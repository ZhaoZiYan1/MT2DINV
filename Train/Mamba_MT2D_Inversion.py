"""
Mamba-based hybrid architecture for 2D magnetotelluric (MT) inversion.

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Mamba import Mamba as RealMamba, MambaConfig
from Utilities import GaussianNormalizer, GaussianNormalizer_out, loadmat_fields
# ==================================================
# Data preprocessing (shared by training and prediction to ensure consistency between the two stages)
# ==================================================
# Fixed order of the four input channels: Two apparent resistivity components and two phase components.
MT_COMPONENT_ORDER = ['Rhoxy', 'Phsxy', 'Rhoyx', 'Phsyx']
# Indices of the apparent resistivity channels (log10 transformed). Phase channels retain their original values.
RHO_CHANNELS = (0, 2)
def preprocess_mt_inputs(raw: dict, n: int) -> torch.Tensor:
    """
    Concatenate the four MT components from the MAT file and apply a log10 transformation to apparent resistivity.
    Args:
        raw: Dictionary mapping component names to arrays with shape (N, n_freq, n_obs).
        n: Number of samples to select from the beginning.
    Returns:
        x: Tensor with shape (n, n_freq, n_obs, 4), with log10-transformed apparent resistivity channels.
    """
    x = torch.tensor(np.stack([raw[k][:n] for k in MT_COMPONENT_ORDER], axis=-1),dtype=torch.float32)
    for c in RHO_CHANNELS:
        x[..., c] = torch.log10(x[..., c] + 1e-10)
    return x
def preprocess_resistivity_targets(sig_k: np.ndarray, target_nz: int, target_ny: int) -> torch.Tensor:
    """
    Convert conductivity models to log10 resistivity models and resample them to the target grid.

    Args:
        sig_k: Conductivity models with shape (N, nz, ny).
    Returns:
        y: Log10 resistivity models with shape (N, target_nz, target_ny), expressed in log10(Ω·m).
    """
    y = torch.tensor(sig_k, dtype=torch.float32)
    # If the original grid dimensions differ from the target dimensions, resample the conductivity model using bilinear interpolation.
    if y.shape[1] != target_nz or y.shape[2] != target_ny:
        y = F.interpolate(y.unsqueeze(1), size=(target_nz, target_ny),mode='bilinear', align_corners=True).squeeze(1)
    # Convert conductivity to resistivity (rho = 1/sigma), followed by a log10 transformation.
    return torch.log10(1.0 / (y + 1e-10))
def load_and_preprocess_data(train_path, val_path, test_path, ntrain, nval, ntest, target_nz, target_ny):
    """
    Load and preprocess training, validation, and test datasets from three independent MAT files.
    Fit the normalizers using statistics from the training dataset only. Apply the fitted normalizers to the validation and test datasets
    without recalculating statistics to prevent data leakage.
    Args:
        train_path / val_path / test_path: Paths to the three dataset files (val_path may be None).
        ntrain / nval / ntest: Number of samples used from each dataset.
    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, x_normalizer, y_normalizer
    """
    fields = list(MT_COMPONENT_ORDER) + ['sig_k']
    print(f"Loading training data: {train_path}")
    train_raw = loadmat_fields(train_path, fields)
    x_train = preprocess_mt_inputs(train_raw, ntrain)
    y_train = preprocess_resistivity_targets(train_raw['sig_k'][:ntrain], target_nz, target_ny)
    # Validation dataset: Load from an independent file. Leave it empty if val_path is None or nval=0.
    if val_path and nval > 0:
        print(f"Loading validation data: {val_path}")
        val_raw = loadmat_fields(val_path, fields)
        x_val = preprocess_mt_inputs(val_raw, nval)
        y_val = preprocess_resistivity_targets(val_raw['sig_k'][:nval], target_nz, target_ny)
    else:
        x_val, y_val = x_train[:0], y_train[:0]
    print(f"Loading test data: {test_path}")
    test_raw = loadmat_fields(test_path, fields)
    x_test = preprocess_mt_inputs(test_raw, ntest)
    y_test = preprocess_resistivity_targets(test_raw['sig_k'][:ntest], target_nz, target_ny)
    # Gaussian normalization: Calculate normalization statistics exclusively from the training dataset.
    x_normalizer = GaussianNormalizer_out(x_train, flag='Point')
    y_normalizer = GaussianNormalizer(y_train, flag='Point')
    x_train = x_normalizer.encode(x_train)
    x_val = x_normalizer.encode(x_val) if nval > 0 else x_val
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)
    y_val = y_normalizer.encode(y_val) if nval > 0 else y_val
    y_test = y_normalizer.encode(y_test)
    print(f"x_train: {tuple(x_train.shape)}, x_val: {tuple(x_val.shape)}, "f"x_test: {tuple(x_test.shape)}")
    print(f"y_train: {tuple(y_train.shape)}, y_val: {tuple(y_val.shape)}, "f"y_test: {tuple(y_test.shape)}")
    return x_train, y_train, x_val, y_val, x_test, y_test, x_normalizer, y_normalizer
# ==================================================
# Module 1: Continuous positional encoder
# ==================================================
class ContinuousPositionalEncoder(nn.Module):
    """
    Map continuous (obs, freq) coordinates to Fourier features, followed by a linear projection to encoding_dim dimensions.
    Generate sine and cosine basis functions at freqs_per_dim frequencies for each input coordinate dimension.
    The total number of Fourier features is 2 * input_dim * freqs_per_dim, which are subsequently projected to encoding_dim dimensions.
    """
    def __init__(self, input_dim: int = 2, encoding_dim: int = 32, max_freq: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        freqs_per_dim = encoding_dim // (2 * input_dim)  # Number of frequencies assigned to each coordinate dimension.
        freqs = torch.linspace(1.0, max_freq, freqs_per_dim)
        self.register_buffer('freqs', freqs)
        self.projection = nn.Linear(freqs_per_dim * 2 * input_dim, encoding_dim)
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (batch, n_points, input_dim)  Normalized coordinates with shape
        Returns:
            encoded: (batch, n_points, encoding_dim)
        """
        B, N, _ = coords.shape
        # Generate angular arguments at multiple frequencies for each coordinate dimension.
        angles = 2 * np.pi * coords.unsqueeze(-1) * self.freqs  # (B, N, D, F)
        # Concatenate sine and cosine components to construct the Fourier features.
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.projection(fourier.reshape(B, N, -1))
# ==================================================
# Mamba-related modules
# ==================================================
class MambaLayer(nn.Module):
    """Single Mamba layer (a wrapper around Mamba.py with n_layers=1).
    Requires the official fused CUDA kernel (selective_scan_fn) from mamba_ssm, which provides an observed speedup of approximately 5–6 times.
    The kernel is imported during initialization, and an error is raised immediately if it is unavailable. Forward propagation requires GPU tensors.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand_factor: int = 2):
        super().__init__()
        config = MambaConfig(
            d_model=d_model,
            n_layers=1,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=4,)
        self.mamba = RealMamba(config)
    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("Mamba forward propagation requires CUDA tensors. Move both the model and input to the GPU. No pure PyTorch CPU fallback is provided.No pure PyTorch CPU fallback is provided.")
        return self.mamba(x)
class SpatialMambaBlock(nn.Module):
    """
       Row-wise and column-wise bidirectional Mamba: Scan the feature grid along horizontal and vertical directions separately, followed by feature fusion.
       Treat the 2D feature map (H, W) as two sets of sequences corresponding to the horizontal and vertical directions:
       -Horizontal direction: Treat each row (length W) as an independent sequence processed by Mamba.
       -Vertical direction: Treat each column (length H) as an independent sequence processed by Mamba.
       Concatenate and fuse the outputs from both directions to capture anisotropic long-range dependencies.
       When bidirectional=True (recommended), an additional reverse scan is performed along each direction.
       The input sequence is reversed and processed by the same Mamba layer with shared weights.
       This provides reverse information paths missing from unidirectional scanning (R->L and B->T, in addition to L->R and T->B), resulting in four-direction feature fusion.
    """
    def __init__(self, d_model: int = 128, d_state: int = 16, expand_factor: int = 2, bidirectional: bool = False, reverse: tuple = (False, False)):
        """
        bidirectional=True: Perform forward and reverse scans along both rows and columns, producing four-direction feature fusion. Weights are shared, with approximately 1.8 times the computational cost.
        bidirectional=False: Perform a single scan per direction, with scan orientation specified by reverse.
            reverse[0]=True -> Use right-to-left (R->L) row scanning by reversing the input sequence.
            reverse[1]=True -> Use bottom-to-top (B->T) column scanning.
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.reverse = reverse
        self.row_mamba = MambaLayer(d_model, d_state, expand_factor)
        self.col_mamba = MambaLayer(d_model, d_state, expand_factor)
        n_dir = 4 if bidirectional else 2
        self.fusion = nn.Sequential(nn.Linear(n_dir * d_model, d_model), nn.SiLU(), nn.Dropout(0.1))
        self.norm = nn.LayerNorm(d_model)
    @staticmethod
    def _scan(mamba, seq, backward: bool):
        if backward:
            return mamba(torch.flip(seq, dims=[1])).flip(dims=[1])
        return mamba(seq)
    def forward(self, x):
        """
        x: (B, H, W, C) -> (B, H, W, C)
        """
        B, H, W, C = x.shape
        # Horizontal direction: Treat each row as an independent sequence with shape (B*H, W, C).
        # Vertical direction: Treat each column as an independent sequence with shape (B*W, H, C).
        row_inp = x.reshape(B * H, W, C)
        col_inp = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
        row_outs = [self._scan(self.row_mamba, row_inp, self.reverse[0])]
        col_outs = [self._scan(self.col_mamba, col_inp, self.reverse[1])]
        if self.bidirectional:
            # Perform forward and reverse scans along each direction for four-direction feature fusion.
            row_outs.append(self._scan(self.row_mamba, row_inp, not self.reverse[0]))
            col_outs.append(self._scan(self.col_mamba, col_inp, not self.reverse[1]))
        feats = [o.reshape(B, H, W, C) for o in row_outs] + \
                [o.reshape(B, W, H, C).permute(0, 2, 1, 3) for o in col_outs]
        # Fuse output features from two or four scan directions.
        out = self.fusion(torch.cat(feats, dim=-1))
        # Residual connection followed by LayerNorm.
        return self.norm(x + out)
class SpatialMambaEncoder(nn.Module):
    # Stack multiple SpatialMambaBlocks to construct the encoder.
    def __init__(self, d_model: int = 128, n_layers: int = 2, d_state: int = 16, expand_factor: int = 2, bidirectional=False):
        super().__init__()
        if bidirectional == 'alternate':
            self.layers = nn.ModuleList([
                SpatialMambaBlock(d_model, d_state, expand_factor, bidirectional=False, reverse=(i % 2 == 1, i % 2 == 1))
                for i in range(n_layers)])
        else:
            self.layers = nn.ModuleList([SpatialMambaBlock(d_model, d_state, expand_factor, bidirectional) for _ in range(n_layers)])
    def forward(self, x):
        """x: (B, H, W, C)"""
        for layer in self.layers:
            x = layer(x)
        return x
# ==================================================
# Transformer ablation module (structurally aligned with SpatialMamba, replacing the SSM core with standard MHSA)
# ==================================================
class SpatialTransformerBlock(nn.Module):
    """
    Row-wise and column-wise self-attention block: A comparison architecture used to replace SpatialMambaBlock in ablation experiments.
    The data flow is consistent with SpatialMambaBlock: Internal residual connections in the row and column branches,
    followed by concatenation, feature fusion, an external residual connection, and LayerNorm.
    This design aims to isolate differences between self-attention and the SSM core rather than peripheral structures.
    Capacity alignment: Mamba internally expands the hidden dimension by the configurable expansion factor (d_inner = expand * d_model).
    The attention-based version uses a standard feed-forward network (FFN, mlp_ratio=4) after feature fusion to provide corresponding model capacity.
    For ablation comparisons, keep d_model, the number of layers, and the expansion configuration fixed across variants to isolate the effect of the encoder core.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.row_norm = nn.LayerNorm(d_model)
        self.row_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.col_norm = nn.LayerNorm(d_model)
        self.col_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(2 * d_model, int(mlp_ratio * d_model)),nn.GELU(),nn.Dropout(dropout),nn.Linear(int(mlp_ratio * d_model), d_model),)
        self.fusion = nn.Sequential(nn.Linear(d_model, d_model),nn.SiLU(),nn.Dropout(0.1))
        self.norm = nn.LayerNorm(d_model)
    @staticmethod
    def _attn(attn, norm, x):
        # Apply pre-normalized self-attention along one direction, followed by an internal residual connection.
        h = norm(x)
        out, _ = attn(h, h, h, need_weights=False)
        return x + out
    def forward(self, x):
        """
        x: (B, H, W, C) -> (B, H, W, C)
        """
        B, H, W, C = x.shape
        # Horizontal direction: Treat each row as an independent sequence, consistent with row_mamba.
        row_out = self._attn(self.row_attn, self.row_norm, x.reshape(B * H, W, C)).reshape(B, H, W, C)
        # Vertical direction: Treat each column as an independent sequence, consistent with col_mamba.
        x_col = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
        col_out = self._attn(self.col_attn, self.col_norm, x_col).reshape(B, W, H, C).permute(0, 2, 1, 3)
        # Feature fusion, MLP, and residual connection.
        out = self.fusion(self.mlp(torch.cat([row_out, col_out], dim=-1)))
        return self.norm(x + out)
class SpatialTransformerEncoder(nn.Module):
    # Row-wise and column-wise Transformer encoder used for ablation comparison (encoder_type='attention').
    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([SpatialTransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
    def forward(self, x):
        #x: (B, H, W, C)
        for layer in self.layers:
            x = layer(x)
        return x
# ==================================================
# CNN modules
# ==================================================
class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block: 7x7 depthwise convolution, pointwise convolutions, and layer-scaled residual connection.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)   # Apply GroupNorm with one group to normalize features within each sample.
        self.pwconv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
    def forward(self, x):
        shortcut = x
        x = self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))
        return shortcut + self.gamma.view(1, -1, 1, 1) * x
class SeparableConvBlock(nn.Module):
    """Double depthwise separable convolution block: (DW 3x3 + PW 1x1 + BN + SiLU) x 2
       Compared with a standard block containing two dense 3x3 convolutions, the parameter count and computational cost are reduced to approximately 1/8–1/9.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),)
    def forward(self, x):
        return self.block(x)
class SpatialAttention(nn.Module):
    # Lightweight spatial attention: Depthwise convolution performs channel-wise spatial filtering, while a 1x1 convolution reduces the features to a single spatial attention map.
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid())
    def forward(self, x):
        return x * self.attn(x)
# ==================================================
# Module 3: Lightweight FPN decoder
# ==================================================
class FPNDecoder(nn.Module):
    """
    Lightweight feature pyramid network (FPN) decoder based on depthwise separable convolutions.
    - Replace all conventional ConvBlocks with SeparableConvBlocks, reducing the parameter count and FLOPs by approximately an order of magnitude.
    - Redesign the output head as "low-resolution refinement -> upsampling -> 1x1 output projection".
    Input (B, in_channels, H, W) -> Output (B, out_channels, *out_size)
    """
    def __init__(self, in_channels=128, out_channels=1, base_channels=64, out_size=(32, 32)):
        super().__init__()
        c = base_channels
        self.out_size = tuple(out_size)
        # ---- Bottom-up feature pyramid: Channels 64 -> 128 -> 256 (Dense convolutions are used for downsampling and channel adjustment) ----
        self.c1 = SeparableConvBlock(in_channels, c)           # H x W
        self.down1 = nn.Sequential(nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1), nn.SiLU()) # -> H/2
        self.c2 = SeparableConvBlock(2 * c, 2 * c)
        self.down2 = nn.Sequential(nn.Conv2d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1), nn.SiLU()) # -> H/4
        self.c3 = SeparableConvBlock(4 * c, 4 * c)
        # ---- Lateral connections: Project features at all pyramid levels to 2c channels using computationally efficient 1x1 convolutions. ----
        self.lat3 = nn.Conv2d(4 * c, 2 * c, kernel_size=1)
        self.lat2 = nn.Conv2d(2 * c, 2 * c, kernel_size=1)
        self.lat1 = nn.Conv2d(c, 2 * c, kernel_size=1)
        # ---- Attention-based refinement and smoothing using depthwise separable convolutions. ----
        self.attn2 = SpatialAttention(2 * c)
        self.attn1 = SpatialAttention(2 * c)
        self.smooth2 = SeparableConvBlock(2 * c, 2 * c)
        self.smooth1 = SeparableConvBlock(2 * c, 2 * c)
        # ---- Output head: Low-resolution refinement -> Upsampling -> 1x1 output projection. ----
        self.head = nn.Sequential(nn.Conv2d(2 * c, c, kernel_size=1), nn.BatchNorm2d(c), nn.SiLU(), nn.Conv2d(c, c, kernel_size=3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)
    def forward(self, x):
        # Bottom-up feature extraction pathway.
        c1 = self.c1(x)               # Original spatial resolution.
        c2 = self.c2(self.down1(c1))  # Half spatial resolution.
        c3 = self.c3(self.down2(c2))  # Quarter spatial resolution.
        # Top-down feature fusion with upsampling and lateral connections.
        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.smooth2(self.attn2(p2))
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.smooth1(self.attn1(p1))
        # Refine features at low resolution, upsample them to the target grid, and apply a final 1x1 convolution for output prediction.
        feat = self.head(p1)
        out = F.interpolate(feat, size=self.out_size, mode='bilinear', align_corners=False)
        return self.out_conv(out)
# ==================================================
# Complete network architecture
# ==================================================
class MambaMT2DInversion(nn.Module):
    """
    CNN–Mamba hybrid network for 2D MT inversion.
    Input: MT observation data with shape (B, n_freq, n_obs, n_comp)
    Output: Normalized log10 resistivity model with shape (B, target_nz, target_ny)
    """
    def __init__(self,n_freq: int = 21,
                      n_obs: int = 21,
                      n_comp: int = 4,
                      target_nz: int = 32,
                      target_ny: int = 32,
                      d_model: int = 128,
                      n_mamba_layers: int = 2,
                      freq_range: tuple = (0.01, 1e3),
                      freq_descending: bool = True,
                      freq_values: list = None,
                      fpn_base_channels: int = 64,
                      mamba_d_state: int = 16,
                      mamba_expand: int = 2,
                      mamba_bidirectional: bool = False,
                      use_cnn_branch: bool = True,
                      encoder_type: str = 'mamba',
                      attention_mlp_ratio: float = 4.0):
        super().__init__()
        self.n_freq = n_freq
        self.n_obs = n_obs
        self.n_comp = n_comp
        self.target_nz = target_nz
        self.target_ny = target_ny
        self.d_model = d_model
        self.freq_range = tuple(freq_range)
        self.freq_descending = freq_descending
        self.freq_values = list(freq_values) if freq_values is not None else None
        # ---------- Input encoding ----------
        # Project the MT input components to d_model//4-dimensional feature representations.
        self.data_projection = nn.Linear(n_comp, d_model // 4)
        # Continuous positional encoder: Map (obs, freq) coordinates to d_model//4 dimensions.
        self.pos_encoder = ContinuousPositionalEncoder(input_dim=2, encoding_dim=d_model // 4, max_freq=10.0)
        # Concatenate data and positional features, then project them to d_model dimensions.
        self.feature_fusion = nn.Linear(2 * (d_model // 4), d_model)
        # Optional CNN branch for local feature extraction. This branch is skipped when use_cnn_branch=False in ablation experiments.
        self.use_cnn_branch = use_cnn_branch
        if use_cnn_branch:
            self.cnn_branch = nn.Sequential(ConvNeXtBlock(d_model), ConvNeXtBlock(d_model),)
            self.cnn_fusion = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.SiLU(), nn.Dropout(0.1))
        # Context encoder for ablation experiments: mamba / attention / none.
        if encoder_type == 'mamba':
            self.context_encoder = SpatialMambaEncoder(
                d_model=d_model,
                n_layers=n_mamba_layers,
                d_state=mamba_d_state,
                expand_factor=mamba_expand,
                bidirectional=mamba_bidirectional,)
        elif encoder_type == 'attention':
            self.context_encoder = SpatialTransformerEncoder(
                d_model=d_model,
                n_layers=n_mamba_layers,
                n_heads=d_model // 32,   # head_dim=32, d_model=128 -> 4
                mlp_ratio=attention_mlp_ratio,
                dropout=0.1,)
        elif encoder_type == 'none':
            self.context_encoder = nn.Identity()  # Identity mapping
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type} " f"(available options 'mamba' | 'attention' | 'none')")
        # FPN decoder with output dimensions determined by the target grid.
        self.decoder = FPNDecoder(in_channels=d_model, out_channels=1, base_channels=fpn_base_channels, out_size=(target_nz, target_ny))
        # Precompute the input grid coordinates (obs, freq) and register them as a persistent buffer to include them in the model state_dict.
        self.register_buffer('input_coords', self._build_input_coords(), persistent=True)
    def _build_input_coords(self) -> torch.Tensor:
        # Generate normalized coordinates with shape (n_freq*n_obs, 2). Observed station coordinates are linearly normalized, while frequency coordinates are normalized in log10 space.
        obs_coords = torch.linspace(0.0, 1.0, self.n_obs)
        if self.freq_values is not None:
            # Field frequencies: Directly use the frequency values stored in the data file because the observed frequencies are not strictly logarithmically spaced.
            freq_coords = torch.log10(torch.tensor(self.freq_values, dtype=torch.float32))
        else:
            f_min, f_max = sorted(self.freq_range)
            if self.freq_descending:
                # Match the generator's logspace(3, -2) convention: Data index 0 corresponds to the highest frequency.
                freq_values = torch.logspace(np.log10(f_max), np.log10(f_min), self.n_freq)
            else:
                freq_values = torch.logspace(np.log10(f_min), np.log10(f_max), self.n_freq)
            # Apply the log10 transformation to frequencies.
            freq_coords = torch.log10(freq_values)
        freq_coords = (freq_coords - freq_coords.min()) / (freq_coords.max() - freq_coords.min())
        # Construct the coordinate grid and flatten it into shape (n_freq*n_obs, 2).
        freq_grid, obs_grid = torch.meshgrid(freq_coords, obs_coords, indexing='ij')
        return torch.stack([obs_grid.reshape(-1), freq_grid.reshape(-1)], dim=-1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # 1. Flatten the input into a sequence of shape (B, L, n_comp), where L = n_freq * n_obs.
        sequence_data = x.reshape(B, self.n_freq * self.n_obs, self.n_comp)
        coords = self.input_coords.unsqueeze(0).expand(B, -1, -1)
        # 2. Apply data projection and positional encoding, followed by feature fusion.
        data_features = self.data_projection(sequence_data)
        pos_features = self.pos_encoder(coords)
        fused = self.feature_fusion(torch.cat([data_features, pos_features], dim=-1))
        # 3. Optional CNN branch: Reshape the sequence into a 2D feature map, extract convolutional features, and fuse them back into the sequence.
        if self.use_cnn_branch:
            spatial = fused.view(B, self.n_freq, self.n_obs, self.d_model).permute(0, 3, 1, 2)
            cnn_features = self.cnn_branch(spatial)
            cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(B, self.n_freq * self.n_obs, self.d_model)
            fused = self.cnn_fusion(torch.cat([fused, cnn_features], dim=-1))
        # 4. Apply context encoding (Spatial Mamba) to obtain the encoded feature map.
        latent = self.context_encoder(fused.view(B, self.n_freq, self.n_obs, self.d_model))
        # 5. Apply the FPN decoder to reconstruct the resistivity model with shape (B, target_nz, target_ny).
        resistivity = self.decoder(latent.permute(0, 3, 1, 2))  # Permute to (B, C, H, W) for the FPN decoder.
        return resistivity.squeeze(1) # Remove the singleton channel dimension because out_channels=1.
# ==================================================
# Loss functions and utility functions
# ==================================================
def _sobel_kernels(device) -> tuple:
    # Return Sobel convolution kernels on the specified device for edge detection.
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    return sobel_x, sobel_y
def compute_gradient(x: torch.Tensor, sobel_x, sobel_y) -> torch.Tensor:
    # Compute the Sobel gradient magnitude of an image tensor with shape (B, H, W).
    grad_x = F.conv2d(x.unsqueeze(1), sobel_x, padding=1)
    grad_y = F.conv2d(x.unsqueeze(1), sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6).squeeze(1)
def edge_loss(pred: torch.Tensor, target: torch.Tensor, sobel_x, sobel_y) -> torch.Tensor:
    # L1 loss between Sobel gradient magnitudes to enhance boundary reconstruction.
    return F.l1_loss(compute_gradient(pred, sobel_x, sobel_y), compute_gradient(target, sobel_x, sobel_y))
def edge_weighted_l1(pred: torch.Tensor, target: torch.Tensor, sobel_x, sobel_y, alpha: float = 3.0, tau_quantile: float = 0.9) -> torch.Tensor:
    pred, target = pred.float(), target.float()
    sobel_x, sobel_y = sobel_x.float(), sobel_y.float()
    with torch.no_grad():
        grad = compute_gradient(target, sobel_x, sobel_y)
        tau = torch.quantile(grad.reshape(-1), tau_quantile)
        w = 1.0 + alpha * (grad > tau).float()
    err = (pred - target).abs()
    return (w * err).sum() / w.sum()
def count_parameters(model: nn.Module) -> int:
    # Count the number of trainable model parameters.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def build_param_groups(model: nn.Module, weight_decay: float) -> list:
    # Construct parameter groups for the AdamW optimizer. Exclude biases, normalization parameters, and SSM state parameters (A_log and D) from weight decay.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith('A_log'):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},]
# ==================================================
# Self-test
# ==================================================
def test_mamba_network():
    # Construct the model, print the parameter count of each module, and verify forward propagation.
    print("Testing the CNN–Mamba MT inversion network...")
    model = MambaMT2DInversion()
    total = count_parameters(model)
    print(f"{'Module':<24}{'Parameters':>14}{'Percentage':>8}")
    print("-" * 48)
    for name, child in model.named_children():
        c = count_parameters(child)
        print(f"{name:<26}{c:>14,}{c / total:>7.1%}")
    print("-" * 48)
    print(f"{'Total':<26}{total:>14,}")
    x = torch.randn(4, 21, 21, 4)
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    else:
        print("\nNo CUDA environment is available. Skip the forward propagation test because Mamba requires the mamba_ssm CUDA kernel.")
        return
    with torch.no_grad():
        y = model(x)
    print(f"\nInput {tuple(x.shape)} -> Output {tuple(y.shape)}")
    assert y.shape == (4, 32, 32), "Output shape mismatch"
    print("Network test completed!")
if __name__ == "__main__":
    test_mamba_network()
