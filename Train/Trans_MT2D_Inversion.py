"""
Standalone Transformer network module for 2D magnetotelluric (MT) inversion (independent of Mamba_MT2D_Inversion.py)
=============================================================================
Controlled comparison with the standalone Mamba network (MambaMT2DInversion in Mamba_MT2D_Inversion.py):
    - Identical frontend and decoder: data projection / continuous positional encoding / feature fusion / FPN decoder
      (copied from the Mamba module to keep implementations independent and allow separate debugging)
    - Only difference: the context encoder uses standard row/column multi-head self-attention (SpatialTransformerEncoder),
      with no CNN branch
    - Capacity matching: the training script adjusts the FFN expansion ratio mlp_ratio by binary search to match the
      Mamba(expand) encoder parameter count (measured at d_model=128, 2 layers: expand=2 -> r≈2.38, expand=4 -> r≈7.1)
Actual architecture:
    Input (B, n_freq, n_obs, 4): [Rhoxy, Phsxy, Rhoyx, Phsyx], four MT observation channels
    |
    |-- Data projection:         4 components -> d_model//4-dimensional features
    |-- Continuous encoding:    normalized (obs, freq) coordinates -> Fourier features -> d_model//4 dimensions
    |-- Feature fusion:         concatenate and project to d_model, then reshape to the (n_freq, n_obs) feature grid
    |-- Context encoder:        standard row/column multi-head self-attention (MHSA + FFN)
    |-- FPN decoder:            lightweight feature pyramid (base_channels=64, depthwise separable convolutions), upsampled to the target grid
    Output (B, target_nz, target_ny): log10 resistivity model in normalized space
This module includes data preprocessing and loss utilities identical to those in the Mamba module; training and
prediction scripts only need to import this module and Utilities.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utilities import GaussianNormalizer, GaussianNormalizer_out, loadmat_fields
# ==================================================
# Data preprocessing (shared by training and prediction to ensure consistency)
# ==================================================
# Fixed order of the four input channels: two apparent resistivities and two phases
MT_COMPONENT_ORDER = ['Rhoxy', 'Phsxy', 'Rhoyx', 'Phsyx']
# Channel indices for apparent resistivity (apply log10); phase channels remain unchanged
RHO_CHANNELS = (0, 2)
def preprocess_mt_inputs(raw: dict, n: int) -> torch.Tensor:
    """
    Concatenate the four MT components from the .mat file and apply log10 transformation
    Args:
        raw: dictionary {component_name: ndarray (N, n_freq, n_obs)}
        n: number of samples to take from the beginning
    Returns:
        x: (n, n_freq, n_obs, 4), with log10 applied to the apparent-resistivity channels
    """
    x = torch.tensor(np.stack([raw[k][:n] for k in MT_COMPONENT_ORDER], axis=-1), dtype=torch.float32)
    for c in RHO_CHANNELS:
        x[..., c] = torch.log10(x[..., c] + 1e-10)
    return x
def preprocess_resistivity_targets(sig_k: np.ndarray, target_nz: int, target_ny: int) -> torch.Tensor:
    """
    Convert conductivity models to log10 resistivity models and resample to the target grid
    Args:
        sig_k: (N, nz, ny) conductivity
    Returns:
        y: (N, target_nz, target_ny) log10 resistivity (log10(Ω·m))
    """
    y = torch.tensor(sig_k, dtype=torch.float32)
    # Resample using bilinear interpolation if the original grid size differs from the target
    if y.shape[1] != target_nz or y.shape[2] != target_ny:
        y = F.interpolate(y.unsqueeze(1), size=(target_nz, target_ny), mode='bilinear', align_corners=True).squeeze(1)
    # Conductivity -> resistivity (rho = 1/sigma), followed by log10
    return torch.log10(1.0 / (y + 1e-10))
def load_and_preprocess_data(train_path, val_path, test_path, ntrain, nval, ntest, target_nz, target_ny):
    """
    Load and preprocess training/validation/test data (three separate .mat files)
    Fit normalization statistics only on the training set (encode validation/test data without refitting) to prevent data leakage.
    Args:
        train_path / val_path / test_path: paths to the three data files (val_path can be None)
        ntrain / nval / ntest: number of samples used from each file
    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, x_normalizer, y_normalizer
    """
    fields = list(MT_COMPONENT_ORDER) + ['sig_k']
    print(f"Loading training data: {train_path}")
    train_raw = loadmat_fields(train_path, fields)
    x_train = preprocess_mt_inputs(train_raw, ntrain)
    y_train = preprocess_resistivity_targets(train_raw['sig_k'][:ntrain], target_nz, target_ny)
    # Validation set: separate file (empty if val_path is None or nval=0)
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
    # Gaussian normalization: compute statistics from the training set only
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
# Module 1: Continuous positional encoder (identical to the Mamba module)
# ==================================================
class ContinuousPositionalEncoder(nn.Module):
    """
    Map continuous (obs, freq) coordinates to Fourier features and linearly project to encoding_dim
    Generate freqs_per_dim sine/cosine frequency bases for each input dimension;
    the total number of features is 2 * input_dim * freqs_per_dim before projection to encoding_dim.
    This encoding captures continuous coordinate variations more flexibly than discrete embeddings.
    """
    def __init__(self, input_dim: int = 2, encoding_dim: int = 32, max_freq: float = 10.0):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        freqs_per_dim = encoding_dim // (2 * input_dim)  # Number of frequencies per coordinate dimension
        freqs = torch.linspace(1.0, max_freq, freqs_per_dim)
        self.register_buffer('freqs', freqs)       # Register as a buffer: not trainable, but included in saved model state
        self.projection = nn.Linear(freqs_per_dim * 2 * input_dim, encoding_dim)
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (batch, n_points, input_dim) normalized coordinates
        Returns:
            encoded: (batch, n_points, encoding_dim)
        """
        B, N, _ = coords.shape
        # Generate angles at multiple frequencies for each coordinate dimension
        angles = 2 * np.pi * coords.unsqueeze(-1) * self.freqs  # (B, N, D, F)
        # Concatenate sine and cosine components to obtain Fourier features
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.projection(fourier.reshape(B, N, -1))
# ==================================================
# Transformer context encoder (consistent with the ablation implementation in the Mamba module)
# ==================================================
class SpatialTransformerBlock(nn.Module):
    """
       Row/column self-attention block: apply standard multi-head self-attention independently along rows and columns, then fuse
       The data flow matches the standalone Mamba SpatialMambaBlock (each row/column branch includes
       an internal residual connection -> concatenation and fusion -> external residual connection + LayerNorm).
       This isolates the difference between attention and the SSM kernel rather than the surrounding architecture.
       Capacity matching: Mamba layers internally expand the hidden dimension by expand (d_inner = expand * d_model);
       the attention variant uses a standard FFN (mlp_ratio, applied after fusion) to provide comparable capacity.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.row_norm = nn.LayerNorm(d_model)
        self.row_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.col_norm = nn.LayerNorm(d_model)
        self.col_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(2 * d_model, int(mlp_ratio * d_model)), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(mlp_ratio * d_model), d_model),)
        self.fusion = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Dropout(0.1))
        self.norm = nn.LayerNorm(d_model)
    @staticmethod
    def _attn(attn, norm, x):
        # Apply pre-norm self-attention with an internal residual connection along one direction
        h = norm(x)
        out, _ = attn(h, h, h, need_weights=False)
        return x + out
    def forward(self, x):
        """
        x: (B, H, W, C) -> (B, H, W, C)
        """
        B, H, W, C = x.shape
        # Horizontal direction: each row is treated as a sequence
        row_out = self._attn(self.row_attn, self.row_norm, x.reshape(B * H, W, C)).reshape(B, H, W, C)
        # Vertical direction: each column is treated as a sequence
        x_col = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
        col_out = self._attn(self.col_attn, self.col_norm, x_col).reshape(B, W, H, C).permute(0, 2, 1, 3)
        # Fusion + MLP + residual connection
        out = self.fusion(self.mlp(torch.cat([row_out, col_out], dim=-1)))
        return self.norm(x + out)
class SpatialTransformerEncoder(nn.Module):
    # Row/column Transformer encoder
    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([SpatialTransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)])
    def forward(self, x):
        #x: (B, H, W, C)
        for layer in self.layers:
            x = layer(x)
        return x
# ==================================================
# FPN decoder module (identical to the Mamba module)
# ==================================================
class SeparableConvBlock(nn.Module):
    """Double depthwise separable convolution block: (DW 3x3 + PW 1x1 + BN + SiLU) x 2"""
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
    # Lightweight spatial attention: depthwise convolution filters each channel spatially; 1x1 convolution produces an attention map
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid())
    def forward(self, x):
        return x * self.attn(x)
class FPNDecoder(nn.Module):
    """
    Lightweight FPN decoder (depthwise separable variant)
    Input (B, in_channels, H, W) -> output (B, out_channels, *out_size)
    """
    def __init__(self, in_channels=128, out_channels=1, base_channels=64, out_size=(32, 32)):
        super().__init__()
        c = base_channels
        self.out_size = tuple(out_size)
        # ---- Bottom-up pyramid: channels 64 -> 128 -> 256 (standard convolution changes channels during downsampling) ----
        self.c1 = SeparableConvBlock(in_channels, c)           # H x W
        self.down1 = nn.Sequential(nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1), nn.SiLU()) # -> H/2
        self.c2 = SeparableConvBlock(2 * c, 2 * c)
        self.down2 = nn.Sequential(nn.Conv2d(2 * c, 4 * c, kernel_size=3, stride=2, padding=1), nn.SiLU()) # -> H/4
        self.c3 = SeparableConvBlock(4 * c, 4 * c)
        # ---- Lateral connections: reduce all features to 2c channels (inexpensive 1x1 convolutions) ----
        self.lat3 = nn.Conv2d(4 * c, 2 * c, kernel_size=1)
        self.lat2 = nn.Conv2d(2 * c, 2 * c, kernel_size=1)
        self.lat1 = nn.Conv2d(c, 2 * c, kernel_size=1)
        # ---- Attention refinement and separable smoothing ----
        self.attn2 = SpatialAttention(2 * c)
        self.attn1 = SpatialAttention(2 * c)
        self.smooth2 = SeparableConvBlock(2 * c, 2 * c)
        self.smooth1 = SeparableConvBlock(2 * c, 2 * c)
        # ---- Output head: low-resolution refinement -> upsampling -> 1x1 output projection ----
        self.head = nn.Sequential(nn.Conv2d(2 * c, c, kernel_size=1), nn.BatchNorm2d(c), nn.SiLU(), nn.Conv2d(c, c, kernel_size=3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)
    def forward(self, x):
        # Bottom-up path
        c1 = self.c1(x)               # Original resolution
        c2 = self.c2(self.down1(c1))  # Half resolution
        c3 = self.c3(self.down2(c2))  # Quarter resolution
        # Top-down fusion (upsampling + lateral connections)
        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.smooth2(self.attn2(p2))
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.smooth1(self.attn1(p1))
        # Refine at low resolution, upsample to the target size, then apply the final 1x1 convolution
        feat = self.head(p1)
        out = F.interpolate(feat, size=self.out_size, mode='bilinear', align_corners=False)
        return self.out_conv(out)
# ==================================================
# Complete network
# ==================================================
class TransMT2DInversion(nn.Module):
    """
    Standalone Transformer network for 2D MT inversion (no CNN branch)
    The frontend (data projection / continuous positional encoding / feature fusion) and FPN decoder are identical
    to the standalone Mamba network; only the context encoder uses standard row/column multi-head self-attention.
    Input: (B, n_freq, n_obs, n_comp) MT observations
    Output: (B, target_nz, target_ny) normalized log10 resistivity model
    """
    def __init__(self, n_freq: int = 31,
                       n_obs: int = 25,
                       n_comp: int = 4,
                       target_nz: int = 32,
                       target_ny: int = 64,
                       d_model: int = 128,
                       n_layers: int = 2,
                       freq_range: tuple = (0.01, 1e3),
                       freq_descending: bool = True,
                       freq_values: list = None,
                       fpn_base_channels: int = 64,
                       n_heads: int = None,
                       mlp_ratio: float = 4.0):
        super().__init__()
        if n_heads is None:
            n_heads = max(1, d_model // 32)   # head_dim=32, d_model=128 -> 4 heads
        self.n_freq = n_freq
        self.n_obs = n_obs
        self.n_comp = n_comp
        self.target_nz = target_nz
        self.target_ny = target_ny
        self.d_model = d_model
        self.freq_range = tuple(freq_range)
        self.freq_descending = freq_descending
        # Measured frequencies (Hz): if provided, positional encoding uses log10 of the actual frequencies instead of
        # the log-uniform approximation from freq_range/freq_descending; order must match the input frequency axis
        self.freq_values = list(freq_values) if freq_values is not None else None
        # ---------- Input encoding (identical to the standalone Mamba network) ----------
        # Project the four MT components to d_model//4 dimensions
        self.data_projection = nn.Linear(n_comp, d_model // 4)
        # Continuous positional encoder (obs, freq) -> d_model//4 dimensions
        self.pos_encoder = ContinuousPositionalEncoder(input_dim=2, encoding_dim=d_model // 4, max_freq=10.0)
        # Concatenate data and positional features, then project to d_model
        self.feature_fusion = nn.Linear(2 * (d_model // 4), d_model)
        # Context encoder: standard row/column multi-head self-attention (mlp_ratio is the capacity counterpart of Mamba expand)
        self.context_encoder = SpatialTransformerEncoder(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=0.1,)
        # FPN decoder (output dimensions follow the target grid)
        self.decoder = FPNDecoder(in_channels=d_model, out_channels=1, base_channels=fpn_base_channels, out_size=(target_nz, target_ny))
        # Precompute input-grid coordinates (obs, freq) and register them as a buffer in state_dict
        self.register_buffer('input_coords', self._build_input_coords(), persistent=True)
    def _build_input_coords(self) -> torch.Tensor:
        # Generate normalized coordinates (n_freq*n_obs, 2): linearly normalized obs and log10-normalized freq
        obs_coords = torch.linspace(0.0, 1.0, self.n_obs)
        if self.freq_values is not None:
            # Measured frequencies: directly use the supplied frequency sequence (not necessarily log-uniform)
            freq_coords = torch.log10(torch.tensor(self.freq_values, dtype=torch.float32))
        else:
            f_min, f_max = sorted(self.freq_range)
            if self.freq_descending:
                # Match the generator logspace(3, -2): data index 0 corresponds to the highest frequency
                freq_values = torch.logspace(np.log10(f_max), np.log10(f_min), self.n_freq)
            else:
                freq_values = torch.logspace(np.log10(f_min), np.log10(f_max), self.n_freq)
            # Apply log10 to frequencies
            freq_coords = torch.log10(freq_values)
        freq_coords = (freq_coords - freq_coords.min()) / (freq_coords.max() - freq_coords.min())
        # Construct the coordinate grid and flatten it to (n_freq*n_obs, 2)
        freq_grid, obs_grid = torch.meshgrid(freq_coords, obs_coords, indexing='ij')
        return torch.stack([obs_grid.reshape(-1), freq_grid.reshape(-1)], dim=-1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # 1. Flatten input into sequences (B, L, 4), where L = n_freq * n_obs
        sequence_data = x.reshape(B, self.n_freq * self.n_obs, self.n_comp)
        coords = self.input_coords.unsqueeze(0).expand(B, -1, -1)
        # 2. Data projection + positional encoding -> fusion
        data_features = self.data_projection(sequence_data)
        pos_features = self.pos_encoder(coords)
        fused = self.feature_fusion(torch.cat([data_features, pos_features], dim=-1))
        # 3. Context encoding (row/column self-attention) -> feature map
        latent = self.context_encoder(fused.view(B, self.n_freq, self.n_obs, self.d_model))
        # 4. FPN decoding -> resistivity model (B, target_nz, target_ny)
        resistivity = self.decoder(latent.permute(0, 3, 1, 2))  # Convert to (B, C, H, W) for the FPN decoder
        return resistivity.squeeze(1) # Remove the channel dimension (out_channels=1)
# ==================================================
# Loss and utility functions (identical to the Mamba module)
# ==================================================
def _sobel_kernels(device) -> tuple:
    # Return Sobel kernels on the specified device for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    return sobel_x, sobel_y
def compute_gradient(x: torch.Tensor, sobel_x, sobel_y) -> torch.Tensor:
    # Calculate the Sobel gradient magnitude of an image with shape (B, H, W)
    grad_x = F.conv2d(x.unsqueeze(1), sobel_x, padding=1)
    grad_y = F.conv2d(x.unsqueeze(1), sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6).squeeze(1)
def edge_loss(pred: torch.Tensor, target: torch.Tensor, sobel_x, sobel_y) -> torch.Tensor:
    # L1 loss on Sobel gradients (improves boundary reconstruction)
    return F.l1_loss(compute_gradient(pred, sobel_x, sobel_y), compute_gradient(target, sobel_x, sobel_y))
def edge_weighted_l1(pred: torch.Tensor, target: torch.Tensor, sobel_x, sobel_y, alpha: float = 3.0, tau_quantile: float = 0.9) -> torch.Tensor:
    """
    Gradient-weighted L1 loss: assign a weight of 1 + alpha to pixels with large ground-truth gradients (anomaly boundaries)
    - Use the tau_quantile of ground-truth gradient magnitudes as threshold tau; pixels with |grad(target)| > tau
      are boundary pixels with L1 weight 1 + alpha; all others have weight 1. Return sum(w*|err|)/sum(w).
    - Weights depend only on ground truth (no gradients); calculations use fp32 for stable quantiles under AMP.
    """
    pred, target = pred.float(), target.float()
    sobel_x, sobel_y = sobel_x.float(), sobel_y.float()
    with torch.no_grad():
        grad = compute_gradient(target, sobel_x, sobel_y)
        tau = torch.quantile(grad.reshape(-1), tau_quantile)
        w = 1.0 + alpha * (grad > tau).float()
    err = (pred - target).abs()
    return (w * err).sum() / w.sum()
def count_parameters(model: nn.Module) -> int:
    # Count trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def build_param_groups(model: nn.Module, weight_decay: float) -> list:
    # Construct AdamW parameter groups: exclude bias and normalization weights from weight decay
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0},]
# ==================================================
# Self-test
# ==================================================
def test_trans_network():
    # Build the model, print per-module parameter counts, and verify the forward pass
    print("Testing the standalone Transformer MT inversion network...")
    model = TransMT2DInversion()
    total = count_parameters(model)
    print(f"{'Module':<24}{'Parameters':>14}{'Share':>8}")
    print("-" * 48)
    for name, child in model.named_children():
        c = count_parameters(child)
        print(f"{name:<26}{c:>14,}{c / total:>7.1%}")
    print("-" * 48)
    print(f"{'Total':<26}{total:>14,}")
    x = torch.randn(4, 31, 25, 4)
    with torch.no_grad():
        y = model(x)
    print(f"\nInput {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (4, 32, 64), "Unexpected output shape"
    print("Network test completed!")
if __name__ == "__main__":
    test_trans_network()
