"""
Conventional U-Net module for 2D MT inversion (standalone network module; independent of Mamba_MT2D_Inversion.py)
=========================================================================
End-to-end implementation of the standard U-Net (Ronneberger et al., 2015):
    Input (B, n_freq, n_obs, n_comp): [Rhoxy, Phsxy, Rhoyx, Phsyx] (normalized)
    |-- Encoder: progressive downsampling with DoubleConv + MaxPool, channels b -> 2b -> 4b
    |-- Bottleneck: third MaxPool + DoubleConv(4b -> 8b) (minimum resolution: 3x3)
    |-- Decoder: transposed-convolution upsampling + concatenated skip connections + DoubleConv;
    |             bilinear interpolation for spatial alignment
    |-- Output head: 1x1 convolution -> bilinear interpolation to (target_nz, target_ny)
    Output (B, target_nz, target_ny): log10 resistivity model in normalized space
Comparison with the standalone Mamba network (controlled variables):
    - Same: data, normalization, input/output interfaces, loss, and training protocol
      (enforced by the training scripts)
    - Different: the network architecture itself: a purely convolutional encoder-decoder
      with skip connections, without data projection, continuous positional encoding,
      row/column context encoding, or FPN; this defines the conventional U-Net baseline
Architectural notes:
    - Three downsampling stages rather than the conventional four: the 31x25 input grid
      is relatively small, and a fourth MaxPool would reduce the bottleneck to 1x1.
      Three stages retain a 3x3 bottleneck with a receptive field covering the input grid
    - SiLU activation and BatchNorm are used, consistent with this project's FPN decoder
    - With base_channels=32, the model has approximately 1.93M parameters, comparable
      to standalone Mamba (expand=2: 1.28M; expand=4: 1.74M). Parameter counts are
      reported in the training logs. Use --base-channels for capacity-matching tests
      (base=26: 1,274,105 parameters vs. 1,277,827 for Mamba e2)
This module includes data preprocessing and loss utilities consistent with the Mamba
module. Training and prediction scripts only need to import this module and Utilities.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utilities import GaussianNormalizer, GaussianNormalizer_out, loadmat_fields
# ==================================================
# Data preprocessing (shared by training and prediction; consistent with the Mamba module)
# ==================================================
# Fixed order of four input channels: two apparent-resistivity and two phase components
MT_COMPONENT_ORDER = ['Rhoxy', 'Phsxy', 'Rhoyx', 'Phsyx']
# Indices of apparent-resistivity channels (log10-transform); phase channels remain unchanged
RHO_CHANNELS = (0, 2)
def preprocess_mt_inputs(raw: dict, n: int) -> torch.Tensor:
    """
    Concatenate the four MT components from a .mat file and apply the log10 transformation
    Args:
        raw: Dictionary {component_name: ndarray (N, n_freq, n_obs)}
        n: Number of samples to use from the beginning of the dataset
    Returns:
        x:   (n, n_freq, n_obs, 4), with log10-transformed apparent-resistivity channels
    """
    x = torch.tensor(np.stack([raw[k][:n] for k in MT_COMPONENT_ORDER], axis=-1), dtype=torch.float32)
    for c in RHO_CHANNELS:
        x[..., c] = torch.log10(x[..., c] + 1e-10)
    return x
def preprocess_resistivity_targets(sig_k: np.ndarray, target_nz: int, target_ny: int) -> torch.Tensor:
    """
    Convert conductivity models to log10 resistivity and resample to the target grid
    Args:
        sig_k: (N, nz, ny) conductivity
    Returns:
        y: (N, target_nz, target_ny) log10 resistivity (log10(Ω·m))
    """
    y = torch.tensor(sig_k, dtype=torch.float32)
    if y.shape[1] != target_nz or y.shape[2] != target_ny:
        y = F.interpolate(y.unsqueeze(1), size=(target_nz, target_ny), mode='bilinear', align_corners=True).squeeze(1)
    return torch.log10(1.0 / (y + 1e-10))
def load_and_preprocess_data(train_path, val_path, test_path, ntrain, nval, ntest, target_nz, target_ny):
    """
    Load and preprocess training, validation, and test data (three separate .mat files)
    Fit normalization statistics only on the training set (encode validation/test data without refitting) to prevent data leakage.
    """
    fields = list(MT_COMPONENT_ORDER) + ['sig_k']
    print(f"Loading training data: {train_path}")
    train_raw = loadmat_fields(train_path, fields)
    x_train = preprocess_mt_inputs(train_raw, ntrain)
    y_train = preprocess_resistivity_targets(train_raw['sig_k'][:ntrain], target_nz, target_ny)
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
    # Gaussian normalization: statistics are computed exclusively from the training set
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
# U-Net architecture
# ==================================================
class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> SiLU) x 2, basic U-Net convolutional block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU())
    def forward(self, x):
        return self.block(x)
class Down(nn.Module):
    """MaxPool2d(2) downsampling + DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.block(x)
class Up(nn.Module):
    """Transposed-convolution upsampling + concatenated skip connection + DoubleConv.
    For odd-sized inputs (e.g., 31x25), the transposed-convolution output may differ
    from the skip feature by one pixel. Bilinear interpolation aligns their sizes
    without relying on the center-cropping convention of the original U-Net"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(2 * out_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
class UNetMT2DInversion(nn.Module):
    """
    Conventional U-Net for 2D MT inversion
    Input: (B, n_freq, n_obs, n_comp) MT observations (same interface as Mamba)
    Output: (B, target_nz, target_ny) normalized log10 resistivity model
    """
    def __init__(self, n_freq: int = 31,
                       n_obs: int = 25,
                       n_comp: int = 4,
                       target_nz: int = 32,
                       target_ny: int = 64,
                       base_channels: int = 32):
        super().__init__()
        self.n_freq = n_freq
        self.n_obs = n_obs
        self.n_comp = n_comp
        self.target_nz = target_nz
        self.target_ny = target_ny
        self.base_channels = base_channels
        b = base_channels
        # ---------- Encoder (three downsampling stages) ----------
        self.inc = DoubleConv(n_comp, b)       # Full resolution (n_freq, n_obs)
        self.down1 = Down(b, 2 * b)            # 1/2
        self.down2 = Down(2 * b, 4 * b)        # 1/4
        self.bottleneck = Down(4 * b, 8 * b)   # 1/8 (third MaxPool and bottleneck)
        # ---------- Decoder (three upsampling stages with skip connections) ----------
        self.up2 = Up(8 * b, 4 * b)
        self.up1 = Up(4 * b, 2 * b)
        self.up0 = Up(2 * b, b)
        # ---------- Output head ----------
        self.head = nn.Conv2d(b, 1, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_freq, n_obs, n_comp) -> (B, target_nz, target_ny)
        x = x.permute(0, 3, 1, 2)              # (B, n_comp, n_freq, n_obs)
        s0 = self.inc(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.bottleneck(s2)
        u2 = self.up2(s3, s2)
        u1 = self.up1(u2, s1)
        u0 = self.up0(u1, s0)
        out = self.head(u0)
        out = F.interpolate(out, size=(self.target_nz, self.target_ny), mode='bilinear', align_corners=False)
        return out.squeeze(1)                  # Remove the singleton channel dimension (out_channels=1)
# ==================================================
# Loss functions and utilities (consistent with the Mamba module)
# ==================================================
def _sobel_kernels(device) -> tuple:
    # Return Sobel convolution kernels on the specified device for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    return sobel_x, sobel_y
def compute_gradient(x: torch.Tensor, sobel_x, sobel_y) -> torch.Tensor:
    # Compute Sobel gradient magnitudes for (B,H,W) images
    grad_x = F.conv2d(x.unsqueeze(1), sobel_x, padding=1)
    grad_y = F.conv2d(x.unsqueeze(1), sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6).squeeze(1)
def edge_loss(pred: torch.Tensor, target: torch.Tensor, sobel_x, sobel_y) -> torch.Tensor:
    # L1 loss on Sobel gradients to improve boundary reconstruction
    return F.l1_loss(compute_gradient(pred, sobel_x, sobel_y), compute_gradient(target, sobel_x, sobel_y))
def edge_weighted_l1(pred: torch.Tensor, target: torch.Tensor, sobel_x, sobel_y, alpha: float = 3.0, tau_quantile: float = 0.9) -> torch.Tensor:
    """
    Gradient-weighted L1 loss: pixels with large target gradients (anomaly boundaries)
    receive a weight of 1 + alpha.
    - Use the specified quantile of target gradient magnitudes as threshold tau.
      Boundary pixels satisfy |grad(target)| > tau and receive weight 1 + alpha;
      other pixels receive weight 1. Return sum(w*|err|)/sum(w).
    - Weights depend only on the target (no gradient tracking); calculations are
      performed in fp32 to maintain stable quantile estimation under AMP.
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
    # Build AdamW parameter groups: exclude biases and normalization weights from weight decay
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},]
# ==================================================
# Self-test
# ==================================================
if __name__ == "__main__":
    model = UNetMT2DInversion()
    total = count_parameters(model)
    print(f"U-Net MT inversion network (base_channels={model.base_channels}) total parameters: {total:,}")
    x = torch.randn(4, 31, 25, 4)
    with torch.no_grad():
        y = model(x)
    print(f"Input {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (4, 32, 64), "Unexpected output shape"
    print("Network test completed!")
