import math
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Module hierarchy:
- Mamba: The complete model consisting of multiple stacked ResidualBlocks.
- ResidualBlock: Consists of a MambaBlock, RMSNorm, and a residual connection.
- MambaBlock: The core computational unit.
  Input (B, L, D) is first expanded to (B, L, 2*ED) through in_proj.
  The projected features are split into x and z, forming a gating mechanism.
  x passes through a 1D convolution, SiLU activation, and a selective state-space model (SSM).
  z passes through SiLU activation.
  Final output = SSM(x) * SiLU(z), followed by projection back to D dimensions.
- Dependency: Requires the official fused CUDA kernel (selective_scan_fn)
  from mamba_ssm, which is imported during initialization.
  An error is raised if the kernel is unavailable.
  No pure PyTorch (pscan) fallback is provided.
"""
# ------------------------------------------------------------
# 1. Configuration class (MambaConfig)
# ------------------------------------------------------------
@dataclass
class MambaConfig:
    # Basic dimensions
    d_model: int          # D: Hidden dimension of the model (input/output dimension)
    n_layers: int         # Number of Mamba layers (stacked ResidualBlocks)
    # Core SSM hyperparameters
    dt_rank: Union[int, str] = 'auto'   # Projection rank of delta. When set to 'auto', it is automatically
    d_state: int = 16       # N: State-space dimension (hidden state dimension)
    expand_factor: int = 2  # E: Expansion factor, with the internal dimension ED = E * D
    d_conv: int = 4         # Kernel size of the 1D convolution (typically small, e.g., 4, for local contextual feature mixing)
    # Initialization range of the delta parameter
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    # Normalization and initialization
    rms_norm_eps: float = 1e-5
    # Whether to use bias terms
    bias: bool = False
    conv_bias: bool = True
    def __post_init__(self):
        # Calculate the internal expanded dimension:
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
# ------------------------------------------------------------
# 2. Main model (Mamba) - Stacked layers
# ------------------------------------------------------------
class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        # Use ModuleList to store multiple residual blocks
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
    def forward(self, x):
        # x: (B, L, D) -> Output: (B, L, D)
        for layer in self.layers:
            x = layer(x)  # Perform forward propagation through each layer
        return x
# ------------------------------------------------------------
# 3. Residual block (ResidualBlock)
# ------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        # Core Mamba block
        self.mixer = MambaBlock(config)
        # Apply RMSNorm before the Mamba block (Pre-Norm architecture) to improve training stability.
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
    def forward(self, x):
        # Residual connection: Output = Mixer(Norm(x)) + x
        output = self.mixer(self.norm(x)) + x
        return output
# ------------------------------------------------------------
# 4. Core Mamba block (MambaBlock) - Main computational component
# ------------------------------------------------------------
class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        # 1. Input projection: Project D-dimensional features into 2 * ED dimensions to separate the x and z branches.
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)
        # 2. One-dimensional depthwise convolution (Depthwise Conv1d)
        # Perform convolution along the sequence dimension L. Setting groups=d_inner ensures independent convolution for each feature channel.
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,      # Grouped convolution to reduce the number of parameters
                              padding=config.d_conv - 1)  # Apply padding to preserve the sequence length; excess elements will be truncated afterward.
        # 3. Projection from x to delta, B, and C
        # Project input x with shape (B, L, ED) into (dt_rank + 2*N) dimensions to generate input-dependent parameters.
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)
        # 4. Delta projection: Project from dt_rank to ED dimensions
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)
        # 5. Delta initialization (critical for model convergence)
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize the delta bias using the inverse softplus transformation to ensure that the initial delta values lie within [dt_min, dt_max].
        dt = torch.exp(torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)).clamp(min=config.dt_init_floor)
        # Inverse softplus: x = y + log(-expm1(-y)), ensuring that softplus(bias) = dt.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # 6. SSM parameters A (state transition matrix) and D (skip connection parameter)
        # A has shape (ED, N). Although the effective state transition matrix is negative, the implementation initializes A_log using log(1..N) and computes -exp(A_log) during the forward pass.
        # The logarithmic parameterization ensures that the effective state transition matrix A remains negative, promoting stable exponential decay by maintaining negative diagonal entries.
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Learnable parameter
        self.A_log._no_weight_decay = True       # Exclude this parameter from weight decay regularization
        # D is the skip connection parameter with shape (ED).
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True
        # 7. Output projection: Project from ED back to D dimensions
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        # 8. Load the official fused CUDA kernel. This is the only supported computational path. The kernel is imported during initialization, and an error is raised immediately if it is unavailable.
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            self.selective_scan_cuda = selective_scan_fn
        except ImportError as e:
            raise ImportError(
                "Mamba forward propagation requires the official CUDA kernel (selective_scan_fn) from mamba_ssm."
                "Run this model in a GPU environment with mamba_ssm installed. No pure PyTorch fallback is provided.") from e
    # ------------------------------------------------------------
    # Main entry point for forward propagation
    # ------------------------------------------------------------
    def forward(self, x):
        # Shape of x: (B, L, D)
        _, L, _ = x.shape
        # Step 1: Input projection and feature splitting
        xz = self.in_proj(x)       # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # Each has shape (B, L, ED)
        # Step 2: x branch -> 1D convolution along the sequence dimension
        # Transpose (B, L, ED) to (B, ED, L) to match the input format required by Conv1d.the input format required by Conv1d.
        x = x.transpose(1, 2)         # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  # Retain the first L elements after convolution to remove excess elements introduced by padding.
        x = x.transpose(1, 2)         # (B, L, ED)
        x = F.silu(x)                 # Apply SiLU activation (Swish)
        # Step 3: Apply the selective state-space model (SSM). The CUDA kernel internally integrates the z gating operation (y * SiLU(z)).
        y = self.ssm(x, z)
        # Step 4: Project the output back to D dimensions
        return self.out_proj(y)
    # ------------------------------------------------------------
    # Core computation of the selective state-space model (SSM)
    # ------------------------------------------------------------
    def ssm(self, x, z):
        # x: (B, L, ED), z: (B, L, ED); The CUDA kernel internally fuses the z gating operation, and the kernel output is returned directly.
        # Prepare the SSM parameters. The CUDA kernel requires A and D to be in fp32 format.
        A = -torch.exp(self.A_log.float())  # (ED, N) Ensure A < 0 to maintain stable state transitions with negative real parts of the eigenvalues.
        D = self.D.float()                  # (ED)
        # 1. Generate delta, B, and C through the projection of x (input-dependent parameters).
        deltaBC = self.x_proj(x)     # (B, L, dt_rank + 2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        # 2. Compute delta (discretization step size): Project from dt_rank to ED dimensions.
        # Bias addition and softplus activation are omitted here because they are fused into the CUDA kernel.
        delta = self.dt_proj.weight @ delta.transpose(1, 2)  # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        #  3. Transpose tensors into the (B, *, L) layout required by the CUDA kernel and invoke the kernel.
        #  The CUDA kernel fuses softplus(delta + bias) and the y * SiLU(z) gating operation.
        x = x.transpose(1, 2)     # (B, ED, L)
        B = B.transpose(1, 2)     # (B, N, L)
        C = C.transpose(1, 2)     # (B, N, L)
        z = z.transpose(1, 2)     # (B, ED, L)
        y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
        return y.transpose(1, 2)  # Transpose back to (B, L, ED)
# ------------------------------------------------------------
# 5. RMSNorm (Root Mean Square Normalization)
# ------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Standard RMSNorm includes a learnable scaling parameter.
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # RMS: Compute the root mean square.
        # x / RMS(x) performs adaptive scaling without mean subtraction, which distinguishes RMSNorm from LayerNorm.
        return output * self.weight
