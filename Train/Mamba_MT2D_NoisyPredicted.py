"""
Standalone Mamba Network for 2D MT Inversion Prediction — Noise Sensitivity Test
=========================================================================

"""
import argparse
import os
import sys
import time
# Resolve paths relative to this directory (HERE) and the project root (ROOT),
# so that data and outputs can be located regardless of the working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')  # Save figures even in headless environments.
import matplotlib.pyplot as plt
import numpy as np
import torch
# Import the network, data preprocessing functions, and utilities from the Mamba modules.
from Mamba_MT2D_Inversion import (MambaMT2DInversion, MT_COMPONENT_ORDER, preprocess_mt_inputs, preprocess_resistivity_targets)
from Utilities import GaussianNormalizer, GaussianNormalizer_out, loadmat_fields
# ==================================================
# Random seed and acceleration settings
# ==================================================
torch.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================================================
# Model loading and utilities
# ==================================================
def _synchronize_device():
    # Synchronize the GPU for accurate timing.
    if device.type == 'cuda':
        torch.cuda.synchronize()
def load_model_and_normalizers(model_path: str):
    """Load the trained model and normalizers (the checkpoint contains model_config)."""
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['model_config']
    print("Model configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    model = MambaMT2DInversion(
        n_freq=cfg['n_freq'],
        n_obs=cfg['n_obs'],
        n_comp=cfg['n_comp'],
        target_nz=cfg['target_nz'],
        target_ny=cfg['target_ny'],
        d_model=cfg['d_model'],
        n_mamba_layers=cfg['n_mamba_layers'],
        freq_range=tuple(cfg.get('freq_range', (0.01, 1e3))),
        freq_descending=cfg.get('freq_descending', True),
        fpn_base_channels=cfg.get('fpn_base_channels', 64),
        mamba_d_state=cfg.get('mamba_d_state', 16),
        mamba_expand=cfg.get('mamba_expand', 2),
        mamba_bidirectional=cfg.get('mamba_bidirectional', False),
        use_cnn_branch=cfg.get('use_cnn_branch', True),
        encoder_type=cfg.get('encoder_type', 'mamba'),
        attention_mlp_ratio=cfg.get('attention_mlp_ratio', 4.0)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # Reconstruct normalizers from stored statistics. Dummy tensors establish the
    # required shapes before their statistics are overwritten by checkpoint values.
    dummy_x = torch.zeros(1, cfg['n_freq'], cfg['n_obs'], cfg['n_comp'])
    dummy_y = torch.zeros(1, cfg['target_nz'], cfg['target_ny'])
    x_normalizer = GaussianNormalizer_out(dummy_x)
    x_normalizer.mean = checkpoint['x_normalizer_mean'].to(device)
    x_normalizer.std = checkpoint['x_normalizer_std'].to(device)
    y_normalizer = GaussianNormalizer(dummy_y)
    y_normalizer.mean = checkpoint['y_normalizer_mean'].to(device)
    y_normalizer.std = checkpoint['y_normalizer_std'].to(device)
    print(f"Model loaded successfully. Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, x_normalizer, y_normalizer, cfg
# ---- SSIM convention: consistent with MATLAB Draw_Inversion_Results.m / SSIM_Distribution_Plot.py ----
# Default MATLAB IPT ssim() settings: 11x11 Gaussian kernel with sigma=1.5 /
# replicate padding / double-precision input with L=1 / K=[0.01,0.03] /
# mean of the full-image SSIM map (inputs are log10 resistivity values).
# Numerically verified against MATLAB R2024b (error <4e-14). Note: SSIM values
# cannot be directly compared with the previous dynamic data_range implementation,
# because using L=1 systematically lowers the scores.
_SSIM_SIGMA = 1.5
_SSIM_R = int(np.ceil(3 * _SSIM_SIGMA))              # 5 -> kernel size 2*5+1 = 11
_ax = np.arange(-_SSIM_R, _SSIM_R + 1, dtype=np.float64)
_g = np.exp(-(_ax ** 2) / (2.0 * _SSIM_SIGMA ** 2))
_SSIM_KERNEL = np.outer(_g, _g)
_SSIM_KERNEL /= _SSIM_KERNEL.sum()                   # == fspecial('gaussian', 11, 1.5)
def compute_ssim(img1, img2, dynamic_range=1.0, K1=0.01, K2=0.03):
    """Compute structural similarity (SSIM) using the same convention as
    ssim(log10(rho_inv), log10(rho_true)) in MATLAB Draw_Inversion_Results.m
    (default MATLAB IPT ssim settings)."""
    from scipy.ndimage import convolve
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    C1 = (K1 * dynamic_range) ** 2
    C2 = (K2 * dynamic_range) ** 2
