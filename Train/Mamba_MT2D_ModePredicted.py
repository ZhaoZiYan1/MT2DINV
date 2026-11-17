"""
Single-Mamba network for 2D magnetotelluric (MT) inversion prediction — TE/TM/TEM mode version.
================================================================
Designed to work with Mamba_MT2D_ModeTrain.py
"""
import argparse
import os
import sys
import time
# Path configuration: Define the current directory (HERE) and project root directory (ROOT) to ensure that data and output files can be located from any working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
# Import the network definition, data preprocessing functions, and utilities from the Mamba-specific network module.
from Mamba_MT2D_Inversion import (MambaMT2DInversion, MT_COMPONENT_ORDER, preprocess_mt_inputs, preprocess_resistivity_targets)
from Utilities import GaussianNormalizer, GaussianNormalizer_out, loadmat_fields
# ==================================================
# Inversion mode definitions (consistent with Mamba_MT2D_ModeTrain.py)
# ==================================================
MODE_CHANNEL_IDX = {'TE': [0, 1], 'TM': [2, 3], 'TEM': [0, 1, 2, 3]}
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
# Model loading and utility functions
# ==================================================
def _synchronize_device():
    # Synchronize the GPU for accurate inference-time measurement.
    if device.type == 'cuda':
        torch.cuda.synchronize()
def load_model_and_normalizers(model_path: str):
    """Load a mode-specific trained model and its normalizers.
    The checkpoint contains model_config and mode.
    Return (model, x_normalizer, y_normalizer, cfg, mode, ch_idx)."""
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['model_config']
    # Inversion mode: Read the mode stored in the checkpoint. For older checkpoints without a mode field, use TEM as the default fallback.
    mode = checkpoint.get('mode', 'TEM')
    ch_idx = checkpoint.get('mode_channels', MODE_CHANNEL_IDX.get(mode, [0, 1, 2, 3]))
    if mode not in MODE_CHANNEL_IDX:
        raise ValueError(f"Unrecognized mode={mode} in checkpoint (available modes {'/'.join(MODE_CHANNEL_IDX)})")
    if len(ch_idx) != cfg['n_comp']:
        raise ValueError(f"Checkpoint mode_channels ({ch_idx}) does not match n_comp({cfg['n_comp']}) ")
    print(f"Inversion mode: {mode} -> Input channels {[MT_COMPONENT_ORDER[i] for i in ch_idx]}")
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
    # Reconstruct the normalizers using saved statistics.
    dummy_x = torch.zeros(1, cfg['n_freq'], cfg['n_obs'], cfg['n_comp'])
    dummy_y = torch.zeros(1, cfg['target_nz'], cfg['target_ny'])
    x_normalizer = GaussianNormalizer_out(dummy_x)
    x_normalizer.mean = checkpoint['x_normalizer_mean'].to(device)
    x_normalizer.std = checkpoint['x_normalizer_std'].to(device)
    y_normalizer = GaussianNormalizer(dummy_y)
    y_normalizer.mean = checkpoint['y_normalizer_mean'].to(device)
    y_normalizer.std = checkpoint['y_normalizer_std'].to(device)
    print(f"Model loaded successfully; number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, x_normalizer, y_normalizer, cfg, mode, ch_idx
# Default parameters of MATLAB IPT ssim()
_SSIM_SIGMA = 1.5
_SSIM_R = int(np.ceil(3 * _SSIM_SIGMA))              # 5 -> Kernel size: 2*5+1 = 11.
_ax = np.arange(-_SSIM_R, _SSIM_R + 1, dtype=np.float64)
_g = np.exp(-(_ax ** 2) / (2.0 * _SSIM_SIGMA ** 2))
_SSIM_KERNEL = np.outer(_g, _g)
_SSIM_KERNEL /= _SSIM_KERNEL.sum()                   # == fspecial('gaussian', 11, 1.5)
def compute_ssim(img1, img2, dynamic_range=1.0, K1=0.01, K2=0.03):
    from scipy.ndimage import convolve
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    C1 = (K1 * dynamic_range) ** 2
    C2 = (K2 * dynamic_range) ** 2
    kw = dict(mode='nearest')                         # imfilter 'replicate'
    mu1 = convolve(img1, _SSIM_KERNEL, **kw)
    mu2 = convolve(img2, _SSIM_KERNEL, **kw)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = convolve(img1 ** 2, _SSIM_KERNEL, **kw) - mu1_sq
    sigma2_sq = convolve(img2 ** 2, _SSIM_KERNEL, **kw) - mu2_sq
    sigma12 = convolve(img1 * img2, _SSIM_KERNEL, **kw) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())
# ==================================================
# Batch prediction (mode-specific input channel selection)
# ==================================================
def predict_test_set(model, x_normalizer, y_normalizer, data_path: str, model_config: dict,
                     ch_idx, batch_size: int = 16):
    """Perform batch prediction on the entire test dataset and return the predicted results, SSIM scores, inference times, and related information.
    First preprocess the inputs as standard four-channel MT data (applying log10 to apparent resistivity), then select the channels corresponding to the inversion mode."""
    print(f"\nLoading test data: {data_path}")
    raw = loadmat_fields(data_path, list(MT_COMPONENT_ORDER) + ['sig_k'])
    n_samples = raw['Rhoxy'].shape[0]
    print(f"Number of test samples: {n_samples}")
    x_raw = preprocess_mt_inputs(raw, n_samples)[..., ch_idx]   # (N, n_freq, n_obs, n_comp) with channels selected according to the inversion mode.
    # Verify that the input data dimensions match the training configuration.
    expected_geom = (model_config['n_freq'], model_config['n_obs'], model_config['n_comp'])
    if tuple(x_raw.shape[1:]) != expected_geom:
        raise ValueError(
            f"Input dimensions after channel selection {tuple(x_raw.shape[1:])} do not match the model configuration {expected_geom}  "
            f"Please check whether --data and --model are compatible.")
    y_true = preprocess_resistivity_targets(
        raw['sig_k'], model_config['target_nz'], model_config['target_ny'])  # (N, nz, ny) log10 resistivity.
    x_normalized = x_normalizer.encode(x_raw.to(device))
    preds = []
    ssim_scores = []
    per_sample_ms = []
    print("\nStarting batch prediction...")
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            x_batch = x_normalized[i:i + batch_size]
            # ---- Timing interval: Forward propagation and decoding only ----
            _synchronize_device()
            t_start = time.perf_counter()
            out = model(x_batch)
            y_pred = y_normalizer.decode(out).cpu()
            t_end = time.perf_counter()
            preds.append(y_pred)
            n_in_batch = y_pred.shape[0]
            per_sample_ms.extend([(t_end - t_start) * 1000.0 / n_in_batch] * n_in_batch)
            for j in range(y_pred.shape[0]):
                y_true_np = y_true[i + j].numpy()
                y_pred_np = y_pred[j].numpy()
                ssim_scores.append(compute_ssim(y_true_np, y_pred_np))
            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= n_samples:
                print(f"Processed: {min(i + batch_size, n_samples)}/{n_samples} samples")
    inference_time_ms_arr = np.array(per_sample_ms)
    inference_time = float(inference_time_ms_arr.sum()) / 1000.0
    y_pred_all = torch.cat(preds, dim=0)
    print(f"\nPrediction completed!")
    print(f"Total inference time (forward + decoding, excluding SSIM): {inference_time:.2f} seconds")
    print(f"Per-sample inference time: Mean {inference_time_ms_arr.mean():.2f} | "
          f"Min {inference_time_ms_arr.min():.2f} | Max {inference_time_ms_arr.max():.2f} ms "
          f"(averaged within each batch, batch_size={batch_size})")
    print(f"Mean SSIM: {np.mean(ssim_scores):.6f} ± {np.std(ssim_scores):.6f}")
    # ===== Rank SSIM scores in descending order =====
    ssim_arr = np.array(ssim_scores)
    sort_order = np.argsort(-ssim_arr, kind='stable')
    n_top = min(100, n_samples)
    print(f"\nTop {n_top} samples ranked by SSIM (descending order):")
    for rank, idx in enumerate(sort_order[:n_top], start=1):
        print(f" Rank{rank:>2d}: Sample index {idx}, SSIM = {ssim_arr[idx]:.6f}")
    n_bottom = min(100, n_samples)
    print(f"Bottom {n_bottom} samples ranked by SSIM:")
    for k, idx in enumerate(sort_order[-n_bottom:][::-1], start=1):
        print(f" Bottom rank{k}: Sample index {idx}, SSIM = {ssim_arr[idx]:.6f}")
    return {
        'x_raw': x_raw.numpy(),
        'y_true': y_true.numpy(),
        'y_pred': y_pred_all.numpy(),
        'ssim': ssim_arr,
        'ssim_sorted': ssim_arr[sort_order],
        'ssim_sorted_sample_idx': sort_order,
        'inference_time': inference_time,
        'inference_time_ms': inference_time_ms_arr,
        'n_samples': n_samples
    }
# ==================================================
# Prediction entry point
# ==================================================
def run_prediction(model_name: str, data_name: str, batch_size: int = 16, samples=(891,)):
    """
    Args:
        model_name: Model filename (relative to the ../Model directory) or an absolute path. The checkpoint should be generated by ModeTrain and contain the mode field.
        data_name:  Test dataset filename (relative to the ../Training_dataset directory) or an absolute path. The noise-free test dataset is used by default.
        batch_size: Batch size for inference.
        samples:    List of sample indices for which individual figures and MAT files should be generated and saved.
    """
    model_path = model_name if os.path.isabs(model_name) else os.path.join(ROOT, "Model", model_name)
    test_data_path = data_name if os.path.isabs(data_name) else os.path.join(ROOT, "Training_dataset", data_name)
    model_basename = os.path.splitext(os.path.basename(model_path))[0]
    mode_base_dir = os.path.join(ROOT, "Result", "Mamba", "Mode")   # Organize results into separate TE, TM, and TEM subdirectories according to the inversion mode.
    os.makedirs(mode_base_dir, exist_ok=True)
    print("=" * 70)
    print("Single-Mamba 2D MT inversion prediction (TE/TM/TEM mode version).")
    print("=" * 70)
    print(f"Model path: {model_path}")
    print(f"Test data: {test_data_path}")
    print(f"Results root directory: {mode_base_dir}")
    print("=" * 70)
    # ==================== Load model (automatic inversion mode identification) ====================
    try:
        model, x_normalizer, y_normalizer, model_config, mode, ch_idx = load_model_and_normalizers(model_path)
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    result_dir = os.path.join(mode_base_dir, mode)
    os.makedirs(result_dir, exist_ok=True)
    print(f"Mode-specific results directory ({mode}): {result_dir}")
    # ==================== Batch prediction ====================
    try:
        results = predict_test_set(model, x_normalizer, y_normalizer, test_data_path, model_config,
                                   ch_idx, batch_size=batch_size)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    # ==================== Generate comparison figures and MAT files for specified sample indices ====================
    sample_indices = list(samples)
    print(f"\nGenerating comparison figures for specified samples...")
    print(f"Specified sample indices: {sample_indices}")
    ch_names = [MT_COMPONENT_ORDER[i] for i in ch_idx]   # Names of the input channels used by the selected mode.
    from scipy.io import savemat
    for sample_idx in sample_indices:
        if sample_idx < 0 or sample_idx >= results['n_samples']:
            print(f" Sample index {sample_idx} is out of range (0-{results['n_samples'] - 1}), skipping.")
            continue
        y_true_sample = results['y_true'][sample_idx]
        y_pred_sample = results['y_pred'][sample_idx]
        ssim_sample = results['ssim'][sample_idx]
        ssim_rank = int(np.where(results['ssim_sorted_sample_idx'] == sample_idx)[0][0]) + 1
        # Create a four-panel comparison figure: Ground-truth model | Predicted model | Relative error | Scatter plot.
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_basename} ({mode} mode) - Sample {sample_idx} (SSIM={ssim_sample:.6f}, Rank {ssim_rank}/{results["n_samples"]})',
                     fontsize=15, fontweight='bold')
        error = np.abs((10 ** y_pred_sample - 10 ** y_true_sample) / (10 ** y_true_sample + 1e-10))
        vmin = min(y_true_sample.min(), y_pred_sample.min())
        vmax = max(y_true_sample.max(), y_pred_sample.max())
        im1 = axes[0, 0].imshow(y_true_sample, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('True Resistivity Model', fontweight='bold')
        axes[0, 0].set_xlabel('Y Direction')
        axes[0, 0].set_ylabel('Z Direction (Depth)')
        plt.colorbar(im1, ax=axes[0, 0], label='log10(ρ / Ω·m)')
        im2 = axes[0, 1].imshow(y_pred_sample, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Predicted Resistivity Model', fontweight='bold')
        axes[0, 1].set_xlabel('Y Direction')
        axes[0, 1].set_ylabel('Z Direction (Depth)')
        plt.colorbar(im2, ax=axes[0, 1], label='log10(ρ / Ω·m)')
        im3 = axes[1, 0].imshow(error, aspect='auto', cmap='viridis', vmin=0, vmax=np.percentile(error, 95))
        axes[1, 0].set_title('Relative Error', fontweight='bold')
        axes[1, 0].set_xlabel('Y Direction')
        axes[1, 0].set_ylabel('Z Direction (Depth)')
        plt.colorbar(im3, ax=axes[1, 0], label='|ρ_pred - ρ_true| / ρ_true')
        axes[1, 1].scatter(y_true_sample.flatten(), y_pred_sample.flatten(), alpha=0.6, s=1)
        axes[1, 1].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 1].set_xlabel('True log10(ρ / Ω·m)')
        axes[1, 1].set_ylabel('Predicted log10(ρ / Ω·m)')
        axes[1, 1].set_title('True vs Predicted', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].text(0.02, 0.98, f'SSIM: {ssim_sample:.6f} ({mode})', transform=axes[1, 1].transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout()
        fig_path = os.path.join(result_dir, f"{model_basename}_Sample{sample_idx}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Saved: {fig_path}")
        # Save an individual MAT file for this sample. Only the input channels used by the selected mode are saved. Apparent resistivity values are converted back to the linear domain.
        inputs_save = {}
        for j, name in enumerate(ch_names):
            ch_data = results['x_raw'][sample_idx][..., j]
            inputs_save[name] = (10 ** ch_data) if name.startswith('Rho') else ch_data
        sample_save = {
            'sample_idx': sample_idx,
            'ssim': float(ssim_sample),
            'ssim_rank': ssim_rank,
            'mode': mode,
            'mode_channels': list(ch_idx),
            'inference_time_ms': float(results['inference_time_ms'][sample_idx]),
            'rho_true': 10 ** y_true_sample,
            'sig_k_true': 1.0 / (10 ** y_true_sample),
            'rho_pred': 10 ** y_pred_sample,
            'sig_k_pred': 1.0 / (10 ** y_pred_sample),
            'y_true_log10': y_true_sample,
            'y_pred_log10': y_pred_sample,
        }
        sample_save.update(inputs_save)
        sample_mat_path = os.path.join(result_dir, f"{model_basename}_Sample{sample_idx}.mat")
        savemat(sample_mat_path, sample_save)
        print(f"  ✅ Saved: {sample_mat_path}")
    print(f"\n✅ Prediction completed!")
    print(f"\n💡 Note: To compare the three modes, run --model Mamba_MT2D_Best_nz32_ny64_TE.pth / _TM.pth / _TEM.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single-Mamba 2D MT inversion prediction (TE/TM/TEM mode version, with automatic mode from the checkpoint)')
    parser.add_argument('--model', default='Mamba_MT2D_Best_nz32_ny64_TE.pth',
                        help='Model filename (relative to the ../Model directory) or an absolute path. Example ModeTrain checkpoint: Mamba_MT2D_Best_nz32_ny64_TE.pth')
    parser.add_argument('--data', default='Test_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_1000.mat',
                        help='Test dataset filename (relative to the ../Training_dataset directory) or an absolute path')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--samples', type=int, nargs='+', default=[91],
                        help='Sample indices for generating individual figures and saving MAT files (e.g., --samples 891 15 42)')
    args = parser.parse_args()
    run_prediction(model_name=args.model, data_name=args.data, batch_size=args.batch_size, samples=args.samples)
