"""
Single Transformer Network for 2D MT Inversion Training (No CNN Branch) -- Standalone Module Version
=====================================================================
Network module: Trans_MT2D_Inversion.py (TransMT2DInversion). This script contains the complete
training pipeline, operates independently of any shared core, and can be debugged separately. The frontend
(data projection/positional encoding) and FPN decoder are identical to those of the standalone Mamba network
(Mamba_MT2D_Inversion.py); the only difference is the context encoder (unidirectional row/column Spatial Mamba replaced by standard row/column multi-head self-attention, MHSA + FFN).

Parameter matching: mlp_ratio is automatically determined by binary search to match the parameter count of the Mamba(expand) encoder
    (measured Mamba targets: expand=2 -> 532,736; expand=4 -> 998,656;
     for d_model=128 and 2 layers: expand=2 -> r≈2.38 (533,088),
     expand=4 -> r≈7.10 (998,938)), minimizing differences in parameter capacity;
    alternatively, --mlp-ratio can be set manually (without automatic matching).

Training protocol (identical to Mamba/UNet and implemented directly in this script):
    batch=20, AdamW(lr=1e-4, wd=1e-4 with parameter groups), 5 warmup epochs followed by cosine annealing,
    gradient clipping at 1.0, AMP(fp16), early stopping with patience=15/min_delta=5e-4, L1+0.2*Sobel loss,
    random seed 123
Outputs (written to ../Model and ../Log):
    Default --expand 4 (matched to Mamba e4, accuracy-oriented) -> tag=Trans
    --expand 2 (matched to Mamba e2)                     -> tag=TransAlignE2
"""
import argparse
import os
import sys
import time
# Path resolution: HERE is this script directory; ROOT is the project root, allowing execution from any working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')  # Save figures in headless environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# Import the network definition, data preprocessing, losses, and utilities from the standalone Transformer network module.
from Trans_MT2D_Inversion import (TransMT2DInversion, SpatialTransformerEncoder,
                                  _sobel_kernels, build_param_groups, count_parameters,
                                  edge_loss, edge_weighted_l1, load_and_preprocess_data)
from Utilities import GaussianNormalizer, GaussianNormalizer_out
# ==================================================
# Random seeds and acceleration settings
# ==================================================
torch.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================================================
# Fixed training protocol (identical to the Mamba/UNet training scripts)
# ==================================================
PROTOCOL = dict(
    # ---- Data ----
    ntrain=10000, nval=1000, ntest=1000,
    target_nz=32, target_ny=64,
    # ---- Shared network dimensions ----
    n_freq=31, n_obs=25, n_comp=4,
    d_model=128, n_layers=2,
    freq_range=(0.01, 1000.0), freq_descending=True,
    fpn_base_channels=64,
    # ---- Training ----
    batch_size=20, learning_rate=1e-4, weight_decay=1e-4,
    epochs=200, warmup_epochs=5, patience=15, min_delta=5e-4,
    use_amp=True, loss_type='l1_edge',
    edge_weight_alpha=3.0, edge_tau_quantile=0.9,
)
# ==================================================
# Parameter matching: use binary search to find the attention mlp_ratio matching the Mamba(expand) encoder parameter count.
# ==================================================
# Matching targets: measured parameter counts of SpatialMambaEncoder in Mamba_MT2D_Inversion.py
# (d_model=128 / 2 layers / d_state=16 / unidirectional scanning). Since this standalone Transformer module
# does not import the Mamba module, these values are stored as constants; update them if the Mamba encoder changes.
MAMBA_ENCODER_PARAMS = {2: 532_736, 4: 998_656}
def matched_mlp_ratio(d_model: int, n_layers: int, expand: int) -> float:
    """Find an attention mlp_ratio by binary search to match the Mamba(expand) encoder parameter count.
    Measured at d_model=128 with 2 layers: expand=2 -> r≈2.38 (533,088), expand=4 -> r≈7.10 (998,938)."""
    target = MAMBA_ENCODER_PARAMS[expand]
    lo, hi = 1.0, 16.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if count_parameters(SpatialTransformerEncoder(d_model, n_layers, d_model // 32, mid)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
# ==================================================
# Loss function wrapper
# ==================================================
def compute_loss(outputs, targets, sobel_x, sobel_y, loss_type, edge_alpha=3.0, edge_tau=0.9):
    """
    Compute the loss according to loss_type:
    'l1_edge'          = L1 + 0.2*Sobel (default)
    'mse'              = MSE only
    'weighted_l1_edge' = gradient-weighted L1 + 0.2*Sobel (emphasizes anomaly boundaries)
    """
    if loss_type == 'l1_edge':
        return F.l1_loss(outputs, targets) + 0.2 * edge_loss(outputs, targets, sobel_x, sobel_y)
    elif loss_type == 'mse':
        return F.mse_loss(outputs, targets)
    elif loss_type == 'weighted_l1_edge':
        return edge_weighted_l1(outputs, targets, sobel_x, sobel_y, alpha=edge_alpha, tau_quantile=edge_tau) \
               + 0.2 * edge_loss(outputs, targets, sobel_x, sobel_y)
    raise ValueError(f"Unknown loss_type: {loss_type}")
def evaluate(model, data_iter, sobel_x, sobel_y, use_amp=False, loss_type='mse', edge_alpha=3.0, edge_tau=0.9):
    # Compute the mean loss on a dataset with the same loss function and AMP setting as training for consistent monitoring.
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in data_iter:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = compute_loss(outputs, targets, sobel_x, sobel_y, loss_type, edge_alpha, edge_tau)
            losses.append(loss.float())
    return torch.stack(losses).mean().item()
# ==================================================
# Plot loss curves
# ==================================================
def plot_loss_curves(train_losses, val_losses, learning_rates, best_epoch, save_path, tag='', test_losses=None):
    # Two y-axes: loss on the left (log scale) and learning rate on the right.
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, train_losses, 'b-', label='Train loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val loss')
    if test_losses:
        ax1.plot(epochs, test_losses, 'm-', alpha=0.8, label='Test loss')
    if best_epoch and best_epoch <= len(train_losses):
        ax1.axvline(best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best epoch ({best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='center right')
    ax2 = ax1.twinx()
    ax2.plot(epochs, learning_rates, 'g:', alpha=0.6, label='Learning rate')
    ax2.set_ylabel('Learning rate')
    ax2.legend(loc='upper right')
    fig.suptitle(f'{tag} MT2D Inversion Training Curves')
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
# ==================================================
# Main training procedure
# ==================================================
def run_training(expand: int = 4, mlp_ratio=None, dry_run: bool = False):
    """
    Args:
        expand: Parameter-matching target, i.e., the Mamba(expand) encoder parameter count
                (default 4 for accuracy-oriented training, tag=Trans; 2 matches e2, tag=TransAlignE2)
        mlp_ratio: FFN width multiplier; None=automatic binary-search matching (recommended)
        dry_run: Build the model and print configuration, parameter counts, and output filenames without loading data or training
    """
    tag = 'Trans'
    P = PROTOCOL
    # ==================== Data configuration ====================
    train_path = os.path.join(ROOT, "Training_dataset", "Train_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_10000.mat")
    val_path = os.path.join(ROOT, "Training_dataset", "Validate_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_1000.mat")
    test_path = os.path.join(ROOT, "Training_dataset", "Test_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_1000.mat")
    # ==================== Parameter matching ====================
    if mlp_ratio is None:
        attention_mlp_ratio = matched_mlp_ratio(P['d_model'], P['n_layers'], expand)
        aligned_note = 'Automatically matched to Mamba encoder parameter count'
    else:
        attention_mlp_ratio = float(mlp_ratio)
        aligned_note = 'Manually specified (not matched)'
    # Add a suffix for nondefault matching target (4) to avoid overwriting default checkpoints and logs.
    if expand != 4:
        tag = tag + f'AlignE{expand}'
    # Model configuration saved in the checkpoint for reconstruction during prediction.
    model_config = {
        'n_freq': P['n_freq'], 'n_obs': P['n_obs'], 'n_comp': P['n_comp'],
        'target_nz': P['target_nz'], 'target_ny': P['target_ny'],
        'd_model': P['d_model'], 'n_layers': P['n_layers'],
        'freq_range': P['freq_range'], 'freq_descending': P['freq_descending'],
        'fpn_base_channels': P['fpn_base_channels'],
        'n_heads': P['d_model'] // 32,          # head_dim=32, d_model=128 -> 4 heads
        'mlp_ratio': attention_mlp_ratio,
        'aligned_to_expand': expand}            # Parameter-matching target (metadata only; ignored during prediction)
    if dry_run:
        model = TransMT2DInversion(**{k: v for k, v in model_config.items() if k != 'aligned_to_expand'})
        print(f"[dry-run] Standalone Transformer variant ({tag}): no CNN branch")
        print(f"[dry-run] encoder=attention, CNN branch=False, mlp_ratio={attention_mlp_ratio:.4f} "
              f"({aligned_note}), matched against Mamba expand={expand}")
        print(f"[dry-run] Context encoder parameters: {count_parameters(model.context_encoder):,} | "
              f"Target Mamba(expand={expand}) encoder parameters: {MAMBA_ENCODER_PARAMS[expand]:,} | "
              f"Total parameters: {count_parameters(model):,}")
        print(f"[dry-run] Outputs: ../Model/{tag}_MT2D_Best_nz{P['target_nz']}_ny{P['target_ny']}.pth | "
              f"../Log/{tag}_MT2D_Loss_log.csv + {tag}_MT2D_LossCurves_nz{P['target_nz']}_ny{P['target_ny']}.png")
        return
    # ==================== Unpack protocol parameters ====================
    ntrain, nval, ntest = P['ntrain'], P['nval'], P['ntest']
    target_nz, target_ny = P['target_nz'], P['target_ny']
    batch_size, learning_rate, weight_decay = P['batch_size'], P['learning_rate'], P['weight_decay']
    epochs, warmup_epochs = P['epochs'], P['warmup_epochs']
    patience, min_delta = P['patience'], P['min_delta']
    use_amp, loss_type = P['use_amp'], P['loss_type']
    edge_weight_alpha, edge_tau_quantile = P['edge_weight_alpha'], P['edge_tau_quantile']
    # ==================== Print configuration ====================
    print("=" * 60)
    print("2D MT inversion training: standalone Transformer (no CNN branch)")
    print(f"Network configuration (encoder=attention, CNN branch=False, mlp_ratio={attention_mlp_ratio:.4f} "
          f"({aligned_note}), Mamba matching target expand={expand})")
    print("Controlled settings: data/sample counts/batch/optimizer/scheduler/early stopping/AMP/loss/seed are identical to standalone Mamba")
    print("=" * 60)
    print(f"Training samples: {ntrain}, validation samples: {nval}, test samples: {ntest}")
    print(f"Target resolution: {target_nz} x {target_ny}")
    print(f"Batch size: {batch_size}, learning rate: {learning_rate}, epochs: {epochs}")
    print(f"Early stopping patience: {patience}, mixed precision AMP: {use_amp}, loss: {loss_type}")
    print("=" * 60)
    # ==================== Data loading and preprocessing ====================
    try:
        (x_train, y_train, x_val, y_val, x_test, y_test,
         x_normalizer, y_normalizer) = load_and_preprocess_data(
            train_path, val_path, test_path, ntrain, nval, ntest, target_nz, target_ny)
    except Exception as e:
        print(f"Data loading failed: {e}")
        print("Please check the data paths")
        return
    # Create DataLoaders (num_workers=0 is a Windows-safe setting).
    data_iter_train = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    data_iter_val = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    data_iter_test = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # ==================== Model initialization ====================
    print(f"\nCreating {tag} MT inversion model...")
    model = TransMT2DInversion(**{k: v for k, v in model_config.items() if k != 'aligned_to_expand'}).to(device)
    print(f"Context encoder parameters: {count_parameters(model.context_encoder):,} | Total model parameters: {count_parameters(model):,}")
    # ==================== Output directories ====================
    MODEL_DIR = os.path.join(ROOT, "Model")   # Best model (.pth)
    LOG_DIR = os.path.join(ROOT, "Log")       # Loss log CSV and loss curve PNG
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    loss_fig_path = os.path.join(LOG_DIR, f"{tag}_MT2D_LossCurves_nz{target_nz}_ny{target_ny}.png")
    # ==================== Optimizer and learning-rate scheduler ====================
    optimizer = torch.optim.AdamW(build_param_groups(model, weight_decay), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    x_normalizer.to(device)
    y_normalizer.to(device)
    sobel_x, sobel_y = _sobel_kernels(device)
    # ==================== Logging and best-model checkpointing ====================
    log_filename = os.path.join(LOG_DIR, f'{tag}_MT2D_Loss_log.csv')
    log_header = "epoch,train_loss,val_loss,test_loss,learning_rate,time_sec\n"
    if os.path.isfile(log_filename):
        with open(log_filename) as f:
            if f.readline() != log_header:
                backup = log_filename.replace('.csv', '_old.csv')
                try:
                    os.replace(log_filename, backup)
                    print(f"Previous log format differs; backed up to: {backup}")
                except PermissionError:
                    log_filename = log_filename.replace('.csv', time.strftime('_%Y%m%d_%H%M%S.csv'))
                    print(f"Previous log is in use and cannot be backed up; writing this run to: {log_filename}")
    if not os.path.isfile(log_filename):
        with open(log_filename, 'w') as f:
            f.write(log_header)
    model_save_path = os.path.join(MODEL_DIR, f"{tag}_MT2D_Best_nz{target_nz}_ny{target_ny}.pth")
    def save_best_model(path, epoch):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': avg_val_loss,
                    'x_normalizer_mean': x_normalizer.mean,
                    'x_normalizer_std': x_normalizer.std,
                    'y_normalizer_mean': y_normalizer.mean,
                    'y_normalizer_std': y_normalizer.std,
                    'model_config': model_config,}, path)
    # ==================== Training loop ====================
    print("\nStarting training...")
    print("=" * 80)
    train_losses, val_losses, test_losses, learning_rates = [], [], [], []
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        # Linear learning-rate warmup over the first warmup_epochs epochs.
        if epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            current_lr = warmup_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']
        # ========== Training phase ==========
        model.train()
        batch_losses = []
        for inputs, targets in data_iter_train:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = compute_loss(outputs, targets, sobel_x, sobel_y, loss_type, edge_weight_alpha, edge_tau_quantile)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(loss.detach())
        avg_train_loss = torch.stack(batch_losses).mean().item()
        train_losses.append(avg_train_loss)
        learning_rates.append(current_lr)
        # ========== Validation and test evaluation (excluded from training) ==========
        avg_val_loss = evaluate(model, data_iter_val, sobel_x, sobel_y, use_amp, loss_type, edge_weight_alpha, edge_tau_quantile)
        val_losses.append(avg_val_loss)
        avg_test_loss = evaluate(model, data_iter_test, sobel_x, sobel_y, use_amp, loss_type, edge_weight_alpha, edge_tau_quantile)
        test_losses.append(avg_test_loss)
        # Cosine annealing after warmup.
        if epoch >= warmup_epochs:
            scheduler.step()
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch: {epoch + 1}/{epochs} | Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f} | Test: {avg_test_loss:.6f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
        with open(log_filename, 'a') as f:
            f.write(f"{epoch + 1},{avg_train_loss:.6f},{avg_val_loss:.6f},"
                    f"{avg_test_loss:.6f},{current_lr:.6f},{epoch_time:.2f}\n")
        if (epoch + 1) % 10 == 0:
            plot_loss_curves(train_losses, val_losses, learning_rates, best_epoch, loss_fig_path, tag, test_losses)
        # ========== Early stopping and best-model checkpointing ==========
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0
            save_best_model(model_save_path, epoch + 1)
            print(f"Validation loss improved; saved best model: {model_save_path}")
        else:
            early_stop_counter += 1
            print(f"Validation loss did not improve ({early_stop_counter}/{patience})")
            if early_stop_counter >= patience:
                print(f"No improvement for {patience} consecutive epochs; early stopping triggered")
                break
    # ==================== Training complete: evaluate best model on test set ====================
    total_time = time.time() - total_start_time
    plot_loss_curves(train_losses, val_losses, learning_rates, best_epoch, loss_fig_path, tag, test_losses)
    print(f"Loss curve saved: {loss_fig_path}")
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = evaluate(model, data_iter_test, sobel_x, sobel_y, use_amp, loss_type, edge_weight_alpha, edge_tau_quantile)
    print("=" * 80)
    print("Training complete!")
    print(f"Total training time: {total_time:.2f}s ({total_time / 60:.1f} minutes), epochs completed: {len(train_losses)}, best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss (best model): {test_loss:.6f}")
    print("=" * 80)
    print(f"Best model: {model_save_path}")
    print(f"Training log: {log_filename}")
    print(f"Loss curves: {loss_fig_path}")
    print("=" * 80)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standalone Transformer 2D MT inversion training')
    parser.add_argument('--expand', type=int, default=4, choices=(2, 4),
                        help='Parameter-matching target: match Mamba(expand) encoder parameters; default 4 (accuracy-oriented, tag=Trans); '
                             '2 matches the e2 baseline (tag=TransAlignE2)')
    parser.add_argument('--mlp-ratio', type=float, default=None,
                        help='Manually set the attention FFN width multiplier (default None=automatic binary-search matching, recommended)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build the model and print configuration, parameter counts, and output filenames without loading data or training')
    args = parser.parse_args()
    run_training(expand=args.expand, mlp_ratio=args.mlp_ratio, dry_run=args.dry_run)
