"""
Standalone training script for a standard U-Net baseline for 2D magnetotelluric (MT) inversion
=========================================================================================
Network module: UNet_MT2D_Inversion.py (UNetMT2DInversion, a standard end-to-end U-Net
encoder-decoder with three downsampling stages, transposed-convolution upsampling, and
skip connections; no data projection, continuous positional encoding, row/column context
encoder, or FPN). This script implements the complete training pipeline independently
and can be debugged without a shared core.
Training protocol (identical to Mamba/Transformer for a controlled comparison):
    batch=20, AdamW(lr=1e-4, wd=1e-4 with parameter groups), five warmup epochs followed
    by cosine annealing, gradient clipping at 1.0, AMP(fp16), early stopping with
    patience=15/min_delta=5e-4, L1+0.2*Sobel loss, and random seed 123.
Parameter counts (base_channels is configurable and counts are reported in the logs):
    base=32 (default): 1,928,705 | base=26: 1,274,105 (approximately equal to the
    1,277,827 parameters of standalone Mamba expand=2, a 0.3% difference suitable for
    capacity-matched experiments) | base=24: 1,085,953 | base=16: 483,585.
Outputs (saved under ../Model and ../Log):
    Default --base-channels 32 -> tag=UNet; otherwise tag=UNetB{N} to avoid overwriting
    the default artifacts.
"""
import argparse
import os
import sys
import time
# Resolve this directory (HERE) and the project root (ROOT) for location-independent paths.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')  # Save figures in headless environments.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
# Import the network, preprocessing, losses, and utilities from the standalone U-Net module.
from UNet_MT2D_Inversion import (UNetMT2DInversion, _sobel_kernels, build_param_groups,
                                 count_parameters, edge_loss, edge_weighted_l1,
                                 load_and_preprocess_data)
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
# Fixed training protocol (identical to the Mamba/Transformer training scripts)
# ==================================================
PROTOCOL = dict(
    # ---- Data ----
    ntrain=10000, nval=1000, ntest=1000,
    target_nz=32, target_ny=64,
    # ---- Common network dimensions ----
    n_freq=31, n_obs=25, n_comp=4,
    # ---- Training ----
    batch_size=20, learning_rate=1e-4, weight_decay=1e-4,
    epochs=200, warmup_epochs=5, patience=15, min_delta=5e-4,
    use_amp=True, loss_type='l1_edge',
    edge_weight_alpha=3.0, edge_tau_quantile=0.9,
    # ---- UNet ----
    base_channels=32)
# ==================================================
# Loss function wrapper
# ==================================================
def compute_loss(outputs, targets, sobel_x, sobel_y, loss_type, edge_alpha=3.0, edge_tau=0.9):
    """
    Compute the loss according to loss_type:
    'l1_edge'          = L1 + 0.2*Sobel (default)
    'mse'              = MSE only
    'weighted_l1_edge' = Gradient-weighted L1 + 0.2*Sobel (emphasizes anomaly boundaries)
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
    # Evaluate mean loss using the same loss function and AMP settings as training.
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
# Loss curve plotting
# ==================================================
def plot_loss_curves(train_losses, val_losses, learning_rates, best_epoch, save_path, tag='', test_losses=None):
    # Dual y-axes: loss (log scale) on the left; learning rate on the right.
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
# Main training pipeline
# ==================================================
def run_training(base_channels: int = 32, dry_run: bool = False):
    """
    Args:
        base_channels: Number of channels in the first U-Net stage (doubled per stage; 32 -> [32,64,128,256]),
                       with 26 channels: 1,274,105 parameters, approximately matching standalone Mamba expand=2 (1,277,827)
        dry_run: Build the model and print its configuration, parameter count, and output filenames without loading data or training
    """
    tag = 'UNet'
    P = PROTOCOL
    # ==================== Data configuration ====================
    train_path = os.path.join(ROOT, "Training_dataset", "Train_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_10000.mat")
    val_path = os.path.join(ROOT, "Training_dataset", "Validate_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_1000.mat")
    test_path = os.path.join(ROOT, "Training_dataset", "Test_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_1000.mat")
    # For non-default base_channels, append a tag for capacity studies to avoid overwriting default checkpoints/logs.
    if base_channels != P['base_channels']:
        tag = tag + f'B{base_channels}'
    # Save the model configuration in the checkpoint for reconstruction during prediction.
    model_config = {
        'n_freq': P['n_freq'], 'n_obs': P['n_obs'], 'n_comp': P['n_comp'],
        'target_nz': P['target_nz'], 'target_ny': P['target_ny'],
        'base_channels': base_channels}
    if dry_run:
        model = UNetMT2DInversion(**model_config)
        print(f"[dry-run] Standard U-Net baseline ({tag}): standard U-Net encoder-decoder with three downsampling stages")
        print(f"[dry-run] base_channels={base_channels} (channels {base_channels}->{2 * base_channels}->{4 * base_channels}->{8 * base_channels})")
        print(f"[dry-run] Total model parameters: {count_parameters(model):,} | "
              f"(reference: standalone Mamba expand=2: 1,277,827 / expand=4: 1,743,747)")
        print(f"[dry-run] Outputs: ../Model/{tag}_MT2D_Best_nz{P['target_nz']}_ny{P['target_ny']}.pth | "
              f"../Log/{tag}_MT2D_Loss_log.csv + {tag}_MT2D_LossCurves_nz{P['target_nz']}_ny{P['target_ny']}.png")
        return
    # ==================== Unpack training protocol ====================
    ntrain, nval, ntest = P['ntrain'], P['nval'], P['ntest']
    target_nz, target_ny = P['target_nz'], P['target_ny']
    batch_size, learning_rate, weight_decay = P['batch_size'], P['learning_rate'], P['weight_decay']
    epochs, warmup_epochs = P['epochs'], P['warmup_epochs']
    patience, min_delta = P['patience'], P['min_delta']
    use_amp, loss_type = P['use_amp'], P['loss_type']
    edge_weight_alpha, edge_tau_quantile = P['edge_weight_alpha'], P['edge_tau_quantile']
    # ==================== Print configuration ====================
    print("=" * 60)
    print("2D MT inversion training: standard U-Net baseline")
    print(f"Network configuration (standard U-Net, base_channels={base_channels}, three downsampling stages, no positional encoding/FPN)")
    print("Controlled settings: data/sample counts/batch/optimizer/scheduler/early stopping/AMP/loss/random seed are identical to standalone Mamba")
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
        print("Please verify the data paths.")
        return
    # Create DataLoaders (num_workers=0 for Windows compatibility).
    data_iter_train = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    data_iter_val = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    data_iter_test = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # ==================== Model initialization ====================
    print(f"\nCreating {tag} MT inversion model...")
    model = UNetMT2DInversion(**model_config).to(device)
    print(f"Total model parameters: {count_parameters(model):,}")
    # ==================== Output directories ====================
    MODEL_DIR = os.path.join(ROOT, "Model")   # Best model (.pth)
    LOG_DIR = os.path.join(ROOT, "Log")       # Loss log (CSV) and loss curves (PNG)
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
                    print(f"Previous log format differs; backup saved to: {backup}")
                except PermissionError:
                    log_filename = log_filename.replace('.csv', time.strftime('_%Y%m%d_%H%M%S.csv'))
                    print(f"Previous log is in use and cannot be backed up; using a new log file: {log_filename}")
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
        # Linearly warm up the learning rate over the first warmup_epochs epochs.
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
        # ========== Validation and test evaluation (no gradient updates) ==========
        avg_val_loss = evaluate(model, data_iter_val, sobel_x, sobel_y, use_amp, loss_type, edge_weight_alpha, edge_tau_quantile)
        val_losses.append(avg_val_loss)
        avg_test_loss = evaluate(model, data_iter_test, sobel_x, sobel_y, use_amp, loss_type, edge_weight_alpha, edge_tau_quantile)
        test_losses.append(avg_test_loss)
        # Apply cosine annealing after warmup.
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
    # ==================== Final test evaluation using the best checkpoint ====================
    total_time = time.time() - total_start_time
    plot_loss_curves(train_losses, val_losses, learning_rates, best_epoch, loss_fig_path, tag, test_losses)
    print(f"Loss curve saved: {loss_fig_path}")
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = evaluate(model, data_iter_test, sobel_x, sobel_y, use_amp, loss_type, edge_weight_alpha, edge_tau_quantile)
    print("=" * 80)
    print("Training completed!")
    print(f"Total training time: {total_time:.2f}s ({total_time / 60:.1f}min), epochs completed: {len(train_losses)} epochs, best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss (best model): {test_loss:.6f}")
    print("=" * 80)
    print(f"Best model: {model_save_path}")
    print(f"Training log: {log_filename}")
    print(f"Loss curves: {loss_fig_path}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standard U-Net training for 2D MT inversion')
    parser.add_argument('--base-channels', type=int, default=32,
                        help='First-stage U-Net channels, default 32 (1,928,705 parameters); ''26 gives 1,274,105 vs. 1,277,827 for standalone Mamba expand=2 (capacity matching)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build the model and print configuration, parameter count, and output filenames without loading data or training')
    args = parser.parse_args()
    run_training(base_channels=args.base_channels, dry_run=args.dry_run)
