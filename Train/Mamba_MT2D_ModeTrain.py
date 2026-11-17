"""
Single-Mamba network training script for 2D magnetotelluric (MT) inversion — TE/TM/TEM input mode comparison experiments.
=================================================================
"""
import argparse
import os
import sys
import time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from Mamba_MT2D_Inversion import (MambaMT2DInversion, MT_COMPONENT_ORDER, preprocess_mt_inputs,
                                  preprocess_resistivity_targets, _sobel_kernels,
                                  build_param_groups, count_parameters, edge_loss, edge_weighted_l1)
from Utilities import GaussianNormalizer, GaussianNormalizer_out, loadmat_fields
# ==================================================
# Random seed and acceleration settings (identical to the original version)
# ==================================================
torch.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================================================
# Inversion mode definitions (channel indices for selecting components from [Rhoxy, Phsxy, Rhoyx, Phsyx])
# ==================================================
MODE_CHANNEL_IDX = {'TE': [0, 1], 'TM': [2, 3], 'TEM': [0, 1, 2, 3]}
MODE_DESC = {
    'TE':  'TE mode: TE channels only [Rhoxy, Phsxy] (2 channels)',
    'TM':  'TM mode: TM channels only [Rhoyx, Phsyx] (2 channels)',
    'TEM': 'TEM combined mode: All four TE+TM channels [Rhoxy, Phsxy, Rhoyx, Phsyx]',
}
# ==================================================
# Fixed training protocol (identical to Mamba_MT2D_Train.py; the only variable is the inversion mode/input channels)
# ==================================================
PROTOCOL = dict(
    # ---- Data configuration ----
    ntrain=10000, nval=1000, ntest=1000,
    target_nz=32, target_ny=64,
    # ---- Shared network dimensions (n_comp is determined by the inversion mode in run_training) ----
    n_freq=31, n_obs=25,
    d_model=128, n_layers=2,
    freq_range=(0.01, 1000.0), freq_descending=True,
    fpn_base_channels=64,
    mamba_d_state=16, mamba_bidirectional=False,   # Unidirectional scanning.
    # ---- Training configuration ----
    batch_size=20, learning_rate=1e-4, weight_decay=1e-4,
    epochs=200, warmup_epochs=5, patience=15, min_delta=5e-4,
    use_amp=True, loss_type='l1_edge',
    edge_weight_alpha=3.0, edge_tau_quantile=0.9)
# ==================================================
# Mode-specific data loading: Preprocess four channels -> Select mode-specific channels -> Estimate normalization statistics
# ==================================================
def load_and_preprocess_data_mode(train_path, val_path, test_path,
                                  ntrain, nval, ntest, target_nz, target_ny, mode):
    ch_idx = MODE_CHANNEL_IDX[mode]
    fields = list(MT_COMPONENT_ORDER) + ['sig_k']
    print(f"Loading training data: {train_path}")
    train_raw = loadmat_fields(train_path, fields)
    x_train = preprocess_mt_inputs(train_raw, ntrain)[..., ch_idx]   # (n, n_freq, n_obs, n_comp)
    y_train = preprocess_resistivity_targets(train_raw['sig_k'][:ntrain], target_nz, target_ny)
    # Validation dataset: Load from an independent file. Leave the validation tensors empty if val_path is None or nval=0.
    if val_path and nval > 0:
        print(f"Loading validation data: {val_path}")
        val_raw = loadmat_fields(val_path, fields)
        x_val = preprocess_mt_inputs(val_raw, nval)[..., ch_idx]
        y_val = preprocess_resistivity_targets(val_raw['sig_k'][:nval], target_nz, target_ny)
    else:
        x_val, y_val = x_train[:0], y_train[:0]
    print(f"Loading test data: {test_path}")
    test_raw = loadmat_fields(test_path, fields)
    x_test = preprocess_mt_inputs(test_raw, ntest)[..., ch_idx]
    y_test = preprocess_resistivity_targets(test_raw['sig_k'][:ntest], target_nz, target_ny)
    # Gaussian normalization: Calculate normalization statistics exclusively from the training dataset after channel selection.
    x_normalizer = GaussianNormalizer_out(x_train, flag='Point')
    y_normalizer = GaussianNormalizer(y_train, flag='Point')
    x_train = x_normalizer.encode(x_train)
    x_val = x_normalizer.encode(x_val) if nval > 0 else x_val
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)
    y_val = y_normalizer.encode(y_val) if nval > 0 else y_val
    y_test = y_normalizer.encode(y_test)
    print(f"x_train: {tuple(x_train.shape)}, x_val: {tuple(x_val.shape)}, x_test: {tuple(x_test.shape)}")
    print(f"y_train: {tuple(y_train.shape)}, y_val: {tuple(y_val.shape)}, y_test: {tuple(y_test.shape)}")
    return x_train, y_train, x_val, y_val, x_test, y_test, x_normalizer, y_normalizer
# ==================================================
# Loss function wrapper (identical to the original version)
# ==================================================
def compute_loss(outputs, targets, sobel_x, sobel_y, loss_type, edge_alpha=3.0, edge_tau=0.9):
    """
    Compute the loss according to loss_type:
    'l1_edge'          = L1 + 0.2*Sobel (default)
    'mse'              = Pure mean squared error (MSE)
    'weighted_l1_edge' = Gradient-weighted L1 + 0.2*Sobel, emphasizing anomaly boundaries.
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
    # Compute the mean loss on the specified dataset using the same loss function and AMP configuration as training to ensure consistent performance monitoring.
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
# Loss curve plotting (identical to the original version)
# ==================================================
def plot_loss_curves(train_losses, val_losses, learning_rates, best_epoch, save_path, tag='', test_losses=None):
    # Dual y-axes: The left axis represents loss on a logarithmic scale, while the right axis represents the learning rate.
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
# Output artifact naming (mode-specific suffixes to distinguish results from the original and noise experiments)
# ==================================================
def build_artifact_names(target_nz, target_ny, mode, expand):
    """Return the model checkpoint filename, CSV log filename, and PNG loss curve filename, and PNG loss curve filename.
    When expand=2 (default), append the mode suffix (TE/TM/TEM). When expand=4, append Expand4 to the mode suffix."""
    suffix = mode if expand == 2 else f'{mode}Expand{expand}'
    model_name = f'Mamba_MT2D_Best_nz{target_nz}_ny{target_ny}_{suffix}.pth'
    log_name = f'Mamba_MT2D_Loss_log_{suffix}.csv'
    fig_name = f'Mamba_MT2D_LossCurves_nz{target_nz}_ny{target_ny}_{suffix}.png'
    return model_name, log_name, fig_name
# ==================================================
# Main training workflow
# ==================================================
def run_training(mode: str = 'TEM', expand: int = 2, dry_run: bool = False):
    """
    Args:
        mode: Inversion mode ('TE', 'TM', or 'TEM'), determining which observation channels are selected.
        expand:  Mamba expansion factor (2 or 4). When expand is not 2, append an Expand suffix to the output filenames to avoid overwriting the default artifacts.
        Construct the model and print its configuration, parameter count, and output filenames without loading data or performing training.
    """
    if mode not in MODE_CHANNEL_IDX:
        raise ValueError(f"Unknown mode: {mode} (available modes {'/'.join(MODE_CHANNEL_IDX)})")
    plot_tag = f'Mamba {mode}'
    P = PROTOCOL
    n_comp = len(MODE_CHANNEL_IDX[mode])
    # ==================== Data configuration (noise-free dataset) ====================
    train_path = os.path.join(ROOT, "Training_dataset", "Train_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_10000.mat")
    val_path = os.path.join(ROOT, "Training_dataset", "Validate_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_1000.mat")
    test_path = os.path.join(ROOT, "Training_dataset", "Test_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_1000.mat")
    mamba_expand = expand
    model_name, log_name, fig_name = build_artifact_names(P['target_nz'], P['target_ny'], mode, mamba_expand)
    # Model configuration dictionary, saved with the checkpoint and used to reconstruct the network during prediction. n_comp is set to 2 or 4 according to the inversion mode.
    model_config = {
        'n_freq': P['n_freq'], 'n_obs': P['n_obs'], 'n_comp': n_comp,
        'target_nz': P['target_nz'], 'target_ny': P['target_ny'],
        'd_model': P['d_model'], 'n_mamba_layers': P['n_layers'],
        'freq_range': P['freq_range'], 'freq_descending': P['freq_descending'],
        'fpn_base_channels': P['fpn_base_channels'],
        'mamba_d_state': P['mamba_d_state'], 'mamba_expand': mamba_expand,
        'mamba_bidirectional': P['mamba_bidirectional'],
        'use_cnn_branch': False,
        'encoder_type': 'mamba',
        'attention_mlp_ratio': 4.0}
    if dry_run:
        model = MambaMT2DInversion(**model_config)
        print(f"[dry-run] Mamba mode comparison ({plot_tag}): {MODE_DESC[mode]}")
        print(f"[dry-run] Number of input channels n_comp={n_comp}, channel indices={MODE_CHANNEL_IDX[mode]}, expand={mamba_expand}")
        print(f"[dry-run] dry-run] Context encoder parameters: {count_parameters(model.context_encoder):,} | "
              f"Total parameters: {count_parameters(model):,}")
        print(f"[dry-run] Data: {train_path}")
        print(f"[dry-run] Data: {val_path}")
        print(f"[dry-run] Data: {test_path}")
        print(f"[dry-run] Outputs: ../Model/{model_name} | "
              f"../Log/{log_name} + {fig_name}")
        return
    # ==================== Unpack training protocol parameters ====================
    ntrain, nval, ntest = P['ntrain'], P['nval'], P['ntest']
    target_nz, target_ny = P['target_nz'], P['target_ny']
    batch_size, learning_rate, weight_decay = P['batch_size'], P['learning_rate'], P['weight_decay']
    epochs, warmup_epochs = P['epochs'], P['warmup_epochs']
    patience, min_delta = P['patience'], P['min_delta']
    use_amp, loss_type = P['use_amp'], P['loss_type']
    edge_weight_alpha, edge_tau_quantile = P['edge_weight_alpha'], P['edge_tau_quantile']
    # ==================== Print training configuration ====================
    print("=" * 60)
    print(f"2D MT inversion training: Single Mamba —— {MODE_DESC[mode]}")
    print(f"Network configuration (encoder=mamba, CNN branch=False, expand={mamba_expand}, n_comp={n_comp})")
    print("Controlled variables: Noise-free dataset, sample counts, batch size, optimizer, scheduler, early stopping, AMP, loss function, and random seed are identical to the original configuration.")
    print("=" * 60)
    print(f"Training samples: {ntrain}, Validation samples: {nval}, Test samples: {ntest} (noise-free observations + sig_k ground-truth labels)")
    print(f"Target resolution: {target_nz} x {target_ny}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Epochs: {epochs}")
    print(f"Early stopping patience: {patience}, AMP: {use_amp}, Loss: {loss_type}")
    print("=" * 60)
    # ==================== Data loading and preprocessing (mode-specific channel selection) ====================
    try:
        (x_train, y_train, x_val, y_val, x_test, y_test,
         x_normalizer, y_normalizer) = load_and_preprocess_data_mode(
            train_path, val_path, test_path, ntrain, nval, ntest, target_nz, target_ny, mode)
    except Exception as e:
        print(f"Data loading failed: {e}")
        print("Check whether the dataset paths are correct and point to the noise-free datasets.")
        return
    # Create DataLoaders (num_workers=0 for Windows compatibility).
    data_iter_train = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    data_iter_val = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    data_iter_test = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # ==================== Model initialization ====================
    print(f"\nCreating {plot_tag} MT inversion model...")
    model = MambaMT2DInversion(**model_config).to(device)
    print(f"Context encoder parameters: {count_parameters(model.context_encoder):,} | Total model parameters: {count_parameters(model):,}")
    # ==================== Output directories ====================
    MODEL_DIR = os.path.join(ROOT, "Model")   # Best model checkpoint (.pth).
    LOG_DIR = os.path.join(ROOT, "Log")       # Loss log (CSV) and loss curve figure (PNG).
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    loss_fig_path = os.path.join(LOG_DIR, fig_name)
    # ==================== Optimizer and learning rate scheduling ====================
    optimizer = torch.optim.AdamW(build_param_groups(model, weight_decay), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    x_normalizer.to(device)
    y_normalizer.to(device)
    sobel_x, sobel_y = _sobel_kernels(device)
    # ==================== Logging and best model checkpoint saving ====================
    log_filename = os.path.join(LOG_DIR, log_name)
    log_header = "epoch,train_loss,val_loss,test_loss,learning_rate,time_sec\n"
    if os.path.isfile(log_filename):
        with open(log_filename) as f:
            if f.readline() != log_header:
                backup = log_filename.replace('.csv', '_old.csv')
                try:
                    os.replace(log_filename, backup)
                    print(f"The old log format differs from the current format. Backup saved to: {backup}")
                except PermissionError:
                    log_filename = log_filename.replace('.csv', time.strftime('_%Y%m%d_%H%M%S.csv'))
                    print(f"The old log file is in use and cannot be backed up. Using a new log file: {log_filename}")
    if not os.path.isfile(log_filename):
        with open(log_filename, 'w') as f:
            f.write(log_header)
    model_save_path = os.path.join(MODEL_DIR, model_name)

    def save_best_model(path, epoch):
        torch.save({'epoch': epoch,
                    'mode': mode,                          # Store the inversion mode and selected channel indices for consistent channel selection during prediction.
                    'mode_channels': MODE_CHANNEL_IDX[mode],
                    'model_state_dict': model.state_dict(),
                    'val_loss': avg_val_loss,
                    'x_normalizer_mean': x_normalizer.mean,
                    'x_normalizer_std': x_normalizer.std,
                    'y_normalizer_mean': y_normalizer.mean,
                    'y_normalizer_std': y_normalizer.std,
                    'model_config': model_config,}, path)

    # ==================== Training loop ====================
    print(f"\nStarting training ({MODE_DESC[mode]})...")
    print("=" * 80)
    train_losses, val_losses, test_losses, learning_rates = [], [], [], []
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        # Learning rate warmup: Linearly increase the learning rate during the first warmup_epochs epochs.
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
        # ========== Validation and test evaluation (excluded from parameter updates) ==========
        avg_val_loss = evaluate(model, data_iter_val, sobel_x, sobel_y, use_amp, loss_type, edge_weight_alpha, edge_tau_quantile)
        val_losses.append(avg_val_loss)
        avg_test_loss = evaluate(model, data_iter_test, sobel_x, sobel_y, use_amp, loss_type, edge_weight_alpha, edge_tau_quantile)
        test_losses.append(avg_test_loss)
        # Learning rate scheduling: Apply cosine annealing after the warmup phase.
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
            plot_loss_curves(train_losses, val_losses, learning_rates, best_epoch, loss_fig_path, plot_tag, test_losses)
        # ========== Early stopping and best model checkpoint saving ==========
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            early_stop_counter = 0
            save_best_model(model_save_path, epoch + 1)
            print(f"Validation loss improved. Best model saved to: {model_save_path}")
        else:
            early_stop_counter += 1
            print(f"Validation loss did not improve ({early_stop_counter}/{patience})")
            if early_stop_counter >= patience:
                print(f"No sufficient improvement for {patience} consecutive epochs. Early stopping triggered.")
                break
    # ==================== Training completed: Evaluate the test dataset using the best model ====================
    total_time = time.time() - total_start_time
    plot_loss_curves(train_losses, val_losses, learning_rates, best_epoch, loss_fig_path, plot_tag, test_losses)
    print(f"Loss curve figure saved to: {loss_fig_path}")
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = evaluate(model, data_iter_test, sobel_x, sobel_y, use_amp, loss_type, edge_weight_alpha, edge_tau_quantile)
    print("=" * 80)
    print("Training completed!")
    print(f"Total training time: {total_time:.2f}s ({total_time / 60:.1f}minutes), Actual training epochs {len(train_losses)}, Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss (best model): {test_loss:.6f}")
    print("=" * 80)
    print(f"Best model: {model_save_path}")
    print(f"Training log: {log_filename}")
    print(f"Loss curve: {loss_fig_path}")
    print("=" * 80)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single-Mamba 2D MT inversion training —— TE/TM/TEM mode comparison (noise-free data)')
    parser.add_argument('--mode', default='TM', choices=('TE', 'TM', 'TEM'),
                        help='Inversion mode: TE=TE channels only (Rhoxy+Phsxy), TM=TM channels only (Rhoyx+Phsyx), TEM=all four TE+TM channels. Default: TM')
    parser.add_argument('--expand', type=int, default=2, choices=(2, 4),
                        help='Mamba expansion factor, default=2; append Expand4 to output filenames when expand=4')
    parser.add_argument('--dry-run', action='store_true',
                        help='Construct the model and print the configuration, parameter count, and output filenames without loading data or performing training')
    args = parser.parse_args()
    run_training(mode=args.mode, expand=args.expand, dry_run=args.dry_run)
