"""
Three-network test-set SSIM distribution comparison (UNet / Mamba / Transformer)
================================================================
Features:
- Compute SSIM for every test sample (consistent with Draw_Inversion_Results.m: log10 resistivity domain,
    default MATLAB IPT ssim() settings: 11x11 Gaussian kernel with sigma=1.5 / replicate padding / double input,
    DynamicRange=1 / K=[0.01,0.03] / mean of the full SSIM map)
- Plot a three-panel distribution figure: (a) violin + box plot, (b) overlaid histograms, (c) empirical CDF (ECDF)
- Plot a raincloud chart (half-violin cloud + box plot + per-sample jittered points, SSIM only)
- Identify samples where Mamba clearly outperforms both Transformer and UNet: require its SSIM to exceed both by
    at least MARGIN_THRESHOLD; export sample indices, per-sample margins, and margin distributions,
    together with win rates, paired Wilcoxon tests, and threshold sensitivity
- Saved outputs (../Result/):
    SSIM_Distribution_3Networks.png          Three-panel distribution chart (300 dpi)
    SSIM_RainCloud_3Networks.png             SSIM raincloud chart (300 dpi)
    SSIM_Comparison_3Networks.mat            Per-sample SSIM arrays
    SSIM_Comparison_3Networks_Summary.csv    Summary SSIM statistics for each network
    SSIM_MambaSuperior_Samples.csv           Samples where Mamba is clearly superior (descending margin)
    SSIM_MambaSuperior.mat                   Per-sample margins and superiority indicators
    SSIM_MambaSuperior_Margins.png           Margin histogram + Mamba versus strongest rival scatter plot
Networks: reuse load_model_and_normalizers from each network's prediction script (independent modules);
          skip networks with missing checkpoints (not yet trained) and report the omission.
Note: absolute SSIM values are not directly comparable with the older dynamic-data-range implementation
      (the fixed L=1 setting shifts values downward); comparisons across these networks use identical settings.
Validation (2025-XX, MATLAB R2024b): for real Mamba/Transformer predictions,
      matlab_ssim agrees with MATLAB IPT ssim() and a manual fspecial+imfilter reference,
      with numerical differences below 4e-14 (floating-point noise level), matching
      ssim(log10(rho_inv), log10(rho_true)) in Draw_Inversion_Results.m to numerical precision.
"""
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
from matplotlib.patches import Rectangle
import numpy as np
import torch
from scipy.io import savemat
from scipy.ndimage import convolve
# Loaders for the three networks come from their prediction scripts (reconstruct each architecture from checkpoint model_config)
import Mamba_MT2D_Predicted as MP
import Trans_MT2D_Predicted as TP
import UNet_MT2D_Predicted as UP
from Mamba_MT2D_Inversion import MT_COMPONENT_ORDER, preprocess_mt_inputs, preprocess_resistivity_targets
from Utilities import loadmat_fields

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Configuration ====================
# Network order: UNet (reference) -> Mamba -> Transformer
# Model-capacity note: default Mamba e2 (1.28M), aligned Transformer e4 (1.74M), and UNet 32 (1.93M) have different parameter counts;
#   For capacity-matched experiments, use after training: Mamba e4 -> 'MambaExpand4_MT2D_Best_nz32_ny64.pth',
#                                              Transformer e2 -> 'TransAlignE2_MT2D_Best_nz32_ny64.pth'
NETS = ['UNet', 'Mamba', 'Trans']
MODEL_FILES = {'UNet': 'UNet_MT2D_Best_nz32_ny64.pth',
               'Mamba': 'Mamba_MT2D_Best_nz32_ny64.pth',
               'Trans': 'Trans_MT2D_Best_nz32_ny64.pth'}
LABELS = {'UNet': 'UNet',
          'Mamba': 'Mamba',
          'Trans': 'Transformer'}
COLORS = {'UNet': '#ff7f0e',          # Orange
          'Mamba': '#1f77b4',         # Blue
          'Trans': '#2ca02c'}         # Green
LOADERS = {'UNet': UP.load_model_and_normalizers,
           'Mamba': MP.load_model_and_normalizers,
           'Trans': TP.load_model_and_normalizers}
BATCH = 20
test_path = os.path.join(ROOT, "Training_dataset", "Test_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_1000.mat")
result_dir = os.path.join(ROOT, "Result")
os.makedirs(result_dir, exist_ok=True)
# ---- Criteria for "Mamba clearly superior" (Mamba must outperform both Transformer and UNet) ----
MARGIN_THRESHOLD = 0.02   # Minimum SSIM margin by which Mamba must exceed both Transformer and UNet
TOP_PRINT =500          # Number of top-margin samples to print to the console

# ==================== SSIM computation (consistent with Draw_Inversion_Results.m) ====================
# MATLAB equivalent: logrho_inv = log10(rho_k_inv);
#                    ssim_val = ssim(logrho_inv, logrho_true)  % Default IPT settings:
#   - Gaussian weighting kernel fspecial('gaussian', 11, 1.5): radius ceil(3*1.5)=5 -> normalized 11x11 kernel
#   - Local statistics via imfilter(..., 'conv', 'replicate'): replicated boundary, original output size
#   - Default DynamicRange = diff(getrangefromclass(double)) = 1
#   - C1 = (0.01*L)^2, C2 = (0.03*L)^2
#   - Scalar SSIM = mean(ssim_map(:)): average local SSIM over the full image (no border cropping)
_SSIM_SIGMA = 1.5                                    # Default MATLAB ssim 'Radius' (kernel sigma)
_SSIM_R = int(np.ceil(3 * _SSIM_SIGMA))              # 5 -> kernel size 2*5+1 = 11
_ax = np.arange(-_SSIM_R, _SSIM_R + 1, dtype=np.float64)
_g = np.exp(-(_ax ** 2) / (2.0 * _SSIM_SIGMA ** 2))
_SSIM_KERNEL = np.outer(_g, _g)
_SSIM_KERNEL /= _SSIM_KERNEL.sum()                   # == fspecial('gaussian', 11, 1.5)


def matlab_ssim(img1, img2, dynamic_range=1.0, K1=0.01, K2=0.03):
    """Compute SSIM consistent with ssim(log10(ρ_inv), log10(ρ_true)) in Draw_Inversion_Results.m.

    Match default MATLAB IPT ssim() settings for double input:
    11x11 Gaussian kernel, sigma=1.5 | replicate padding | DynamicRange=1 | K=[0.01,0.03]
    | Return the mean of the full-image SSIM map
    """
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

# ==================== Load test data (physical domain) ====================
print(f"Loading test data: {test_path}")
raw = loadmat_fields(test_path, list(MT_COMPONENT_ORDER) + ['sig_k'])
n_samples = raw['Rhoxy'].shape[0]
x_raw = preprocess_mt_inputs(raw, n_samples)                      # (N, 31, 25, 4), log10 domain
y_true = preprocess_resistivity_targets(raw['sig_k'], 32, 64)     # (N, 32, 64), log10 resistivity
print(f"Number of test samples: {n_samples}")

# ==================== Inference for each network + per-sample SSIM / relative error ====================
ssim_all = {}
time_all = {}
for name in NETS:
    pth = os.path.join(ROOT, "Model", MODEL_FILES[name])
    if not os.path.exists(pth):
        print(f"  [Skipped] {name}: checkpoint not found at {pth} (network not trained yet?)")
        continue
    model, x_normalizer, y_normalizer, cfg = LOADERS[name](pth)
    x_enc = x_normalizer.encode(x_raw.to(device))
    preds = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n_samples, BATCH):
            out = model(x_enc[i:i + BATCH])
            preds.append(y_normalizer.decode(out).cpu())
    y_pred = torch.cat(preds).numpy()
    t_fwd = time.perf_counter() - t0
# SSIM uses the default MATLAB ssim() settings in Draw_Inversion_Results.m (L=1)
    ssims = np.empty(n_samples)
    for j in range(n_samples):
        ssims[j] = matlab_ssim(y_true[j].numpy(), y_pred[j])
    ssim_all[name] = ssims
    time_all[name] = t_fwd
    print(f"  {name:9s}: SSIM mean={ssims.mean():.6f}  std={ssims.std():.6f}  "
          f"median={np.median(ssims):.6f}  min={ssims.min():.6f}  (inference {t_fwd:.1f}s)")
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

evaluated = [k for k in NETS if k in ssim_all]
if not evaluated:
    print("No networks were evaluated (all checkpoints missing). Exiting.")
    sys.exit(1)

# ==================== Summary table (console + CSV) ====================
print("\n" + "=" * 92)
print("Per-sample SSIM summary for three networks (test set, n=%d)" % n_samples)
print("=" * 92)
header = ("%-10s %9s %8s %9s %9s %9s %9s %10s" %
          ('Network', 'SSIM mean', 'std', 'median', 'P5', 'min', 'max', 'SSIM<0.9'))
print(header)
print("-" * 88)
csv_lines = ["net,ssim_mean,ssim_std,ssim_median,ssim_p5,ssim_min,ssim_max,frac_below_0.9"]
for name in evaluated:
    s = ssim_all[name]
    print("%-10s %9.6f %8.6f %9.6f %9.6f %9.6f %9.6f %6d (%.1f%%)" %
          (name, s.mean(), s.std(), np.median(s), np.percentile(s, 5),
           s.min(), s.max(), int((s < 0.9).sum()), 100.0 * (s < 0.9).mean()))
    csv_lines.append("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f" %
                     (name, s.mean(), s.std(), np.median(s), np.percentile(s, 5),
                      s.min(), s.max(), (s < 0.9).mean()))
print("=" * 92)
csv_path = os.path.join(result_dir, "SSIM_Comparison_3Networks_Summary.csv")
with open(csv_path, 'w') as f:
    f.write("\n".join(csv_lines) + "\n")
print(f"Summary table saved: {csv_path}")

# Save per-sample SSIM arrays (not saved by the individual prediction scripts)
mat_path = os.path.join(result_dir, "SSIM_Comparison_3Networks.mat")
save = {'net_names': evaluated, 'sample_idx': np.arange(n_samples)}
for name in evaluated:
    save[f'ssim_{name}'] = ssim_all[name]
savemat(mat_path, save)
print(f"Per-sample SSIM arrays saved: {mat_path}")

# ==================== Three-panel distribution plot: SSIM + mean relative error ====================
def distribution_row(row_axes, values_by_net, ylabel, fmt, tags,
                     hist_legend_loc='upper left', clip_tail=False):
    """Three-panel distribution: violin+box, histogram overlay, and ECDF; diamond marks the mean (raincloud style)."""
    data = [values_by_net[k] for k in evaluated]
    gmin = min(d.min() for d in data)
    gmax = max(d.max() for d in data)
    tail_cap = np.percentile(np.concatenate(data), 99.5) if clip_tail else gmax
    clipped = clip_tail and gmax > tail_cap

# ---- Violin plot + box plot ----
    ax = row_axes[0]
    vp = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.72)
    for body, k in zip(vp['bodies'], evaluated):
        body.set_facecolor(COLORS[k])
        body.set_alpha(0.45)
        body.set_edgecolor(COLORS[k])
    ax.boxplot(data, widths=0.16, patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.9),
               whiskerprops=dict(color='black', linewidth=0.9),
               capprops=dict(color='black', linewidth=0.9),
               medianprops=dict(color='black', linewidth=1.4),
               flierprops=dict(marker='o', markersize=2.5, markerfacecolor='none',
                               markeredgecolor='gray', alpha=0.6, markeredgewidth=0.6))
    for i, k in enumerate(evaluated):
        m = values_by_net[k].mean()
        ax.scatter([i + 1], [m], marker='D', s=28, color=COLORS[k],
                   zorder=5, edgecolors='black', linewidths=0.6)
        ax.annotate(fmt.format(m), (i + 1, m),
                    textcoords="offset points", xytext=(14, 6),
                    fontsize=8.5, color=COLORS[k], fontweight='bold')
    ax.set_xticks(range(1, len(evaluated) + 1))
    ax.set_xticklabels([LABELS[k] for k in evaluated], rotation=18, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    if clipped:                                        # Clip long-tailed relative error at P99.5 (as in raincloud plot)
        ax.set_ylim(top=tail_cap + 0.03 * (tail_cap - gmin))
        ax.text(0.98, 0.975, 'y-axis clipped at P99.5', transform=ax.transAxes,
                fontsize=7.8, style='italic', color='#999999', va='top', ha='right')
    ax.set_title(f'({tags[0]}) Violin + Box (◆ = mean)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

# ---- Overlaid histograms ----
    ax = row_axes[1]
    bins = np.linspace(gmin, tail_cap if clipped else gmax, 45)
    for k in evaluated:
        ax.hist(values_by_net[k], bins=bins, density=True, alpha=0.45,
                color=COLORS[k], label=LABELS[k], edgecolor=COLORS[k], linewidth=0.7)
    ax.set_xlabel(ylabel)
    ax.set_ylabel('Density')
    ax.set_title(f'({tags[1]}) Histogram' + (' (x clipped at P99.5)' if clipped else ''),
                 fontsize=11)
    ax.legend(fontsize=8.5, loc=hist_legend_loc)
    ax.grid(alpha=0.3)

    # ---- ECDF ----
    ax = row_axes[2]
    for k in evaluated:
        s = np.sort(values_by_net[k])
        ecdf = np.arange(1, s.size + 1) / s.size
        ax.plot(s, ecdf, color=COLORS[k], linewidth=1.6, label=LABELS[k])
        med = np.median(values_by_net[k])
        ax.plot([med], [0.5], marker='o', ms=4.5, color=COLORS[k],
                markeredgecolor='black', markeredgewidth=0.5, zorder=5, linestyle='none')
    ax.set_xlabel(ylabel)
    ax.set_ylabel('Empirical CDF')
    ax.set_title(f'({tags[2]}) ECDF (median: $\\bullet$)', fontsize=11)
    pad = 0.02 * max(gmax - gmin, 1e-9)
    ax.set_xlim(gmin - pad, gmax + pad)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(alpha=0.3)
    return gmin, gmax


fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
gmin_s, _ = distribution_row(axes, {k: ssim_all[k] for k in evaluated},
                             'SSIM', '{:.4f}', 'abc')
axes[0].set_ylim(max(0.0, gmin_s - 0.03), 1.004)        # Fix the SSIM upper limit near 1
fig.suptitle(f'Per-sample SSIM Distribution on Test Set (n = {n_samples}, log10 resistivity domain)',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.95))
png_path = os.path.join(result_dir, "SSIM_Distribution_3Networks.png")
fig.savefig(png_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Distribution plot saved: {png_path}")

# ==================== Raincloud plot: SSIM + mean relative error ====================
def raincloud_panel(ax, values_by_net, ylabel, fmt, higher_better, clip_tail=False):
    """Raincloud panel: half-violin cloud + box plot + per-sample jittered points; diamond marks the mean."""
    data = [values_by_net[k] for k in evaluated]
    gmin = min(d.min() for d in data)
    gmax = max(d.max() for d in data)
    span = max(gmax - gmin, 1e-9)
    rng = np.random.RandomState(7)
    vp = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.8)
    for i, k in enumerate(evaluated):
        c = COLORS[k]
        body = vp['bodies'][i]
        body.set_facecolor(c)
        body.set_alpha(0.40)
        body.set_edgecolor(c)
        clip = Rectangle((i + 0.6, gmin - 0.5 * span), 0.4, 2.0 * span,
                         transform=ax.transData)
        body.set_clip_path(clip)                       # Half-violin cloud: display only the left half
        d = values_by_net[k]
        x = i + 1 + rng.uniform(0.10, 0.42, size=d.size)   # Rain: scatter individual samples on the right
        ax.scatter(x, d, s=2.2, alpha=0.28, color=c, edgecolors='none', zorder=2)
    ax.boxplot(data, widths=0.13, patch_artist=True,
               boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.9),
               whiskerprops=dict(color='black', linewidth=0.9),
               capprops=dict(color='black', linewidth=0.9),
               medianprops=dict(color='black', linewidth=1.3),
               flierprops=dict(marker='o', markersize=2.5, markerfacecolor='none',
                               markeredgecolor='gray', alpha=0.6, markeredgewidth=0.6))
    for i, k in enumerate(evaluated):
        m = values_by_net[k].mean()
        ax.scatter([i + 1], [m], marker='D', s=30, color=COLORS[k], zorder=6,
                   edgecolors='black', linewidths=0.6)
        ax.annotate(fmt.format(m), (i + 1, m), textcoords='offset points',
                    xytext=(16, 5), fontsize=8.5, color=COLORS[k], fontweight='bold')
    ax.set_xticks(range(1, len(evaluated) + 1))
    ax.set_xticklabels([LABELS[k] for k in evaluated], rotation=15, ha='right', fontsize=9.5)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.02, 0.975, '↑ higher is better' if higher_better else '↓ lower is better',
            transform=ax.transAxes, fontsize=8.5, style='italic', color='#666666', va='top')
    if clip_tail:
        cap = np.percentile(np.concatenate(data), 99.5)
        if gmax > cap:
            ax.set_ylim(top=cap)
            ax.text(0.98, 0.975, 'y-axis clipped at P99.5', transform=ax.transAxes,
                    fontsize=7.8, style='italic', color='#999999', va='top', ha='right')
    return gmin

fig, ax = plt.subplots(figsize=(7.2, 5.4))
gmin_s = raincloud_panel(ax, {k: ssim_all[k] for k in evaluated},
                         'SSIM', '{:.4f}', higher_better=True)
ax.set_ylim(max(0.0, gmin_s - 0.03), 1.004)
ax.set_title('SSIM   (cloud = density, rain = per-sample)', fontsize=11)
fig.suptitle(f'Per-sample SSIM Raincloud Comparison (Test set, n = {n_samples})',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.94))
png_path = os.path.join(result_dir, "SSIM_RainCloud_3Networks.png")
fig.savefig(png_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Raincloud plot saved: {png_path}")

# ==================== Samples where Mamba clearly outperforms Transformer and UNet ====================
SUPERIOR_NET = 'Mamba'
RIVAL_NETS = ['Trans', 'UNet']          # Mamba must exceed both rivals to qualify as clearly superior
if SUPERIOR_NET in ssim_all and all(k in ssim_all for k in RIVAL_NETS):
    s_m = ssim_all[SUPERIOR_NET]
    rival = {k: ssim_all[k] for k in RIVAL_NETS}
    margins = {k: s_m - v for k, v in rival.items()}
# min_margin = minimum margin over both rivals: positive only if Mamba exceeds both (weakest-link criterion)
    min_margin = np.minimum(*[margins[k] for k in RIVAL_NETS])
    superior = min_margin >= MARGIN_THRESHOLD
    rank = np.argsort(-min_margin, kind='stable')
    sup_idx = rank[superior[rank]]      # Rank superior samples in descending order of margin
    n_sup = int(superior.sum())

    print("\n" + "=" * 92)
    print(f"Samples where {LABELS[SUPERIOR_NET]} clearly outperforms both rivals (test set, n={n_samples}, "
          f"criterion: SSIM exceeds both rivals by >= {MARGIN_THRESHOLD})")
    print("=" * 92)
    for k in RIVAL_NETS:
        print(f"  vs {LABELS[k]:12s}: Mamba win rate {100.0 * (margins[k] > 0).mean():5.1f}% | "
              f"mean margin {margins[k].mean():+.4f} | median margin {np.median(margins[k]):+.4f}")
    from scipy.stats import wilcoxon
    for k in RIVAL_NETS:
        try:
            stat_w, p_w = wilcoxon(s_m, rival[k])
            print(f"  Paired Wilcoxon signed-rank test vs {LABELS[k]:12s}: W={stat_w:.1f}, p={p_w:.3e}"
                  + ("  (Mamba significantly better overall)" if p_w < 0.05 and margins[k].mean() > 0 else ""))
        except ValueError as e:
            print(f"  Paired Wilcoxon test vs {LABELS[k]} unavailable: {e}")
    print("-" * 92)
    print("  Threshold sensitivity (samples with Mamba SSIM above both rivals by at least thr):")
    for thr in (0.01, 0.02, 0.03, 0.05):
        c = int((min_margin >= thr).sum())
        print(f"    thr={thr:.2f}: {c:4d} samples ({100.0 * c / n_samples:.1f}%)")
    print("-" * 92)
    print(f"  At threshold {MARGIN_THRESHOLD}: {n_sup} samples ({100.0 * n_sup / n_samples:.1f}%) "
          f"where Mamba is clearly superior")
    if n_sup > 0:
        print(f"  Top {min(TOP_PRINT, n_sup)} samples with the largest margins "
              f"(min_margin = the smallest SSIM advantage over the two rivals):")
        print("  %-6s %-8s %-10s %-10s %-10s %-10s %-10s" %
              ('Rank', 'Sample', 'SSIM_Mamba', 'SSIM_Trans', 'SSIM_UNet', 'vs_Trans', 'vs_UNet'))
        for r, idx in enumerate(sup_idx[:TOP_PRINT], start=1):
            print("  %-6d %-8d %-10.4f %-10.4f %-10.4f %+10.4f %+10.4f" %
                  (r, idx, s_m[idx], rival['Trans'][idx], rival['UNet'][idx],
                   margins['Trans'][idx], margins['UNet'][idx]))
        top_ids = ' '.join(str(i) for i in sup_idx[:TOP_PRINT])
        print(f"  Batch plotting: python Mamba_MT2D_Predicted.py --samples {top_ids}")
    else:
        print("  (No qualifying samples; consider lowering MARGIN_THRESHOLD or inspecting threshold sensitivity)")

# ---- CSV: all superior samples in descending margin order ----
    sup_csv = os.path.join(result_dir, "SSIM_MambaSuperior_Samples.csv")
    with open(sup_csv, 'w') as f:
        f.write("rank,sample_idx,ssim_mamba,ssim_trans,ssim_unet,"
                "margin_vs_trans,margin_vs_unet,min_margin\n")
        for r, idx in enumerate(sup_idx, start=1):
            f.write("%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" %
                    (r, idx, s_m[idx], rival['Trans'][idx], rival['UNet'][idx],
                     margins['Trans'][idx], margins['UNet'][idx], min_margin[idx]))
    print(f"  Superior sample list saved: {sup_csv}")

# ---- MAT: per-sample margins and classifications (for subsequent MATLAB analysis) ----
    sup_mat = os.path.join(result_dir, "SSIM_MambaSuperior.mat")
    savemat(sup_mat, {'margin_vs_trans': margins['Trans'],
                      'margin_vs_unet': margins['UNet'],
                      'min_margin': min_margin,
                      'is_superior': superior.astype(np.float64),
                      'ranked_sample_idx': sup_idx.astype(np.float64),
                      'margin_threshold': float(MARGIN_THRESHOLD)})
    print(f"  Per-sample margins saved: {sup_mat}")

# ---- Excel: all 1000 samples sorted by Mamba advantage (min_margin), descending ----
    try:
        import pandas as pd
        rank_all = np.argsort(-min_margin, kind='stable')   # All samples ranked by descending margin
        df_rank = pd.DataFrame({
            'rank':            np.arange(1, n_samples + 1),
            'sample_idx':      rank_all,
            'ssim_mamba':      s_m[rank_all],
            'ssim_trans':      rival['Trans'][rank_all],
            'ssim_unet':       rival['UNet'][rank_all],
            'margin_vs_trans': margins['Trans'][rank_all],
            'margin_vs_unet':  margins['UNet'][rank_all],
            'min_margin':      min_margin[rank_all],
            'is_superior':     superior[rank_all]})
        rows = []
        for k in [SUPERIOR_NET] + RIVAL_NETS:
            s = ssim_all[k]
            rows.append({'network': LABELS[k], 'ssim_mean': s.mean(), 'ssim_std': s.std(),
                         'ssim_median': float(np.median(s)), 'ssim_min': s.min(), 'ssim_max': s.max()})
        df_sum = pd.DataFrame(rows)
        extra = pd.DataFrame([
            {'network': f'Mamba win rate vs {LABELS["Trans"]}', 'ssim_mean': 100.0 * (margins['Trans'] > 0).mean()},
            {'network': f'Mamba win rate vs {LABELS["UNet"]}',  'ssim_mean': 100.0 * (margins['UNet'] > 0).mean()},
            {'network': f'superior count (thr={MARGIN_THRESHOLD})', 'ssim_mean': n_sup},
            {'network': 'n_samples', 'ssim_mean': n_samples}])
        df_sum = pd.concat([df_sum, extra], ignore_index=True)
        xlsx_path = os.path.join(result_dir, "SSIM_Mamba_Advantage_Ranked_1000.xlsx")
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df_rank.to_excel(writer, sheet_name='Ranked_All_1000', index=False)
            df_sum.to_excel(writer, sheet_name='Summary', index=False)
        print(f"  Excel advantage ranking saved: {xlsx_path}")
    except ImportError:
        print("  [Skipped] Excel export: pandas/openpyxl missing (pip install pandas openpyxl)")

# ---- Plots: (a) min_margin histogram, (b) Mamba vs strongest rival scatter plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    lo = min(min_margin.min(), -MARGIN_THRESHOLD)
    hi = max(min_margin.max(), MARGIN_THRESHOLD)
    bins = np.linspace(lo, hi, 60)
    ax.hist(min_margin[~superior], bins=bins, color='#999999', alpha=0.65, label='not superior')
    ax.hist(min_margin[superior], bins=bins, color=COLORS[SUPERIOR_NET], alpha=0.80,
            label=f'superior (n={n_sup}, {100.0 * n_sup / n_samples:.1f}%)')
    ax.axvline(MARGIN_THRESHOLD, color='crimson', linestyle='--', linewidth=1.4)
    ax.axvline(0.0, color='black', linestyle=':', linewidth=0.9, alpha=0.7)
    ax.annotate(f'thr = {MARGIN_THRESHOLD}', xy=(MARGIN_THRESHOLD, 0.95),
                xycoords=('data', 'axes fraction'), fontsize=9, color='crimson',
                va='top', ha='left')
    ax.set_xlabel('min margin: SSIM(Mamba) - max[SSIM(Trans), SSIM(UNet)]')
    ax.set_ylabel('Count')
    ax.set_title(f'(a) Mamba advantage over BOTH rivals (n = {n_samples})', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(alpha=0.3)

    ax = axes[1]
    best_rival = np.maximum(rival['Trans'], rival['UNet'])
    ax.scatter(best_rival[~superior], s_m[~superior], s=9, color='#aaaaaa', alpha=0.55,
               label='not superior')
    ax.scatter(best_rival[superior], s_m[superior], s=13, color=COLORS[SUPERIOR_NET],
               alpha=0.85, edgecolors='none', label=f'superior (n={n_sup})')
    lim_lo = min(best_rival.min(), s_m.min()) - 0.01
    lims = [lim_lo, 1.0]
    ax.plot(lims, lims, 'k--', linewidth=1.0, label='parity (y = x)')
    ax.plot(lims, [v + MARGIN_THRESHOLD for v in lims], color='crimson',
            linestyle=':', linewidth=1.3, label=f'y = x + {MARGIN_THRESHOLD}')
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel('SSIM of best rival  max(Transformer, UNet)')
    ax.set_ylabel('SSIM of Mamba')
    ax.set_title('(b) Mamba vs strongest rival (above red line = superior)', fontsize=11)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(alpha=0.3)
    fig.suptitle(f'Samples where Mamba clearly outperforms both rivals '
                 f'(threshold = {MARGIN_THRESHOLD}, Test set n = {n_samples})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    sup_png = os.path.join(result_dir, "SSIM_MambaSuperior_Margins.png")
    fig.savefig(sup_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Advantage distribution plot saved: {sup_png}")
else:
    missing = [k for k in [SUPERIOR_NET] + RIVAL_NETS if k not in ssim_all]
    print(f"\n[Skipped] Mamba superiority analysis: missing networks {missing} (all three checkpoints required)")

