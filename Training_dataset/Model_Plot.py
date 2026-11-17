import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import scipy.io as scio

# ==================== 1. Global settings ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# ==================== 2. Data loading ====================
model_name = '../Training_dataset/Test_MT2D_64x32_Distance(50km)_Depth(10km)_Fre31_Obs25_1000.mat'
try:
    data = scio.loadmat(model_name)
except FileNotFoundError:
    print(f"Error: File not found: {model_name}")
    raise

# Extract variables
sig = data['sig']  # Conductivity of the entire computational domain
zn = data['zn'].squeeze()  # Vertical nodes of the entire domain (m)
yn = data['yn'].squeeze()  # Horizontal nodes of the entire domain (m)
sig_k = data['sig_k']  # Conductivity of the core region
zn0 = data['zn0'].squeeze()  # Vertical nodes of the core region
yn0 = data['yn0'].squeeze()  # Horizontal nodes of the core region

freqs = data['freq'].squeeze()
obs = data['obs'].squeeze()
Rhoxy = data['Rhoxy']
Phsxy = data['Phsxy']
Rhoyx = data['Rhoyx']
Phsyx = data['Phsyx']

idx = 144   # Select sample index


# ==================== 3. Helper function: Draw actual mesh lines ====================
def draw_mesh_lines(ax, y_nodes, z_nodes, color='Black', linewidth=0.5, alpha=0.5):
    """
    Draw the actual computational mesh lines
    """
    y_km = y_nodes / 1e3
    z_km = z_nodes / 1e3

    # Draw vertical lines
    ax.vlines(x=y_km, ymin=z_km.min(), ymax=z_km.max(),
              colors=color, linewidth=linewidth, alpha=alpha, zorder=5)

    # Draw horizontal lines
    ax.hlines(y=z_km, xmin=y_km.min(), xmax=y_km.max(),
              colors=color, linewidth=linewidth, alpha=alpha, zorder=5)


# ==================== 4. Plotting ====================

# ----------------- Figure 1: Full model (fill the plotting area) -----------------
# Key modification 1: Use layout='constrained' to automatically reduce surrounding whitespace
# Key modification 2: Adjust figsize according to the data aspect ratio (e.g., width/height ~ 1.5)
fig1, ax = plt.subplots(figsize=(12, 8), layout='constrained')

Z, Y = np.meshgrid(zn, yn, indexing='ij')

# 1. Plot the conductivity background
# With shading='flat', the data array must be one cell smaller than the node grid in each dimension; auto or nearest can be used when needed
h = ax.pcolormesh(Y / 1e3, Z / 1e3, sig[idx],
                  norm=colors.LogNorm(vmin=1e-4, vmax=1e0),
                  cmap='jet', shading='flat', alpha=1, zorder=1)

# 2. Draw the actual mesh in the extended domain (light-colored lines can improve contrast on dark backgrounds)
draw_mesh_lines(ax, yn, zn, color='Black', linewidth=0.5, alpha=0.5)

# 3. Draw the black boundary of the core region
y0_min, y0_max = yn0.min() / 1e3, yn0.max() / 1e3
z0_min, z0_max = zn0.min() / 1e3, zn0.max() / 1e3
rect = patches.Rectangle((y0_min, z0_min), y0_max - y0_min, z0_max - z0_min,
                         linewidth=2.0, edgecolor='black', facecolor='none', zorder=10)
ax.add_patch(rect)

# 4. Key modification 3: Fit the axes exactly to the domain boundaries without margins
ax.set_xlim(yn.min() / 1e3, yn.max() / 1e3)  # Align with the left and right boundaries
ax.set_ylim(zn.max() / 1e3, zn.min() / 1e3)  # Align with the top and bottom boundaries (use reversed limits)

# 5. Labels and formatting
cbar1 = fig1.colorbar(h, ax=ax, label='Conductivity (S/m)', pad=0.02)  # Reduce spacing between the colorbar and plot
cbar1.ax.tick_params(labelsize=14)
cbar1.set_label('Conductivity (S/m)', fontsize=16)

ax.set_xlabel('Distance (km)', fontsize=16)
ax.set_ylabel('Depth (km)', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
# ax.set_title(f'Full Model (Sample {idx})', fontsize=18) # Uncomment if a title is needed; it occupies additional space

# ----------------- Figure 2: Core region and MT responses (retain the layout and enable constrained_layout) -----------------
fig2 = plt.figure(figsize=(12, 12), layout='constrained')
ax00 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax10 = plt.subplot2grid((3, 2), (1, 0))
ax11 = plt.subplot2grid((3, 2), (1, 1))
ax20 = plt.subplot2grid((3, 2), (2, 0))
ax21 = plt.subplot2grid((3, 2), (2, 1))

# --- Core region ---
resistivity_core = 1 / sig_k[idx]
resistivity_core = np.log10(resistivity_core)
Z_core, Y_core = np.meshgrid(zn0, yn0, indexing='ij')

h = ax00.pcolormesh(Y_core / 1e3, Z_core / 1e3, resistivity_core,
                    vmin=0, vmax=4, cmap='jet', shading='flat', zorder=1)
ax00.invert_yaxis()
draw_mesh_lines(ax00, yn0, zn0, color='white', linewidth=0.3, alpha=0.3)

cbar00 = fig2.colorbar(h, ax=ax00, pad=0.02)
cbar00.ax.tick_params(labelsize=14)
cbar00.set_label('log₁₀(ρ [Ω·m])', fontsize=16)
ax00.set_xlabel('Distance (km)', fontsize=16)
ax00.set_ylabel('Depth (km)', fontsize=16)
ax00.set_title('Core Model', fontsize=18)
ax00.tick_params(axis='both', labelsize=14)
# Fit the axes exactly to the core-region boundaries
ax00.set_xlim(yn0.min() / 1e3, yn0.max() / 1e3)
ax00.set_ylim(zn0.max() / 1e3, zn0.min() / 1e3)


# --- MT response plots ---
def plot_response(ax, x, y, data, vmin, vmax, label, title):
    c = ax.pcolormesh(x / 1e3, np.log10(y), data, vmin=vmin, vmax=vmax, cmap='jet', shading='auto', zorder=1)
    cb = fig2.colorbar(c, ax=ax, pad=0.02)
    cb.ax.tick_params(labelsize=14)
    cb.set_label(label, fontsize=16)
    ax.set_xlabel('Distance (km)', fontsize=16)
    ax.set_ylabel('log₁₀(Frequency [Hz])', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, which='major', linestyle='--', color='white', linewidth=0.5, alpha=0.3, zorder=5)


plot_response(ax10, obs, freqs, np.log10(Rhoxy[idx]), 0, 4, 'log₁₀(ρₓᵧ)', 'TE Apparent Resistivity')
plot_response(ax11, obs, freqs, Phsxy[idx], 0, 90, 'φₓᵧ (°)', 'TE Phase')
plot_response(ax20, obs, freqs, np.log10(Rhoyx[idx]), 0, 4, 'log₁₀(ρᵧₓ)', 'TM Apparent Resistivity')
plot_response(ax21, obs, freqs, Phsyx[idx], 0, 90, 'φᵧₓ (°)', 'TM Phase')

plt.show()
