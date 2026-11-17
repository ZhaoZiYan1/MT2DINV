"""
================================================================================
Layered Model Validation Script - Using MT2D_Model Professional Grid Design
================================================================================
Functions:
    1. Use the same professional grid design as MT2D_Model (fine shallow discretization)
    2. Test simple layered models
    3. Compare with 1D analytical solution to verify numerical accuracy of 2D FDM forward modeling

Grid alignment (2026-08, consistent with MT2D_Model.py training sample generation):
    - Frequency: 21 points, 1000 -> 0.01 Hz logarithmic spacing
    - Z core region: 32 layers, total depth 10 km, first layer 20 m, geometric growth
    - Y core region: 64 cells, +/-25 km uniform (consistent with target_nz=32, target_ny=64)
    - Air layer: 10 layers, upward to 100 km; bottom padding: 10 layers, downward to 200 km
    - Y padding: 10 layers, left and right each to 100 km
    - Observation point: 0.0 = center point of training observation array np.linspace(-25000, 25000, 25)
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import matplotlib
import scipy.io as sio
from matplotlib import font_manager
from scipy.optimize import fsolve
matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Microsoft YaHei', 'SimHei']
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = ['serif', 'sans-serif']
# ==================== Mock Ray (lightweight handling) ====================
# The MT2D_FDM module uses Ray by default; create a mock class to run on single core
class MockRay:
    @staticmethod
    def init(*args, **kwargs): pass
    @staticmethod
    def remote(cls): return cls
    @staticmethod
    def get(results): return results
sys.modules['ray'] = MockRay()   # Inject Ray module into sys.modules
from MT2D_FDM import MT2DFD      # Import MT2D FDM forward modeling module
# 1. Professional grid design parameters (fully aligned with MT2D_Model.py main())
# -------------------- Grid scale parameters (same values as MT2D_Model.py) --------------------
nza = 10        # Number of air layers
size_b = 10     # Number of padding layers (bottom and both sides)
size_ky = 64    # Number of horizontal cells in core region (aligned with target_ny=64)
size_kz = 32    # Number of depth layers in core region (aligned with target_nz=32)
n_freq = 31     # Number of frequencies (aligned with training data Fre21)
# -------------------- Frequency settings --------------------
# Frequency range: 1000 Hz to 0.01 Hz, 21 points, logarithmic spacing (index 0 = highest frequency)
freq = np.logspace(3, -2, n_freq)
# -------------------- Observation point settings--------------------
# 0.0 is the center point (13th) of training observation array obs = np.linspace(-25000, 25000, 25)
ry = np.array([0.0])
# ==================== Z-direction grid construction (code aligned with MT2D_Model.generate_model) ====================
# Air layer grid, increasing exponentially upward
# NOTE: geomspace uses nza (not nza+1) nodes -> exactly nza air cells ending at z=0.
# The old nza+1 version created one extra 50 m cell [-50,0] that carried EARTH conductivity,
# placing the effective surface (impedance extraction node) at z=-50 m and causing the
# high-frequency phase upturn (~±6.5 deg at 1000 Hz for conductive top layers).
z_air = -np.geomspace(50, 100000, nza)[::-1]
# Core region: simulate 10 km subsurface, non-uniform grid: fine near surface, coarse at depth
z_total, z_first = 10000, 20
# Solve for appropriate growth factor r using geometric series sum formula
def equation(r):
    return z_first * (r ** size_kz - 1) / (r - 1) - z_total
r_z = fsolve(equation, 1.1)[0]
zn0 = np.zeros(size_kz + 1)
# Generate core region depth nodes
for i in range(size_kz):
    zn0[i + 1] = zn0[i] + z_first * r_z ** i
# Bottom padding grid
z_bottom = np.geomspace(zn0[-1], 200000, size_b + 1)[1:]
# Concatenate air padding, core region, and bottom padding nodes
zn = np.concatenate([z_air, zn0, z_bottom])
print(f'r_z: {r_z:.6f} (core region {size_kz} layers, first layer {z_first} m, total depth {zn0[-1]:.1f} m)')
print(f'Z grid: air {nza} layers + core {size_kz} layers + bottom padding {size_b} layers = {len(zn) - 1} cells')

# ==================== 2. Generate horizontal (Y) grid (code aligned with MT2D_Model.generate_model) ====================
y_total = 50000  # Core region total width 50 km (+/-25 km)
yn0 = np.linspace(-y_total / 2, y_total / 2, size_ky + 1)
y_left = -np.geomspace(-yn0[0] + 2500, 100000, size_b + 1)[::-1]
y_right = np.geomspace(yn0[-1] + 2500, 100000, size_b + 1)
yn = np.concatenate([y_left, yn0, y_right])
print(f'Y grid: padding {size_b} cells + core {size_ky} cells + padding {size_b} cells = {len(yn) - 1} cells')
# Cell center depth (layer assignment uses cell centers, consistent with MT2D_Model cell-value convention)
zc = (zn[:-1] + zn[1:]) / 2

# ==================== 1D analytical solution function (ground truth) ====================
def mt1d_analytical(freq, thickness, resistivity):
    """
    Compute MT theoretical response using recursive formula for layered media.
    Algorithm:
        1. Start from the bottom layer, compute its impedance Z_n
        2. Recursively upward, compute upper-layer impedance using reflection coefficient formula
        3. Finally obtain surface impedance
    """
    # -------------------- Constants and initialization --------------------
    mu = 4e-7 * np.pi   # Vacuum magnetic permeability (H/m)
    n_freq = len(freq)  # Number of frequencies
    n_layer = len(resistivity)  # Number of layers
    Z = np.zeros(n_freq, dtype=complex)  # Impedance array
    # Compute impedance for each frequency
    for i, f in enumerate(freq):
        omega = 2 * np.pi * f        # Angular frequency (rad/s)
        # Step 1: Compute impedance of the bottom half-space
        k_n = np.sqrt(1j * omega * mu / resistivity[-1])  #Wavenumber
        Z_n = 1j * omega * mu / k_n  # Bottom layer impedance
        # Step 2: Recursively upward
        for j in range(n_layer - 2, -1, -1):  # Start from second-last layer
            # Compute wavenumber and intrinsic impedance of layer j
            k_j = np.sqrt(1j * omega * mu / resistivity[j])
            Z_j = 1j * omega * mu / k_j
            h_j = thickness[j]  # Thickness of layer j
            # Compute reflection coefficient
            r = (Z_n - Z_j) / (Z_n + Z_j)
            # Recursion: compute impedance at top of layer j
            Z_n = Z_j * (1 + r * np.exp(-2 * k_j * h_j)) / (1 - r * np.exp(-2 * k_j * h_j))
        Z[i] = Z_n  # Store surface impedance
    # -------------------- Compute apparent resistivity and phase --------------------
    # Apparent resistivity formula: ρ_a = |Z|² / (ωμ)
    rho_app = np.abs(Z)**2 / (2 * np.pi * freq * mu)
    # Phase computation (using arctan2 for range -180° to +180°)
    phase = np.arctan2(Z.imag, Z.real) * 180 / np.pi
    return rho_app, phase, Z

# ==================== Test 1: Three-layer model (A-type: low-medium-high)====================
# Used to verify basic stability of the forward algorithm
rho1, rho2, rho3 = 100.0, 500.0, 1000.0
h1, h2 = 2000.0, 5000.0
sig_air = 1e-9  # Air conductivity (S/m), corresponding to resistivity ~10^9 Ω·m
# Analytical solution
rho_ana_1, phase_ana_1, _ = mt1d_analytical(freq, np.array([h1, h2]),np.array([rho1, rho2, rho3]))
# Numerical solution
sig_1 = np.ones((len(zn)-1, len(yn)-1)) / rho1
sig_1[0:nza, :] = sig_air  # Set air layer conductivity
idx1 = np.where(zc >= h1)[0][0]
idx2 = np.where(zc >= (h1 + h2))[0][0]
sig_1[idx1:idx2, :] = 1.0 / rho2
sig_1[idx2:, :] = 1.0 / rho3
model_1 = MT2DFD(nza, zn, yn, freq, ry, sig_1, n_add=10)
Rhoxy_1, Phsxy_1, Zxy_1, Rhoyx_1, Phsyx_1, Zyx_1 = model_1.MT2D(mode="TETM")
# ==================== Test 2: Three-layer model (Q-type: high-medium-low) ====================
rho1, rho2, rho3 = 1000.0, 100.0, 10.0
h1, h2 = 2000.0, 5000.0
# Analytical solution
rho_ana_2, phase_ana_2, _ = mt1d_analytical(freq, np.array([h1, h2]),np.array([rho1, rho2, rho3]))
# Numerical solution
sig_2 = np.ones((len(zn)-1, len(yn)-1)) / rho1
sig_2[0:nza, :] = sig_air
idx1 = np.where(zc >= h1)[0][0]
idx2 = np.where(zc >= (h1 + h2))[0][0]
sig_2[idx1:idx2, :] = 1.0 / rho2
sig_2[idx2:, :] = 1.0 / rho3
model_2 = MT2DFD(nza, zn, yn, freq, ry, sig_2, n_add=10)
Rhoxy_2, Phsxy_2, Zxy_2, Rhoyx_2, Phsyx_2, Zyx_2 = model_2.MT2D(mode="TETM")
# ==================== Test 3: Three-layer model (K-type: low-high-low) ====================
rho1, rho2, rho3 = 100.0, 1000.0, 10.0
h1, h2 = 2000.0, 5000.0
# Analytical solution
rho_ana_3, phase_ana_3, _ = mt1d_analytical(freq, np.array([h1, h2]), np.array([rho1, rho2, rho3]))
# Numerical solution
sig_3 = np.ones((len(zn)-1, len(yn)-1)) / rho1
sig_3[0:nza, :] = sig_air
idx1 = np.where(zc >= h1)[0][0]
idx2 = np.where(zc >= (h1 + h2))[0][0]
sig_3[idx1:idx2, :] = 1.0 / rho2
sig_3[idx2:, :] = 1.0 / rho3
t0 = time.time()
model_3 = MT2DFD(nza, zn, yn, freq, ry, sig_3, n_add=10)
Rhoxy_3, Phsxy_3, Zxy_3, Rhoyx_3, Phsyx_3, Zyx_3 = model_3.MT2D(mode="TETM")
# ==================== Test 4: Three-layer model (H-type: medium-low-high) ====================
rho1, rho2, rho3 = 100.0, 10.0, 1000.0
h1, h2 = 2000.0, 5000.0
# Analytical solution
rho_ana_4, phase_ana_4, _ = mt1d_analytical(freq, np.array([h1, h2]), np.array([rho1, rho2, rho3]))
# Numerical solution
sig_4 = np.ones((len(zn)-1, len(yn)-1)) / rho1
sig_4[0:nza, :] = sig_air
idx1 = np.where(zc >= h1)[0][0]
idx2 = np.where(zc >= (h1 + h2))[0][0]
sig_4[idx1:idx2, :] = 1.0 / rho2
sig_4[idx2:, :] = 1.0 / rho3
model_4 = MT2DFD(nza, zn, yn, freq, ry, sig_4, n_add=10)
Rhoxy_4, Phsxy_4, Zxy_4, Rhoyx_4, Phsyx_4, Zyx_4 = model_4.MT2D(mode="TETM")
# ==================== Plotting ====================
fig = plt.figure(figsize=(16, 10), dpi=100)
# --- Prepare data package ---
# Frequency (shared)
freq_data = freq.reshape(-1, 1)
# Pack analytical solutions, TE responses, TM responses for 4 models into matrices
# Analytical (1D)
rho_ana = np.column_stack([rho_ana_1, rho_ana_2, rho_ana_3, rho_ana_4])
phs_ana = np.column_stack([phase_ana_1, phase_ana_2, phase_ana_3, phase_ana_4])
# Numerical TE mode (take data at observation point 0)
rho_te = np.column_stack([Rhoxy_1[:,0], Rhoxy_2[:,0], Rhoxy_3[:,0], Rhoxy_4[:,0]])
phs_te = np.column_stack([Phsxy_1[:,0], Phsxy_2[:,0], Phsxy_3[:,0], Phsxy_4[:,0]])
# Numerical TM mode
rho_tm = np.column_stack([Rhoyx_1[:,0], Rhoyx_2[:,0], Rhoyx_3[:,0], Rhoyx_4[:,0]])
phs_tm = np.column_stack([Phsyx_1[:,0], Phsyx_2[:,0], Phsyx_3[:,0], Phsyx_4[:,0]])
# --- Save to .mat file ---
save_dict = {'freq': freq_data,      # Frequency array (Hz)
             'rho_ana': rho_ana,     # Theoretical resistivities (4 column vectors)
             'phs_ana': phs_ana,     # Theoretical phases
             'rho_te': rho_te,       # Numerical TE resistivities
             'phs_te': phs_te,       # Numerical TE phases
             'rho_tm': rho_tm,       # Numerical TM resistivities
             'phs_tm': phs_tm,       # Numerical TM phases
             'zn': zn, 'yn': yn,     # Grid nodes (aligned with MT2D_Model.py, for traceability)
             'nza': nza }
sio.savemat('MT_Validation_1DForward.mat', save_dict)
print("\n Data successfully saved to MT_Validation_1DForward.mat")
# Test 1: A-type (low-medium-high)
ax1 = plt.subplot(2, 4, 1)
ax1.loglog(freq, rho_ana_1, 'k-', lw=2, label='Analytical')
ax1.loglog(freq, Rhoxy_1[:, 0], 'ro--', ms=4, label='TE')
ax1.loglog(freq, Rhoyx_1[:, 0], 'bs--', ms=4, label='TM')
ax1.set_xlabel('Frequency (Hz)', fontsize=10, fontfamily='serif')
ax1.set_ylabel('Apparent Resistivity (Ω·m)', fontsize=10)
ax1.set_title('Test 1: Three-layer (A-type)\n10/100/1000 Ω·m, 2km/5km', fontsize=12)
ax1.invert_xaxis()
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

ax2 = plt.subplot(2, 4, 5)
ax2.semilogx(freq, phase_ana_1, 'k-', lw=2, label='Analytical')
ax2.semilogx(freq, Phsxy_1[:, 0], 'ro--', ms=4, label='TE')
ax2.semilogx(freq, Phsyx_1[:, 0], 'bs--', ms=4, label='TM')
ax2.set_xlabel('Frequency (Hz)', fontsize=10, fontfamily='serif')
ax2.set_ylabel('Phase (°)', fontsize=10)
ax2.set_title('Phase', fontsize=12)
ax2.invert_xaxis()
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Test 2: Q-type (high-medium-low)
ax3 = plt.subplot(2, 4, 2)
ax3.loglog(freq, rho_ana_2, 'k-', lw=2, label='Analytical')
ax3.loglog(freq, Rhoxy_2[:, 0], 'ro--', ms=4, label='TE')
ax3.loglog(freq, Rhoyx_2[:, 0], 'bs--', ms=4, label='TM')
ax3.axhline(100, color='gray', ls=':', alpha=0.5)
ax3.axhline(10, color='gray', ls=':', alpha=0.5)
ax3.set_xlabel('Frequency (Hz)', fontsize=10, fontfamily='serif')
ax3.set_ylabel('Rho (Ω·m)', fontsize=10)
ax3.set_title('Test 2: Three-layer (Q-type)\n1000/100/10 Ω·m, 2km/5km', fontsize=12)
ax3.invert_xaxis()
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

ax4 = plt.subplot(2, 4, 6)
ax4.semilogx(freq, phase_ana_2, 'k-', lw=2, label='Analytical')
ax4.semilogx(freq, Phsxy_2[:, 0], 'ro--', ms=4, label='TE')
ax4.semilogx(freq, Phsyx_2[:, 0], 'bs--', ms=4, label='TM')
ax4.set_xlabel('Frequency (Hz)', fontsize=10, fontfamily='serif')
ax4.set_ylabel('Phase (°)', fontsize=10)
ax4.set_title('Phase', fontsize=12)
ax4.invert_xaxis()
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# Test 3: K-type (low-high-low)
ax5 = plt.subplot(2, 4, 3)
ax5.loglog(freq, rho_ana_3, 'k-', lw=2, label='Analytical')
ax5.loglog(freq, Rhoxy_3[:, 0], 'ro--', ms=4, label='TE')
ax5.loglog(freq, Rhoyx_3[:, 0], 'bs--', ms=4, label='TM')
ax5.axhline(100, color='gray', ls=':', alpha=0.5)
ax5.axhline(10, color='gray', ls=':', alpha=0.5)
ax5.set_xlabel('Frequency (Hz)', fontsize=10, fontfamily='serif')
ax5.set_ylabel('Rho (Ω·m)', fontsize=10)
ax5.set_title('Test 3: Three-layer (K-type)\n10/1000/100 Ω·m, 2km/5km', fontsize=12)
ax5.invert_xaxis()
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

ax6 = plt.subplot(2, 4, 7)
ax6.semilogx(freq, phase_ana_3, 'k-', lw=2, label='Analytical')
ax6.semilogx(freq, Phsxy_3[:, 0], 'ro--', ms=4, label='TE')
ax6.semilogx(freq, Phsyx_3[:, 0], 'bs--', ms=4, label='TM')
ax6.set_xlabel('Frequency (Hz)', fontsize=10, fontfamily='serif')
ax6.set_ylabel('Phase (°)', fontsize=10)
ax6.set_title('Phase', fontsize=10)
ax6.invert_xaxis()
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=10)

# Test 4: H-type (medium-low-high)
ax7 = plt.subplot(2, 4, 4)
ax7.loglog(freq, rho_ana_4, 'k-', lw=2, label='Analytical')
ax7.loglog(freq, Rhoxy_4[:, 0], 'ro--', ms=4, label='TE')
ax7.loglog(freq, Rhoyx_4[:, 0], 'bs--', ms=4, label='TM')
ax7.axhline(100, color='gray', ls=':', alpha=0.5)
ax7.axhline(10, color='gray', ls=':', alpha=0.5)
ax7.axhline(1000, color='gray', ls=':', alpha=0.5)
ax7.set_xlabel('Frequency (Hz)', fontsize=10, fontfamily='serif')
ax7.set_ylabel('Rho (Ω·m)', fontsize=10)
ax7.set_title('Test 4: Three-layer (H-type)\n100/10/1000 Ω·m, 2km/5km', fontsize=12)
ax7.invert_xaxis()
ax7.grid(True, alpha=0.3)
ax7.legend(fontsize=10)

ax8 = plt.subplot(2, 4, 8)
ax8.semilogx(freq, phase_ana_4, 'k-', lw=2, label='Analytical')
ax8.semilogx(freq, Phsxy_4[:, 0], 'ro--', ms=4, label='TE')
ax8.semilogx(freq, Phsyx_4[:, 0], 'bs--', ms=4, label='TM')
ax8.set_xlabel('Frequency (Hz)', fontsize=10, fontfamily='serif')
ax8.set_ylabel('Phase (°)', fontsize=10)
ax8.set_title('Phase', fontsize=10)
ax8.invert_xaxis()
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=10)
plt.suptitle('Simple Layered Model Validation', fontsize=12, y=0.995)
plt.tight_layout()
plt.show()




