"""
================================================================================
Gaussian Random Field Generator - For Magnetotelluric 2D Random Resistivity Models
================================================================================
Core algorithm:
    Generate Gaussian random field using frequency-domain power spectrum method
    1. Construct power-law power spectrum in frequency domain: P(k) ~ k^(-alpha/2)
    2. Generate random-phase complex Gaussian white noise
    3. Synthesize in frequency domain and inverse FFT to spatial domain
    4. Normalize and adjust to target distribution (Gaussian or bounded interval)
Main features:
    - Supports arbitrary grid sizes
    - Controls field smoothness and correlation length via alpha parameter
    - Two output modes:
      * random mode: outputs Gaussian distribution N(μ, σ²)
      * bound mode: outputs bounded interval [a, b]
    - Suitable for geostatistical random medium models
Applications:
    - Magnetotelluric random resistivity model generation
    - Seismic velocity model random perturbations
    - Spatial random field simulation in geostatistics
    - Random sample generation for machine learning training data

================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
# ================================================================================
# Core function: Gaussian random field generator
# ================================================================================
# ==================== Configure matplotlib font ====================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'SimHei']
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = ['serif', 'sans-serif']
def gaussian_random_field(alpha=7, shape=(100, 50), mode='random', set_1=0.0, set_2=1.0):
    """
    Generate a 2D Gaussian random field.

    Theoretical background:
        A Gaussian random field is a spatially correlated random process whose
        statistical properties are described by a covariance function or power spectrum.
        This function uses a frequency-domain method, specifying the power spectral
        density P(k) ~ k^(-alpha/2) to control the spatial correlation and smoothness.
    Algorithm steps:
        Step 1: Generate frequency grid (wavenumber space)
        Step 2: Construct power-law power spectrum
        Step 3: Generate random-phase complex Gaussian white noise
        Step 4: Synthesize in frequency domain and inverse FFT to spatial domain
        Step 5: Normalize to standard normal distribution N(0,1)
        Step 6: Transform to target distribution (Gaussian or bounded)
    Parameters:
        alpha : float, default=7
            Power spectrum exponent, controls smoothness and spatial correlation length.
            - Larger alpha: low-frequency components dominate, field becomes smoother,
              structures become larger.
            - Smaller alpha: high-frequency components increase, field becomes rougher,
              structures become finer.
            - Recommended range: 3-15
            - Typical values: alpha=7 (moderately smooth), alpha=3 (rough),
              alpha=12 (very smooth).
        shape : tuple of int, default=(100, 50)
            2D grid size (Nz, Ny)
            - Nz: number of rows, corresponding to depth direction (X-axis)
            - Ny: number of columns, corresponding to horizontal direction (Y-axis)
            - Example: shape=(50, 100) means 50 rows and 100 columns.
        mode : str, default='random'
            Distribution mode for output values.
            - 'random': Gaussian distribution, unbounded range.
            - 'bound': bounded interval [set_1, set_2].
        set_1 : float, default=0.0
            Distribution parameter 1.
            - When mode='random': mean μ.
            - When mode='bound': lower bound a.
        set_2 : float, default=1.0
            Distribution parameter 2.
            - When mode='random': standard deviation σ (> 0 required).
            - When mode='bound': upper bound b (> set_1 required).
    Returns:
        gfield : ndarray, shape (Nz, Ny)
            Generated 2D Gaussian random field.
            - mode='random': follows N(set_1, set_2²) distribution.
            - mode='bound': values strictly within [set_1, set_2].
    Examples:
        # Example 1: Generate standard normal random field
        field1 = gaussian_random_field(alpha=7, shape=(100, 100), mode='random', set_1=0.0, set_2=1.0)
        # Example 2: Generate resistivity field in [10, 1000] Ohm·m
        field2 = gaussian_random_field(alpha=10, shape=(50, 100),mode='bound', set_1=10, set_2=1000)
        # Example 3: Generate log10(resistivity) in [1, 3]
        log_rho = gaussian_random_field(alpha=8, shape=(40, 80), mode='bound', set_1=1.0, set_2=3.0)
        rho = 10 ** log_rho   # convert to resistivity
    Notes:
        1. Choice of alpha:
           - Geological models typically use alpha=6-10, giving moderately smooth structures.
           - Too small alpha (<3) produces noise-like fields.
           - Too large alpha (>15) produces overly smooth fields lacking detail.
        2. Grid size:
           - Larger grids increase computation time (FFT complexity O(N log N)).
           - Recommended grid size between 50×50 and 200×200.
        3. Randomness:
           - Each call generates a different random field.
           - For reproducible results, set random seed before calling: np.random.seed(seed).
    """
    # --------------------------------------------------------------------------------
    # Step 0: Parameter extraction and initialization
    # --------------------------------------------------------------------------------
    Nz, Ny = shape   # Nz = rows (depth), Ny = columns (horizontal)
    # --------------------------------------------------------------------------------
    # Step 1: Generate frequency grid (wavenumber space)
    # --------------------------------------------------------------------------------
    # numpy.fft.fftfreq generates frequency coordinates for discrete Fourier transform.
    # Return range: [-0.5, 0.5) for normalized frequency.
    # Multiply by N to get actual wavenumber range: [-N/2, N/2).
    # Physical meaning:
    #   - freqx[i] is the wavenumber for the i-th frequency component (units: 1/grid spacing)
    #   - Larger wavenumber -> shorter spatial wavelength (high frequency)
    #   - Smaller wavenumber -> longer spatial wavelength (low frequency)
    freqz = np.fft.fftfreq(Nz) * Nz  # wavenumbers in X direction, range [-Nz/2, Nz/2)
    freqy = np.fft.fftfreq(Ny) * Ny  # wavenumbers in Y direction, range [-Ny/2, Ny/2)
    # Generate 2D wavenumber grids
    # meshgrid expands 1D wavenumber arrays into 2D grids.
    # indexing='ij' ensures kx.shape = (Nz, Ny), consistent with spatial grid indexing.
    #   - kx[i, j] = freqx[i]   (wavenumber in X direction)
    #   - ky[i, j] = freqy[j]   (wavenumber in Y direction)
    kz, ky = np.meshgrid(freqz, freqy, indexing='ij')
    # --------------------------------------------------------------------------------
    # Step 2: Construct power-law power spectrum
    # --------------------------------------------------------------------------------
    # Power Spectral Density (PSD) describes the energy distribution of the random field
    # at different frequencies.
    # Formula:
    #   P(k) = C * k^(-alpha/2)
    # where:
    #   - k = sqrt(kx^2 + ky^2) is the radial wavenumber
    #   - alpha is the power spectrum exponent
    #   - C is a normalization constant (not needed here because of later normalization)
    # Relationship between amplitude spectrum and power spectrum:
    #   P(k) = |A(k)|^2
    # Therefore: A(k) = k^(-alpha/4)
    # Physical interpretation:
    #   - alpha/2 = 1: corresponds to 1/f noise (pink noise)
    #   - alpha/2 = 2: corresponds to Brownian motion
    #   - Larger alpha -> stronger low-frequency energy -> smoother field
    k_squared = kz ** 2 + ky ** 2  # square of wavenumber modulus: k^2 = kx^2 + ky^2

    # Compute amplitude spectrum: A(k) = (k^2 + ε)^(-alpha/4)
    # Add a small epsilon to avoid division by zero at k=0
    amplitude = np.power(k_squared + 1e-12, -alpha / 4.0)
    # Set DC component (zero frequency) to zero.
    # Reason: DC component corresponds to constant offset. Setting to zero makes the
    # field have zero mean, facilitating later normalization.
    amplitude[0, 0] = 0
    # --------------------------------------------------------------------------------
    # Step 3: Generate random-phase complex Gaussian white noise
    # --------------------------------------------------------------------------------
    # In frequency domain, each frequency component is a complex number containing
    # amplitude and phase information. To generate a random field, we assign random
    # phases to each frequency component.
    # Method: generate complex Gaussian white noise.
    #   - Real and imaginary parts are independent standard normal N(0,1).
    #   - Thus the phase is uniformly distributed in [0, 2π).
    #   - The squared modulus follows an exponential distribution.
    # Mathematical representation:
    #   noise = Re + i*Im
    #   where Re ~ N(0,1), Im ~ N(0,1) independent.
    noise = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    # --------------------------------------------------------------------------------
    # Step 4: Synthesize in frequency domain and inverse FFT to spatial domain
    # --------------------------------------------------------------------------------
    # Frequency-domain synthesis:
    #   Multiply white noise by target amplitude spectrum to obtain frequency-domain
    #   field with the specified power spectrum.
    #   F(k) = noise(k) * A(k)
    # Inverse FFT:
    #   Transform frequency-domain field to spatial domain.
    #   f(x) = IFFT[F(k)]
    # Take real part:
    #   The input is complex, so the IFFT result is also complex.
    #   However, because we construct a real-valued random field, taking the real part
    #   is correct (the imaginary part is negligible numerical error).
    gfield = np.fft.ifft2(noise * amplitude).real
    # --------------------------------------------------------------------------------
    # Step 5: Normalize to standard normal distribution N(0, 1)
    # --------------------------------------------------------------------------------
    # Purpose:
    #   Normalize the random field to have zero mean and unit standard deviation.
    #   This facilitates subsequent transformation to any target distribution.
    # Steps:
    #   1. Subtract mean: gfield = gfield - mean(gfield)
    #   2. Divide by standard deviation: gfield = gfield / std(gfield)
    # Remove mean
    gfield = gfield - np.mean(gfield)
    # Compute standard deviation and normalize
    std_val = np.std(gfield)
    # Numerical protection: if standard deviation is too small (field nearly constant),
    # set to 1.0 to avoid division by zero.
    if std_val < 1e-14:
        std_val = 1.0
    gfield = gfield / std_val
    # Now gfield follows standard normal distribution N(0, 1)
    # --------------------------------------------------------------------------------
    # Step 6: Transform to target distribution
    # --------------------------------------------------------------------------------
    if mode == 'random':
        # ============================================================================
        # Mode 1: Gaussian distribution N(μ, σ²)
        # ============================================================================
        # Goal: transform N(0,1) to N(set_1, set_2²)
        # Linear transformation formula:
        #   Y = σ * X + μ
        #   where X ~ N(0,1), then Y ~ N(μ, σ²)
        # Parameter mapping:
        #   μ = set_1  (target mean)
        #   σ = set_2  (target standard deviation)
        set_mean = set_1  # target mean
        set_std = set_2   # target standard deviation
        # Linear transformation: multiply by standard deviation, then add mean
        gfield = gfield * set_std + set_mean
        # Output characteristics:
        #   - Follows Gaussian distribution N(set_1, set_2²)
        #   - Unbounded range: (-∞, +∞)
        #   - About 68% of values within [μ-σ, μ+σ]
        #   - About 95% of values within [μ-2σ, μ+2σ]
    elif mode == 'bound':
        # ============================================================================
        # Mode 2: Bounded interval [a, b]
        # ============================================================================
        # Goal: linearly map standard normal distribution to bounded interval [set_1, set_2]
        # Linear mapping formula:
        #   y = (x - x_min) / (x_max - x_min) * (b - a) + a
        # Steps:
        #   1. Find current field's minimum x_min and maximum x_max
        #   2. Normalize to [0, 1]: (x - x_min) / (x_max - x_min)
        #   3. Scale to [a, b]: multiply by (b - a) and add a
        # Parameter mapping:
        #   a = set_1  (lower bound)
        #   b = set_2  (upper bound)
        set_lower = set_1  # lower bound
        set_upper = set_2  # upper bound
        # Compute current field's range
        g_min = np.min(gfield)
        g_max = np.max(gfield)
        # Numerical protection: if field is constant (g_max ≈ g_min), return midpoint of interval
        if g_max - g_min < 1e-14:
            gfield = np.full(shape, (set_lower + set_upper) / 2)
        else:
            # Linear mapping to [set_lower, set_upper]
            gfield = (gfield - g_min) / (g_max - g_min) * (set_upper - set_lower) + set_lower
        # Output characteristics:
        #   - Values strictly within [set_1, set_2]
        #   - No longer Gaussian (distribution is truncated and stretched)
        #   - Suitable for problems with physical constraints (e.g., resistivity must be positive)
    else:
        # Invalid parameter handling
        raise ValueError("mode must be either 'random' or 'bound'")
    return gfield
# ================================================================================
# Test and visualization functions
# ================================================================================
def main():
    """
    Main function: test the Gaussian random field generator and visualize results.
    Functionality:
        1. Generate bound model and random model
        2. Compare visualizations
        3. Output statistics
    """
    print("=" * 80)
    print("Gaussian Random Field Generator Test - Bound Model vs Random Model Comparison")
    print("=" * 80)
    # --------------------------------------------------------------------------------
    # Common parameter settings
    # --------------------------------------------------------------------------------
    test_alpha = 4         # power spectrum exponent (controls smoothness)
    test_shape = (64, 64)  # grid size: 64 rows x 64 columns
    
    # Bound model parameters
    bound_lower = 1.0      # lower bound (e.g., minimum of log10(resistivity))
    bound_upper = 4.0      # upper bound (e.g., maximum of log10(resistivity))
    
    # Random model parameters
    random_mean = 2.0      # mean (e.g., mean of log10(resistivity))
    random_std = 0.5       # standard deviation
    
    # --------------------------------------------------------------------------------
    # Generate Bound model
    # --------------------------------------------------------------------------------
    print(f"\nGenerating Bound model...")
    np.random.seed(123)  # set random seed for reproducibility
    bound_field = gaussian_random_field(
        alpha=test_alpha,
        shape=test_shape,
        mode='bound',
        set_1=bound_lower,
        set_2=bound_upper)
    print(f"Bound model generation complete!")
    # --------------------------------------------------------------------------------
    # Generate Random model
    # --------------------------------------------------------------------------------
    print(f"\nGenerating Random model...")
    np.random.seed(123)  # use same random seed for fair comparison
    random_field = gaussian_random_field(
        alpha=test_alpha,
        shape=test_shape,
        mode='random',
        set_1=random_mean,
        set_2=random_std)
    print(f"Random model generation complete!")
    # --------------------------------------------------------------------------------
    # Comparative visualization
    # --------------------------------------------------------------------------------
    print(f"\nPlotting comparison figures...")
    fig = plt.figure(figsize=(14, 6))
    # Subplot 1: Bound model
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(bound_field, cmap='jet', aspect='auto', origin='upper')
    plt.colorbar(im1, label='Value')
    plt.title('Bound Model', fontsize=15)
    plt.xlabel('y direction', fontsize=15)
    plt.ylabel('z direction', fontsize=15)
    # Subplot 2: Random model
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(random_field, cmap='jet', aspect='auto', origin='upper')
    plt.colorbar(im2, label='Value')
    plt.title('Random Model', fontsize=15)
    plt.xlabel('y direction', fontsize=15)
    plt.ylabel('z direction', fontsize=15)
    plt.tight_layout()
    plt.savefig('Gaussian_Random_Fields_2DModel1.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nTest completed!")
    print("=" * 80)
# ================================================================================
# Program entry point
# ================================================================================
if __name__ == '__main__':
    main()
