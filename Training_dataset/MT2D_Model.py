'''
2D Magnetotelluric Forward Modeling based on Finite Difference Method (FDM)
'''
import ray
import numpy as np
import scipy.io as scio
import scipy.sparse as scipa   # Provides sparse matrix storage and operations, reducing memory and computation time
import scipy.sparse.linalg as scilg  # Provides sparse linear algebra operations
import cmath as cm  # For complex number operations

def cinterp(x, xp, fp):
    return np.interp(x, xp, fp.real)+1j*np.interp(x, xp, fp.imag)
@ray.remote#(num_cpus=10, num_gpus=10)
class MT2DFD(object):
    # 2D Magnetotelluric forward modeling class
    def __init__(self, nza, zn, yn, freq, obs, sig, n_add=10):
        self.nza = nza               # number of air layers
        self.miu = 4.0e-7*np.pi      # vacuum magnetic permeability (physical constant)
        self.II = cm.sqrt(-1)        # imaginary unit j (for frequency-domain calculations)
        self.zn = zn                 # z-direction node coordinates (depth)
        self.nz = len(zn)            # number of z nodes
        self.dz = zn[1:] - zn[:-1]   # grid sizes in z direction (thickness of each layer)
        self.yn = yn                 # y-direction node coordinates (horizontal)
        self.ny = len(yn)            # number of y nodes
        self.dy = yn[1:] - yn[:-1]   # grid sizes in y direction (width of each cell)
        self.freq = freq             # frequency array
        self.nf = np.size(freq)      # number of frequencies
        self.obs = obs               # observation system, positions of measurement points
        self.n_obs = len(obs)        # number of observation points
        self.sig = sig               # conductivity array
        self.n_add = n_add           # number of added points for refinement
        # Check conductivity array size
        if np.shape(sig) != (self.nz-1, self.ny-1):
            raise ValueError("bad size of sigma, must be (nz-1,ny-1)")
        #Background conductivity (using left boundary as background)
        self.sig_back = np.ones_like(sig)*sig[:, 0:1]
        self.sig_diff = self.sig - self.sig_back  # conductivity difference
    def MT2D(self, mode="TETM"): #2D MT forward operator
        dy = self.dy
        dz = self.dz
        sig = self.sig
        sig_diff = self.sig_diff
        yn = self.yn
        obs = self.obs
        nza = self.nza
        n_add = self.n_add
        # Initialize output arrays for impedance Zxy, Zyx, apparent resistivity Rhoxy, Rhoyx, phase Phsxy, Phsyx
        Zxy = np.zeros((self.nf, self.n_obs), dtype=complex)
        Zyx = np.zeros((self.nf, self.n_obs), dtype=complex)
        Rhoxy = np.zeros((self.nf, self.n_obs), dtype=float)
        Phsxy = np.zeros((self.nf, self.n_obs), dtype=float)
        Rhoyx = np.zeros((self.nf, self.n_obs), dtype=float)
        Phsyx = np.zeros((self.nf, self.n_obs), dtype=float)

        # Loop over all frequencies.
        if mode == "TE": # Transverse Electric mode
            #1. Solve for Ex (electric field)
            for kf in range(0, self.nf):
                # 1. Solve for Ex (electric field)
                Ex = self.MT2DTE(self.freq[kf], dy, dz, sig, self.sig_diff, n_add)
                # 2. Compute auxiliary fields Hy, Hz
                Hys, Hzs = self.MT2DHyHz(self.freq[kf], dy, dz, sig, Ex)
                # 3. Extract surface field values
                Exs = Ex[nza,:]  #electric field at the surface
                # 4. Interpolate to observation points
                Exr = cinterp(obs, yn, Exs) #complex interpolation
                Hyr = cinterp(obs, yn, Hys)
                # 5. Compute impedance parameters
                Zxy[kf, :], Rhoxy[kf, :], Phsxy[kf, :] = self.MT2DZxy(self.freq[kf], Exr, Hyr)
            return Rhoxy, Phsxy, Zxy

        elif mode == "TM": #Transverse Magnetic mode (no air layer in TM)
            dz_tm = self.dz[nza:]        # remove air layer thickness
            sig_tm = self.sig[nza:, :]   # remove air layer conductivity
            sig_diff_tm = self.sig_diff[nza:, : ]
            for kf in range(0, self.nf):
                # 1. Solve for Hx (magnetic field)
                Hx = self.MT2DTM(self.freq[kf], dy, dz_tm, sig_tm, sig_diff_tm, n_add)
                # 2. Compute auxiliary fields Ey, Ez
                Eys, Ezs = self.MT2DEyEz(self.freq[kf], dy, dz_tm, sig_tm, Hx)
                # 3. Extract surface field values
                Hxs = Hx[0, :]
                # 4. Interpolate to observation points
                Hxr = cinterp(obs, yn, Hxs)
                Eyr = cinterp(obs, yn, Eys)
                Zyx[kf, :], Rhoyx[kf, :], Phsyx[kf, :] = self.MT2DZyx(self.freq[kf], Hxr, Eyr)
            return Rhoyx, Phsyx, Zyx

        elif mode == "TETM":
            # Execute TE mode code
            for kf in range(0, self.nf):
                Ex = self.MT2DTE(self.freq[kf], dy, dz, sig, sig_diff, n_add)
                Hys, Hzs = self.MT2DHyHz(self.freq[kf], dy, dz, sig, Ex)
                Exs = Ex[nza,:]
                Exr = cinterp(obs, yn, Exs)
                Hyr = cinterp(obs, yn, Hys)
                Zxy[kf, :], Rhoxy[kf, :], Phsxy[kf, :] = self.MT2DZxy(self.freq[kf], Exr, Hyr)

            # Execute TM mode code
            dz_tm = self.dz[nza:]
            sig_tm = self.sig[nza:, :]  #remove air layer
            sig_diff_tm = self.sig_diff[nza:, :]
            for kf in range(0, self.nf):
                Hx = self.MT2DTM(self.freq[kf], dy, dz_tm, sig_tm, sig_diff_tm, n_add)
                Eys, Ezs = self.MT2DEyEz(self.freq[kf], dy, dz_tm, sig_tm, Hx)
                Hxs = Hx[0, :]
                Hxr = cinterp(obs, yn, Hxs)
                Eyr = cinterp(obs, yn, Eys)
                Zyx[kf, :], Rhoyx[kf, :], Phsyx[kf, :] = self.MT2DZyx(self.freq[kf], Hxr, Eyr)
            return Rhoxy, Phsxy, Zxy, Rhoyx, Phsyx, Zyx

    def MT2DTE(self, freq, dy, dz, sig, sig_diff, n_add):
        '''
        Compute the secondary electric field in TE mode.
        Core equation: Total field = primary field (1D) + secondary field (2D scattering)
        Solved equation: (∇² + iωμσ) E_secondary = -iωμ(σ_diff) * E_primary
        '''
        # 1. Basic physical parameters and grid preparation
        omega = 2.0*np.pi*freq   # angular frequency
        ny = self.ny-1           # number of effective cells in y direction
        nz = self.nz-1           # number of effective cells in z direction (excluding boundary nodes)

        # Generate grid size matrices (2D arrays)
        dy0, dz0 = np.meshgrid(dy, dz)

        # Compute cell center distances (staggered grid concept)
        # dyc: distance between centers of two cells perpendicular to the Z-axis
        dyc = (dy0[0:nz-1, 0:ny-1]+dy0[0:nz-1, 1:ny])/2.0
        # dzc: distance between centers of two cells perpendicular to the Y-axis
        dzc = (dz0[0:nz-1, 0:ny-1]+dz0[1:nz, 0:ny-1])/2.0

        #  Compute weighted areas for control volumes (four sub-areas around a node)
        w1 = dy0[0:nz-1, 0:ny-1] * dz0[0:nz-1, 0:ny-1]  # top-left
        w2 = dy0[0:nz-1, 1:ny] * dz0[0:nz-1, 0:ny-1]    # top-right
        w3 = dy0[0:nz-1, 0:ny-1] * dz0[1:nz, 0:ny-1]    # bottom-left
        w4 = dy0[0:nz-1, 1:ny] * dz0[1:nz, 0:ny-1]      # bottom-right
        area = (w1+w2+w3+w4)/4.0    # average area / control volume

        # 2. Physical property interpolation (conductivity averaging)
        # For TE mode, Ex is tangential continuous; current flows mainly parallel to layers.
        # Use arithmetic average for effective conductivity at nodes.
        sigc = (sig[0:nz-1, 0:ny-1]*w1 + sig[0:nz-1,1:ny]*w2 + sig[1:nz,:ny-1]*w3 + sig[1:nz,1:ny]*w4)/(area*4.0)

        # 3. Build sparse matrix coefficients (5-point finite difference)
        # Difference equation: (Left + Right + Up + Down + Self) * E = Source
        # Compute coefficient val (from Laplace operator ∇²)
        # dzc/dy0: part of d²E/dy²
        # dyc/dz0: part of d²E/dz²
        val = dzc/dy0[0:nz-1, 0:ny-1] + dzc/dy0[0:nz-1, 1:ny] + dyc/dz0[0:nz-1, 0:ny-1] +dyc/dz0[1:nz, 0:ny-1]

        # mtx1: main diagonal
        # Includes: Helmholtz term iωμσ*area - Laplace central term val
        mtx1 = self.II*omega*self.miu*sigc*area - val
        mtx1 = mtx1.flatten('F')  # column-major flattening

        # mtx2: coupling in Z direction (d²E/dz²)
        mtx20 = dyc[1:nz-1,0:ny-1]/dz0[1:nz-1, 0:ny-1]
        mtx2 = np.concatenate((mtx20, np.zeros((1, ny-1))),0)
        mtx2 = mtx2.flatten('F')[:-1]   # remove last extra zero

        # mtx3: coupling in Y direction (d²E/dy²)
        mtx3 = dzc[0:nz-1, 1:ny-1]/dy0[0:nz-1,1:ny-1]
        mtx3 = mtx3.flatten('F')

        # Assemble sparse matrix A
        k2 = nz
        mtxA = scipa.diags(mtx1, 0, format='csc') + \
               scipa.diags(mtx2, -1, format='csc') + \
               scipa.diags(mtx2, 1, format='csc') + \
               scipa.diags(mtx3, 1 - k2, format='csc') + \
               scipa.diags(mtx3, k2 - 1, format='csc')

        # 4. Compute right-hand side (RHS) for secondary field source
        # 4.1 Compute primary field (background) Ex1d
        # Use leftmost column as background, subtract anomaly sig_diff
        Ex1d,_ = self.MT1DTE(freq, dz, (sig-sig_diff)[:, 0], n_add)
        # Expand 1D field to 2D (assuming layered background)
        Ex1d = Ex1d.reshape(-1,1)*np.ones((nz+1, ny+1))

        # 4.2 Compute scattering source term: -iωμ * (σ_anomalous) * E_primary
        # Weighted average of anomalous conductivity
        sigc_diff = (sig_diff[0:nz-1, 0:ny-1]*w1 + sig_diff[0:nz-1,1:ny]*w2 + sig_diff[1:nz, :ny-1]*w3 + sig_diff[1:nz,1:ny]*w4)/(area*4.0)
        coef = self.II * omega * self.miu * sigc_diff * area
        rhs = coef * Ex1d[1:nz, 1:ny]     # primary field at cell centers
        rhs = - rhs    # sign after moving term
        rhs = rhs.reshape((ny-1)*(nz-1), 1, order='F')

        # 5. Solve linear system and assemble total field
        Ex, _ = self.equation_solve(mtxA, rhs)
        Ex2d = Ex.reshape(nz-1, ny-1, order='F')
        Ex0d = Ex1d
        # Update interior; boundaries remain as primary field (Dirichlet condition)
        Ex0d[1:nz, 1:ny] = Ex1d[1:nz, 1:ny] + Ex2d

        return Ex0d

    def MT2DTM(self, freq, dy, dz, sig, sig_diff, n_add):
        '''
        Compute the magnetic field Hx in TM mode.
        TM equation involves derivatives of resistivity (rho = 1/sig), more complicated than TE.
        Solved equation: ∇·(ρ∇Hx) + iωμHx = 0 (total field)
        Then transformed to solve secondary field H_sec.
        '''
        # 1. Basic physical parameters and grid
        omega = 2.0 * np.pi*freq
        ny = len(dy)
        nz = len(dz)
        dy0, dz0 = np.meshgrid(dy, dz)

        # Compute half-widths for control volumes around nodes
        dyc = (dy0[0:nz-1, 0:ny-1]+dy0[0:nz-1, 1:ny])/2.0
        dzc = (dz0[0:nz-1, 0:ny-1]+dz0[1:nz, 0:ny-1])/2.0

        w1 = 2 * dz0[0:nz-1, 0:ny-1]   # dz
        w2 = 2 * dz0[1:nz, 0:ny-1]     # dz2
        w3 = 2 * dy0[0:nz-1, 0:ny-1]   # dy1
        w4 = 2 * dy0[0:nz-1, 1:ny]     # dy2
        area = dyc * dzc               # area of control volume

        # Compute transmission coefficients A, B, C, D.
        # A, B ~ dy/dz (Z-direction transmission)
        A = (1.0/sig[0:nz-1, 0:ny-1] * dy0[0:nz-1, 0:ny-1] + 1.0/sig[0:nz-1, 1:ny] * dy0[0:nz-1,1:ny])/w1
        B = (1.0/sig[1:nz, 0:ny-1] * dy0[0:nz-1, 0:ny-1] + 1.0/sig[1:nz, 1:ny] * dy0[0:nz-1, 1:ny])/w2
        # C, D ~ dz/dy (Y-direction transmission)
        C = (1.0/sig[0:nz-1, 0:ny-1] * dz0[0:nz-1, 0:ny-1] + 1.0/sig[1:nz, 0:ny-1] * dz0[1:nz,0:ny-1])/w3
        D = (1.0/sig[0:nz-1, 1:ny] * dz0[0:nz-1, 0:ny-1] + 1.0/sig[1:nz, 1:ny] * dz0[1:nz, 0:ny-1])/w4

        # Main diagonal mtx1
        mtx1 = self.II * omega * self.miu * area - A - B - C - D
        mtx1 = mtx1.flatten('F')

        # Z-direction coupling mtx2 (vertical neighbors)
        mtx = B[0:nz-2, 0:ny-1]
        mtx2 = np.concatenate((mtx, np.zeros((1, ny-1))), 0)
        mtx2 = mtx2.flatten('F')[:-1]

        # Y-direction coupling mtx3 (horizontal neighbors)
        mtx3 = D[0:nz-1, 0:ny-2]
        mtx3 = mtx3.flatten('F')

        k2 = nz # column length (number of rows)
        mtxA = scipa.diags(mtx1, 0, format='csc') + \
               scipa.diags(mtx2, -1, format='csc') + \
               scipa.diags(mtx2, 1, format='csc') + \
               scipa.diags(mtx3, 1 - k2, format='csc') + \
               scipa.diags(mtx3, k2 - 1, format='csc')

        # 2. Compute right-hand side
        Ey1d, Hx1d = self.MT1DTM(freq, dz, (sig-sig_diff)[:,0], n_add)
        Ey1d = Ey1d.reshape(-1,1) * np.ones((nz+1, ny+1))
        Hx1d = Hx1d.reshape(-1,1) * np.ones((nz+1, ny+1))
        # Recompute auxiliary area weights for source term
        A1 = dy0[0:nz-1, 0:ny-1] * dz0[0:nz-1, 0:ny-1]
        A2 = dy0[0:nz-1, 1:ny] * dz0[0:nz-1, 0:ny-1]
        A3 = dy0[0:nz-1, 0:ny-1] * dz0[1:nz, 0:ny-1]
        A4 = dy0[0:nz-1, 1:ny] * dz0[1:nz, 0:ny-1]
        area_src = (A1+A2+A3+A4)/4.0
        sig_scale = sig_diff/sig
        sigc_diff = (sig_scale[0:nz-1, 0:ny-1] * A1 + sig_scale[0:nz-1, 1:ny] * A2 + sig_scale[1:nz, 0:ny-1] * A3 + sig_scale[1:nz,1:ny]*A4)/(area_src * 4.0)
        coef = self.II * omega * self.miu * sigc_diff * area_src
        rhs = coef * Hx1d[1:nz, 1:ny]

        # Compute vertical derivative correction term Ey_d
        sigc_t = (sig_scale[0:nz-1, 0:ny-1] * dy0[0:nz-1,0:ny-1] + sig_scale[0:nz-1,1:ny] * dy0[0:nz-1,1:ny])/(dy0[0:nz-1, 0:ny-1] + dy0[0:nz-1, 1:ny])
        sigc_b = (sig_scale[1:nz, 0:ny-1] * dy0[1:nz, 0:ny-1] + sig_scale[1:nz, 1:ny] * dy0[1:nz, 1:ny])/(dy0[1:nz,0:ny-1]+dy0[1:nz, 1:ny])
        Ey_d = (sigc_b - sigc_t)/((dz0[0:nz-1, 0:ny-1]+dz0[1:nz,0:ny-1])/2.0) * area_src * Ey1d[1:nz, 1:ny]
        rhs = rhs - Ey_d
        rhs = - rhs
        rhs = rhs.reshape((ny-1)*(nz-1), 1, order='F')

        # 3. Solve system
        Hx, _ = self.equation_solve(mtxA, rhs)
        Hx2d = Hx.reshape(nz-1, ny-1, order='F')
        # Total field = primary + secondary
        Hx0d = Hx1d
        Hx0d[1:nz, 1:ny] = Hx1d[1:nz, 1:ny] + Hx2d
        return Hx0d

    def MT2DHyHz(self, freq, dy, dz, sig, Ex):
        '''
        Compute auxiliary magnetic fields Hy and Hz in TE mode.
        Physical principle: Faraday's law ∇ × E = -iωμH
        1. Hy = -1/(iωμ) * (∂Ex/∂z)
        Uses high-order interpolation to handle conductivity jumps at the surface.
        2. Hz =  1/(iωμ) * (∂Ex/∂y)
        Uses central differences.
        Input:
            Ex: total electric field (nz+1, ny+1)
            dy, dz: grid spacings
            sig: conductivity matrix
        '''
        omega = 2.0*np.pi * freq
        ny = np.size(dy)  #number of cells in y direction
        # 1. Compute Hy (tangential magnetic field)
        # Hy is used to compute apparent resistivity (Zxy = Ex / Hy), requires high accuracy.
        Hys = np.zeros((ny+1), dtype=complex)

        kk = self.nza                # node index of the surface (after air layers)
        delz = dz[kk]                # thickness of the first cell below surface
        # 1.1 Hy at left boundary node
        sigc = sig[kk, 0]
        c0 = -1.0/(self.II * omega * self.miu * delz) + (3.0/8.0) * sigc * delz
        c1 = 1.0/(self.II * omega * self.miu * delz) + (1.0/8.0) * sigc * delz
        Hys[0] = c0*Ex[kk, 0] + c1*Ex[kk+1, 0]

        # 1.2 Hy at right boundary node
        sigc = sig[kk, ny-1]
        c0 = 1.0/(self.II * omega * self.miu * delz) + (3.0/8.0) * sigc * delz
        c1 = -1.0/(self.II * omega * self.miu*delz) + (1.0/8.0) * sigc*delz
        Hys[ny] = c0 * Ex[kk, ny] + c1 * Ex[kk+1, ny]

        # 1.3 Internal nodes (vectorized)
        dyj = dy[0:ny-1]+dy[1:ny]     # distance between nodes i-1 and i+1
        sigc = (sig[kk, 0:ny-1] * dy[0:ny-1] + sig[kk, 1:ny] * dy[1:ny])/dyj
        cc = delz/(4.0 * self.II * omega * self.miu * dyj)

        # Coefficients for 6-point stencil
        c0 = -1.0/(self.II * omega * self.miu * delz) + (3.0/8.0) * sigc * delz - cc * 3.0 * (1.0/dy[1:ny] + 1.0/dy[0:ny-1])
        c1 = 1.0/(self.II * omega * self.miu * delz) + (1.0/8.0) * sigc * delz - cc * 1.0 * (1.0/dy[1:ny] + 1.0/dy[0:ny-1])
        c0l = 3.0 * cc/dy[0:ny-1]
        c0r = 3.0 * cc/dy[1:ny]
        c1l = 1.0 * cc/dy[0:ny-1]
        c1r = 1.0 * cc/dy[1:ny]
        Hys[1:ny] = c0l * Ex[kk, 0:ny-1] + c0 * Ex[kk, 1:ny] + c0r * Ex[kk, 2:ny+1] + c1l * Ex[kk+1, 0:ny-1] + c1 * Ex[kk+1, 1:ny] + c1r * Ex[kk+1, 2:ny+1]

        # 2. Compute vertical magnetic field Hz
        # Hz = 1/(iωμ) * (∂Ex/∂y)
        Hzs = np.zeros((ny+1),dtype=complex)
        # 2.1 Boundaries: forward/backward differences
        Hzs[0] = -1.0/(self.II * omega * self.miu) * (Ex[kk, 1]-Ex[kk,0])/dy[0]
        Hzs[ny] = -1.0/(self.II * omega * self.miu) * (Ex[kk, ny]-Ex[kk, ny-1])/dy[ny-1]
        # 2.2Internal: central difference
        Hzs[1:ny] = -1.0/(self.II*omega*self.miu)*(Ex[kk, 2:ny+1]-Ex[kk,0:ny-1])/(dy[0:ny-1]+dy[1:ny])
        return Hys, Hzs

    def MT2DEyEz(self, freq, dy, dz, sig, Hx):
        '''
         Compute auxiliary electric fields Ey and Ez in TM mode.
        Physical principle: Ampere's law ∇ × H = σE
        1. Ey =  1/σ * (∂Hx/∂z) = ρ * (∂Hx/∂z)
        Uses high-order interpolation to handle surface resistivity variations.
        2. Ez = -1/σ * (∂Hx/∂y) = -ρ * (∂Hx/∂y)
        Input:
            Hx: total magnetic field (nz+1, ny+1). In TM mode there is no air layer, so index 0 is the surface.
        '''
        omega = 2.0 * np.pi*freq
        ny = np.size(dy)
        # 1. Compute Ey, tangential electric field
        # Ey used for apparent resistivity (Zyx = Ey / Hx).
        Eys = np.zeros((ny+1), dtype=complex)
        kk = 0         # TM mode no air layer, surface index 0
        delz = dz[kk]  # thickness of first layer below surface

        # 1.1 Ey at left boundary node
        sigc = sig[kk, 0] # resistivity at node
        temp_beta = self.II * omega * self.miu * delz
        temp_1 = sigc * delz
        c0 = -1.0/temp_1 + (3.0/8.0) * temp_beta
        c1 = 1.0/temp_1 + (1.0/8.0) * temp_beta
        Eys[0] = c0 * Hx[kk, 0] + c1 * Hx[kk+1, 0]

        # 1.2 Ey at right boundary node
        sigc = sig[kk, ny-1]
        temp_1 = sigc * delz
        c0 = -1.0/temp_1 + (3.0/8.0) * temp_beta
        c1 = 1.0/temp_1 + (1.0/8.0) * temp_beta
        Eys[ny] = c0 * Hx[kk, ny] + c1 * Hx[kk+1, ny]

        # 1.3 Internal nodes (vectorized)
        dy_sum = dy[0:ny-1] + dy[1:ny]
        dyj = (dy[0:ny-1]+dy[1:ny])/2.0
        tao = 1.0/sig[kk, 0:ny]
        # Weighted average resistivity at node
        taoc = (tao[0:ny-1] * dy[0:ny-1] + tao[1:ny] * dy[1:ny])/dy_sum
        temp_1 = self.II * omega * self.miu * delz
        temp_2 = taoc/delz      # dominant term: ρ/Δz (∂H/∂z)
        temp_3 = delz/dy_sum    # geometric factor
        temp_4 = tao/dy         # weight for horizontal gradient

        c0 = (3.0/8.0) * temp_1 - temp_2
        c1 = (1.0/8.0) * temp_1 + temp_2 - (1.0/8.0) * temp_3 * (temp_4[0:ny-1] + temp_4[1:ny])
        c1l = (1.0/8.0) * temp_3 * temp_4[0:ny-1]
        c1r = (1.0/8.0) * temp_3 * temp_4[1:ny]
        Eys[1:ny] = c0 * Hx[kk, 1:ny] + c1l * Hx[kk+1, 0:ny-1] + c1 * Hx[kk+1, 1:ny] + c1r * Hx[kk+1, 2:ny+1]

        # 2. Compute vertical electric field Ez
        Ezs = np.zeros((ny + 1), dtype=complex)
        rho = 1.0 / sig[kk, :]  # cell resistivities
        # 2.1 Boundaries
        rho_node_0 = rho[0]
        Ezs[0] = -rho_node_0 * (Hx[kk, 1] - Hx[kk, 0]) / dy[0]
        rho_node_ny = rho[ny - 1]
        Ezs[ny] = -rho_node_ny * (Hx[kk, ny] - Hx[kk, ny - 1]) / dy[ny - 1]
        # 2.2 Internal nodes
        # Node resistivity: weighted average of left and right cells
        rho_node_internal = (rho[0:ny-1]*dy[1:ny] + rho[1:ny]*dy[0:ny-1]) / dy_sum
        dH_dy = (Hx[kk, 2:ny + 1] - Hx[kk, 0:ny - 1]) / dy_sum
        Ezs[1:ny] = -rho_node_internal * dH_dy

        return Eys, Ezs

    def MT2DZxy(self, freq, Exr, Hyr):
        #Compute response for TE mode (Ex, Hy)
        omega = 2.0 * np.pi * freq
        # 1. Impedance Zxy
        Zxy = np.array(Exr/Hyr, dtype=complex)
        # 2. Apparent resistivity RhoXY
        RhoTE = abs(Zxy)**2/(omega*self.miu)
        # 3. Phase PhsXY (usually between 0 and 90 degrees for TE)
        PhsTE = np.degrees(np.arctan2(Zxy.imag, Zxy.real))
        PhsTE = -PhsTE
        return Zxy, RhoTE, PhsTE

    def MT2DZyx(self, freq, Hxr, Eyr):
        # Compute response for TM mode (Ey, Hx)
        # Note: Zyx = Ey/Hx, in 1D case Zyx = -Zxy
        omega =2.0 * np.pi * freq
        # 1. Impedance Zyx
        Zyx = np.array(Eyr/Hxr, dtype=complex)
        # 2. Apparent resistivity RhoYX
        RhoTM = abs(Zyx)**2/(omega*self.miu)
        # 3. Phase PhsYX
        PhsTM = np.degrees(np.arctan2(Zyx.imag, Zyx.real))
        PhsTM = 180-PhsTM

        return Zyx, RhoTM, PhsTM

    def MT1DTE(self, freq, dz0, sig0, n_add):
        # Compute 1D TE mode (Ex, Hy) electromagnetic field distribution.
        # Used as left/right boundary conditions for 2D TE mode
        omega = 2.0 * np.pi * freq
        self.miu = 4.0e-7 * np.pi
        # 1. Refine grid
        dz = np.repeat(dz0 / n_add, n_add)
        sig = np.repeat(sig0, n_add)
        nz = len(dz)
        #  2. Extend bottom boundary with one skin depth to simulate infinite half-space
        skin_depth = np.sqrt(2.0 / (sig[-1] * omega * self.miu))
        sig = np.append(sig, sig[-1])
        dz = np.append(dz, skin_depth)
        # 3. Build system matrix
        diagA = self.II * omega * self.miu * (sig[0:nz] * dz[0:nz] + sig[1:nz+1] * dz[1:nz+1]) - 2.0/dz[0:nz] - 2.0/dz[1:nz+1]
        offdiagA = 2.0/dz[1:nz]
        mtxA = scipa.diags(diagA, 0, format='csc') + \
               scipa.diags(offdiagA, 1, format='csc') + \
               scipa.diags(offdiagA, -1, format='csc')

        # 4. RHS with Dirichlet boundary condition Ex[0] = 1.0
        rhs = np.zeros(nz, dtype=complex)
        rhs[0] = -2.0/dz[0] * 1.0

        # 5. Solve
        lup = scilg.splu(mtxA)
        Ex0 = lup.solve(rhs)
        Ex = np.concatenate(([1.0], Ex0))

        # 6. Compute magnetic field Hy using Faraday's law: dEx/dz = -iωμHy
        dEx_dz = (Ex[1:] - Ex[:-1]) / dz[:-1]
        Hy0 = -1.0 / (self.II * omega * self.miu) * dEx_dz
        Hy = np.append(Hy0, Hy0[-1])
        # 7. Extract values at original grid nodes
        idx = np.arange(len(dz0) + 1) * n_add
        return Ex[idx], Hy[idx]

    def MT1DTM(self, freq, dz0, sig0, n_add):
        # Compute 1D TM mode (Ey, Hx) electromagnetic field distribution.
        # Used as boundary conditions for 2D TM mode.
        # Physical principle:
        #     Helmholtz Eq: d/dz(rho * dHx/dz) - iωμHx = 0
        #     Auxiliary field: Ey = rho * (dHx/dz)
        omega = 2.0 * np.pi * freq
        # 1. Refine grid
        dz = np.repeat(dz0 / n_add, n_add)
        sig = np.repeat(sig0, n_add)
        nz = len(dz)
        # 2. Extend bottom boundary with one skin depth
        skin_depth = np.sqrt(2.0 / (sig[-1] * omega * self.miu))
        dz = np.append(dz, skin_depth)
        sig = np.append(sig, sig[-1])

        # 3. Build system matrix for Hx
        diagA = self.II * omega * self.miu * (dz[0:nz]+dz[1:nz+1]) - 2.0/(dz[0:nz]*sig[0:nz]) - 2.0/(dz[1:nz+1]*sig[1:nz+1])
        offdiagA = 2.0/(dz[1:nz]*sig[1:nz])
        mtxA = scipa.diags(diagA, 0, format='csc') + \
               scipa.diags(offdiagA, -1, format='csc') + \
               scipa.diags(offdiagA,1, format='csc')
        # 4.RHS with Dirichlet condition Hx[0] = 1.0
        rhs = np.zeros(nz, dtype=complex)
        rhs[0] = -2.0/(dz[0]*sig[0]) * 1.0
        # 5.Solve
        lup = scilg.splu(mtxA)
        Hx0 = lup.solve(rhs)
        Hx = np.concatenate(([1.0], Hx0))
        # 6. Compute Ey = ρ * dHx/dz
        Ey0 = (Hx[1:]-Hx[:-1])/dz[:-1]/sig[:-1]
        Ey = np.append(Ey0, Ey0[-1])

        # 7. Extract at original nodes
        idx = np.arange(len(dz0) + 1) * n_add
        return Ey[idx], Hx[idx]

    def equation_solve(self, mtxA, rhs):
        # Solve sparse linear system Ax = b.
        # For 2D MT, direct SPLU (SuperLU) solver is recommended.
        # It is more stable than iterative solvers (BiCGSTAB) and has no convergence issues.
        try:
            lu_obj = scilg.splu(mtxA)
            x = lu_obj.solve(rhs)
            return x, 0
        except RuntimeError as e:
            print(f"Matrix solve failed: {e}")
            return np.zeros_like(rhs), 1

def save_model(model_name, zn, yn, freq, obs, sig_log, Rhoxy, Phsxy, Zxy, Rhoyx, Phsyx, Zyx):
    '''
    Save results to a .mat file.
    Input:
        model_name: filename
        obs: observation point coordinates
        sig_log: logarithmic conductivity model
        Rhoxy, Phsxy, ...: computed response data
    '''
    scio.savemat(model_name,{'zn': zn, 'yn': yn, 'freq': freq, 'obs': obs, 'sig': sig_log,
                                   'Rhoxy': Rhoxy, 'Phsxy': Phsxy, 'Zxy': Zxy,
                                   'Rhoyx': Rhoyx, 'Phsyx': Phsyx, 'Zyx': Zyx})
    print(f"Model saved to {model_name}")

def func_remote(nza, zn, yn, freq, obs, sig, n_sample, mode="TETM", np_dtype = np.float64):
    '''
    Compute MT responses for multiple samples in parallel using Ray.
    Input:
        obs: observation Y-coordinate array
        sig: conductivity array of shape (n_sample, nz, ny)
        mode: "TE", "TM", or "TETM"
    '''
    n_freq = np.size(freq)
    n_obs = len(obs)
    Rhoxy = np.zeros((n_sample, n_freq, n_obs), dtype=np_dtype)
    Phsxy = np.zeros((n_sample, n_freq, n_obs), dtype=np_dtype)
    Rhoyx = np.zeros((n_sample, n_freq, n_obs), dtype=np_dtype)
    Phsyx = np.zeros((n_sample, n_freq, n_obs), dtype=np_dtype)
    Zxy = np.zeros((n_sample, n_freq, n_obs), dtype=complex)
    Zyx = np.zeros((n_sample, n_freq, n_obs), dtype=complex)

    # 1.Submit tasks to Ray (asynchronous calls)
    result_ids = []
    print(f"Starting parallel computation for {n_sample} samples with mode={mode}...")

    for ii in range(n_sample):
        model = MT2DFD.remote(nza, zn, yn, freq, obs, sig[ii, :, :])
        result_ids.append(model.MT2D.remote(mode))
    # 2. Get results (blocking)
    result_data = ray.get(result_ids)
    # 3. Unpack results according to mode
    for ii in range(n_sample):
        temp = result_data[ii]
        if mode == "TETM":
            Rhoxy[ii], Phsxy[ii], Zxy[ii] = temp[0], temp[1], temp[2]
            Rhoyx[ii], Phsyx[ii], Zyx[ii] = temp[3], temp[4], temp[5]
        elif mode == "TE":
            Rhoxy[ii], Phsxy[ii], Zxy[ii] = temp[0], temp[1], temp[2]
            # TM parts remain zero
        elif mode == "TM":
            Rhoyx[ii], Phsyx[ii], Zyx[ii] = temp[0], temp[1], temp[2]
            # TE parts remain zero
    print("Computation finished.")
    return Rhoxy, Phsxy, Zxy, Rhoyx, Phsyx, Zyx
