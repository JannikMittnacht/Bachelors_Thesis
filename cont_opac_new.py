# =====================================================================
# Adaptation from cont_opac.f90 of FIRTEZ-DZ-MHS
# Version: 0.0.0
# Author: Andreu V. (+ gemini)
# Date: 2026-04-21
# Last edit: 2026-04-29 Andreu V.
# =====================================================================
import numpy as np

# =====================================================================
# CONSTANTS (CGS Units)
# =====================================================================
KBOL = 1.38064852e-16     # Boltzmann constant [erg/K]
LIGHT = 2.99792458e10     # Speed of light [cm/s]
HPLA = 6.62607004e-27     # Planck constant [erg s]
MELE = 9.10938356e-28     # Electron mass [g]
QELE = 4.80320425e-10     # Electron charge [esu]
MAMU = 1.660539040e-24    # Atomic mass unit [g]
EVOLT = 1.60217662e-12    # Electron volt [erg]

# Saha equation constants
CSAHA1 = (HPLA**2 / (2.0 * np.pi * MELE * KBOL))**1.5  # ~ 4.141e-16 cm^3 K^1.5
CSAHA2 = EVOLT / KBOL                                  # ~ 11604.5 K/eV

NELEM = 92

# =====================================================================
# ATOMIC DATABASE
# =====================================================================
# First ionization potential (For H, this is the binding energy of H-) [eV]
XI = np.array([
    0.754, 24.58, 5.39, 9.32, 8.298, 11.256, 14.529, 13.614, 17.418, 21.559, 
    5.138, 7.644, 5.984, 8.149, 10.474, 10.357, 13.012, 15.755, 4.339, 6.111, 
    6.538, 6.825, 6.738, 6.763, 7.432, 7.896, 7.863, 7.633, 7.724, 9.391, 
    5.997, 7.88, 9.81, 9.75, 11.840, 13.996, 4.176, 5.692, 6.377, 6.838, 
    6.881, 7.10, 7.28, 7.36, 7.46, 8.33, 7.574, 8.991, 5.785, 7.34, 
    8.64, 9.01, 10.454, 12.127, 3.893, 5.210, 5.577, 5.466, 5.422, 5.489, 
    5.554, 5.631, 5.666, 6.141, 5.852, 5.927, 6.018, 6.101, 6.184, 6.254, 
    5.426, 6.650, 7.879, 7.980, 7.870, 8.70, 9.10, 9.00, 9.22, 10.43, 
    6.105, 7.415, 7.287, 8.43, 9.30, 10.745, 4.0, 5.276, 6.9, 6.0, 6.0, 6.0
])

# Second ionization potential (For H, this is the standard first ionization) [eV]
XII = np.array([
    13.595, 54.403, 75.62, 18.21, 25.15, 24.376, 29.59, 35.11, 34.98, 41.07,
    47.290, 15.03, 18.823, 16.34, 19.72, 23.405, 23.798, 27.62, 31.81, 11.868,
    12.891, 13.63, 14.205, 16.493, 15.636, 16.178, 17.052, 18.15, 20.286, 17.96,
    20.509, 15.93, 18.63, 21.50, 21.60, 24.565, 27.50, 11.027, 12.233, 13.13,
    14.316, 16.15, 15.26, 16.76, 18.07, 19.42, 21.48, 16.904, 18.86, 14.63,
    16.50, 18.60, 19.09, 21.20, 25.10, 10.001, 11.060, 10.850, 10.550, 10.730,
    10.899, 11.069, 11.241, 12.090, 11.519, 11.670, 11.800, 11.930, 12.050,
    12.184, 13.900, 14.900, 16.2, 17.7, 16.60, 17.00, 20.00, 18.56, 20.50, 
    18.75, 20.42, 15.03, 16.68, 19.0, 20.0, 20.0, 22.0, 10.144, 12.1, 12.0, 
    12.0, 12.0
])

# Abundances (THEVENIN 1989)
ABUND = np.array([
    12.00, 11.00, 1.00, 1.15, 2.60, 8.69, 7.99, 8.91, 4.56, 8.00, 6.28,
    7.53, 6.43, 7.50, 5.45, 7.21, 5.50, 6.58, 5.05, 6.36, 2.99, 4.88, 
    3.91, 5.61, 5.47, 7.46, 4.85, 6.18, 4.24, 4.60, 2.88, 3.57, 2.39, 
    3.35, 2.63, 3.21, 2.60, 2.93, 2.18, 2.46, 1.46, 2.10, 0.00, 1.78, 
    1.10, 1.69, 0.94, 1.86, 1.66, 2.00, 1.00, 2.25, 1.51, 2.19, 1.12, 
    2.18, 1.07, 1.58, 0.76, 1.40, 0.00, 0.88, 0.48, 1.13, 0.20, 1.07, 
    0.26, 0.93, 0.00, 1.08, 0.76, 0.88, -0.09, 0.98, 0.26, 1.45, 1.36, 
    1.80, 1.13, 1.27, 0.90, 1.90, 0.71, -8.0, -8.0, -8.0, -8.0, -8.0, 
    -8.0, 0.02, -8.0, -0.47
])

# Atomic mass [amu]
MATOM = np.array([
    1.008, 4.003, 6.939, 9.012, 10.811, 12.011, 14.007, 16.0, 18.998,
    20.183, 22.99, 24.312, 26.982, 28.086, 30.974, 32.064, 35.453, 39.948,
    39.102, 40.08, 44.956, 47.90, 50.942, 51.996, 54.938, 55.847, 58.933,
    58.71, 63.54, 65.37, 69.72, 72.59, 74.92, 78.96, 79.91, 83.80, 85.47, 
    87.62, 88.905, 91.22, 92.906, 95.94, 99.00, 101.07, 102.9, 106.4, 
    107.87, 112.40, 114.82, 118.69, 121.75, 127.6, 126.9, 131.3, 132.9, 
    137.34, 138.91, 140.12, 140.91, 144.24, 147.00, 150.35, 151.96, 157.25, 
    158.92, 162.50, 164.93, 167.26, 168.93, 173.04, 174.97, 178.49, 180.95, 
    183.85, 186.2, 190.2, 192.2, 195.09, 196.97, 200.59, 204.37, 207.19, 
    208.98, 210.0, 211.0, 222.0, 223.0, 226.1, 227.1, 232.04, 231.0, 238.03
])

DENO_SA = np.sum(10.0**(ABUND - 12.0))
DENO_SAM = np.sum(MAMU * MATOM * 10.0**(ABUND - 12.0))

# =====================================================================
# PARTITION FUNCTIONS
# =====================================================================
def get_partition_functions(Z, T):
    """
    Returns partition functions (U1, U2, U3) for element Z at temperature T.
    """
    U1, U2, U3 = 1.0, 1.0, 1.0
    x = np.log(5040.0 / T)
    y = 1e-3 * T
    
    if Z == 1: # H
        U1 = 1.0 # H-
        U2 = 2.0 # H
        if T > 1.3e4: U2 = 1.51 + 3.8e-5 * T
        if T > 1.62e4: U2 = 11.41 + T * (-1.1428e-3 + T * 3.52e-8)
        U3 = 1.0 # H+
    elif Z == 2: # He
        U1 = 1.0
        if T > 3e4: U1 = 14.8 + T * (-9.4103e-4 + T * 1.6095e-8)
        U2 = 2.0
        U3 = 1.0
    elif Z == 6: # C
        U1 = 8.6985 + y * (2.0485e-2 + y * (1.7629e-2 - 3.9091e-4 * y))
        if T > 1.2e4: U1 = 13.97 + T * (-1.3907e-3 + T * 9.0844e-8)
        U2 = 5.838 + 1.6833e-5 * T
        if T > 2.4e4: U2 = 10.989 + T * (-6.9347e-4 + T * 2.0861e-8)
        U3 = 1.0
        if T > 1.95e4: U3 = -0.555 + 8e-5 * T
    elif Z == 8: # O
        U1 = 8.29 + 1.10e-4 * T
        if T > 1.9e4: U1 = 66.81 + T * (-6.019e-3 + T * 1.657e-7)
        U2 = max(4.0, 3.51 + 8e-5 * T)
        if T > 3.64e4: U2 = 68.7 + T * (-4.216e-3 + T * 6.885e-8)
        U3 = 7.865 + 1.1348e-4 * T
    elif Z == 11: # Na
        U1 = max(2.0, 1.72 + 9.3e-5 * T)
        if T > 5400.0: U1 = -0.83 + 5.66e-4 * T
        if T > 8.5e3: U1 = 4.5568 + T * (-1.2415e-3 + T * 1.3861e-7)
        U2 = 1.0
        U3 = 5.69 + 5.69e-6 * T
    elif Z == 12: # Mg
        U1 = 1.0 + np.exp(-4.027262 - x * (6.173172 + x * (2.889176 + x * (2.393895 + 0.784131 * x))))
        if T > 8e3: U1 = 2.757 + T * (-7.8909e-4 + T * 7.4531e-8)
        U2 = 2.0 + np.exp(-7.721172 - x * (7.600678 + x * (1.966097 + 0.212417 * x)))
        if T > 2e4: U2 = 7.1041 + T * (-1.0817e-3 + T * 4.7841e-8)
        U3 = 1.0
    elif Z == 20: # Ca
        U1 = 1.0 + np.exp(-1.731273 - x * (5.004556 + x * (1.645456 + x * (1.326861 + 0.508553 * x))))
        U2 = 2.0 + np.exp(-1.582112 - x * (3.996089 + x * (1.890737 + 0.539672 * x)))
        U3 = 1.0
    elif Z == 26: # Fe
        U1 = 9.0 + np.exp(2.930047 + x * (-0.979745 + x * (0.76027 + 0.118218 * x)))
        if T < 4e3: U1 = 15.85 + T * (1.306e-3 + T * 2.04e-7)
        if T > 9e3: U1 = 39.149 + T * (-9.5922e-3 + T * 1.2477e-6)
        U2 = 10.0 + np.exp(3.501597 + x * (-0.612094 + 0.280982 * x))
        if T > 1.8e4: U2 = 68.356 + T * (-6.1104e-3 + T * 5.1567e-7)
        U3 = 17.336 + T * (5.5048e-4 + T * 5.7514e-8)
    else:
        # Fallback approximate constant weights for remaining trace elements to preserve structural completion
        U1, U2, U3 = 1.0, 2.0, 1.0 
        
    return U1, U2, U3

# =====================================================================
# CHEMICAL EQUILIBRIUM SOLVER
# =====================================================================
def get_rho(T, Pg):
    """
    Computes Density, Ne, and Nh from T and Pgas iteratively.
    """
    # Initial guesses
    ne_new = Pg / (101.0 * KBOL * T)
    if (ne_new * KBOL * T) > Pg:
        ne_new = (Pg * 0.5) / (KBOL * T)
    
    alpha1 = np.zeros(NELEM)
    alpha2 = np.zeros(NELEM)
    
    lamelec = CSAHA1 * (T**(-1.5))
    
    # Precompute Saha factors
    for i in range(NELEM):
        u1, u2, u3 = get_partition_functions(i + 1, T)
        expo1 = np.clip(CSAHA2 * XI[i] / T, -650.0, 650.0)
        expo2 = np.clip(CSAHA2 * XII[i] / T, -650.0, 650.0)
        alpha1[i] = (u1 / (2.0 * u2)) * lamelec * np.exp(expo1)
        alpha2[i] = (u2 / (2.0 * u3)) * lamelec * np.exp(expo2)

    error = 1.0
    iter_count = 1
    nh = 0.0

    while error > 1e-4 and iter_count <= 200:
        ne_old = ne_new
        nume = Pg - ne_new * KBOL * T
        nh = nume / (DENO_SA * KBOL * T)
        
        nt = nh * 10.0**(ABUND - 12.0)
        ne_est = 0.0
        
        # Calculate ionizations
        n_I = nt / (1.0 + (1.0 / (ne_new * alpha1)) * (1.0 + (1.0 / (ne_new * alpha2))))
        n_II = nt / (ne_new * alpha1 + 1.0 + (1.0 / (ne_new * alpha2)))
        n_III = nt / (1.0 + ne_new * alpha2 * (ne_new * alpha1 + 1.0))
        
        # H is index 0. n_I is H-, n_III is H+
        ne_est = n_III[0] - n_I[0]
        ne_est += np.sum(n_II[1:]) + np.sum(n_III[1:])
        
        ne_new = (max(ne_est, 0.0) + ne_old) / 2.0
        if ne_new > 0:
            error = abs((ne_new - ne_old) / ne_new)
        else:
            error = 0.0
        iter_count += 1
        
    nelec = ne_new
    return nelec, nh

# =====================================================================
# CONTINUUM OPACITY CONTRIBUTORS
# =====================================================================
def opac_hminus_ff(T, Ne, lambda0):
    lambda0_mic = lambda0 / 1e4
    theta = 5040.0 / T
    
    A1 = [0, 2483.346, -3449.889, 2200.04, -696.271, 88.283]
    B1 = [0, 285.827, -1158.382, 2427.719, -1841.4, 444.517]
    C1 = [0, -2054.291, 8746.523, -13651.105, 8624.97, -1863.864]
    D1 = [0, 2827.776, -11485.632, 16755.524, -10051.53, 2095.288]
    E1 = [0, -1341.537, 5303.609, -7510.494, 4400.067, -901.788]
    F1 = [0, 208.952, -812.939, 1132.738, -655.02, 132.985]
    
    A2 = [518.1021, 473.2636, -482.2089, 115.5291]
    B2 = [-734.8666, 1443.4137, -737.1616, 169.6374]
    C2 = [1021.1775, -1977.3395, 1096.8827, -245.649]
    D2 = [-479.0721, 922.3575, -521.1341, 114.243]
    E2 = [93.1373, -178.9275, 101.7963, -21.9972]
    F2 = [-6.4285, 12.36, -7.0571, 1.5097]

    if lambda0_mic < 0.3645:
        c2 = (np.array(A2) * lambda0_mic**2 + np.array(B2) + np.array(C2)/lambda0_mic +
              np.array(D2)/lambda0_mic**2 + np.array(E2)/lambda0_mic**3 + np.array(F2)/lambda0_mic**4)
        part1 = c2[0]*theta + c2[1]*theta**1.5 + c2[2]*theta**2 + c2[3]*theta**2.5
    else:
        l, l2, l_inv = lambda0_mic, lambda0_mic**2, 1.0/lambda0_mic
        c1 = (np.array(A1) * l2 + np.array(B1) + np.array(C1)*l_inv + np.array(D1)*l_inv**2 + np.array(E1)*l_inv**3 + np.array(F1)*l_inv**4)
        part1 = (c1[0]*theta + c1[1]*theta**1.5 + c1[2]*theta**2 + c1[3]*theta**2.5 +
                 c1[4]*theta**3 + c1[5]*theta**3.5)
                 
    return 1e-29 * part1 * KBOL * Ne * T


def opac_hminus_bf(T, Ne, lambda0):
    lambda_p = 1.6419
    cte = 0.75e-18
    alpha = (HPLA * LIGHT / KBOL) * 1e4
    cc = [152.519, 49.534, -118.858, 92.536, -34.194, 4.982]
    lambda0_mic = lambda0 / 1e4

    if lambda0_mic < lambda_p:
        com = 1.0/lambda0_mic - 1.0/lambda_p
        sigma = (cc[0] + cc[1]*com**0.5 + cc[2]*com + cc[3]*com**1.5 + 
                 cc[4]*com**2 + cc[5]*com**2.5)
        sigma = cte * sigma * lambda0_mic**3 * com**1.5
        part = T**(-2.5) * np.exp(alpha/(T*lambda_p)) * (1.0 - np.exp(-alpha/(T*lambda0_mic)))
        return part * sigma * Ne * KBOL * T
    return 0.0

def opac_thomson(Ne, Nh):
    ct = (8.0 * np.pi / 3.0) * ((QELE / LIGHT)**4) / (MELE**2)
    return ct * (Ne / Nh)

def opac_rayleigh_h(lambda0):
    cc = [5.799e-13, 1.422e-6, 2.784]
    return (cc[0] + (cc[1] + cc[2]/lambda0**2) / lambda0**2) / lambda0**4

def opac_hydrogen(T, lambda0):
    iopot = 13.595
    c1 = 1e-8 * iopot * EVOLT / (HPLA * LIGHT)
    c2 = (1e-8)**3 * 64.0 * np.pi**4 * MELE * QELE**10 / (3.0 * np.sqrt(3.0) * LIGHT**4 * HPLA**6)
    c3 = HPLA * LIGHT / (KBOL * 1e-8)
    c4 = iopot * EVOLT / KBOL
    
    n0 = 1 + int(np.floor(np.sqrt(c1 * lambda0)))
    sum_val = 0.0
    
    if n0 <= 8:
        sum_val = np.exp(c4 / (T * n0**2)) * (n0**-3)
        for i in range(n0, 9):
            sum_val += np.exp(c4 / (T * i**2)) * (i**-3)
        sum_val += (0.117 + np.exp(c4 / (T * 9.0**2))) * (T / (2.0 * c4))
    else:
        sum_val = (0.117 + np.exp(c4 / (T * n0**2))) * (T / (2.0 * c4))

    opac = c2 * sum_val * (1.0 - np.exp(-c3 / (T * lambda0))) * np.exp(-c4 / T) * lambda0**3
    return opac

# =====================================================================
# MAIN CONTINUUM OPACITY FUNCTION
# =====================================================================
def get_continuum_opacity(T, Pg, lambda0):
    """
    Computes continuum opacity in LTE for given atmospheric properties.
    
    Parameters:
    - T: Temperature [K]
    - Pg: Gas pressure [dyn/cm^2]
    - lambda0: Wavelength [Angstroms]
    
    Returns:
    - kc: Continuum opacity [cm^-1]
    """
    # Chemical Equilibrium Solver
    Ne, Nh = get_rho(T, Pg)
    
    # Extract Individual Opacity Components (Per Hydrogen Atom)
    o_hmin_ff = opac_hminus_ff(T, Ne, lambda0)
    o_hmin_bf = opac_hminus_bf(T, Ne, lambda0)
    o_thomson = opac_thomson(Ne, Nh)
    o_rayleigh = opac_rayleigh_h(lambda0)
    o_hyd = opac_hydrogen(T, lambda0)
    
    factor = o_hmin_ff + o_hmin_bf + o_thomson + o_rayleigh + o_hyd
    
    # Total Continuum Opacity [cm^-1]
    kc = Nh * factor
    return kc, Nh, Ne

# Example usage:
# kappa = get_continuum_opacity(5000.0, 1.0e5, 5000.0)