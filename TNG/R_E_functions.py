import numpy as np
from scipy.optimize import root_scalar

from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
from astropy.cosmology import Planck15
import astropy.units as u
from astropy import constants as const

# Important Cosmological & Simulation Parameters
h = 0.6774
G = 4.30091e-6  # Gravitational constant in kpc * (km/s)^2 / Msun
p_mass = 3.64755609660833*10**5 / h # Individual particle mass in Msun (TNG specific)

def get_sigma_crit_for_snapshot(z_L, z_S=2.0):
    """
    Calculates the Critical Surface Mass Density (Sigma_crit) for a given 
    lens and source redshift configuration.
    
    Parameters:
    -----------
    z_L : float
        The redshift of the lens (e.g., the TNG snapshot redshift).
    z_S : float, optional
        The redshift of the background light source. Default is 2.0.
        MUST be strictly greater than z_L.
        
    Returns:
    --------
    Sigma_crit_val : float
        Critical surface mass density in units of M_sun / kpc^2.
    """
    
    # Sanity check: Lensing is physically impossible if the source is in front of the lens
    if z_L >= z_S:
        raise ValueError(f"Snapshot redshift (z_L={z_L:.2f}) is >= Source redshift (z_S={z_S}). Lensing cannot occur.")
    
    # 1. Calculate angular diameter distances using TNG's native cosmology (Planck15)
    D_L = Planck15.angular_diameter_distance(z_L)       # Distance to Lens
    D_S = Planck15.angular_diameter_distance(z_S)       # Distance to Source
    D_LS = Planck15.angular_diameter_distance_z1z2(z_L, z_S) # Distance between Lens and Source
    
    # 2. Calculate Sigma_crit using the physical formula: c^2 / (4 * pi * G) * (D_S / (D_L * D_LS))
    Sigma_crit_qty = (const.c**2 / (4 * np.pi * const.G)) * (D_S / (D_L * D_LS))
    
    # 3. Convert units to M_sun / kpc^2 to match TNG simulation particle mass units
    Sigma_crit_val = Sigma_crit_qty.to(u.M_sun / u.kpc**2).value
    
    return Sigma_crit_val

def get_host_background_kappa(host_coords, host_center, subhalo_center, Sigma_crit):
    """
    Calculates the local background convergence (kappa_bg) contributed by the 
    host cluster at the 2D projected position of a subhalo.
    
    Parameters:
    -----------
    host_coords : numpy.ndarray
        Nx3 array containing the 3D coordinates of all particles in the host cluster.
    host_center : numpy.ndarray
        1x3 array of the host cluster's center coordinates (x, y, z).
    subhalo_center : numpy.ndarray
        1x3 array of the subhalo's center coordinates (x, y, z).
    Sigma_crit : float
        The critical surface mass density for the system (M_sun / kpc^2).
        
    Returns:
    --------
    kappa_bg : float
        The dimensionless local background convergence at the subhalo's position.
    """
    
    # 1. Calculate the 2D projected distance between the host center and subhalo center
    # Assuming projection along the z-axis (taking only x and y coordinates)
    rel_pos = subhalo_center[:2] - host_center[:2]
    R_sub_proj = np.linalg.norm(rel_pos)
    
    # 2. Get 2D distances of all host particles from the host center
    host_rel_coords = host_coords - host_center
    R_host_2d = np.linalg.norm(host_rel_coords[:, :2], axis=1)
    
    # 3. Define a narrow annulus (ring) around the subhalo's distance
    # Using a +/- 10 kpc window to get a smooth average of the local density
    dR = 10.0 
    mask = (R_host_2d > (R_sub_proj - dR)) & (R_host_2d < (R_sub_proj + dR))
    
    # 4. Calculate the total mass inside this annulus
    M_annulus = np.sum(mask) * p_mass
    
    # 5. Calculate the geometric area of the 2D annulus
    Area = np.pi * ((R_sub_proj + dR)**2 - (R_sub_proj - dR)**2)
    
    # 6. Calculate local surface density (Sigma) and normalize by Sigma_crit to get kappa
    Sigma_host_local = M_annulus / Area
    kappa_bg = Sigma_host_local / Sigma_crit
    
    return kappa_bg

def convert_rmax_vmax_to_nfw(R_max_kpc, V_max_kms):
    """
    Converts observational subhalo parameters (R_max and V_max) into 
    structural NFW profile parameters (scale radius r_s and characteristic density rho_s).
    
    Parameters:
    -----------
    R_max_kpc : float
        Radius at which the subhalo profile reaches maximum circular velocity (kpc).
    V_max_kms : float
        The maximum circular velocity of the subhalo (km/s).
        
    Returns:
    --------
    r_s : float
        NFW profile scale radius in kpc.
    rho_s_val : float
        NFW characteristic density in M_sun / kpc^3.
    """
    
    # 1. Calculate scale radius (r_s) based on the structural properties of an NFW profile
    # For a pure NFW profile, R_max occurs at roughly 2.163 * r_s
    r_s = R_max_kpc / 2.16258
    
    # 2. Assign astropy units to handle G conversion and dimensional analysis safely
    V_max_qty = V_max_kms * u.km / u.s
    r_s_qty = r_s * u.kpc
    
    # 3. Calculate characteristic density (rho_s) using the analytical NFW V_max equation
    rho_s_qty = (V_max_qty**2) / (2.7170 * const.G * r_s_qty**2)
    
    # 4. Convert to standard TNG simulation units (M_sun / kpc^3)
    rho_s_val = rho_s_qty.to(u.M_sun / u.kpc**3).value
    
    return r_s, rho_s_val

def nfw_kappa_bar(x, kappa_s):
    """
    Calculates the analytical average enclosed convergence (mean surface density 
    divided by critical density) inside a dimensionless radius x for an NFW profile.
    
    Parameters:
    -----------
    x : float
        Dimensionless radius scaled by the scale radius (x = R / r_s).
    kappa_s : float
        Dimensionless characteristic convergence scale (rho_s * r_s / Sigma_crit).
        
    Returns:
    --------
    kappa_bar : float
        The average enclosed convergence inside radius x.
    """
    # Handle the piecewise analytical solutions of the projected NFW profile
    if x < 1.0:
        # Inner profile solution (sub-scale radius)
        term1 = (2.0 / np.sqrt(1.0 - x**2)) * np.arctanh(np.sqrt((1.0 - x) / (1.0 + x)))
        return (4.0 * kappa_s / x**2) * (term1 + np.log(x / 2.0))
    elif x > 1.0:
        # Outer profile solution (super-scale radius)
        term1 = (2.0 / np.sqrt(x**2 - 1.0)) * np.arctan(np.sqrt((x - 1.0) / (1.0 + x)))
        return (4.0 * kappa_s / x**2) * (term1 + np.log(x / 2.0))
    else: 
        # Boundary condition exactly at the scale radius (x == 1.0)
        return 4.0 * kappa_s * (1.0 + np.log(0.5))

def get_analytical_einstein_radius(r_s, rho_s, Sigma_crit, kappa_bg=0.0):
    """
    Finds the Einstein radius for an NFW profile superposed on a smooth background 
    using numerical 1D root-finding.
    
    Parameters:
    -----------
    r_s : float
        NFW scale radius in kpc.
    rho_s : float
        NFW characteristic density in M_sun / kpc^3.
    Sigma_crit : float
        Critical surface mass density (M_sun / kpc^2).
    kappa_bg : float, optional
        Local background convergence from the host environment. Default is 0.0.
        
    Returns:
    --------
    R_E : float
        The physical Einstein radius in kpc. Returns 0.0 if the profile is subcritical.
    optimal_kappa_bar : float
        The targeted average enclosed convergence value at the Einstein radius.
    """
    # Calculate the scale convergence factor
    kappa_s = (rho_s * r_s) / Sigma_crit
    
    # Define the objective function for root finding: kappa_total_enclosed(x) - 1 = 0
    def objective_function(x):
        return nfw_kappa_bar(x, kappa_s) + kappa_bg - 1.0
    
    # First, check if the extreme center is supercritical.
    # NFW kappa diverges at x -> 0, but if it's already negative at 1e-4, no lensing can occur.
    if objective_function(1e-4) < 0:
        return 0.0, 0.0  # Entire profile is subcritical (not dense enough to lens)
        
    # Use Brent's method to find where the objective function crosses zero.
    # We look for a root bounded between x = 1e-4 and x = 100.0 scale radii.
    try:
        x_E = root_scalar(objective_function, bracket=[1e-4, 100.0], method='brentq').root
        R_E = x_E * r_s  # Convert dimensionless root back to physical kpc
        optimal_kappa_bar = 1.0 - kappa_bg
        return R_E, optimal_kappa_bar
    except ValueError:
        # If no root is found within the physical bounds, it's not an effective lens
        return 0.0, 0.0

def X(fb, cs, param):
    """
    Calculates the scaling factor for structural parameters under tidal stripping 
    according to the Green and van den Bosch (2019) model.
    
    Parameters:
    -----------
    fb : float
        The bound mass fraction remaining (M_current / M_infall).
    cs : float
        The NFW concentration parameter evaluated at the time of infall.
    param : str
        Determines which scaling factor to return: 'v' for V_max or 'r' for R_max.
        
    Returns:
    --------
    X_factor : float
        The dimensionless evolutionary scaling parameter for R_max or V_max.
    """
    mu, nu = gb_mu_nu(fb, cs, param)
    return (2**mu) * (fb**nu) / ((1 + fb)**mu)

def gb_mu_nu(fb, cs, param):
    """
    Calculates the fitting parameters mu and nu used in the Green & van den Bosch (2019) 
    tidal stripping model. These parameters are functions of bound mass fraction and concentration.
    
    Parameters:
    -----------
    fb : float
        The bound mass fraction (M_current / M_infall).
    cs : float
        The NFW halo concentration at infall.
    param : str
        Tracking flag: 'v' for V_max parameters, 'r' for R_max parameters.
        
    Returns:
    --------
    mu : float
        Fitting exponent function mu.
    nu : float
        Fitting exponent function nu.
    """
    # Load empirical fitting constants depending on the parameter of interest
    if param == 'v':
        p0, p1, p2, p3, p4 = 2.980, 0.310, -0.223, -3.308, -0.079
        q0, q1, q2 = 0.176, -0.008, 0.452
    elif param == 'r':
        p0, p1, p2, p3, p4 = 1.021, 1.463, 0.099, -4.643, -0.250
        q0, q1, q2 = -0.525, -0.065, 0.083
    else:
        raise ValueError("Param must be either 'v' or 'r'.")
    
    # Compute the analytical functions mapping structural evolution
    mu = p0 + p1 * (cs**p2) * np.log10(fb) + p3 * (cs**p4)
    nu = q0 + q1 * (cs**q2) * np.log10(fb)
    return mu, nu

def get_GVB_R_E(fb, norm_rmax, norm_vmax, inf_c, Sigma_crit, kappa_bg):
    """
    Calculates the predicted Einstein radius and average convergence of a subhalo 
    after correcting its R_max and V_max profiles for tidal stripping using 
    the Green and van den Bosch (2019) model.
    
    Parameters:
    -----------
    fb : float
        The bound mass fraction (M_current / M_infall).
    norm_rmax : float
        The baseline or infall reference R_max value (kpc).
    norm_vmax : float
        The baseline or infall reference V_max value (km/s).
    inf_c : float
        The halo concentration parameter at infall. 
        (Note: Removed the explicit array indexing [t] to ensure standalone function safety).
    Sigma_crit : float
        Critical surface mass density (M_sun / kpc^2).
    kappa_bg : float
        Local background convergence from the host environment.
        
    Returns:
    --------
    R_E_analytical : float
        The calculated physical Einstein radius (kpc) after tidal corrections.
    kappa_analytical : float
        The target average enclosed convergence at the modified Einstein radius.
    """
    # 1. Get the expected normalized evolutionary fractions from the GVB model
    r_GVB = X(fb, inf_c, 'r')
    v_GVB = X(fb, inf_c, 'v')
    
    # 2. Scale up the normalized fractions using the baseline infall properties
    r = r_GVB * norm_rmax
    v = v_GVB * norm_vmax

    # 3. Convert the modified R_max and V_max values back into NFW parameters
    r_s, rho_s = convert_rmax_vmax_to_nfw(r, v)

    # 4. Numerically solve for the new corrected Einstein Radius
    R_E_analytical, kappa_analytical = get_analytical_einstein_radius(
        r_s, rho_s, Sigma_crit, kappa_bg=kappa_bg
    )
    
    return R_E_analytical, kappa_analytical