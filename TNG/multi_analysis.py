import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.cm import ScalarMappable
import seaborn as sns
import random
import pickle
from collections.abc import Iterable
# from tqdm.notebook import tqdm
import pickle

from scipy import stats
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy.optimize import root_scalar, curve_fit
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from boruta import BorutaPy

from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
from astropy.cosmology import Planck15
import astropy.units as u
from astropy import constants as const

print('--Completed Imports--')

"""Data Configuration"""

#important parameters
h = 0.6774
G = 4.30091e-6  # kpc (km/s)^2 / Msun
p_mass = 3.64755609660833*10**5 / h #Msun

# referring to Green and van den Bosch 2019 as GVB
# referring to Penarrubia et al. 2010 as PN

#fit parameters depending on gamma [mu r, eta r, mu v, eta v] for PN model
g_fit = {1.5:[0,.48,0.4,0.24],
                1.0: [-0.3,0.4,0.4,0.3],
                0.5: [-0.4,0.27,0.4,0.35],
                0.0: [-1.3,0.05,0.4,0.37]}

#reported relative error for gvb
gvb_error = {'rmax':0.03,'vmax':0.01}

# load in data files
print('--Loading Data--')

# load subhalo data
with open('data/valid_subhalos_multihost.pkl', 'rb') as f:
    final_analysis_store = pickle.load(f)

# load snapshot conversion data
with open('data/snap_conversions.pkl', 'rb') as f:
    snap_conversions = pickle.load(f)

snap_to_a = snap_conversions['a']
snap_to_red = snap_conversions['red']
snap_to_time = snap_conversions['time']

# load structural feature data
tidal_features = pd.read_csv('data/tidal_features_multihost.csv')

print('--Data Loaded--')

"""helpful functions"""

#convert a distance value or list of values from ckpc/h to kpc
def to_physical(dist_val,snap_num):
    """
    dist_val: value to convert to physical units (in ckpc/h)
    snap_num: snapshot number of dist_val
    return: dist_val converted to physical units (kpc)
    """   
    a = snap_to_a[snap_num]
    return np.array(dist_val)*a/h

#convert snapshot number into id to index that snapshot in subhalo's data arrays
def snap_to_id(snap_num,tree):
    """
    snap_num: snapshot to convert to index
    tree: subhalo tree to get index for
    return: id of snapshot in subhalo arrays
    """
    return np.where(tree['SnapNum']==snap_num)[0][0]

# convert mass into particle count
def mass_to_count(mass,mass_unit='1e10Msol/h'):
    """
    Uses global p_mass variable
    mass: mass value to convert
    mass_unit: units of mass variable, either 1e10Msol/h or Msol
    """
    if mass_unit == '1e10Msol/h':
        return mass*1e10/h/p_mass
    elif mass_unit == 'Msol':
        return mass/p_mass


"""Theoretical Lines"""

# power law tidal function from pn
def g(x,mu,nu):
    """
    x: current bound mass fraction
    mu and nu: fit parameters (stored in global g_fit variable)
    """
    return (2**mu)*(x**nu)/((1+x)**mu)

#model from Green and van den Bosch 2019
def X(fb,cs,param):
    """
    #fb:  the bound mass fraction
    #cs: the concentration at infall
    #mu and nu:  fit parameters and functions of fb and cs
    param: either 'v' or 'r' to indicate vmax or rmax
    """
    mu,nu = gb_mu_nu(fb,cs,param)
    return (2**mu)*(fb**nu)/((1+fb)**mu)

#function for mu and nu
def gb_mu_nu(fb,cs,param):
    """
    fb: bound mass (current mass / infall mass)
    cs: current nfw concentration
    param: either 'v' for vmax or 'r' for rmax
    """
    if param == 'v':
        p0 = 2.980
        p1 = 0.310
        p2 = -0.223
        p3 = -3.308
        p4 = -0.079
        q0 = 0.176
        q1 = -0.008
        q2 = 0.452
    elif param == 'r':
        p0 = 1.021
        p1 = 1.463
        p2 = 0.099
        p3 = -4.643
        p4 = -0.250
        q0 = -0.525
        q1 = -0.065
        q2 = 0.083
    
    mu = p0 + p1*(cs**p2)*np.log10(fb)+p3*(cs**p4)
    nu = q0 + q1*(cs**q2)*np.log10(fb)
    return mu,nu

# function to retrieve data from both PN and GVB models from infall down to a specified minimum bound mass
def get_theoretical_points(m_ret_min,cs):
    """
    m_ret_min: minimum retained mass (equivalent to bound mass fraction)
    cs: nfw concentration for GVB model
    """
    # make fractional bound mass values from m_ret_min/100 up to 1
    if m_ret_min == 0:
        bound_mass_frac = np.logspace(-2,0,20)
    else:
        bound_mass_frac = np.logspace(np.log10(m_ret_min/100), 0, 20)
    # print(bound_mass_frac)

    # Generate penarrubia points for all their listed gamma
    pn_rmax_all = []
    pn_vmax_all = []
    #theoretical model from Peñarrubia et. al. 2010
    
    for gamma in g_fit:
        rmax = [g(x,g_fit[gamma][0],g_fit[gamma][1]) for x in bound_mass_frac]
        pn_rmax_all.append(rmax)
        vmax = [g(x,g_fit[gamma][2],g_fit[gamma][3]) for x in bound_mass_frac]
        pn_vmax_all.append(vmax)
    #convert mnorm to my mass loss convention
    th_mloss = [(1-m)*100 for m in bound_mass_frac]

    # retained mass for theoretical models (for coloring)
    th_mretained = np.array([100 - loss for loss in th_mloss])

    gvb_th_vmax = [X(fb,cs,'v') for fb in bound_mass_frac]
    gvb_th_rmax = [X(fb,cs,'r') for fb in bound_mass_frac]

    return gvb_th_rmax, gvb_th_vmax, pn_rmax_all, pn_vmax_all, th_mretained

"""Analysis Functions"""

# creates scatter plot of tidal track colored by retained mass for all subhalos
def tidal_scatter(final_analysis_store,N_min=3000,m_ret_min=0,med_line=True,filename=None):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl
    N_min: minimum particle count to consider
    m_ret_min: minimum retained mass to consider
    med_line: if True, will calculate and display the median line, binned by rmax
    filename: output file name, will use a default value if filename=None
    """
    norm = mcolors.LogNorm(vmin=1, vmax=100)  # can't be zero
    cmap = plt.cm.viridis

    # 1. Initialize master lists to collect all data across all trees
    all_rmax = []
    all_vmax = []
    m_retained = []

    # loop over trees
    for host_id, data in final_analysis_store.items():
        sat_trees = data['satellite_trees']
        for idx, tree in enumerate(sat_trees):
            infall = data['infall_metrics']
            inf_snap = infall['snap'][idx]
            m0 = infall['mass'][idx] # already in msol
            for i, snap in enumerate(tree['SnapNum']):
                m = tree['SubhaloMass'][i]*1e10/h
                if  m / p_mass >= N_min and m/m0 >= m_ret_min:
                    # find normalized rmax and vmax
                    r0 = infall['rmax'][idx]
                    v0 = infall['vmax'][idx]
                    
                    r = to_physical(tree['SubhaloVmaxRad'][i], snap) / r0
                    v = tree['SubhaloVmax'][i] / v0

                    all_rmax.append(r)
                    all_vmax.append(v)
                    m_retained.append(100*m/m0) # convert to percentage

                if snap <= inf_snap:
                    break

    sc = plt.scatter(all_rmax, all_vmax, c=m_retained, cmap=cmap, norm=norm,s=2)
    
    # median line
    if med_line:
        # Convert to numpy arrays and filter out any invalid/zero values for log-math
        all_rmax = np.array(all_rmax)
        all_vmax = np.array(all_vmax)
        valid_mask = np.isfinite(all_rmax) & np.isfinite(all_vmax) & (all_rmax > 0)
        all_rmax = all_rmax[valid_mask]
        all_vmax = all_vmax[valid_mask]

        # 2. Create log-spaced bins for Rmax/Rmax0 
        # (You can increase or decrease '20' to adjust the smoothness of the line)
        bins = np.logspace(np.log10(all_rmax.min()), np.log10(all_rmax.max()), 20)
        bin_centers = np.sqrt(bins[:-1] * bins[1:]) # Geometric centers for log scale

        # 3. Group the Vmax data into the Rmax bins and calculate statistics
        indices = np.digitize(all_rmax, bins)
        med_vmax = []
        p16_vmax = [] # 16th percentile (approx -1 sigma)
        p84_vmax = [] # 84th percentile (approx +1 sigma)
        valid_centers = []

        for i in range(1, len(bins)):
            in_bin = all_vmax[indices == i]
            if len(in_bin) > 5:  # Only plot a bin if it has a reliable number of points
                med_vmax.append(np.median(in_bin))
                p16_vmax.append(np.percentile(in_bin, 16)) 
                p84_vmax.append(np.percentile(in_bin, 84)) 
                valid_centers.append(bin_centers[i-1])

        # 4. Plot the median line and the shaded error band
        plt.plot(valid_centers, med_vmax, color='blue', linewidth=2, label='Median Line')

    # get theoretical points
    cs = 5
    gvb_th_rmax, gvb_th_vmax, pn_rmax_all, pn_vmax_all, th_mretained = get_theoretical_points(m_ret_min=m_ret_min,cs=cs)

    #plot the green and van den bosch prediction line for cs = 23.1
    plt.scatter(gvb_th_rmax,gvb_th_vmax,c=th_mretained,cmap=cmap, norm=norm,\
                marker='o',edgecolors='black',s=200,linewidths=0.5,label=rf'GVB ($c_s$={cs})')

    # plot error bars
    # Compute absolute errors based on percentages for green and van den bosch
    vmax_err = gvb_error['vmax'] * np.array(gvb_th_vmax)  # 1% error on vmax
    rmax_err = gvb_error['rmax'] * np.array(gvb_th_rmax)  # 3% error on rmax

    # plot the penarrubia prediction lines for gamma = 1, 1.5
    colors = ['red','orange','purple','black']
    for i, gamma in enumerate(g_fit):
        if gamma == 1 or gamma == 1.5:
            plt.plot(pn_rmax_all[i], pn_vmax_all[i], linestyle="--", label=f'PN gamma={gamma}',color=colors[i])

    # Plot with error bars
    plt.errorbar(
        gvb_th_rmax, gvb_th_vmax,
        xerr=rmax_err, yerr=vmax_err,
        fmt='none', markersize=10, color='none', ecolor='black', capsize=3,
        markeredgecolor='black'
        )

    plt.xlabel(r"$R_{max}/R_{max,0}$")
    plt.xscale('log')
    plt.yscale('log')
    cbar = plt.colorbar(sc)
    cbar.set_label("Mass Retained [%]")
    plt.ylabel(r"$V_{max}/V_{max,0}$")
    plt.legend(fontsize='small')
    # plt.title("Tidal Track of TNG subhalos and Emperical Models")
    if filename is None:
        plt.savefig(f"tidal_N{N_min}_m{m_ret_min}.png", bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')
    plt.clf()

# creates band plot for tidal track data (median line with 1 sigma band)
def tidal_band(final_analysis_store,N_min=3000,m_ret_min=0,filename=None):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl
    N_min: minimum particle count to consider
    m_ret_min: minimum retained mass to consider
    filename: output file name, will use a default value if filename=None
    """
    # tidal track band plot
    norm = mcolors.LogNorm(vmin=1, vmax=100)  # can't be zero
    cmap = plt.cm.viridis

    bound_mass_frac = np.logspace(0,2,20)/100
    th_mloss = [(1-m)*100 for m in bound_mass_frac]
    cs = 5
    gvb_th_vmax = [X(fb,cs,'v') for fb in bound_mass_frac]
    gvb_th_rmax = [X(fb,cs,'r') for fb in bound_mass_frac]

    # 1. Initialize master lists to collect all data across all trees
    all_rmax = []
    all_vmax = []

    # loop over trees
    for host_id, data in final_analysis_store.items():
        sat_trees = data['satellite_trees']
        for idx, tree in enumerate(sat_trees):
            for i, snap in enumerate(tree['SnapNum']):
                infall = data['infall_metrics']
                inf_snap = infall['snap'][idx]
                # find normalized rmax and vmax
                r0 = infall['rmax'][idx]
                v0 = infall['vmax'][idx]
                
                r = to_physical(tree['SubhaloVmaxRad'][i], snap) / r0
                v = tree['SubhaloVmax'][i] / v0

                all_rmax.append(r)
                all_vmax.append(v)

                if snap <= inf_snap:
                    break

    # Convert to numpy arrays and filter out any invalid/zero values for log-math
    all_rmax = np.array(all_rmax)
    all_vmax = np.array(all_vmax)
    valid_mask = np.isfinite(all_rmax) & np.isfinite(all_vmax) & (all_rmax > 0)
    all_rmax = all_rmax[valid_mask]
    all_vmax = all_vmax[valid_mask]

    # 2. Create log-spaced bins for Rmax/Rmax0 
    # (You can increase or decrease '20' to adjust the smoothness of the line)
    bins = np.logspace(np.log10(all_rmax.min()), np.log10(all_rmax.max()), 20)
    bin_centers = np.sqrt(bins[:-1] * bins[1:]) # Geometric centers for log scale

    # 3. Group the Vmax data into the Rmax bins and calculate statistics
    indices = np.digitize(all_rmax, bins)
    med_vmax = []
    p16_vmax = [] # 16th percentile (approx -1 sigma)
    p84_vmax = [] # 84th percentile (approx +1 sigma)
    valid_centers = []

    for i in range(1, len(bins)):
        in_bin = all_vmax[indices == i]
        if len(in_bin) > 5:  # Only plot a bin if it has a reliable number of points
            med_vmax.append(np.median(in_bin))
            p16_vmax.append(np.percentile(in_bin, 16)) 
            p84_vmax.append(np.percentile(in_bin, 84)) 
            valid_centers.append(bin_centers[i-1])

    # 4. Plot the median line and the shaded error band
    plt.plot(valid_centers, med_vmax, color='blue', linewidth=2, label='Simulation Median')
    plt.fill_between(valid_centers, p16_vmax, p84_vmax, color='blue', alpha=0.3, label='1-$\sigma$ Spread')

    # theoretical models
    gvb_th_rmax, gvb_th_vmax, pn_rmax_all, pn_vmax_all, th_mretained = get_theoretical_points(m_ret_min=m_ret_min,cs=cs)

    # plot the penarrubia prediction lines for gamma = 1, 1.5
    colors = ['red','orange','purple','black']
    for i, gamma in enumerate(g_fit):
        if gamma == 1 or gamma == 1.5:
            plt.plot(pn_rmax_all[i], pn_vmax_all[i], linestyle="--", label=rf'PN $\gamma$={gamma}',color=colors[i])

    # plot the green and van den bosch prediction line for cs = 23.1
    # Assign this scatter to 'sc' to act as the mappable for the colorbar below
    sc = plt.scatter(gvb_th_rmax, gvb_th_vmax, c=th_mretained, cmap=cmap, norm=norm,
                marker='o', edgecolors='black', s=200, linewidths=0.5, label=rf'GVB ($c_s$={cs})')
    
    # Compute absolute errors based on percentages for green and van den bosch
    vmax_err = gvb_error['vmax'] * np.array(gvb_th_vmax)  # 1% error on vmax
    rmax_err = gvb_error['rmax'] * np.array(gvb_th_rmax)  # 3% error on rmax

    # Add error bars to GVB
    plt.errorbar(
        gvb_th_rmax, gvb_th_vmax,
        xerr=rmax_err, yerr=vmax_err,
        fmt='none', markersize=10, color='none', ecolor='black', capsize=3,
        markeredgecolor='black'
        )

    plt.xlabel("Rmax/Rmax0")
    plt.xscale('log')
    plt.yscale('log')

    # Colorbar now relies on the Green/Van den Bosch scatter points
    cbar = plt.colorbar(sc)
    cbar.set_label("Mass Retained [%]")

    plt.ylabel("Vmax/Vmax0")
    plt.legend()
    if filename is None:
        plt.savefig(f"tidal_band_N{N_min}_m{m_ret_min}.png", bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')
    plt.clf()

# calculates chi squared, reduced chi squared, error in reduced chi squared, and p value
# optionally plots distribution of residuals
def chi_squared(final_analysis_store,tidal_features,N_min=3000,m_ret_min=0,m_ret_max=100,m_rel_max=np.inf,plot_residuals=False,filename=None):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl)
    tidal_features: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
    N_min: minimum particle count to consider
    m_ret_min: minimum retained mass to consider
    m_ret_max: maximum retained mass to consider
    m_rel_max: maximum subhalo mass / host mass to consider (for dynamical friction issues)
    plot_residuals: if True, generates histogram of residuals compared to normal distribution from GVB errors
    filename: output file name, will use a default value if filename=None (in case of plot_residuals=True)
    """
    # loop over trees and create global list of rmax, vmax, and retained mass
    rmax = []
    vmax = []
    m_retained = []
    concentrations = []
    concentration_lookup = tidal_features.set_index(['subhalo_id','snapshot'])['concentration_inf'].to_dict()

    for host_id, data in final_analysis_store.items():
        trees = data['satellite_trees']
        for idx, tree in enumerate(trees):
            infall = data['infall_metrics']
            inf_snap = infall['snap'][idx]
            mass_inf = infall['mass'][idx]
            vmax_inf = infall['vmax'][idx]
            rmax_inf = infall['rmax'][idx]
            c_inf = concentration_lookup.get((tree['SubhaloID'][0],99),None)

            for i,snap in enumerate(tree['SnapNum']):
                #apply N = 3000 resolution minimum
                if (tree['SubhaloMass'][i]*1e10/h)/p_mass >= N_min:
                    m = tree["SubhaloMass"][i]*1e10 / h # in 10^10 Msol/h so convert
                    cent_id = snap_to_id(snap,data['central_tree'])
                    m_host = data['central_tree']["SubhaloMass"][cent_id]*1e10 / h # in 10^10 Msol
                    # apply cutoff on retained mass percent
                    m_ret = 100 * (tree['SubhaloMass'][i]*1e10/h / mass_inf)
                    
                    # cut at minimum retained mass, also ignore the initial point because it is always 1,1
                    if m_ret_max > m_ret >= m_ret_min and m/m_host < m_rel_max:
                        # find normalized rmax and vmax, note rmax_inf is already in kpc but we need to convert the current rmax
                        r = to_physical(tree['SubhaloVmaxRad'][i],snap) / rmax_inf
                        v = tree['SubhaloVmax'][i] / vmax_inf

                        rmax.append(r)
                        vmax.append(v)          
                        m_retained.append(m_ret)
                        concentrations.append(c_inf)

                    if snap <= inf_snap:
                        break


    both_chi2 = []
    both_red_chi2 = []
    both_p_value = []
    for key in ['r','v']:
        
        # 1. Calculate the predicted values from your theoretical model
        bound_mass = np.array(m_retained) / 100
        expected_values = [X(fb,cs,key) for fb,cs in zip(bound_mass,concentrations)]
        
        # Convert lists to numpy arrays for element-wise operations
        if key == 'r':
            obs_data = np.array(rmax)
            # If the error is a single value, this ensures it works mathematically with arrays
            error = np.array(gvb_error['rmax'])*expected_values
        if key == 'v':
            obs_data = np.array(vmax)
            error = np.array(gvb_error['vmax'])*expected_values

        # 2. Calculate the Chi-Square statistic
        # Formula: Sum of ((Observed - Expected) / Error)^2

        chi2 = np.sum(((obs_data - expected_values) / error)**2)
        both_chi2.append(chi2)
        
        # 3. Calculate degrees of freedom (N - k)
        # N is number of data points, k is number of fitted parameters
        dof = len(obs_data)
        if dof == 0:
            return -1,-1,-1,-1,-1
        
        # 4. Calculate the reduced Chi-Square
        reduced_chi2 = chi2 / dof
        both_red_chi2.append(reduced_chi2)
        sigma_red_chi2 = np.sqrt(2/dof)

        log_p_value = stats.chi2.logsf(chi2, dof)    
        log10_p_value = log_p_value / np.log(10)
        both_p_value.append(log10_p_value)

        if plot_residuals:
            # Calculate normalized residuals
            normalized_residuals = (obs_data - expected_values) / error

            # Plot histogram
            plt.hist(normalized_residuals, bins=50, density=True, alpha=0.6, color='b', label='TNG Data')

            # Overplot a standard normal curve for comparison
            x = np.linspace(-5, 5, 100)
            plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, label="Standard Normal")
            plt.xlim(-40,40)
            plt.xlabel("Normalized Residual")
            plt.ylabel("Fraction of Points")
            plt.yscale("log")
            plt.ylim(1e-3,1)
            # if key == 'r':
                # plt.title("Normalized Rmax Residuals")
            # if key == 'v':
                # plt.title("Normalized Vmax Residuals")
            plt.legend()
            if filename is None:
                plt.savefig(f'residuals_{key}.png',bbox_inches='tight') 
            else:
                plt.savefig(filename,bbox_inches='tight')
            plt.clf()
    
    return both_chi2, both_red_chi2, sigma_red_chi2, both_p_value, len(rmax)

# calculates median of rmax over epsilon given a minimum particle count 
def median_rmax_over_epsilon(final_analysis_store,N_min,m_ret_min=0,m_ret_max=100):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl)
    N_min: minimum particle count to consider
    m_ret_min: minimum retained mass to consider
    m_rel_max: maximum subhalo mass / host mass to consider (for dynamical friction issues)
    """
    # comoving softening length (valid until z=1)
    co_softening_length = 0.575*h # ckpc/h

    #physical softening length set after z = 1
    phys_softening_length = 0.288 # kpc

    abs_rmax = []
    softening_lengths = []
    particle_counts = []

    for host_id, data in final_analysis_store.items():
        trees = data['satellite_trees']
        for idx, tree in enumerate(trees):
            infall = data['infall_metrics']
            inf_snap = infall['snap'][idx]
            mass_inf = infall['mass'][idx]
            rmax_inf = infall['rmax'][idx]
            for i,snap in enumerate(tree['SnapNum']):
                #apply N = 3000 resolution minimum
                if mass_to_count(tree['SubhaloMass'][i]) >= N_min:
                    # apply cutoff on retained mass
                    m_ret = 100 * (tree['SubhaloMass'][i] / mass_inf)
                    
                    # cut at minimum retained mass, also ignore the initial point because it is always 1,1
                    if m_ret_max > m_ret >= m_ret_min:
                        # find normalized rmax and vmax, note rmax_inf is already in kpc but we need to convert the current rmax
                        r = to_physical(tree['SubhaloVmaxRad'][i],snap) / rmax_inf
                        # v = tree['SubhaloVmax'][i] / vmax_inf

                        # rmax.append(r)
                        abs_rmax.append(r*rmax_inf)
                        # vmax.append(v)          
                        particle_counts.append(mass_to_count(tree['SubhaloMass'][i]))
                        
                        # z = 1 occurs at snapshot 50
                        if snap >= 50:
                            softening_lengths.append(phys_softening_length)
                        else:
                            softening_lengths.append(co_softening_length)

                    if snap <= inf_snap:
                        break
        
    r_over_epsilon = np.array(abs_rmax) / np.array(softening_lengths)
    return np.median(r_over_epsilon)

# plots percent error as a function of the minimum particle count
def p_err_N_min(final_analysis_store,tidal_features,m_ret_min=0,m_ret_max=100,m_rel_max=np.inf,filename=None):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl)
    tidal_features: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
    m_ret_min: minimum retained mass to consider
    m_ret_max: maximum retained mass to consider
    m_rel_max: maximum subhalo mass / host mass to consider (for dynamical friction issues)
    filename: output file name, will use a default value if filename=None
    """
    norm = mcolors.LogNorm(vmin=1, vmax=100)  # can't be zero
    cmap = plt.cm.viridis

    #range of N values
    N_range = np.logspace(3,7,20)

    red_chi_r = []
    red_chi_v = []
    sigma = []
    valid_N = []
    point_counts = []
    all_r_over_epsilon = []

    for n in N_range:
        _, red, sig, both_p, point_count = chi_squared(final_analysis_store,tidal_features,
                                                       N_min=n,m_ret_min=m_ret_min,m_ret_max=m_ret_max,
                                                       m_rel_max=m_rel_max)
        if red == -1:
            continue
        all_r_over_epsilon.append(median_rmax_over_epsilon(final_analysis_store,N_min=n,m_ret_max=m_ret_max))
        red_chi_r.append(red[0])
        red_chi_v.append(red[1])
        sigma.append(sig)
        valid_N.append(n)
        point_counts.append(point_count)
        # print(both_p)

    z_score_r = np.sqrt(red_chi_r)
    z_score_v = np.sqrt(red_chi_v)

    percent_r = z_score_r*gvb_error['rmax']*100 #multiply by 100 to make it a percent
    percent_v = z_score_v*gvb_error['vmax']*100

    z_err_r = 0.5*(1/np.sqrt(red_chi_r))*(np.sqrt(2)/np.sqrt(point_counts))
    z_err_v = 0.5*(1/np.sqrt(red_chi_v))*(np.sqrt(2)/np.sqrt(point_counts))

    percent_err_r = z_err_r*gvb_error['rmax']
    percent_err_v = z_err_v*gvb_error['vmax']

    #create plotting axes
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot with error bars
    ax1.errorbar(
        valid_N, percent_r,
        xerr=0, yerr=percent_err_r,
        fmt='None', markersize=10, ecolor='red', capsize=3,
        markeredgecolor='black', label=f'Rmax', zorder=2
        )
    # Plot over the error bars colored points with the rmax over epsilon as the color
    sc = ax1.scatter(valid_N, percent_r, edgecolors='black', c=all_r_over_epsilon, cmap=cmap, norm=norm, zorder=1, s=100)

    # Same as above but for vmax residuals
    ax1.errorbar(
        valid_N, percent_v,
        xerr=0, yerr=percent_err_v,
        fmt='None', markersize=10, ecolor='blue', capsize=3,
        markeredgecolor='black', label=f'Vmax', zorder=2
        )
    ax1.scatter(valid_N, percent_v, edgecolors='black', c=all_r_over_epsilon, cmap=cmap, norm=norm, zorder=1, s=100)
    
    ### configure the plot
    
    ax1.axvline(1048576,linestyle='--',color='black',label='DASH particle count')

    ax1.set_xlabel("Minimum Particle Count")
    ax1.set_ylabel("% Error")
    ax1.set_xscale("log")
    ax1.legend()
    
    #secondary axis for number of points
    ax2 = ax1.twinx()
    ax2.plot(valid_N,point_counts,linestyle=':',color='gray')
    ax2.set_ylabel("Number of Points",color='gray')
    ax2.set_yscale("log")
    ax2.tick_params(axis='y', labelcolor='gray')

    #colorbar
    cbar = fig.colorbar(sc, ax=ax2, pad=0.15)
    cbar.set_label(r"Median $R_{max}/\epsilon$")

    # plt.title("Percent Error Function of Resolution")
    fig.tight_layout()
    if filename is None:
        plt.savefig(f"percent_N_min_color_rel_max={m_rel_max}.png",bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')
    plt.clf()

# plots the same as p_err_N_min however with one host at a time
def p_err_N_min_per_host(final_analysis_store, tidal_features, m_ret_min=0, m_ret_max=100):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl)
    tidal_features: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
    m_ret_min: minimum retained mass to consider
    m_ret_max: maximum retained mass to consider
    plot_residuals: if True, generates histogram of residuals compared to normal distribution from GVB errors
    no filename argument because many plots are generated, the default format can be modified at the bottom of the function
    """
    norm = mcolors.LogNorm(vmin=1, vmax=100)  # can't be zero
    cmap = plt.cm.viridis

    # range of N values
    N_range = np.logspace(3, 7, 20)

    # 1. Loop over each host in the dictionary
    for host_id, host_data in final_analysis_store.items():
        print(f"Processing Host ID: {host_id}...")
        
        # 2. Isolate the data for just this host so your helper functions only see one tree
        single_host_store = {host_id: host_data}

        red_chi_r = []
        red_chi_v = []
        sigma = []
        valid_N = []
        point_counts = []
        all_r_over_epsilon = []

        for n in N_range:
            # Pass the isolated host dictionary instead of the full store
            _, red, sig, both_p, point_count = chi_squared(
                single_host_store, tidal_features,
                N_min=n, m_ret_min=m_ret_min, m_ret_max=m_ret_max
            )
            
            if red == -1:
                continue
                
            all_r_over_epsilon.append(median_rmax_over_epsilon(
                single_host_store, N_min=n, m_ret_max=m_ret_max
            ))
            
            red_chi_r.append(red[0])
            red_chi_v.append(red[1])
            sigma.append(sig)
            valid_N.append(n)
            point_counts.append(point_count)

        # 3. Safety check: If a host has no valid points after cuts, skip plotting to prevent crashes
        if not valid_N:
            print(f"  -> Skipping Host {host_id}: No valid data points found across N_range.")
            continue

        z_score_r = np.sqrt(red_chi_r)
        z_score_v = np.sqrt(red_chi_v)

        percent_r = z_score_r * gvb_error['rmax'] * 100 # multiply by 100 to make it a percent
        percent_v = z_score_v * gvb_error['vmax'] * 100

        z_err_r = 0.5 * (1/np.sqrt(red_chi_r)) * (np.sqrt(2)/np.sqrt(point_counts))
        z_err_v = 0.5 * (1/np.sqrt(red_chi_v)) * (np.sqrt(2)/np.sqrt(point_counts))

        # Note: Added * 100 here so the error bars are scaled to match the percentage values!
        percent_err_r = z_err_r * gvb_error['rmax'] * 100
        percent_err_v = z_err_v * gvb_error['vmax'] * 100

        # create plotting axes
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Plot with error bars
        ax1.errorbar(
            valid_N, percent_r,
            xerr=0, yerr=percent_err_r,
            fmt='None', markersize=10, ecolor='red', capsize=3,
            markeredgecolor='black', label='Rmax', zorder=2
        )
        # Plot over the error bars colored points with the rmax over epsilon as the color
        sc = ax1.scatter(valid_N, percent_r, edgecolors='black', c=all_r_over_epsilon, cmap=cmap, norm=norm, zorder=1, s=100)

        # Same as above but for vmax residuals
        ax1.errorbar(
            valid_N, percent_v,
            xerr=0, yerr=percent_err_v,
            fmt='None', markersize=10, ecolor='blue', capsize=3,
            markeredgecolor='black', label='Vmax', zorder=2
        )
        ax1.scatter(valid_N, percent_v, edgecolors='black', c=all_r_over_epsilon, cmap=cmap, norm=norm, zorder=1, s=100)
        
        ### configure the plot
        ax1.axvline(1048576, linestyle='--', color='black', label='DASH particle count')

        ax1.set_xlabel("Minimum Particle Count")
        ax1.set_ylabel("% Error")
        ax1.set_xscale("log")
        ax1.legend()
        
        # secondary axis for number of points
        ax2 = ax1.twinx()
        ax2.plot(valid_N, point_counts, linestyle=':', color='gray')
        ax2.set_ylabel("Number of Points", color='gray')
        ax2.set_yscale("log")
        ax2.tick_params(axis='y', labelcolor='gray')

        # colorbar
        cbar = fig.colorbar(sc, ax=ax2, pad=0.15)
        cbar.set_label(r"Median $R_{max}/\epsilon$")

        # 4. Dynamically update title and filename per host
        # plt.title(f"Percent Error Function of Resolution (Host {host_id})")
        fig.tight_layout()
        plt.savefig(f"percent_N_min_color_individuals/percent_N_min_color_host_{host_id}.png",bbox_inches='tight')
        plt.clf()
        plt.close(fig) # Free up memory after each host

# plot residuals as a function of rmax / epsilon (epsilon is softening length)
def soft_length_res(final_analysis_store,tidal_features,N_min=0,m_ret_max=100,m_ret_min=0,num_bins=15,min_points=20):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl)
    tidal_features: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
    N_min: minimum particle count to consider
    m_ret_min: minimum retained mass to consider
    m_ret_max: maximum retained mass to consider
    num_bins: number of bins at which to calculate the median line and quartiles
    min_points: minimum number of points to validate each bin
    filename is not an argument as multiple plots with different names are generated, the default scheme can be changed at the bottom of the function
    """
    
    # comoving softening length (valid until z=1)
    co_softening_length = 0.575*h # ckpc/h

    #physical softening length set after z = 1
    phys_softening_length = 0.288 # kpc

    abs_rmax = []
    rmax = []
    vmax = []
    m_retained = []
    concentrations = []
    softening_lengths = []
    particle_counts = []

    concentration_lookup = tidal_features.set_index(['subhalo_id','snapshot'])['concentration_inf'].to_dict()

    for host_id, data in final_analysis_store.items():
        for idx, tree in enumerate(data['satellite_trees']):
            infall = data['infall_metrics']
            inf_snap = infall['snap'][idx]
            mass_inf = infall['mass'][idx]
            vmax_inf = infall['vmax'][idx]
            rmax_inf = infall['rmax'][idx]
            c_inf = concentration_lookup.get((tree['SubhaloID'][0],99))

            for i,snap in enumerate(tree['SnapNum']):
                #apply N = 3000 resolution minimum
                if mass_to_count(tree['SubhaloMass'][i]) >= N_min:
                    # apply cutoff on retained mass
                    m_ret = 100 * (tree['SubhaloMass'][i]*1e10/h / mass_inf)
                    
                    # cut at minimum retained mass, also ignore the initial point because it is always 1,1
                    if m_ret_max > m_ret >= m_ret_min:
                        # find normalized rmax and vmax, note rmax_inf is already in kpc but we need to convert the current rmax
                        r = to_physical(tree['SubhaloVmaxRad'][i],snap) / rmax_inf
                        v = tree['SubhaloVmax'][i] / vmax_inf

                        rmax.append(r)
                        abs_rmax.append(r*rmax_inf)
                        vmax.append(v)          
                        m_retained.append(m_ret)
                        concentrations.append(c_inf)
                        particle_counts.append(mass_to_count(tree['SubhaloMass'][i]))
                        
                        # z = 1 occurs at snapshot 50
                        if snap >= 50:
                            softening_lengths.append(phys_softening_length)
                        else:
                            softening_lengths.append(co_softening_length)

                if snap <= inf_snap:
                        break

    for key in ['r','v']:
        
        # 1. Calculate the predicted values from your theoretical model
        bound_mass = np.array(m_retained) / 100
        expected_values = [X(fb,cs,key) for fb,cs in zip(bound_mass,concentrations)]
        
        # Convert lists to numpy arrays for element-wise operations
        if key == 'r':
            obs_data = np.array(rmax)
            # If the error is a single value, this ensures it works mathematically with arrays
            error = np.array(gvb_error['rmax'])*expected_values
        if key == 'v':
            obs_data = np.array(vmax)
            error = np.array(gvb_error['vmax'])*expected_values

        normalized_residuals = (obs_data - expected_values) / error

        rm_over_softening = np.array(abs_rmax) / np.array(softening_lengths)

        fig, ax = plt.subplots(figsize=(8, 6))

        x_data = rm_over_softening
        y_data = normalized_residuals

        # 1. Plot the raw scatter in the background (faint)
        ax.scatter(x_data, y_data, alpha=0.1, color='gray', s=10, label='Raw TNG Data')
        
        # 2. Create LOG-SPACED bins spanning the absolute min to max of your x-data
        # This guarantees the bins cover the entire tail.
        min_x, max_x = 1, 300
        bin_edges = np.logspace(np.log10(min_x), np.log10(max_x), num_bins + 1)
        
        bin_centers = []
        medians = []
        p16 = [] 
        p84 = [] 
        
        # 3. Loop through the bins and calculate statistics
        for i in range(num_bins):
            # Find points that fall in the current bin
            mask = (x_data >= bin_edges[i]) & (x_data < bin_edges[i+1])
            y_in_bin = y_data[mask]
            x_in_bin = x_data[mask]

            # Clean this specific bin of math errors so np.median doesn't return NaN and break the line
            valid_mask = np.isfinite(y_in_bin) & np.isfinite(x_in_bin)
            y_clean = y_in_bin[valid_mask]
            x_clean = x_in_bin[valid_mask]
            
            # Use max(1, min_points) to guarantee we don't pass an empty array to np.median
            if len(y_clean) >= min_points:
                
                bin_center = np.median(x_clean)
                bin_centers.append(bin_center)
                
                medians.append(np.median(y_clean))
                p16.append(np.percentile(y_clean, 16))
                p84.append(np.percentile(y_clean, 84))
                
        # 5. Plot the running median and scatter bands
        if len(bin_centers) > 0:
            ax.plot(bin_centers, medians, 'b-', lw=3, label='Median Residual')
            ax.plot(bin_centers, p16, 'b--', lw=2, label=r'16th / 84th Percentile ($\pm 1\sigma$)')
            ax.plot(bin_centers, p84, 'b--', lw=2)
            
            # Fill between for a nice visual band
            ax.fill_between(bin_centers, p16, p84, color='blue', alpha=0.2)
        
        # 6. Add a horizontal line at 0 for reference
        ax.axhline(0, color='red', linestyle=':', lw=2, label='Perfect Agreement (0 Residual)')
        # vertical line at 2.8 epsilon as reasonable physical minimum
        ax.axvline(2.8, color='black', linestyle=':', lw=2, label=r'$R_{max}=2.8\epsilon$ (Resolution Minimum)')

        ax.set_xscale('log')
        ax.set_xlim(min_x,max_x)
        ax.set_ylim(-30,100)
        ax.set_xlabel(r'Resolution Ratio ($R_{max} / \epsilon$)')
        if key == 'r':
            ax.set_ylabel(r'Normalized Residual for $R_{max}$')
        elif key == 'v':
            ax.set_ylabel(r'Normalized Residual for $V_{max}$')
            
        # ax.set_title(rf'Residuals as a Function of Softening Length ($N_{{min}}=${N_min})')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"softening_residual_{key}_Nmin_{N_min}.png",bbox_inches='tight')
        plt.clf()

# calculate residual between gvb and tng data point
def calculate_residual(mass,inf_mass,inf_c,rmax,vmax):
    """
    mass: mass of tng subhalo
    inf_mass: infall mass of tng subhalo
    inf_c: infall concentration of tng subhalo
    rmax: normalized rmax (rmax/rmax_inf) of TNG subhalo
    vmax: normalized vamx (vmax/vmax_inf) of TNG subhalo
    """
    fb = mass/inf_mass
    expected_r = X(fb,inf_c,'r')
    expected_v = X(fb,inf_c,'v')

    error_r = gvb_error['rmax']*expected_r
    error_v = gvb_error['vmax']*expected_v

    res_r = ((rmax-expected_r)/error_r)**2
    res_v = ((vmax-expected_v)/error_v)**2

    return res_r, res_v

# plots infall gamma as a function of minimum particle count
def N_v_gamma(final_analysis_store, tidal_features, filename=None):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl)
    tidal_features: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
    filename: output file name, will use a default value if filename=None
    """
    # range of N values
    N_range = np.logspace(3, 7, 20)

    gamma_median = []
    gamma_iqr_low = []
    gamma_iqr_high = []
    valid_n = []
    counts = []

    gamma_lookup = tidal_features.set_index(['subhalo_id','snapshot'])['gamma_inf'].to_dict()

    for N_min in N_range:
        gammas = []
        for host_id, data in final_analysis_store.items():
            for idx, tree in enumerate(data['satellite_trees']):
                gamma_inf = gamma_lookup.get((tree['SubhaloID'][0],99))

                infall = data['infall_metrics']
                mass_inf = infall['mass'][idx]

                if mass_inf / p_mass >= N_min: #mass_inf has already been converted to Msol
                    gammas.append(gamma_inf)

        if len(gammas) > 0:
            gammas = np.array(gammas)

            median = np.median(gammas)
            q1 = np.percentile(gammas, 25)
            q3 = np.percentile(gammas, 75)

            gamma_median.append(median)

            # asymmetric IQR error bars
            gamma_iqr_low.append(median - q1)
            gamma_iqr_high.append(q3 - median)

            valid_n.append(N_min)
            counts.append(len(gammas))

    fig, ax1 = plt.subplots()

    # Median + IQR plot
    ax1.errorbar(
        valid_n,gamma_median,
        yerr=[gamma_iqr_low, gamma_iqr_high],
        fmt="o", ecolor='black', capsize=3, markeredgecolor='black',
        label=r"Median $\gamma$ with IQR"
    )

    ax1.set_xlabel(r"$N_{min}$")
    ax1.set_ylabel(r"Median $\gamma$")
    ax1.set_xscale('log')

    # Second axis for number of points
    ax2 = ax1.twinx()

    ax2.plot(
        valid_n,
        counts,
        marker='none',
        linestyle='--',
        color='gray',
        alpha=0.5,
        label="Number of points"
    )

    ax2.set_ylabel("Number of points", color='gray')
    ax2.tick_params(axis='y', colors='gray')
    ax2.set_yscale("log")

    # Make right spine gray
    ax2.spines['right'].set_color('gray')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2)
    plt.title(r"Infall $\gamma$ Vs Minimum Particle Count")
    if filename is None:
        plt.savefig("N_v_gamma.png",bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')

    plt.clf()

# perform random forest analysis on tidal features
# outputs histogram of relative importances
def reg_random_forest(data,filename=None):
    """
    data: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
        equivalent to tidal_features elsewhere
    filename: output file name, will use a default value if filename=None
    """
    # Setup clean dataset using same strict exclusions
    base_drop_cols = [
        'host_id', 'subhalo_id', 'snapshot', 'time_inf', 'time',
        'residual_r', 'residual_v', 'concentration_inf', 'rmax', 'vmax', 
        'm_retained', 'rmax_inf', 'vmax_inf', 'rmax_gvb', 'vmax_gvb', 'mass'
    ]
    
    df = data.dropna(subset=['residual_r', 'residual_v'])
    existing_drops = [c for c in base_drop_cols if c in df.columns]
    
    X = df.drop(columns=existing_drops).dropna()
    
    # Sync target with final cleaned features
    df_clean = df.loc[X.index]
    y = (df_clean['residual_r'] + df_clean['residual_v']) / np.sqrt(2)

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    # Evaluate results
    y_pred = rf.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    print(f"R² on full data: {r2:.4f}")
    print(f"RMSE on full data: {rmse:.4f}")

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature order:")
    print(importances)

    importances.plot(kind='bar')
    plt.ylabel("Feature Importance")
    plt.title("Random Forest Feature Importance of Structural Parameters")
    plt.tight_layout()
    if filename is None:
        plt.savefig("titled_random_forest_importance.png",bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')
    plt.clf()

# train and test data split for random forest analyses (also calculates log residuals)
def forest_data_split(data):
    """
    Cleans, normalizes, and separates the multi-host dataset into
    pure physical features, targets, and subhalo track groups.
    data: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
        equivalent to tidal_features elsewhere
    """
    # 1. CALCULATE LOG RESIDUALS
    data['log_res_r'] = np.log10(data['rmax_gvb']) - np.log10(data['rmax_norm'])
    data['log_res_v'] = np.log10(data['vmax_gvb']) - np.log10(data['vmax_norm'])

    # 2. DEFINE THE DROP LIST (Targets, Leakage, and Metadata Identifiers)
    base_drop_cols = [
        # irrelevant parameters
        'host_id', 'subhalo_id', 'snapshot',
        # Optimization Targets
        'residual_r', 'residual_v', 'log_res_r', 'log_res_v',
        # Trivial / Mass-Scaling Hallmarks
        'concentration_inf', 'rmax_norm', 'vmax_norm', 'm_retained', 
        'rmax_inf', 'vmax_inf', 'rmax_gvb', 'vmax_gvb', 'mass'
    ]
    
    # Clean rows where target residuals are NaN
    df = data.dropna(subset=['log_res_r', 'log_res_v'])
    
    # 3. GENERATE STRUCTURAL CROSS-VALIDATION GROUPS
    # Combine host_id and subhalo_id to uniquely isolate a subhalo's entire historical track
    groups = df['host_id'].astype(str) + "_" + df['subhalo_id'].astype(str)
    
    # Filter the drop list to columns that exist in the dataframe
    existing_drops = [c for c in base_drop_cols if c in df.columns]
    
    # 4. CREATE PURE STRUCTURAL FEATURE MATRIX
    X_structural = df.drop(columns=existing_drops).dropna()
    
    # Sync targets and groups with the cleaned feature index
    y_r = df.loc[X_structural.index, 'log_res_r'].values
    y_v = df.loc[X_structural.index, 'log_res_v'].values
    groups = groups.loc[X_structural.index].values
    
    return X_structural, y_r, y_v, groups

# random forest analysis using boruta method to test predictability of individual features 
# as well as their relative importances
# outputs a histogram of importances colored by confirmed vs not confirmed from boruta
def random_forest_with_boruta(data,filename=None):
    """
    data: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
        equivalent to tidal_features elsewhere
    filename: output file name, will use a default value if filename=None
    """
    X_structural, y_r, y_v, _ = forest_data_split(data)
    X_arr = X_structural.values

    # Max depth is constrained so Boruta can effectively benchmark against noise
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)

    # RUN BORUTA FOR R_MAX RESIDUALS
    print("--- Running Boruta for R_max Residuals (Pure Structure) ---")
    boruta_r = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=42)
    boruta_r.fit(X_arr, y_r)

    r_confirmed = X_structural.columns[boruta_r.support_].tolist()
    r_tentative = X_structural.columns[boruta_r.support_weak_].tolist()
    print(f"Confirmed R_max Drivers: {r_confirmed}")
    print(f"Tentative R_max Drivers: {r_tentative}\n")

    # RUN BORUTA FOR V_MAX RESIDUALS
    print("--- Running Boruta for V_max Residuals (Pure Structure) ---")
    boruta_v = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=42)
    boruta_v.fit(X_arr, y_v)

    v_confirmed = X_structural.columns[boruta_v.support_].tolist()
    v_tentative = X_structural.columns[boruta_v.support_weak_].tolist()
    print(f"Confirmed V_max Drivers: {v_confirmed}")
    print(f"Tentative V_max Drivers: {v_tentative}\n")

    # VISUALIZE CLEANED RANKINGS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ranks_r = pd.Series(boruta_r.ranking_, index=X_structural.columns).sort_values()
    color_r = ['green' if r==1 else 'yellow' if c in r_tentative else 'red' for r, c in zip(ranks_r, ranks_r.index)]
    ranks_r.plot(kind='bar', ax=axes[0], color=color_r)
    axes[0].set_title("Boruta Structure Ranks (R_max)\nGreen = Confirmed | Red = Rejected")
    axes[0].set_ylabel("Rank (Lower is Better)")

    ranks_v = pd.Series(boruta_v.ranking_, index=X_structural.columns).sort_values()
    color_v = ['green' if r==1 else 'yellow' if c in v_tentative else 'red' for r, c in zip(ranks_v, ranks_v.index)]
    ranks_v.plot(kind='bar', ax=axes[1], color=color_v)
    axes[1].set_title("Boruta Structure Ranks (V_max)\nGreen = Confirmed | Red = Rejected")
    
    plt.tight_layout()
    if filename is None:
        plt.savefig("boruta_pure_structure_importance.png",bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')
    plt.clf()

    return boruta_r, boruta_v

# cross validation method to check generalizeability of prediction metrics for random forest
# prints output values
def check_effect_size(data):
    """
    data: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
        equivalent to tidal_features elsewhere
    """
    X_structural, y_r, y_v, groups = forest_data_split(data)

    rf_eval = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
    
    # Use GroupKFold to guarantee whole tracks stay grouped together
    group_kfold = GroupKFold(n_splits=5)
    
    # Run Leakage-Free Group Cross Validation for R_max
    r2_scores_r = cross_val_score(
        rf_eval, X_structural, y_r, groups=groups, cv=group_kfold, scoring='r2'
    )
    
    # Run Leakage-Free Group Cross Validation for V_max
    r2_scores_v = cross_val_score(
        rf_eval, X_structural, y_v, groups=groups, cv=group_kfold, scoring='r2'
    )
    
    print(f"R_max Residual Predictive Power (Group Out-of-Sample R²): {r2_scores_r.mean():.4f}")
    print(f"V_max Residual Predictive Power (Group Out-of-Sample R²): {r2_scores_v.mean():.4f}")

    print("R_max fold scores:", r2_scores_r)
    print("V_max fold scores:", r2_scores_v)

# 2D histogram of subhalo mass divided by host (central tree) mass
def rel_mass_hist(final_analysis_store,filename=None):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl)
    filename: output file name, will use a default value if filename=None
    """

    sub_masses = []
    host_masses = []

    for host_id, data in final_analysis_store.items():
        trees = data['satellite_trees']
        for idx, tree in enumerate(trees):
            infall = data['infall_metrics']
            inf_snap = infall['snap'][idx]

            for i,snap in enumerate(tree['SnapNum']):
                
                m = tree["SubhaloMass"][i]*1e10 / h # in 10^10 Msol/h so convert
                cent_id = snap_to_id(snap,data['central_tree'])
                m_host = data['central_tree']["SubhaloMass"][cent_id]*1e10 / h # in 10^10 Msol
                
                # print(f"sub: {m}, host: {m_host}")
                
                sub_masses.append(m)
                host_masses.append(m_host)

                if snap <= inf_snap:
                    break

    sub_masses = np.array(sub_masses)
    host_masses = np.array(host_masses)

    # 1. Define logarithmic bins
    bins_x = np.logspace(np.log10(sub_masses.min()), np.log10(sub_masses.max()), 20)
    bins_y = np.logspace(np.log10(host_masses.min()), np.log10(host_masses.max()), 20)

    # 2. Plot the 2D Histogram
    # Using LogNorm so the color scale also reflects log-density
    plt.hist2d(sub_masses, host_masses, bins=[bins_x, bins_y], cmap='viridis', norm=mcolors.LogNorm())
    plt.colorbar(label='Number of Subhalo Instances')

    # 3. Add your threshold line
    xline = np.logspace(8, 12, 20)
    yline = xline * 100
    plt.plot(xline, yline, linestyle='--', color='red', linewidth=4, label='$M_{host} = 100 \\times M_{sub}$')

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$M_{sub}$")
    plt.ylabel(r"$M_{host}$")
    plt.legend(loc='lower right')
    if filename is None:
        plt.savefig("mass_hist.png",bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')
    plt.clf()

# histogram of infall gamma values
def gamma_hist(final_analysis_store,tidal_features,filename=None):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl)
    tidal_features: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
    filename: output file name, will use a default value if filename=None
    """
    gamma_lookup = tidal_features.set_index(['subhalo_id','snapshot'])['gamma_inf'].to_dict()

    gammas = []
    for host_id, data in final_analysis_store.items():
        for idx, tree in enumerate(data['satellite_trees']):
            gamma_inf = gamma_lookup.get((tree['SubhaloID'][0],99))
            gammas.append(gamma_inf)

    
    plt.hist(gammas,bins=50)
    plt.xlabel(r"Infall $\gamma$")
    plt.ylabel("Number of Subhalos")
    plt.yscale('log')
    if filename is None:
        plt.savefig("gamma_hist.png",bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')
    plt.clf()

# plots tidal scatter plot with median line for a specified minimum and maximum infall gamma
def gamma_binned_tidal(final_analysis_store,tidal_features,gamma_bin,filename=None):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl)
    tidal_features: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
    gamma_bin: tuple or two-element array with minimum and maximum gamma values
    filename: output file name, will use a default value if filename=None
    """
    gamma_lookup = tidal_features.set_index(['subhalo_id','snapshot'])['gamma_inf'].to_dict()

    # tidal track band plot
    norm = mcolors.LogNorm(vmin=1, vmax=100)  # can't be zero
    cmap = plt.cm.viridis

    bound_mass_frac = np.logspace(0,2,20)/100
    th_mloss = [(1-m)*100 for m in bound_mass_frac]
    cs = 5
    gvb_th_vmax = [X(fb,cs,'v') for fb in bound_mass_frac]
    gvb_th_rmax = [X(fb,cs,'r') for fb in bound_mass_frac]

    # 1. Initialize master lists to collect all data across all trees
    all_rmax = []
    all_vmax = []

    # loop over trees
    for host_id, data in final_analysis_store.items():
        sat_trees = data['satellite_trees']
        for idx, tree in enumerate(sat_trees):
            gamma_inf = gamma_lookup.get((tree['SubhaloID'][0],99))
            if gamma_bin[0] <= gamma_inf <= gamma_bin[1]:
                infall = data['infall_metrics']
                inf_snap = infall['snap'][idx]
                r0 = infall['rmax'][idx]
                v0 = infall['vmax'][idx]
                for i, snap in enumerate(tree['SnapNum']):
                    # find normalized rmax and vmax
                    r = to_physical(tree['SubhaloVmaxRad'][i], snap) / r0
                    v = tree['SubhaloVmax'][i] / v0

                    all_rmax.append(r)
                    all_vmax.append(v)

                    if snap <= inf_snap:
                        break

    # Convert to numpy arrays and filter out any invalid/zero values for log-math
    all_rmax = np.array(all_rmax)
    all_vmax = np.array(all_vmax)
    valid_mask = np.isfinite(all_rmax) & np.isfinite(all_vmax) & (all_rmax > 0)
    all_rmax = all_rmax[valid_mask]
    all_vmax = all_vmax[valid_mask]

    # 2. Create log-spaced bins for Rmax/Rmax0 
    # (You can increase or decrease '20' to adjust the smoothness of the line)
    bins = np.logspace(np.log10(all_rmax.min()), np.log10(all_rmax.max()), 20)
    bin_centers = np.sqrt(bins[:-1] * bins[1:]) # Geometric centers for log scale

    # 3. Group the Vmax data into the Rmax bins and calculate statistics
    indices = np.digitize(all_rmax, bins)
    med_vmax = []
    p16_vmax = [] # 16th percentile (approx -1 sigma)
    p84_vmax = [] # 84th percentile (approx +1 sigma)
    valid_centers = []

    for i in range(1, len(bins)):
        in_bin = all_vmax[indices == i]
        if len(in_bin) > 5:  # Only plot a bin if it has a reliable number of points
            med_vmax.append(np.median(in_bin))
            p16_vmax.append(np.percentile(in_bin, 16)) 
            p84_vmax.append(np.percentile(in_bin, 84)) 
            valid_centers.append(bin_centers[i-1])

    # 4. Plot the median line and the shaded error band
    plt.plot(valid_centers, med_vmax, color='blue', linewidth=2, label='Simulation Median')
    plt.fill_between(valid_centers, p16_vmax, p84_vmax, color='blue', alpha=0.3, label='1-$\sigma$ Spread')

    # theoretical models
    gvb_th_rmax, gvb_th_vmax, pn_rmax_all, pn_vmax_all, th_mretained = get_theoretical_points(m_ret_min=0,cs=cs)


    # plot the penarrubia prediction lines all gamma
    colors = ['red','orange','purple','black']
    for i, gamma in enumerate(g_fit):
        plt.plot(pn_rmax_all[i], pn_vmax_all[i], linestyle="--", label=f'PN gamma={gamma}',color=colors[i])

    # plot the green and van den bosch prediction line for cs = 23.1
    # Assign this scatter to 'sc' to act as the mappable for the colorbar below
    sc = plt.scatter(gvb_th_rmax, gvb_th_vmax, c=th_mretained, cmap=cmap, norm=norm,
                marker='o', edgecolors='black', s=200, linewidths=0.5, label=rf'GVB ($c_s$={cs})')
    
    # Compute absolute errors based on percentages for green and van den bosch
    vmax_err = gvb_error['vmax'] * np.array(gvb_th_vmax)  # 1% error on vmax
    rmax_err = gvb_error['rmax'] * np.array(gvb_th_rmax)  # 3% error on rmax

    # Add error bars to GVB
    plt.errorbar(
        gvb_th_rmax, gvb_th_vmax,
        xerr=rmax_err, yerr=vmax_err,
        fmt='none', markersize=10, color='none', ecolor='black', capsize=3,
        markeredgecolor='black'
        )

    plt.xlabel("Rmax/Rmax0")
    plt.xscale('log')
    plt.yscale('log')

    # Colorbar now relies on the Green/Van den Bosch scatter points
    cbar = plt.colorbar(sc)
    cbar.set_label("Mass Retained [%]")

    plt.ylabel("Vmax/Vmax0")
    plt.legend()
    if filename is None:
        plt.savefig(f"gamma_bin_tidal_{gamma_bin}.png",bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')
    plt.clf()

# performs the following two cuts from numerical artifact criteria from van den Bosch and Ogiya 2018:
#   r_h / epsilon > 0.62 * c^(1.26)/f(c)
#       # r_h is half-mass radius, epsilon is softening length, c is concentration, and f(c) = ln(1+c)-c/(1+c)
#   N > 80 * (N0^0.2)
#       N is current particle count, N0 is infall particle count (accretion particle count in paper)
def num_cuts(final_analysis_store,tidal_features,filename=None):
    """
    final_analysis_store: output from TNG sample selection (stored in valid_subhalos_multihost.pkl)
    tidal_features: csv output from TNG structural parameter calculations (stored in tidal_features_multihost.csv)
    filename: output file name, will use a default value if filename=None
    """
    c_lookup = tidal_features.set_index(['subhalo_id','snapshot'])['concentration_inf'].to_dict()

    # comoving softening length (valid until z=1)
    co_softening_length = 0.575*h # ckpc/h

    #physical softening length set after z = 1
    phys_softening_length = 0.288 # kpc

    valid_rmax = []
    valid_vmax = []
    valid_mret = []

    # loop over trees
    for host_id, data in final_analysis_store.items():
        sat_trees = data['satellite_trees']
        for idx, tree in enumerate(sat_trees):
            infall = data['infall_metrics']
            inf_snap = infall['snap'][idx]
            m0 = infall['mass'][idx]
            N0 = m0/p_mass
            # get infall halfmass radius (not in infall dict)
            inf_id = snap_to_id(inf_snap,tree)
            rh0 = tree['SubhaloHalfmassRad'][inf_id]
            c = c_lookup.get((tree['SubhaloID'][0],99))
            for i, snap in enumerate(tree['SnapNum']):
                #gather parameters for criteria
                m = tree['SubhaloMass'][i]*1e10/h
                N = m/p_mass
                rh = to_physical(tree['SubhaloHalfmassRad'][i],snap)
                fb = m/m0

                if snap >= 50:
                    epsilon = phys_softening_length
                else:
                    epsilon = co_softening_length

                f_of_c = np.log(1+c)-c/(1+c)

                # apply critera from Bosch and Ogiya 2018
                rh_criteria = (rh/epsilon) > (0.62 * c**1.26 / f_of_c)
                N_criteria = N > (80 * N0**0.2)

                if N_criteria and rh_criteria:
                    # find normalized rmax and vmax
                    r0 = infall['rmax'][idx]
                    v0 = infall['vmax'][idx]
                    
                    r = to_physical(tree['SubhaloVmaxRad'][i], snap) / r0
                    v = tree['SubhaloVmax'][i] / v0

                    valid_rmax.append(r)
                    valid_vmax.append(v)
                    valid_mret.append(100*m/m0) # convert to percentage

                if snap <= inf_snap:
                    break
    # plot results
    norm = mcolors.LogNorm(vmin=1, vmax=100)  # can't be zero
    cmap = plt.cm.viridis

    sc = plt.scatter(valid_rmax, valid_vmax, c=valid_mret, cmap=cmap, norm=norm,s=2)
    
    # median line
    # Convert to numpy arrays and filter out any invalid/zero values for log-math
    all_rmax = np.array(valid_rmax)
    all_vmax = np.array(valid_vmax)
    valid_mask = np.isfinite(all_rmax) & np.isfinite(all_vmax) & (all_rmax > 0)
    all_rmax = all_rmax[valid_mask]
    all_vmax = all_vmax[valid_mask]

    # 2. Create log-spaced bins for Rmax/Rmax0 
    # (You can increase or decrease '20' to adjust the smoothness of the line)
    bins = np.logspace(np.log10(all_rmax.min()), np.log10(all_rmax.max()), 20)
    bin_centers = np.sqrt(bins[:-1] * bins[1:]) # Geometric centers for log scale

    # 3. Group the Vmax data into the Rmax bins and calculate statistics
    indices = np.digitize(all_rmax, bins)
    med_vmax = []
    p16_vmax = [] # 16th percentile (approx -1 sigma)
    p84_vmax = [] # 84th percentile (approx +1 sigma)
    valid_centers = []

    for i in range(1, len(bins)):
        in_bin = all_vmax[indices == i]
        if len(in_bin) > 10:  # Only plot a bin if it has a reliable number of points
            med_vmax.append(np.median(in_bin))
            p16_vmax.append(np.percentile(in_bin, 16)) 
            p84_vmax.append(np.percentile(in_bin, 84)) 
            valid_centers.append(bin_centers[i-1])

    # 4. Plot the median line and the shaded error band
    plt.plot(valid_centers, med_vmax, color='blue', linewidth=2, label='Median Line')

    # get theoretical points
    cs = 5
    gvb_th_rmax, gvb_th_vmax, pn_rmax_all, pn_vmax_all, th_mretained = get_theoretical_points(m_ret_min=0,cs=cs)

    #plot the green and van den bosch prediction line for cs = 23.1
    plt.scatter(gvb_th_rmax,gvb_th_vmax,c=th_mretained,cmap=cmap, norm=norm,\
                marker='o',edgecolors='black',s=200,linewidths=0.5,label=rf'GVB ($c_s$={cs})')

    # plot error bars
    # Compute absolute errors based on percentages for green and van den bosch
    vmax_err = gvb_error['vmax'] * np.array(gvb_th_vmax)  # 1% error on vmax
    rmax_err = gvb_error['rmax'] * np.array(gvb_th_rmax)  # 3% error on rmax

    # plot the penarrubia prediction lines for gamma = 1, 1.5
    colors = ['red','orange','purple','black']
    for i, gamma in enumerate(g_fit):
        if gamma == 1 or gamma == 1.5:
            plt.plot(pn_rmax_all[i], pn_vmax_all[i], linestyle="--", label=f'PN gamma={gamma}',color=colors[i])

    # Plot with error bars
    plt.errorbar(
        gvb_th_rmax, gvb_th_vmax,
        xerr=rmax_err, yerr=vmax_err,
        fmt='none', markersize=10, color='none', ecolor='black', capsize=3,
        markeredgecolor='black'
        )

    plt.xlabel(r"$R_{max}/R_{max,0}$")
    plt.xscale('log')
    plt.yscale('log')
    cbar = plt.colorbar(sc)
    cbar.set_label("Mass Retained [%]")
    plt.ylabel(r"$V_{max}/V_{max,0}$")
    plt.legend(fontsize='small')
    # plt.title("Tidal Track of TNG subhalos and Emperical Models")
    if filename is None:
        plt.savefig(f"tidal_num_cut.png", bbox_inches='tight')
    else:
        plt.savefig(filename,bbox_inches='tight')
    plt.clf()

print("--Generating Plots--")

"""Function calls to generate plots from the thesis report"""

# Figure 2
tidal_band(final_analysis_store)

# Figure 3
chi_squared(final_analysis_store,tidal_features,plot_residuals=True)

# Figure 4
gamma_hist(final_analysis_store,tidal_features)

# Figure 5
gamma_bins = [[0,0.75],[0.75,1.25],[1.25,np.inf]]
for gamma_bin in gamma_bins:
    gamma_binned_tidal(final_analysis_store,tidal_features,gamma_bin)

#Figure 6
tidal_scatter(final_analysis_store,N_min=0)
tidal_scatter(final_analysis_store,N_min=3000)
num_cuts(final_analysis_store,tidal_features)

# Figure 7
p_err_N_min(final_analysis_store,tidal_features)

# Figure 8 (generates for all hosts, more than what is displayed in report)
p_err_N_min_per_host(final_analysis_store,tidal_features)

# Figure 9
N_v_gamma(final_analysis_store,tidal_features)

# Figure 10
soft_length_res(final_analysis_store,tidal_features,N_min=3000)
soft_length_res(final_analysis_store,tidal_features,N_min=0)

# Figure 11
rel_mass_hist(final_analysis_store)

# print info about hosts (map ids to masses, used in Figure 8):
total = 0
for host, data in final_analysis_store.items():
    print(f"ID: {host}, Mass (10^10 Msol): {data['host_metadata']['GroupMass_Msol']*1e-10}")
    total += 1
print("Number of Hosts Used:",total)



