# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: darknanograv
#     language: python
#     name: darknanograv
# ---

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # run_spheric
#
# This file represents a notebook/script used to run and test spheric.
#
# Note that this notebook has been paired with a python script file in the `percent` format using `jupytext`.  Use that file for comparing revisions. See [Collaborating on notebooks with git](https://github.com/mwouts/jupytext#collaborating-on-notebooks-with-git) for more details.

# %% [markdown]
# # SpherIC documentation

# %% [markdown]
# Here is the CLI help menu (Note the nonstandard `-help` and `-version` options are not mentioned):
#
# ```
# -halo               : generate a dark matter halo
# -Nhalo <value>      : number of halo particles
# -Mhalo <value>      : total mass of the halo (default Mhalo = 1)
# -a <value>          : alpha parameter in the halo density profile
# -b <value>          : beta parameter in the halo density profile
# -c <value>          : gamma parameter in the halo density profile
# -rs <value>         : scale radius (default rs = 1)
# -rcutoff <value>    : cutoff radius for cutoff halo models (i.e. beta >= 3)
# -king               : generate a stellar component with a King profile
# -hernquist          : generate a stellar component with a Hernquist profile
# -plummer            : generate a stellar component with a Plummer profile
# -starabg            : generate a 'stellar' component with a (alpha,beta,gamma) profile
# -Nstar <value>      : number of star particles
# -Mstar <value>      : total stellar mass 
# -rc <value>         : king core radius
# -rt <value>         : king tidal radius
# -rhern <value>      : scale radius for Hernquist profile
# -rp <value>         : scale radius for Plummer profile
# -as <value>         : alpha parameter in the 'star' abg profile
# -bs <value>         : beta parameter in the 'star' abg profile
# -cs <value>         : gamma parameter in the 'star' abg profile
# -rss <value>        : scale radius in the 'star' abg profile (default rss = 1)
# -rcuts <value>      : cutoff radius for cutoff 'star' abg models (i.e. beta >= 3)
# -MBH <value>        : mass of black hole (default MBH = 0)
# -name <value>       : name of the output file
# -dx/dy/dz <value>   : position offset for the initial conditions (default All = 0)
# -dvx/dvy/dvz <value>: velocity offset for the initial conditions (default All = 0)
# -ogr                : set this flag for outputting grid in r in an ASCII file
# -ogdf               : set this flag for outputting grid for distribution function in an ASCII file
# -ogb                : set this flag for generating a GADGET2 initial conditions binary file
# -ogh                : set this flag for generating a GIZMO initial conditions HDF5 file
# -otb                : set this flag for generating a TIPSY initial conditions binary file
# -oift               : set this flag to write positions for IFRIT binary file 
# -opfs               : set this flag to write a table of density profiles in an ASCII file
# -nostarpot          : set this flag for excluding the stellar potential 
# -randomseed <value> : set this flag for setting a value for a random seed (default: random value)
# -dorvirexact        : set this flag for calculating rvir exactly via N^2 sum - Warning: time consuming for large N!
#
# Note: The -starabg flag is only intended for use with GIZMO initial conditions (-ogh). Usage elsewhere is not tested.
# ```
#
# From running and code analysis, I think the `rs` parameter actually defaults to `-1` aka no default.
#
# Note: SpherIC assumes units of $10^{10}$ M$_{\odot}$, kpc, and km/s. This is **very** important! If you want to change the Halo mass later (by e.g. changing UnitMass in GIZMO) you need to rerun SpherIC. Otherwise, your sim will not function (tested from personal experience)

# %% [markdown]
# # All initialization code - Put reused functions here!
# We will put any common definitions and imports here. We'll still try to break them up by section though

# %%
import numpy as np
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 150

# %% [markdown]
#

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Generic helper definitions
# These are useful for any Jupyter notebook

# %%
import numpy as np
from itertools import chain


class StopExecution(Exception):
    """Placeholder class for stopping cells prematurely"""

    def _render_traceback_(self):
        pass


def exit():
    """This function lets you stop a notebook cell without causing an error or quitting the kernel"""
    raise StopExecution


def flatten_list(deep_list: list[list[object]]):
    """Convert a list of lists to single list"""
    return list(chain.from_iterable(deep_list))


def signif(x, p):
    """This function rounds numbers to the provided number of significant figures"""
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

from unyt.array import unyt_array,unyt_quantity
def latex_float(f):
    ''' 
    Convert scientific notation to latex notation (useful for plots otherwise just use siunit package). 
    From https://stackoverflow.com/a/13490601 with modifications
    '''
    if isinstance(f,unyt_array) or isinstance(f,unyt_quantity):
        f = f.v
    float_str = f"{f:.2g}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if base=="1":
            return f'{r"10^{"}{int(exponent)}{r"}"}'
        if int(exponent)==0:
            return f'{base}'
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


# %% [markdown]
# ## SpherIC

# %% [markdown]
# The SpherIC code has been moved to a seperate file: `spheric.py`

# %%
from spheric import SphericOptions,spheric

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Combine Halo ICs from SpherIC

# %% [markdown]
# This should be straightforward, since they're just hdf5 files.

# %%
import h5py as h5py
import numpy as np

def combineICs(ic1name,ic2name,outname):
    with h5py.File(ic1name,'r') as ic1,h5py.File(ic2name,'r') as ic2,h5py.File(outname,'w') as out:
        h1 = ic1['/Header']
        h2 = ic2['/Header']
        ho = out.create_group("Header")
        for k in h1.attrs:
            if 'NumPart' in k:
                # Need to update when combining
                x1 = h1.attrs[k]
                x2 = h2.attrs[k]
                xo = [y1+y2 for y1,y2 in zip(x1,x2)]
            else:
                # Can copy directly
                xo = h1.attrs[k]
            ho.attrs[k] = xo
        numpart1 = sum(h1.attrs['NumPart_Total'])
        for pt in ['PartType1','PartType2','PartType4','PartType5']:
            for var in ['Coordinates','Velocities','Masses','ParticleIDs']:
                n = f"/{pt}/{var}"
                x1 = x2 = []
                if n in ic1:
                    x1 = ic1[n][:]
                if n in ic2:
                    x2 = ic2[n][:]
                    if 'Part' in var:
                        x2 = x2 + numpart1
                xo = np.concatenate((x1, x2))
                if len(xo)==0:
                    continue
                assert len(xo) == len(x1) + len(x2),f"{len(xo)}!={len(x1)}+{len(x2)}"
                out.create_dataset(n,data=xo)
    pass


# %% [markdown]
# ## Profiles

# %% [markdown]
# ### density

# %%
import yt
import numpy as np
from scipy.special import gamma as gammafun
from scipy.special import hyp2f1
from unyt import kiloparsec as kpc
from unyt.array import unyt_array, unyt_quantity

pt1 = "PartType1"

def get_αβγ_prof(r,α=1,β=3,γ=1,*,sphereopts=None,rs = 1,rcut=100,Mtot=1):
    if sphereopts is not None:
        α = sphereopts.alpha
        β = sphereopts.beta
        γ = sphereopts.gamma
        rs = sphereopts.rs
        rcut = sphereopts.rcutoff
        Mtot = sphereopts.Mhalo
    rdec = 0.3 * rcut
    δ = 10/3 - (γ + β * (rcut/rs)**α)/(1 + (rcut/rs)**α)

    # we'll assume rcut/rs is large
    # Note that apparently Im is given by the hypergeometric function
    # q^(3-γ) * hypergeom_2F_1( (3-γ)/α, (β-γ)/α; (α-γ+3)/α; -q^α ) / (3-γ)
    # where q is rcut/rs - not sure why Zemp didn't just use this, possibly
    # because it's more complicated
    q = rcut/rs
    if isinstance(q,unyt_array) or isinstance(q,unyt_quantity):
        Im = q.v**(3-γ) * hyp2f1( (3-γ)/α, (β-γ)/α, (α-γ+3)/α, -q.v**α ) / (3-γ)
    else:
        Im = q**(3-γ) * hyp2f1( (3-γ)/α, (β-γ)/α, (α-γ+3)/α, -q**α ) / (3-γ)
    Imcut = 0
    ρ_0 = Mtot / (4*np.pi * rs**3 * (Im + Imcut))
    
    ρ_low = (ρ_0)/((r/rs)**γ * (1 + (r/rs)**α)**((β-γ)/α))
    ρcut = (ρ_0)/((rcut/rs)**γ * (1 + (rcut/rs)**α)**((β-γ)/α))
    ρ_high = ρcut * (r/rcut)**δ * np.exp(-(r-rcut)/rdec)
    prof = ρ_low  * (r<=rcut) + ρ_high * (r>rcut)
    #print(f"{ρ_0=}\n{ρ_low=}\n{ρ_high=}\n{prof=}")
    return prof

def get_sphere(ds,radius=(274,"pc"),*,center=None,refine=False,ref_radius=None):
    if center is None:
        center = ds.all_data().quantities.center_of_mass(
            use_gas=False,use_particles=True)
    if refine:
        #refine center to ignore some of the outer stuff
        if not (type(center) is unyt_array or type(center) is unyt_quantity):
            if isinstance(center, tuple):
                center = center[0]
            center = center * kpc
        print(f"0: {center.to('kpc')}")
        if ref_radius is None:
            ref_radius = radius
        if isinstance(ref_radius, tuple):
            new_radius = ref_radius[0]*kpc
        for i in range(1,3):
            new_radius=new_radius*0.75
            center = ds.sphere(center,new_radius).quantities.center_of_mass(
                use_gas=False,use_particles=True)
        print(f'2: {center.to("kpc")}')
    return ds.sphere(center,radius)

def rho_prof(*,ds=None,radius=None,center=None,sphere=None,
             pt=pt1,override_bins=None,stretch=True):
    if sphere is None  and (ds is None or radius is None):
        raise Exception("Need to provide ds and radius or sphere")
    if sphere is None:
        sphere = get_sphere(ds,radius,center=center)
    rhofield = (pt,"density")
    numfield = (pt,"particle_ones") 
    rfield = (pt, "particle_radius")
    wfield = numfield
    volume_normal = False
    if rhofield not in sphere.ds.field_list:
        #print(f'{rhofield} not in sphere.ds.field_list. Using ({pt},"Masses") instead.')
        rhofield = (pt,"Masses")
        volume_normal = True
        wfield = None
    #wfield = (pt,"Masses")
    if not isinstance(override_bins,dict):
        override_bins = {rfield:override_bins}
    prof = yt.create_profile(
            sphere,
            [rfield],
            fields=[rhofield],
            weight_field=wfield,
            deposition='ngb',
            accumulation=False,
            override_bins=override_bins,
        )
    numprof = yt.create_profile(
            sphere,
            [rfield],
            fields=[numfield],
            weight_field=None,
            accumulation=True,
            override_bins=override_bins,
        )
    rho = prof[rhofield]
    npart = numprof[numfield]
    r = prof.x.to("pc")
    if stretch:
        # Depending on the particle spacing/resolution, some bins won't
        # have particles and thus n, T, and mu would be 0. In those cases
        # copy the value from the first non-zero inner bin 
        for i,ni in enumerate(npart):
            if stretch=='nan':
                if rho[i]<=0:
                    rho[i] = np.nan
            elif stretch=='skip':
                continue
            else:
                if i==0 or ni>npart[i-1]:
                    continue
                rho[i] = rho[i-1]
        #if stretch=='skip':
            # need to remove bins with no particles
    if volume_normal:
        # rho is currently just the total mass in the bin
        # Need to divide by the volume of the bin shell
        rp0 = [0,*r]
        vol = [4/3 * np.pi * (rp0[i]**3-rp0[i-1]**3) for i in range(len(rp0)) if i>0]
        rho = rho / vol
    return rho,(prof,npart)


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### velocity

# %%
# This should be very similar to the dens_prof function
def vel_prof(*,ds=None,radius=None,center=None,sphere=None,
             pt=pt1,override_bins=None,stretch=True):
    if sphere is None  and (ds is None or radius is None):
        raise Exception("Need to provide ds and radius or sphere")
    if sphere is None:
        sphere = get_sphere(ds,radius,center=center)
    bulk_velocity = sphere.quantities.bulk_velocity(use_gas=False,use_particles=True)
    sphere.set_field_parameter('bulk_velocity',bulk_velocity)
    field = (pt,"particle_velocity_spherical_radius")
    numfield = (pt,"particle_ones") 
    wfield = numfield
    prof = yt.create_profile(
            sphere,
            [(pt, "particle_radius")],
            fields=[field],
            weight_field=wfield,
            deposition='ngb',
            accumulation=False,
            override_bins=override_bins,
        )
    numprof = yt.create_profile(
            sphere,
            [(pt, "particle_radius")],
            fields=[numfield],
            weight_field=None,
            accumulation=True,
            override_bins=override_bins,
        )
    vel = prof[field]
    npart = numprof[numfield]
    r = prof.x.to("pc")
    if stretch:
        # Depending on the particle spacing/resolution, some bins won't
        # have particles and thus n, T, and mu would be 0. In those cases
        # copy the value from the first non-zero inner bin 
        for i,ni in enumerate(npart):
            if stretch=='nan':
                if vel[i]<=0:
                    vel[i] = np.nan
            elif stretch=='skip':
                continue
            else:
                if i==0 or ni>npart[i-1]:
                    continue
                vel[i] = vel[i-1]
        #if stretch=='skip':
            # need to remove bins with no particles
    return vel,(prof,npart)


# %% [markdown]
# ### shells

# %%
class Shell:
    inner=None
    outer=None
    ri=0
    ro=0
    dd=None
    def __init__(self,*,ds=None,inner=None,outer=None,ri=0,ro=None):
        if ds is None and outer is None:
            raise TypeError('Must provide one of (ds,r1=0,r2) or (outer)')
        if ds is not None and ro is None:
            raise TypeError('If ds is specified, must provided at least ro')
        
        if ds is not None:
            center = [0,0,0]
            if not hasattr(ds,'sphere'):
                if not hasattr(ds,'ds'):
                    raise TypeError(f'ds (type:{type(ds)} has no way to get a subsphere')
                center = ds.center
                ds = ds.ds
            outer = ds.sphere(radius=ro,center=center)
            if ri>0:
                inner = ds.sphere(radius=ri,center=center)
        
        if inner is None:
            dd = outer
        else:
            dd = outer-inner
        self.inner = inner
        self.outer = outer
        self.ri = 0*outer.radius if inner is None else inner.radius
        self.ro = outer.radius
        self.dd = dd
        
    def numparts(self,pt='all'):
        return self.dd.quantities.total_quantity((pt,'particle_ones'))

def get_shells(*,ds=None,radius=None,center=None,sphere=None,
             pt='all',override_bins=None,stretch="merge"):
    if sphere is None  and (ds is None or radius is None):
        raise Exception("Need to provide ds and radius or sphere")
    if sphere is None:
        sphere = get_sphere(ds,radius,center=center)
    numfield = (pt,"particle_ones") 
    rfield = (pt, "particle_radius")
    wfield = numfield
    if not isinstance(override_bins,dict):
        override_bins = {rfield:override_bins}
    numprof = yt.create_profile(
            sphere,
            [rfield],
            fields=[numfield],
            weight_field=None,
            accumulation=False,
            override_bins=override_bins,
            logs={rfield:True},
        )
    npart = numprof[numfield]
    r = numprof.x.to("pc")
    # Depending on the particle spacing/resolution, some bins won't
    # have particles. How we deal with them depends on stretch. We'll
    # default to leaving the bins in -note that this might result in 
    # empty shells. Alternatively we can merge adjacent empty shells
    # together or remove them from the list
    bins = np.hstack((unyt_array([0*r.units,*r[:-1]]),r)).T
    bin_mask = [np>0 for np in npart]
    if stretch=="merge":
        bins = bins[bin_mask]
        for i,b in enumerate(bins):
            if i<len(bins)-1 and b[1]<bins[i+1][0]:
                bins[i][1]=bins[i+1][0]
    elif stretch=='skip':
        print(f'Skipping {len(bin_mask)-sum(bin_mask)}-particle bins')
        bins = bins[bin_mask]
    #spheres = [sphere.ds.sphere(radius=b[1],center=sphere.center) for b in bins]
    #shells = [s-spheres[i-1] if i>0 else s for i,s in enumerate(spheres) ]
    shells = [Shell(ds=sphere,ri=b[0],ro=b[1]) for b in bins]
    return shells

def get_shell_bins(shells):
    bins = unyt_array([shells[0].ri,*(s.ro for s in shells)]).to('pc')
    return bins

def plot_num_particles(shells,*,ax=None,**kwargs):
    if ax is None:
        fig=plt.figure();
        ax=fig.add_subplot();
    if shells is None:
        shells = get_shells(**kwargs)
    bins = get_shell_bins(shells)
    hr = 'PartType4' if 'PartType4' in shells[0].dd.ds.particle_types else 'PartType1'
    lr = 'PartType1' if 'PartType4' in shells[0].dd.ds.particle_types else 'PartType2'
    nhr = [s.numparts(pt=hr) for s in shells]
    nlr = [s.numparts(pt=lr) for s in shells]
    hrline = ax.stairs(nhr,edges=bins,label='HR',lw=2);
    lrline = ax.stairs(nlr,edges=bins,label='LR',edgecolor=hrline.get_edgecolor(),
              facecolor=hrline.get_facecolor(),lw=1);
    ax.set_xscale('symlog',linthresh=1e-2)
    ax.set_yscale('symlog')
    ax.set_xlabel(f'r ({bins.units})')
    ax.set_ylabel('# Particles')
    ax.legend()
    return hrline,lrline


# %% [markdown]
# ### dispersion

# %%
def dispersion(shells=None,*,pt='all',p=2,**kwargs):
    if shells is None:
        shells = get_shells(**kwargs)
    vrfield = (pt,'particle_velocity_spherical_radius')
    #s20 = shells[0].std(vrfield)
    #s2 = np.zeros(len(shells)) * s2.units
    #for i,s in enumerate(shells):
    #    s2[i] = s.quantites.std(vrfield)
    s2 = unyt_array([s.dd.std(vrfield)**p for s in shells]).to(f'km**{p}/s**{p}')
    return s2

def plot_dispersion(shells,*,ax=None,p=2,**kwargs):
    if ax is None:
        fig=plt.figure();
        ax=fig.add_subplot();
    if shells is None:
        shells = get_shells(**kwargs)
    bins = get_shell_bins(shells)
    hr = 'PartType4' if 'PartType4' in shells[0].dd.ds.particle_types else 'PartType1'
    lr = 'PartType1' if 'PartType4' in shells[0].dd.ds.particle_types else 'PartType2'
    s2hr = dispersion(shells,pt=hr,p=p)
    s2lr = dispersion(shells,pt=lr,p=p)
    hrline = ax.stairs(s2hr,edges=bins,label='HR',lw=2);
    lrline = ax.stairs(s2lr,edges=bins,label='LR',edgecolor=hrline.get_edgecolor(),
              facecolor=hrline.get_facecolor(),lw=1);
    ax.set_xscale('symlog',linthresh=1e-2)
    ax.set_yscale('symlog')
    ax.set_xlabel(f'r ({bins.units})')
    ax.set_ylabel(r'$\sigma^'f'{p}'r'$ (km$^'f'{p}'r'$/s$^'f'{p}'r'$)')
    ax.legend()
    return hrline,lrline


# %% [markdown]
# ## Mean free path
# Defined as 
#
# $$ l = \frac{1}{n  \sigma v} = (\rho  \frac{\sigma}{m}  v)^{-1} $$
#
# Since we don't have local versions of $n/\rho$, we need to use the profiles
#
# So this will be the average mean free path in spherical shell

# %%
from yt.units import centimeter as cm
from yt.units import gram as g
def _relative_particle_speed(field, data, ftype,  ):
    return np.sqrt(np.sum(data[ftype,"relative_particle_velocity"]**2,axis=1))

from functools import partial
def add_particle_speed(ds,ftype="PartType1",force_override=False):
    ds.add_field(
        (ftype, "relative_particle_speed"),
        function=partial(_relative_particle_speed,ftype=ftype),
        sampling_type="particle",
        display_name=r"\bar{v}",
        units='auto',
        force_override=force_override,
        validators=[ValidateDataField((ftype,"Velocities"))],
    )

from yt.fields.derived_field import ValidateDataField
def mean_free_path(shells=None,*,sigma_over_m=None,pt='PartType4',**kwargs):
    # first define the shells
    if shells is None:
        shells = get_shells(**kwargs)
    if sigma_over_m is None:
        sigma_over_m = lambda v:10*cm**2/g
    vrfield = (pt,'relative_particle_speed')
    #s20 = shells[0].std(vrfield)
    #s2 = np.zeros(len(shells)) * s2.units
    #for i,s in enumerate(shells):
    #    s2[i] = s.quantites.std(vrfield)
    if vrfield not in shells[0].dd.ds.derived_field_list:
        add_particle_speed(shells[0].dd.ds,ftype=pt)
    v = unyt_array([s.dd.mean(vrfield) for s in shells]).to(f'km/s')
    bins = get_shell_bins(shells)
    ρ,_ = rho_prof(override_bins=bins,pt=pt,**kwargs)
    l = 1/(ρ*sigma_over_m(v))
    return l, shells

def plot_mean_free_path(*,ax=None,**kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    lhr,shellshr = mean_free_path(**kwargs)
    bins = get_shell_bins(shellshr)
    hrline = ax.stairs(lhr,edges=bins,lw=2,label='HR')
    llr,shellslr = mean_free_path(pt='PartType1',**kwargs)
    bins = get_shell_bins(shellslr)
    lrline = ax.stairs(llr,edges=bins,lw=1,label='LR',edgecolor=hrline.get_edgecolor(),
                     facecolor=hrline.get_facecolor(),)
    ax.set_xscale('symlog',linthresh=1e-1)
    ax.set_yscale('symlog',linthresh=1e-4)
    ax.set_xlabel(f'r ({bins.units})')
    ax.set_ylabel(f'$l$ ({llr.units})')
    ax.legend()
    return hrline,lrline
    


# %% [markdown]
# ### Relaxation time
# $t_r = \frac{1}{\rho (\sigma/m) v} $

# %%
def relax_time(*,pt='PartType4',**kwargs):
    l, shells = mean_free_path(**kwargs)
    vrfield = (pt,'relative_particle_speed')
    v = unyt_array([s.dd.mean(vrfield) for s in shells]).to(f'km/s')
    tr = (l/v).to('Myr')
    return tr, shells

def plot_relax_time(*,ax=None,**kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    vhr,shellshr = relax_time(**kwargs)
    bins = get_shell_bins(shellshr)
    hrline = ax.stairs(vhr,edges=bins,lw=2,label='HR')
    vlr,shellslr = relax_time(pt='PartType1',**kwargs)
    bins = get_shell_bins(shellslr)
    lrline = ax.stairs(vlr,edges=bins,lw=1,label='LR',edgecolor=hrline.get_edgecolor(),
                      facecolor=hrline.get_facecolor(),)
    ax.set_xscale('symlog',linthresh=1e-1)
    ax.set_yscale('symlog',)
    ax.set_xlabel(f'r ({bins.units})')
    ax.set_ylabel(f'$t_r$ ({vlr.units})')
    ax.legend()
    return hrline,lrline


# %%

# %% [markdown]
# ## Various plotting commands

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Make animations
# These are another set of fragile definitions, those in this case more because of how yt is implemented. `make_animation_from_images` should always work, but it won't create an animation that can be paused, rewound, etc. `make_animation_directly` will only work for plots that are based on SlicePlots and ProjectionPlots, it won't work on things like ParticlePhasePlots

# %%
import os
import tempfile
import imageio.v3 as imageio
from pathlib import Path
from IPython.display import Image,Video
from matplotlib.animation import FuncAnimation


def make_animation_from_images(dss,plotcommand,filename:str,*,img_dir=None):
    # img_dir is where the intermediate images are generated. If None, images
    # are saved in a temporary directory that is deleted when the function returns
    with tempfile.TemporaryDirectory() as tempdirname:
        filelist = []
        if img_dir is None:
            img_dir = tempdirname
        else:
            os.makedirs(img_dir)
        print(f"Saving figures to {img_dir}")
        # generate temporary figures
        for idx,ds in enumerate(dss):
            print(f"Frame {idx} of {len(dss)}", end="\r")
            p = plotcommand(ds)
            if isinstance(p,tuple):
                p = p[0]
            fn = os.path.join(img_dir,f"ds{idx}.png")
            filelist.append(fn)
            p.save(fn)
        print(f"Frame {len(dss)} of {len(dss)}")
        # build gif from files
        fpath = Path(filename)
        if not fpath.parent.exists():
            fpath.parent.mkdir()
        #with imageio.get_writer(filename, mode='I') as writer: # This is obsolete with ImageIO v3
        with imageio.imopen(filename,"w") as writer:
            images = []
            for fn in filelist:
                images.append(imageio.imread(fn))
            writer.write(images,loop=0)
    return load_animation_from_file(filename)


def make_animation_directly(dss,plotcommand,save_filename:str = None):
    # Warning: this function does not currently work with any plots based
    # on PhasePlot (it's supposed to, so bug the yt developers to fix it, see
    # issue #4291)
    plot,col_field,timetxt = plotcommand(dss[0])
    if timetxt is None:
        plot.annotate_timestamp(time_unit="Myr",draw_inset_box=True)
    fig = plot.plots[col_field].figure

    # animate must accept an integer frame number. We use the frame number
    # to identify which dataset in the time series we want to load
    def animate(i):
        ds = dss[i]
        print(f"Drawing frame {i}", end = "\r")
        if timetxt is not None:
            time = ds.current_time.to("Myr")
            timetxt._plot_text[col_field]=f"T={time:0.3g}"
        # ParticlePhasePlot has no _recreate_frb method, so this isn't
        # sufficient
        plot._switch_ds(ds)
    animation = FuncAnimation(fig, animate, frames=len(dss))
    if save_filename is not None:
        sfpath = Path(save_filename)
        if not sfpath.parent.exists():
            sfpath.parent.mkdir()
        animation.save(save_filename)
    return animation


def load_animation_from_file(filename:str):
    if "gif" in str(filename):
        return Image(filename=filename)
    return Video(filename=filename)


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Color varying line def

# %%
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_color_varying_line(x,y,t,*,z=None,fig=None,ax=None,resize=True,label=None,cmap='viridis',**kwargs):
    if ax is None:
        if fig is None:
            fig = plt.figure(**kwargs)
        ax = fig.add_subplot(**kwargs)
    # from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    if z is None:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
    else:
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(t.min(), t.max())
    if z is None:
        lc = LineCollection(segments, cmap=cmap,norm=norm,label=label )
    else:
        lc = Line3DCollection(segments, cmap=cmap,
                    norm=norm,label=label)
    # Set the values used for colormapping
    lc.set_array(t)
    lc.set_linewidth(2)
    if z is None:
        line = ax.add_collection(lc)
    else:
        line = ax.add_collection3d(lc)
    #fig.colorbar(line, ax=ax)
    if resize:
        ax.set_xlim(x.min(),x.max())
        ax.set_ylim(y.min(),y.max())
        if z is not None:
            ax.set_zlim(z.min(),z.max())
    return line


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Plot Random Walk

# %%
import matplotlib.pyplot as plt

def plot_random_walk(x,y,t,*,inspos=[-0.2, -0.17, 0.15, 0.12]):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    #plot = ax.scatter(x,y,5,t)
    #cb = plt.colorbar(plot)
    #cb.set_label('Time [Myr]')
    line=plot_color_varying_line(x[:,0],y[:,0],t,fig=fig,ax=ax,label="All CoM")
    line.figure.colorbar(line).set_label('Time [Myr]')
    line.set_linewidth(3)
    line=plot_color_varying_line(x[:,1],y[:,1],t,fig=fig,ax=ax,resize=True,label="DM CoM")
    line.set_linestyle('dashed')
    line.set_linewidth(1)
    xl = line.axes.get_xlim()
    yl = line.axes.get_ylim()
    line=plot_color_varying_line(x[:,2],y[:,2],t,fig=fig,ax=ax,resize=True,label="BH")
    line.set_linestyle('dotted')
    line.set_linewidth(1)
    ax.set_xlabel('x (kpc)')
    ax.set_ylabel('y (kpc)')
    ax.legend()

    if inspos != -1:
        axins = ax.inset_axes(
            inspos,transform=ax.transData,
            xlim=(xl[0], xl[1]), ylim=(yl[0], yl[1]), xticklabels=[], yticklabels=[])
        line=plot_color_varying_line(x[:,0],y[:,0],t,fig=fig,ax=axins,resize=False)
        line.set_linewidth(3)
        line=plot_color_varying_line(x[:,1],y[:,1],t,fig=fig,ax=axins,resize=False)
        line.set_linestyle('dashed')
        line.set_linewidth(1)
        line=plot_color_varying_line(x[:,2],y[:,2],t,fig=fig,ax=axins,resize=False)
        line.set_linestyle('dotted')
        line.set_linewidth(1)
        ax.indicate_inset_zoom(axins, edgecolor="black")
    return fig


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Plot Single DS (similar to plotmerger or Simulation.plotds)

# %%
def plotds(ds,*,width=(300,'kpc'),**kwargs):
    col_field = ("PartType1","Masses")
    ad = ds.all_data() # make sure the data is loaded and particle fields are populated
    hrpt = 'PartType1' if 'PartType2' in ds.particle_types else 'PartType4'
    lrpt = 'PartType2' if 'PartType2' in ds.particle_types else 'PartType1'
    plot = yt.ParticleProjectionPlot(ds,"z",(hrpt,'particle_ones'),col_field,width=width,window_size=(3,3),origin='native',**kwargs)
    if ('PartType5','Coordinates') in ds.field_list:
        plot.annotate_particles(20,ptype="PartType5",col="orange",p_size=10,alpha=0.75)
    if (lrpt,'Coordinates') in ds.field_list:
        plot.annotate_particles(20,ptype=lrpt,col='red',p_size=10,alpha=0.1)
    plot.annotate_timestamp(time_unit="Myr",draw_inset_box=True)
    return plot,col_field,None


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Plot single ds density profile (like Simulation.prof_ds)

# %%
def prof_ds(ds,*,ax=None,**kwargs):
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot()
    ds.all_data()
    hrpt = 'PartType1' if 'PartType2' in ds.particle_types else 'PartType4'
    lrpt = 'PartType2' if 'PartType2' in ds.particle_types else 'PartType1'
    rhohr,(prof,npart) = rho_prof(ds=ds,radius=(500,'kpc'),stretch='nan',pt=hrpt)
    rhohr = rhohr.to("code_mass/kpc**3")
    r = prof.x.to("pc")
    hrl, = ax.loglog(r,rhohr,'.-',label=f"t={signif(ds.current_time.to('Myr'),2)} HR DM ")
    r0 = r
    rholr,(prof,npart) = rho_prof(ds=ds,radius=(500,'kpc'),stretch='nan',pt=lrpt)
    rholr = rholr.to("code_mass/kpc**3")
    rlr = prof.x.to("pc")
    r0 = np.unique([*r0,*rlr]) * rlr.units
    lrl, = ax.loglog(rlr,rholr,'.',markersize=2,label=f"_Lo-res DM",color=hrl.get_color())
    
    rho_αβγ130 = get_αβγ_prof(r0.to('kpc').v,)
    rho_αβγNFW = get_αβγ_prof(r0.to('kpc').v,γ=0)
    corel, = ax.loglog(r0, rho_αβγ130,label=f'({1},{3},{0})')
    cuspl, = ax.loglog(r0, rho_αβγNFW,label=f'({1},{3},{1})')
    mass_unit = ds.mass_unit.to("Msun")
    mus = latex_float(mass_unit)
    ax.set_xlabel(f"r ({r.units})")
    ax.set_ylabel(f'ρ$(r)$ (${mus}$' r' M$_{{\odot}}$/kpc$^3$)')
    ax.legend()
    return hrl,lrl,corel,cuspl


# %% [markdown]
# ## COM and COMs class defs

# %%
from functools import total_ordering

@total_ordering
class COM:
    filename = None
    all = None
    name_mapping = None
    ptype_list = None
    __is_COM__ = True
    def __init__(self,ds,*,name_mapping={'dm':'PartType1','stars':'PartType4','bh':'PartType5'}):
        self.filename = ds.filename
        ad = ds.all_data()
        self.time = ds.current_time
        self.all = ad.quantities.center_of_mass(use_gas=False,use_particles=True,)
        for ptype in ds.particle_fields_by_type:
            self.__dict__[ptype] = ad.quantities.center_of_mass(
                use_gas=False,use_particles=True,particle_type=ptype)
        self.name_mapping = name_mapping
        mapped = []
        for p1,p2 in name_mapping.items():
            if p2 in self.__dict__:
                self.__dict__[p1] = self.__dict__[p2]
                mapped.append(p1)
        self.ptype_list = ["all",*ds.particle_fields_by_type,*mapped]

    def __repr__(self):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        s = "COM("
        for a in attributes:
            match a:
                case (_,str()):
                    s = s + f'{a[0]}="{a[1]}",'
                case _:
                    s = s + f"{a[0]}={a[1]},"
        s = s + ")"
        return s

    def __lt__(self,other):
        if hasattr(other,"time"):
            return self.time<other.time
        else:
            return NotImplemented
        
    def __getitem__(self,name):
        return self.__dict__[name]

# want time-ordered list of COM - would be nice to insert in order, but can also just sort at end
# want method to obtain bulk position (possibly at given times) - use interpolation?
# could use plot_random_walk as inherent plotting method
from sortedcontainers import SortedKeyList
from collections.abc import Iterable
from scipy.interpolate import splprep,splev
from unyt import unyt_array
from operator import attrgetter
import yt

class COMs:
    splunit = 1
    def __init__(self,lst=None):
        self.coms = SortedKeyList(key=attrgetter('time'))
        self.spldict = {}
        if lst is not None:
            if not getattr(lst[0],'__is_COM__',False):
                lst = [COM(l) for l in (lst.piter() if hasattr(lst,'piter') else iter(lst))]
            self.coms.update(lst)
            self.make_splines()
        
    def copy_from(self,old):
        self.coms = old.coms
        self.spldict = old.spldict
        self.splunit = old.splunit

    def make_splines(self):
        com0 = self.coms[0]
        t = unyt_array([com.time for com in self.coms])
        ptypes = com0.ptype_list
        for ptype in ptypes:
            xyz = unyt_array([getattr(com,ptype) for com in self.coms])
            # need to remove nan values before interpolating
            # We will assume a nan in x corresponds to nans in y, z
            # while this might not be true, it should be considered true
            # as otherwise doesn't make sense physically
            inds = np.ravel(np.argwhere(np.logical_not(np.isnan(xyz[:,0]))))
            xyz = xyz[inds,:]
            # note that t is shared, so we don't want resize it
            x,y,z = xyz.v[:,0],xyz.v[:,1],xyz.v[:,2]
            self.splunit = xyz.units
            try:
                #Note - s=0 is necessary to force interpolation. Since the data
                # is accurate, we don't actually want any smoothing
                self.spldict[ptype],_ = splprep([x,y,z],u=np.ravel(t[inds]),s=0)
                #self.spldict[ptype],_ = splprep([x,y,z],u=t,s=0)
            except ValueError:
                # Likely 1 or more points are duplicated (see e.g. 
                # https://stackoverflow.com/questions/47948453/scipy-interpolate-splprep-error-invalid-inputs)
                # We'll add a very small amount of noise to each point and try again
                x = x + np.random.random(np.shape(x))*100*np.finfo(np.float64).eps
                y = y + np.random.random(np.shape(y))*100*np.finfo(np.float64).eps
                z = z + np.random.random(np.shape(z))*100*np.finfo(np.float64).eps
                self.spldict[ptype],_ = splprep([x,y,z],u=np.ravel(t[inds]),s=0)
                #self.spldict[ptype],_ = splprep([x,y,z],u=t,s=0)

    def add(self,com):
        ''' Add CoM after initial creation. Since this recomputes the spline functions, it is _highly_ recommended to add multiple CoMs at once'''
        if isinstance(com,Iterable):
            self.coms.update(com)
        else:
            self.coms.add(com)
        self.make_splines()

    def __iter__(self):
        return iter(self.coms)
    
    def __getitem__(self,item):
        return self.coms[item]

    def __len__(self):
        return len(self.coms)
    
    def get_bulk(self,time=None,*,ptype="all"):
        ''' 
        Get bulk position as a spline interpolation of time. 
        If time is None, return all. Can specify to only use ptype CoM
        '''
        if time is None:
            time = unyt_array([com.time for com in self.coms])
        return np.transpose(unyt_array(splev(time,self.spldict[ptype],),self.splunit))
    
    
    def get_bulk_velocity(self,time=None,*,ptype="all"):
        '''
        Get bulk velocity as the 3d derivative of the bulk position.
        If time is None, return all. Can specify to only use ptype CoM
        '''
        if time is None:
            time = unyt_array([com.time for com in self.coms])
        return np.transpose(unyt_array(splev(time,self.spldict[ptype],der=1),self.splunit/time.units))
        
    def plot(self,*,inspos=[-0.2, -0.17, 0.15, 0.12]):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot()
        xl = []
        yl = []
        ls = ["solid","dashed","dotted","dashdot"]
        lw = [3,1,1,1]
        t = unyt_array([com.time for com in self.coms])
        for ind,ptype in enumerate(["all","dm","bh","stars"]):
            if ptype not in self.spldict.keys():
                continue
            xyz = self.get_bulk(ptype=ptype)
            line=plot_color_varying_line(xyz[:,0],xyz[:,1],t,fig=fig,ax=ax,resize=True,label=f"{ptype} CoM")
            line.set_linewidth(lw[ind])
            line.set_linestyle(ls[ind])
            if ind==0:
                line.figure.colorbar(line).set_label('Time [Myr]')
            xl.append(line.axes.get_xlim())
            yl.append(line.axes.get_ylim())
        ax.set_xlabel('x (kpc)')
        ax.set_ylabel('y (kpc)')
        ax.legend()
        #print(yl)
        xl = np.array(xl)
        yl = np.array(yl)
        xls = np.transpose([sorted(xl[:,0],reverse=True),sorted(xl[:,1])])
        yls = np.transpose([sorted(yl[:,0],reverse=True),sorted(yl[:,1])])
        #print(yls)
        #print(yls[0])
        ax.set_xlim(xls[-1])
        ax.set_ylim(yls[-1])

        if inspos != -1:
            axins = ax.inset_axes(
                inspos,transform=ax.transData,
                xlim=xls[0], ylim=yls[0], xticklabels=[], yticklabels=[])
            
            for ind,ptype in enumerate(["all","dm","bh"]):
                xyz = self.get_bulk(ptype=ptype)
                line=plot_color_varying_line(xyz[:,0],xyz[:,1],t,fig=fig,ax=ax,resize=True,label=f"{ptype} CoM")
                line.set_linewidth(lw[ind])
                line.set_linestyle(ls[ind])
            ax.indicate_inset_zoom(axins, edgecolor="black")
        return fig,xl,yl

# %% [markdown]
# ## Various utility stuff

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Add particle_acceleration_[xyz]

# %%
from yt.units import centimeter as cm
from yt.units import second as s
from yt.utilities.exceptions import YTFieldNotFound 
def _particle_acceleration(field, data, ftype, axis, units = cm/s**2):
    return data[ftype,"Acceleration"][:,axis] * units

from functools import partial
def add_particle_acceleration(ds,ftype="PartType5",force_override=False):
    units = ds.unit_system["velocity"]/ds.unit_system["time"]
    for i,ax in enumerate("xyz"):
        ds.add_field(
            (ftype, f"particle_acceleration_{ax}"),
            function=partial(_particle_acceleration,ftype=ftype,axis=i,units=units),
            sampling_type="particle",
            display_name=f"a_{ax}",
            units=units,
            force_override=force_override,
        )

def add_all_acceleration(ds):
    #print('Adding accelerations')
    #print(f'Available particle types: {ds.particle_types}')
    for i in range(6):
        pt = f'PartType{i}'
        if pt not in ds.particle_types:
            continue
        #print(f'Adding acceleration for {pt}')
        add_particle_acceleration(ds,ftype=pt)
        
# I cannot get the following to work. As far as I can tell, I'm using validators
# correctly 
# add as new field going forward
#from yt.fields.derived_field import ValidateDataField
#units = cm/s**2
#for ftype in ['PartType1','PartType2','PartType4','PartType5']:
#    for i,ax in enumerate("xyz"):
#        yt.add_field(
#            (ftype, f"particle_acceleration_{ax}"),
#            function=partial(_particle_acceleration,ftype=ftype,axis=i,units=units),
#            sampling_type="particle",
#            display_name=f"a_{ax}",
#            units=units,
#            validators=[ValidateDataField((ftype,"Acceleration"))],
#            force_override=True,
#        )


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Remove bulk coords/velocity - Deprecated - use COM stuff instead

# %%
from unyt.array import unyt_array
from unyt import megayear as Myr
#@deprecated('Use the COM(s) structures instead')
def remove_bulk_xyz(xo,yo,zo,vxo,vyo,vzo,t):
    # This is a time-based removal. x,y,z,vx,vy,vz are expected to be x=x(t),y=y(t),etc
    # Doing transposes to get the broadcasting correctly
    x = (xo[:,0:2].T - xo[:,2].T).T
    y = (yo[:,0:2].T - yo[:,2].T).T
    z = (zo[:,0:2].T - zo[:,2].T).T
    lun = x.units
    if t is None:
        tun = Myr
    else:
        tun = t.units
    # remove bulk velocity
    com = unyt_array([xo[:,2],yo[:,2],zo[:,2]]).T
    dcom = (np.gradient(com.v,t.v,axis=0) * lun/tun)
    vx = (vxo.T - dcom[:,0].T).T
    vy = (vyo.T - dcom[:,1].T).T
    vz = (vzo.T - dcom[:,2].T).T
    return x,y,z,vx,vy,vz,

#@deprecated('Use the COM(s) structures instead')
def remove_bulk_coords(coords,vels,weights=None):
    # Assume coords,vels are Nx3
    if weights is None:
        weights = np.ones((len(coords),1))

    com = np.sum((weights.T*coords.T).T,axis=0)/np.sum(weights)
    dcom = np.sum(vels,axis=0)/len(vels)

    newcoords = coords - com
    newvels = vels - dcom
    return newcoords, newvels, 



# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Triaxiality computations

# %%
# get principal axes (stolen from Andrew Wetzel's utilities and modified)
# Note that for the triaxiality parameter T = (1-p^2)/(1-q^2), p=b/a, q=c/a, where c<=b<=a, 
# axis_ratios=[c/a,c/b,b/a], so [a,b,c] = [1,axis_ratios[2],axis_ratios[0]]
def get_principal_axes(position_vectors, weights=None, use_moi=False, verbose=True):
    '''
    Compute principal axes of input position_vectors (which should be wrt a center),
    defined via the moment of inertia tensor.
    Get reverse-sorted rotation_tensor and axis ratios of these principal axes.

    Parameters
    ----------
    position_vectors : array (object number x dimension number)
        position[s] or distance[s] wrt a center
    weights : array
        weight for each position (usually mass) - if None, assume all have same weight
    use_moi : bool
        whether to use the moment of inertia tensor, instead of the second moment of the
        mass distribution, forthe diagonal components input to get the rotation tensor
        this choice only affect the resultant axis ratios, not the resultant rotation tensor
    verbose : bool
        whether to print axis ratios

    Returns
    -------
    rotation_tensor : array
        max, med, min eigen-vectors that define the rotation tensor
    axis_ratios : array
        ratios of principal axes
    '''
    if weights is None or len(weights) == 0:
        weights = 1
    else:
        weights = weights / np.median(weights)

    if position_vectors.shape[1] == 3:
        # 3-D
        if use_moi:
            # use moment of inertia to define for diagonal terms
            xx = np.sum(weights * (position_vectors[:, 1] ** 2 + position_vectors[:, 2] ** 2))
            yy = np.sum(weights * (position_vectors[:, 0] ** 2 + position_vectors[:, 2] ** 2))
            zz = np.sum(weights * (position_vectors[:, 0] ** 2 + position_vectors[:, 1] ** 2))
            xy = yx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 1])
            xz = zx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 2])
            yz = zy = np.sum(weights * position_vectors[:, 1] * position_vectors[:, 2])

            moi_tensor = [[xx, -xy, -xz], [-yx, yy, -yz], [-zx, -zy, zz]]
        else:
            # default: use second moment of mass distribution for diagonal terms
            xx = np.sum(weights * position_vectors[:, 0] ** 2)
            yy = np.sum(weights * position_vectors[:, 1] ** 2)
            zz = np.sum(weights * position_vectors[:, 2] ** 2)
            xy = yx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 1])
            xz = zx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 2])
            yz = zy = np.sum(weights * position_vectors[:, 1] * position_vectors[:, 2])

            moi_tensor = [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]

    elif position_vectors.shape[1] == 2:
        # 2-D
        xx = np.sum(weights * position_vectors[:, 0] ** 2)
        yy = np.sum(weights * position_vectors[:, 1] ** 2)
        xy = yx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 1])

        moi_tensor = [[xx, xy], [yx, yy]]

    eigen_values, rotation_tensor = np.linalg.eig(moi_tensor)

    # order eigen-vectors by eigen-values, from largest to smallest
    eigen_indices_sorted = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[eigen_indices_sorted]
    # eigen_values /= eigen_values.max()  # renormalize to 1
    # make rotation_tensor[0, 1, 2] be eigen_vectors that correspond to eigen_values[0, 1, 2]
    rotation_tensor = rotation_tensor.transpose()[eigen_indices_sorted]
    # ensure that rotation tensor satisfies right-hand rule
    rotation_tensor[2] = np.cross(rotation_tensor[0], rotation_tensor[1])

    if position_vectors.shape[1] == 3:
        axis_ratios = np.sqrt(
            [
                eigen_values[2] / eigen_values[0],
                eigen_values[2] / eigen_values[1],
                eigen_values[1] / eigen_values[0],
            ]
        )

        if verbose:
            print(
                '* principal axes:  min/maj = {:.3f}, min/med = {:.3f}, med/maj = {:.3f}'.format(
                    axis_ratios[0], axis_ratios[1], axis_ratios[2]
                )
            )

    elif position_vectors.shape[1] == 2:
        axis_ratios = eigen_values[1] / eigen_values[0]

        if verbose:
            print('* principal axes:  min/maj = {:.3f}'.format(axis_ratios))

    return rotation_tensor, axis_ratios

from unyt import kiloparsec as kpc
def triax_vs_r(ds,*,radii=None,weights=None):
    if radii is None:
        radii = np.linspace(10,400,num=10)
    ad = ds.all_data()
    coords,vels = remove_bulk_coords(ad['all','Coordinates'],ad['all','Velocities'],weights=weights)
    rcoords = np.sqrt(np.sum(coords**2,axis=1))
    rotation_tensor = []
    axis_ratios = []
    for r in radii:
        inds = np.where(rcoords<r)
        rt, ar = get_principal_axes(coords[inds],weights=weights[inds])
        rotation_tensor.append(rt)
        axis_ratios.append(ar)
    axis_ratios = np.array(axis_ratios)
    return rotation_tensor,axis_ratios,radii


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### load_timeseries_from_folder

# %%
import yt
import numpy as np
import re
from pathlib import Path
from yt.data_objects.time_series import DatasetSeries
def load_timeseries_from_folder(foldername,*,num_snaps=None,log_level='warning',**kwargs):
    yt.set_log_level(log_level)
    # If foldername is a directory, convert to glob pattern
    if Path(foldername).is_dir():
        foldername = f"{foldername}/snapshot_*.hdf5"
    # The default dataset sorting doesn't do a very good job, it just sorts
    # lexographically on the entire filename. Thus snapshot_1000 comes *before*
    # snapshot_999. So instead of giving a glob pattern to yt.load, we'll sort
    # the file list first by snapshot number, using the snapshot index only
    # and then pass the file list to a new DatasetSeries constructor
    dsnamelist = DatasetSeries._get_filenames_from_glob_pattern(foldername)
    tsinds = np.argsort([int(re.search(r'snapshot_(\d+)',n)[1]) for n in dsnamelist])
    # Sample the given names if we only want a subset
    print(f'Sorted {len(dsnamelist)} snapshots')
    if num_snaps is not None:
        if isinstance(num_snaps,int):
            idxs = np.round(np.linspace(0,len(dsnamelist)-1,min(len(dsnamelist),num_snaps))).astype(int)
        else:
            idxs = num_snaps
        tsinds = tsinds[idxs].astype(int)
        print(f'Downsampled to {num_snaps if isinstance(num_snaps,int) else len(num_snaps)} snapshots')
        #print(f'{tsinds} {tsinds}')
    dsnamelist = [dsnamelist[i] for i in tsinds]
    ts = DatasetSeries(dsnamelist,**kwargs)
    print(f'Loaded {len(ts)} snapshots')
    return ts,(tsinds,dsnamelist)


# %% [markdown]
# ## Simulation and Run processing
# I expect this section will get much bigger over time, so this might be put into it's own file or something else. At least it'll probably move up to header 2.

# %%
import yt
from yt import DatasetSeries
from operator import attrgetter
from pathlib import Path
from collections import namedtuple
import dill as pickle

class FakeParticleTrajectory:
    indices = None
    _keys = None
    trajs = None
    def __init__(self,trajs):
        self.trajs = []
        for t in trajs:
            self.trajs.append(t)
        if len(trajs)>0:
            self._keys = trajs.keys()
            self.indices = trajs.indices
    
    def __iter__(self):
        return iter(self.trajs)
    
    def __getitem__(self,item):
        return self.trajs[item]
    
    def __len__(self):
        return len(self.trajs)
    
    def keys(self):
        return self._keys

class Simulation:
    folder = None
    ts = None
    trajs = None
    coms = None
    names = None
    inds = None
    
    def __init__(self,simname,*,force_reload=False,**kwargs):
        
        # check if run has already been processed
        self.folder = Simulation._remove_file_glob(Path(simname).expanduser())
        self.simdatafile = self.folder / "sim.bin"
        
        is_processed = self.simdatafile.exists()
        if not is_processed or force_reload:
            self.process_sim(**kwargs)
            self.save_sim(**kwargs)
        else:
            self.load_sim(**kwargs)
    
    @staticmethod
    def _remove_file_glob(path):
        if not path.is_dir():
            path = path.resolve().parent
        return path
    
    def simname(self):
        return self.folder.name
    
    def __repr__(self):
        return str(self.folder)
    
    def process_sim(self,*,unit_base=None,bounding_box=None,**kwargs):
        if unit_base is None:
            unit_base = {
                "length": (1.0, "kpc"),
                "velocity": (1.0, "km/s"),
                "mass": (1e10, "Msun"),
                "temperature": (1.0, "K"),
            }
        if bounding_box is None:
            bounding_box = [[-600, 600]] * 3
        
        print(f'Initial processing of simulation {self.folder}')
        # We want to load all available snapshots
        ts,(tsinds,dsnamelist) = load_timeseries_from_folder(self.folder,
                                                             num_snaps=None,
                                                             log_level='warning',
                                                             bounding_box=bounding_box,
                                                             unit_base=unit_base,
                                                             setup_function=add_all_acceleration,
                                                            **kwargs)
        print('Computing CoMs...',end='')
        coms = COMs(ts)
        print('done')
        print(f'Sim lasts for {signif(coms[-1].time.to("Myr").v,3)} Myr')

        ptype='PartType5'
        fields = [
            (ptype, "particle_position_x"),
            (ptype, "particle_position_y"),
            (ptype, "particle_position_z"),
            (ptype, "particle_velocity_x"),
            (ptype, "particle_velocity_y"),
            (ptype, "particle_velocity_z"),
            #(ptype, "particle_acceleration_x"),
            #(ptype, "particle_acceleration_y"),
            #(ptype, "particle_acceleration_z"),
            #(ptype, "Acceleration"),
            (ptype, "Masses"),
        ]
        ds = ts[0]
        # We expect indices to look something like [100001,200002], depending
        # on the number of particles. This way is more flexible, assuming the 
        # of BHs stays small
        # This section only works if BHs are present
        if (ptype,'ParticleIDs') in ds.field_list:
            indices = ds.r[ptype,'ParticleIDs'].v
            trajs = ts.particle_trajectories(indices, fields=fields, ptype=ptype)
            print(f'Loaded {len(trajs)} trajectories')
        else:
            trajs = []
            print(f'No BHs present')
        self.ts = ts
        self.coms = coms
        self.trajs = trajs
        self.inds = tsinds
        self.names = dsnamelist
        self.ptype = ptype
        # Saving these so that the ts can be reloaded in the same way if necessary
        self.bounding_box=bounding_box
        self.unit_base=unit_base

    def save_sim(self,**kwargs):
        print('Saving sim....',end='')
        # Need to save coms, trajectories, names, bounding_box, and unit_base
        # For now, we'll assume any other arguments to loading the ts can be
        # ignored. We're also not saving simdatafile, folder, or ts. The first
        # two are just paths, and the third can be quickly regenerated using
        # load_timeseries_from_folder
        outlist = ['coms','names','inds','bounding_box','unit_base','ptype']
        outdict = {k:v for k,v in vars(self).items() if k in outlist}
        # Note trajs is a particle_trajectories object, which contains
        # references to the originating time series. This is unpickleable 
        # and annoying.
        # So we'll need to save it out on its own
        trajs = FakeParticleTrajectory(self.trajs)
        outdict['trajs'] = trajs
        
        # Note we could save everything to dataframes and such, but that's 
        # complicated and we don't _really_ need it. So we'll just pickle 
        # everything instead
        with open(self.simdatafile,'wb') as simdatafile: 
            pickle.dump(outdict,simdatafile,**kwargs)
        print('done')
    
    def load_sim(self,**kwargs):
        print(f"Loading from already processed sim {self.simdatafile}")
        with open(self.simdatafile,'rb') as simdatafile:
            indict = pickle.load(simdatafile,**kwargs)
        for k,v in indict.items():
            setattr(self,k,v)
        # need to reload self.trajs in case we've redefined
        # FakeParticleTrajectory
        self.trajs = FakeParticleTrajectory(self.trajs)
        # we only load the original saved datasets here so that the COMs and 
        # trajs match
        self.ts,_ = load_timeseries_from_folder(self.folder,
                                              num_snaps=self.inds,
                                              log_level='warning',
                                              bounding_box=self.bounding_box,
                                              unit_base=self.unit_base,
                                              setup_function=add_all_acceleration,
                                              **kwargs) 
        print('Finished')
        
    def update_sim(self,*,save_sim=False,**kwargs):
        ''' Run this to add snapshots that weren't present on initial run '''
        print('Updating simulation')
        fullts,(allinds,allnames) = load_timeseries_from_folder(self.folder,
                                                                num_snaps=None,
                                                                log_level='warning',
                                                                bounding_box=self.bounding_box,
                                                                unit_base=self.unit_base,
                                                                setup_function=add_all_acceleration,
                                                                **kwargs)
        missing_inds = np.setdiff1d(allinds,self.inds)
        print(f'Missing {len(missing_inds)} snapshots')
        newts,(newinds,newnames) = load_timeseries_from_folder(self.folder,
                                                                num_snaps=missing_inds,
                                                                log_level='warning',
                                                                bounding_box=self.bounding_box,
                                                                unit_base=self.unit_base,
                                                                **kwargs)
        print(f"Adding {len(newts)} new snapshots")
        print(f"Generating new COMs...",end="")
        newcoms = COMs(newts)
        self.coms.add(newcoms)
        print(f"done")
        # We should be able to add just the new stuff in using newts.particle_trajectories, but for now
        # we'll just regenerate them. Note that this could take a *long* time.
        self.trajs = fullts.particle_trajectories(self.trajs.indices, 
                                         fields=self.trajs.keys(), 
                                         ptype=self.ptype)
        print(f'Loaded {len(self.trajs)} trajectories')
        self.ts = fullts
        self.names = allnames
        self.inds = allinds
        if save_sim:
            self.save_sim()
        else:
            print(f"Sim updated. Don't forget to save it!")
    
    def _read_param_file(self):
        import re
        file = sim.folder/Path(f'{sim.simname()}.params')
        paramfile = file.read_text()
        lines = re.split('\n+',paramfile)
        params = {}
        for line in lines:
            if line.startswith('%') or line.startswith('#'):
                continue
            line = re.sub(r'\%.*','',line)
            if len(line)==0:
                continue
            m = re.match(r'(\w+)\s+(\S+)\s*',line)
            params[m.group(1)] = m.group(2)
        return params
    
    def get_parameter(self,param):
        if not hasattr(self,'_params'):
            self._params = self._read_param_file()
        return self._params[param]

    def downsample_ts(self,num_snaps):
        idxs = np.round(np.linspace(0,len(self.names)-1,min(len(self.names),num_snaps))).astype(int)
        inds = self.inds[idxs]
        # not clear if I should use inds or just idxs directly
        ts = [self.ts[i] for i in inds]
        return ts
    
    def get_index_from_time(self,time):
        ''' Return our best guess of the index into ts/coms/trajs corresponding to a particular time'''
        # We will probably change how this is calculated in the future, but the
        # behaviour should be the same
        times = [com.time for com in self.coms]
        
        # get the closest index for each time
        try: # I dislike EAFP but apparently unyt_quantity is an Iterable
            ind = np.zeros_like(time)
            for i,t in enumerate(time):
                ind[i] = np.argmin(np.abs(times-t))
        except TypeError:
            ind = np.argmin(np.abs(times-time))
        return ind
    
    def plotds(self,ds=None,*,width=(300,'kpc'),**kwargs):
        if ds is None:
            ds = self.ts[-1]
        if isinstance(ds,int):
            ds = self.ts[ds]
        col_field = ("PartType1","Masses")
        ad = ds.all_data() # make sure the data is loaded and particle fields are populated
        hrpt = 'PartType1' if 'PartType2' in ds.particle_types else 'PartType4'
        lrpt = 'PartType2' if 'PartType2' in ds.particle_types else 'PartType1'
        plot = yt.ParticleProjectionPlot(ds,"z",(hrpt,'particle_ones'),col_field,width=width,window_size=(3,3),origin='native',**kwargs)
        if ('PartType5','Coordinates') in ds.field_list:
            plot.annotate_particles(20,ptype="PartType5",col="orange",p_size=10,alpha=0.75)
        if (lrpt,'Coordinates') in ds.field_list:
            plot.annotate_particles(20,ptype=lrpt,col='red',p_size=10,alpha=0.1)
        plot.annotate_timestamp(time_unit="Myr",draw_inset_box=True)
        return plot,col_field,None
    
    def _get_frame_inds(self,*,num_frames=20,tsinds=None,**kwargs):
        ts = self.ts
        idxs = np.linspace(0,len(ts)-1,num=min(num_frames,len(ts))).astype(int)
        return idxs
    
    @staticmethod
    def _gen_animation_name(figpath,idxs):
        return Path(f'{figpath}_f{idxs[-1]}_n{len(idxs)}.gif')
    
    def make_animation(self,*,figpath=None,tsinds=None,**kwargs):
        ts = self.ts
        idxs = self._get_frame_inds(**kwargs)
        if tsinds is None:
            ts = [ts[i] for i in idxs]
        else:
            ts = [ts[tsinds[i]] for i in idxs]
        if figpath is None:
            figpath = (Path.home() / Path(f'storage/figures/darknanograv/{self.simname()}'))
            figpath = figpath.expanduser().resolve()
            figpath = Simulation._gen_animation_name(figpath,idxs)
            print(f'Creating new animation: {figpath}')
        ani = make_animation_from_images(ts,partial(self.plotds,**kwargs),figpath)
        return ani
    
    def animate(self,*,figpath=None,overwrite=False,**kwargs):
        if figpath is None:
            figpath = (Path.home() / Path(f'storage/figures/darknanograv/{self.simname()}'))
        idxs = self._get_frame_inds(**kwargs)
        figpath = Simulation._gen_animation_name(figpath,idxs)
        if figpath.exists() and not overwrite:
            print(f'Loading animation from file: {figpath}')
            ani = load_animation_from_file(figpath)
        else:
            print(f'Creating animation {figpath}')
            ani = self.make_animation(figpath=figpath,**kwargs)
        return ani
        
    def prof_ds(self,dsind,*,ax=None,**kwargs):
        raise NotImplementedError('Still under construction')
        newfig = False
        if ax is None:
            fig = plt.figure(**kwargs)
            ax = fig.add_subplot()
            newfig = True
        ds = self.ts[dsind]
        ad = ds.all_data()
        hrpt = 'PartType1' if 'PartType2' in ds.particle_types else 'PartType4'
        lrpt = 'PartType2' if 'PartType2' in ds.particle_types else 'PartType1'
        rhohr,(prof,npart) = rho_prof(ds=ds,radius=(500,'kpc'),stretch='nan',pt=hrpt)
        
        
    def make_density_prof(self,*,ax=None,num_lines=5,center=None,
                          save_fig=False,figpath=None,
                          **kwargs):
        newfig = False
        if ax is None:
            fig = plt.figure(**kwargs)
            ax = fig.add_subplot()
            newfig = True
        # TODO: need to figure out center stuff
        if center is None:
            center='all'
        ts = self.downsample_ts(num_lines)
        for i,ds in enumerate(ts):
            ds.all_data()
            time = ds.current_time
            print(f'Looking at ds={Path(ds.filename).stem} at {time.v=} Myr',end='')
            hrpt = 'PartType1' if 'PartType2' in ds.particle_types else 'PartType4'
            lrpt = 'PartType2' if 'PartType2' in ds.particle_types else 'PartType1'
            c = self.coms.get_bulk(time=time,ptype=center).to('kpc')
            sph = ds.sphere(center=c,radius=(1,'Mpc'))
            print(f' centered at {sph.center}')
            rhohr,(prof,npart) = rho_prof(sphere=sph,stretch='nan',pt=hrpt)
            rhohr = rhohr.to("code_mass/kpc**3")
            r = prof.x.to("pc")
            hr, = ax.loglog(r,rhohr,'.-',label=f"t={signif(time.to('Myr'),2)} HR DM ")
            if i==0:
                r0 = r
            else:
                r0 = np.unique([*r0,*r])*r.units
            
            rholr,(prof,npart) = rho_prof(sphere=sph,stretch='nan',pt=lrpt)
            rholr = rholr.to("code_mass/kpc**3")
            rlr = prof.x.to("pc")
            r0 = np.unique([*r0,*rlr]) * rlr.units
            ax.loglog(rlr,rholr,'.',markersize=2,label=f"_Lo-res DM",color=hr.get_color())

        rho_αβγ130 = get_αβγ_prof(r0.to('kpc').v,)
        rho_αβγNFW = get_αβγ_prof(r0.to('kpc').v,γ=0)
        ax.loglog(r0, rho_αβγ130,label=f'({1},{3},{0})')
        ax.loglog(r0, rho_αβγNFW,label=f'({1},{3},{1})')
        mass_unit = ds.mass_unit.to("Msun")
        mus = latex_float(mass_unit)
        ax.set_xlabel(f"r ({r.units})")
        ax.set_ylabel(f'ρ$(r)$ (${mus}$' r' M$_{{\odot}}$/kpc$^3$)')
        ax.legend()
        #handles, labels = ax1.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='outside right')
        #fig.tight_layout()
        #fig.subplots_adjust(wspace=0.6*4)
        if save_fig:
            if figpath is None:
                figpath = (Path.home() / Path(f'storage/figures/darknanograv/{self.simname()}_dens_prof'))
                tend = signif(self.ts[-1].current_time.to('Myr'),2).astype(int)
                figpath = Path(f'{figpath}_tf{tend}.pdf')
            fig = ax.get_figure()
            fig.savefig(figpath,bbox_inches='tight')
            return fig
        if newfig:
            return ax


# %%

# %% [markdown] toc-hr-collapsed=true
# # Testing

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Verifying gizmo/gadget equivalence

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Generate test file

# %%
so = SphericOptions(MBH=1e-2*0,dx=-5,dvx=1,ogb=True,Nhalo=1e4,ogh=True)
print(f"Using spheric options: {so.generateOptionString()}")
comproc = spheric(so)
print(comproc.stdout.decode())
print(comproc.stderr.decode())
#./spheric -ogb -ogr -opfs -halo -Nhalo 500000 -Mhalo 0.15 -a 1 -b 3 -c 1 -rs 1.18 -rcutoff 118.0 -name p5e5_m1e9_vmax24_rc100rs

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Load in IC file using yt

# %%
import yt

ic = yt.load("runs/IC-gizmo.hdf5")

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

ad = ic.all_data()
x = ad["PartType1","particle_position_x"]
y = ad["PartType1","particle_position_y"]
z = ad["PartType1","particle_position_z"]

try:
    sx = ad["PartType4","particle_position_x"]
    sy = ad["PartType4","particle_position_y"]
    sz = ad["PartType4","particle_position_z"]
except:
    sx = None
    sy = None
    sz = None

try:
    bhx = ad["PartType5","particle_position_x"]
    bhy = ad["PartType5","particle_position_y"]
    bhz = ad["PartType5","particle_position_z"]
except:
    bhx = None
    bhy = None
    bhz = None

fig = plt.figure(figsize=(12,3))
ax = fig.add_subplot(1,3,1)
ax.plot(x,y,'.')
if sx is not None:
    ax.plot(sx,sy,'.')
if bhx is not None:
    ax.plot(bhx,bhy,'.')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax = fig.add_subplot(1,3,2)
ax.plot(x,z,'.')
if sx is not None:
    ax.plot(sx,sz,'.')
if bhx is not None:
    ax.plot(bhx,bhy,'.')
ax.set_xlabel("x")
ax.set_ylabel("z")
ax = fig.add_subplot(1,3,3)
ax.plot(z,y,'.')
if sx is not None:
    ax.plot(sz,sy,'.')
if bhx is not None:
    ax.plot(bhx,bhy,'.')
ax.set_xlabel("z")
ax.set_ylabel("y")
fig.subplots_adjust(wspace=0.4)

# %%
print(ad.quantities.center_of_mass(use_gas=False,use_particles=True))
print(ad.quantities.center_of_mass(use_gas=False,use_particles=True,particle_type="PartType1"))
ad["PartType5","Coordinates"]

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Verifying gizmo/gadget equivalence

# %%
import yt

ds = yt.load("../../gizmo-public/output/spheric_test_gizmo/snapshot_000.hdf5",bounding_box=[[-300,300]]*3)

plot = yt.ParticleProjectionPlot(ds,"z",("PartType1","Masses"),window_size=(4,4))
#plot.annotate_particles(20,ptype="PartType5",col="orange",p_size=25)
plot.show()

# %%
import yt

ds = yt.load("../../gizmo-public/output/spheric_test_gadget/snapshot_000.hdf5",bounding_box=[[-300,300]]*3)

plot = yt.ParticleProjectionPlot(ds,"z",("PartType1","Masses"),window_size=(4,4))
#plot.annotate_particles(20,ptype="PartType5",col="orange",p_size=25)
plot.show()

# %%
import yt

ds_giz = yt.load("../../gizmo-public/output/spheric_test_gizmo/snapshot_000.hdf5",bounding_box=[[-300,300]]*3)
ds_gad = yt.load("../../gizmo-public/output/spheric_test_gadget/snapshot_000.hdf5",bounding_box=[[-300,300]]*3)

ad_giz = ds_giz.all_data()
ad_gad = ds_gad.all_data()

print(min(ad_giz["PartType1","ParticleIDs"]))
print(min(ad_gad["PartType1","ParticleIDs"]))

# %%
# Note for some reason the gadget particle ids are like logarithmically distributed. I have no idea why
print(ad_giz["PartType1","ParticleIDs"][0:4])
print(ad_gad["PartType1","ParticleIDs"][0:4]-1065353216)
gizpid = ad_giz["PartType1","ParticleIDs"]
gadpid = ad_gad["PartType1","ParticleIDs"]
giz1 = np.argwhere(gizpid==min(gizpid))[0][0]
gad1 = np.argwhere(gadpid==min(gadpid))[0][0]
print(f"{giz1=} {gad1=}")
gizpx = ad_giz['PartType1','particle_position_x']
gadpx = ad_gad['PartType1','particle_position_x']
print(f"{gizpx[giz1]} {gadpx[gad1]}")
gizvx = ad_giz['PartType1','particle_velocity_x']
gadvx = ad_gad['PartType1','particle_velocity_x']
gizc = ad_giz['PartType1','Coordinates']
gadc = ad_gad['PartType1','Coordinates']
gizv = ad_giz['PartType1','Velocities']
gadv = ad_gad['PartType1','Velocities']
print(f"{gizvx[giz1]} {gadvx[gad1]}")

idxgiz = np.argsort(gizpid)
idxgad = np.argsort(gadpid)

def dispvecmag(v1,v2,s=1):
    dv = [v1[i]-s*v2[i] for i,_ in enumerate(v1)]
    dvm = [v*v for v in dv]
    return sum(dvm)

posdiff = np.any(np.abs(gizpx[idxgiz]-gadpx[idxgad])>0)
veldiff = np.any(np.abs(gizvx[idxgiz]-gadvx[idxgad])>0)
print(f"Any x-position differences: {posdiff}")
print(f"Any velocity differences: {veldiff}")

if posdiff:
    plt.plot(gizpx[idxgiz]-gadpx[idxgad],'.')
if veldiff:
    dvm = [dispvecmag(gizv[idxgiz[i]],gadv[idxgad[i]]) for i,_ in enumerate(idxgiz)]
    plt.plot(dvm,'.')


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Testing bulk movement

# %% [markdown]
# ### Generate test file

# %%
so = SphericOptions(MBH=1e-2,dx=-5,dvx=10,name="runs/IC-testmove",Nhalo=1e4,ogh=True)
print(f"Using spheric options: {so.generateOptionString()}")
comproc = spheric(so)
print(comproc.stdout.decode())
print(comproc.stderr.decode())

# %% [markdown]
# ### Testing movement
# We'll look at the center of mass of the different snapshots. It should be moving at $10 \text{ km/s} = 0.0102 \text{ kpc/Myr}$ in the x direction. Since the final snapshot occurs at 167 Myr, it should have moved ~1.67 kpc.

# %%
import yt

yt.set_log_level('warning')
ts = yt.load("../../gizmo-public/output/testmove/snapshot_???.hdf5")
print(len(ts))

# %%
com = []
t = []
for ds in ts:
    ad = ds.all_data()
    t.append(ds.current_time.to("Myr"))
    com.append(ad.quantities.center_of_mass(use_gas=False,use_particles=True,particle_type='PartType1').to('kpc'))
    #print(f"t={t[-1]} COM:{com[-1]}")

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import unyt

x = [c[0] for c in com]
y = [c[1] for c in com]
z = [c[2] for c in com]

#with matplotlib_support: # don't have newest unyt yet
fig = plt.figure(figsize=(12,3))
ax = fig.add_subplot(1,3,1)
plot = ax.scatter(x,y,20,t)
ax = fig.add_subplot(1,3,2)
plot = ax.scatter(x,z,20,t)
ax = fig.add_subplot(1,3,3)
plot = ax.scatter(y,z,20,t)
plt.colorbar(plot)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Testing core profile random walk

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Generate test file
# Need a cored profile, not NFW/cusp. We'll use $(\alpha,\beta,\gamma)=(1,3,0)$ (approximating from [Lazar 2020](https://doi.org/10.1093/mnras/staa2101)). 

# %% jupyter={"outputs_hidden": true}
so = SphericOptions(MBH=1e-4,dx=0,dvx=0,name="runs/IC-randomNFW",Nhalo=1e5,ogh=True)
so130 = SphericOptions(MBH=1e-4,dx=0,dvx=0,name="runs/IC-random130",beta=3,gamma=0,Nhalo=1e4,ogh=True)
print(f"Using spheric options: {so.generateOptionString()}")
comproc = spheric(so)
# Note if this errors/is blank, spheric probably segfaulted
print(comproc.stdout.decode()) 
print(comproc.stderr.decode())

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Density Profile

# %%

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
import matplotlib as mpl
import matplotlib.pyplot as plt

unit_base = {
            "length": (1.0, "kpc"),
            "velocity": (1.0, "km/s"),
            "mass": (1e10, "Msun"),
            "temperature": (1.0, "K"),
        }

ds = yt.load("../../gizmo-public/output/random/snapshot_000.hdf5",unit_base=unit_base,bounding_box=[[-300,300]]*3)

sph = get_sphere(ds=ds,radius=(300,"kpc"),center=([0,0,0],"kpc"))
rhodm,(prof,npart) = rho_prof(sphere=sph,stretch=False)
rhodm = rhodm.to("code_mass/kpc**3")

r = prof.x.to("kpc")

rho_αβγ = get_αβγ_prof(r.v,sphereopts=so)

fig = plt.figure()
ax = fig.add_subplot()
ax.loglog(r,r**0 * rhodm,'.-',label="Gizmo")
ax.loglog(r,r**0 * rho_αβγ,label=f'({so.alpha},{so.beta},{so.gamma})')
ax.set_xlabel(f"r (kpc)")
ax.set_ylabel(r"$r^2 \rho(r)$ ($10^{10}$ M$_{\odot}$/kpc)")
ax.legend()


# %% jupyter={"outputs_hidden": true, "source_hidden": true}
unit_base = {
            "length": (1.0, "kpc"),
            "velocity": (1.0, "km/s"),
            "mass": (1e10, "Msun"),
            "temperature": (1.0, "K"),
        }
ds = yt.load("../../gizmo-public/output/random130/snapshot_000.hdf5",unit_base=unit_base,bounding_box=[[-300,300]]*3)

sph = get_sphere(ds=ds,radius=(300,"kpc"),center=([-0,0,0],"kpc"))
rhodm,(prof,npart) = rho_prof(sphere=sph,stretch=True)
rhodm = rhodm.to("code_mass/kpc**3")

r = prof.x.to("kpc")

rho_αβγ = get_αβγ_prof(r.v,sphereopts=so130)

fig = plt.figure()
ax = fig.add_subplot()
ax.loglog(r,r**0 * rhodm,'.-',label="Gizmo")
ax.loglog(r,r**0 * rho_αβγ,label=f'({so130.alpha},{so130.beta},{so130.gamma})')
ax.set_xlabel(f"r (kpc)")
ax.set_ylabel(r"$r^2 \rho(r)$ ($10^{10}$ M$_{\odot}$/kpc)")
ax.legend()

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
from operator import attrgetter
unit_base = {
            "length": (1.0, "kpc"),
            "velocity": (1.0, "km/s"),
            "mass": (1e10, "Msun"),
            "temperature": (1.0, "K"),
        }
ts = yt.load('../../gizmo-public/output/randomNFW/snapshot_*.hdf5',unit_base=unit_base,bounding_box=[[-300,300]]*3)
idxs = np.round(np.linspace(0,len(ts)-1,5)).astype(int)
ts = sorted([ts[i] for i in idxs],key=attrgetter('current_time'))


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
r0 = []
for ix,ds in enumerate(ts):
    sph = get_sphere(ds=ds,radius=(300,"kpc"),
                     #center=([-0,0,0],"kpc"),
                     center=None,
                    )
    rhodm,(prof,npart) = rho_prof(sphere=sph,stretch=False)
    rhodm = rhodm.to("code_mass/kpc**3")

    r = prof.x.to("kpc")
    if ix==0:
        r0 = r
    rho130 = get_αβγ_prof(r0.v,sphereopts=so130)
    rhoNFW = get_αβγ_prof(r0.v,sphereopts=so)
    ax.loglog(r,rhodm,'.-',label=f"{ds.current_time.to('Myr'):.4g}")
    ax2.loglog(r,r**2 * rhodm,'.-',label=f"{ds.current_time.to('Myr'):.4g}")
    ax3.loglog(r,rhodm/rhoNFW,label=f"{ds.current_time.to('Myr'):.4g}")
    ax4.loglog(r,rhodm/rho130,label=f"{ds.current_time.to('Myr'):.4g}")
        
rho_αβγ130 = get_αβγ_prof(r0.v,sphereopts=so130)
rho_αβγNFW = get_αβγ_prof(r0.v,sphereopts=so)

ax.loglog(r0, rho_αβγ130,label=f'({so130.alpha},{so130.beta},{so130.gamma})')
ax.loglog(r0, rho_αβγNFW,label=f'({so.alpha},{so.beta},{so.gamma})')
ax2.loglog(r0,r0**2 * rho_αβγ130,label=f'({so130.alpha},{so130.beta},{so130.gamma})')
ax2.loglog(r0,r0**2 * rho_αβγNFW,label=f'({so.alpha},{so.beta},{so.gamma})')
ax.set_xlabel(f"r (kpc)")
ax.set_ylabel(r"$\rho(r)$ ($10^{10}$ M$_{\odot}$/kpc)")
ax.legend()
ax2.set_xlabel(f"r (kpc)")
ax2.set_ylabel(r"$r^2 \rho(r)$ ($10^{10}$ M$_{\odot}\cdot$kpc)")
ax2.legend()
ax3.set_xlabel(f"r (kpc)")
ax3.set_ylabel(r"$\rho(r)/\rho_{NFW}$")
ax3.legend()
ax4.set_xlabel(f"r (kpc)")
ax4.set_ylabel(r"$\rho(r)/\rho_{130}$")
ax4.legend()

# %% [markdown]
# ### Test random walk

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Plot random walks

# %%
from unyt import kiloparsec as kpc
from unyt import megayear as Myr
unit_base = {
            "length": (1.0, "kpc"),
            "velocity": (1.0, "km/s"),
            "mass": (1e10, "Msun"),
            "temperature": (1.0, "K"),
        }
ts = yt.load('../../gizmo-public/output/random130/snapshot_???.hdf5',unit_base=unit_base,bounding_box=[[-300,300]]*3)
x = np.zeros((len(ts),3)) * kpc
y = np.zeros((len(ts),3)) * kpc
z = np.zeros((len(ts),3)) * kpc
t = np.zeros(len(ts)) * Myr
for ix,ds in enumerate(ts):
    ad = ds.all_data()
    t[ix] = ds.current_time.to("Myr")
    comAll = ad.quantities.center_of_mass(use_gas=False,use_particles=True,)
    comDM = ad.quantities.center_of_mass(use_gas=False,use_particles=True,particle_type="PartType1")
    comBH = ad.quantities.center_of_mass(use_gas=False,use_particles=True,particle_type="PartType5")
    x[ix,:] = [comAll[0],comDM[0],comBH[0]]
    y[ix,:] = [comAll[1],comDM[1],comBH[1]]
    z[ix,:] = [comAll[2],comDM[2],comBH[2]]

fig = plot_random_walk(x,y,t)
fig.gca().set_title(r"$(1,3,0)$ profile")

# %%
import yt
from unyt.array import unyt_array
from unyt import kiloparsec as kpc
from unyt import megayear as Myr
unit_base = {
            "length": (1.0, "kpc"),
            "velocity": (1.0, "km/s"),
            "mass": (1e10, "Msun"),
            "temperature": (1.0, "K"),
        }
yt.set_log_level('warning')
ts = yt.load('../../runs/randomNFW/outputs/snapshot_???.hdf5',unit_base=unit_base,bounding_box=[[-300,300]]*3)
x = np.zeros((len(ts),3)) * kpc
y = np.zeros((len(ts),3)) * kpc
z = np.zeros((len(ts),3)) * kpc
t = np.zeros(len(ts)) * Myr
for ix,ds in enumerate(ts):
    ad = ds.all_data()
    t[ix] = ds.current_time.to("Myr")
    comAll = ad.quantities.center_of_mass(use_gas=False,use_particles=True,)
    comDM = ad.quantities.center_of_mass(use_gas=False,use_particles=True,particle_type="PartType1")
    comBH = ad.quantities.center_of_mass(use_gas=False,use_particles=True,particle_type="PartType5")
    x[ix,:] = [comAll[0],comDM[0],comBH[0]]
    y[ix,:] = [comAll[1],comDM[1],comBH[1]]
    z[ix,:] = [comAll[2],comDM[2],comBH[2]]

fig = plot_random_walk(x,y,t,inspos=-1)
fig.gca().set_title(r"NFW profile")


# %% [markdown]
# ### Animate halo
# Might be worth trying to track individual particles

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
def plotHalo(ds,*,width=(300,'kpc'),**kwargs):
    col_field = ("PartType1","Masses")
    plot = yt.ParticleProjectionPlot(ds,"z",col_field,width=width,window_size=(3,3),origin='native',**kwargs)
    plot.annotate_particles(20,ptype="PartType5",col="orange",p_size=10,alpha=0.75)
    plot.annotate_timestamp(time_unit="Myr",draw_inset_box=True)
    plot.set_zlim(col_field,1e-5,5e-4)
    return plot,col_field,None

plot,_,_ = plotHalo(ts[4],width=(.5,'kpc'),center=([-0.2,0,0],'kpc'))
plot.show()


# %% jupyter={"outputs_hidden": true, "source_hidden": true}
def animate_halo(ts,*,num_frames=5,filename,plot_function=plotHalo,**kwargs):
    idxs = np.linspace(0,len(ts)-1,num=num_frames).astype(int)
    ts = [ts[i] for i in idxs]
    ani = make_animation_from_images(ts,lambda ds:plot_function(ds,**kwargs),filename)
    return ani

animate_halo(ts,num_frames=20,width=(20,'kpc'),center=([0,0,0],'kpc'),filename='../../figures/nanograv/randomNFW.gif')

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Track particles

# %%
import yt
from yt import DatasetSeries
from operator import attrgetter
unit_base = {
            "length": (1.0, "kpc"),
            "velocity": (1.0, "km/s"),
            "mass": (1e10, "Msun"),
            "temperature": (1.0, "K"),
        }
yt.set_log_level('warning')
ts = yt.load('../../runs/IgorTest/outputs/snapshot_*.hdf5',unit_base=unit_base,bounding_box=[[-300,300]]*3)
idxs = np.round(np.linspace(0,len(ts)-1,min(len(ts),20))).astype(int)
ts = sorted([ts[i] for i in idxs],key=attrgetter('current_time'))
ts = DatasetSeries(ts)
coms = COMs(ts)

fields = [
    ("all", "particle_position_x"),
    ("all", "particle_position_y"),
    ("all", "particle_position_z"),
    ("all", "particle_velocity_x"),
    ("all", "particle_velocity_y"),
    ("all", "particle_velocity_z"),
]
ds = ts[0]
init_sphere = ds.sphere(coms.get_bulk(ds.current_time,ptype='bh'), (.1, "kpc"))
indices = init_sphere[("PartType1", "particle_index")].astype("int64")
trajs = ts.particle_trajectories(indices, fields=fields)
print(f'Loaded {len(trajs)} trajectories')

# %%
fig = plt.figure(figsize=(7,7))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
rfinal = np.zeros(len(trajs))
for ind,t in enumerate(trajs):
    time = t["particle_time"]
    bp = coms.get_bulk(time,ptype='bh')
    x = t[('all','particle_position_x')].to('kpc')-bp[:,0]
    y = t[('all','particle_position_y')].to('kpc')-bp[:,1]
    z = t[('all','particle_position_z')].to('kpc')-bp[:,2]
    r = np.sqrt(x**2+y**2+z**2)
    inds = np.argsort(time)
    time = time[inds]
    r = r[inds]
    x = x[inds]
    y = y[inds]
    z = z[inds]
    rfinal[ind] = r[-1]
    ax1.plot(x,y)
    ax2.plot(time.to('Myr'),r,label=f'id:{t["particle_index"].v}')
ax1.set_xlabel('x (kpc)')
ax1.set_ylabel('y (kpc)')
ax2.set_xlabel('t (Myr)')
ax2.set_ylabel('r (kpc)')
if len(trajs)<7:
    ax2.legend()
#fig.subplots_adjust(wspace=0.2)
ax3 = fig.add_subplot(2,2,(3,4))
ax3.hist(rfinal,bins=np.geomspace(rfinal.min(),rfinal.max(),10),density=False)
ax3.set_xscale('log')
ax3.set_xlabel('r (kpc)')
ax3.set_ylabel('Counts')

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Generate combined test data

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### basic halo

# %%
from pathlib import Path
folder = Path("~/projects/spheric/runs").expanduser()
so1 = SphericOptions(MBH=1e-3,dx=0,dy=0,name=f"{folder}/IC-sideA",Nhalo=1e5,ogh=True)
so2 = SphericOptions(MBH=.5e-3,dx=50,dy=-10,dvx=-10,name=f"{folder}/IC-sideB",Nhalo=1e5,ogh=True)
print(f"Using for halo 1: {so1.generateOptionString()}")
print(f"Using for halo 2: {so2.generateOptionString()}")
comproc = spheric(so1)
# Note if this errors/is blank, spheric probably segfaulted
print(comproc.stdout.decode()) 
print(comproc.stderr.decode())
comproc = spheric(so2)
# Note if this errors/is blank, spheric probably segfaulted
print(comproc.stdout.decode()) 
print(comproc.stderr.decode())

# %% [markdown]
# ### 2-component halo
#
# Basic idea is:
# 1. Generate a single 2-component halo with same overall properties as single basic halo
# 2. Let evolve for ~1 Gyr
# 3. Check properties (density profile, triaxiality, etc)
# 4. Merge evolved halos as normal

# %% [markdown]
# Some notes:
# According to [1402.0005](https://arxiv.org/abs/1402.0005) (Shapiro's paper on cusps), the BH eats any DM particle at $r<4M=\frac{4GM}{c^2}\sim10^{-5}\text{ pc }\left(\frac{M}{10^8 M_{\odot}}\right)$ and any star with radius $R$, mass $m$ at $r<R(M/m)^{1/3}$

# %% [markdown]
# #### Generate a single halo first
# To generate a two component single halo, we need to figure out some things: 
# 1. Number of low-res and high-res DM particles - $10^4$ and $10^5$?
# 2. `Mstar`- $\int_0^{r_i}4\pi r^2\rho_{\text{NFW}+\text{spike}}(r)dr$
# 3. $r_s$ and $r_{cutoff}$ - use values from analytic: 80 pc=0.08 LU and 10 kpc=10 LU

# %%
from yt.units import parsec as pc
from yt.units import kiloparsec as kpc
from yt.units import Msun
Nlow = 1e4
Nhigh = 2e5
# Since get_αβγ_prof defaults to (1,3,1)=NFW (and cored is just (1,3,0)), we only need to specify α, β, γ for the high res stuff
α = 1
β = 4 # β>3 is a finite mass model
γ = 2 # This can vary from 0 (core) to <3 (divergent)
rs = 80 * pc
rcut = 1e2 * kpc
Mhalo = 1 * 1e10*Msun
Mstar = Mhalo / 1e3

r = np.logspace(-2,8) * pc
nfw = get_αβγ_prof(r,rs=1*kpc,rcut=100*kpc,Mtot=Mhalo).to('Msun/Mpc**3')
core = get_αβγ_prof(r,γ=0,rs=1*kpc,rcut=100*kpc,Mtot=Mhalo).to('Msun/Mpc**3')
hires = get_αβγ_prof(r,α=α,β=β,γ=γ,rs=rs,rcut=rcut,Mtot=Mstar).to('Msun/Mpc**3')

fig = plt.figure(figsize=(10,3))
ax = fig.add_subplot(121)
ax2= fig.add_subplot(122)
ax.loglog(r,nfw,ls='dashed',label='NFW')
ax.loglog(r,core,ls='dashdot',label='Core')
ax.loglog(r,hires,ls='dotted',label='Spike')
ax.loglog(r,core+hires,label='Core+Spike')

ax.axvspan(min(r),(rcut)/(1*r.units),alpha=0.3,label='Mstar region')

ax.set_xlabel(f'r ({r.units})')
ax.set_ylabel(f'ρ ({nfw.units})')

ax.set_ylim(1e13,1e33)

import scipy.integrate as integrate
from functools import partial
def rho_combined(r,*,α=1,β=4,γ=2,cuspcore=1,Mtot=1e10*Msun,Mstar=1e10/4e3,rs=1*kpc,rcut=100*kpc,rss=80*pc,rcuts=10*kpc):
    return get_αβγ_prof(r*kpc,γ=cuspcore,rs=rs,rcut=rcut,Mtot=Mtot) + get_αβγ_prof(r*kpc,α=α,β=β,γ=γ,rs=rss,rcut=rcuts,Mtot=Mstar)
def mass_enc_int(r,rho_fun,**kwargs):
    return 4*np.pi*r**2*rho_fun(r,**kwargs)*kpc**2
rho_fun = partial(rho_combined,α=α,β=β,γ=γ,Mtot=Mhalo,cuspcore=1,Mstar=Mstar,rss=rs,rcuts=rcut)
mass_enc = partial(mass_enc_int,rho_fun=rho_fun)
ax.loglog(r,rho_fun((r.to('kpc')).v).to('Msun/Mpc**3'),label='ρ combined')
ax.legend()

Mstar_calc,calc_err = integrate.quad(mass_enc,0*0.001e-3,1) # 0-1 kpc 
print(f'{Mstar_calc/(Mhalo)=} with error {calc_err/Mhalo} (expected {Mstar/(Mhalo)})')

mass_enc_res = [integrate.quad(mass_enc,0,ro.v) for ro in r.to('kpc')]
me = [m for m,_ in mass_enc_res]
Mhigh = Mstar/Nhigh
num_parts = me / Mhigh
ax2.loglog(r,num_parts)
ax2.set_xlabel(f'r ({r.units})')
ax2.set_ylabel(f'# of particles')

# %% [markdown]
# Above results suggest we need ~$2\times10^{5}$ high res particles, $M_{star} = M_{halo}/1000$, $α = 1$, $β = 4$ (a finite mass model), $γ = 1$ (to match NFW), $rs = 80$ pc, and $rcuts = 1e3$ kpc
#
# **Update 03/04** Upping number of hires particles to $10^5$, $10^6$ to see if this helps with spike collapse. 

# %%
from pathlib import Path
folder = Path("~/workspace/Research/projects/spheric/runs").expanduser()
so_noBH = SphericOptions(MBH=0,dx=0,dy=0,name=f"{folder}/IC-twosideA-noBH",Nhalo=1e4,starabg=True,
                    Nstar=2e5,Mstar=1e-3,star_alpha=1,star_beta=4,star_gamma=2,star_rs=80/1e3,star_rcutoff=1e2,ogh=True)
so_wiBH = SphericOptions(MBH=1e-3,dx=0,dy=0,name=f"{folder}/IC-twosideA-wiBH",Nhalo=1e4,starabg=True,
                    Nstar=2e5,Mstar=1e-3,star_alpha=1,star_beta=4,star_gamma=2,star_rs=80/1e3,star_rcutoff=1e2,ogh=True)
so_hern = SphericOptions(MBH=1e-3,dx=0,dy=0,name=f"{folder}/IC-twosideA-hern",Nhalo=1e4,hernquist=True,
                    Nstar=2e5,Mstar=1e-3,rhern=0.08,ogh=True)
so_habg = SphericOptions(MBH=1e-3,dx=0,dy=0,name=f"{folder}/IC-twosideA-habg",Nhalo=1e4,starabg=True,
                    Nstar=2e5,Mstar=1e-3,star_alpha=1,star_beta=4,star_gamma=1,star_rs=80/1e3,star_rcutoff=1e2,ogh=True)
so_s5BH = SphericOptions(MBH=1e-3,dx=0,dy=0,name=f"{folder}/IC-twosideA-s5BH",Nhalo=2e5,starabg=True,
                    Nstar=2e5,Mstar=1e-3,star_alpha=1,star_beta=4,star_gamma=2,star_rs=80/1e3,star_rcutoff=1e2,ogh=True)
so_s6BH = SphericOptions(MBH=1e-3,dx=0,dy=0,name=f"{folder}/IC-twosideA-s6BH",Nhalo=2e6,starabg=True,
                    Nstar=2e5,Mstar=1e-3,star_alpha=1,star_beta=4,star_gamma=2,star_rs=80/1e3,star_rcutoff=1e2,ogh=True)
#so2 = SphericOptions(MBH=.5e-3,dx=50,dy=-10,dvx=-10,name=f"{folder}/IC-twosideB",Nhalo=1e5,ogh=True)
print(f"Using for halo w/o BH: {so_noBH.generateOptionString()}")
print(f"Using for halo w/  BH: {so_wiBH.generateOptionString()}")
print(f"Using for hern: {so_hern.generateOptionString()}")
print(f"Using for hern ABG: {so_habg.generateOptionString()}")
print(f"Using for 10^5 Halo particles: {so_s5BH.generateOptionString()}")
print(f"Using for 10^6 Halo particles: {so_s6BH.generateOptionString()}")
#print(f"Using for halo 2: {so2.generateOptionString()}")
#print('Not currently working. Try from commandline instead')
#exit()
for so in [so_noBH,so_wiBH,so_hern,so_habg,so_s5BH,so_s6BH]:
    comproc = spheric(so)
    # Note if this errors/is blank, spheric probably segfaulted
    print(comproc.stdout.decode()) 
    print(comproc.stderr.decode())
#comproc = spheric(so2)
# Note if this errors/is blank, spheric probably segfaulted
#print(comproc.stdout.decode()) 
#print(comproc.stderr.decode())

# %%
import yt

yt.set_log_level('warning')
#ds = yt.load('../../runs/spike_test_nBH/IC-twosideA-noBH-gizmo.hdf5',bounding_box=[[-600,600]]*3)
ds = yt.load('runs/IC-twosideA-habg-gizmo.hdf5',bounding_box=[[-600,600]]*3)

ad = ds.all_data()
hrpt = 'PartType1' if 'PartType2' in ds.particle_types else 'PartType4'
lrpt = 'PartType2' if 'PartType2' in ds.particle_types else 'PartType1'

plot = yt.ParticleProjectionPlot(ds,'z',(hrpt,'particle_ones'),origin='native',width=(.5,'kpc'),window_size=(10,3))
if ('PartType5','Coordinates') in ds.field_list:
    plot.annotate_particles(20,ptype="PartType5",col="orange",p_size=10,alpha=0.75)
if (lrpt,'Coordinates') in ds.field_list:
    plot.annotate_particles(20,ptype=lrpt,col='red',p_size=10,alpha=0.5)
#plot.show()

#fig=plt.figure(figsize=(4,4))
fig = plot.export_to_mpl_figure((1,1))
ax1 = fig.add_subplot(122)
rhohr,(prof,npart) = rho_prof(ds=ds,radius=(500,'kpc'),stretch='nan',pt=hrpt)
rhohr = rhohr.to("code_mass/kpc**3")
r = prof.x.to("kpc")
ax1.loglog(r,rhohr,'.-',label=f"Hi-res DM")
rho_αβγ130 = get_αβγ_prof(r.v,)
rho_αβγNFW = get_αβγ_prof(r.v,γ=0)
rholr,(prof,npart) = rho_prof(ds=ds,radius=(500,'kpc'),stretch='nan',pt=lrpt)
rholr = rholr.to("code_mass/kpc**3")
rlr = prof.x.to("kpc")
ax1.loglog(rlr,rholr,'.-',label=f"Lo-res DM")

ax1.loglog(r, rho_αβγ130,label=f'({1},{3},{0})')
ax1.loglog(r, rho_αβγNFW,label=f'({1},{3},{1})')
mass_unit = ds.mass_unit.to("Msun")
mus = latex_float(mass_unit)
ax1.set_xlabel(f"r ({r.units})")
ax1.set_ylabel(f'ρ$(r)$ (${mus}$ M$_{{\odot}}$/kpc)')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend()
#fig.legend(handles, labels, loc='outside right')
fig.tight_layout()
fig.subplots_adjust(wspace=0.6*4)

# %% [markdown]
# ```
# Nstar     = 200000
# starPmass =  5.0000000e-09 MU
# Mstar     =  1.0000000e-03 MU
# K         =  8.8425986e-03 MU LU^-3  ->  Normalization constant for the stellar model. 
# rp        =  3.0000000e-01 LU
# ```
#

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Combine Files

# %%
from pathlib import Path
folder = Path("~/projects/spheric/runs").expanduser()
combineICs(f"{so1.name}-gizmo.hdf5",f"{so2.name}-gizmo.hdf5",f'{folder}/IC-combined.hdf5')
#combineICs(f"{so1.name}-gizmo.hdf5",f"{so2.name}-gizmo.hdf5",'runs/IC-combined-6.hdf5')

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Test combined file

# %%
import yt

ds = yt.load('../../gizmo-public/output/combined/snapshot_000.hdf5',bounding_box=[[-600,600]]*3)

# %%
plot = yt.ParticleProjectionPlot(ds,"z",("PartType1","Masses"),origin='native',window_size=(4,4),width=(100,'kpc'),center=([25,0,0],'kpc'))
plot.annotate_particles(20,ptype='PartType5',col='orange',p_size=25)
plot.show()

# %% jupyter={"source_hidden": true}
sph1 = get_sphere(ds=ds,radius=(800,"kpc"),center=([50,-10,0],"kpc"),refine=True,ref_radius=(20,'kpc'))
rhodm1,(prof1,npart1) = rho_prof(sphere=sph1,stretch=False)
rhodm1 = rhodm1.to("code_mass/kpc**3")

sph2 = get_sphere(ds=ds,radius=(800,"kpc"),center=([0,0,0],"kpc"),refine=True,ref_radius=(20,'kpc'))
rhodm2,(prof2,npart2) = rho_prof(sphere=sph2,stretch=False)
rhodm2 = rhodm2.to("code_mass/kpc**3")

r = prof1.x.to("kpc")

rho_αβγ = get_αβγ_prof(r.v,sphereopts=so1)

fig = plt.figure()
ax = fig.add_subplot()
ax.loglog(r,rhodm1,'*-',label="Halo1")
ax.loglog(r,rhodm2,'.-',label='Halo2')
ax.loglog(r,rho_αβγ,label=f'({so1.alpha},{so1.beta},{so1.gamma})')
ax.set_xlabel(f"r (kpc)")
ax.set_ylabel(r"$\rho(r)$ ($10^{10}$ M$_{\odot}$/kpc)")
ax.legend()

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Merging animation

# %%
try:
    del ts
except:
    pass
ts = yt.load('../../gizmo-public/output/combined/snapshot_*.hdf5',bounding_box=[[-600, 600]] * 3)
ts = sorted(ts,key=attrgetter('current_time'))
print(f'Loaded {len(ts)} snapshots')


# %%
def plotmerger(ds,*,width=(300,'kpc'),**kwargs):
    col_field = ("PartType1","Masses")
    plot = yt.ParticleProjectionPlot(ds,"z",col_field,width=width,window_size=(3,3),origin='native',**kwargs)
    plot.annotate_particles(20,ptype="PartType5",col="orange",p_size=10,alpha=0.75)
    plot.annotate_timestamp(time_unit="Myr",draw_inset_box=True)
    return plot,col_field,None

plot,_,_ = plotmerger(ts[-1],width=(10,'kpc'),center=([8,-5,0],'kpc'))
plot.show()


# %%
def animate_merge(ts,*,num_frames=5,**kwargs):
    idxs = np.linspace(0,len(ts)-1,num=num_frames).astype(int)
    ts = [ts[i] for i in idxs]
    ani = make_animation_from_images(ts,lambda ds:plotmerger(ds,**kwargs),'../../figures/nanograv/basic_merger.gif')
    return ani

animate_merge(ts,num_frames=20,width=(100,'kpc'),center=([15,0,0],'kpc'))

# %% [markdown]
# Note that the following should probably be done using [DatasetSeries.particle_trajectories](https://yt-project.org/doc/reference/api/yt.data_objects.time_series.html#yt.data_objects.time_series.DatasetSeries.particle_trajectories) instead of manually. 

# %%
from unyt import kiloparsec as kpc
from unyt import megayear as Myr
from unyt import kilometer as km
from unyt import second,Msun
from tqdm import tqdm
kmps = km/second

x = np.zeros((len(ts),3)) * kpc
y = np.zeros((len(ts),3)) * kpc
z = np.zeros((len(ts),3)) * kpc
vx = np.zeros((len(ts),2)) * kmps
vy = np.zeros((len(ts),2)) * kmps
vz = np.zeros((len(ts),2)) * kmps
masses = np.zeros((len(ts),2)) * Msun
t = np.zeros(len(ts)) * Myr
for ix,ds in enumerate(tqdm(ts)):
    ad = ds.all_data()
    t[ix] = ds.current_time.to("Myr")
    pids = ad['PartType5','ParticleIDs']
    coords = ad['PartType5','Coordinates']
    vels = ad['PartType5','Velocities']
    mas = ad['PartType5','Masses']
    com = ad.quantities.center_of_mass(use_gas=False,use_particles=True,
                                       particle_type="PartType5",
                                      )
    # need this because particle array location is not consistent between snapshots
    i1 = np.where(pids==10000)[0][0]
    i2 = 1-i1
    x[ix,:] = [coords[i1][0],coords[i2][0],com[0]]
    y[ix,:] = [coords[i1][1],coords[i2][1],com[1]]
    z[ix,:] = [coords[i1][2],coords[i2][2],com[2]]
    vx[ix,:] = [vels[i1][0],vels[i2][0]]
    vy[ix,:] = [vels[i1][1],vels[i2][1]]
    vz[ix,:] = [vels[i1][2],vels[i2][2]]
    masses[ix,:] = [mas[i1],mas[i2]]


# %% jupyter={"source_hidden": true}
# Plot merging halos in CoM frame
def plot_merging_bhs(xo,yo,t,*,zo=None,inspos=[1, -9.75, 18, 3.5],ixl=None,iyl=None):
    fig = plt.figure(figsize=(12,8))
    if zo is None:
        ax = fig.add_subplot()
    else:
        ax = fig.add_subplot(projection='3d')
    #plot = ax.scatter(x,y,5,t)
    #cb = plt.colorbar(plot)
    #cb.set_label('Time [Myr]')
    # Defining here so I don't destroy possible outer variables
    x = y = []
    # Doing transposes to get the broadcasting correctly
    x = (xo[:,0:2].T - xo[:,2].T).T
    y = (yo[:,0:2].T - yo[:,2].T).T
    if zo is not None:
        z = (zo[:,0:2].T - zo[:,2].T).T
    for i in range(2):
        if zo is None:
            zt = None
        else:
            zt = z[:,i]
        line=plot_color_varying_line(x[:,i],y[:,i],t,z=zt,fig=fig,ax=ax,resize=False,label=f"BH {i}")
        if i==1:
            line.set_linestyle('dotted')
        else:
            line.figure.colorbar(line).set_label('Time [Myr]')
        #line.set_linewidth(3)
    #line=plot_color_varying_line(x[:,1],y[:,1],t,fig=fig,ax=ax,resize=False,label="BH 2")
    #line.set_linestyle('dashed')
    #line.set_linewidth(1)
    #line=plot_color_varying_line(x[:,2],y[:,2],t,fig=fig,ax=ax,resize=False,label="BH CoM")
    #line.set_linestyle('dotted')
    #line.set_linewidth(1)
    ax.set_xlabel('x (kpc)')
    ax.set_ylabel('y (kpc)')
    ax.legend()
    xl = [np.min(x),np.max(x)]
    yl = [np.min(y),np.max(y)]
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    if zo is not None:
        zl = [np.min(z),np.max(z)]
        ax.set_zlim(zl)
        ax.set_zlabel('z (kpc)')

    if inspos != -1 and zo is None:
        if ixl is None:
            ixl = xl
        if iyl is None:
            iyl = yl
        axins = ax.inset_axes(
            inspos,transform=ax.transData,
            xlim=(ixl[0], ixl[1]), ylim=(iyl[0], iyl[1]), 
            #xticklabels=[], yticklabels=[]
        )
        line=plot_color_varying_line(x[:,0],y[:,0],t,fig=fig,ax=axins,resize=False)
        #line.set_linewidth(3)
        line=plot_color_varying_line(x[:,1],y[:,1],t,fig=fig,ax=axins,resize=False)
        line.set_linestyle('dashed')
        #line.set_linewidth(1)
        #line=plot_color_varying_line(x[:,2],y[:,2],t,fig=fig,ax=axins,resize=False)
        #line.set_linestyle('dotted')
        #line.set_linewidth(1)
        ax.indicate_inset_zoom(axins, edgecolor="black")
    return fig

fig = plot_merging_bhs(x,y,t,#zo=z,
                       inspos=[-21,-4.55,19,4],
                       #inspos=-1,
                       ixl=[-.4,.4],iyl=[-.4,.4])
fig.gca().set_title('BH movement (CoM frame)')

# %% jupyter={"source_hidden": true}
from functools import partial

def animate_merger(xo,yo,t,*,num_points=50,step=5,offset=0):
    # Defining here so I don't destroy possible outer variables
    x = y = []
    # Doing transposes to get the broadcasting correctly
    x = (xo[:,0:2].T - xo[:,2].T).T
    y = (yo[:,0:2].T - yo[:,2].T).T

    n = len(x)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot()
    bh1, = ax.plot([],[],'.-')
    bh2, = ax.plot([],[],'.-')
    txt = ax.text(0,0,'')

    def animate(i,bh1,bh2,x,y,t,offset,step,num_points,ax):
        i1 = np.maximum(0,i*step - num_points) + offset
        i2 = i*step+1 + offset
        bh1.set_xdata(x[i1:i2,0])
        bh1.set_ydata(y[i1:i2,0])
        bh2.set_xdata(x[i1:i2,1])
        bh2.set_ydata(y[i1:i2,1])
        txt.set_text(f't={int(t[i2-1])}')
        ax.set_xlim((np.min(x[i1:i2,:]),np.max(x[i1:i2,:])))
        ax.set_ylim((np.min(y[i1:i2,:]),np.max(y[i1:i2,:])))
        return bh1,bh2,ax,

    animate(0,bh1=bh1,bh2=bh2,x=x,y=y,t=t,offset=offset,
            step=step,num_points=num_points,ax=ax)
    
    ani = FuncAnimation(
        fig, partial(animate,bh1=bh1,bh2=bh2,
                     x=x,y=y,t=t,step=step,offset=offset,
                     num_points=num_points,ax=ax),
        int((len(t)-offset)/step), blit=False)
    return ani

ani = animate_merger(x,y,t,num_points=20,step=4,offset=00)
ani


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Circular Analysis

# %%
def compute_circles(xo,yo,zo,vxo,vyo,vzo,t):
    # Defining here so I don't destroy possible outer variables
    #x = y = z = r = []
    x,y,z,vx,vy,vz = remove_bulk_xyz(xo,yo,zo,vxo,vyo,vzo,t)
    lun = x.units
    tun = t.units

    # h is the specific angular momentum (L=m*h)
    hvec = np.cross([x[:,0],y[:,0],z[:,0]],[vx[:,0],vy[:,0],vz[:,0]],axis=0).T
    h = np.sqrt(np.sum(hvec**2,axis=1))
    hhat = (hvec.T / h.T).T
    #print(h.shape)
    
    rcom = np.sqrt(x**2 + y**2 + z**2)
    r = np.sqrt((x[:,0] - x[:,1])**2 + (x[:,0] - x[:,1])**2 + (x[:,0] - x[:,1])**2)
    θ = np.arccos(z / rcom)
    ϕ = np.arctan2(y,x)
    ϕ = np.unwrap(ϕ,axis=0)

    drdt = np.gradient(r.v,t.v,axis=0) * lun/tun
    drcomdt = np.gradient(rcom.v,t.v,axis=0) * lun/tun
    dθdt = np.gradient(θ,t.v,axis=0) / tun
    dϕdt = np.gradient(ϕ,t.v,axis=0) / tun
    dhdt = np.gradient(h,t.v,axis=0) * lun**2/tun

    # ω = dϕdt, so ωdot = d/dt(dϕdt)
    ωdot = np.gradient(dϕdt.v,t.v,axis=0) / tun**2

    return r,θ,ϕ,drdt,dθdt,dϕdt,ωdot,rcom,drcomdt,h,dhdt

r,θ,ϕ,drdt,dθdt,ω,ωdot,rcom,drcomdt,h,dhdt = compute_circles(x,y,z,vx,vy,vz,t)
fig=plt.figure(figsize=(12,14))
ax=fig.add_subplot(3,2,1)
ax.semilogy(t,bn.move_mean(np.abs(drdt),4,axis=0),label=r"$\frac{dr_{sep}}{dt}$")
ax.semilogy(t,bn.move_mean(np.abs(drcomdt[:,0]),4,axis=0),label=r"$\frac{dr_{com}}{dt}$")
ax.legend(loc='lower left')
ax.set_xlim(1500,4800)
ax.set_xlabel('Time (Myr)')
ax.set_ylabel(r'$\frac{dr}{dt}$ (kpc/Myr)')
ax.set_title(r'$\frac{dr}{dt}$')
ax2=fig.add_subplot(3,2,2)
#for i in range(3):
#    ax2._get_lines.get_next_color()
ax2.semilogy(t,bn.move_mean(r,1,axis=0),label=r"$r_{sep}$")
ax2.semilogy(t,bn.move_mean(rcom[:,0],1,axis=0),label=r"$r_{com}$")
ax2.legend()
ax2.set_xlabel(ax.get_xlabel())
ax2.set_xlim(ax.get_xlim())
ax2.set_ylabel(r'r (kpc)')
ax2.set_title(r'$r$')
ax=fig.add_subplot(3,2,3)
ax.plot(t,bn.move_mean(dθdt,4,axis=0))
ax.plot(t,bn.move_mean(np.sum(dθdt,axis=1),4,axis=0))
ax.set_xlim(000,4800)
ax.set_title(r'$\frac{d\theta}{dt}$')
ax=fig.add_subplot(3,2,4)
ax.plot(t,bn.move_mean(ω,4,axis=0))
ax.set_xlim(000,4800)
ax.set_title(r'$\frac{d\phi}{dt}=\omega$')
ax=fig.add_subplot(3,2,6)
ax.plot(t,bn.move_mean(dhdt,1,axis=0),'.')
ax.set_xlim(2500,4800)
ax.set_yscale('symlog',linthresh=1e-4)
ax.set_title(r'$\frac{dh}{dt}$')
ax2=fig.add_subplot(3,2,5)
ax2.semilogy(t,bn.move_mean(h,1,axis=0))
ax2.set_xlim(ax.get_xlim())
ax2.set_ylim(0.1,1)
fig.subplots_adjust(wspace=0.4,hspace=0.4)
ax2.set_title(r'$h$ (specific angular momentum)')
fig.subplots_adjust(wspace=0.4,hspace=0.4)

# %% [markdown]
# Lets look at $\frac{dE}{dt}=\frac{d}{dt}\left(\frac{1}{2}m v^2 - \frac{Gm}{r}\right)$

# %%
from unyt import gravitational_constant as G
def computeEnergyStuff(xo,yo,zo,vxo,vyo,vzo,t,m1,m2):
    x,y,z,vx,vy,vz = remove_bulk_xyz(xo,yo,zo,vxo,vyo,vzo,t)
    lun = x.units
    tun = t.units

    v2 = vx**2 + vy**2 + vz**2
    r = np.sqrt(x**2+y**2+z**2)
    E1 = 1/2*m1*v2[:,0] -  G*m1*m1/r[:,0]
    E2 = 1/2*m2*v2[:,1] -  G*m1*m2/r[:,1]
    E = np.array([E1,E2]).T
    dEdt = np.gradient(E,axis=0)
    return E,dEdt

E,dEdt = computeEnergyStuff(x,y,z,vx,vy,vz,t,masses[0,0],masses[0,1])
fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.plot(t,E/E[0,:])
ax = fig.add_subplot(1,2,2)
ax.plot(t,dEdt)

# %% [markdown]
# Since $W_{gw} = \frac{32}{5} G \mu^2 \omega^6 r^4$, $\frac{dE}{d\omega}=\frac{W_{gw}}{\dot{\omega}}$ and we now have $r$, $\omega$, and $\dot{\omega}$, we can calculate $\frac{dE}{d\omega}$ directly. But then $h^2(\omega)\sim \frac{16\pi G}{c^2 \omega} \frac{dE}{d\omega}$

# %% jupyter={"source_hidden": true}
from unyt import gravitational_constant as G
from unyt import speed_of_light as c

def compute_h(μ,ωo,ro,ωdoto,*,n=4):
    N0 = kpc**-3
    r = bn.move_mean(ro,n,axis=0)
    ω = bn.move_mean(ωo,n,axis=0)
    ωdot = bn.move_mean(ωdoto,n,axis=0)
    Wgw = 32/5* G * μ**2 * ω**6 * r**4 / c**5
    dEdω = Wgw / ωdot

    h2 = 16*np.pi*G/(c**2*np.abs(ω)) * N0 * dEdω
    assert np.all(h2>0), "Negative h2 value"
    h = np.sqrt(h2)
    return h
masses = ts[0].all_data()['PartType5','Masses'].to('Msun')
μ = np.product(masses)/np.sum(masses)
h = compute_h(μ,ω,r,ωdot)
fig = plt.figure()
ax = fig.add_subplot()
idx = np.where(t>3000)
ax.plot(np.abs(ω[idx]).to('nanohertz')/(2*np.pi),h[idx],'.')
#ax.set_xlim(3000,3600)

# %% [markdown]
# # Analyzing bridges2 combinations

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Overall animations

# %% jupyter={"source_hidden": true}
import yt
import re
#from operator import attrgetter
from yt.data_objects.time_series import DatasetSeries
try:
    del ts
except:
    pass
yt.set_log_level('warning')
foldername = '~/storage/runs/IgorTest/outputs-public/snapshot_*.hdf5'
#foldername = '~/storage/runs/IgorTest/outputs/snapshot_*.hdf5'
#ts = yt.load(foldername,bounding_box=[[-600, 600]] * 3)
#print(f'Loaded {len(ts)} snapshots')
#dsnamelist = DatasetSeries._get_filenames_from_glob_pattern(foldername)
#tsinds = np.argsort([int(re.search(r'snapshot_(\d+)',n)[1]) for n in dsnamelist])
#ts = [ts[i] for i in dsinds]
#ts = sorted(ts,key=attrgetter('current_time'))
#ts = sorted(ts, key=lambda ds: int(re.search(r'snapshot_(\d+)',ds.filename)[1]))
#print(f'Sorted {len(ts)} snapshots')
ts,(tsinds,dsnamelist) = load_timeseries_from_folder(foldername,bounding_box=[[-600, 600]]*3)


# %%
def plotmerger(ds,*,width=(300,'kpc'),**kwargs):
    col_field = ("PartType1","Masses")
    ad = ds.all_data() # make sure the data is loaded and particle fields are populated
    hrpt = 'PartType1' if 'PartType2' in ds.particle_types else 'PartType4'
    lrpt = 'PartType2' if 'PartType2' in ds.particle_types else 'PartType1'
    plot = yt.ParticleProjectionPlot(ds,"z",(hrpt,'particle_ones'),col_field,width=width,window_size=(3,3),origin='native',**kwargs)
    if ('PartType5','Coordinates') in ds.field_list:
        plot.annotate_particles(20,ptype="PartType5",col="orange",p_size=10,alpha=0.75)
    if (lrpt,'Coordinates') in ds.field_list:
        plot.annotate_particles(20,ptype=lrpt,col='red',p_size=10,alpha=0.1)
    plot.annotate_timestamp(time_unit="Myr",draw_inset_box=True)
    return plot,col_field,None

plot,_,_ = plotmerger(sim.ts[-1],width=(10,'kpc'),center=([12,-5,0],'kpc'))
plot.show()


# %%
def animate_merge(ts,*,num_frames=5,tsinds=None,**kwargs):
    idxs = np.linspace(0,len(ts)-1,num=min(num_frames,len(ts))).astype(int)
    if tsinds is None:
        ts = [ts[i] for i in idxs]
    else:
        ts = [ts[tsinds[i]] for i in idxs]
    ani = make_animation_from_images(ts,lambda ds:plotmerger(ds,**kwargs),f'{Path.home()}/storage/figures/darknanograv/Igor_merger.gif')
    return ani

animate_merge(sim.ts,num_frames=20,tsinds=sim.inds,width=(100,'kpc'),center=([15,0,0],'kpc'))

# %%

# %% jupyter={"source_hidden": true}
from yt.units import parsec as pc
from yt.utilities.exceptions import YTFieldNotFound
ds = sim.ts[-1]
ad = ds.all_data()
try:
    q = np.quantile(ad['PartType1','TimeStep'],0.0001)
except YTFieldNotFound:
    print('Field TimeStep does not exist. Check that GIZMO was compiled with OUTPUT_TIMESTEP')
    exit()
lti = np.argwhere(ad['PartType1','TimeStep']<=q)[0]
fig=plt.figure(figsize=(8,4))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
bh1 = ad['PartType5','particle_position'][0].to('pc')
bh2 = ad['PartType5','particle_position'][1].to('pc')
ax1.plot(bh1[0],bh1[1],marker='s',ls='none')
ax2.plot(bh2[0],bh2[1],marker='s',ls='none')
def add_bracket(x,r,bh,ax):
    mid = (x+bh)/2
    theta = np.arccos((x[0]-bh[0])/r)
    print(f'{mid=} \n {theta=}')
    xy = (mid[0],mid[1]) + r/10*(np.cos(theta+np.pi/2),np.sin(theta+np.pi/2))
    xyt = (mid[0],mid[1]) + r/5*(np.cos(theta+np.pi/2),np.sin(theta+np.pi/2))
    fs = 6
    ax.annotate(f'{r:.3}', xy=xy, xytext=xyt, xycoords='data', 
            fontsize=fs*1.5, ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white', color='k'),
            arrowprops=dict(arrowstyle=f'-[, widthB={5}, lengthB=.5, angleB={270+180/np.pi*theta}', lw=2.0, color='k'))
maxr = 1
minr = 1e3
counts = 0
for i in lti:
    x = ad['PartType1','particle_position'][i].to('pc')
    r1 = np.sqrt(np.sum((x-bh1)**2))
    r2 = np.sqrt(np.sum((x-bh2)**2))
    maxr = r1 if r1>maxr else maxr
    minr = r1 if r1<minr else minr
    maxr = r2 if r2>maxr else maxr
    minr = r2 if r2<minr else minr
    if r1<1e3:
        ax1.plot(x[0],x[1],marker='.',ls='none')
        add_bracket(x,r1,bh1,ax1)
    if r2<1e3:
        ax2.plot(x[0],x[1],marker='.',ls='none')
        add_bracket(x,r2,bh2,ax2)
    counts = counts + (1 if r1<1e3 or r2<1e3 else 0)
print(f"Closest low timebin: {minr}. Furthest low timebin: {maxr}")
print(f"Should be plotting {counts} points")
maxr = min(maxr,1e3)
maxr = 10**(np.ceil(np.log10(maxr)))/4
xw = maxr*pc
yw = maxr*pc
ax1.set_xlim(bh1[0]-xw,bh1[0]+xw)
ax1.set_ylim(bh1[1]-yw,bh1[1]+yw)
ax2.set_xlim(bh2[0]-xw,bh2[0]+xw)
ax2.set_ylim(bh2[1]-yw,bh2[1]+yw)

# %% jupyter={"source_hidden": true}
from functools import partial

def animate_merger(sim,*,num_points=50,step=5,offset=0):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot()
    bh1, = ax.plot([],[],'.-')
    bh2, = ax.plot([],[],'.-')
    bv1 = [ax.arrow([],[],[],[]) for i in range(0,num_points)]
    bv2 = [ax.arrow([],[],[],[]) for i in range(0,num_points)]
    txt = ax.text(0,0,'')

    def animate(i,bh1,bh2,bv1,bv2,sim,offset,step,num_points,ax):
        i1 = np.maximum(0,i*step - num_points) + offset
        i2 = i*step+1 + offset
        trajs = [t for t in sim.trajs]
        t1 = trajs[0]
        t2 = trajs[1]
        time=unyt_array([sim.coms[i].time for i in range(i1,i2)])
        bp = sim.coms.get_bulk(time,ptype='bh')
        x1 = (t1['PartType5','particle_position_x'][i1:i2]-bp[:,0]).to('pc')
        y1 = (t1['PartType5','particle_position_y'][i1:i2]-bp[:,1]).to('pc')
        bv = sim.coms.get_bulk_velocity(time,ptype='bh')
        if len(time)<2:
            dt = time
        else:
            dt = [*np.diff(time), time[-1]-time[-2]]
        dx1 = ((t1['PartType5','particle_velocity_x'][i1:i2]-bv[:,0]) * dt).to('pc')
        dy1 = ((t1['PartType5','particle_velocity_y'][i1:i2]-bv[:,1]) * dt).to('pc')
        bh1.set_xdata(x1)
        bh1.set_ydata(y1)
        for i,a in enumerate(bv1):
            if i>=len(x1):
                continue
            a.set_data(x=x1[i].v,y=y1[i].v,dx=dx1[i].v,dy=dy1[i].v)
        x2 = (t2['PartType5','particle_position_x'][i1:i2]-bp[:,0]).to('pc')
        y2 = (t2['PartType5','particle_position_y'][i1:i2]-bp[:,1]).to('pc')
        bh2.set_xdata(x2)
        bh2.set_ydata(y2)
        txt.set_text(f't={int(time[-1].to("Myr"))}')
        ax.set_xlim((np.min([*x1,*x2]),np.max([*x1,*x2])))
        ax.set_ylim((np.min([*y1,*y2]),np.max([*y1,*y2])))
        return bh1,bh2,ax,

    animate(0,bh1=bh1,bh2=bh2,bv1=bv1,bv2=bv2,sim=sim,offset=offset,
            step=step,num_points=num_points,ax=ax)
    num_snaps = len(sim.coms)
    ani = FuncAnimation(
        fig, partial(animate,bh1=bh1,bh2=bh2,bv1=bv1,bv2=bv2,
                     sim=sim,step=step,offset=offset,
                     num_points=num_points,ax=ax),
        int((num_snaps-offset)/step), blit=False)
    return ani

ani = animate_merger(sim,num_points=5,step=1,offset=390)
ani

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Circular analysis

# %% jupyter={"source_hidden": true}
foldername = '~/storage/runs/IgorTest/outputs-public/snapshot_*.hdf5'
sim = Simulation(foldername)
exit()

import yt
from yt import DatasetSeries
from operator import attrgetter
import re
unit_base = {
            "length": (1.0, "kpc"),
            "velocity": (1.0, "km/s"),
            "mass": (1e10, "Msun"),
            "temperature": (1.0, "K"),
        }
try:
    del ts
except:
    pass

yt.set_log_level("warning")

foldername = '~/storage/runs/IgorTest/outputs-public/snapshot_*.hdf5'
ts,(tsinds,dsnamelist) = load_timeseries_from_folder(foldername,num_snaps=None,bounding_box=[[-600, 600]] * 3)
print('Computing CoMs...',end='')
coms = COMs(ts)
print('done')

ptype='PartType5'
fields = [
    (ptype, "particle_position_x"),
    (ptype, "particle_position_y"),
    (ptype, "particle_position_z"),
    (ptype, "particle_velocity_x"),
    (ptype, "particle_velocity_y"),
    (ptype, "particle_velocity_z"),
    (ptype, "Masses"),
]
ds = ts[0]
# this is excessive, we really only need 10001 and 20002 I think?
# Depends on how many particles - need a more robust method. Since
# there are only 2 BHs just use all_data
#init_sphere = ds.sphere(coms.get_bulk(ds.current_time,ptype='bh'), (.1, "kpc"))
#indices = init_sphere[("PartType5", "particle_index")].astype("int64")
#indices = [100001,200002]
indices = ds.r[ptype,'ParticleIDs'].v
trajs = ts.particle_trajectories(indices, fields=fields, ptype=ptype)
print(f'Loaded {len(trajs)} trajectories')

# %%
from yt.units import gravitational_constant as G
from yt.units import second, year, Msun
from yt.units import kilometer as km
from yt.units import speed_of_light as c
from yt.units import parsec as pc
fig = plt.figure(figsize=(10,10.5))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(3,2,(3))
ax4 = fig.add_subplot(3,2,4)
ax5 = fig.add_subplot(3,2,5)
ax6 = fig.add_subplot(3,2,6)
indices=sim.trajs.indices
ptype='PartType5'
mass_dict = {t['particle_index']:[t[('PartType5','Masses')][0],indices[0] if t['particle_index']==indices[1] else indices[1]] for t in sim.trajs}
for ind,t in enumerate(sim.trajs):
    time = t["particle_time"]
    bp = sim.coms.get_bulk(time,ptype='bh')
    x = t[(ptype,'particle_position_x')].to('kpc')-bp[:,0]
    y = t[(ptype,'particle_position_y')].to('kpc')-bp[:,1]
    z = t[(ptype,'particle_position_z')].to('kpc')-bp[:,2]
    ri = np.sqrt(x**2+y**2+z**2)
    bpv = sim.coms.get_bulk_velocity(time,ptype='bh')
    vix = t[(ptype,'particle_velocity_x')].to('km/s')-bpv[:,0]
    viy = t[(ptype,'particle_velocity_y')].to('km/s')-bpv[:,1]
    viz = t[(ptype,'particle_velocity_z')].to('km/s')-bpv[:,2]
    inds = np.argsort(time)
    time = time[inds]
    ri = ri[inds]
    x = x[inds]
    y = y[inds]
    z = z[inds]
    vix = vix[inds]
    viy = viy[inds]
    viz = viz[inds]
    # These are not the quantities we're looking for, move along
    #omega_vec = r x v = omega * omega_hat 
    # omega = sqrt(omega . omega)
    #omega_vec = (np.cross([x,y,z],[vx,vy,vz],axis=0) *x.units*vx.units / r**2).T
    #omega = np.sqrt(np.einsum('ij, ij->i',omega_vec.v,omega_vec.v)) * omega_vec.units
    # E = 1/2 M1 v^2 - G M1 M2 /r
    M1 = mass_dict[t['particle_index']][0]
    M2 = mass_dict[mass_dict[t['particle_index']][1]][0]
    M = M1+M2
    μ = M1*M2/M
    d = M/M1 * ri # Since M1 is always the mass of _this_ BH, we're always effectively looking at r1
    vx = M/M2 * vix
    vy = M/M2 * viy
    vz = M/M2 * viz
    v2 = vx**2 + vy**2 + vz**2
    v = np.sqrt(v2)
    a = 1 / (2/d - v2/(G*M))
    omega = np.sqrt(G*M/a**3)
    # from 1807.11489
    # h_vec = J_vec / μ
    h_vec = (np.cross(unyt_array([vx,vy,vz]),M/M1 * [x,y,z],axis=0)).T # need to keep the units using unyt_array
    h_mag2 = np.einsum('ij, ij->i',h_vec.v,h_vec.v) * h_vec.units**2
    # μ*cross(v_vec,r_vec) = J = μ*sqrt(G*M*a*(1-e^2)) --> e = sqrt(1 - cross(v_vec,r_vec)/(G*M*a)
    e = np.sqrt(1-h_mag2/(G*M*a))
    
    E = 1/2*μ*v2 - G*M*μ/d
    # currently double-checking the time unit needed when using np.gradient
    # omegadot = np.gradient(omega_circ.v,time.v,axis=0) * omega.units / time.units # This is averaging over **MANY** periods
    # dEdt = np.gradient(E.v,time.v,axis=0) *E.units / time.units # Same problem
    # We'll assume M_dot ~ 0
    # a_dot = a^-2 * (2*d_dot/d^2 + 2*v/(G*M)*v_dot)
    # omegadot = -3/2 * ω * a_dot / a
    # If E = -G*M*μ/(2*a) then
    # E_dot = G*M*μ/(2*a^2) * a_dot
    v_dot = 0 * km/second/second
    a_dot = 2*a**-2 * (v/d**2 + v/(G*M)*v_dot)
    omega_dot = -3/2 * omega / a * a_dot
    dEdt = G*M*μ/(2*a**2) * a_dot
    
    h = 4*G*np.pi/(c**2 * omega * omega_dot) * dEdt
    
    # plotting
    ax1.plot(x,y)
    M1s = f'${latex_float(signif(M1.to('Msun'),2))}$ M{r"$_{\odot}$"}'
    # remember no factor of 2 for omega -> f
    liner, =ax2.plot(time.to('Myr'),ri.to('pc'),label=f'id:{M1s}')
    nonneg = np.argwhere(a>0) # better to use binary mask but eh
    neg = np.argwhere(a<0)
    ttemp = time.to('Myr')
    ttemp[neg]=np.nan
    linea, = ax2.plot(ttemp,a.to('pc'),ls='dashed',label=f'_(-)a',color=liner.get_color())
    ttemp = time.to('Myr')
    ttemp[nonneg]=np.nan
    ax2.plot(ttemp,-a.to('pc'),ls='dotted',label=f'_-a',color=liner.get_color())
    ax3.plot(time.to('Myr'),e,label=f'id:{M1s}')
    l=ax4.plot(time.to('Myr'),np.abs(omega/(np.pi)).to('1/yr'),label=f'id:{M1s}')
    #ax4.plot(time.to('Myr'),(omega_circ/(np.pi)).to('1/yr'),color=l[0].get_color(),ls='dashed',label=f'id:{t["particle_index"]} circ')
    #ax5.plot(time.to('Myr'),np.abs(dEdt.to("Msun*km**2/s**3")),label=f'id:{t["particle_index"]}')
    ax5.plot(time.to('Myr'),np.abs(h).to('Mpc**3'),label=f'id:{M1s}')
    ax6.plot((omega/(np.pi)).to('1/yr'),np.abs(h).to('Mpc**3'),label=f'id:{M1s}')
def next_color(ax):
    null_line=ax.plot(np.nan,np.nan)
    return null_line[0].get_color()
ax1.set_xlabel('x (kpc)')
ax1.set_ylabel('y (kpc)')
ax2.set_xlabel('t (Myr)')
ax2.set_ylabel('r (pc)')
ax2.set_yscale('log')
yl = ax2.get_ylim()
ax2.fill_between(time.to('Myr'),y1=yl[0],y2=yl[1],where=E>=0,alpha=0.2,color='gray',label=r'E$\geq$0')
ax2.axhline(0.0001*pc,label=f'Min Softening',color=next_color(ax2)) # Note this value is in pc
bhs_interact = 30*pc*np.sqrt(M1/(1e8*Msun))
ax2.axhline(bhs_interact,label=f'BHs Interact',color=next_color(ax2))
ax2.set_ylim(yl)
ax3.set_xlabel('t (Myr)')
ax3.set_ylabel('eccentricity (e)')
ax3.set_yscale('log')
ax4.set_yscale('log')
xl = ax4.get_xlim()
yl = ax4.get_ylim()
ptafr=ax4.axhspan(1/year,.5e-1/year,alpha=0.5,label='PTA Freq',color=next_color(ax4))
stelfr=ax4.axhspan(1e-3/year,1e-9/year,alpha=0.5,label='Stellar',color=next_color(ax4))
dts = np.unique(signif(np.diff(time),2)) * time.units
for dt in dts:
    ax4.axhline(5/dt.to('yr'),label='Time resolution',color=next_color(ax4))
ax4.set_xlim(xl)
ax4.set_ylim(yl)
ax4.set_xlabel('t (Myr)')
ax4.set_ylabel(r'$f=\omega/(2 \pi)$ yr$^{-1}$')
ax5.set_xlabel('t (Myr)')
#ax5.set_ylabel(r'dE/dt ($M_{\odot}\frac{km^2}{s^3}$)')
ax5.set_ylabel('|h|')
ax5.set_yscale('log')
ax6.set_xscale('log')
ax6.set_yscale('log')
xl = ax6.get_xlim()
yl = ax6.get_ylim()
ax6.axvspan(1/year,5e-1/year,alpha=0.5,label='PTA Freq',color=ptafr.get_facecolor())
ax6.axhspan(1e-3/year,1e-5/year,alpha=0.5,label='Stellar',color=stelfr.get_facecolor())
ax6.set_xlim(xl)
ax6.set_ylim(yl)
ax6.set_xlabel(r'f (yr$^{-1}$)')
ax6.set_ylabel('|h|')

if len(sim.trajs)<7:
    ax2.legend(loc='lower left')
    ax4.legend()
#fig.subplots_adjust(wspace=0.2)

# %% [markdown]
# ## Density profiles

# %% jupyter={"source_hidden": true}
#from operator import attrgetter
#unit_base = {
#            "length": (1.0, "kpc"),
#            "velocity": (1.0, "km/s"),
#            "mass": (1e10, "Msun"),
#            "temperature": (1.0, "K"),
#        }
foldername = '~/storage/runs/IgorTest/outputs-public-cdm/snapshot_*.hdf5'
#ts,(tsinds,dsnamelist) = load_timeseries_from_folder(foldername,
#                                                     num_snaps=5,
#                                                     unit_base=unit_base,
#                                                     bounding_box=[[-300,300]]*3)
sim = Simulation(foldername)
ts = sim.downsample_ts(5)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
r0 = []
sonfw = SphericOptions(alpha=1,beta=3,gamma=1,Mhalo=2)
so130 = SphericOptions(alpha=1,beta=3,gamma=0,Mhalo=2)
for ix,ds in enumerate(ts):
    center = sim.coms.get_bulk(ds.current_time,ptype='bh')
    sph = get_sphere(ds=ds,radius=(300,"kpc"),
                     #center=([-0,0,0],"kpc"),
                     center=center,
                    )
    rhodm,(prof,npart) = rho_prof(sphere=sph,stretch='nan')
    rhodm = rhodm.to("code_mass/kpc**3")

    r = prof.x.to("kpc")
    if ix==0:
        r0 = r
    else:
        r0 = np.unique([*r0,*r])*r.units
    rho130 = get_αβγ_prof(r.v,sphereopts=so130)
    rhoNFW = get_αβγ_prof(r.v,sphereopts=sonfw)
    ax.loglog(r,rhodm,'.-',label=f"{ds.current_time.to('Myr'):.4g}")
    ax2.loglog(r,r**2 * rhodm,'.-',label=f"{ds.current_time.to('Myr'):.4g}")
    ax3.loglog(r,rhodm/rhoNFW,label=f"{ds.current_time.to('Myr'):.4g}")
    ax4.loglog(r,rhodm/rho130,label=f"{ds.current_time.to('Myr'):.4g}")
        
rho_αβγ130 = get_αβγ_prof(r0.v,sphereopts=so130)
rho_αβγNFW = get_αβγ_prof(r0.v,sphereopts=sonfw)

ax.loglog(r0, rho_αβγ130,label=f'({so130.alpha},{so130.beta},{so130.gamma})')
ax.loglog(r0, rho_αβγNFW,label=f'({sonfw.alpha},{sonfw.beta},{sonfw.gamma})')
ax2.loglog(r0,r0**2 * rho_αβγ130,label=f'({so130.alpha},{so130.beta},{so130.gamma})')
ax2.loglog(r0,r0**2 * rho_αβγNFW,label=f'({sonfw.alpha},{sonfw.beta},{sonfw.gamma})')
mass_unit = ds.mass_unit.to("Msun")
mus = latex_float(mass_unit)
ax.set_xlabel(f"r ({r0.units})")
ax.set_ylabel(f'{r"$\rho(r)$ ($"}{mus}{r"$ M$_{\odot}$/kpc)"}')
ax2.set_xlabel(f"r ({r0.units})")
ax2.set_ylabel(f'{r"$r^2 \rho(r)$ ($"}{mus}{r"$ M$_{\odot}\cdot$kpc)"}')
ax3.set_xlabel(f"r ({r0.units})")
ax3.set_ylabel(r"$\rho(r)/\rho_{NFW}$")
ax4.set_xlabel(f"r ({r0.units})")
ax4.set_ylabel(r"$\rho(r)/\rho_{130}$")
#ax.legend()
#ax2.legend()
#ax3.legend()
#ax4.legend()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='right')

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Compare cdm and sidm

# %%
folder = '/jet/home/mryan1/storage/runs/IgorTest/outputs-public'
simcdm = Simulation(f'{folder}-cdm')
simidm = Simulation(folder)
sample_rate = 5
tscdm=simcdm.downsample_ts(sample_rate)

# double check times match up 
#print(f'CDM t100:{simcdm.ts[100].current_time.to("Myr")} SIDM t100:{simidm.ts[100].current_time.to("Myr")}')

# Add plot (maybe second plot, maybe second axis on main plot) showing number of particles in bin

fig=plt.figure(figsize=(8,4))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2,sharex=ax1,sharey=ax1)
sonfw = SphericOptions(alpha=1,beta=3,gamma=1,Mhalo=2)
so130 = SphericOptions(alpha=1,beta=3,gamma=0,Mhalo=2)

for ix,dsc in enumerate(tscdm):
    time = dsc.current_time
    dsi = simidm.ts[simidm.get_index_from_time(time)]
    cencdm = simcdm.coms.get_bulk(time,ptype='bh')
    cenidm = simidm.coms.get_bulk(time,ptype='bh')
    sphcdm = get_sphere(ds=dsc,radius=(300,"kpc"),
                        center=cencdm,
                        #center=([0,0,0],'kpc')
                        )
    sphidm = get_sphere(ds=dsi,radius=(300,"kpc"),
                        center=cenidm,
                        #center=([0,0,0],'kpc')
                       )
    rhocdm,(prof,npart) = rho_prof(sphere=sphcdm,stretch='nan')
    rhocdm = rhocdm.to("code_mass/kpc**3")
    r = prof.x.to("kpc")
    if ix==0:
        r0 = r
    else:
        r0 = np.unique([*r0,*r])*r.units
    ax1.loglog(r,rhocdm,'.-',label=f"{dsc.current_time.to('Myr'):.4g}")
    rhoidm,(prof,npart) = rho_prof(sphere=sphidm,stretch='nan')
    rhoidm = rhoidm.to("code_mass/kpc**3")
    r = prof.x.to("kpc")
    r0 = np.unique([*r0,*r])*r.units
    ax2.loglog(r,rhoidm,'.-',label=f"{dsi.current_time.to('Myr'):.4g}")
rho_αβγ130 = get_αβγ_prof(r0.v,sphereopts=so130)
rho_αβγNFW = get_αβγ_prof(r0.v,sphereopts=sonfw)

ax1.loglog(r0, rho_αβγ130,label=f'({so130.alpha},{so130.beta},{so130.gamma})')
ax1.loglog(r0, rho_αβγNFW,label=f'({sonfw.alpha},{sonfw.beta},{sonfw.gamma})')
ax2.loglog(r0, rho_αβγ130,label=f'({so130.alpha},{so130.beta},{so130.gamma})')
ax2.loglog(r0, rho_αβγNFW,label=f'({sonfw.alpha},{sonfw.beta},{sonfw.gamma})')
ax1.set_title('CDM')
ax2.set_title('SIDM')
mass_unit = dsc.mass_unit.to("Msun")
mus = latex_float(mass_unit)
ax1.set_xlabel(f"r ({r0.units})")
ax1.set_ylabel(f'{r"$\rho(r)$ ($"}{mus}{r"$ M$_{\odot}$/kpc)"}')
ax2.set_xlabel(f"r ({r0.units})")
ax2.set_ylabel(f'{r"$\rho(r)$ ($"}{mus}{r"$ M$_{\odot}$/kpc)"}')
handles, labels = ax1.get_legend_handles_labels()
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
fig.legend(handles, labels, loc='outside right')
fig.subplots_adjust(wspace=0.3)

# %%

# %% [markdown]
# ## Comparing velocity dispersions and also hern vs herndm

# %%
ishern = yt.load('/ocean/projects/phy240001p/mryan1/runs/spike_test_hedm/IC-twosideA-hern-gizmo.hdf5',bounding_box=[[-2e3,2e3]]*3)
ishabg = yt.load('/ocean/projects/phy240001p/mryan1/runs/spike_test_habg/IC-twosideA-habg-gizmo.hdf5',bounding_box=[[-2e3,2e3]]*3)
iswiBH = yt.load('/ocean/projects/phy240001p/mryan1/runs/spike_test_wBH/IC-twosideA-wiBH-gizmo.hdf5',bounding_box=[[-2e3,2e3]]*3)
fig = plt.figure(figsize=(9,3))
axes = [fig.add_subplot(121+i) for i in range(2)]
ics = [ishern,ishabg]
ics = [ishern,iswiBH]
for ds in ics:
    hrl,lrl,corel,cuspl = prof_ds(ds,ax=axes[0])
    lrl.set_label('_LR')
    hrl.set_label(f'{Path(ds.filename).parent.stem}')
    corel.set_label('_core')
    cuspl.set_label('_cusp')
    hrl,lrl = plot_dispersion(None,ax=axes[1],p=1,ds=ds,radius=(1,'Mpc'))
    lrl.set_label('_LR')
    hrl.set_label(f'{Path(ds.filename).parent.stem}')
for ax in axes:
    ax.legend()

# %%
folder = '/ocean/projects/phy240001p/mryan1/runs/spike_test_hern/'
simhern = Simulation(folder)
folder = '/ocean/projects/phy240001p/mryan1/runs/spike_test_hedm/'
simhedm = Simulation(folder)
folder = '/ocean/projects/phy240001p/mryan1/runs/spike_test_habg/'
simhabg = Simulation(folder)
folder = '/ocean/projects/phy240001p/mryan1/runs/spike_test_hesl/'
simhesl = Simulation(folder)
folder = '/ocean/projects/phy240001p/mryan1/runs/spike_test_wBH/'
simwiBH = Simulation(folder)
fig = plt.figure(figsize=(8,8))
axes=[fig.add_subplot(221+i) for i in range(4)]
sima = [simhern,simhedm]
sima = [simhern,simhabg]
sima = [simhesl,simhabg]
sima = [simhesl,simwiBH]
for i,s in enumerate(sima):
    for j,dsi in enumerate([0,-1]):
        ds = s.ts[dsi]
        ai = i+2*j
        #plot_num_particles(None,ax=axes[ai],ds=ds,radius=(1,'Mpc'))
        plot_dispersion(None,ax=axes[ai],p=1,ds=ds,radius=(1,'Mpc'))
        axes[ai].set_title(f'{s.simname()} t={signif(ds.current_time.to("Myr"),2)} Myr')
fig.subplots_adjust(wspace=0.3,hspace=0.3)

# %% [markdown]
# ### Density Profile

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
#simhedm.animate()
simwiBH.make_density_prof(num_lines=9,center='bh')

# %%
fig=plt.figure(figsize=(9,4))
axes = [fig.add_subplot(121+i) for i in range(2)]
simhern.make_density_prof(num_lines=5,ax=axes[0])
simwiBH.make_density_prof(num_lines=5,ax=axes[1],center='bh')
axes[0].set_title(r'no BH ($v_0=20$)')
axes[0].set_title(r'Hern (CDM)')
axes[1].set_title(r'w/ BH ($v_0=10$)')
fig.subplots_adjust(wspace=0.3)

# %% [markdown]
# ### Number of particles

# %%
sim = simwiBH
fig = plt.figure(figsize=(9,4))
axes=[fig.add_subplot(121+i) for i in range(2)]
for i,dsi in enumerate([0,-1]):
    ds = sim.ts[dsi]
    com = sim.coms[dsi]
    ax = axes[i]
    shells = get_shells(ds=ds,radius=(1,'Mpc'))
    bins = get_shell_bins(shells)
    for center in ['bh','all','stars','dm']:
        if not hasattr(com,center):
            continue
        c = com.__dict__[center]
        hrl,lrl = plot_num_particles(None,ds=ds,radius=(1,'Mpc'),center=c,override_bins=bins,ax=ax,)
        hrl.set_label(f'{center} at {[signif(x,3) for x in c.to("pc").v]} pc')
        lrl.set_label(f'_{lrl.get_label()}')
    ax.set_title(f't={signif(ds.current_time.to("Myr").v,3)} Myr')
    # note we've hard-coded the units here. Be careful!
    ax.set_xlabel('r from center (pc)')
    ax.legend()

# %% [markdown]
# ### Mean free path

# %%
from yt.units import kilometer as km
from yt.units import second
sim = simwiBH
v0,sigma0=(float(sim.get_parameter('DM_InteractionVelocityScale'))*km/second,float(sim.get_parameter('DM_InteractionCrossSection'))*cm**2/g)
print(f'Sim uses {v0=} and {sigma0=}')
#v0,sigma0=(20*km/second,1*cm**2/g)
sigma_over_m = lambda v,sigma0,v0:sigma0/(1+(v/v0)**(4))
fig=plt.figure(figsize=(9,4))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
for i,dsi in enumerate([0,-1]):
    ds = sim.ts[dsi]
    hrline,lrline = plot_mean_free_path(ds=ds,ax=ax,radius=(1,'Mpc'),sigma_over_m=partial(sigma_over_m,sigma0=sigma0,v0=v0))
    hrline.set_label(f't={signif(ds.current_time.to("Myr").v,3)} Myr')
    lrline.set_label(f'_{lrline.get_label()}')
    hrline,lrline = plot_relax_time(ds=ds,ax=ax2,radius=(1,'Mpc'),sigma_over_m=partial(sigma_over_m,sigma0=sigma0,v0=v0))
    hrline.set_label(f't={signif(ds.current_time.to("Myr").v,3)} Myr')
    lrline.set_label(f'_{lrline.get_label()}')
ax.legend()
ax2.legend()
fig.subplots_adjust(wspace=0.3)

# %% [markdown]
# ### Notes
# Previous run of habg goes about 20 Myr. The relaxation time is on the order of a Gyr. So we need to be simulating for 50 times as long if the current rate holds up. That is $128*5*50=32,000$ cpu hours. That's not super tenable for generating ICs. Hopefully the rankfile will help, but it looks like it's only doubling the generation rate, instead of doing any better. Might be able to do better than 2-1 with a different rank file. Mike Grudic's example was using 4-1 I think.
#
# #### 02/25
#  - HABG problems and other might be related to softening length? but at the same time, the _initial_ velocity dispersion is clearly way different. I have no idea why though!
#  - I'm rerunning habg with substantially increased softening length (using the same as was being used for the stars). I also upped the max timestep, that seems to be speeding up the sim too, which is nice, but slightly concerning. 
#    - Definitely not the softening length. Sim still explodes. Though maybe faster than before? Probably a symptom of upping the max timestep
# #### 03/04
#  - BH influence radius ($r_h$) is $\sim100$ pc. The mfp is supposed to be significantly greater than that.
#  - Try running hires with 10^5 and 10^6 particles
#  - Send Manoj link to spheric github
#  - Add list of updates to spheric repo

# %%

# %% [markdown]
# # Todo List

# %% [markdown]
# (Not in any particular order)
#
# - [ ] Generate SIDM "inner" (sub 1 kpc) profile either through:
#    1. Adjusting the profile calculation in SpherIC _or_
#    2. Evolving halos separately, then combining (could also be used if needed for BH expansion)
# - [ ] Determine halo ICs
#    1. Masses - nanograv BHs are $\sim 10^8 - 10^9 \, M_{\odot}$ - 1810.04184 gives a BH to Bulge relationship of $M_{BH}=\mathcal{N}\{M_*\left(\frac{M_{bulge}}{10^11\;M_{\odot}}\right)^{\alpha_*}\}$ for $\log_{10}(M_*)=8.17_{-0.32}^{+0.35}$, $\alpha_*=1.01_{-0.10}^{+0.08}$, and $\epsilon=[0.3,0.5]$ and Bulge to stellar mass ratio of
#    
#      $$
#      \frac{M_{bulge}}{M_{stellar}} = \left\{
#      \begin{array}{ll}
#      \frac{\sqrt{6.9}}{(\log M-10)^{1.5}}\exp\left(\frac{-3.45}{\log M-10}\right) + 0.615 & \log M > 10 \\
#      0.615 & \log M <10 \\
#      \end{array}
#      \right.
#      $$
#
#    2. Number of particles - $10^5$?
#    3. Collision velocities
#    4. Initial separation
# - [ ] Correct Config flags/parameters
#    1. Also good time to test/switch to Igor's gizmo version
#    2. Need proper SIDM parameters
# - [ ] Final parsec problem - is this something we need to worry about or not? - Answer doesn't seem to be an issue?
#    1. But a decent portion of Cline's paper [2401.14450](https://www.arxiv.org/abs/2401.14450) is showing how SIDM solves the final parsec problem
# - [ ] Fix analysis code - am I calculating $\omega\rightarrow\frac{dE}{d\omega}\rightarrow h_c(\omega)$ correctly? - Probably not, continuing to work on this
# - [x] Figure out meetings with G and K - when2meet poll
# - [x] Apply for time on access
#    1. Applied for and received Explore Access grant
#    2. Applied for time (95k hrs) & storage (5 TB) on Bridges2 - still waiting on confirmation - chosen over ookami for ease of use (cmd & gui access, x86_64 toolchain, etc), storage
#    3. Need [x] G & K usernames to add 
# - [ ] Figure out how to divvy up work with G/K?
# - [x] Evolve NFW isolated halo longer - try to get to 10 Gyr
#    1. Tracking particle trajectories seems to show CDM particles are being scattered out of close SMBH neighborhood, causing deflection from NFW profile
# - [ ] Talk to Igor about dynamical friction, GW emission
#    1. Can turn on dynamical friction flags: 2 options `BH_DYNFRICTION` or `BH_DYNFRICTION_FROMTREE`. The FROMTREE option appears to be the one Igor was referring to - `BH_DYNFRICTION_FROMTREE` is the way to go for this, it's the more up to date and faster version
#    2. Thesis by Faheel Kahn
#    3. GW wave calculations might be unnecessary. Since GW emission is not really relevant until <$0.1$ pc ~ yr$^{-1}$, we can probably just run the sim until the orbits achieve that separation and then claim the GW takes over
# - [x] See if triaxial halos after merger - if not need to switch
#    1. Isolated NFW - no (probably as expected), also no significant change over time or radially
#    2. Merging halos - yes/no. Initial halos are spherical (SpherIC outputs spherical halos after all). Merged halo at small radii is - values lessen as radii increases
# - [ ] Compute $\frac{dL}{dt} \frac{t_{orbit}}{ \omega }$
# - [x] Add SIDM - probably also need to talk to Igor about
# - [ ] Try using Seaborn library for any statistical plotting

# %%
