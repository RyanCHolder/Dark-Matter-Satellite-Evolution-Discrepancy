import inspect
import numbers
import subprocess
import numpy as np
import copy
from pathlib import Path

class SphericOptions:
    halo = True        # generate a dark matter halo
    Nhalo = 1e5        # number of halo particles
    Mhalo = 1          # total mass of the halo (default Mhalo = 1)
    alpha = 1          # alpha parameter in the halo density profile
    beta = 3           # beta parameter in the halo density profile
    gamma = 1          # gamma parameter in the halo density profile
    rs = 1             # scale radius (default rs = 1)
    rcutoff = 100      # cutoff radius for cutoff halo models (i.e. beta >= 3)
    king = False       # generate a stellar component with a King profile
    hernquist = False  # generate a stellar component with a Hernquist profile
    plummer = False    # generate a stellar component with a Plummer profile
    starabg = False    # generate a stellar component with an (alpha,beta,gamma) profile
    Nstar = 0          # number of star particles
    Mstar = 0          # total stellar mass 
    rc = np.nan        # king core radius
    rt = np.nan        # king tidal radius
    rhern = np.nan     # scale radius for Hernquist profile
    rp = np.nan        # scale radius for Plummer profile
    star_alpha = 1     # alpha parameter in the halo density profile
    star_beta = 3      # beta parameter in the halo density profile
    star_gamma = 1     # gamma parameter in the halo density profile
    star_rs = 1        # scale radius (default rs = 1)
    star_rcutoff = 100 # cutoff radius for cutoff halo models (i.e. beta >= 3)
    MBH = 0            # mass of black hole (default MBH = 0)
    name = ""          # name of the output file
    dx,dy,dz = 0,0,0   # position offset for the initial conditions (default All = 0)
    dvx,dvy,dvz = 0,0,0# velocity offset for the initial conditions (default All = 0)
    ogr = False        # set this flag for outputting grid in r in an ASCII file
    ogdf = False       # set this flag for outputting grid for distribution function in an ASCII file
    ogb = False        # set this flag for generating a GADGET2 initial conditions binary file
    ogh = False        # set this flag for generating a GIZMO initial conditions HDF5 file
    otb = False        # set this flag for generating a TIPSY initial conditions binary file
    oift = False       # set this flag to write positions for IFRIT binary file 
    opfs = False       # set this flag to write a table of density profiles in an ASCII file
    nostarpot = False  # set this flag for excluding the stellar potential 
    randomseed = -1     # set this flag for setting a value for a random seed (default: random value)
    dorvirexact = False# set this flag for calculating rvir exactly via N^2 sum - Warning: time consuming for large N!
    
    
    def __init__(self,*,name=None,randomseed=-1,**kwargs):
        if name is None:
            name = "runs/IC"
        self.name = name
        self.randomseed = randomseed
        self.__dict__.update(kwargs)
        if self.Nstar < 1:
            self.Mstar = 0

    def __repr__(self):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        s = "SpericOptions("
        for a in attributes:
            match a:
                case (_,str()):
                    s = s + f'{a[0]}="{a[1]}",'
                case _:
                    s = s + f"{a[0]}={a[1]},"
        s = s + ")"
        return s

    def copy(self,**kwargs):
      s = copy.copy(self)
      s.__dict__.update(kwargs)
      return s
  
    def generateOptionString(self):
        # Determine attributes programmatically since maybe options will change in future
        # Code pulled from https://stackoverflow.com/a/9058322
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        optStr = ""
        for a in attributes:
            match a:
                case (_,None):
                    pass
                case (_,bool()):
                    if a[1]:
                        optStr = optStr + f"-{a[0]} "
                case ("a"|"alpha",x):
                    optStr = optStr + f"-a {a[1]} "
                case ("b"|"beta",x):
                    optStr = optStr + f"-b {a[1]} "
                case ("c"|"gamma",x):
                    optStr = optStr + f"-c {a[1]} "
                case ("star_alpha",x): # probably a better way of dealing with this
                    optStr = optStr + f"-as {a[1]} "
                case ("star_beta",x):
                    optStr = optStr + f"-bs {a[1]} "
                case ("star_gamma",x):
                    optStr = optStr + f"-cs {a[1]} "
                case ("star_rs",x):
                    optStr = optStr + f"-rss {a[1]} "
                case ("star_rcutoff",x):
                    optStr = optStr + f"-rcuts {a[1]} "
                #case ("Mhalo",1):
                #    pass
                case ("randomseed", -1):
                    pass
                case (("MBH"|"Nstar"|"Mstar"|"dx"|"dy"|"dz"|"dvx"|"dvy"|"dvz"),0): # Deal with default 0 values
                    pass
                case (_,numbers.Number()):
                    if not np.isnan(a[1]):
                        optStr = optStr + f"-{a[0]} {a[1]} "
                case _:
                    optStr = optStr + f"-{a[0]} {a[1]} "
        return optStr

import warnings
def spheric(opts=None):
    if opts is None:
        opts = SphericOptions()
    cmd = "./spheric"
    args = opts.generateOptionString()
    p = subprocess.run([cmd,*args.split()],capture_output=True)
    if p.returncode:
        warnings.warn(f'SpherIC crashed (returned error code {p.returncode}). Check from command line to see any additional info')
        warnings.warn('Note if you get an error about hdf5 version mismatch and you are in a conda environment,'
                      ' try deactivating (possibly reactivating) and recompiling. Sometimes that fixes it')
        return p
    # This generates a filename called {opts.name}.out which is a text file. We'll change it to {opts.name}_out.txt
    print(f"Changing filename {opts.name}.out to {opts.name}_out.txt")
    Path(f"{opts.name}.out").rename(f"{opts.name}_out.txt")
    return p