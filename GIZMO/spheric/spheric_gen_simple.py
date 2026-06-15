import h5py
import spheric
import pv_fix
import pandas as pd

# generates an hdf5 initial condition for the specified parameters

out_file = "file_name"
Mhalo = 1 # subhalo mass, 1e10 Msol
rs = 1 # subhalo scale radius, kpc
pos = (500,200,300) # subhalo position, 1x3, kpc
vel = (100,200,-100) # subhalo velocity, 1x3, km/s

#generate spheric ic
s = spheric.SphericOptions(
    name = out_file,
    ogh = True,
    Mhalo = Mhalo,
    rs = rs
)
s.generateOptionString()
spheric.spheric(s)

pv_fix.GIZMO_fix(out_file, pos, vel)