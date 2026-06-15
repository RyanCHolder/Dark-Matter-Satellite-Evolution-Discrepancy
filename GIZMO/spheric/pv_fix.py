import numpy as np
import h5py

#helper function to fix the initial setting of positions and velocities causing a bug in spheric
#manually adds a position and velocity to every particle

def GIZMO_fix(file_name,pos,vel):
    file_name = file_name + "-gizmo.hdf5"

    with h5py.File(file_name, "r+") as f:
        #adjust position and velocity
        coords = f["PartType1/Coordinates"][:]
        vels = f["PartType1/Velocities"][:]
        f["PartType1/Coordinates"][:] = coords + pos
        f["PartType1/Velocities"][:] = vels + vel
