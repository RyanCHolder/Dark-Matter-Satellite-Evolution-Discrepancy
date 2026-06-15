import h5py
import spheric
import pv_fix
import pandas as pd
import numpy as np
import argparse

# load the following modules to run this script:
# module load hdf5 gsl

# === Command-line argument parsing ===
parser = argparse.ArgumentParser(description="Generate GIZMO ICs for a halo from CSV.")
parser.add_argument(
    "--tree_id",
    type=int,
    default=9,
    help="Tree ID of the halo to generate (default: 9)",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/work2/10833/rholder/stampede3/ics/data/sub_init_cond.csv",
    help="CSV file containing halo data (default: /work2/10833/rholder/stampede3/ics/data/sub_init_cond.csv)",
)
parser.add_argument(
    "--file_name",
    type=str,
    default=None,
    help="Specify output filename, do not include extension (default: /work2/10833/rholder/stampede3/ics/tng-subs/TreeId{tree_id})"
)
args = parser.parse_args()

# === file name default depends on tree_id argument ===
if args.file_name is None:
    args.file_name = f"/work2/10833/rholder/stampede3/ics/tng-subs/TreeId{args.tree_id}"

#get values from command line
tree_id = args.tree_id
out_file = args.file_name
sub_data_path = args.data_path

#load data file
df = pd.read_csv(sub_data_path)

# Find the row corresponding to the given tree_id
halo_row = df.loc[df["tree_id"] == tree_id].squeeze()

#check that tree_id is a valid id in the data
if halo_row.empty:
    raise ValueError(f"No entry found for tree_id {tree_id}")

# gather subhalo parameters
Mhalo = halo_row["mass"]
rs = halo_row["rs"]

try:
    pos = [float(halo_row["pos_x"]), float(halo_row["pos_y"]), float(halo_row["pos_z"])]
    vel = [float(halo_row["vel_x"]), float(halo_row["vel_y"]), float(halo_row["vel_z"])]
except KeyError as e:
    raise KeyError(f"Missing expected position or velocity column: {e}")

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
