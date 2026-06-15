#!/usr/bin/bash

# ---------------------------------------
# Settings
# ---------------------------------------
rundir=/scratch/10833/rholder/rundir/
CONTFLAG=0   # 0 = (re)start, 1 = resume

# Ensure output directory exists
mkdir -p "$rundir/output"

# Load modules if needed (depends on your system)
module purge
module load intel impi phdf5 gsl fftw3

# ---------------------------------------
# Record start time
# ---------------------------------------
if [ $CONTFLAG -ne 0 ]; then
  date >> "$rundir/output/runtime.txt"
else
  date > "$rundir/output/runtime.txt"
fi

# ---------------------------------------
# Run GIZMO
# ---------------------------------------
# Replace 'ibrun' with 'mpirun' for local/interactive runs:
mpirun -n 24 "$rundir/GIZMO" "$rundir/params.txt" $CONTFLAG \
    1>"$rundir/output/gizmo.out" \
    2>"$rundir/output/gizmo.err"

# ---------------------------------------
# Record end time
# ---------------------------------------
date >> "$rundir/output/runtime.txt"
