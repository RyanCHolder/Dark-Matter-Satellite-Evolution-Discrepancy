#!/usr/bin/bash
# Stampede3
#SBATCH --job-name=Test2
#SBATCH --partition=skx-dev
#SBATCH -N 1 #number of nodes
#SBATCH -n 24   ## total MPI tasks , tasks/node*maxmemsize < max memory/node, have 48 cores/node on Stampede3 (should be 48*N)
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2:00:00
#SBATCH --mail-user=rholder@uci.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-type=begin
#SBATCH --account=TG-PHY250197

module purge #unloads all modules
module load intel impi phdf5 gsl fftw3

rundir=/path/to/example_rundir
CONTFLAG=0 # 0-(re)start 1-resume

if [ $CONTFLAG -ne 0 ]; then
  date >> $rundir/output/runtime.txt
else
  date > $rundir/output/runtime.txt
fi
ibrun $rundir/GIZMO $rundir/params.txt $CONTFLAG 1>$rundir/output/gizmo.out 2>$rundir/output/gizmo.err
date >> $rundir/output/runtime.txt