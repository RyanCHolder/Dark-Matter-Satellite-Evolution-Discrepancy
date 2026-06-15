# Readme

This is a basic Readme for the SpherIC package, written by Miguel Rocha. 
If you use this software, please cite [10.1093/mnras/stt984](https://doi.org/10.1093/mnras/stt984)
aka [1301.3137](https://arxiv.org/abs/1301.3137).

This version of SpherIC contains edits by Michael Ryan (mryan4@uci.edu). 

For the main SpherIC documentation, see doc/spherIC.pdf.

# Primary edits
1. Added gizmo hdf5 output (using the `-ogh` flag). If `MBH`>0, this will 
	include a central Black Hole (as `PartType5` particle). Since the gadget
	binary file does not include one, this will lead to a difference in the
	initial particle velocities seen by GIZMO (as seen by comparing 
	snapshot_000).
2. Black hole particle affected by translation (the `-dx/dy/dz` and 
	`-dvx/dvy/dvz` arguments)
3. Fixed gadget binary not being written under certain compilers due to 
	improper use of assert
4. **EXPERIMENTAL** Added generic $(\alpha,\beta,\gamma)$-type profile
    for stellar particles. This is in active development and may not work
    correctly. It currently seems to be able to reproduce the Hernquist and
    Plummer output profiles correctly but YMMV. The arguments for using this
    profile are `-starabg`, `-as #`, `-bs #`, `-cs #`, `-rss #`, and `-rcuts #`
    with the same meanings as for the Halo inputs.
5. Added `spheric.py` to run SpherIC from within python. The python file
    contains the `SphericOptions` class and the `spheric([options])` function.
    Please read the file for more information about the options.

**WARNING**: The gizmo output does not currently support the `-nostarpot`
option.

Some additional changes have been made, mostly for bugfixes.

# Compiling Notes
Since we now include hdf5 output, the hdf5 libraries need to be included. This
has been accomplished by switching the default compiler from gcc to h5cc. This
doesn't work on the Bridges2 cluster, so some logic has been added to the 
Makefile to test and account for running SpherIC on that system. If you develop
a workaraound for another system that needs special Makefile instructions, 
please submit a PR or send me a message, and I'll try to add it in to the main
branch.

Note that depending on your compiler (for example, the version of `gcc` 
provided in conda), the `assert(expression)` statement may or may not actually
run `expression`. Since much of the output files were written using 
`assert(fwrite(...) == # of writes)`, this can result in SpherIC writing 0-byte
output files. This has been "fixed" for the gadget binary output and is not an
issue for the gizmo output, but does still affect e.g. the ascii output. 
