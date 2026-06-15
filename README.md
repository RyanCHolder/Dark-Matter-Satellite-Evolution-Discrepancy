# Dark-Matter-Satellite-Evolution-Discrepancy
Analyzing structural and numerical contributions to residuals between TNG-50-1-Dark subhalos and emperical models from isolated simulations.

Below are the details of reproducing the analyses of the TNG-50-1-Dark subhalos. We also began simulations of our own similar to the DASH library using the GIZMO simulation tool, scripts and info on using GIZMO are below as well.

# TNG Data Download

Firstly, you will need to get access to the TNG API to use the same scripts we did. You can register for an account to get an API key here: https://www.tng-project.org/users/register/
Once you have an account and an API key, you can request access to the Jupyter lab server here: https://www.tng-project.org/data/lab/

With both of these configured, downloading the same sample we did of TNG-50-1-Dark hosts, you will simply need to run the Jupyter notebook in this repository, TNG/sample_selection.ipynb
An explanation of the process this script goes through can be found in the thesis report as well as in the comments on this notebook.

Note at the end of this notebook there are a couple of additional functions that were written for analysing subhalo interactions with the environment and potential disturbances thereof, which may be of interest but the implications and validity of these parameters have not been thoroughly explored.

# TNG Data Analysis

To reproduce our analyses on the structural and numerical influences tidal tracks compared with existing models, download the "TNG/data" fold and go to "TNG/multi_analysis.py" where a number of functions have been created (with comments explaining what they do and how they work). By default, running this python script without any arguments will reproduce the plots generated in the thesis report (assuming you run it in the same directory as the "TNG/data" folder contained here. "TNG/data" contains all the data that comes out of running "TNG/sample_selection.ipynb" on the TNG Jupyter lab server.

The code for plotting the gNFW profile used in Figure 1 in the thesis report is found at the end of the "TNG/sample_selection.ipynb" file under a heading "Optional Function to Visualize gNFW Fit." This requires the TNG particle data, hence why it is intended for the notebook to go on the TNG server.

# Einstein Radius Calculations

An additional set of functions for calculating einstein radii is contained in "TNG/R_E_functions.py" which calculate einstein radii using particle data for the subhalo, catalog data, and catalog data corrected by the Green and van den Bosch 2019 model. Note, while these functions will work in any environment, passing in the particle data will require accessing and/or downloading particle data through the TNG API. Please see the "TNG/sample_selection.ipynb" code to see how to access these data.

These einstein radius calculations are currently in an in-progress state and their results are not confirmed to be accurate, corrections to these functions are likely necessary.

# GIZMO and SPHERIC

To run simulations similar to the DASH library used for the Green and van den Bosch 2019 study, we used a combination of SPHERIC to generate initial conditions, and GIZMO to run our own N-body simulation.

Firstly, to download the latest version of GIZMO, go here: https://github.com/pfhopkins/gizmo-public

Second, you need to edit the configuration files to correspond to the type of simulation are trying to run: dark matter only with a single host potential. You will find in the GIZMO folder in this repository three files, GIZMO_config.h, config.sh, and analytic_gravity.h. GIZMO_config.h and config.sh should replace the default files from the downloaded gizmo-public repository to follow our desired simulation. analytic_gravity.h is contained in the gravity folder of the gizmo-public repository, our version contains two additional functions, GravAccel_DynamicNFW and GravAccel_TriaxialNFW. The triaxialNFW function is intended to model an elliptical host, but it is in need of modification to properly handle a triaxial host. The DynamicNFW function is working and will enable you to pass an array of M200 masses and concentrations with a corresponding array of times, the host potential will then interpolate between these values to create a time-evolving host.

To actually run GIZMO, copy the GIZMO/example_rundir. You will need to copy your compiled version of the GIZMO file into this folder (replace the placeholder one there currently), and modify the params.txt and run.sbatch files. In the params.txt, the relevant things to change are as follows:

1. Replace the path for InitCondFile to the path of your actual initial condition (do not include the .hdf5 extension). Instructions on generating this initial condition with SPHERIC will follow.
2. Replace the TimeMax with your desired length for the simulation, the unit is gigayears. Currently it is set to 15 Gyr (about the same as TNG-50)
3. Replace OutputDir with the path to where you want the output text and snapshot files to go
4. Replace TimeBetSnapshot with your desired interval between snapshots, current setting is 0.2 Gyr
5. If potential memory issues occur, you may want to reduce the MaxMemSize parameter

No in the run.sbatch command (this is designed for running on stampede3, on a different system you will have to make modifications)
1. Replace --mail-user and --account= with your login email and allocation account id
2. Change rundir to the path of your run directory
3. Then, on stampede3, run gizmo by calling sbatch run.sbatch:
4. If you are running a longer simulation (currently it is set to a maximum time of 2 hours): delete or comment out the --partition, it is currently set to a partition with a 2 hour maximum to reduce wait time; replace --time with your desired maximum run time.

Lastly for running SPHERIC, it is easiest to download the "GIZMO/spheric" folder here as this contains additional functions that fix a significant bug and facilitate easily generating initial conditions. The original code can also be found online here: https://bitbucket.org/migroch/spheric/src/main/. Our code has a major bugfix, as setting a nonzero position or velocity for the subhalo puts the subhalo out of equilibrium. The easiest way to create an initial condition is to use the spheric_gen_simple.py script. Just go into this script, modify the mass, scale radius, position, velocity, and output file parameters and run the script. Make sure that you run this in the directory it is set up in here containing spheric itself and the pv_fix.py, which is the script to fix the initial condition issue.

Another option is to use the spheric_gen.py script. This takes in a few command line arguments detailed within the file. It also expects a csv file of data for the subhalos, an example of such a csv is provided in GIZMO/sub_init_cond.csv. The units of the values in this csv are also detailed in GIZMO/sub_init_cond_units.json to help you in creating such a CSV. The data for this example CSV are taken from TNG, but we recommend you select data yourself as the conditions of the subhalos used here may not be ideal depending on your purposes, but they could serve as suitable test cases. 

Copy and paste the output hdf5 file into your run directory for GIZMO, make sure params.txt in your run directory has the correct file specified (again without the .hdf5 extension). This should run your simulation.

# References and Data Sources

The data contained in the "TNG" folder are from the IllustrisTNG public dataset which can be accessed from their website: https://www.tng-project.org/data/

We use code that is not ours from Philip Hopkins for GIZMO, again it is from this GitHub: https://github.com/pfhopkins/gizmo-public; and the spherIC code from Miguel Rocha: https://bitbucket.org/migroch/spheric/src/main/

------

If you have any questions, please reach out to me at rcholder@umich.edu, or contact me through the PI on this project, mkapling@uci.edu
