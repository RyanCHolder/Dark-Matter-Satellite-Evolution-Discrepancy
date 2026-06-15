# Dark-Matter-Satellite-Evolution-Discrepancy
Analyzing structural and numerical contributions to residuals between TNG-50-1-Dark subhalos and emperical models from isolated simulations.

Below are the details of reproducing the analyses of the TNG-50-1-Dark subhalos. We also began simulations of our own similar to the DASH library using the GIZMO simulation tool, scripts and info on using GIZMO are below as well.

# TNG Data Download

Firstly, you will need to get access to the TNG API to use the same scripts we did. You can register for an account to get an API key here: https://www.tng-project.org/users/register/
Once you have an account and an API key, you can request access to the Jupyter lab server here: https://www.tng-project.org/data/lab/

With both of these configured, downloading the same sample we did of TNG-50-1-Dark hosts, you will simply need to run the Jupyter notebook in this repository, TNG/sample_download.ipynb
Note at the end of this notebook there are a couple of additional functions that were written for analysing subhalo interactions with the environment and potential disturbances thereof, which may be of interest but the implications and validity of these parameters have not been thoroughly explored.
