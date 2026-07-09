### Viraj Pandya - July 2026 

This directory contains ipynbs and scripts needed to reproduce the analysis and figures of Pandya+26. 

==========================================================================================

plotting/ 

- contains Jupyter notebooks to reproduce every figure of Pandya+26 
- the ipynb filenames are self-explanatory
- many ipynbs read in data saved after completion of expensive inference routines
    - inference is time-consuming and must be run separately using the scripts/ folder (see below)
    - some data like HMC posteriors are saved in ../../data/posteriors/pandya26/

==========================================================================================

scripts/fit_obs/ and scripts/fit_mocks/

The remainder of the files/subdirectories are for inference with adam and numpyro
- there are 3 kinds of files separated into fit_mocks vs. fit_obs subdirectories 
    - config.yaml specifies sapphire configuration (different for mocks vs obs, and variations within each below)
    - disbatch_tasks uses Flatiron Institute's disBatch (https://github.com/flatironinstitute/disBatch) to automate mocks / HMC chains 
    - sub_disbatch.sh is the actual slurm job submission script 

Pandya+26 deliberately carried out a simplified proof-of-concept analysis (w/ future extensibility in mind): 
- we always fit the same free parameters drawn from fixed random priors 
- we only fit to z=0 and 3 quasi-observables which minimizes the # of config variations
- for mock tests, we assume constant mock errors and systematic bias shifts 
- the loss function was essentially hard-coded to deal with all above variations 

config.yaml is the main file that needs to be repeatedly modified:
- what combos of constraints to fit (SMHM, fgas, MZR)
    - these are toggled by the flag_XXX at the bottom of config.yaml
    - for example flag_smhm=1 with all others to 0 will only fit the SMHM relation
    - Pandya+26 did all 7 flag combos for the no measurement uncertainty case, then fit all 3 together w/ varying errors (see next)
- what uncertainty to assume 
    - these are set by mock_err_XXX which get added in quadrature to the intrinsic standard error from Nadaraya-Watson regression
    - the defaults are 0.1 dex for SMHM, 0.2 dex for fgas, 0.3 dex for MZR (constant vs. mass for simplicity)
    - for the case of fitting all 3 constraints together, two additional cases involve cutting all 3 errors by a factor of 2 or 10
- whether you are fitting observations or mocks
    - this is decided by either fit_mock=True or fit_obs=True (only one or the other)
    - if fitting mocks, also set mock_num, rng_sample, rng_init, etc. (this is automated by slurm scripts below)
- output_suffix to summarize the variations above in the filename
    - by default this will append the 3 flags to the out filename but you can/should modify this for different sets 

fit_mocks/disbatch_tasks_adam 
- automates N slurm jobs for N mocks, each with a different random seed for drawing mock parameters 
- uses sapphire's command line mode to point to a config.yaml with different mock_num 
- should be resubmitted for as many different config files you have
    - change the config filename AND the output log filename in this file

fit_mocks/sub_disbatch_adam.sh 
- the actual slurm job submission script 
- run it by doing "sbatch sub_disbatch_adam.sh" on the command line on a cluster
- the slurm options at the top are tailored for the Flatiron rusty cluster and should be modified accordingly for other clusters
- the directory you start this from will have lots of slurm and disbatch log files generated that you can delete after
    - the main useful files are the piped .log files specified in disbatch_tasks_adam, use these to monitor progress, errors, etc. 

fit_obs/sub_adam.sh
- analogous to above but for fitting a single data combo of quasi-observables (rather than N mocks across parameter space) 
- uses command line mode with path to fit_obs/config_obs_adam.yaml 
- re-run multiple times with variations of config_obs_adam.yaml for flag_XXX and scale_err_XXX 

fit_obs/disbatch_tasks_numpyro 
- uses disBatch to run sapphire on command line for N chains on N different nodes 
- use with fit_obs/config_obs_hmc.yaml
- re-run multiple times with variations of config_obs_hmc.yaml for flag_XXX, scale_err_XXX, shift_XXX

fit_obs/sub_disbatch_numpyro.sh 
- slurm job submission script to request N nodes for N disBatch tasks defined in fit_obs/disbatch_tasks_numpyro 


==========================================================================================

scripts/benchmark_runtime/

config_benchmark.yaml
- similar to above except with runtype=benchmark (sapphire/utils/benchmark_runtime.py does the rest) 

sub_benchmark.sh 
- this calls sapphire on the command line with path to config_benchmark.yaml 
- needs to be run multiple times for different architectures 
    - change the slurm config options at the top accordingly for CPU vs 1 GPU vs N GPUs 
    - for gpu, make sure suitable JAX mamba environment is activated


#