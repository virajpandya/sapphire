"""
sapphire -- a next-generation multi-zone dynamical model of galaxy formation 

This is the main driver module that takes an input parameter dictionary or JSON config filepath 
and then calls the relevant package modules in the order required. 
"""

# load general modules unrelated to jax
import os, sys

# set for jax on GPU, doesn't affect CPU
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' 

from glob import glob
import multiprocess
from functools import partial 
import numpy as np
import pandas as pd
import seaborn as sns
import arviz as az
from chainconsumer import Chain, ChainConsumer, make_sample, PlotConfig, Truth, ChainConfig
import argparse
import json
import h5py 
from timeit import default_timer as timer

import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['figure.dpi'] = 120
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

# load sapphire modules that do not require dependency injection
# from .coolfunc import read_coolfunc 
# from .utils import writer 

# NOTE: this should be further modularized as needed, including any dependency injections (loading of modules based on config dict)
def run(config):
    """
    driver module that calls all the other submodules in the required order.
    
    config: user-provided dict or JSON filename giving model/runtime config 
    
    NOTE: at some point I plan to offload a lot of the if/else checks on the config dict to another module 
    """
    
    # FIRST: parse the input config object (must be a dict or a string giving the path of config JSON file that will be converted to dict)
    # print('Parsing input config...',flush=True)
    if type(config) == dict: # later will add an argparse util module for making sure the provided dict is sensible
        pass 
    elif type(config) == str: 
        raise NotImplementedError('JSON config filepath option not yet implemented')
    else:
        raise ValueError('config must be provided as either a python dict or JSON filename')

    print('your requested config:\n',config,flush=True)
    
    # if multiple CPU cores requested, set environment flag (idiosyncrasy of jax on CPUs)
    if config['num_cpus'] > 1:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=50"

    ### now load jax
    # NOTE: there has to be a cleaner, compact way to load all of these jax packages up top... 
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True) # required to accurately solve and take gradients through our diffeqs 

    import jax    
    import jax.numpy as jnp
    print('jax version',jax.__version__,flush=True)
    
    # # if output directory does not already exist, create it
    # if os.path.exists(config['output_path']) == False:
    #     os.mkdir(config['output_path'])
    
    # load the relevant tree reading module
    if config['tree_type'] == 'tng':
        from sapphire.trees import jax_read_tng as tree_reader    
    else: # NOTE: glob and print a list of available tree_type strings based on modules available in read_trees subdirectory
        raise ValueError('tree_type must be one of the types implemented in the trees module') 
        
    # now read the trees into a dict where the key is halo name/ID and element is an astropy Table
    # read and interpolate into the required jax halo matrix format
    print('Reading trees and converting to jax matrices...',flush=True)
    halo_matrix, halo_coeff_matrix, halo_tinit, halo_index, ts_interp = tree_reader.read_trees(config)
        
    # use dependency injection to read in the requested physical model, then the associated parameter functions using dependency injection
    # NOTE: this parsing / dependency injection will eventually be moved to another utility module 

    if config['model'] == 'jax_thermal':
        from sapphire.models import jax_thermal as model
    else:
        raise ValueError('you must enter the name of an existing model within the sapphire models module') 

    integrator, saveat_fn = model.setup(config)

    ### decide how to run: single fixed parameter set, sampling, or inference
    from sapphire.utils import setup_parameters 

    if config['runtype'] not in ['single','sampling','inference']:
        raise ValueError('config.runtype must be one of single, sampling or inference')

    
    elif config['runtype'] == 'single': 
        # user requested only a single run with input fixed parameters, so turn them into the jnp.array with 10**A_X, etc.
        # note: batch_solve above should have already been configured internally for this single-run mode 
        param_samples = setup_parameters.get(config)
        
    elif config['runtype'] == 'sampling':
        # use requested sampling strategy to choose N samples of parameters
        param_samples = setup_parameters.get(config)

    ### now set up the ODE solver
    if config['solver_config']['engine'] == 'diffrax':
        from sapphire.solvers import diffrax as solver
    else:
        raise ValueError('other diffeq solver engines are not yet implemented') 
    
    batch_solve = solver.setup(config,integrator,saveat_fn,halo_matrix,halo_coeff_matrix,halo_tinit,ts_interp)

    ### Finally solve the ODEs for single or multiple halos
    # this is clunky, can push the solve down to sapphire.solvers.diffrax itself, returning batch_solve only for inference later 
    if config['runtype'] in ['single','sampling']: 
        print('solving ODEs for runtype=%s...'%config['runtype'],flush=True) 
        
        ### first for full batch without commenting out vmap
        tstart = timer()
        sol = batch_solve(halo_index,param_samples)
        print(sol.ys[0][0][0],flush=True)
        print('initial sol took %.3f sec'%(timer()-tstart),flush=True)
        
        tstart = timer()
        sol = batch_solve(halo_index,param_samples)
        print(sol.ys[0][0][0],flush=True)
        print('jitted sol took %.3f sec'%(timer()-tstart),flush=True)
        

    ### alternatively run inference if requested
    elif config['runtype'] in ['inference']:

        if config['inference_config']['engine'] == 'adam':
            print('calling adam for MAP+Fisher...')

        elif config['inference_config']['engine'] == 'hmc':
            print('calling numpyro for HMC...')

        
        else:
            raise ValueError('inference_config.engine must be either adam or hmc')
    



        
    # print('Finished -- thank you for using sapphire!',flush=True)

    
