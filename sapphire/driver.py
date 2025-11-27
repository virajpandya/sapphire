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

    # just to benchmark entire runtime from start to finish
    tstart0 = timer()
    
    
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
    ### Can this now be done afterwards using jax_config below?
    if config['num_cpus'] > 1:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=%s"%config['num_cpus']

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

    ### decide how to run: single set of parameters or sampling many params simultaneously
    from sapphire.utils import setup_parameters 

    if config['runtype'] not in ['single','sampling','inference']:
        raise ValueError('config.runtype must be one of single, sampling or inference')
        
    else:
        # this module will automatically return params based on runtype
        param_samples = setup_parameters.get(config)

    ### now set up the ODE solver to run on single parameter set or batched parameter set 
    if config['solver_config']['engine'] == 'diffrax':
        from sapphire.solvers import diffrax as solver
    else:
        raise ValueError('other diffeq solver engines are not yet implemented') 
    
    batch_solve = solver.setup(config,integrator,saveat_fn,halo_matrix,halo_coeff_matrix,halo_tinit,ts_interp)
    
    ### automatically run 
    print('benchmarking full-batch ODE runtime for runtype=%s...'%config['runtype'],flush=True) 

    ### Finally solve the ODEs for single or multiple halos
    # this is clunky, can push the solve down to sapphire.solvers.diffrax itself, returning batch_solve only for inference later 

    ### TO DO: remove this in favor of minibatched below? and/or can push down to future TSG/ILI module otherwise?
    
    tstart = timer()
    sol = batch_solve(halo_index,param_samples)
    print(sol.ys[0][0][0],flush=True)
    print('full-batch initial jit+sol took %.3f sec'%(timer()-tstart),flush=True)
    
    tstart = timer()
    sol = batch_solve(halo_index,param_samples)
    print(sol.ys[0][0][0],flush=True)
    print('full-batch jitted sol took %.3f sec'%(timer()-tstart),flush=True)

    """ 
    add module here to compress+save data directly for TSG/ILI 
    """
    
    if config['runtype'] in ['single','sampling']: 
        print('returning sol...',flush=True)
        return sol
    
    ### alternatively run inference if requested
    ######### Can this whole thing be put into a sapphire.inference wrapper module to keep driver / __ clean?
    if config['runtype'] in ['inference']: # can merge Lucas' ILI option in the future

        from sapphire import inference
        
        inference_config = config['inference_config']

        params_fixed_astro = config['params_fixed_astro']
        params_bounds = config['sampling_config']['params_bounds']
        params_free = list(params_bounds.keys())

        print('[minibatched] solving mock ODEs...',flush=True)
        ### move this earlier -- maybe to read_trees or a utils module ?
        minibatch_halo_index = jax.random.choice(jax.random.key(0), halo_index, (inference_config['Nbatch'],), replace=False)
        
        ### mock mode 
        ### push this down to sapphire.utils.setup_parameters 
        
        true_params = jnp.full(len(params_free),jnp.nan) # by default, there is no truth if not running in mock mode
        
        if inference_config['mock'] is True:

            ##### TO DO: replace this with either using input user mock params, or random lhs single w/ rng_key
            ##### TO DO: push this down to utils.setup_params ?
            true_params = jnp.array([params_fixed_astro[k] for k in params_free])
            print('true mock parameters\n',true_params,flush=True)

            # note: batch_solve requires the full input parameters, not just subset true_params
            tstart = timer()
            mocksol = batch_solve(minibatch_halo_index,param_samples)
            print(mocksol.ys[0][0][0],flush=True)
            print('[minibatched] initial jit+sol took %.3f sec'%(timer()-tstart),flush=True)
            
            tstart = timer()
            mocksol = batch_solve(minibatch_halo_index,param_samples)
            print(mocksol.ys[0][0][0],flush=True)
            print('[minibatched] jitted sol took %.3f sec'%(timer()-tstart),flush=True)            
            
            print('summarizing mock data...',flush=True)
            # TO DO: push this import earlier, and use __init__ to load only summaries parent module
            import sapphire.summaries.gaussian_kernel_regression as gkr # can generalize later

            # although these are mocks, we use the same obs_ prefix for consistency with rest of code below
            # NOTE: return order of obs_stats is same order as expected by sapphire.inference module below
            obs_stats = gkr.summarize_mock(mocksol)

        """ otherwise add module to summarize input observations here """
        ### obs_stats = gkr.summarize_obs(config) 
        
        print('setting up model for inference...',flush=True)

        ### should push this down to sapphire.inference based on config, for explicit or ILI etc. 
        loss_func, grad_loss_func, hess_loss_func = inference.explicit_likelihood.setup(config,minibatch_halo_index,obs_stats,batch_solve)

        ### change this to also benchmark for non-mock (fitting obs)
        ### and push this down to sapphire.utils or a new inference.coverage module or something 
        if inference_config['mock'] is True:
            tstart = timer()
            true_loss = loss_func(true_params)
            tjit_loss = timer()-tstart
            
            tstart = timer()
            true_loss = loss_func(true_params)
            tloss = timer()-tstart

            print('[minibatched] true loss=%s, jit %.5f sec, post-jit %.5f sec'%(true_loss,tjit_loss,tloss),flush=True)

            tstart = timer()
            true_grad_loss = grad_loss_func(true_params)
            tjit_grad = timer()-tstart

            tstart = timer()
            true_grad_loss = grad_loss_func(true_params)
            tgrad = timer()-tstart            
            
            print('[minibatched] true grad loss=%s, jit %.5f sec, post-jit %.5f sec'%(true_grad_loss,tjit_grad,tgrad),flush=True)

            tstart = timer()
            true_hess_loss = hess_loss_func(true_params)
            tjit_hess = timer()-tstart

            tstart = timer()
            true_hess_loss = hess_loss_func(true_params)
            thess = timer()-tstart            
            
            print('[minibatched] true hess loss=%s, jit %.5f sec, post-jit %.5f sec'%(true_hess_loss,tjit_hess,thess),flush=True)

            true_hess_flag = jnp.all(jnp.linalg.eigvalsh(true_hess_loss)>0)
            print('true_hess_flag',true_hess_flag,flush=True)
            # return true_hess_flag

            true_Finv = jnp.linalg.inv(true_hess_loss)
            print('true_Finv',true_Finv,flush=True)

        ####### TO DO: push this to a unit test somewhere
        # params_bounds = config['sampling_config']['params_bounds']
        # params_free = list(params_bounds.keys())

        # print('param_samples\n',param_samples,flush=True)

        # test_params = jnp.array([param_samples[0],param_samples[1],
        #                          param_samples[4],param_samples[5],
        #                          param_samples[8],param_samples[9],
        #                          param_samples[12],param_samples[13]])

        # test_params = test_params.at[0].set(jnp.log10(test_params[0]))
        # test_params = test_params.at[2].set(jnp.log10(test_params[2]))
        # test_params = test_params.at[4].set(jnp.log10(test_params[4]))
        # test_params = test_params.at[6].set(jnp.log10(test_params[6]))        
        
        # print('test_params\n',test_params,flush=True)

        
        # test_params_dict = {params_free[i]:test_params[i] for i in range(len(params_free))}
        # print('test_params_dict\n',test_params_dict,flush=True)

        
        # if inference_config['backend'] == 'numpyro':
        #     test_loss = loss_func(test_params_dict)
        #     test_grads = grad_loss_func(test_params_dict)
        # elif inference_config['backend'] == 'manual':
        #     test_loss = loss_func(test_params)
        #     test_grads = grad_loss_func(test_params)
        
        # print('test_loss',test_loss,flush=True)
        # print('test_grads\n',test_grads,flush=True)
        
        # return test_grads
        
        if config['inference_config']['engine'] == 'adam':
            print('calling adam for MAP+Fisher...',flush=True)

            out_adam = inference.run_adam.setup(config,loss_func,grad_loss_func)

            ### compute MAP and fisher/covariance matrix
            out_best = inference.map_fisher.from_adam(config,hess_loss_func,out_adam,true_params)

            # return out_best
                

            ### figure making / save results if requested

        

        elif config['inference_config']['engine'] == 'hmc':
            print('calling numpyro for HMC...',flush=True)

        
        else:
            raise ValueError('inference_config.engine must be either adam or hmc',flush=True)
    

    ### somehow have this total runtime always print upon program exit, even if return was up above somewhere 
    print('Finished in %.1f min -- thank you for using sapphire!'%((timer()-tstart0)/60.),flush=True)

    
