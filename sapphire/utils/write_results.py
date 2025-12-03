"""
This module includes writer functions that will be loaded based on the types of trees being run on.
For large-volume simulations with multiple indivdual subvolumes, output files will be split with a
subvolume at the end of the filename. For small numbers of halos from zoom simulations, we can save 
all outputs into a single output file.
"""

import numpy as np 
import os, shutil
import multiprocess 
import pandas as pd
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 
from functools import partial 

# in case user loads module separately from sapphire.run()
from jax import config as jax_config
jax_config.update("jax_enable_x64", True) # required to accurately solve and take gradients through our diffeqs 

import jax
import jax.numpy as jnp
from jax._src.third_party.scipy.interpolate import RegularGridInterpolator as jax_RegularGridInterpolator
from jax import jit, grad, vmap, pmap, debug, jvp, vjp, jacrev, jacfwd, make_jaxpr, hessian, value_and_grad
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline    
from jax.experimental.ode import odeint
from jax.lax import fori_loop, while_loop
from jax.scipy.integrate import trapezoid
from jax.random import PRNGKey, key    
from diffrax import diffeqsolve, ODETerm, PIDController, SaveAt, Kvaerno3, Bosh3, Dopri5, Tsit5, DirectAdjoint, RecursiveCheckpointAdjoint, BacksolveAdjoint
from diffrax import backward_hermite_coefficients, CubicInterpolation    
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec, PositionalSharding, NamedSharding



# this creates output subdirectories if they don't exist 
def create_output_subdirs(config):

    if 'output_path' not in config.keys() or config['output_path'] == '':
        print('output_path not provided so will not write results...',flush=True)
        return None

    ### this can probably be cleaned up with a for loop over subdir names
    for subdir in ['outputs','figures','logs']:
        
        if os.path.exists(os.path.join(config['output_path'],subdir)) == False:
            print('creating %s'%os.path.join(config['output_path'],subdir),flush=True)
            os.makedirs(os.path.join(config['output_path'],subdir))


# this saves adam/MAP/fisher output as an npz file
# NOTE: this should also save the observables...
def write_adam_map_fisher(config,out_map_fisher,full_params_arr,free_params_arr,
                          true_loss,true_grad_loss,obs_stats,post_preds_map):

    ### unpack out_map_fisher
    (best_adam_loss, theta_map, adam_hess_arr, adam_hess_flag, Finv_adam, best_adam_index,
     best_finali, best_adam_trace_loss, best_adam_trace_params, best_adam_trace_grads,
     hess_true_flag,Finv_true,sigma_true, accuracy_map) = out_map_fisher    
    
    ### MISSING: dict-format for true/input parameters
    # for now, just store params_order and params_free list
    full_params_order = ['A_M','alpha0_M','alphaz_M','beta_M',
                         'A_E','alpha0_E','alphaz_E','beta_E',
                         'A_SF','alpha0_SF','alphaz_SF','beta_SF',
                         'A_Z','alpha0_Z','alphaz_Z','beta_Z'] 

    params_bounds = config['sampling_config']['params_bounds']
    params_free = list(params_bounds.keys())    

    ### unpack obs_stats (from mock or observations)
    (obs_avg_smhm,obs_err_smhm,
     obs_avg_fgas,obs_err_fgas,
     obs_avg_mzr,obs_err_mzr,
     obs_x0_mvir,obs_bw_mvir,obs_x0_mstar,obs_bw_mstar) = obs_stats     
    
    ### unpack posterior predictives at theta_map 
    (map_z0_Mvir, map_z0_smhm, map_Nfail, map_z0_Mstar, map_z0_fgas, map_z0_mzr,
     map_avg_smhm, map_err_smhm, map_avg_fgas, map_err_fgas, map_avg_mzr, map_err_mzr) = post_preds_map
    
    # set up npz filename
    if 'mock_num' in config.keys():
        prefix = 'mock%s'%config['mock_num']
    else:
        prefix = ''

    ### user specified suffix 
    # TO DO: write a wrapper that generalizes this, printing par names (not just values) where requested
    suffix = config['output_suffix'].format(**config)

    # NOTE: this can be improved by auto-appending whatever params the user specifies in config 
    fname = os.path.join(config['output_path'],'outputs','%s_%s.npz'%(prefix,suffix))
    
    # finally save npz
    jnp.savez(fname,
              ### true/input/mock parameters (nan's if not mock mode)
              full_params_arr = full_params_arr, # true/input/mock full transformed params
              free_params_arr = free_params_arr, # just the true/input/mock free parameters (non-transformed)
              true_loss = true_loss, 
              true_grad_loss = true_grad_loss,
              hess_true_flag = hess_true_flag,
              Finv_true = Finv_true,
              sigma_true = sigma_true,
              
              ### results of adam
              theta_map = theta_map,
              Finv_adam = Finv_adam,
              adam_hess_arr = adam_hess_arr,
              adam_hess_flag = adam_hess_flag,
              best_adam_loss = best_adam_loss,
              best_adam_trace_loss = best_adam_trace_loss,
              best_adam_trace_params = best_adam_trace_params,
              best_adam_trace_grads = best_adam_trace_grads,

              ### if mock mode -- accuracy/bias vs true parameter values (normalized by sigma_true), nan's otherwise
              accuracy_map = accuracy_map,

              ### save summary stats from obs/mock and at theta_map 
              # again this needs to be generalized for arbitrary quantities and summary statistics
              obs_avg_smhm = obs_avg_smhm, 
              obs_err_smhm = obs_err_smhm,
              obs_avg_fgas = obs_avg_fgas, 
              obs_err_fgas = obs_err_fgas,
              obs_avg_mzr = obs_avg_mzr,
              obs_err_mzr = obs_err_mzr,
              obs_x0_mvir = obs_x0_mvir,
              obs_bw_mvir = obs_bw_mvir,
              obs_x0_mstar = obs_x0_mstar,
              obs_bw_mstar = obs_bw_mstar,
              
              map_z0_Mvir = map_z0_Mvir, 
              map_z0_smhm = map_z0_smhm, 
              map_Nfail = map_Nfail, 
              map_z0_Mstar = map_z0_Mstar, 
              map_z0_fgas = map_z0_fgas, 
              map_z0_mzr = map_z0_mzr,
              map_avg_smhm = map_avg_smhm, 
              map_err_smhm = map_err_smhm, 
              map_avg_fgas = map_avg_fgas,  
              map_err_fgas = map_err_fgas, 
              map_avg_mzr = map_avg_mzr, 
              map_err_mzr = map_err_mzr, 

              
             )
              
    print('saved adam-map-fisher outputs in %s'%fname,flush=True)

    



###