"""
this module defines the model either with numpyro/blackjax or manually

"""

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

# this keeps it clean
import sapphire.summaries.gaussian_kernel_regression as gkr
from . import run_numpyro # in case user requests numpyro-based loss

def setup(config,minibatch_halo_index,obs_stats,batch_solve):

    inference_config = config['inference_config']
    sampling_config = config['sampling_config']
    params_fixed_astro = config['params_fixed_astro']

    # required parameter order for sapphire
    # NOTE: change API to just use dicts throughout (care is needed for gradients)
    full_params_order = ['A_M','alpha0_M','alphaz_M','beta_M',
                         'A_E','alpha0_E','alphaz_E','beta_E',
                         'A_SF','alpha0_SF','alphaz_SF','beta_SF',
                         'A_Z','alpha0_Z','alphaz_Z','beta_Z'] 

    params_bounds = sampling_config['params_bounds']
    params_free = list(params_bounds.keys())
    Nfree = len(params_free)
    
    lower_bounds = jnp.array([params_bounds[k][0] for k in params_free])
    upper_bounds = jnp.array([params_bounds[k][1] for k in params_free])  

    Lflag_smhm = config['flag_smhm'] 
    Lflag_fgas = config['flag_fgas'] 
    Lflag_mzr = config['flag_mzr']   

    Nbatch = inference_config['Nbatch']

    ### unpack input observed (or mock) summary statistics
    (obs_x0_smhm,obs_bw_smhm,obs_avg_smhm,obs_err_smhm,
     obs_x0_fgas,obs_bw_fgas,obs_avg_fgas,obs_err_fgas,
     obs_x0_mzr,obs_bw_mzr,obs_avg_mzr,obs_err_mzr) = obs_stats

    # print('raw mock obs_err',obs_err_smhm,obs_err_fgas,obs_err_mzr,flush=True)
    
    ### IF MOCK MODE -- change obs_err_* to the user supplied constant or scalar
    #   (or have user add x-dependent function in sapphire.summaries and import here)
    if inference_config['fit_mock'] is True and inference_config['fit_obs'] is False: # checking both as safeguard

        # unclear how to use "scale" since that needs to be multiplied by intrinsic stderr(y) inside 
        # might need to add that as static (non-jit-traced) input to logL functions below
        obs_err_smhm = config['obs_err_smhm']
        obs_err_fgas = config['obs_err_fgas']
        obs_err_mzr = config['obs_err_mzr']
    
    # uniform prior
    # @jit
    def log_prior(batch_params): 
    
        # logp = jnp.full_like(test_params,1.0) # 1.0
        logp = jnp.log(1.0/(upper_bounds-lower_bounds)) # this is what numpyro and pymc do
    
        # Dec 30 -- add quadratic penalty for EVERY parameter to ensure it stays within initial latin hypercube mock prior range
        lower_penalty = jnp.sum(jnp.maximum(lower_bounds - batch_params, 0)**2)
        upper_penalty = jnp.sum(jnp.maximum(batch_params - upper_bounds, 0)**2)
    
        """ july 15 -- raw grads become ~1e8, so 1e6 penalty is not high enough """
        penalty = 1e8 * Nbatch * (lower_penalty + upper_penalty)    
        
        return jnp.sum(logp - penalty)
    
    # print('log_prior(test_params)',log_prior(test_params),flush=True)
    
    ### independent (iid) gaussian log-L 
    # @jit
    def compute_logL(batch_params):
        """ 
        batch_params = params to be passed to batch_solve 
        extra arguments are for likelihood parameters, observables, etc. 
            smhm_sigma = constant or Mhalo/redshift dependent uncertainty in log10(SMHM) [dex]
    
        Nov 15 -- for optimization purposes, some parameters are assumed to be log10 to prevent negatives 
        Nov 16 -- if not fitting to different redshifts, fix redshift dependent parameters 
        """
    
        """ July 24 -- for automatic 6 and 4 parameter case, can't hard-code batch_params index numbers """

        ## converting to dict for easier manipulation
        batch_params_dict = {params_free[i]:batch_params[i] for i in range(Nfree)}
        full_params_dict = {**params_fixed_astro, **batch_params_dict} # second dict overwrites first one for keys in common
        full_params = jnp.array([full_params_dict[k] for k in full_params_order])

        # jax.debug.print('batch_params={} full_params={}',batch_params,full_params)

        # NOTE: this needs to be generalized using dicts
        full_params = full_params.at[0].set(10**full_params[0]) # 10**A_M
        full_params = full_params.at[4].set(10**full_params[4]) # 10**A_E
        full_params = full_params.at[8].set(10**full_params[8]) # 10**A_SF
        full_params = full_params.at[12].set(10**full_params[12]) # 10**A_Z
        full_params = jnp.concatenate([full_params,jnp.array([0])]) # realization #
        
        # first call the batch solve 
        shsol = batch_solve(minibatch_halo_index, full_params)
    
        """ July 14 -- more compact using functions above """
        z0_Mvir, z0_smhm, fail_flag, Nfail, z0_Mstar, z0_fgas, z0_mzr = gkr.extract_quantities(shsol)
    
        pred_avg_smhm, pred_err_smhm = gkr.nadaraya_watson(z0_Mvir, z0_smhm, obs_x0_smhm, obs_bw_smhm)
        pred_avg_fgas, pred_err_fgas = gkr.nadaraya_watson(z0_Mstar, z0_fgas, obs_x0_fgas, obs_bw_fgas)
        pred_avg_mzr, pred_err_mzr = gkr.nadaraya_watson(z0_Mstar, z0_mzr, obs_x0_mzr, obs_bw_mzr)    

        ### NEED TO ADD ACTUAL OBS QUADRATURE SUM OPTION HERE (AS for numpyro below)
        tot_err_smhm = jnp.sqrt(pred_err_smhm**2 + obs_err_smhm**2)
        tot_err_fgas = jnp.sqrt(pred_err_fgas**2 + obs_err_fgas**2)
        tot_err_mzr = jnp.sqrt(pred_err_mzr**2 + obs_err_mzr**2)        
    
        # compute masked summed logL for each observable separately
        logL_smhm = -0.5*jnp.sum(((pred_avg_smhm-obs_avg_smhm)/tot_err_smhm)**2 + jnp.log(2*jnp.pi*tot_err_smhm**2))
        logL_fgas = -0.5*jnp.sum(((pred_avg_fgas-obs_avg_fgas)/tot_err_fgas)**2 + jnp.log(2*jnp.pi*tot_err_fgas**2))
        logL_mzr  = -0.5*jnp.sum(((pred_avg_mzr-obs_avg_mzr)/tot_err_mzr)**2 + jnp.log(2*jnp.pi*tot_err_mzr**2))
        
        # Dec 6 -- return logL itself, not negative, since we will convert to log_posterior later
        # July 24 -- here multiply by Lflag_XXX to zero out different constraints 
        return Lflag_smhm*logL_smhm + Lflag_fgas*logL_fgas + Lflag_mzr*logL_mzr
    
    
    ### our loss will be negative log-posterior = negative * (log-likelihood + log_prior)
    
    # @jit
    def negative_log_posterior(batch_params):
        
        logL = compute_logL(batch_params)
        
        logprior = log_prior(batch_params)
        
        log_posterior = logL + logprior
    
        # return LOSS := negative log_posterior 
        return -log_posterior

    

    # return jitted loss, grad(loss) and hessian(loss) functions 
    if inference_config['adam_logL'] == 'numpyro':
        # TO DO: adam params needs to be dict for numpyro, not array 
        loss_func = run_numpyro.setup(config,minibatch_halo_index,obs_stats,batch_solve) # returns numpyro_loss func
    elif inference_config['adam_logL'] == 'manual':
        loss_func = negative_log_posterior # else use the manual one above (user should check that both numpyro/manual are equivalent)
    return jax.jit(loss_func), jax.jit(jax.jacfwd(loss_func)), jax.jit(jax.jacfwd(jax.jacfwd(loss_func)))



    

###  