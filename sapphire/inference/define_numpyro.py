""" 
this module shows how to define a numpyro model (based on Pandya+26) 
it is analogous to explicit_likelihood.py
this module can also return a numpyro-based loss function for adam

users are welcome (encouraged) to actually define their own numpyro model
outside of this internal file and then feed it to the separate run_numpyro.py 
module which actually runs inference (NUTS, AIES, SVI, etc.)
"""

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15
from functools import partial 
import os
import pandas as pd
from timeit import default_timer as timer

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 120
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

# in case user loads module separately from sapphire.run()
from jax import config as jax_config
jax_config.update("jax_enable_x64", True) # required to accurately solve and take gradients through our diffeqs 

import jax
import jax.numpy as jnp
from jax._src.third_party.scipy.interpolate import RegularGridInterpolator as jax_RegularGridInterpolator
from jax import jit, grad, vmap, pmap, debug, jvp, vjp, jacrev, jacfwd, make_jaxpr, hessian, value_and_grad
# from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline    
from jax.experimental.ode import odeint
from jax.lax import fori_loop, while_loop
from jax.scipy.integrate import trapezoid
from jax.random import PRNGKey, key    
from diffrax import diffeqsolve, ODETerm, PIDController, SaveAt, Kvaerno3, Bosh3, Dopri5, Tsit5, DirectAdjoint, RecursiveCheckpointAdjoint, BacksolveAdjoint
from diffrax import backward_hermite_coefficients, CubicInterpolation    
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, AIES, ESS
from chainconsumer import Chain, ChainConsumer, make_sample, PlotConfig, Truth, ChainConfig

# this keeps it clean
import sapphire.summaries.gaussian_kernel_regression as gkr



def setup(config,halo_index,obs_stats,batch_solve):

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
    Lflag_sfms = config['flag_sfms']   
    Lflag_mzr_gas = config['flag_mzr_gas']       

    Nbatch = config['Nbatch']

    ### unpack input observed (or mock) summary statistics
    (obs_x0_smhm,obs_bw_smhm,obs_avg_smhm,obs_err_smhm,
     obs_x0_fgas,obs_bw_fgas,obs_avg_fgas,obs_err_fgas,
     obs_x0_mzr,obs_bw_mzr,obs_avg_mzr,obs_err_mzr, 
     obs_x0_sfms,obs_bw_sfms,obs_avg_sfms,obs_err_sfms, 
     obs_x0_mzr_gas,obs_bw_mzr_gas,obs_avg_mzr_gas,obs_err_mzr_gas) = obs_stats

    ### optionally, if running in mock mode, zero out obs_errs and let user do their mock_err below [or this can be kept, if user desires]
    if inference_config['fit_mock'] == True:
        obs_err_smhm = 0.0
        obs_err_fgas = 0.0
        obs_err_mzr = 0.0
        obs_err_sfms = 0.0
        obs_err_mzr_gas = 0.0

    ### package only the needed subset into a numpyro_model_args 
    model_args = (obs_avg_smhm,obs_err_smhm,obs_avg_fgas,obs_err_fgas,obs_avg_mzr,obs_err_mzr,
                  obs_avg_sfms,obs_err_sfms,obs_avg_mzr_gas,obs_err_mzr_gas)

    
    # print('raw mock obs_err',obs_err_smhm,obs_err_fgas,obs_err_mzr,flush=True)
    
    # these are added in quadrature (should be 0 by default)
    mock_err_smhm = config['mock_err_smhm']
    mock_err_fgas = config['mock_err_fgas']
    mock_err_mzr = config['mock_err_mzr']
    mock_err_sfms = config['mock_err_sfms']
    mock_err_mzr_gas = config['mock_err_mzr_gas']    
        
        
    """ define the numpyro model """
    
    # note: these input obs can be different from the obs_ ones returned above
    def model(obs_avg_smhm,obs_err_smhm,obs_avg_fgas,obs_err_fgas,obs_avg_mzr,obs_err_mzr,
              obs_avg_sfms,obs_err_sfms,obs_avg_mzr_gas,obs_err_mzr_gas):
        
        ### sample the free parameters assuming their respective Uniform priors 
        # the user should change the prior sampling function as needed...
        params_dict = {}
        for i, name in enumerate(params_free):
            params_dict[name] = numpyro.sample(name,dist.Uniform(lower_bounds[i], upper_bounds[i]))

        # build full parameter dict and array 
        full_params_dict = {**params_fixed_astro, **params_dict}
        full_params = jnp.array([full_params_dict[k] for k in full_params_order])
        
        full_params = full_params.at[0].set(10**full_params[0]) # 10**A_M
        full_params = full_params.at[4].set(10**full_params[4]) # 10**A_E
        full_params = full_params.at[8].set(10**full_params[8]) # 10**A_SF
        full_params = full_params.at[12].set(10**full_params[12]) # 10**A_Z
        full_params = jnp.concatenate([full_params,jnp.array([0])]) # realization #

        ### for some reason with adam, Uniform hard-cutoff is not enforced
        ### so here add a soft quadratic penalty just like i was doing manually, using numpyro.factor
        parr = jnp.array([params_dict[name] for name in params_free])
        lower_penalty = jnp.sum(jnp.maximum(lower_bounds - parr, 0)**2)
        upper_penalty = jnp.sum(jnp.maximum(parr - upper_bounds, 0)**2)
        penalty = 1e8 * Nbatch * (lower_penalty + upper_penalty) # only gradient should matter, not normalization I think
        numpyro.factor('penalty', -penalty)
        
        # first call the batch solve 
        shsol = batch_solve(halo_index, full_params)
    
        """ July 14 -- more compact using functions above """
        z0_Mvir, z0_smhm, fail_flag, Nfail, z0_Mstar, z0_fgas, z0_mzr, z0_Mdot_sfr, z0_mzr_gas = gkr.extract_quantities(shsol)
    
        numpyro.deterministic('Nfail', Nfail)    
    
        ### July 24 -- evaluate now using observed x0 and bin(=band) widths
        pred_avg_smhm, pred_err_smhm = gkr.nadaraya_watson(z0_Mvir, z0_smhm, obs_x0_smhm, obs_bw_smhm)
        pred_avg_fgas, pred_err_fgas = gkr.nadaraya_watson(z0_Mstar, z0_fgas, obs_x0_fgas, obs_bw_fgas)
        # pred_avg_mzr, pred_err_mzr = gkr.nadaraya_watson(z0_Mstar, z0_mzr, obs_x0_mzr, obs_bw_mzr) 
        # pred_avg_sfms, pred_err_sfms = gkr.nadaraya_watson(z0_Mstar, z0_Mdot_sfr, obs_x0_sfms, obs_bw_sfms) 
        pred_avg_mzr_gas, pred_err_mzr_gas = gkr.nadaraya_watson(z0_Mstar, z0_mzr_gas, obs_x0_mzr_gas, obs_bw_mzr_gas) 
    
        """ July 24 -- quadrature sum intrinsic model standard error with observed uncertainty """ 
        ### NEED TO UPDATE THIS TO GENERALLY HANDLE MOCKED OR ACTUAL OBSERVED EXTRA ERROR 
        tot_err_smhm = jnp.sqrt(pred_err_smhm**2 + obs_err_smhm**2 + mock_err_smhm**2) 
        tot_err_fgas = jnp.sqrt(pred_err_fgas**2 + obs_err_fgas**2 + mock_err_fgas**2) 
        # tot_err_mzr = jnp.sqrt(pred_err_mzr**2 + obs_err_mzr**2 + mock_err_mzr**2) 
        # tot_err_sfms = jnp.sqrt(pred_err_sfms**2 + obs_err_sfms**2 + mock_err_sfms**2) 
        tot_err_mzr_gas = jnp.sqrt(pred_err_mzr_gas**2 + obs_err_mzr_gas**2 + mock_err_mzr_gas**2) 

        # to save as part of outputs 
        numpyro.deterministic('pred_avg_smhm',pred_avg_smhm)
        numpyro.deterministic('pred_avg_fgas',pred_avg_fgas)
        # numpyro.deterministic('pred_avg_mzr',pred_avg_mzr)
        # numpyro.deterministic('pred_avg_sfms',pred_avg_sfms)
        numpyro.deterministic('pred_avg_mzr_gas',pred_avg_mzr_gas)
        
        numpyro.deterministic('pred_err_smhm',pred_err_smhm)
        numpyro.deterministic('pred_err_fgas',pred_err_fgas)
        # numpyro.deterministic('pred_err_mzr',pred_err_mzr)    
        # numpyro.deterministic('pred_err_sfms',pred_err_sfms)    
        numpyro.deterministic('pred_err_mzr_gas',pred_err_mzr_gas)            
        
        if obs_avg_smhm is None: # for prior and posterior predictive checks
    
            numpyro.sample('obs_smhm',dist.Normal(pred_avg_smhm,tot_err_smhm)) 
            numpyro.sample('obs_fgas',dist.Normal(pred_avg_fgas,tot_err_fgas)) 
            numpyro.sample('obs_mzr',dist.Normal(pred_avg_mzr,tot_err_mzr)) 
            numpyro.sample('obs_sfms',dist.Normal(pred_avg_sfms,tot_err_sfms)) 
            numpyro.sample('obs_mzr_gas',dist.Normal(pred_avg_mzr_gas,tot_err_mzr_gas))  
            
        else:
    
            # sample gaussian likelihoods independently (these will be internally summed by numpyro like we'd do manually)
            # May 15 -- only compute each logL if requested based on command line argument
            if Lflag_smhm:
                obs_smhm = numpyro.sample('obs_smhm',dist.Normal(pred_avg_smhm,tot_err_smhm), obs=obs_avg_smhm)
            
            if Lflag_fgas:
                obs_fgas = numpyro.sample('obs_fgas',dist.Normal(pred_avg_fgas,tot_err_fgas), obs=obs_avg_fgas)
    
            if Lflag_mzr: 
                obs_mzr = numpyro.sample('obs_mzr',dist.Normal(pred_avg_mzr,tot_err_mzr), obs=obs_avg_mzr)

            if Lflag_sfms: 
                obs_sfms = numpyro.sample('obs_sfms',dist.Normal(pred_avg_sfms,tot_err_sfms), obs=obs_avg_sfms)

            if Lflag_mzr_gas: 
                obs_mzr_gas = numpyro.sample('obs_mzr_gas',dist.Normal(pred_avg_mzr_gas,tot_err_mzr_gas), obs=obs_avg_mzr_gas)    
                

    # in case we want to compute loss for adam based on numpyro model
    # this hard-codes the input obs constraint as inputs to numpyro model() function
    @jit
    def numpyro_loss(params):
        # numpyro.infer.util.log_density returns (log_density, aux) tuple.
        logp, _ = numpyro.infer.util.log_density(model, 
                                                 (*model_args, ), 
                                                 {}, params)
        
        # take negative of numpyro log-posterior so it is a loss for adam
        return -logp    

    # if requested, just return numpyro loss function but don't do any inference
    if inference_config['engine'] == 'adam' and inference_config['adam_logL'] == 'numpyro':

        ########### before proceeding, benchmark jit-time and post-jit runtime so we know what to expect 
        test_keys = jax.random.split(key(config['rng_init']), len(params_bounds))
        test_params = {name: jax.random.uniform(k,shape=(),minval=lower,maxval=upper)
                       for k, (name, (lower, upper)) in zip(test_keys, params_bounds.items())}
    
        tstart = timer()
        test_loss = numpyro_loss(test_params).block_until_ready()
        print('jit numpyro loss_fn took %.3f sec with value=%.3f'%(timer()-tstart,test_loss),flush=True)
    
        tstart = timer()
        test_loss = numpyro_loss(test_params).block_until_ready()
        print('post-jit numpyro loss_fn took %.3f sec with value=%.3f'%(timer()-tstart,test_loss),flush=True)    
        
        return numpyro_loss
    
    ### otherwise return numpyro model for use in run_numpyro.py module

    return (model, model_args, params_free)


    
###