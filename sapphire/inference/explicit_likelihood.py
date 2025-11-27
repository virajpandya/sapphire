"""
this module defines the model either with numpyro/blackjax or manually

"""

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15
from functools import partial 

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

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, AIES, ESS

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

    Lflag_smhm = inference_config['flag_smhm'] 
    Lflag_fgas = inference_config['flag_fgas'] 
    Lflag_mzr = inference_config['flag_mzr']   

    Nbatch = inference_config['Nbatch']

    ### move this outside to utils or something, for driver.py benchmarking 
    minibatch_halo_index = jax.random.choice(key(0), halo_index, (Nbatch,), replace=False)

    ### unpack input observed (or mock) summary statistics
    (obs_avg_smhm,obs_err_smhm,
     obs_avg_fgas,obs_err_fgas,
     obs_avg_mzr,obs_err_mzr,
     obs_x0_mvir,obs_bw_mvir,obs_x0_mstar,obs_bw_mstar) = obs_stats

    # print('raw mock obs_err',obs_err_smhm,obs_err_fgas,obs_err_mzr,flush=True)
    
    ### IF MOCK MODE -- change obs_err_* to the user supplied constant or scalar
    #   (or have user add x-dependent function in sapphire.summaries and import here)
    if inference_config['mock'] is True:
        mock_err = inference_config['mock_err']

        # unclear how to use "scale" since that needs to be multiplied by intrinsic stderr(y) inside 
        # might need to add that as static (non-jit-traced) input to logL functions below
        obs_err_smhm = mock_err['smhm']['constant']
        obs_err_fgas = mock_err['fgas']['constant']
        obs_err_mzr = mock_err['mzr']['constant']

        # print('input constant mock obs_err',obs_err_smhm,obs_err_fgas,obs_err_mzr,flush=True)
    
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
        
        full_params = full_params.at[0].set(10**full_params[0]) # 10**A_M
        full_params = full_params.at[4].set(10**full_params[4]) # 10**A_E
        full_params = full_params.at[8].set(10**full_params[8]) # 10**A_SF
        full_params = full_params.at[12].set(10**full_params[12]) # 10**A_Z
        full_params = jnp.concatenate([full_params,jnp.array([0])]) # realization #
        
        # first call the batch solve 
        shsol = batch_solve(minibatch_halo_index, full_params)
    
        """ July 14 -- more compact using functions above """
        z0_Mvir, z0_smhm, fail_flag, Nfail, z0_Mstar, z0_fgas, z0_mzr = gkr.extract_quantities(shsol)
    
        pred_avg_smhm, pred_err_smhm = gkr.nadaraya_watson(z0_Mvir, z0_smhm, obs_x0_mvir, obs_bw_mvir)
        pred_avg_fgas, pred_err_fgas = gkr.nadaraya_watson(z0_Mstar, z0_fgas, obs_x0_mstar, obs_bw_mstar)
        pred_avg_mzr, pred_err_mzr = gkr.nadaraya_watson(z0_Mstar, z0_mzr, obs_x0_mstar, obs_bw_mstar)    

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
    
    
    """ alternatively, the equivalent model in numpyro """
    
    # note: these input obs can be different from the obs_ ones returned above
    def model(obs_avg_smhm,obs_err_smhm,obs_avg_fgas,obs_err_fgas,obs_avg_mzr,obs_err_mzr):
        
        ### sample the free parameters assuming their respective Uniform priors 
        # the user should change the prior sampling function as needed...
        params_dict = {}
        for i, name in enumerate(params_free):
            params_dict[name] = numpyro.sample(name,dist.Uniform(lower_bounds[i], upper_bounds[i]))

        # build full parameter dict and array 
        full_params_dict = {**params_dict, **params_fixed_astro}
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
        shsol = batch_solve(minibatch_halo_index, full_params)
    
        """ July 14 -- more compact using functions above """
        z0_Mvir, z0_smhm, fail_flag, Nfail, z0_Mstar, z0_fgas, z0_mzr = gkr.extract_quantities(shsol)
    
        numpyro.deterministic('Nfail', Nfail)    
    
        ### July 24 -- evaluate now using observed x0 and bin(=band) widths
        pred_avg_smhm, pred_err_smhm = gkr.nadaraya_watson(z0_Mvir, z0_smhm, obs_x0_mvir, obs_bw_mvir)
        pred_avg_fgas, pred_err_fgas = gkr.nadaraya_watson(z0_Mstar, z0_fgas, obs_x0_mstar, obs_bw_mstar)
        pred_avg_mzr, pred_err_mzr = gkr.nadaraya_watson(z0_Mstar, z0_mzr, obs_x0_mstar, obs_bw_mstar) 
    
        """ July 24 -- quadrature sum intrinsic model standard error with observed uncertainty """ 
        ### NEED TO UPDATE THIS TO GENERALLY HANDLE MOCKED OR ACTUAL OBSERVED EXTRA ERROR 
        tot_err_smhm = jnp.sqrt(pred_err_smhm**2 + obs_err_smhm**2)
        tot_err_fgas = jnp.sqrt(pred_err_fgas**2 + obs_err_fgas**2)
        tot_err_mzr = jnp.sqrt(pred_err_mzr**2 + obs_err_mzr**2)

        # to save as part of outputs 
        numpyro.deterministic('pred_avg_smhm',pred_avg_smhm)
        numpyro.deterministic('pred_avg_fgas',pred_avg_fgas)
        numpyro.deterministic('pred_avg_mzr',pred_avg_mzr)
        numpyro.deterministic('pred_err_smhm',pred_err_smhm)
        numpyro.deterministic('pred_err_fgas',pred_err_fgas)
        numpyro.deterministic('pred_err_mzr',pred_err_mzr)    
        
        if obs_avg_smhm is None: # for prior and posterior predictive checks
    
            numpyro.sample('obs_avg_smhm',dist.Normal(pred_avg_smhm,tot_err_smhm)) 
            numpyro.sample('obs_avg_fgas',dist.Normal(pred_avg_fgas,tot_err_fgas)) 
            numpyro.sample('obs_avg_mzr',dist.Normal(pred_avg_mzr,tot_err_mzr)) 
            
        else:
    
            # sample gaussian likelihoods independently (these will be internally summed by numpyro like we'd do manually)
            # May 15 -- only compute each logL if requested based on command line argument
            if Lflag_smhm:
                obs_avg_smhm = numpyro.sample('obs_avg_smhm',dist.Normal(pred_avg_smhm,tot_err_smhm), obs=obs_avg_smhm)
            
            if Lflag_fgas:
                obs_avg_fgas = numpyro.sample('obs_avg_fgas',dist.Normal(pred_avg_fgas,tot_err_fgas), obs=obs_avg_fgas)
    
            if Lflag_mzr: 
                obs_avg_mzr = numpyro.sample('obs_avg_mzr',dist.Normal(pred_avg_mzr,tot_err_mzr), obs=obs_avg_mzr)
                

    # in case we want to compute loss for adam based on numpyro model
    # this hard-codes the input obs constraint as inputs to numpyro model() function
    def numpyro_loss(params):
        # numpyro.infer.util.log_density returns (log_density, aux) tuple.
        logp, _ = numpyro.infer.util.log_density(model, 
                                                 (obs_avg_smhm,obs_err_smhm,
                                                  obs_avg_fgas,obs_err_fgas,
                                                  obs_avg_mzr,obs_err_mzr), 
                                                 {}, params)
        
        # take negative of numpyro log-posterior so it is a loss for adam
        return -logp
    

    # return jitted loss, grad(loss) and hessian(loss) functions 
    if inference_config['backend'] == 'numpyro':
        loss_func = numpyro_loss
    elif inference_config['backend'] == 'manual':
        loss_func = negative_log_posterior 

    return jax.jit(loss_func), jax.jit(jax.jacfwd(loss_func)), jax.jit(jax.jacfwd(jax.jacfwd(loss_func)))
    # return loss_func, jax.jacfwd(loss_func), jax.jacfwd(jax.jacfwd(loss_func))



    

###  