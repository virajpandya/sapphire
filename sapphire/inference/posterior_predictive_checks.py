"""
this module provides convenience functions for posterior predictive checks
including extracting the best point-wise and summary statistics from adam's MAP+fisher
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
from jax.sharding import Mesh, PartitionSpec, NamedSharding

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, AIES, ESS

# this keeps it clean
import sapphire.summaries.gaussian_kernel_regression as gkr


# NOTE: need to generalize what summary statistic to use 
def adam_map_fisher(config,obs_stats,out_map_fisher,batch_solve,minibatch_halo_index):

    # unpack input observed (or mock) summary statistics (this provides bandwidths)
    (obs_x0_smhm,obs_bw_smhm,obs_avg_smhm,obs_err_smhm,
     obs_x0_fgas,obs_bw_fgas,obs_avg_fgas,obs_err_fgas,
     obs_x0_mzr,obs_bw_mzr,obs_avg_mzr,obs_err_mzr) = obs_stats
    
    
    # unpack out_map_fisher (see return signature of sapphire/inference/map_fisher.py)
    (best_adam_loss, theta_map, adam_hess_arr, adam_hess_flag, Finv_adam, best_adam_index,
     best_finali, best_adam_trace_loss, best_adam_trace_params, best_adam_trace_grads,
     hess_true_flag,Finv_true,sigma_true, accuracy_map)  = out_map_fisher

    # adapt sapphire/inference/compute_logL to merge theta_map with any fixed parameters to get full_params_arr 
    # NOTE: this should be pushed to sapphire/utils/setup_parameters.py

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
    
    ## converting to dict for easier manipulation
    theta_map_dict = {params_free[i]:theta_map[i] for i in range(Nfree)}
    full_params_dict = {**params_fixed_astro, **theta_map_dict} # second dict overwrites first one for keys in common
    full_params = jnp.array([full_params_dict[k] for k in full_params_order])

    # NOTE: this needs to be generalized using dicts
    full_params = full_params.at[0].set(10**full_params[0]) # 10**A_M
    full_params = full_params.at[4].set(10**full_params[4]) # 10**A_E
    full_params = full_params.at[8].set(10**full_params[8]) # 10**A_SF
    full_params = full_params.at[12].set(10**full_params[12]) # 10**A_Z
    full_params = jnp.concatenate([full_params,jnp.array([0])]) # realization #
    
    # now evaluate ODEs at best final adam theta_map
    sol_map = batch_solve(minibatch_halo_index, full_params)

    # extract discrete pointwise quantities and compute summary statistics (using obs/mock bandwidths)
    # NOTE: this needs to be generalized for other quantities, summary statistics, etc. 
    map_z0_Mvir, map_z0_smhm, map_fail_flag, map_Nfail, map_z0_Mstar, map_z0_fgas, map_z0_mzr = gkr.extract_quantities(sol_map)
    
    map_avg_smhm, map_err_smhm = gkr.nadaraya_watson(map_z0_Mvir, map_z0_smhm, obs_x0_smhm, obs_bw_smhm)
    map_avg_fgas, map_err_fgas = gkr.nadaraya_watson(map_z0_Mstar, map_z0_fgas, obs_x0_fgas, obs_bw_fgas)
    map_avg_mzr, map_err_mzr = gkr.nadaraya_watson(map_z0_Mstar, map_z0_mzr, obs_x0_mzr, obs_bw_mzr)
    
    return (map_z0_Mvir, map_z0_smhm, map_Nfail, map_z0_Mstar, map_z0_fgas, map_z0_mzr,
            map_avg_smhm, map_err_smhm, map_avg_fgas, map_err_fgas, map_avg_mzr, map_err_mzr)





###