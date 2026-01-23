"""
this module does univariate gaussian kernel regression from sapphire batched outputs 
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


### write a function that returns the data want
@jit
def extract_quantities(sol):
    ### sol is the diffrax solution object from batch_solve above

    # first compute relevant quantities
    z0_Mvir = jnp.log10(sol.ys[1]['Mvir'][:,-1])
    z0_Mstar = sol.ys[0][:,-1,0]
    z0_smhm = jnp.log10(10**z0_Mstar / 10**z0_Mvir)
            
    # ### extract quantities of interest
    nsteps = sol.stats['num_steps']
    fail_flag = jnp.where(nsteps>=4096,1,0)
    Nfail = jnp.sum(fail_flag) # where True (failed solve), yield 1, else 0. Then we can just sum
    # print('true mock Nfail=%s'%Nfail)
    
    # April 28 -- need to do this for kde, otherwise inf/nan Mvir or smhm leads to all-nan kde
    z0_smhm = jnp.where(fail_flag, -99.0, z0_smhm) 
    z0_Mvir = jnp.where(fail_flag, -99.0, z0_Mvir) 

    ### July 14 -- add stuff for fgas and MZR

    z0_Mstar = jnp.where(fail_flag, -99.0, z0_Mstar) 
    
    z0_Mism = sol.ys[0][:,-1,1]
    z0_fgas = jnp.log10(10**z0_Mism / 10**z0_Mstar)
    z0_fgas = jnp.where(fail_flag, -99.0, z0_fgas) 
    
    z0_MZstar = sol.ys[0][:,-1,4]
    z0_mzr = jnp.log10(10**z0_MZstar / 10**z0_Mstar / 0.02)    
    z0_mzr = jnp.where(fail_flag, -99.0, z0_mzr)       

    return z0_Mvir, z0_smhm, fail_flag, Nfail, z0_Mstar, z0_fgas, z0_mzr


### New function that pre-processes stuff we need as inputs for nadaraya_watson regression function
### Haven't figured out way to avoid jax ConcretizationTypeError for jnp.arange -- doesn't need jit
### This is only needed once for original mock truth or the observed scaling relations, do not need autodiff/jit
# @jit 
def get_bandwidths(fail_flag, x_data, prob_sel=None, prob_sel_thresh=None):
    ### x_data is usually z0_Mvir or z0_Mstar
    # for obs, assume fail_flag = jnp.zeros_like(x_data) 
    # prob_sel is an optional selection probability function of same shape as x_data
    # prob_sel_thresh can be used to only include x_data that satisfies prob_sel > prob_sel_thresh 

    ### first for rand batch
    # April 28 -- to avoid failed solves
    ind_success = jnp.where(fail_flag==0) # avoid any failed halos (whose values were set to -99 above)
    
    ### compute the gaussian kernal bandwidth from scott's rule and the spread of the logmvir distribution
    scotts_factor = len(x_data[ind_success])**(-1./(1+4)) # note: we add 1 to 4 in exponent since nadaraya-watson is univariate regression 
    std_xdata = jnp.std(x_data[ind_success])
    prefac = 1.0 # 0.5 
    bw_xdata = prefac * std_xdata * scotts_factor
    
    # July 21 -- for better behaved regression, choose x0 based on ~5-95 percentiles of true Mvir and Mstar distribution
    # otherwise can get gaussian bins/kernels at extreme low/high end of true distribution which can even make Fisher undefined at truth
    # note: doesn't matter that jnp.percentile is non-differentiable/jittable --> we're just computing this once BEFORE beginning adam, HMC, etc. 
    if prob_sel == None:
        xmin, xmax = jnp.percentile(x_data[ind_success], jnp.array([5.0,95.0]))
        x0 = jnp.arange(xmin,xmax+bw_xdata,bw_xdata)    

    # Dec 10 -- choose x0 only using galaxies that fall within selection window
    # doesn't matter that this is not differentiable/jittable -- it is only done once to set up obs_stats constraints
    else:
        x_data_success = x_data[ind_success]
        prob_sel_success = prob_sel[ind_success]
        xmin, xmax = jnp.percentile(x_data_success[prob_sel_success>prob_sel_thresh], jnp.array([5.0,95.0]))
        x0 = jnp.arange(xmin,xmax+bw_xdata,bw_xdata)       

    return bw_xdata, x0


### constant 1D gaussian kernel regression 
@jit
def nadaraya_watson(x_data, y_data, x0, bandwidth):

    # compute normed weight of gaussian kernel given each data point's x-distance from each query point x0
    diffs = x0[:, None] - x_data[None, :] # shape = (x0.shape, x_data.shape) 
    weights = jnp.exp(-0.5 * (diffs / bandwidth)**2) / (bandwidth*jnp.sqrt(2*jnp.pi))
    weights_normed = weights / jnp.sum(weights,axis=1,keepdims=True) 
    
    # conditional mean of y at each x0
    means = weights_normed @ y_data  
    
    # conditional standard deviation of y at each x0 
    res_sq = (y_data[None, :] - means[:, None])**2
    stds = jnp.sqrt(jnp.sum(weights_normed * res_sq, axis=1))
    
    # effective sample size comes from the inverse-sum of squared weights_normed
    neff = 1.0 / jnp.sum(weights_normed**2, axis=1) 

    # standard error on means
    stderrs = stds / jnp.sqrt(neff)
    
    return means, stderrs


### locally-linear 1D gaussian kernel regression
@jit
def local_linear_regression(x_data, y_data, x0, bandwidth):

    # compute normed weight of gaussian kernel given each data point's x-distance from each query point x0
    diffs = x0[:, None] - x_data[None, :]  # shape = (x0.shape, x_data.shape) 
    weights = jnp.exp(-0.5 * (diffs / bandwidth) ** 2) / (bandwidth * jnp.sqrt(2 * jnp.pi))
    weights_normed = weights / jnp.sum(weights, axis=1, keepdims=True)

    # compute gaussian kernel-weighted average of x_data and y_data 
    x_mean = jnp.sum(weights_normed * x_data[None, :], axis=1)
    y_mean = jnp.sum(weights_normed * y_data[None, :], axis=1)

    # compute deviation of every (x,y) from (x_mean, y_mean) 
    # this part is different from nadaraya-watson since we'll need the local linear x-dependence 
    x_centered = x_data[None, :] - x_mean[:, None]  # (x0.shape, x_data.shape) 
    y_centered = y_data[None, :] - y_mean[:, None]

    # weighted squared sums for local linear kernel regression 
    Sxx = jnp.sum(weights * x_centered**2, axis=1)
    Sxy = jnp.sum(weights * x_centered * y_centered, axis=1)
    
    # formulas for slope and intercept of local linear fit
    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean

    # compute the predicted y for each x0 location
    y_pred = intercept + slope * x0

    # also compute gaussian kernel-weighted standard error at each x0
    y_fit = intercept[:, None] + slope[:, None] * (x_data[None, :] - x0[:,None])
    res_sq = (y_data[None, :] - y_fit)**2

    # divide sqrt(var) by sqrt(neff) as for nadaraya-watson above
    var = jnp.sum(weights * res_sq, axis=1) / jnp.sum(weights, axis=1)
    neff = 1.0 / jnp.sum(weights_normed**2, axis=1)
    stderr = jnp.sqrt(var / neff)

    return y_pred, stderr


### first, a function that returns bandwidths, x0's and scaling relations for mock truth or observations that we will fit to
# this does not need to be jitted, and cannot anyway due to get_bandwidths issue above
def summarize_mock(sol,obs_stats=None):
    """
    yet to be added: forward model statistical uncertainties

    obs_stats -- if provided, then x0 and bw are taken from this (assumed to be same format as rest of sapphire)
    """

    # extract what we need
    mock_z0_Mvir, mock_z0_smhm, mock_fail_flag, mock_Nfail, mock_z0_Mstar, mock_z0_fgas, mock_z0_mzr = extract_quantities(sol)
    
    # compute bandwidths and centers for gaussian kernels for mvir and mstar 
    # Jan 16 -- x0 and bw can be defined by input obs_stats [for prior predictive checks]
    if obs_stats == None:
        # Jan 16 -- rename for compatibility with obs_stats, allowing x0 and bw vary per observable
        mock_bw_smhm, mock_x0_smhm = get_bandwidths(mock_fail_flag, mock_z0_Mvir)
        mock_bw_fgas, mock_x0_fgas = get_bandwidths(mock_fail_flag, mock_z0_Mstar) 
        mock_bw_mzr, mock_x0_mzr = get_bandwidths(mock_fail_flag, mock_z0_Mstar) 
    else: # based on return signature in sapphire.utils.read_obs
        mock_bw_smhm, mock_x0_smhm = obs_stats[1], obs_stats[0]
        mock_bw_fgas, mock_x0_fgas = obs_stats[5], obs_stats[4]
        mock_bw_mzr, mock_x0_mzr = obs_stats[9], obs_stats[8]
        
    # do gaussian kernel regression 
    mock_avg_smhm, mock_err_smhm = nadaraya_watson(mock_z0_Mvir, mock_z0_smhm, mock_x0_smhm, mock_bw_smhm)
    mock_avg_fgas, mock_err_fgas = nadaraya_watson(mock_z0_Mstar, mock_z0_fgas, mock_x0_fgas, mock_bw_fgas)
    mock_avg_mzr, mock_err_mzr = nadaraya_watson(mock_z0_Mstar, mock_z0_mzr, mock_x0_mzr, mock_bw_mzr)    

    # NOTE: the return order should be same as expected by sapphire.inference modules (for the "obs_stats" collection)
    # Dec 15 -- in the most general case, each obs may have different x0 and bw, so for now here just repeat
    return (mock_x0_smhm, mock_bw_smhm, mock_avg_smhm, mock_err_smhm,
            mock_x0_fgas, mock_bw_fgas, mock_avg_fgas, mock_err_fgas,
            mock_x0_mzr, mock_bw_mzr, mock_avg_mzr, mock_err_mzr)


    

#