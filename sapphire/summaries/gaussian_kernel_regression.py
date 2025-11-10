"""
this module does univariate gaussian kernel regression from sapphire batched outputs 
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
def local_linear(x_data, y_data, x0, bandwidth):

    raise ValueError('not implemented yet')




#