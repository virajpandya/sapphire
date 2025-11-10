"""
this module provides convenience functions for generating mock parameters given prior bounds
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


# user requested only a single run, so just transform their input dict into jnp.array format with 10**A_X, etc. 
def single_run(config):

    params_fixed_astro = config['params_fixed_astro']

    # required parameter order for sapphire
    # NOTE: change API to just use dicts throughout (care is needed for gradients)
    params_order = ['A_M','alpha0_M','alphaz_M','beta_M',
                    'A_E','alpha0_E','alphaz_E','beta_E',
                    'A_SF','alpha0_SF','alphaz_SF','beta_SF',
                    'A_Z','alpha0_Z','alphaz_Z','beta_Z']  

    # create jnp.array and append
    # NOTE: change API to get rid of automatically assign realization # at end somehow if user requested written outputs
    params_arr = jnp.array([params_fixed_astro[k] for k in params_order]+[0])
    print('params_arr\n',params_arr,flush=True)    

    # transform 10**A_X 
    # NOTE: this can be a one-liner, or done above in the list comprehension
    params_arr = params_arr.at[0].set(10**params_arr[0]) # A_M
    params_arr = params_arr.at[4].set(10**params_arr[4]) # A_E
    params_arr = params_arr.at[8].set(10**params_arr[8]) # A_SF
    params_arr = params_arr.at[12].set(10**params_arr[12]) # A_Z        

    return params_arr
    


##### latin hypercube sampling [this is pseudo-lhs... need to put in the random shuffling step]
def latin_hypercube_sampling(config):

    sampling_config = config['sampling_config']

    ### first, if on CPU or multiple GPUs, enforce Nsamples = integer multiple of Ndevices (num_cpus or num_gpus)
    if jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) > 1:
        if sampling_config['Nsamples'] % jax.local_device_count():
            raise ValueError('Nsamples must equal to or an integer multiple of num_cpus or num_gpus unless num_gpus=1')
    
    params_bounds = sampling_config['params_bounds']
    params_free = list(params_bounds.keys())

    lower_bounds = jnp.array([params_bounds[k][0] for k in params_free])
    upper_bounds = jnp.array([params_bounds[k][1] for k in params_free])
    print('lower_bounds',lower_bounds,flush=True)
    print('upper_bounds',upper_bounds,flush=True)
    
    # first create a base random key based on mock_num, then split into as many different keys as free parameters we want
    base_key = jax.random.key(sampling_config['rng_seed'])
    keys = jax.random.split(base_key, len(params_free))

    Nsamples = sampling_config['Nsamples']
    
    # Generate mock truth values (log10 for A_X, linear for alpha0_X)
    params_lhs = {pname: jax.random.uniform(keys[i], shape=(Nsamples,), minval=plow, maxval=phigh)
                  for i, (pname, (plow, phigh)) in enumerate(params_bounds.items())}
    
    # convert input params_fixed so every element is a jax array for later below
    params_fixed = {k: jnp.array(v,dtype='float64') for k, v in config['params_fixed_astro'].items()} 

    # now merge fixed and free params to create Nguess params dicts 
    # note: for duplicated keys between params_fixed and params_lhs, the value of params_lhs will be used 
    params_merged = [{**params_fixed, **{k: float(params_lhs[k][i]) for k in params_free}} for i in range(Nsamples)] 
    
    # first define order of parameters expected by sapphire
    params_order = ['A_M','alpha0_M','alphaz_M','beta_M',
                    'A_E','alpha0_E','alphaz_E','beta_E',
                    'A_SF','alpha0_SF','alphaz_SF','beta_SF',
                    'A_Z','alpha0_Z','alphaz_Z','beta_Z'] 
    
    # now convert those Nguess params dicts into 2D jax array of parameters 
    params_lharr = jnp.array([[params_merged[i][k] for k in params_order] for i in range(Nsamples)])
    
    # as usual hstack the final column of all zeros for realization # placeholder
    params_lharr = jnp.concatenate([params_lharr, jnp.zeros((params_lharr.shape[0],1))],axis=1)

    # finally change the A_X to 10**A_X since that's what solve_halo expects
    params_lharr = params_lharr.at[:,0].set(10**params_lharr[:,0]) # A_M
    params_lharr = params_lharr.at[:,4].set(10**params_lharr[:,4]) # A_E
    params_lharr = params_lharr.at[:,8].set(10**params_lharr[:,8]) # A_SF
    params_lharr = params_lharr.at[:,12].set(10**params_lharr[:,12]) # A_Z       

    # print('params_lharr\n',params_lharr,flush=True)
    print('params_lharr.shape\n',params_lharr.shape,flush=True)
    
    return params_lharr




def get(config):

    sampling_config = config['sampling_config']

    if config['runtype'] == 'single':
        return single_run(config)
    
    elif sampling_config['method'] == 'lhs':
        return latin_hypercube_sampling(config)

    else:
        raise NotImplementedError('that sampling strategy is not implemented, check sapphire/utils/sample_parameters.py')




# 
    