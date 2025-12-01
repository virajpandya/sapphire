"""
this module provides convenience functions for generating mock parameters given prior bounds
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


# user requested only a single run, so just transform their input dict into jnp.array format with 10**A_X, etc. 
def single_fixed(config):

    # first get just the free parameters non-transformed (as would be input into adam, NUTS, etc.)
    params_bounds = config['sampling_config']['params_bounds']
    params_free = list(params_bounds.keys())    
    params_fixed_astro = config['params_fixed_astro']
    
    free_params_arr = jnp.array([params_fixed_astro[k] for k in params_free])

    ### now set up the full transformed params arr
    # required parameter order for sapphire
    # NOTE: change API to just use dicts throughout (care is needed for gradients)
    params_order = ['A_M','alpha0_M','alphaz_M','beta_M',
                    'A_E','alpha0_E','alphaz_E','beta_E',
                    'A_SF','alpha0_SF','alphaz_SF','beta_SF',
                    'A_Z','alpha0_Z','alphaz_Z','beta_Z']  

    # create jnp.array and append
    # NOTE: change API to get rid of automatically assign realization # at end somehow if user requested written outputs
    full_params_arr = jnp.array([params_fixed_astro[k] for k in params_order]+[0])
    print('full_params_arr\n',full_params_arr,flush=True)    

    # transform 10**A_X 
    # NOTE: this can be a one-liner, or done above in the list comprehension
    # NOTE: this needs to be generalized using dictionaries 
    full_params_arr = full_params_arr.at[0].set(10**full_params_arr[0]) # A_M
    full_params_arr = full_params_arr.at[4].set(10**full_params_arr[4]) # A_E
    full_params_arr = full_params_arr.at[8].set(10**full_params_arr[8]) # A_SF
    full_params_arr = full_params_arr.at[12].set(10**full_params_arr[12]) # A_Z        

    # (full parameters array, just the free non-transformed parameters)
    return full_params_arr, free_params_arr
    

### different from above -- this can all probably be combined neatly later 
def single_random(config):

    inference_config = config['inference_config']
    sampling_config = config['sampling_config']

    params_bounds = sampling_config['params_bounds']
    params_free = list(params_bounds.keys())

    # first generate single random set of free parameter values (non-transformed)
    base_key = jax.random.key(sampling_config['rng_sample'])
    keys = jax.random.split(base_key, len(params_free))
    free_params_dict = {pname: jax.random.uniform(keys[i], shape=(), minval=plow, maxval=phigh)
                        for i, (pname, (plow, phigh)) in enumerate(params_bounds.items())}    

    # convert free parameters to array as expected by rest of sapphire, adam, nuts, etc.
    free_params_arr = jnp.array([free_params_dict[k] for k in params_free])

    # now convert to full transformed parameters array
    # required parameter order for sapphire
    # NOTE: change API to just use dicts throughout (care is needed for gradients)
    params_order = ['A_M','alpha0_M','alphaz_M','beta_M',
                    'A_E','alpha0_E','alphaz_E','beta_E',
                    'A_SF','alpha0_SF','alphaz_SF','beta_SF',
                    'A_Z','alpha0_Z','alphaz_Z','beta_Z'] 
    
    params_fixed_astro = config['params_fixed_astro']
    
    params_merged = {**params_fixed_astro, **free_params_dict} # second dict overwrites first dict
    
    full_params_arr = jnp.array([params_merged[k] for k in params_order]+[0])
    print('full_params_arr\n',full_params_arr,flush=True)    

    # transform 10**A_X 
    # NOTE: this can be a one-liner, or done above in the list comprehension
    # NOTE: this needs to be generalized using dictionaries 
    full_params_arr = full_params_arr.at[0].set(10**full_params_arr[0]) # A_M
    full_params_arr = full_params_arr.at[4].set(10**full_params_arr[4]) # A_E
    full_params_arr = full_params_arr.at[8].set(10**full_params_arr[8]) # A_SF
    full_params_arr = full_params_arr.at[12].set(10**full_params_arr[12]) # A_Z        

    # (full parameters array, just the free non-transformed parameters)
    return full_params_arr, free_params_arr

    


##### latin hypercube sampling [this is pseudo-lhs... need to put in the random shuffling step]
def latin_hypercube_sampling(config):

    sampling_config = config['sampling_config']

    ### first, if on CPU or multiple GPUs, enforce Nsamples = integer multiple of Ndevices (num_cpus or num_gpus)
    # Dec 1 -- this check is only needed if running in TSG mode, otherwise for inference mode only return a single random
    if (jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) > 1) and (config['runtype']=='sampling'):
        if sampling_config['Nsamples'] % jax.local_device_count():
            raise ValueError('Nsamples must equal to or an integer multiple of num_cpus or num_gpus unless num_gpus=1')
    
    params_bounds = sampling_config['params_bounds']
    params_free = list(params_bounds.keys())

    lower_bounds = jnp.array([params_bounds[k][0] for k in params_free])
    upper_bounds = jnp.array([params_bounds[k][1] for k in params_free])
    print('lower_bounds',lower_bounds,flush=True)
    print('upper_bounds',upper_bounds,flush=True)
    
    # first create a base random key based on mock_num, then split into as many different keys as free parameters we want
    base_key = jax.random.key(sampling_config['rng_sample'])
    keys = jax.random.split(base_key, len(params_free))

    Nsamples = sampling_config['Nsamples'] 
    
    # Generate mock truth values (log10 for A_X, linear for alpha0_X)
    free_params_dict = {pname: jax.random.uniform(keys[i], shape=(Nsamples,), minval=plow, maxval=phigh)
                        for i, (pname, (plow, phigh)) in enumerate(params_bounds.items())}
    
    # convert input params_fixed so every element is a jax array for later below
    params_fixed = {k: jnp.array(v,dtype='float64') for k, v in config['params_fixed_astro'].items()} 

    # now merge fixed and free params to create Nguess params dicts 
    # note: for duplicated keys between params_fixed and params_lhs, the value of params_lhs will be used 
    params_merged = [{**params_fixed, **{k: float(free_params_dict[k][i]) for k in params_free}} for i in range(Nsamples)] 
    
    # first define order of parameters expected by sapphire
    params_order = ['A_M','alpha0_M','alphaz_M','beta_M',
                    'A_E','alpha0_E','alphaz_E','beta_E',
                    'A_SF','alpha0_SF','alphaz_SF','beta_SF',
                    'A_Z','alpha0_Z','alphaz_Z','beta_Z'] 
    
    # now convert those Nguess params dicts into 2D jax array of parameters 
    full_params_arr = jnp.array([[params_merged[i][k] for k in params_order] for i in range(Nsamples)])
    
    # as usual hstack the final column of all zeros for realization # placeholder
    full_params_arr = jnp.concatenate([full_params_arr, jnp.zeros((full_params_arr.shape[0],1))],axis=1)

    # finally change the A_X to 10**A_X since that's what solve_halo expects
    full_params_arr = full_params_arr.at[:,0].set(10**full_params_arr[:,0]) # A_M
    full_params_arr = full_params_arr.at[:,4].set(10**full_params_arr[:,4]) # A_E
    full_params_arr = full_params_arr.at[:,8].set(10**full_params_arr[:,8]) # A_SF
    full_params_arr = full_params_arr.at[:,12].set(10**full_params_arr[:,12]) # A_Z       

    # print('params_lharr\n',params_lharr,flush=True)
    print('full_params_arr.shape\n',full_params_arr.shape,flush=True)

    ### convert params_lhs to arr for other processing 
    free_params_arr = arr = jnp.column_stack([free_params_dict[k] for k in params_free])
    
    # (full_params array, free non-transformed params arr)
    return full_params_arr, free_params_arr




def get(config):

    sampling_config = config['sampling_config']
    inference_config = config['inference_config'] 

    # both single and inference runtypes only evaluate a single parameter set at a time 
    if config['runtype'] == 'single' or (config['runtype']=='inference' and inference_config['random_mock']==False):
        return single_fixed(config)
    
    elif config['runtype'] == 'inference' and inference_config['random_mock']==True:
        return single_random(config)
    
    elif config['runtype'] == 'sampling': 
        # can implement alternative samplers later
        return latin_hypercube_sampling(config)

    else:
        raise NotImplementedError('that sampling strategy is not implemented, check sapphire/utils/sample_parameters.py')




# 
    