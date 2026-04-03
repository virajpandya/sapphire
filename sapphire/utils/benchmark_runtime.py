"""
this module benchmarks runtime for solving and auto-diffing through ODEs
for different numbers of halos/params and on CPU vs single GPU vs multi GPUs 
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

import pandas as pd
import os

from timeit import default_timer as timer
import gc

# user requested only a single run, so just transform their input dict into jnp.array format with 10**A_X, etc. 
def start(config,batch_solve,halo_index,full_params_arr,free_params_arr):

    ### when runtype='benchmark' batch_solve is actually (batch_solve, single_solve)
    ### so that we can manually jit and run single_solve for individual halos as well as benchmark its jacobian
    batch_solve, batch_jacfwd, single_solve, single_jacfwd = batch_solve

    # for jacfwd, slice subset of full_params_arr to just be the free indices 
    # TO DO: generalize this and clean it up
    inds_wanted = jnp.array([0,1,4,5,8,9,12,13])
    full_params_arr_wanted = full_params_arr[:,inds_wanted]
    
    ### set up a function that returns N shuffled combos of (params, halo_indices) 
    Nparams, Nhalos = full_params_arr.shape[0], halo_index.shape[0]
    def get_random_combos(Ncombos):
    
        # get N rand indices 1-1 mapped between params and halo_index
        param_idx = jax.random.randint(key(0), (Ncombos,), 0, Nparams) # note randint is exclusive of maxval
        halo_idx  = jax.random.randint(key(1), (Ncombos,), 0, Nhalos)
    
        return full_params_arr[param_idx], halo_index[halo_idx], full_params_arr_wanted[param_idx]

    ### single solves only relevant on CPU or single-GPU
    if jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) == 1:
        
        # for single solves, we'll take average runtimes of 1K random param-halo pairs
        Ncombos = 100 
        shuffled_params, shuffled_halos, shuffled_params_wanted = get_random_combos(Ncombos)    
        
        ### benchmark single halo solve 
    
        print('benchmarking single halo solve...',flush=True)
        
        # clear caches just in case 
        jax.clear_caches() 
        
        tsolves = []
        for i in range(Ncombos):
            if i%100 == 0: print('finished %i/%i'%(i,Ncombos),flush=True)
            tstart = timer()
            _ = single_solve(shuffled_halos[i],shuffled_params[i])
            _.ys[0].block_until_ready()
            tfinal = timer()
            tsolves.append((tfinal-tstart))

        tsolves_jit = tsolves[0]
        tsolves_avg = jnp.mean(jnp.array(tsolves[1:]))
        print('initial compile=%.5f sec, average post-jit=%.5f sec'%(tsolves_jit, tsolves_avg),flush=True)
    
        #### repeat for single jacfwd 
    
        print('benchmarking single halo jacfwd...',flush=True)
        
        # clear caches just in case 
        jax.clear_caches() 
        
        tjacs = []
        for i in range(Ncombos):
            if i%100 == 0: print('finished %i/%i'%(i,Ncombos),flush=True)
            tstart = timer()
            _ = single_jacfwd(shuffled_params_wanted[i],shuffled_halos[i],shuffled_params[i])
            _.block_until_ready()
            tfinal = timer()
            tjacs.append((tfinal-tstart))
        
        tjacs_jit = tjacs[0]
        tjacs_avg = jnp.mean(jnp.array(tjacs[1:]))
        print('jacfwd initial compile=%.5f sec, average post-jit=%.5f sec'%(tjacs_jit, tjacs_avg),flush=True)    

    else:
        tsolves_jit = jnp.nan
        tsolves_avg = jnp.nan
        tjacs_jit = jnp.nan
        tjacs_avg = jnp.nan
        
        
    ##### now benchmark shard-mapped halo solves
    
    print('benchmarking shard-mapped halo solves',flush=True)

    # Choose a base array of candidate Ncombos sizes to explore w/ shard_map
    # TO DO: generalize beyond hard-coded Ndevices=50 from cpu for 1-1 data point comparison    
    Ndevices = jax.local_device_count()
    if jax.devices()[0].platform == 'cpu':
        Ncombos_shmap = jnp.hstack([50,10**jnp.arange(2,6)]) 
    elif len(jax.devices('gpu')) in [1,4]:
        Ncombos_shmap = jnp.hstack([4,40,10**jnp.arange(2,7)]) 
    elif len(jax.devices('gpu')) == 8:
        Ncombos_shmap = jnp.hstack([8,8*10**jnp.arange(1,7)]) 
    else:
        print('ERROR in setting up Ncombos_shmap',flush=True) 


    ### now loop over different # ODE solves (x-axis) w/ batch_solve and record runtime for each 
    
    tsolves_batch = [] # runtimes 
    tsolves_batch_jit = []
    
    tstart0 = timer()
    
    for Ncombos in Ncombos_shmap:
        print('-----> working on Ncombos=%s'%Ncombos,flush=True)
    
        # first get shuffled params and halos for this Nbatch
        shuffled_params, shuffled_halos, shuffled_params_wanted = get_random_combos(Ncombos)
    
        # initial jit compile
        tstart = timer()
        _ = batch_solve(shuffled_halos, shuffled_params) 
        _.ys[0].block_until_ready()
        telapsed = timer()-tstart
        print('initial jit compile = %.2f'%telapsed,flush=True)
        tsolves_batch_jit.append(telapsed)
    
        # now time it 
        tstart = timer()
        _ = batch_solve(shuffled_halos, shuffled_params) 
        _.ys[0].block_until_ready()
        telapsed = timer()-tstart    
        print('subsequent runtime = %.2f'%telapsed,flush=True)
        tsolves_batch.append(telapsed)
    
        ### clear cache just in case
        jax.clear_caches()
        batch_solve._clear_cache()
        # del _, shuffled_params, shuffled_halos
        gc.collect()
        
        print('finished Ncombos=%s in %.2f sec'%(Ncombos,telapsed),flush=True)
    
    print('finished all Ncombos in %.2f sec'%(timer()-tstart0),flush=True)
        
    
    ##### batched jacfwds 
    
    print('benchmarking shard-mapped jacfwds',flush=True)    

    tjacs_batch = [] # runtimes 
    tjacs_batch_jit = []
    
    tstart0 = timer()
    
    for Ncombos in Ncombos_shmap:
        print('-----> working on Ncombos=%s'%Ncombos,flush=True)
    
        # first get shuffled params and halos for this Nbatch
        shuffled_params, shuffled_halos, shuffled_params_wanted = get_random_combos(Ncombos)
    
        # initial jit compile
        tstart = timer()
        _ = batch_jacfwd(shuffled_params_wanted, shuffled_halos, shuffled_params) 
        _.block_until_ready()
        telapsed = timer()-tstart
        print('initial jit compile = %.2f'%telapsed,flush=True)
        tjacs_batch_jit.append(telapsed)
    
        # now time it 
        tstart = timer()
        _ = batch_jacfwd(shuffled_params_wanted, shuffled_halos, shuffled_params) 
        _.block_until_ready()
        telapsed = timer()-tstart    
        print('subsequent runtime = %.2f'%telapsed,flush=True)
        tjacs_batch.append(telapsed)
    
        ### clear cache just in case
        jax.clear_caches()
        batch_jacfwd._clear_cache()
        # del _, shuffled_params, shuffled_halos
        gc.collect()
        
        print('finished Ncombos=%s in %.2f sec'%(Ncombos,telapsed),flush=True)
    
    print('finished all Ncombos in %.2f sec'%(timer()-tstart0),flush=True)
            
    
    
    ##### save in file
    
    # NOTE: this can be improved by auto-appending whatever params the user specifies in config 
    fname = os.path.join(config['output_path'],'outputs','benchmark_%s.npz'%(config['output_suffix']))
    
    # finally save npz
    jnp.savez(fname,
              tsolves_jit = tsolves_jit,
              tsolves_avg=tsolves_avg, 
              tjacs_jit = tjacs_jit,
              tjacs_avg = tjacs_avg,
              Ncombos_shmap = Ncombos_shmap,
              tsolves_batch_jit = tsolves_batch_jit,
              tsolves_batch = tsolves_batch,
              tjacs_batch_jit = tjacs_batch_jit,
              tjacs_batch = tjacs_batch)
              
              
    
    
    
    
    
    
    
    
    
    
    






# 
    