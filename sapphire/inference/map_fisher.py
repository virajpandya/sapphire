"""
this module identifies the best of N adam's (maximum a posterior aka MAP)
and computes Fisher information matrix at that local point
"""

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15
from functools import partial 
from timeit import default_timer as timer

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

import optax


### NOTE: this currently not only computes MAP/Fisher but also packages and returns trace, etc. for saving -- move that to utils?
def from_adam(config,hess_loss_func,out_adam,true_params):

    sampling_config = config['sampling_config']
    params_bounds = sampling_config['params_bounds']
    params_free = list(params_bounds.keys())    
    
    ### TO DO: change all names from _3 to more generic since user may request any arbitrary Nguess trajectories
    (finali_3, final_params_3,
     trace_params_3, trace_loss_3, trace_grads_3,trace_updatenorm_3,trace_dlogL_3) = out_adam  


    # can just get this from finali_3.shape ... 
    Nchain = config['inference_config']['Nchain']
    
    @jit
    @vmap
    def compute_hessians(final_params):
        ### final_params = shape (Nguess, Nparams)
        
        hess_arr = hess_loss_func(final_params)

        # if all eigvalsh are positive, then this is a minimum (if not, its saddle or max)
        hess_flag = jnp.all(jnp.linalg.eigvalsh(hess_arr)>0) 
    
        return hess_arr, hess_flag
    
    
    tstart = timer()
    hess_arrs, hess_flags = compute_hessians(final_params_3)
    print('%2f sec, compile hess_flags=%s'%(timer()-tstart,hess_flags),flush=True)
    
    tstart = timer()
    hess_arrs, hess_flags = compute_hessians(final_params_3)
    print('%2f sec, run hess_flags=%s'%(timer()-tstart,hess_flags),flush=True)
        
        
    ### jittable, vmappable function that selects lowest loss minimum
    
    @jit
    # @partial(vmap,in_axes=(0,0,0,0))
    def identify_best(final_params,final_losses,final_hessians, final_flags):
    
        # when there is at least one true minimum, identify it and return its stuff
        def true_branch(_):
            masked_losses = jnp.where(final_flags, final_losses, jnp.inf) # inf-out non-minima
            min_idx = jnp.argmin(masked_losses)
            hess = final_hessians[min_idx,:,:]
            Finv = jnp.linalg.inv(hess)
            return (final_losses[min_idx],final_params[min_idx],hess,final_flags[min_idx],Finv,min_idx)
    
        # if there are no true minima at all, just return lowest loss saddle and Finv = all nan's
        def false_branch(_):
            min_idx = jnp.argmin(final_losses)
            hess = final_hessians[min_idx,:,:]
            Finv = jnp.full_like(hess, jnp.nan)
            return (final_losses[min_idx],final_params[min_idx],hess,final_flags[min_idx],Finv,min_idx)
    
        return jax.lax.cond(jnp.any(final_flags), true_branch, false_branch, operand=None)    
        
    
    # store just final losses
    final_losses_3 = jnp.array([trace_loss_3[i][finali_3[i]-1] for i in range(Nchain)])
    
    # test call -- need to add vmap batch dimension (the vmap is over different Nmocks, not Nguesses)
    tstart = timer()
    # out_best = identify_best(final_params_3[None,:],final_losses_3[None,:],hess_arrs[None,:],hess_flags[None,:])
    # July 17 -- no longer doing multiple mocks, so dont need vmap
    # Nov 27 -- can add vmap back in for multi-mock mode on multi-GPUs
    out_best = identify_best(final_params_3,final_losses_3,hess_arrs,hess_flags)
    print('finished identify_best in %2f sec'%(timer()-tstart),flush=True)
    
    # print('out_best\n',out_best,flush=True)
    
    ### finally a non-jax function that extracts and saves what we need in npz
    
    # unpack out_best
    best_adam_loss, theta_map, adam_hess_arr, adam_hess_flag, Finv_adam, best_adam_index  = out_best    

    print('best_adam_index',best_adam_index,flush=True)

    ### print best parameter +/- Fisher uncertainty and (if mock) true parameter 
    sigma_map = jnp.sqrt(jnp.diag(Finv_adam))

    ### if mock mode, compute Finv at truth
    if config['inference_config']['mock'] is True:

        hess_true_arr = hess_loss_func(true_params)

        # if all eigvalsh are positive, then this is a minimum (if not, its saddle or max)
        hess_true_flag = jnp.all(jnp.linalg.eigvalsh(hess_true_arr)>0) 

        print('hess_true_flag',hess_true_flag,flush=True)
        
        if hess_true_flag == True:
            Finv_true = jnp.linalg.inv(hess_true_arr)
        else:
            Finv_true = jnp.full_like(hess_true_arr, jnp.nan)

        print('Finv_true',Finv_true,flush=True)   

        sigma_true = jnp.sqrt(jnp.diag(Finv_true))
        print('sigma_true',sigma_true,flush=True)

        ### compute accuracy (bias) := # standard deviations theta_map is away from truth
        accuracy_map = (theta_map-true_params)/sigma_true
    
    else:
        hess_true_flag = jnp.nan
        Finv_true = jnp.full((len(true_params),len(true_params)), jnp.nan)
        sigma_true = jnp.full(len(true_params),jnp.nan)    
        accuracy_map = jnp.full(len(true_params),jnp.nan)    

    
    for i in range(len(theta_map)):
        print('%s = %.3f +/- %.3f, true=%.3f, sigma_true=%.3f, accuracy=%.3f'%(params_free[i],theta_map[i],sigma_map[i],
                                                                               true_params[i],sigma_true[i],accuracy_map[i]),flush=True)

    
    # get trace of loss and params for best adam, cut off at 
    best_finali = finali_3[best_adam_index] # - 1 # to cut off padded part of trace
    best_adam_trace_loss = trace_loss_3[best_adam_index][:best_finali]
    best_adam_trace_params = trace_params_3[best_adam_index][:best_finali]
    best_adam_trace_grads = trace_grads_3[best_adam_index][:best_finali]
    
    # print('finali=%s, best finali=%s'%(finali_3,best_adam_trace_loss.shape),flush=True)

    # TO DO: clean this up 
    return (*out_best, best_finali, best_adam_trace_loss, best_adam_trace_params, best_adam_trace_grads,
            hess_true_flag,Finv_true,sigma_true, accuracy_map)
    


    

###