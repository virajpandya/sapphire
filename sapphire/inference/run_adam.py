"""
this module runs adam 

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

import optax


def setup(config,loss_func,grad_loss_func,obs_stats,loss_and_grad_func=None):
    """ 
    July 24 2025 -- vmap on single GPU, otherwise sequential on CPU (not enough cores)
    NOTE: this should be updated to use shard_map for multi-GPU 

    June 4 2026 -- new loss_and_grad_func returns both (loss, grad_loss), supersedes the other two
    """

    inference_config = config['inference_config']
    sampling_config = config['sampling_config']

    params_bounds = sampling_config['params_bounds']
    params_free = list(params_bounds.keys())
    Nfree = len(params_free)
    
    ### generate initial guess
    base_key = jax.random.key(int(config['rng_init']))
    init_keys = jax.random.split(base_key, len(list(params_bounds.keys())))
    
    # generate init param dict
    # the conditional for shape makes it easier to deal with single guess (no need to access [0]) -- can be generalized later
    
    initial_params_dict = {pname: jax.random.uniform(init_keys[i], 
                                                shape=(inference_config['Nadam'],),
                                                minval=plow, maxval=phigh)
                           for i, (pname, (plow, phigh)) in enumerate(params_bounds.items())}

    ### rewrite initial_params from dict to array of shape (Nadam,Nfree) preserving parameter order
    
    initial_params_arr = jnp.stack([initial_params_dict[k] for k in params_free], 
                                   axis=1)
    
    print('initial_params_arr\n',initial_params_arr,flush=True)    

    ### June 2026 -- minimal wrapper that returns both loss and grad(loss) if needed
    if loss_and_grad_func is None:

        # assumes loss and grad funcs are already jitted and/or jit-compatible...
        @jit
        def loss_and_grad_func(params,obs_stats):
            return loss_func(params,obs_stats), grad_loss_func(params,obs_stats)
    

    def run_adam_while(init_params,loss_and_grad_func):
    
        max_loops = 1000
    
        """ 
        July 21 -- need faster initial gradient, but steep decay later to descend deep minimum
        instead of init_value=1e-2 and clip(1) -- use clip(1-10) and init_value=1e-1 , and smaller transition steps
        """
        
        # learning rate schedule
        schedule = optax.exponential_decay(init_value=1e-1,
                                           decay_rate=0.8,
                                           transition_steps=25,
                                           end_value=1e-4)

        # chain together gradient transformations 
        optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                                optax.scale_by_adam(),
                                optax.scale_by_schedule(schedule),
                                optax.scale(-1.0))
    
        opt_state = optimizer.init(init_params)
        
        # initialize loss for jax.lax.while_loop
        loss0, grad0 = loss_and_grad_func(init_params,obs_stats)
        # print('loss0',loss0,flush=True)
    
        # allocate (max_loops, Nfree) trace arrays
        trace_params = jnp.zeros((max_loops, Nfree))
        trace_loss = jnp.zeros((max_loops,))
        trace_grads = jnp.zeros((max_loops, Nfree))
        trace_updatenorm = jnp.zeros((max_loops,))
        trace_dlogL = jnp.zeros((max_loops,))
    
        # Initial state tuple: (step, params, opt_state, loss, converged_flag, trace_array)
        init_state = (0, init_params, opt_state, loss0, False, 
                      trace_params,trace_loss,trace_grads,trace_updatenorm,trace_dlogL)

        # condition function for stopping jax.lax.while_loop
        def cond_fun(state):
            (i, params, opt_state, prev_loss, converged, 
            trace_params,trace_loss,trace_grads,trace_updatenorm,trace_dlogL) = state
            return jnp.logical_and(i < max_loops, ~converged)

        # body function for jax.lax.while_loop
        def body_fun(state):
            (i, params, opt_state, prev_loss, converged, 
             trace_params,trace_loss,trace_grads,trace_updatenorm,trace_dlogL) = state
            
            # eval loss and grad loss
            loss, grads = loss_and_grad_func(params,obs_stats)

            # apply updates to parameters based on adam grad transforms
            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # compute norm of parameter update vector and change in loss
            update_norm = jnp.linalg.norm(updates)
            # update_norm = jnp.linalg.norm(jnp.asarray([updates[k] for k in params_free]))
            dlogL = jnp.abs(prev_loss - loss) / (jnp.abs(loss) + 1e-12)

            # compute converged flag
            converged = jnp.logical_and(update_norm < 1e-3, dlogL < 1e-3)

            # jax.debug.print("i={} params={} loss={} grads={} update_norm={} dlogL={} converged={}",i,params,
            #                 loss,grads,update_norm,dlogL,converged)            

            # update trace arrays that we will return
            trace_params = trace_params.at[i].set(new_params)
            trace_loss = trace_loss.at[i].set(loss)
            trace_grads = trace_grads.at[i].set(grads)
            trace_updatenorm = trace_updatenorm.at[i].set(update_norm)
            trace_dlogL = trace_dlogL.at[i].set(dlogL)
            
            return i + 1, new_params, opt_state, loss, converged, trace_params, trace_loss, trace_grads, trace_updatenorm, trace_dlogL

        # call jax.lax.while_loop for adam iterations
        (final_i, final_params, _, _, _, 
         final_trace_params,final_trace_loss,final_trace_grads,
         final_trace_updatenorm,final_trace_dlogL) = jax.lax.while_loop(cond_fun, body_fun, init_state)
    
        # return final_params, final_trace[:final_i]  # only return valid steps
        return final_i,final_params,final_trace_params,final_trace_loss,final_trace_grads,final_trace_updatenorm,final_trace_dlogL


    #### TO DO: clean this up... 
    if jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) == 1:
        print('---------> sequential run_adam_while on CPU or single GPU',flush=True)

        # no vmap over multiple initial guesses / trajectories if running on single CPU node
        # TBD whether we allow multiple adams on single GPU
        run_adam_while = jit(run_adam_while,static_argnums=(1,)) # don't jit-trace the input loss and grad(loss) functions
    
        # we'll stack outputs afterwards so all are same shape (3, XXX) and name as vmap-GPU version
        finali_3,final_params_3,trace_params_3,trace_loss_3,trace_grads_3,trace_updatenorm_3,trace_dlogL_3 = [],[],[],[],[],[],[]
    
        tstart = timer()

        ### on single CPU node (and maybe single GPU), process multiple adams sequentially
        for ic_num in range(inference_config['Nadam']):

            ### get initial params dict for current chain
            # initial_params_this_dict = {k:initial_params[k][ic_num] for k in initial_params.keys()}
            initial_params_this = initial_params_arr[ic_num]
            print('ic_num, initial_params_this\n',ic_num,initial_params_this,flush=True)
            
            
            tstarti = timer()
        
            # call adam 
            out_adam = run_adam_while(initial_params_this,loss_and_grad_func)
            this_finali = out_adam[0].block_until_ready()
            telapsedi = timer()-tstarti 
            print('finished ic_num=%s in %.2f sec, finali=%s'%(ic_num,telapsedi,this_finali),flush=True)

            print('final params\n',out_adam[1],flush=True)
            
            finali_3.append(out_adam[0])
            final_params_3.append(out_adam[1])
            trace_params_3.append(out_adam[2])
            trace_loss_3.append(out_adam[3])
            trace_grads_3.append(out_adam[4])
            trace_updatenorm_3.append(out_adam[5])
            trace_dlogL_3.append(out_adam[6])
        
        # convert the above into jnp arrays
        finali_3 = jnp.asarray(finali_3)
        final_params_3 = jnp.asarray(final_params_3)
        trace_params_3 = jnp.asarray(trace_params_3)
        trace_loss_3 = jnp.asarray(trace_loss_3)
        trace_grads_3 = jnp.asarray(trace_grads_3)
        trace_updatenorm_3 = jnp.asarray(trace_updatenorm_3)
        trace_dlogL_3 = jnp.asarray(trace_dlogL_3)
    
    
    elif len(jax.devices('gpu')) > 1: #== 1 or :

        raise NotImplementedError('multi-GPU adam not ported over yet...')
        
        print('---------> vmap run_adam_while over multi-GPU',flush=True)
    
        run_adam_while = jit(vmap(run_adam_while,in_axes=(0,None,None)),static_argnums=(1,))
    
        tstart = timer()
        
        (finali_3, final_params_3,
         trace_params_3, trace_loss_3, trace_grads_3,
         trace_updatenorm_3,trace_dlogL_3) = run_adam_while(initial_params_arr,loss_and_grad_func)


    
    return (finali_3, final_params_3,
            trace_params_3, trace_loss_3, trace_grads_3,trace_updatenorm_3,trace_dlogL_3)



###