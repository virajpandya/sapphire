"""
this is a port of Viraj Pandya's early custom-built, minimal adaptive RK23 solver written from scratch in jax
that can optionally compute the augmented ODE system to get a variety of time-dependent Jacobians 

this follows section II.4 of Hairer 1993 and section 13.5.2 of Corless & Fillon 2013 and uses
- jax.lax.while_loop
- pre-allocated solution array size with max_steps
- adaptive error control (this makes jacfwd the best option currently)

this module is designed for any arbitrary input ODE system, initial conditions, parameters, forcing functions
it is intended for pedagogical and small-scope problems, and will be expanded in the future as needed

almost always this module is designed to be imported and run separately from the rest of sapphire like any
other diffeq solver package -- its minimalist nature helps expose the essentials of differentiable dynamics.
it is not designed for command line / config file like the default diffrax-based sapphire.run

for production-level work it is still best to use Patrick Kidger's diffrax package which implements much more
efficient, scalable solver and gradient methods in JAX -- see example usage in sapphire/solvers/diffrax.py 

"""

# in case user loads module separately from sapphire.run()
import os, sys
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # set for jax on GPU, doesn't affect CPU
from jax import config as jax_config
jax_config.update("jax_enable_x64", True) # required to accurately solve and take gradients through our diffeqs 

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, pmap, debug, jvp, vjp, jacrev, jacfwd, make_jaxpr, hessian, value_and_grad
from jax.lax import scan, fori_loop, while_loop
from jax.scipy.integrate import trapezoid
from jax.random import PRNGKey, key
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from interpax import Interpolator1D

from functools import partial 
from timeit import default_timer as timer
import numpy as np


# NOTE: more functionalities to be ported over soon for 
# time-dependent parameters and forcing functions, t_eval grid, dense interpolation
def setup(integrator,init_state,init_time,parameters,forcing_matrix,
          final_time,dt0,rtol,atol,max_steps,compute_jacobians=False,
          interp_nsteps=100):

    tstart0 = timer()
    
    ### following section II.4 of Hairer 1993 and section 13.5.2 of Corless & Fillon 2013
    ### adapted from Viraj Pandya's 2023 implementation from sapphire-jax prototype 
    
    # Bogacki-Shampine RK23 Butcher tableau
    A = jnp.array([[0.,   0.,   0., 0.],
                   [1/2,  0.,   0., 0.],
                   [0.,   3/4,  0., 0.],
                   [2/9,  1/3,  4/9, 0.]])
    
    c = jnp.array([0., 1/2, 3/4, 1.]) # step sizes (Runge-Kutta stages)
    b3 = jnp.array([2/9, 1/3, 4/9, 0.]) # 3rd order weights
    b2 = jnp.array([7/24, 1/4, 1/3, 1/8]) # 2nd order weights

    ### single rk stepper for any arbitrary butcher tableau for future extensibility if desired
    ### currently ignores FSAL (First Same As Last) optimization for RK23: k1=k4 for accepted steps
    
    def jax_rk_step(f, t, x, dt, fargs):
        """
        follows section 13.5.2 of Corless & Fillon 2013
    
        f = rhs of ODE system
        t = current time
        x = current state vector
        dt = overall timestep size
        fargs: (parameters, interps) input args to f
        """
        
        nstages = len(c)
        ndim = x.shape[0]
    
        # preallocate stage array
        k_init = jnp.zeros((nstages, ndim))
    
        def stage_body(i, k):
    
            ti = t + c[i] * dt
    
            # accumulate previous stages
            def accum_body(j, xi):
    
                return xi + dt * A[i, j] * k[j]
    
            xi = jax.lax.fori_loop(0,i,accum_body,x)
    
            ki = f(ti, xi, fargs)
            k = k.at[i].set(ki)
    
            return k
    
        # compute all stages
        k = jax.lax.fori_loop(0,nstages,stage_body,k_init)
    
        # combine stages
        x3 = x + dt * jnp.tensordot(b3,k,axes=1)
        x2 = x + dt * jnp.tensordot(b2,k,axes=1)
        
        return x3, x2   

    ### adaptive timestepper
    # set up safety factors for adaptive stepsizes 
    facsafe = 0.9
    facmin = 0.2
    facmax = 10.0
    
    def adapt_dt(dt,xnow,x2,x3,atol,rtol):
        """
        follows section II.4 of Hairer 1993
    
        dt: proposed step size
        ynow = current state vector
        y2 and y3 = proposed 2nd and 3rd order RK23 state vectors
        atol and rtol = assumed absolute and relative error tolerances
        """
        
        # compute maximum allowable error -- note that this is element-wise maximum 
        maxerr = atol + rtol*jnp.maximum(jnp.abs(xnow),jnp.abs(x3)) 
        
        # compute mean absolute error between RK3 relative to RK2 
        # this could alternatively be done by grouping and subtracting individual k stages with coefficients
        # note this is normalized element-wise by maxerr above 
        err = jnp.sqrt(jnp.mean(((x3 - x2)/maxerr)**2)) 
    
        err = jnp.maximum(err, 1e-16)
        
        # compute multiplicative factor for new optimal step size (taking into account safety factors)
        hfactor = facsafe * (1/err)**(1/3.)
        
        hfactor = jnp.clip(hfactor,facmin,facmax)
        
        # new corrected step size
        dt_new = hfactor * dt
        
        return dt_new, err

    
    # treat as static: rhs function f, max_steps
    @partial(jit,static_argnums=(0,8)) 
    def adaptive_rk23(f,t0,x0,fargs,
                      dt0,t_final,
                      atol,rtol,max_steps):
    
        # initialize time and state vectors
        ts = jnp.zeros(max_steps)
        xs = jnp.zeros((max_steps, x0.shape[0]))
        
        # set first elements to ICs 
        ts = ts.at[0].set(t0)
        xs = xs.at[0].set(x0)
        
        # initialize dict as pytree to track state 
        state = {"t": t0, # current time
                 "x": x0, # current state 
                 "dt": dt0, # current adaptive timestep
                 "istep": 0, # current step number (to compare to max_steps)
                 "ts": ts, # adaptive time vector
                 "xs": xs, # state vector time series
                 # "success": False # will stay False if max_steps reached = failed solution
                 # add accepted which is 1 if yes, else 0
                }     
        
        ### set up cond_fn and body_fn based on inputs
        def cond_fn(state):
    
            # if both are satisfied, then can keep trying more steps if needed, otherwise break
            return ((state["t"] < t_final) & (state["istep"] < max_steps - 1))
    
        def body_fn(state):
        
            # get current state
            t = state["t"]
            x = state["x"]
            dt = state["dt"]
            istep = state["istep"]
        
            # if approaching t_final, don't overshoot
            dt = jnp.minimum(dt, t_final - t)
        
            # do single rk23 step
            x3, x2 = jax_rk_step(f,t,x,dt,fargs)
        
            # compute new adaptive timestep and error between rk23
            dt_new, err = adapt_dt(dt,x,x2,x3,atol,rtol)
        
            # accept updates only if normalized err <= 1.0 
            accept = err <= 1.0
    
            def accept_fn(state):
    
                t_accept = t + dt
                istep_accept = istep + 1
                ts_accept = state["ts"].at[istep_accept].set(t_accept)
                xs_accept = state["xs"].at[istep_accept].set(x3)
    
                return {"t": t_accept,
                        "x": x3,
                        "dt": dt_new,
                        "istep": istep_accept,
                        "ts": ts_accept,
                        "xs": xs_accept,
                       }
    
            def reject_fn(state):
                
                return {"t": t,
                        "x": x,
                        "dt": dt_new,
                        "istep": istep,
                        "ts": state["ts"],
                        "xs": state["xs"],
                       }
                
        
            ### choose between accepted vs. keeping current values 
            new_state = jax.lax.cond(accept,accept_fn,reject_fn,state)
            
            return new_state
        
        # solve
        final_state = jax.lax.while_loop(cond_fn,body_fn,state)
    
        return final_state


    ##### interpolate all onto time grid with same # of output steps for vmap/shardmap purposes
    # could have user specify interp_t, including whether its just outputting at final time... 
    # currently not using dense interpolation (based on storing intermediate RK stages)
    interp_t_traj = jnp.linspace(init_time,final_time,interp_nsteps) 

    def interpfunc_traj(t_traj,vals_traj):
        interp_traj = Interpolator1D(t_traj,vals_traj,method="cubic2",extrap=(vals_traj[0], vals_traj[-1]))
        return interp_traj.__call__(interp_t_traj)

    # interp vmapped state vector 
    interpfunc_xvector = vmap(interpfunc_traj,in_axes=(None,1),out_axes=1)
    
    # for jacobian matrices, interp with double nested vmap over rows and columns
    interpfunc_matrix = vmap(vmap(interpfunc_traj,in_axes=(None,1),out_axes=(None,1)),in_axes=(None,1),out_axes=(None,1))

    
    ### need to set up and solve augmented ODE system -- will jit/vmap/shardmap below
    # inputs here are batched, the others like integrator, max_steps, ... are taken from global setup(inputs) 
    def solve_augmented(init_state,init_time,parameters,forcing_matrix):

        ### forcings is a matrix of shape (Nforcings+1, Ntimes) 
        ### the first row must be time (same units as integrator f's input time) 
        
        # convert forcing_matrix into list of interpolators for integrator rhs function
        # these can be static-jitted/vmapped
        forcing_interps = [Interpolator1D(forcing_matrix[0],forcing_values,method="cubic2",
                                          extrap=(forcing_values[0], forcing_values[-1]),) for forcing_values in forcing_matrix[1:]] 
        
        ### package fargs = (theta, forcings)
        fargs = (parameters, forcing_interps)  
        

        ### initialize augmented RHS
        Nx = init_state.shape[0]
        Ntheta = parameters.shape[0]

        # initial sensitivity matrix
        S0 = jnp.zeros((Nx, Ntheta))

        # augmented vector of state variables + flattened sensitivity matrix variables 
        x0_aug = jnp.concatenate([init_state,S0.reshape(-1)])

        # forward augmented sensitivity ODE approach
        # e.g., https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec09.pdf
        def augmented_integrator(t, x_aug, fargs):
            ### x_aug := jnp.array(x,S.flattened) 
        
            # unpack args
            theta, forcing_interps = fargs
        
            # unpack physical state vector portion of x_aug
            x = x_aug[:Nx]
        
            # unpack sensitivity matrix portion of x_aug and reshape from flattened
            S = x_aug[Nx:].reshape((Nx, Ntheta))
        
            # physical RHS
            dxdt = integrator(t, x, fargs)
        
            # Jacobian of rhs f wrt state using autodiff
            Jx = jax.jacfwd(lambda x_: integrator(t, x_, fargs))(x)
        
            # Jacobian of rhs f wrt parameters -- need to expose theta as an arg to autodiff wrt
            Jtheta = jax.jacfwd(lambda theta_: integrator(t, x, (theta_, forcing_interps)))(theta)
        
            # sensitivity matrix evolution ODE
            dSdt = Jx @ S + Jtheta
        
            # repack augmented RHS
            return jnp.concatenate([dxdt, dSdt.reshape(-1)])
                
        
        ### solve this augmented ODE system like any other 
        sol = adaptive_rk23(augmented_integrator,
                            init_time,x0_aug,fargs,
                            dt0,final_time,
                            atol,rtol,max_steps)

        ### extract x(t) and S(t) 
        def unpack_aug_sol(sol, init_state, parameters):
        
            xs = sol["xs"]
        
            Nx = init_state.shape[0]
            Ntheta = parameters.shape[0]
        
            x_traj = xs[:, :Nx]
            S_traj = xs[:, Nx:].reshape(-1, Nx, Ntheta)
        
            # limit to only the final # of steps actually solved, excluding padded max_steps
            ### TO DO: also restrict to only accepted steps
            t_traj = sol['ts'][:sol["istep"]+1]
            x_traj = x_traj[:sol["istep"]+1]
            S_traj = S_traj[:sol["istep"]+1]
        
            return t_traj, x_traj, S_traj
        
        t_traj, x_traj, S_traj = unpack_aug_sol(sol,init_state,parameters)

        ### compute df/dx along solution trajectory using vmap over x(t) 
        
        @partial(jit,static_argnums=(0,))
        def compute_Jx_traj(integrator, ts, xs):
        
            def Jx_at_point(t, x):
                return jax.jacfwd(lambda x_: integrator(t, x_, fargs))(x)
        
            return jax.vmap(Jx_at_point,in_axes=(0,0))(ts, xs)
        
        Jx_traj = compute_Jx_traj(integrator,t_traj,x_traj)
                
    
        ### similarly compute df/dtheta along solution trajectory using vmap over x(t) 
        
        @partial(jit,static_argnums=(0,))
        def compute_Jtheta_traj(integrator, ts, xs):
        
            parameters, forcing_interps = fargs
        
            def Jth_at_point(t, x):
                return jax.jacfwd(lambda par: integrator(t, x, (par, forcing_interps)))(parameters)
        
            return jax.vmap(Jth_at_point,in_axes=(0,0))(ts, xs)
        
        Jtheta_traj = compute_Jtheta_traj(integrator,t_traj,x_traj)


        ##### interpolate all onto time grid with same # of output steps for vmap/shardmap purposes
        
        # interp vmapped state vector 
        interp_x_traj = interpfunc_xvector(t_traj,x_traj) 
        
        # for jacobian matrices, interp with double nested vmap over rows and columns
        interp_S_traj = interpfunc_matrix(t_traj,S_traj)
        interp_Jx_traj = interpfunc_matrix(t_traj,Jx_traj)
        interp_Jtheta_traj = interpfunc_matrix(t_traj,Jtheta_traj)
        
        ### return sol dict, t_traj, x(t), S(t), Jx(t), Jtheta(t)
        out_sapphire = {'sol':sol,
                        't':interp_t_traj,
                        'x':interp_x_traj,
                        'S':interp_S_traj,
                        'Jx':interp_Jx_traj,
                        'Jtheta':interp_Jtheta_traj}

        return out_sapphire

    ### only need to solve baseline ODE system, no sensitivities, etc.
    def solve_original(init_state,init_time,parameters,forcing_matrix):

        ### forcings is a matrix of shape (Nforcings+1, Ntimes) 
        ### the first row must be time (same units as integrator f's input time) 
        
        # convert forcing_matrix into list of interpolators for integrator rhs function
        # these can be static-jitted/vmapped
        forcing_interps = [Interpolator1D(forcing_matrix[0],forcing_values,method="cubic2",
                                          extrap=(forcing_values[0], forcing_values[-1]),) for forcing_values in forcing_matrix[1:]] 
        
        ### package fargs = (theta, forcings)
        fargs = (parameters, forcing_interps)  
        
        ### solve the user-input ODE system like any other 
        sol = adaptive_rk23(integrator,
                            init_time,init_state,fargs,
                            dt0,final_time,
                            atol,rtol,max_steps)

        ### extract x(t) excluding extra padded max_steps 
        # TO DO: also exclude any rejected steps
        def unpack_sol(sol, init_state):
        
            xs = sol["xs"]
        
            Nx = init_state.shape[0]
        
            x_traj = xs[:, :Nx]
        
            # limit to only the final # of steps actually solved, excluding padded max_steps
            ### TO DO: also restrict to only accepted steps
            t_traj = sol['ts'][:sol["istep"]+1]
            x_traj = x_traj[:sol["istep"]+1]
        
            return t_traj, x_traj
        
        t_traj, x_traj = unpack_sol(sol,init_state)

        # interp vmapped state vector 
        interp_x_traj = interpfunc_xvector(t_traj,x_traj)       

        ## return signature
        out_sapphire = {'sol':sol,'t':interp_t_traj,'x':interp_x_traj}

        return out_sapphire


    ### port over vmap and shard_map to try many ICs, params, etc.
    if compute_jacobians is True:
        out_sapphire = jit(solve_augmented)(init_state,init_time,parameters,forcing_matrix)

    elif compute_jacobians is False:
        tstart = timer()
        out_sapphire = jit(solve_original)(init_state,init_time,parameters,forcing_matrix)
        print('compiled in %.5f sec'%(timer()-tstart),flush=True)

        tstart = timer()
        out_sapphire = jit(solve_original)(init_state,init_time,parameters,forcing_matrix)
        print('jitted ran in %.5f sec'%(timer()-tstart),flush=True)    
    
    print('finished in %.5f sec'%(timer()-tstart0),flush=True)
    return out_sapphire
    
        
#