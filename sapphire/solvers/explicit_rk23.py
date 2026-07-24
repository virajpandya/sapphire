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


def setup(integrator,rhs_terms,init_state,init_time,parameters,forcing_matrix,
          final_time,dt0,rtol,atol,max_steps,output_times,
          compute_jacobians=False,apply_shard_map=False):
    """    
    integrator = main rhs function f that returns state derivatives (based on modular rhs_terms)
    rhs_terms = auxiliary function that returns all rhs terms along solution trajectory
    init_state = initial state vector x0 
    init_time = initial time t0
    parameters = free parameters in format expected by f 
    forcing_matrix = (Ntimes, Ninputs)
    final_time = final time t1
    dt0 = initial timestep guess (recommended instead of adaptive guessers, typically 1e-10)
    rtol = relative error tolerance for RK23 adaptive timestepper
    atol = absolute error tolerance for RK23 adaptive timestepper
    max_steps = maximum number of steps to try
    output_times = jnp array of times at which to output dense solution
    compute_jacobians = whether to return various jacobians at output_times
    apply_shard_map = whether to shard parameter*halos solves over multiple devices 

    the following can be vmapped/shardmapped if the leading dimension is all same batch size:
    init_state, init_time, parameters, forcing_matrix, output_times
    """

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
        
        return x3, x2, k 

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

        # jax.debug.print("xnow={}", xnow)
        # jax.debug.print("x2={}", x2)
        # jax.debug.print("x3={}", x3)
        # jax.debug.print("maxerr={}", maxerr)
        # jax.debug.print('err={}',err)
        
        err = jnp.maximum(err, 1e-16)
        
        # compute multiplicative factor for new optimal step size (taking into account safety factors)
        hfactor = facsafe * (1/err)**(1/3.)
        
        hfactor = jnp.clip(hfactor,facmin,facmax)
        
        # new corrected step size
        dt_new = hfactor * dt
        
        return dt_new, err


    ### new manual local cubic hermite interpolator for dense_output along ODE solution
    # this manual one is necessary since we re-use the RK23 stages for endpoint derivatives of interpolation
    # https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    # https://erikerlandson.github.io/blog/2013/03/16/smooth-gradients-for-cubic-hermite-splines/
    # https://www.rose-hulman.edu/~finn/CCLI/Notes/day09.pdf
    def cubic_hermite_interp(t, t0, t1, x0, x1, k0, k1):

        # first normalize timestep [t0,t1] to [0,1], then convert t into fractional interval step tau
        h = t1 - t0
        tau = (t - t0) / h
        
        # compute Hermite basis functions
        h00 = 2*tau**3 - 3*tau**2 + 1
        h10 = tau**3 - 2*tau**2 + tau
        h01 = -2*tau**3 + 3*tau**2
        h11 = tau**3 - tau**2
        
        return h00*x0 + h10*h*k0 + h01*x1 + h11*h*k1

    
    
    # treat as static: rhs function f, max_steps
    @partial(jit,static_argnums=(0,9)) 
    def adaptive_rk23(f,t0,x0,fargs,
                      dt0,t_final,output_ts,
                      atol,rtol,max_steps):
        
        # initialize dense interpolated solution state vector
        output_xs = jnp.zeros((output_ts.shape[0], x0.shape[0]))        
        
        # initialize dict as pytree to track state 
        state = {"t": t0, # current time
                 "x": x0, # current state 
                 "dt": dt0, # current adaptive timestep
                 "istep": 0, # current step number (to compare to max_steps)
                 "output_xs": output_xs, # state vector time series
                 # "success": False # will stay False if max_steps reached = failed solution
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
            x3, x2, k = jax_rk_step(f,t,x,dt,fargs)
        
            # compute new adaptive timestep and error between rk23
            dt_new, err = adapt_dt(dt,x,x2,x3,atol,rtol)
        
            # accept updates only if normalized err <= 1.0 
            accept = err <= 1.0
    
            def accept_fn(state):

                # use masking to identify all output_times within (t,t_next) where we need to interpolate solution
                t_next = t + dt
                in_interval = (output_ts >= t) & (output_ts <= t_next) & (t_next > t)
                mask = jnp.expand_dims(in_interval, axis=-1) # for broadcasting purposes w/ and w/o vmap
                
                # vmappable function to interpolate at all output_ts 
                def single_time_interp(t_target):
                    return cubic_hermite_interp(t_target, t, t_next, x, x3, k[0], k[-1])

                # compute interpolated solutions xs at all output_ts
                ### WARNING: this is expensive, interpolating over all ts, instead could do some kind of index mapping?
                x_interp_all = jax.vmap(single_time_interp)(output_ts)

                # use mask to update output_xs for output_ts within [t,t_next] 
                updated_output_xs = jnp.where(mask, x_interp_all, state["output_xs"])

                # update to next time, new state vector, new dt, increment step, and update output_xs
                return {"t": t_next,
                        "x": x3,
                        "dt": dt_new,
                        "istep": state["istep"] + 1,
                        "output_xs": updated_output_xs
                       }

            # keep everything same except change to dt_new and increment istep 
            def reject_fn(state):
                
                return {"t": t,
                        "x": x,
                        "dt": dt_new,
                        "istep": state["istep"] + 1, # increment istep even for rejects, in case we hit total max_steps
                        "output_xs": state["output_xs"],
                       }
                
        
            ### choose between accepted vs. keeping current values 
            new_state = jax.lax.cond(accept,accept_fn,reject_fn,state)
            
            return new_state
        
        # solve
        final_state = jax.lax.while_loop(cond_fn,body_fn,state)
    
        return final_state

    
    ### need to set up and solve augmented ODE system -- will jit/vmap/shardmap below
    # inputs here are batched, the others like integrator, max_steps, ... are taken from global setup(inputs) 
    def solve_augmented(init_state,init_time,parameters,forcing_matrix):

        ### forcings is a matrix of shape (Nforcings+1, Ntimes) 
        ### the first row must be time (same units as integrator f's input time) 
        
        # convert forcing_matrix into list of jittable/mappable interpolators for integrator rhs function
        # forcing_interps = [Interpolator1D(forcing_matrix[0],forcing_values,method="cubic2",
        #                                   extrap=(forcing_values[0], forcing_values[-1]),) for forcing_values in forcing_matrix[1:]] 
        
        ### package fargs = (theta, forcings)
        # fargs = (parameters, forcing_interps)  
        fargs = (parameters, forcing_matrix)  

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
            theta, forcings = fargs
        
            # unpack physical state vector portion of x_aug
            x = x_aug[:Nx]
        
            # unpack sensitivity matrix portion of x_aug and reshape from flattened
            S = x_aug[Nx:].reshape((Nx, Ntheta))
        
            # physical RHS
            dxdt = integrator(t, x, fargs)
        
            # Jacobian of rhs f wrt state using autodiff
            Jx = jax.jacfwd(lambda x_: integrator(t, x_, fargs))(x)
        
            # Jacobian of rhs f wrt parameters -- need to expose theta as an arg to autodiff wrt
            Jtheta = jax.jacfwd(lambda theta_: integrator(t, x, (theta_, forcings)))(theta)
        
            # sensitivity matrix evolution ODE
            dSdt = Jx @ S + Jtheta
        
            # repack augmented RHS
            return jnp.concatenate([dxdt, dSdt.reshape(-1)])
                
        
        ### solve this augmented ODE system like any other 
        sol = adaptive_rk23(augmented_integrator,
                            init_time,x0_aug,fargs,
                            dt0,final_time,output_times,
                            atol,rtol,max_steps)
        
        ### extract x(t) and S(t) 
        def unpack_aug_sol(sol, init_state, parameters):
        
            output_xs = sol["output_xs"]
        
            Nx = init_state.shape[0]
            Ntheta = parameters.shape[0]
        
            x_traj = output_xs[:, :Nx]
            S_traj = output_xs[:, Nx:].reshape(-1, Nx, Ntheta)
        
            return x_traj, S_traj
        
        x_traj, S_traj = unpack_aug_sol(sol,init_state,parameters)

        ### compute df/dx along solution trajectory using vmap over x(t) 
        
        @partial(jit,static_argnums=(0,))
        def compute_Jx_traj(integrator, ts, xs):
        
            def Jx_at_point(t, x):
                return jax.jacfwd(lambda x_: integrator(t, x_, fargs))(x)
        
            return jax.vmap(Jx_at_point,in_axes=(0,0))(ts, xs)
        
        Jx_traj = compute_Jx_traj(integrator,output_times,x_traj)
                
    
        ### similarly compute df/dtheta along solution trajectory using vmap over x(t) 
        
        @partial(jit,static_argnums=(0,))
        def compute_Jtheta_traj(integrator, ts, xs):
        
            parameters, forcings = fargs
        
            def Jth_at_point(t, x):
                return jax.jacfwd(lambda par: integrator(t, x, (par, forcings)))(parameters)
        
            return jax.vmap(Jth_at_point,in_axes=(0,0))(ts, xs)
        
        Jtheta_traj = compute_Jtheta_traj(integrator,output_times,x_traj)

        ### update sol object to include the new jacobians
        sol['output_xs'] = x_traj
        sol['output_S'] = S_traj
        sol['output_Jx'] = Jx_traj
        sol['output_Jtheta'] = Jtheta_traj

        return sol

    ### only need to solve baseline ODE system, no sensitivity jacobians, etc.
    def solve_original(init_state,init_time,parameters,forcing_matrix):

        ### forcings is a matrix of shape (Nforcings+1, Ntimes) 
        ### the first row must be time (same units as integrator f's input time) 
        
        # convert forcing_matrix into list of jittable/mappable interpolators for integrator rhs function
        # this automatically deals with empty forcing_matrix inputs like ()
        # forcing_interps = [Interpolator1D(forcing_matrix[0],forcing_values,method="cubic2",
        #                                   extrap=(forcing_values[0], forcing_values[-1]),) for forcing_values in forcing_matrix[1:]] 
        
        ### package fargs = (theta, forcings)
        # fargs = (parameters, forcing_interps)  
        fargs = (parameters, forcing_matrix)  
        
        ### solve the user-input ODE system like any other 
        sol = adaptive_rk23(integrator,
                            init_time,init_state,fargs,
                            dt0,final_time,output_times,
                            atol,rtol,max_steps)

        return sol

    ### also define a second function that returns all auxiliary rhs terms along solution trajectory
    def aux_evaluator(logt,logy,parameters,forcing_matrix):
        """
        logt = array of solution output times, shape = (Ntimes,)
        logy = array of solution state vector, shape = (Nbatch, Ntimes, Nstate) if batched, else (Ntimes, Nstate)
        parameters = array of parameters, shape = (Nbatch, Nparams) if batched, else (Nparams,)
        forcing_matrix = array of forcing matrices, shape = (Nbatch, Nforcings+1, Ntimes) if batched, else (Nforcings+1, Ntimes)
    
        output shape for dict elements = (Nbatch, Ntimes) if batched, else (Ntimes,) 
        """

        # convert forcing_matrix into list of jittable/mappable interpolators for integrator rhs function
        # this automatically deals with empty forcing_matrix inputs like ()
        # forcing_interps = [Interpolator1D(forcing_matrix[0],forcing_values,method="cubic2",
        #                                   extrap=(forcing_values[0], forcing_values[-1]),) for forcing_values in forcing_matrix[1:]] 
        
        ### package fargs = (theta, forcings)
        # fargs = (parameters, forcing_interps)  
        fargs = (parameters, forcing_matrix)  

        ### this is necessary in case Nstate>1 for broadcasting inside rhs_terms -- but abstracting it away from user
        logy = jnp.swapaxes(logy,-1,-2) 
        
        terms = rhs_terms(logt,logy,fargs)
    
        return terms        
    

    ### auto-applies jit and vmap based on shapes of input init_state, init_time, parameters, forcing_matrix
    # NOTE: could extend to map over different final_time, etc.
    # NOTE: can make this more elegant ... 
    def apply_jit_vmap(solve_func,aux_func):

        ### there are many other combinations that can be added as needed     

        if len(init_state.shape) == 1 and len(parameters.shape) == 1:
            print('only jit, no vmap',flush=True)
            return (jit(solve_func), 
                    jit(aux_func))

        if len(init_state.shape) == 1 and len(parameters.shape) > 1:
            print('vmap over parameters but not ICs',flush=True)
            return (jit(vmap(solve_func,in_axes=(None,None,0,None))),
                    jit(vmap(aux_func,in_axes=(None,None,0,None))))

        elif len(init_state.shape) > 1 and len(parameters) == 0 and len(forcing_matrix)==0:
            print('only vmap over ICs with empty parameters/forcings',flush=True)
            return (jit(vmap(solve_func,in_axes=(0,None,None,None))), 
                    jit(vmap(aux_func,in_axes=(None,0,None,None))))
        
        elif len(init_state.shape) > 1 and len(parameters.shape) == 1 and len(forcing_matrix.shape)==1:
            print('only vmap over ICs',flush=True)
            return (jit(vmap(solve_func,in_axes=(0,None,None,None))),
                    jit(vmap(aux_func,in_axes=(None,0,None,None))))

        elif len(init_state.shape) > 1 and len(parameters.shape) > 1 and len(forcing_matrix.shape)==1:
            print('vmap over ICs and parameters',flush=True)
            return (jit(vmap(solve_func,in_axes=(0,None,0,None))),
                    jit(vmap(aux_func,in_axes=(None,0,0,None))))

        elif len(init_state.shape) > 1 and len(parameters.shape) == 1 and len(forcing_matrix.shape) > 1:
            print('vmap over ICs and forcings',flush=True)
            return (jit(vmap(solve_func,in_axes=(0,None,None,0))),
                    jit(vmap(aux_func,in_axes=(None,0,None,0))))

        elif len(init_state.shape) > 1 and len(parameters.shape) > 1 and len(forcing_matrix.shape) > 1:
            print('vmapping over ICs, params, and forcings',flush=True)
            return (jit(vmap(solve_func,in_axes=(0,None,0,0))),
                    jit(vmap(aux_func,in_axes=(None,0,0,0))))

    ### alternative function that auto-applies jit, shard_map and nested vmaps 
    ### this is useful for parallelizing ODE solves over multiple devices
    ### user must make sure that # of halos or parameter sets = integer multiple of Ndevices
    def apply_jit_shardmap(solve_func,aux_func):
        """
        currently this only does two main use cases: 
        1. shard over params (for predictive checks and training set generation)
        2. shard over ICs for a single parameter set (for optimization/inference)

        #2 also accounts for cases where params is expanded to match # ICs
        e.g., if base parameters has some that depend on each object (e.g., final halo mass)
        """

        ### first auto-detect what use case we are in based on global inputs to this module
        ### and set up inner/outer vmap and shard_map axes 

        if len(parameters) == 0 and len(forcing_matrix) == 0:

            print('applying shard_map over many ICs with EMPTY parameters array',flush=True)

            vmapped_solver = vmap(solve_func,in_axes=(0,None,None,None))
            vmapped_aux_evaluator = vmap(aux_func,in_axes=(None,0,None,None))        

            shmap_specs_solver = (PartitionSpec('i'),PartitionSpec(),PartitionSpec(),PartitionSpec())
            shmap_specs_aux = (PartitionSpec(),PartitionSpec('i'),PartitionSpec(),PartitionSpec())                 
            
        
        elif len(parameters.shape) == 1: 

            # shard over many ICs for a single parameter set -- classic optimization/inference case
            # parameters.shape=(Nparams), init_state.shape=(Nhalos,Nstate), forcing_matrix.shape=(Nhalos,Nforcings+1,Ntimes)

            print('applying shard_map over many ICs for single parameter set',flush=True)

            ### note that input args for (logt,logy) are reversed for aux_evaluator vs solve_func
            ### and note here that sol_xs axis also has a new leading batch dimension
            vmapped_solver = vmap(solve_func,in_axes=(0,None,None,0))
            vmapped_aux_evaluator = vmap(aux_func,in_axes=(None,0,None,0))        

            shmap_specs_solver = (PartitionSpec('i'),PartitionSpec(),PartitionSpec(),PartitionSpec('i'))
            shmap_specs_aux = (PartitionSpec(),PartitionSpec('i'),PartitionSpec(),PartitionSpec('i'))         
            
        elif len(parameters.shape) == 2:

            # shard over many ICs and matching 1-1 number of parameter sets (and forcing functions)
            # the behavior here requires user to specify cartesian vs pairwise combo of parameters and init_state 
            # parameters.shape=(Nprior,Nparams), init_state.shape=(Nhalos,Nstate), forcing_matrix.shape=(Nhalos,Nforcings+1,Ntimes)

            print('applying %s shard_map over ICs and parameter sets'%apply_shard_map,flush=True)

            if apply_shard_map == 'pairwise':
                # this version is when each parameter is 1-1 mapped with init_state 
                vmapped_solver = vmap(solve_func,in_axes=(0,None,0,0))
                vmapped_aux_evaluator = vmap(aux_func,in_axes=(None,0,0,0))

                shmap_specs_solver = (PartitionSpec('i'),PartitionSpec(),PartitionSpec('i'),PartitionSpec('i'))         
                shmap_specs_aux = (PartitionSpec(),PartitionSpec('i'),PartitionSpec('i'),PartitionSpec('i'))        
            
            elif apply_shard_map == 'cartesian':
                # this version is when you want to cross every parameter set with every halo
                # this assumes ICs and forcing_matrix are 1-1 mapped -- later can add cases for no forcing, etc. 
                vmapped_solver = vmap(vmap(solve_func,in_axes=(0,None,None,0)),in_axes=(None,None,0,None))
                vmapped_aux_evaluator = vmap(vmap(aux_func,in_axes=(None,0,None,0)),in_axes=(None,None,0,None))

                shmap_specs_solver = (PartitionSpec(),PartitionSpec(),PartitionSpec('i'),PartitionSpec())         

                # for aux evaluator, assume sol_xs has new leading batch dimension of Nprior
                shmap_specs_aux = (PartitionSpec(),PartitionSpec('i'),PartitionSpec('i'),PartitionSpec())
                
            else:
                raise ValueError('apply_shard_map must be None, False, cartesian or pairwise')            
        
        elif len(parameters.shape) == 3:

            # shard over many parameters for predictive checks or training set generation
            # this version is when a single set of base parameters are tiled for each IC so extra inner vmap is needed
            # parameters.shape=(Nprior,Nhalos,Nparams), init_state.shape=(Nhalos,Nstate), forcing_matrix.shape=(Nhalos,Nforcings+1,Ntimes)        

            print('applying shard_map over many parameters',flush=True)

            
            vmapped_solver = vmap(vmap(solve_func,in_axes=(0,None,0,0)),in_axes=(None,None,0,None))
            vmapped_aux_evaluator = vmap(vmap(aux_func,in_axes=(None,0,0,0)),in_axes=(None,0,0,None))

            shmap_specs_solver = (PartitionSpec(None),PartitionSpec(),PartitionSpec('i'),PartitionSpec(None))
            
            # for aux evaluator, assume sol_xs has new leading batch dimension of Nprior
            shmap_specs_aux = (PartitionSpec(),PartitionSpec('i'),PartitionSpec('i'),PartitionSpec(None))  

        else:
            
            raise NotImplementedError('combo of parameters.shape and init_state.shape not yet implemented')
            
        """ 
        WARNING: one use case is missing where len(parameters.shape)==2 and parameters.shape[0]!=init_state.shape[0] 
        but you still want combos w/ outer_vmap 
        """            
        
        ### set up device mesh [so far this is universal to any runtype below but may be made more sophisticated in future]
        Ndevices = jax.local_device_count()
        
        mesh = Mesh(mesh_utils.create_device_mesh((Ndevices,)), axis_names=('i',))  
        
        if jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) > 1:
            print('applying multi-CPU/GPU shard_map over %s devices'%Ndevices,flush=True)

            batch_solve = jit(shard_map(vmapped_solver,
                                        mesh=mesh,
                                        in_specs=shmap_specs_solver,
                                        out_specs=PartitionSpec('i'),check_rep=False,))
            
            batch_aux = jit(shard_map(vmapped_aux_evaluator,
                                        mesh=mesh,
                                        in_specs=shmap_specs_aux, 
                                        out_specs=PartitionSpec('i'),check_rep=False))

            return batch_solve, batch_aux
        
        elif len(jax.devices('gpu')) == 1: 
            raise NotImplementedError('still need to port this over')


    ### return requested solver function and aux_evaluator
    ### can probably add in_axes arg to apply_jit_map and call separate for solver vs aux_evaluator
    if compute_jacobians is True:
        
        if apply_shard_map in [None,False]:
            return apply_jit_vmap(solve_augmented,aux_evaluator)
            
        else:
            return apply_jit_shardmap(solve_augmented,aux_evaluator)
            
    elif compute_jacobians is False:
        
        if apply_shard_map in [None,False]:
            return apply_jit_vmap(solve_original,aux_evaluator)
            
        else:
            return apply_jit_shardmap(solve_original,aux_evaluator)
        
    
        
#