"""
sapphire -- a next-generation multi-zone dynamical model of galaxy formation 

this defines the run() function to be used when importing sapphire as a module
__main__.py imports run() for use on the command line with python -m sapphire <inputs> 
"""


# load general modules unrelated to jax
import os, sys

# set for jax on GPU, doesn't affect CPU
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' 

from glob import glob
import multiprocess
import numpy as np
from timeit import default_timer as timer

# load sapphire modules that do not require dependency injection
from sapphire.utils import read_config

# NOTE: this should be further modularized as needed, including any dependency injections (loading of modules based on config dict)
def run(config,custom_solver_inputs=None):
    """
    driver module that calls all the other submodules in the required order.
    
    config: user-provided dict possibly with .yaml baseline config file 

    custom_solver_inputs = (integrator,init_state,init_time,parameters,forcings,
                            final_time,dt0,rtol,atol,max_steps,compute_Jacobians) 
                            
    if custom_solver_inputs is not None, then everything is handled by sapphire/solvers/explicit_rk23.py
    otherwise sapphire is run according to the input config 
    
    NOTE: at some point I plan to offload a lot of the if/else checks on the config dict to another module 
    """

    if custom_solver_inputs is not None:
        print('sending custom_solver_inputs to sapphire/solvers/explicit_rk23.py',flush=True)

        from sapphire.solvers import explicit_rk23
        return explicit_rk23.setup(*custom_solver_inputs) 
        
    
    # just to benchmark entire runtime from start to finish
    tstart0 = timer()

    ### read config based on user input
    config = read_config.get(config,verbose=True) 
    
    # if multiple CPU cores requested, set environment flag (idiosyncrasy of jax on CPUs)
    # NOTE: there has to be a cleaner, compact way to load all of these jax packages up top... 
    
    import jax   
    jax.config.update("jax_enable_x64", True) # required to accurately solve and take gradients through our diffeqs 
    jax.config.update('jax_num_cpu_devices', config['num_cpus']) # newer jax versions don't need the XLA environment flag above
    import jax.numpy as jnp
    
    print('jax version',jax.__version__,flush=True)
    print('jax devices',jax.devices(),flush=True)

    ### first create the output subdirs if requested
    from sapphire.utils import write_results
    write_results.create_output_subdirs(config)
    
    # now read the trees into a dict where the key is halo name/ID and element is an astropy Table
    # read and interpolate into the required jax halo matrix format
    print('Reading trees and converting to jax matrices...',flush=True)
    from sapphire.utils import read_trees
    halo_matrix, halo_coeff_matrix, halo_tinit, halo_index, ts_interp = read_trees.get(config)
        
    # use dependency injection to read in the requested physical model, then the associated parameter functions using dependency injection
    from sapphire.models import model_loader
    integrator, saveat_fn, list_parameterizations = model_loader.get(config,verbose=True)

    ### decide how to run: single set of parameters or sampling many params simultaneously
    # this module will automatically return params based on runtype and inference_config['random_mock'] etc. 
    # if mock=True, free_params_arr is the "true" mock parameter array (non-transformed, ready to input into loss func, adam, nuts, etc.)    
    from sapphire.utils import setup_parameters 
    full_params_arr, free_params_arr = setup_parameters.get(config)

    ### now set up the ODE solver to run on single parameter set or batched parameter set 
    ### TO DO: move this parsing / dependency injection to sapphire/solvers
    if config['solver_config']['engine'] == 'diffrax':
        from sapphire.solvers import diffrax as solver
    else:
        raise ValueError('other diffeq solver engines are not yet implemented') 

    # this should already be configured to be compatible with full_params_arr and free_params_arr based on config['runtype'] 
    batch_solve = solver.setup(config,integrator,saveat_fn,halo_matrix,halo_coeff_matrix,halo_tinit,ts_interp)


    if config['runtype'] == 'benchmark':
        from sapphire.utils import benchmark_runtime
        out_sapphire = benchmark_runtime.start(config,batch_solve,halo_index,full_params_arr,free_params_arr)


    ##### modularize the following (especially runtype=inference with engine=nuts)
    
    # NOTE: also add option to return extracted quantities and summary statistics
    if config['runtype'] in ['single','sampling']: 

        ### automatically run 
        print('benchmarking full-batch ODE runtime for runtype=%s...'%config['runtype'],flush=True) 

        ### in this case, batch_solve is actually (batch_solve, batch_jacfwd) -- see solvers/diffrax.py
        batch_solve, batch_jacfwd = batch_solve
    
        ### Finally solve the ODEs for single or multiple halos
        # this is clunky, can push the solve down to sapphire.solvers.diffrax itself, returning batch_solve only for inference later 
        
        tstart = timer()
        sol = batch_solve(halo_index,full_params_arr)
        print(sol.ys[0][0][0],flush=True)
        print('full-batch initial jit+sol took %.3f sec'%(timer()-tstart),flush=True)
    
        if config['post_jit_benchmark'] is True: 
            tstart = timer()
            sol = batch_solve(halo_index,full_params_arr)
            print(sol.ys[0][0][0],flush=True)
            print('full-batch jitted sol took %.3f sec'%(timer()-tstart),flush=True)

        
        # prepare generic out_sapphire tuple to be returned at end below
        out_sapphire = (sol, free_params_arr,)

        ### if requested, also compute sensitivity jacobians
        if config['return_sensitivity_jacobians'] is True:

            print('computing sensitivity jacobians...',flush=True)

            # for jacfwd, slice subset of full_params_arr to just be the free indices 
            # TO DO: generalize this and clean it up  [just like in utils/benchmark_runtime.py]
            inds_wanted = jnp.array([0,1,4,5,8,9,12,13])
            if config['runtype'] == 'single':
                full_params_arr_wanted = full_params_arr[inds_wanted]
            elif config['runtype'] == 'sampling':
                full_params_arr_wanted = full_params_arr[:,inds_wanted]
            
            ### note the different call signature / args for batch_jacfwd (see solvers/diffrax.py)    
            tstart = timer()
            sol_jac = batch_jacfwd(full_params_arr_wanted,halo_index,full_params_arr)
            print(sol_jac[0],flush=True)
            print('full-batch initial jit jacobian took %.3f sec'%(timer()-tstart),flush=True)

            if config['post_jit_benchmark'] is True:
                tstart = timer()
                sol_jac = batch_jacfwd(full_params_arr_wanted,halo_index,full_params_arr)
                print(sol_jac[0],flush=True)
                print('full-batch jitted jacobian took %.3f sec'%(timer()-tstart),flush=True)

            # append jacobian to return
            out_sapphire += (jnp.squeeze(sol_jac),)
            
    """ 
    add module here to optionally compress+save data directly for TSG/ILI when in 'sampling' mode
    """
    
    ### alternatively run inference if requested
    ######### Can this whole thing be put into a sapphire.inference wrapper module to keep driver / __ clean?
    if config['runtype'] in ['inference']: # can merge Lucas' ILI option in the future

        ### in this case, batch_solve is actually (batch_solve, batch_jacfwd) -- see solvers/diffrax.py
        # for runtype=inference, we only need batch_solve -- sapphire/inference module will compute grad(loss) on its own
        batch_solve, batch_jacfwd = batch_solve
        
        from sapphire import inference
        
        inference_config = config['inference_config']

        params_fixed_astro = config['params_fixed_astro']
        params_bounds = config['sampling_config']['params_bounds']
        params_free = list(params_bounds.keys())
        
        ### set up mock or obs constraints

        # NOTE: not sure if defining this is necessary (can just use free_params_arr directly)?
        true_params = jnp.full(len(params_free),jnp.nan) # by default, there is no truth if not running in mock mode
        
        if inference_config['fit_mock'] is True:

            true_params = free_params_arr # NOTE: is this renaming necessary for below? check when modularizing/refactoring... 
            print('true mock parameters\n',free_params_arr,flush=True)          
            
            print('summarizing mock data...',flush=True)
            # TO DO: push this import earlier, and use __init__ to load only summaries parent module
            import sapphire.summaries.gaussian_kernel_regression as gkr # can generalize later

            # need to solve true mock sol
            tstart = timer()
            sol = batch_solve(halo_index,full_params_arr) 
            print(sol.ys[0][0][0],flush=True)
            print('full-batch initial jit+sol took %.3f sec'%(timer()-tstart),flush=True)
            
            # although these are mocks, we use the same obs_ prefix for consistency with rest of code below
            # NOTE: return order of obs_stats is same order as expected by sapphire.inference module below
            obs_stats = gkr.summarize_mock(sol)

        ### alternatively get obs_stats for data
        elif inference_config['fit_obs'] is True: 
            
            print('reading observational constraints for %s'%config['obs_name'],flush=True)
            from sapphire.utils import read_obs
            obs_stats = read_obs.get(config)

        if config['inference_config']['engine'] == 'adam':
            
            print('setting up model for adam...',flush=True)
    
            ### should push this down to sapphire.inference based on config, for explicit or ILI etc. 
            loss_func, grad_loss_func, hess_loss_func = inference.explicit_likelihood.setup(config,halo_index,obs_stats,batch_solve)
    
            ### push this down to sapphire.utils or a new inference.coverage module or something 
            
            # by default nan's if not running in mock mode 
            true_loss, true_grad_loss = jnp.nan, jnp.full(len(free_params_arr),jnp.nan)

            ### if mock mode, print true loss, grad, etc. to verify things work at true minimum, and for benchmarking
            # TO DO: can this block be combined with the other fit_mock==True block above?
            if inference_config['fit_mock'] is True:
                
                tstart = timer()
                true_loss = loss_func(true_params)
                tjit_loss = timer()-tstart

                tstart = timer()
                true_loss = loss_func(true_params)
                tloss = timer()-tstart
    
                print('true loss=%s, jit %.5f sec, post-jit %.5f sec'%(true_loss,tjit_loss,tloss),flush=True)
    
                tstart = timer()
                true_grad_loss = grad_loss_func(true_params)
                tjit_grad = timer()-tstart

                tstart = timer()
                true_grad_loss = grad_loss_func(true_params)
                tgrad = timer()-tstart            
                
                print('true grad loss=%s, jit %.5f sec, post-jit %.5f sec'%(true_grad_loss,tjit_grad,tgrad),flush=True)
    
                tstart = timer()
                true_hess_loss = hess_loss_func(true_params)
                tjit_hess = timer()-tstart

                tstart = timer()
                true_hess_loss = hess_loss_func(true_params)
                thess = timer()-tstart            
                
                print('true hess loss=%s, jit %.5f sec, post-jit %.5f sec'%(true_hess_loss,tjit_hess,thess),flush=True)
    
                true_hess_flag = jnp.all(jnp.linalg.eigvalsh(true_hess_loss)>0)
                print('true_hess_flag',true_hess_flag,flush=True)
                # return true_hess_flag
    
                true_Finv = jnp.linalg.inv(true_hess_loss)
                print('true_Finv',true_Finv,flush=True)
                
        
            print('calling adam for MAP+Fisher...',flush=True)

            out_adam = inference.run_adam.setup(config,loss_func,grad_loss_func)

            ### compute MAP and fisher/covariance matrix
            out_map_fisher = inference.map_fisher.from_adam(config,hess_loss_func,out_adam,true_params)

            ### compute posterior predictive checks at MAP 
            post_preds_map = inference.posterior_predictive_checks.adam_map_fisher(config,obs_stats,out_map_fisher,
                                                                                   batch_solve,halo_index)
            
            ### write file if requested
            if inference_config['write_output'] == True:
                write_results.write_adam_map_fisher(config,out_map_fisher,full_params_arr,free_params_arr,
                                                    true_loss,true_grad_loss,obs_stats,post_preds_map)
        
            
            # return outputs for interactive analysis
            # NOTE: this should be improved, so the timer print statement below always run
            # return config, out_map_fisher, full_params_arr, free_params_arr

            # collect outputs to return for interactive analysis if desired
            out_sapphire = (config, out_map_fisher, full_params_arr, free_params_arr,
                            true_loss, true_grad_loss, obs_stats, post_preds_map)
                

            ### additional figure making -- though maybe this can be imported and done separately


        ### this whole thing (w/ adam above) needs to be pushed down to sapphire/inference module
        elif config['inference_config']['engine'] in ['nuts','aies']:
            print('calling numpyro...',flush=True)

            # out_numpyro = inference.run_numpyro.setup(config,halo_index,obs_stats,batch_solve)
            numpyro_model, numpyro_model_args, numpyro_params_free = inference.define_numpyro.setup(config,halo_index,obs_stats,batch_solve)
            out_numpyro = inference.run_numpyro.setup(config,numpyro_model,numpyro_model_args,numpyro_params_free,savefigs=True)

            # TO DO: fix general out for interactive analysis
            out_sapphire = out_numpyro
        
        else:
            raise ValueError('inference_config.engine must be either adam, nuts or aies',flush=True)
    

    ### somehow have this total runtime always print upon program exit, even if return was up above somewhere 
    print('Finished in %.1f min -- thank you for using sapphire!'%((timer()-tstart0)/60.),flush=True)

    return out_sapphire



###