"""
this module computes finite-difference Jacobians and Hessians 
and compares them to auto-diff gradients following Appendix B of Pandya+26 
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
def jacobians(config,halo_index,halo_matrix,batch_solve):

    ### when runtype='finitediff' batch_solve is actually (parallel_jac_autodiff, parallel_jac_finitediff, single_solve) 
    parallel_jac_autodiff, parallel_jac_finitediff, single_solve = batch_solve

    ### pick a single random MW-mass from halo_index 
    inds_mw = jnp.where((halo_matrix[:,1,-1]>12.0) & (halo_matrix[:,1,-1]<12.3))[0]
    rand_index = inds_mw[0][None,...] # add placeholder leading vmap batch dimension 

    ### set up arrays of tol_ode and finite-diff epsilons
    tols_ode = jnp.array([1e-6,1e-8,1e-10,1e-12])
    epsilons = jnp.array([1e-1,1e-2,1e-3,1e-4,1e-5,1e-6])

    ### sample parameters from latin hypercube 
    # assumes 8 GPUs so 4 batches of 125 params each
    from sapphire.utils import setup_parameters
    sampling_config = config['sampling_config']
    sampling_config['method'] = 'latin_hypercube'
    sampling_config['posterior_file'] = ''
    sampling_config['Nsamples'] = 1000
    config['sampling_config'] = sampling_config

    full_params_lharr, free_params_lharr = setup_parameters.latin_hypercube_sampling(config)
    print('latin hypercube param shapes',full_params_lharr.shape, free_params_lharr.shape,flush=True)

    # slice subset of params_lh to just be the free indices 
    inds_wanted = jnp.array([0,1,4,5,8,9,12,13]) # TO DO: generalize this and clean it up
    full_params_lharr_wanted = full_params_lharr[:,inds_wanted]    

    ### first auto-diff 
    with jax.log_compiles(): 
        tstart = timer()
        jac_autodiff_lh = parallel_jac_autodiff(full_params_lharr_wanted,rand_index,full_params_lharr, tols_ode)
        print(jac_autodiff_lh[0][0][0])
        print('finished in %.2f sec'%(timer()-tstart))                     
    
    jac_autodiff_lh = jnp.squeeze(jac_autodiff_lh)
    
    print('jac_autodiff_lh.shape',jac_autodiff_lh.shape,flush=True)

    ### now finite-diff
    with jax.log_compiles():
        tstart = timer()
        jac_finitediff_lh = parallel_jac_finitediff(full_params_lharr,rand_index,tols_ode,epsilons)
        print(jac_finitediff_lh[0][0][0])
        print('finished in %.2f sec'%(timer()-tstart))
    
    jac_finitediff_lh = jnp.squeeze(jac_finitediff_lh)
    
    print('jac_finitediff_lh.shape',jac_finitediff_lh.shape,flush=True)

    ##### compute error matrix summaries for plotting in ipynb

    ### first compute residual matrix between autodiff 1e-6, 1e-8 and 1e-10 vs. 1e-12 
    jac_err_adself = jnp.array([(jac_autodiff_lh[:,i] - jac_autodiff_lh[:,3])/(0.5*(jnp.abs(jac_autodiff_lh[:,i])+jnp.abs(jac_autodiff_lh[:,3])))
                                for i in range(3)])
    print('jac_err_adself.shape',jac_err_adself.shape,flush=True)

    ### now finite-difference vs autodiff using same atol=rtol for each
    ### for a given eps_fdiff: for the same atol=rtol, compute residual jacobian matrix, take its norm, take nanmean of all
    
    jac_err_compare = jnp.array([(jac_autodiff_lh - jac_finitediff_lh[:,:,i,:,:])/(0.5*(jnp.abs(jac_autodiff_lh)+jnp.abs(jac_finitediff_lh[:,:,i,:,:])))
                                 for i in range(jac_finitediff_lh.shape[2])])
    
    print('jac_err_compare.shape',jac_err_compare.shape,flush=True)

    """ repeat above to get 16-50-84 percentiles of residual matrix norms """
    
    jac_err_adself_norm = jnp.nanpercentile(jnp.linalg.norm(jac_err_adself,axis=(2,3)),jnp.array([16,50,84]),axis=1)
    print('jac_err_adself_norm.shape, jac_err_adself_norm',jac_err_adself_norm.shape, jac_err_adself_norm,flush=True)

    jac_err_compare_norm = jnp.nanpercentile(jnp.linalg.norm(jac_err_compare, axis=(3, 4)),jnp.array([16,50,84]),axis=1)
    print('jac_err_compare_norm.shape, jac_err_compare_norm',jac_err_compare_norm.shape, jac_err_compare_norm,flush=True)

    
    ############## now joint Mhalo-theta 2D grid sampling
    ############## NOTE: maybe this should be moved to a separate function since it is more expensive than above 

    print('now working on jointly sampling Mhalo-parameter space...',flush=True)

    ### since this is very expensive we will only do two tolerances, and fix epsilon=1e-4
    tols_ode2 = jnp.array([1e-8,1e-12])

    ### to make it manageable on memory/speed/logging, we'll do 10 batches of 100 halos each
    rand_halo_batches100 = jnp.split(halo_index,10)    

    ##### first autodiff
    # empty list to store each batch's jacobians of shape (Nrealizations, Ntols, Nhalos, Nstate, Nparams) = (1000, 4, 100, 7, 9)
    batches_jac_autodiff100 = []
    
    with jax.log_compiles():
        tstart0 = timer()
        for ibatch, rand_halo_batch in enumerate(rand_halo_batches100):
            tstart = timer()
            batches_jac_autodiff100.append(parallel_jac_autodiff(full_params_lharr_wanted,rand_halo_batch,full_params_lharr,tols_ode2))
            print('finished autodiff batch %i in %.2f sec'%(ibatch,timer()-tstart),flush=True) 
    
    print('finished all autodiff batches in %.2f sec'%(timer()-tstart0),flush=True)

    ### now concatenate into the usual shape (Nrealizations,Ntols,Nhalos,Nstate,Nparams) preserving order
    # note: Ntols and Nhalos axis order got switched just because of how I nested my vmaps 
    
    concat_jac_autodiff = jnp.concatenate(batches_jac_autodiff100, axis=1)
    concat_jac_autodiff = jnp.squeeze(concat_jac_autodiff)
    print('concat_jac_autodiff.shape',concat_jac_autodiff.shape,flush=True)

    ##### now finitediff
    # empty list to store each batch's jacobians of shape (Nrealizations, Ntols, Nhalos, Nstate, Nparams) = (1000, 4, 100, 7, 9)
    batches_jac_finitediff100 = []
    
    with jax.log_compiles():
        tstart0 = timer()
        for ibatch, rand_halo_batch in enumerate(rand_halo_batches100):
            tstart = timer()
            batches_jac_finitediff100.append(parallel_jac_finitediff(full_params_lharr,rand_halo_batch,tols_ode2,jnp.array([1e-4])))
            print('finished finitediff batch %i in %.2f sec'%(ibatch,timer()-tstart),flush=True) 
    
    print('finished all finitediff batches in %.2f sec'%(timer()-tstart0),flush=True)

    ### now concatenate as before
    concat_jac_finitediff = jnp.concatenate(batches_jac_finitediff100, axis=1)
    concat_jac_finitediff = jnp.squeeze(concat_jac_finitediff)
    print('concat_jac_finitediff.shape',concat_jac_finitediff.shape,flush=True)

    ### again compute a bunch of summary statistics for error matrices

    ### first autodiff 1e-8 vs 1e-12 
    ad8, ad12 = concat_jac_autodiff[:,:,0,:,:], concat_jac_autodiff[:,:,1,:,:]    
    ad_err = (ad8-ad12) / (0.5*(jnp.abs(ad8)+jnp.abs(ad12)))    

    ### now finite (eps=1e-4) vs auto diff for atol=rtol=1e-8
    fd8 = concat_jac_finitediff[:,:,0,:,:]
    fd8_err = (fd8-ad8) / (0.5*(jnp.abs(fd8)+jnp.abs(ad8)))    

    ### repeat for fd vs ad with 1e-12
    fd12 = concat_jac_finitediff[:,:,1,:,:]
    fd12_err = (fd12-ad12) / (0.5*(jnp.abs(fd12)+jnp.abs(ad12)))
        
    ##### finally compute extra arrays for external plotting
    params_jac = full_params_lharr_wanted.copy()
    params_jac = params_jac.at[:,0].set(jnp.log10(params_jac[:,0]))
    params_jac = params_jac.at[:,2].set(jnp.log10(params_jac[:,2]))
    params_jac = params_jac.at[:,4].set(jnp.log10(params_jac[:,4]))
    params_jac = params_jac.at[:,6].set(jnp.log10(params_jac[:,6]))
    print('params_jac.shape, params_jac',params_jac.shape, params_jac,flush=True)

    final_mvirs = halo_matrix[:,1,-1]
              
    
    #### return everything needed for plotting as a dict
    return {'tols_ode':tols_ode,
            'epsilons':epsilons,
            'full_params_lharr':full_params_lharr,
            'free_params_lharr':free_params_lharr,
            'full_params_lharr_wanted':full_params_lharr_wanted,
            'jac_autodiff_lh':jac_autodiff_lh,
            'jac_finitediff_lh':jac_finitediff_lh,
            'jac_err_adself':jac_err_adself,
            'jac_err_compare':jac_err_compare,
            'jac_err_adself_norm':jac_err_adself_norm,
            'jac_err_compare_norm':jac_err_compare_norm,
            'tols_ode2':tols_ode2,
            'concat_jac_autodiff':concat_jac_autodiff,
            'concat_jac_finitediff':concat_jac_finitediff,
            'ad8':ad8,
            'ad12':ad12,
            'ad_err':ad_err,
            'fd8':fd8,
            'fd8_err':fd8_err,
            'fd12':fd12,
            'fd12_err':fd12_err,
            'params_jac':params_jac,
            'final_mvirs':final_mvirs, 
            'config':config}
    

def hessians(config,halo_index,batch_solve8,batch_solve12,Nsamples):

    ###### import other sapphire modules we'll need
    from sapphire.utils import setup_parameters    
    import sapphire.summaries.gaussian_kernel_regression as gkr    
    from sapphire import inference
    
    
    ### unpack the two solvers w/ different ODE tolerance settings
    parallel_jac_autodiff8, parallel_jac_finitediff8, single_solve8 = batch_solve8
    parallel_jac_autodiff12, parallel_jac_finitediff12, single_solve12 = batch_solve12    

    ### generate Nsamples latin hypercube parameter sets
    sampling_config = config['sampling_config']    
    sampling_config['method'] = 'latin_hypercube'
    sampling_config['Nsamples'] = Nsamples   
    config['sampling_config'] = sampling_config

    full_params_arr, free_params_arr = setup_parameters.latin_hypercube_sampling(config)
    print('latin hypercube param shapes',full_params_arr.shape, free_params_arr.shape,flush=True)

    params_free = list(sampling_config['params_bounds'].keys())    

    ### initialize loss func for two sets of tol_ode
    loss_func8, grad_loss_func8, hess_loss_func8 = inference.explicit_likelihood.setup(config,halo_index,single_solve8)
    loss_func12, grad_loss_func12, hess_loss_func12 = inference.explicit_likelihood.setup(config,halo_index,single_solve12)    


    ### wrapper to get evaluate loss, grad, hessian and fisher with autodiff for two sets of tol_ode solvers
    ### this is tested here for a single parameter set 
    
    @partial(jit, static_argnums=(2,3,4))
    def get_autodiff(params,obs_stats,loss_func,grad_loss_func,hess_loss_func):
    
        loss = loss_func(params,obs_stats)
        grad_loss = grad_loss_func(params,obs_stats)
        hess_loss = hess_loss_func(params,obs_stats)
        hess_flag = jnp.all(jnp.linalg.eigvalsh(hess_loss)>0)
        Finv = jnp.linalg.inv(hess_loss)
    
        return loss, grad_loss, hess_loss, hess_flag, Finv

    """
    now set up finite-difference hessian function
    this uses a standard central finite-difference scheme
    e.g., section 5 of Numerical Recipes (Press 2007), section 8.1 of Nocedal 2007
    
    see also various Python and Julia libraries
    e.g., https://docs.sciml.ai/FiniteDiff/dev/hessians/
    """
    
    ### finite-difference epsilon 
    eps = 1e-4
    
    @partial(jit, static_argnums=2)
    def finite_diff_hessian(p,obs_stats,loss_func):
        N = p.shape[0] # how many free parameters
        f0 = loss_func(p,obs_stats)
        
        # Preallocate Hessian matrix
        H = jnp.zeros((N, N))
        
        def diag_entry(i):
            ei = jax.nn.one_hot(i, N, dtype=p.dtype)
            f_pos = loss_func(p + eps * ei,obs_stats)
            f_neg = loss_func(p - eps * ei,obs_stats)
            return (f_pos - 2.0 * f0 + f_neg) / (eps**2)
        
        def offdiag_entry(i, j):
            ei = jax.nn.one_hot(i, N, dtype=p.dtype)
            ej = jax.nn.one_hot(j, N, dtype=p.dtype)
            f_pp = loss_func(p + eps * ei + eps * ej,obs_stats)
            f_pm = loss_func(p + eps * ei - eps * ej,obs_stats)
            f_mp = loss_func(p - eps * ei + eps * ej,obs_stats)
            f_mm = loss_func(p - eps * ei - eps * ej,obs_stats)
            return (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
        
        # Diagonal entries
        H = H.at[jnp.diag_indices(N)].set(jax.vmap(diag_entry)(jnp.arange(N)))
        
        # Off-diagonal entries: use nested loops
        for i in range(N):
            for j in range(i+1, N):
                val = offdiag_entry(i, j)
                H = H.at[i, j].set(val)
                H = H.at[j, i].set(val)  # exploit symmetry
    
        return H


    ### now loop over and repeat for many mock parameter realizations, then plot average
    ##### NOTE: this could be shard_mapped or vmapped, but that code optimization is not necessary right now
    
    hessians_ad8, hessians_ad12 = [], []
    hessians_fd8, hessians_fd12 = [], []
    
    for pindex in range(Nsamples):
    
        print('-----> working on %s'%(pindex),flush=True)
    
        #### first need to re-solve ODEs for current params, compute mock obs_stats, and loss function
        tstart = timer()
        
        sol8 = single_solve8(halo_index,full_params_arr[pindex])
        obs_stats8 = gkr.summarize_mock(sol8)
        
        sol12 = single_solve12(halo_index,full_params_arr[pindex])
        obs_stats12 = gkr.summarize_mock(sol12)
    
        print('remade mock stats and loss func in %.3f sec'%(timer()-tstart),flush=True)
    
        tstart = timer()
        hessians_ad8.append(get_autodiff(free_params_arr[pindex],obs_stats8,loss_func8,grad_loss_func8,hess_loss_func8)[2].block_until_ready())
        print('ad8 finished in %.3f sec'%(timer()-tstart),flush=True)
    
        tstart = timer()
        hessians_ad12.append(get_autodiff(free_params_arr[pindex],obs_stats12,loss_func12,grad_loss_func12,hess_loss_func12)[2].block_until_ready())
        print('ad12 finished in %.3f sec'%(timer()-tstart),flush=True)
    
        tstart = timer()
        hessians_fd8.append(finite_diff_hessian(free_params_arr[pindex],obs_stats8,loss_func8).block_until_ready())
        print('fd8 finished in %.3f sec'%(timer()-tstart),flush=True)
    
        tstart = timer()
        hessians_fd12.append(finite_diff_hessian(free_params_arr[pindex],obs_stats12,loss_func12).block_until_ready())
        print('fd12 finished in %.3f sec'%(timer()-tstart),flush=True)
    
    print('finished all',flush=True)

    
    ### first stack relative arrays
    arr_hessians_ad8 = jnp.asarray(hessians_ad8)
    arr_hessians_ad12 = jnp.asarray(hessians_ad12)
    
    arr_hessians_fd8 = jnp.asarray(hessians_fd8)
    arr_hessians_fd12 = jnp.asarray(hessians_fd12)
    
    print('arr_hessians_ad8.shape, arr_hessians_ad12.shape, arr_hessians_fd8.shape, arr_hessians_fd12.shape',
          arr_hessians_ad8.shape, arr_hessians_ad12.shape, arr_hessians_fd8.shape, arr_hessians_fd12.shape,flush=True)

    ### now compute absolute value of relative symmetric error of these matrices (like fractional error)
    
    err_hessians_ad = jnp.abs((arr_hessians_ad8 - arr_hessians_ad12)/(0.5*(jnp.abs(arr_hessians_ad8)+jnp.abs(arr_hessians_ad12))))
    err_hessians_fd8 = jnp.abs((arr_hessians_fd8 - arr_hessians_ad8)/(0.5*(jnp.abs(arr_hessians_fd8)+jnp.abs(arr_hessians_ad8))))
    err_hessians_fd12 = jnp.abs((arr_hessians_fd12 - arr_hessians_ad12)/(0.5*(jnp.abs(arr_hessians_fd12)+jnp.abs(arr_hessians_ad12))))
    
    print('err_hessians_ad.shape, err_hessians_fd8.shape, err_hessians_fd12.shape',
          err_hessians_ad.shape, err_hessians_fd8.shape, err_hessians_fd12.shape,flush=True)

    ### compute median (across parameter realizations) of absolute value of relative symmetric error (like a fractional error) 
    
    median_ad_err = jnp.median(err_hessians_ad,axis=0)
    median_fd8_err = jnp.median(err_hessians_fd8,axis=0)
    median_fd12_err = jnp.median(err_hessians_fd12,axis=0)
    
    print('median_ad_err.shape, median_fd8_err.shape, median_fd12_err.shape',
          median_ad_err.shape, median_fd8_err.shape, median_fd12_err.shape,flush=True)

    ### also compute distribution of norms of these hessian relative symmetric error matrices (like jacobian figures 21-22)
    norm_ad_err = jnp.linalg.norm(err_hessians_ad,axis=(1,2))
    norm_fd8_err = jnp.linalg.norm(err_hessians_fd8,axis=(1,2))
    norm_fd12_err = jnp.linalg.norm(err_hessians_fd12,axis=(1,2))    


    ##### return everything needed for plotting
    return {'params_free':params_free,
            'full_params_arr':full_params_arr, 
            'free_params_arr':free_params_arr,
            'arr_hessians_ad8':arr_hessians_ad8, 
            'arr_hessians_ad12':arr_hessians_ad12,
            'arr_hessians_fd8':arr_hessians_fd8,
            'arr_hessians_fd12':arr_hessians_fd12,
            'err_hessians_ad':err_hessians_ad,
            'err_hessians_fd8':err_hessians_fd8,
            'err_hessians_fd12':err_hessians_fd12,
            'median_ad_err':median_ad_err,
            'median_fd8_err':median_fd8_err,
            'median_fd12_err':median_fd12_err,
            'norm_ad_err':norm_ad_err,
            'norm_fd8_err':norm_fd8_err,
            'norm_fd12_err':norm_fd12_err}
            
            
        




# 
    