"""
This modules reads in an npz file of TNG trees that were compactified and written 
by the sapphire/read_trees/jaxify_trees.py 

"""

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 
from functools import partial 
from timeit import default_timer as timer
import os

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
from jax.sharding import Mesh, PartitionSpec, NamedSharding


def get(config):
    
    """ July 10 -- switch to my new batched subvolume library for TNG100 (125 subvolumes) """
    
    list_halo_tinit = []
    list_halo_matrix = []
    list_halo_rootid = []
    list_halo_subvolume = []
    list_ts_full = [] # just to verify they're all same 
    
    tstart0 = timer()
    
    # Nbatches = 6 for TNG50, 5 for TNG100, 7 for TNG300 -- can also glob
    for ibatch in range(5):
        tstart = timer()
        # *absolute* path to directory containing jaxified tree files 
        # NOTE: should generalize this for different tree_name subdirs using '%s' wildcard as in visualization module
        tree_path = os.path.join(config['data_path'],'trees/%s/tng100_subvolumes%s_jax.npz'%(config['tree_type'],ibatch)) 
        npz = jnp.load(tree_path,allow_pickle=True)
        
        list_halo_tinit.append(npz['halo_tinit'])
        list_halo_matrix.append(npz['halo_matrix'])
        list_halo_rootid.append(npz['halo_rootid'])
        list_halo_subvolume.append(npz['halo_subvolume'])
    
        list_ts_full.append(npz['ts_full'])
        
        npz.close()
        print('finished reading %s in %.2f sec'%(tree_path,timer()-tstart),flush=True)

    print('finished reading all batches in %.2f sec'%(timer()-tstart0),flush=True)
    
    # so just pick a single ts_full since all identical for a given box
    # I verified that this works 
    ts_full = list_ts_full[0]
    
    # and it becomes ts_interp following above
    ts_interp = jnp.asarray(ts_full)
    
    # print('ts_interp',ts_interp.shape, ts_interp,flush=True)
    
    # size of each
    tot_sum = 0
    for i in range(len(list_halo_tinit)):
        this_sum = len(list_halo_tinit[i])
        tot_sum += this_sum
        print(i,this_sum,flush=True)

    print('total # trees',tot_sum,flush=True)

    ### merge all lists along leading Nhalos axis into jax jnp.arrays
    
    halo_tinit = jnp.concatenate(list_halo_tinit)
    halo_matrix = jnp.concatenate(list_halo_matrix)
    halo_rootid = jnp.concatenate(list_halo_rootid)
    # halo_subvolume = np.concatenate(list_halo_subvolume) # needs to be normal np.array since its strings
    
    # print(halo_tinit.shape,halo_matrix.shape,halo_rootid.shape,halo_subvolume.shape,flush=True)   
    # print(halo_tinit.shape,halo_matrix.shape,halo_rootid.shape,flush=True)   


    ### jax-specific function to store Cubic Hermite coeff matrix that can be vmapped over
    
    @jit
    @partial(vmap,in_axes=(0,))
    def interp_single(halo_matrix):
        """
        input is one halo's interpolated matrix of shape (N_inputs,len(ts_interp))
        this function will vmap over the 1st axis to create Cubic Hermite coeffs for each of the N time series for N inputs
        
        the return should be of shape (N_inputs, 4, len(ts_interp)-1) where 4 is the # of coeffs needed for Cubic Hermite
        and len(ts_interp)-1 because thstee spline is characterized by coeffs at # knots minus 1
        """
        
        # here, halo_matrix is a single row = single input's time series
        return jnp.asarray(backward_hermite_coefficients(ts_interp, halo_matrix)) 
        
        
    
    @jit
    @partial(vmap,in_axes=(0,))
    def interp_coeffs(halo_matrix):
        """
        input is the halo matrix for all halos of shape (Nhalos, N_inputs, len(ts_interp))
        this will vmap over the first axis (each halo) and then call the nested vmap interp_single() 
        to return Cubic Hermite coeff matrix of shape  (Nhalos, N_inputs, 4, len(ts_interp)-1) where 4 is the # coeffs needed for cubic hermite
        """
        
        # this here will return Cubic Hermite coeff matrix for a single halo (single row of halo_matrix)
        return interp_single(halo_matrix)
    
    
    @jit
    @partial(vmap,in_axes=(0,None,None))
    def assign_probs(filt_halo_matrix,full_filt_halo_matrix,sigma):
        """
        filt_halo_matrix input twice since vmap will be over first axis, 
        but we'll want to filter the rest of the matrix to compute # halos
        with z=0 logmvir within +/- sigma dex of each halo for probability
        
        return value should be of shape (Nhalos,) where each entry is # halos with similar mass
        """
        
        # three-arg version of jnp.where: return 1 if condition true to count that halo, 0 otherwise
        # then we sum over to count # of halos
        
        return jnp.sum(jnp.where((full_filt_halo_matrix[:,1,-1]>filt_halo_matrix[1,-1]-sigma) & (full_filt_halo_matrix[:,1,-1]<filt_halo_matrix[1,-1]+sigma),1,0))
        
    
    # compute cubic hermite interp coeffs 
    tstart = timer()
    coeff_matrix = interp_coeffs(halo_matrix)
    # print('finished in %.2f sec'%(timer()-tstart),flush=True)    
    
    # tstart = timer()
    # coeff_matrix = interp_coeffs(halo_matrix)
    # print('finished in %.2f sec'%(timer()-tstart),flush=True)    
    
    print('coeff_matrix.shape',coeff_matrix.shape,flush=True)

    ### suppose instead I just wanted a random subset of all halos with z=0 logmvir~10-12
    
    # first filter to mass range
    inds_filt = jnp.where((halo_matrix[:,1,-1]>10) & (halo_matrix[:,1,-1]<12.3))[0]
    filt_halo_matrix = halo_matrix[inds_filt]
    filt_coeff_matrix = coeff_matrix[inds_filt]
    filt_halo_tinit = halo_tinit[inds_filt]
    print('coeff_matrix.shape after halo mass cuts',filt_coeff_matrix.shape,flush=True)
    
    ### finally, the above reflects the halo mass function, but what if we wanted to sample UNIFORMLY in z=0 logmvir~10-12? 
    
    # one way is to create a probability to pass to jnp.random.choice 
    # the probability should be lower for lower-mass halos
    # i.e., probability of drawing each halo should be inversely proportional to # of halos with similar mass

    # initial compile
    tstart = timer()    
    draw_probs = assign_probs(filt_halo_matrix,filt_halo_matrix,0.05)
    print('assigning random draw probs took %.2f sec'%(timer()-tstart),flush=True)  
    
    # actual run
    # tstart = timer()    
    # draw_probs = assign_probs(filt_halo_matrix,filt_halo_matrix,0.05)
    # print('finished in %.2f sec'%(timer()-tstart),flush=True)  
    
    # print(jnp.min(draw_probs),flush=True) # want to make sure we don't divide by 0 if prob ~ 1/N

    """ 
    July 10 -- increase this now to 10K for full uniform rand batch 
    from this we'll generate Ndevice-compatabiel batch and minibatch below 
    """ 
    
    # then choose random subset of these
    # NOTE: user should take care to compare their Nbatch to the total # of trees -- as Nbatch-->Ntot, get HMF/poisson sampling effects
    inds_rand = jax.random.choice(key(config['rng_halos']),jnp.arange(len(filt_halo_matrix)),
                                  shape=(config['Nbatch'],),replace=False,p=1/draw_probs)
    
    # finally rand_matrix
    rand_halo_matrix = filt_halo_matrix[inds_rand]
    rand_coeff_matrix = filt_coeff_matrix[inds_rand]
    rand_halo_tinit = filt_halo_tinit[inds_rand]
    print('rand_coeff_matrix.shape',rand_coeff_matrix.shape,flush=True)

    rand_halo_index = jnp.arange(len(rand_halo_tinit))


    # return the downsampled matrices
    return rand_halo_matrix, rand_coeff_matrix, rand_halo_tinit, rand_halo_index, ts_interp


#


