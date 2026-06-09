"""
this standalone module can be run to re-write dark matter halo merger tree files into a sapphire-specific
jax-friendly array .npz format files for efficient vectorization/parallelization with vmap/shardmap.

as of june 2026 - the examples below are for Rockstar/consistent-trees (Behroozi+13a,b) outputs
and tied to the publicly available TNG trees. currently we use Britton Smith's ytree module.
ytree natively works with both tree_*.dat and isotree_*.dat (subhalos removed) files.
work is in progress (but low priority) to create a native jax-based tree reader template code. 

see example usage / simple tutorial in sapphire/demo/jaxify_trees.py

pull requests are encouraged to add support for additional simulations/codes.
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

""" 
a lot of stuff in here is pure python for simplicity/ease.
it is taken from viraj pandya's old ipynb without generalization for now.
however the final saved .npz files are designed to be jax-compatible.
"""
import numpy as np
import multiprocess
from glob import glob
import h5py 
import ytree
from tqdm import tqdm
from astropy.cosmology import Planck15, FlatLambdaCDM
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import InterpolatedUnivariateSpline as scipy_InterpolatedUnivariateSpline
from scipy.ndimage import gaussian_filter1d


def run(tree_path,prefix,subvolumes,min_root_logmvir,zs_full,min_snaps,out_fname,params_cosmo=None,num_readers=None):
    """
    this version is only for consistent-trees with ytree.
    will be refactored / extended to work with other tree formats and remove ytree dependence.
    
    tree_path = *absolute* path to the directory containing the tree files
    prefix = string denoting beginning of filenames, e.g., 'tree' for 'tree_0_0_0.dat' [TO DO: generalize with *]
    subvolumes = list of subvolume suffix strings e.g., ['0_0_0','0_0_1']
    min_root_logmvir = minimum root halo mass in log10(M/Msun) above which halos will be included
    zs_full = full list of snapshot redshifts based on simulation outputs 
    min_snaps = trees that span less than this number of snapshots/times will be discarded 
    out_fname = *absolute* path and filename for output .npz file
    params_cosmo = dict of cosmological parameters (follows sapphire's config.yaml), Planck15 if None
    num_readers = number of subvolumes to read in parallel using multiprocess module -- if None, min(# cpus, # subvolumes)

    TO DO: this needs to be refactored/expanded to account for subhalos, plus other "# TO DO" comments below
    """

    ### set up cosmology
    ### TO DO: offload to read_cosmology module
    ### TO DO: switch to Robinson, Pandya & Bryan custom jax functions
    if params_cosmo is None:
        print('since params_cosmo=None, using Planck15',flush=True)
        cosmo = Planck15
    else:
        # see Planck15 defaults defined in sapphire's config.yaml
        cosmo = FlatLambdaCDM(H0=params_cosmo['h0']*100, Om0=params_cosmo['Om0'],
                              Tcmb0=params_cosmo['Tcmb0'], Ob0=params_cosmo['Ob0'], Neff=params_cosmo['Neff'])

    ### convert zs_full to ts_full -- cosmic age in Gyr from low to high
    ts_full = cosmo.age(zs_full).value[::-1] 

    ### legacy purposes - compute zmax 
    # TO DO: this should be input by user to filter subset of trees as desired (e.g., zmin-zmax, logmvir range, environment)
    zmax = np.max(zs_full) 
    
    ### define constants and conversions 
    const_mp = const.m_p.to('Msun').value 
    cm_to_kpc = u.cm.to('kpc') # multiply something in cm by this to get to kpc 
    kpc_to_cm = u.kpc.to('cm') # multiply something in kpc, it becomes units of cm
    const_kB = const.k_B.to('erg/K').value # so that k*T = erg by default
    yr_to_s = u.yr.to('s') # if you multiply something in yr by this, you get it in units of sec
    s_to_yr = u.s.to('yr')
    Msun_to_g = u.Msun.to('g') 
    f_b = cosmo.Ob0 / cosmo.Om0 # ~15 %
    littleh = cosmo.h
    Om0 = cosmo.Om0
    G = const.G.to('cm**3 / (g * s**2)').value
    t0 = cosmo.age(0).value    
    
    ### first define a function that reads a SINGLE subvolume's tree file with ytree
    # below this will be called in parallel for all input "subvolumes" using multiprocess module
    def read_single(subvolume):

        tstart = timer()

        # use ytree to read in this file 
        a = ytree.load(os.path.join(tree_path,'%s_%s.dat'%(prefix,subvolume))) 
    
        # Storage lists
        halo_matrix_list = []
        halo_tinit_list = []
        halo_rootid_list = []
        halo_subvolume_list = [] # normal np.array of strings
        
        for t in a:
            # ignore this tree if its root halo mass is below our threshold
            if (t['Mvir']/cosmo.h < 10**min_root_logmvir):
                continue
    
            # do not include if the halo is included in < half TNG output snapshots -- these appear to be artifact halos losing mass that are found only at low-z
            if len(t['prog','redshift']) < min_snaps: # can choose any time series column...
                continue
            
            # get rootid for 1-1 mapped halo_rootid 
            rootid = t['Tree_root_ID']
    
            # get time series of quantities we need -- reversed so cosmic age increases from low to high (high-z to low-z)
            zarr = np.array(t['prog','redshift'][::-1]) # dimensionless, from high-z to low-z
    
            # July 10 -- to ensure we stick to ts_full range 
            # TO DO: this is leftover from legacy code, not sure whether it's still necessary
            indz = np.where(zarr<=zmax)[0]
            zarr = zarr[indz]
            
            cosmic_age = cosmo.age(zarr).value # Gyr   
    
            # apply same indz filtering to these (making sure reverse is done before)
            mvir = np.array(t['prog','Mvir'],dtype=np.float64)[::-1][indz] / cosmo.h # Msun
            rvir = np.array((t['prog','scale'] * t['prog','Rvir']),dtype=np.float64)[::-1][indz] / cosmo.h # proper kpc
            cNFW = np.array((t['prog','Rvir'] / t['prog','Rs_Klypin']))[::-1][indz] # dimensionless 
            vvir = np.sqrt(G * mvir * Msun_to_g / (rvir * kpc_to_cm)) * 1e-5 # proper km/s
            
            # use finite differencing of Mvir time series to compute net halo DM accretion rate 
            # NOTE: this pads the differenced arrays to be same length as original arrays, then clips rates to some small number > 0 Msun/yr so we can take log10 during interpolation
            # TO DO: change this to the interpolator-based derivative that we started using for Gabrielpillai, Pandya+ 
            mar = np.clip(np.append(np.diff(mvir) / np.diff(cosmic_age * 1e9), 0), 1e-10, None) # Msun/yr
    
            # smooth + interpolate each quantity onto common full time grid
            # def interp_log_feat(x):  # smoothing + log10 + spline to ts_full
            #     return scipy_InterpolatedUnivariateSpline(cosmic_age, gaussian_filter1d(np.log10(x), 10), k=3, ext='zeros')(ts_full)
    
            # note that for MAR i did log10 outside gaussian_filter, for everything else do gaussian_filter OF log10 quantity
            # NOTE: ts_full is defined ahead of time below using one snapshot that must have at least 1 halo with all 100 TNG snapshot times
            # TO DO: introduce input parameter for smoothing that depends on time or redshift instead of snapshot #
            # TO DO: create new function above that returns interpolations, instead of here for readability purposes 
            # TO DO: add more features as desired, and expand to new dim for subhalos 
            features = np.vstack([scipy_InterpolatedUnivariateSpline(cosmic_age, np.log10(gaussian_filter1d(mar, 10)), k=3, ext='zeros')(ts_full),
                                  scipy_InterpolatedUnivariateSpline(cosmic_age, gaussian_filter1d(np.log10(mvir), 10), k=3, ext='zeros')(ts_full),
                                  scipy_InterpolatedUnivariateSpline(cosmic_age, gaussian_filter1d(np.log10(rvir), 10), k=3, ext='zeros')(ts_full),
                                  scipy_InterpolatedUnivariateSpline(cosmic_age, gaussian_filter1d(np.log10(vvir), 10), k=3, ext='zeros')(ts_full),
                                  scipy_InterpolatedUnivariateSpline(cosmic_age, gaussian_filter1d(np.log10(cNFW), 10), k=3, ext='zeros')(ts_full)
                                 ])  # shape (5, 100)
    
            ### finally append this halo's arrays to global list 
            halo_matrix_list.append(features)
            halo_tinit_list.append(cosmic_age[0])
            halo_rootid_list.append(rootid)
            halo_subvolume_list.append(subvolume)
        
        ### finally turn into numpy Arrays (we'll convert to jax arrays after processing halos from all subvolumes in this batch)
        halo_tinit = np.asarray(halo_tinit_list) # shape (Nhalos,)
        halo_rootid = np.asarray(halo_rootid_list) # shape (Nhalos,)
        halo_matrix = np.asarray(halo_matrix_list) # shape (Nhalos, 5, 100)
        halo_subvolume = np.asarray(halo_subvolume_list) # shape (Nhalos,) -- normal np array of strings
        
        print('finished subvolume %s in %.2f sec'%(subvolume,timer()-tstart),flush=True)
    
        return halo_tinit, halo_matrix, halo_rootid, halo_subvolume        
        


    # set up multiprocessing for reading trees in parallel 
    ### TO DO: parallelize over total # of trees instead of subvolume files? though we are IO bound 
    if num_readers is None:
        num_readers = np.min([len(subvolumes),multiprocess.cpu_count()])
        print('setting num_readers from None to %s'%num_readers,flush=True)
    
    tstart0 = timer()
    with multiprocess.Pool(processes=num_readers) as pool:
    
        out = pool.map(read_single, subvolumes)
    
    ### stack into np arrays with leading dimension (Nhalos,)  -- can easily do jnp.asarray when reading in for sapphire
    
    halo_tinit = np.concatenate([out[i][0] for i in range(len(out))])
    halo_matrix = np.concatenate([out[i][1] for i in range(len(out))])
    halo_rootid = np.concatenate([out[i][2] for i in range(len(out))])
    halo_subvolume = np.concatenate([out[i][3] for i in range(len(out))]) # normal np array

    print('finished all in %.2f sec'%(timer()-tstart0),flush=True)

    # name this file with subvolume's leading # X of X_Y_Z taken from any str
    np.savez(out_fname, 
             halo_tinit=halo_tinit,
             halo_matrix=halo_matrix,
             halo_rootid=halo_rootid,
             halo_subvolume=halo_subvolume,
             ts_full=ts_full)

    print('saved %s'%out_fname,flush=True)



    
# 