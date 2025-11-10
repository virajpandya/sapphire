"""
This module reads in one or multiple subvolume tree files for the TNG50/100/300 
simulation using ytree and (optional) parallelization. 

This script should also compute any additional quantities needed by sapphire 
such as Vvir=sqrt(GMvir/Rvir) if not already available in the tree file.
"""

import numpy as np
from astropy.table import Table
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 
import os 
import ytree 
import multiprocessing 
from functools import partial

kpc_to_cm = u.kpc.to('cm') # multiply something in kpc, it becomes units of cm
Msun_to_g = u.Msun.to('g')
G = const.G.to('cm**3 / (g * s**2)').value

def read_single_subvolume(subvolume,tree_path,min_root_mass):
    """
    This reads in and returns data in required format for a single TNG subvolume.
    
    subvolume is a single string denoting the subvolume suffix of the isotree file, e.g., '0_0_0'
    tree_path is the *absolute* path to the directory containing the isotree_$subvolume.dat files
    min_root_mass is the minimum root halo mass in Msun below above which halos will be included
    
    Generally you want min_root_mass ~ 1000 * DM resolution so it has a sufficient number of (lower mass) progenitors
    """
    
    print('Reading in trees for subvolume=%s'%subvolume,flush=True)
    
    # use ytree to read in this file 
    a = ytree.load(os.path.join(tree_path,'isotree_%s.dat'%subvolume)) 
    
    # create list of root halo IDs for all halos satisfying min root mass cut 
    trees_rootID = [t['Tree_root_ID'] for t in a if t['Mvir']/Planck15.h > min_root_mass] 
    
    # construct a column where all values are the subvolume string that this halo belongs to
    trees_subvolume = np.full(len(trees_rootID),subvolume)    
    
    # now create list of lists with required progenitor property time series for each halo
    # NOTE: this is only for the most massive progenitor lineage ordered from high-z to low-z
    # NOTE: Mvir and Rvir are read as float32 by default but must be float64 otherwise Vvir can become inf due to overflow [ytree bug?]
    trees_haloID = [t['prog','id'][::-1] for t in a if t['Mvir']/Planck15.h > min_root_mass] # unique halo ID across whole simulation
    trees_redshift = [t['prog','redshift'][::-1] for t in a if t['Mvir']/Planck15.h > min_root_mass] # dimensionless
    trees_Mvir = [np.array(t['prog','Mvir'],dtype=np.float64)[::-1]/Planck15.h for t in a if t['Mvir']/Planck15.h > min_root_mass] # Msun
    trees_Rvir = [np.array(t['prog','scale']*t['prog','Rvir'],dtype=np.float64)[::-1]/Planck15.h for t in a if t['Mvir']/Planck15.h > min_root_mass] # proper kpc
    trees_cNFW = [np.array(t['prog','Rvir']/t['prog','Rs_Klypin'])[::-1] for t in a if t['Mvir']/Planck15.h > min_root_mass] # dimensionless
    
    # convert redshift to cosmic age (time since BB) in Gyr for the ODE solver 
    trees_age = [Planck15.age(t_z).value for t_z in trees_redshift]
    
    # compute Vvir from sqrt(GMvir/Rvir) in proper km/s
    trees_Vvir = [np.sqrt(G*t_Mvir*Msun_to_g / (t_Rvir*kpc_to_cm))*1e-5 for (t_Mvir,t_Rvir) in zip(trees_Mvir,trees_Rvir)]   
    
    # use finite differencing of Mvir time series to compute net halo DM accretion rate 
    # NOTE: this pads the differenced arrays to be same length as original arrays, then clips rates to some small number > 0 Msun/yr so we can take log10 during interpolation
    trees_MAH = [np.clip(np.append(np.diff(t_Mvir) / np.diff(t_Time*1e9), 0),1e-10,None) for (t_Mvir,t_Time) in zip(trees_Mvir,trees_age)]
    
    # loop over every root halo ID, construct its minimal astropy table, and add it to a dict indexed by the root halo ID
    # NOTE: the column names MUST be what is expected from the other sapphire modules and log10 must be taken where expected 
    tables_tng = {} 
    
    for rootnum,rootID in enumerate(trees_rootID): 
        
        # do not include if the halo is included in < half TNG output snapshots -- these appear to be artifact halos losing mass that are found only at low-z
        if len(trees_haloID[rootnum]) < 50: 
            continue
        
        tables_tng[rootID] = Table([trees_haloID[rootnum],
                                   np.full(len(trees_haloID[rootnum]),rootID),
                                   np.full(len(trees_haloID[rootnum]),trees_subvolume[rootnum]),
                                   trees_redshift[rootnum],
                                   trees_age[rootnum],
                                   np.log10(trees_Mvir[rootnum]),
                                   np.log10(trees_Rvir[rootnum]),
                                   np.log10(trees_cNFW[rootnum]),
                                   np.log10(trees_Vvir[rootnum]),
                                   trees_MAH[rootnum]],
                                   names=('haloID','root_haloID','subvolume','redshift','cosmic_age','log_mvir','log_rvir','log_cNFW','log_vvir','dm_accretion_rate',))
        
    return tables_tng 

    
    
def read_trees(parameters):
    """
    Reads in the consistent-trees isotree file for a single subvolume using ytree
    Imposes a cut on minimum root (z=0) halo mass.
    NOTE:   This currently only reads in trees of halos that are still classified as centrals above min_root_mass limit at z=0.
            This does not currently allow for reading in halos that were centrals down to some other redshift, and then became
            subhalos (aside: subhalos are not recorded in the isotrees we read). If we wanted to include centrals at higher
            redshifts then we would need to generalize the way we are using ytree to define "root" halos and their MMP histories.
            
    By default multiprocessing is enabled but the user can do num_readers=1 to effectively read the trees serially (one after the other)
    For reference: the first 6 subvolumes (0_0_x) are ~15 GB, take 10 min to read in parallel but 40 min to read serially (on rusty)

    input is the user-provided parameters dict which contains the parameters we need 
    
    Returns a dict of Astropy Table objects with keys being the root (z=0) halo IDs (halos from all subvolumes are grouped into same dict)
    """
    
    # parse the parameters dict for parameters we need 
    tree_path = parameters['tree_path'] # *absolute* path to directory containing isotree_$subvolume.dat files 
    subvolumes = parameters['subvolumes'] # list of isotree suffixes for individual subvolumes, e.g., ['0_0_0','0_1_0','2_3_4']
    min_root_mass = parameters['min_root_mass'] # minimum root (z=0) halo mass in units of Msun (no little-h)
    num_readers = parameters['num_readers'] # number of tree files (subvolumes) to read in parallel

    # set up multiprocessing for reading trees in parallel or serial (latter if num_readers=1)
    pool = multiprocessing.Pool(processes=num_readers) 

    # create a functools.partial object where tree_path and min_root_mass are fixed and only subvolume varies between pool map calls
    pool_reader = partial(read_single_subvolume,tree_path=tree_path,min_root_mass=min_root_mass)
    
    # map the read_single_halo function to each pair of (halo_name,tree_path) in parallel
    out = pool.map(pool_reader,subvolumes)

    # close pool processes to free memory
    pool.close()
    pool.join()            
    
    # use dict comprehension to merge the ID:table pairs of all halos from all subvolumes into a single huge dictionary
    tables_tng = {key:value for d in out for key,value in d.items()}        

    return tables_tng


    
