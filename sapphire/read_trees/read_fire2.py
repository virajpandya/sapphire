"""
This module reads the raw consistent-trees merger trees of the core FIRE-2 halos
created by Viraj for Pandya+20. These files are each huge (~couple GB) compared to the
stripped down Pandya+22 files used in read_fire2_pandya22.py.

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

all_halos = ['m10q','m10y','m10z','m11a','m11b','m11c','m11q','m11v','m11f','m12i','m12f','m12m']

def get_main_tree(trees):
    """
    This function returns the tree of the main halo in the zoom simulation by choosing 
    the halo that has the lowest contamination from low-res DM particles.
    
    Viraj has manually/visually sanity checked this in the past.
    for m10y, m10z, m11c, m11v, m12m: the 1st index tree is the correct one
    for the other halos, the 0th index tree is the correct one
    """
    
    # store the z=0 Mvir and hires/(lowres+hires) mass fraction for the first 10 trees (our main halo is)
    z0_Mvir = np.array([])
    z0_Mfrac = np.array([])
    
    for treenum in np.arange(10):

        tree = trees[treenum]

        Mvir = tree['Mvir'] # z=0 Mvir 
        logMhires = np.log10(tree['Mvir_hires']) # z=0 Mvir counting only high-res DM particles
        logMlowres = np.log10(tree['Mvir_lowres']) # z=0 Mvir counting only low-res DM particles
        Mfrac = 10**logMhires / (10**logMlowres + 10**logMhires) # Fraction of total Mvir from high-res DM particles only
        
        z0_Mvir = np.append(z0_Mvir,Mvir)
        z0_Mfrac = np.append(z0_Mfrac,Mfrac)
        
    # find index that maximizes product Mvir*Mfrac (i.e., find the halo with mostly high-res DM making up its mass)
    ind_main = np.where(z0_Mvir * z0_Mfrac == np.max(z0_Mvir * z0_Mfrac))[0][0] 
    
    # return that index tree
    return trees[ind_main]

def read_single_halo(halo_name,tree_path):
    """
    This reads in and returns data in required format for a single halo's tree
    
    halo_name is a string (one of the names in the global all_halos list above)
    tree_path is the *absolute* path to the directory containing the tree_$halo_name.dat files 
    It is assumed each file is named tree_path+'tree_$halo_name.dat'
    """
    
    print('Reading in tree for %s'%halo_name,flush=True)
    
    # use ytree to read in this file 
    trees = ytree.load(os.path.join(tree_path,'tree_%s.dat'%halo_name)) 
    
    # automatically identify and retrieve tree of the main FIRE-2 zoom halo 
    t = get_main_tree(trees)
    
    # retrieve lists of redshift, Mvir, Rvir, cNFW from MMP history ordered from high-z to low-z
    tree_redshift = t['prog','redshift'][::-1] # dimensionless
    tree_Mvir = np.array(t['prog','Mvir'])[::-1]/Planck15.h # Msun
    tree_Rvir = np.array(t['prog','scale']*t['prog','Rvir'])[::-1]/Planck15.h # proper kpc
    tree_cNFW = np.array(t['prog','Rvir']/t['prog','Rs_Klypin'])[::-1] # dimensionless
    
    # convert redshift to cosmic age (time since BB) in Gyr for the ODE solver 
    tree_age = Planck15.age(tree_redshift).value
    
    # compute Vvir from sqrt(GMvir/Rvir) in proper km/s
    tree_Vvir = np.sqrt(G*tree_Mvir*Msun_to_g / (tree_Rvir*kpc_to_cm))*1e-5 
    
    # use finite differencing of Mvir time series to compute net halo DM accretion rate 
    # NOTE: this pads the differenced arrays to be same length as original arrays, then clips rates to some small number > 0 Msun/yr so we can take log10 during interpolation
    tree_MAH = np.clip(np.append(np.diff(tree_Mvir) / np.diff(tree_age*1e9), 0),1e-10,None)
    
    # construct astropy table with above needed columns
    # NOTE: the column names MUST be what is expected from the other sapphire modules and log10 must be taken where expected 
    tfire = Table([tree_redshift,
                   tree_age,
                   np.log10(tree_Mvir),
                   np.log10(tree_Rvir),
                   np.log10(tree_cNFW),
                   np.log10(tree_Vvir),
                   tree_MAH],
                  names=('redshift','cosmic_age','log_mvir','log_rvir','log_cNFW','log_vvir','dm_accretion_rate',))
    
    # return dict with halo_name : tfire pair for this single halo
    return {halo_name:tfire}
    

def read_trees(parameters):
    
    """
    This reads in the FIRE-2 consistent-trees files for all or a subset of halo names in all_halos
    
    By default multiprocessing is enabled but the user can do num_readers=1 to effectively read
    each tree in serially. For the fastest read time, set num_readers = len(halo_names) so that 
    num_readers readers will be spawned and each will read in a single tree. This assumes there is 
    enough memory available -- for reference, the tree files of all 12 core halos together are ~50 GB.
        
    input is the user-provided parameters dict which contains the parameters we need 
    
    Returns a dict of Astropy Table objects with keys being the halo names
    """
    
    # parse the parameters dict for parameters we need 
    tree_path = parameters['tree_path'] # *absolute* path to directory containing the tree_$halo_name.dat files 
    halo_names = parameters['halo_names'] # list of halo names to read, if [] or None, process all 12 core FIRE-2 halos
    num_readers = parameters['num_readers'] # number of tree files to read in parallel
    
    # if halo_names is empty or None, do all halos
    if halo_names == None or len(halo_names) == 0: 
        halo_names = all_halos.copy()    
    
    # raise error if halo_names contains strings not in the global all_halos list above
    if all(halo_name in all_halos for halo_name in halo_names) == False:
        raise ValueError('one or more halo names in not in %s'%all_halos) 
    
    # read trees in parallel (or in serial if num_readers = 1)
    # initialize multiprocessing pool of workers
    pool = multiprocessing.Pool(processes=num_readers) 

    # create a functools.partial object where tree_path is fixed and only halo_name will vary between pool map calls
    pool_reader = partial(read_single_halo,tree_path=tree_path)
    
    # map the read_single_halo function to each pair of (halo_name,tree_path) in parallel
    out = pool.map(pool_reader,halo_names)

    # close pool processes to free memory
    pool.close()
    pool.join()            
    
    # use dict comprehension to merge all the individual halos' name:table pairs together into a single dict
    tables_fire = {key:value for d in out for key,value in d.items()}
    
    return tables_fire

