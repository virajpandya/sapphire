"""
This module reads in the FIRE-2 merger tree for one or multiple core halos
from the custom ASCII file format created by Viraj to store both dark matter
and baryonic properties vs. time for the most massive progenitor halo. This 
custom file format is much smaller in size (~1 MB) vs the full consistent-trees
output (several GB) and includes the gross DM accretion rate at Rvir from 
particle tracking fluxes.

This script should also filter out halos below requested Mvir resolution limit, bad snapshots, etc.

This script should also compute any additional quantities needed by sapphire 
such as Vvir=sqrt(GMvir/Rvir) if not already available in the tree file (with the correct names for sapphire).
"""

import numpy as np
from astropy.table import Table
import os 

all_halos = ['m10q','m10y','m10z','m11a','m11b','m11c','m11q','m11v','m11f','m12i','m12f','m12m']
    
def read_trees(tree_path, min_root_mass=None, halo_names=all_halos):
    """
    Reads Viraj's custom merger tree ASCII file for the core FIRE-2 halos created for Pandya+22
    
    tree_path is the *absolute* path to the directory containing Viraj's files    
    min_root_mass is irrelevant for fire2 sapphire trees (its for large volume simulations)
    halo_names is a list of strings that can be a subset of all_halos above 
    
    Returns a dict of Astropy Table objects with keys being the halo names
    """
    
    # if halo_names is empty, do all halos
    if halo_names == None or len(halo_names) == 0: # process all 12 core FIRE-2 halos
        halo_names = all_halos.copy()    
    
    # raise error if halo_names contains strings not in the global all_halos list above
    if all(halo_name in all_halos for halo_name in halo_names) == False:
        raise ValueError('one or more halo names in not in %s'%all_halos) 
    
    ### in a for loop, call read_halo() for each individual halo_name
    tables_fire = {}
    
    for halo_name in halo_names:
        # using os.path.join to avoid ambiguity with slashes (note: filename itself should NOT start with slash)
        t = Table.read(os.path.join(tree_path,'pandya22_%s.dat'%halo_name),format='ascii')
        
        # create any new columns needed for interpolate_trees.py (using the column names that module expects)
        t['log_mvir'] = t['logmvir'] # log10 Mvir in Msun (changing to universal column name)
        t['log_rvir'] = np.log10(10**t['logrvir']*t['scale']) # log10 Rvir in proper kpc [NOTE: new column name has underscore in it]
        t['log_vvir'] = np.log10(t['Vvir_DM']) # log10 Vvir = sqrt(GMvir/Rvir) in k/s 
        t['log_cNFW'] = np.log10(t['halo_conc']) # log10 of halo NFW concentration = Rvir/Rs_klypin
        t['dm_accretion_rate'] = t['mdot_in_halo_dm_tracked'] # halo mass accretion rate in Msun/yr [NOTE: not log10 since this can be <= 0 if net rate]
        
        # append this halo's tree Table to combined list of all trees
        tables_fire[halo_name] = t
        
    return tables_fire 


