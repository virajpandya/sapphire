"""
This module reads in one or multiple subvolume tree files for the TNG50/100/300 
simulation using ytree. 

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

kpc_to_cm = u.kpc.to('cm') # multiply something in kpc, it becomes units of cm
Msun_to_g = u.Msun.to('g')
G = const.G.to('cm**3 / (g * s**2)').value

# NOTE: add an option here (with default) for list of strings of subvolumes to read 
def read_trees(tree_path, min_root_mass,halo_names=[]):
    """
    Reads in the consistent-trees isotree file for a single subvolume using ytree
    Imposes a cut on minimum root (z=0) halo mass.
    NOTE: since this reads isotrees, no cut on central vs. subhalo classification is required.
    
    halo_names = irrelevant for tng trees (it was for reading in specific fire halos)
    isotree_path = *absolute* path to isotree file 
    min_root_mass = minimum root (z=0) halo mass in units of Msun (no little-h)
    """

    # use ytree to read in this file 
    a = ytree.load(tree_path)
    
    # create list of root halo IDs for all halos satisfying min root mass cut 
    trees_rootID = [t['Tree_root_ID'] for t in a if t['Mvir']/Planck15.h > min_root_mass] 
    
    # now create list of lists with required progenitor property time series for each halo
    # NOTE: this is only for the most massive progenitor lineage ordered from high-z to low-z
    trees_haloID = [t['prog','id'][::-1] for t in a if t['Mvir']/Planck15.h > min_root_mass] # unique halo ID across whole simulation
    trees_redshift = [t['prog','redshift'][::-1] for t in a if t['Mvir']/Planck15.h > min_root_mass] # dimensionless
    trees_Mvir = [np.array(t['prog','Mvir'])[::-1]/Planck15.h for t in a if t['Mvir']/Planck15.h > min_root_mass] # Msun
    trees_Rvir = [np.array(t['prog','scale']*t['prog','Rvir'])[::-1]/Planck15.h for t in a if t['Mvir']/Planck15.h > min_root_mass] # proper kpc
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
        
        tables_tng[rootID] = Table([trees_haloID[rootnum],
                                   np.full_like(trees_haloID[rootnum],rootID),
                                   trees_redshift[rootnum],
                                   trees_age[rootnum],
                                   np.log10(trees_Mvir[rootnum]),
                                   np.log10(trees_Rvir[rootnum]),
                                   np.log10(trees_cNFW[rootnum]),
                                   np.log10(trees_Vvir[rootnum]),
                                   trees_MAH[rootnum]],
                                   names=('haloID','root_haloID','redshift','cosmic_age','log_mvir','log_rvir','log_cNFW','log_vvir','dm_accretion_rate',))
        

    return tables_tng


    
