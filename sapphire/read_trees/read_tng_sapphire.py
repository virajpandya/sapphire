"""
This modules reads in an npz file of TNG trees that were compactified and written 
by a previous run of sapphire. 

NOTE:   this currently assumes that there is only a single npz file to read in. That can be easily updated later. 
        But in general, the compactified npz format is so efficient that we can store 10K~100K trees in a single <1 GB file.
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
    
def read_trees(parameters):
    """
    Reads in the npz file created by sapphire in a previous run which includes a dict of all TNG trees 
    from 1 or more subvolumes as astropy tables.
    
    input is the user-provided parameters dict which contains the filepath and the subvolumes we want to read in
    
    Returns a dict of Astropy Table objects with keys being the root (z=0) halo IDs (halos from all subvolumes are grouped into same dict)
    """
    
    # parse the parameters dict for parameters we need 
    tree_path = parameters['tree_path'] # *absolute* path to directory containing isotree_$subvolume.dat files 
    
    # read in the single npz file 
    npz = np.load(tree_path,allow_pickle=True)
    tables_tng = npz['arr_0'].item() # this is a dict of astropy tables indexed by root halo ID 
    npz.close()
    
    # if subvolumes != None, use dict comprehension to keep only the trees requested 
    # WARNING: there is no error check here for whether list of subvolumes makes sense ... should be list of strings like ['0_0_0','0_0_1'] 
    if parameters['subvolumes'] != None: 
        tables_tng = {key:value for key,value in tables_tng.items() if value['subvolume'][0] in parameters['subvolumes']}
    
    return tables_tng


    
