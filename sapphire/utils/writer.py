"""
This module includes writer functions that will be loaded based on the types of trees being run on.
For large-volume simulations with multiple indivdual subvolumes, output files will be split with a
subvolume at the end of the filename. For small numbers of halos from zoom simulations, we can save 
all outputs into a single output file.
"""

import numpy as np 
import os
from astropy.table import Table
import multiprocessing
from functools import partial 
import pandas as pd

def write_subvolume(subvolume,dict_results,dict_subvols,output_path):
    """
    parallelized function to write results of all halos belonging to same subvolume into the same output file.
    
    June 5, 2023: convert to pandas dataframe and write as hdf5 
    """
    
    print('Writing results for subvolume=%s'%subvolume,flush=True)
    
    # filter dict_results to only contain results for objects in current subvolume
    dict_results_filtered = {key:value for key,value in dict_results.items() if value['sol_success'][0]!=False}
    
    # convert filtered dict to pandas dataframe and write as hdf5 
    df_results = pd.DataFrame.from_dict(dict_results_filtered,orient='index')
    df_results.to_hdf(os.path.join(output_path+'output_subvolume_%s.h5'%subvolume),key='data',complevel=9)
        
    

def write(dict_results,tree_tables,parameters):
    """
    this will write solutions for halos of different subvolumes into different output files if needed.
    
    as of jan 6, 2023: the output file is a dict of dictionaries indexed by halo ID
    and is a npz binary file -- we will switch to hdf5 or msgpack in the near future 
    
    June 5, 2023: convert to pandas dataframe and write as hdf5 
    """
    
    if parameters['tree_type'] in ['fire2_pandya22','fire2']:
        # just write a single output file since there are only 12 core halos
        
        df_results = pd.DataFrame.from_dict(dict_results,orient='index')
        df_results.to_hdf(os.path.join(parameters['output_path']+'output_%s.h5'%parameters['tree_type']),key='data',complevel=9)
        
    elif parameters['tree_type'] in ['tng']:
        
        # construct dict of halo_ID:subvolume pairs 
        dict_subvols = {str(key):value['subvolume'][0] for key,value in tree_tables.items()}
        subvolumes = np.unique(list(dict_subvols.values()))
        
        # write in sequence
        # NOTE: this writing needs to be parallelized but I am having issues as of Jan 9, 2023
        for subvolume in subvolumes:
            write_subvolume(subvolume,dict_results,dict_subvols,parameters['output_path']) 
        
    print('Finished writing output files to %s'%parameters['output_path'],flush=True)
    