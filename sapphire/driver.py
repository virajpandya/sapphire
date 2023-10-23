"""
sapphire -- a next-generation multi-zone model of galaxy formation 

This is the main driver module that takes an input parameter dictionary 
or JSON file path and then calls the relevant package modules in the order required. 
"""

# load general modules
import numpy as np
np.seterr(all='ignore') # comment out when debugging
from glob import glob
import multiprocessing
from functools import partial 
import os
import pandas as pd

# load sapphire modules that do not require dependency injection
from .coolfunc import read_coolfunc 
from .parameter_functions import uvb_filtering
from .read_trees import interpolate_trees
from .utils import writer 

# NOTE: this should be further modularized as needed, including any dependency injections (loading of modules based on parameters dict)
def run(parameters):
    """
    driver module that calls all the other submodules in the required order.
    
    parameters: user-provided dict or JSON filename giving model/runtime parameters 
    
    NOTE: at some point I plan to offload a lot of the if/else checks on the parameters dict to another module 
    """
    
    # parse the input parameters object (must be a dict or a string giving the path of parameters JSON file that will be converted to dict)
    #print('Parsing input parameters...',flush=True)
    if type(parameters) == dict: # later will add an argparse util module for making sure the provided dict is sensible
        pass 
    elif type(parameters) == str: 
        raise NotImplementedError('JSON input filename option not yet implemented')
    else:
        raise ValueError('parameters must be provided as either a python dict or JSON filename')
    
    # if output directory does not already exist, create it
    if os.path.exists(parameters['output_path']) == False:
        os.mkdir(parameters['output_path'])
    
    # load the relevant tree reading module
    if parameters['tree_type'] == 'fire2_pandya22':
        from .read_trees import read_fire2_pandya22 as tree_reader
    elif parameters['tree_type'] == 'fire2':
        from .read_trees import read_fire2 as tree_reader 
    elif parameters['tree_type'] == 'tng':
        from .read_trees import read_tng as tree_reader
    elif parameters['tree_type'] == 'tng_sapphire':
        from .read_trees import read_tng_sapphire as tree_reader
    else: # NOTE: glob and print a list of available tree_type strings based on modules available in read_trees subdirectory
        raise ValueError('tree_type must be one of the types implemented in the read_trees module') 
        
    # now read the trees into a dict where the key is halo name/ID and element is an astropy Table
    #print('Reading trees...',flush=True)
    tree_tables = tree_reader.read_trees(parameters)
    
    # create smooth interpolator functions for required halo properties vs time (redshift, logMAR, logMvir, logRvir, logVvir, logcNFW)
    # this is a dict of lists where each key is a halo name/ID
    #print('Interpolating trees...',flush=True)
    tree_interpolators = interpolate_trees.run(tree_tables)
    
    # retrieve the list of UVB filtering mass and collapse fraction functions
    list_uvb_filtering = uvb_filtering.get()
    
    # read cooling function and create N-dimensional interpolator object
    coolfunc = read_coolfunc.return_coolfunc(parameters['coolfunc']) 
    
    #print('Evolving model halos...',flush=True)
    
    # by default we will assume a single node and use all cores on that node to solve every tree in parallel 
    # NOTE: later we will add an option to use mpi4py to distribute integrations (and tree reading, and parameter variations) across cores of multiple nodes
    
    # use dependency injection to read in the requested physical model, then the associated parameter functions using dependency injection
    # NOTE: this parsing / dependency injection will eventually be moved to another utility module 
    if parameters['physical_model'] == 'pandya22':
        from .physical_models import pandya22 as physical_model
        from .parameter_functions import pandya22 as param_functions        
    elif parameters['physical_model'] == 'thermal_general':
        from .physical_models import thermal_general as physical_model
        from .parameter_functions import thermal_general as param_functions
    elif parameters['physical_model'] == 'thermal_general_AGN':
        from .physical_models import thermal_general_AGN as physical_model
        from .parameter_functions import thermal_general as param_functions        
        
    # retrieve the list of parameter fitting functions in the order expected for unpacking by the integrator
    list_param_functions = param_functions.get()    
        
    # use functools.partial to fix most arguments to the integrator function except the tuple (halo_name, tree_interpolators[halo_name])
    pool_integrator = partial(physical_model.integrator,parameters=parameters,parameter_functions=list_param_functions,uvb_model=list_uvb_filtering,coolfunc=coolfunc)
    
    # zip list of tuples of (halo_name, tree_interpolators[halo_name]) pairs for pool_integrator function
    halo_data_tuples = [(k,v) for k,v in tree_interpolators.items()]
    
    # set up for python single-node parallel processing -- by default this will use the full number of cores available since it is not memory-intensive
    PoolProcesses = multiprocessing.cpu_count()
    #print('We will process %s trees in parallel...'%PoolProcesses,flush=True)

    # start pool process 
    pool = multiprocessing.Pool(processes=PoolProcesses) 

    # call the pool_integrator function in parallel with each pair in halo_data_tuples as inputs
    out = pool.map(pool_integrator,halo_data_tuples)

    # close pool processes to free memory
    pool.close()
    pool.join()

    # collapse the list of dicts into a single dict with halo_name:results_dict pairs 
    dict_results = {key:value for d in out for key,value in d.items()}

    if parameters['return_or_write'] == 'write':            
        # finally save all dictionaries into a single npz file with keys being the halo names/IDs
        # NOTE: this is a temporary way of saving -- will create a separate writer module function and output will be either hdf5 or msgpack
        writer.write(dict_results,tree_tables,parameters)
        
        return None
    elif parameters['return_or_write'] == 'return':
        # simply return dict_results
        #print('Returning dictionary of results directly instead of writing output files',flush=True)
        
        # June 5, 2023: filter out bad solutions and return as pandas dataframe         
        dict_results_filtered = {key:value for key,value in dict_results.items() if value['sol_success'][0]!=False}
        
        df_results = pd.DataFrame.from_dict(dict_results_filtered,orient='index')
        
        return df_results
        
    # print('Finished -- thank you for using sapphire!',flush=True)

    
