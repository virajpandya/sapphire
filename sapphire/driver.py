"""
sapphire -- a next-generation multi-zone model of galaxy formation 

This is the main driver module that takes an input parameter dictionary 
or JSON file path and then calls the relevant package modules in the order required. 
"""

# load general modules
import numpy as np
from glob import glob

# load sapphire modules that do not require dependency injection
from .coolfunc import read_coolfunc 
from .parameter_functions import uvb_filtering
from .read_trees import interpolate_trees
from .physical_models import fiducial_model 

# NOTE: this should be further modularized as needed, including any dependency injections (loading of modules based on parameters dict)
def run(parameters):
    """
    driver module that calls all the other submodules in the required order.
    
    parameters: user-provided dict or JSON filename giving model/runtime parameters 
    """
    
    # parse the input parameters object (must be a dict or a string giving the path of parameters JSON file that will be converted to dict)
    if type(parameters) == dict: # later will add an argparse util module for making sure the provided dict is sensible
        pass 
    elif type(parameters) == str: 
        raise NotImplementedError('JSON input filename option not yet implemented')
    else:
        raise ValueError('parameters must be provided as either a python dict or JSON filename')
        
    # load the relevant tree reading module
    if parameters['tree_type'] == 'fire2_sapphire':
        from .read_trees import read_fire2_sapphire as tree_reader
    elif parameters['tree_type'] == 'bolshoi_planck':
        raise NotImplementedError('bolshoi-planck trees not yet implemented')
    else: # NOTE: glob and print a list of available tree_type strings based on modules available in read_trees subdirectory
        raise ValueError('tree_type must be one of the types implemented in the read_trees module') 
        
    # now read the trees into a dict where the key is halo name/ID and element is an astropy Table
    tree_tables = tree_reader.read_halos(halo_names=parameters['halo_names'],tree_dir=parameters['tree_dir'])
    
    # create smooth interpolator functions for required halo properties vs time (redshift, logMAR, logMvir, logRvir, logVvir, logcNFW)
    # this is a dict of lists where each key is a halo name/ID
    tree_interpolators = interpolate_trees.run(tree_tables) 
    
    # read in the requested parameter functions module
    if parameters['parameter_functions'] == 'fire2': 
        from .parameter_functions import fire2 as param_functions 
        
    # retrieve the list of parameter fitting functions in the order expected for unpacking by the integrator
    list_param_functions = param_functions.get()
    
    # retrieve the list of UVB filtering mass and collapse fraction functions
    list_uvb_filtering = uvb_filtering.get()
    
    # read cooling function and create N-dimensional interpolator object
    coolfunc = read_coolfunc.return_coolfunc(parameters['coolfunc'])
    
    # finally solve for the baryonic evolution of each tree and collect the resulting dict objects into a dict with halo_names/IDs being the keys
    # NOTE: this can and should be parallelized, along with the read_trees and interpolate_trees steps 
    dict_results = {}
    
    for halo_name in tree_tables.keys(): 
        
        comb_dict = fiducial_model.integrator(parameters,tree_interpolators[halo_name],list_param_functions,list_uvb_filtering,coolfunc)
        
        dict_results[halo_name] = comb_dict
        
        print('>>>>> FINISHED halo=%s'%halo_name)
                
    # finally save all dictionaries into a single npz file with keys being the halo names/IDs
    # NOTE: this is a temporary way of saving -- it will be moved to another module and either hdf5 or msgpack
    np.savez(parameters['output_file'], **dict_results)
    
    print('Finished -- thank you for using sapphire!',flush=True)

    return None
