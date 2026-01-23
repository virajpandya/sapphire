"""
this module reads the config (nested dict) and/or yaml file 
"""

import yaml
from pprint import pprint
import sys


### function that reads user-supplied config, merging with .yaml file if provided
def get(config,verbose=False):

    if 'path_config' in config.keys():

        if verbose is True:
            print('reading %s'%config['path_config'],flush=True)
        
        # load the baseline config file (must be yaml)
        with open(config['path_config'], 'r') as f:
            yaml_config = yaml.safe_load(f)

        
        # merge/override any baseline configs with the config dict itself
        config = {**yaml_config, **config} 

    if 'path_config' not in config.keys() and len(config.keys()) == 0:
        raise ValueError('must provide config via yaml file and/or manual dict')

    ### pretty-print parsed config 
    if verbose is True:
        print('\nyour full sapphire config:\n',flush=True)
        pprint(config, indent=4, compact=False) #, width=80, 
        print('\n',flush=True)

    ##### add safeguard checks here

    # fit_mock and fit_obs cannot both be True (they can both be False if user doesn't want inference)
    if config['inference_config']['fit_mock'] and config['inference_config']['fit_obs']:
        sys.exit('fit_mock and fit_obs cannot both be True')
    
    return config 