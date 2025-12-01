"""
this module reads the config (nested dict) and/or yaml file 
"""

import yaml
from pprint import pprint


### function that reads user-supplied config, merging with .yaml file if provided
def get(config):

    if 'path_config' in config.keys():

        print('reading %s'%config['path_config'],flush=True)
        
        # load the baseline config file (must be yaml)
        with open(config['path_config'], 'r') as f:
            yaml_config = yaml.safe_load(f)

        
        # merge/override any baseline configs with the config dict itself
        config = {**yaml_config, **config} 

    if 'path_config' not in config.keys() and len(config.keys()) == 0:
        raise ValueError('must provide config via yaml file and/or manually')

    ### print parsed config 
    print('\nyour input config:\n',flush=True)
    pprint(config, indent=4, compact=False) #, width=80, 

    print('\n',flush=True)
    
    return config 