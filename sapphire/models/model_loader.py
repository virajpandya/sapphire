"""
this loads the ODE module specified by config['model'] and returns integrator, saveat_fn, list_parameterizations

as of jan 6, 2026, the model registry includes
- pandya23.py

"""

import importlib 

def get(config,verbose=False):

    ### this can probably be automated  using importlib.load_module
    # if config['model'] == 'pandya23':
    #     from sapphire.models import pandya23 as model
    #     integrator, saveat_fn, list_parameterizations = model.setup(config,verbose)
    # else:
    #     raise ValueError('you must enter the name of an existing model within the sapphire models module') 
    

    try:
        
        model = importlib.import_module('sapphire.models.%s'%config['model'])
        integrator, saveat_fn, list_parameterizations = model.setup(config)
        
        if verbose is True:
            print('successfully loaded model=%s'%config['model'],flush=True)
            
        return integrator, saveat_fn, list_parameterizations
        
    except:
        
        raise ModuleNotFoundError('you must enter the name of an existing model within the sapphire models module')
        

####