"""
This is the simplest test of the code -- define the dict of input parameters, import sapphire, and call sapphire.run(parameters)
"""

### delete output file expected from this script if it already exists
import os 
path_abs = os.path.dirname(os.path.abspath(__file__)) # absolute path of the directory containing this test file (should be "/PATH/sapphire/tests/")
path_outfile = os.path.join(path_abs,'test_output.npz') # absolute path of the output file to be created by this test_run.py script
if os.path.exists(path_outfile) == True:
    os.remove(path_outfile) 

# get absolute path to the included test data directory (should be "/PATH/sapphire/tests/test_data/")
path_data = os.path.join(path_abs,'test_data')

# test whether sapphire can be imported 
import sapphire

# initialize a dict of input parameters for testing sapphire 
parameters = {'output_file':path_outfile, # name of output file, NOTE: this will be rewritten as a directory location, and our I/O module will take care writing
              'tree_type':'fire2_sapphire', # type of trees -- this determines what read_trees module gets loaded
              'tree_dir':os.path.join(path_data,'example_merger_trees/'), # directory containing merger trees
              'halo_names':['m10q','m11c','m12m'], # optional for certain simulations -- list of halo names/IDs to process (default=empty=process all trees for given tree_type and tree_dir)
              'parallelize_read_trees':False, # whether to use parallelization to read in (and interpolate) merger tree files
              'parallelize_integration':False, # whether to use paralelization to integrate the evolution of every halo
              'parallelize_mode':None, # whether to use multiprocessing or mpi4py for parallelization (default: None)
              'parallelize_options':None, # list of optional args for multiprocessing or mpi4py such as number of nodes, cores, etc. (default: None)
              'parameter_functions':'fire2', # name of module in parameter_functions with fitting functions for free parameters
              'coolfunc':'wiersma09', # which cooling function to use (wiersma09, sd93, ploeckinger20)
              'alpha_n':-3/2., # slope of CGM density power law
              'alpha_T':0.0, # slope of CGM temperature power law 
              'tau_escape':1.0, # scales the halo outflow timescale tdyn=tau*Rvir/Vvir
              'f_recycle':0.4, # instantaneous stellar-->ISM recycling fraction 
              'Rturb_transition':7.0, # time in Gyr where Rturb logistic function drops to ~half its max value
              'Rturb_slope':2.0, # exponential slope for smoothness/abruptness of Rturb logistic transition
              'f_supp':1.0, # multiply vturb in tff,eff by this to get more turb pressure support for a given vturb
              'e_wind_halo':1.0, # specific energy of halo wind relative to Ecgm/Mcgm              
              'yZ':0.02, # metal yield of 1 SN per 100 Msun of stars formed (2 Msun / 100 Msun = 0.02 for 10 Msun of SN ejecta)
              'Rturb_type':'RvirLogistic', # either Rvir so Rturb(t)=Rvir(t), or fiducial Rvir-->Rmax logistic from Pandya+22
              'verbose':False, # whether to print values of several properties at each timestep of integration (for debugging)
              'return_all':False} # False for initial solve_ivp call; will be re-run with True to store ALL properties for output 

# finally simply call the main driver function of sapphire to test whether it works
sapphire.run(parameters)

