"""
This is the second test that
1. reads in the new test output file saved by test_run.py
2. reads in a pre-generated output file for the exact same test performed during code development
3. asserts whether several numpy arrays in the new test output file are exactly equal to the arrays in the pre-generated output file
4. saves an output png file that the user can visually compare to a pre-provided png if they wish (see documentation)
"""

# make sure the relevant general python packages can be imported for testing (this is mainly a test of the user environment)
import numpy as np 
import os 

# get absolute path of directory containing this test script (should be "/PATH/sapphire/tests/")
path_abs = os.path.dirname(os.path.abspath(__file__)) 

# load the new test output file created by test_run.py 
path_out_test = os.path.join(path_abs,'test_output.npz') 
npz_test = np.load(path_out_test,allow_pickle=True)

path_out_ref = os.path.join(path_abs,'test_data/solutions_fire2_viraj.npz') # exact same file as test_output.npz but generated locally during code development
npz_ref = np.load(path_out_ref,allow_pickle=True)

# define the 3 FIRE-2 halo names we used for testing 
# NOTE: this needs to be generalized to any other halo IDs and tests that we performed (including if test_run.py generates multiple outputs)
halo_names = ['m10q','m11c','m12m']

# begin automatic test of comparing test solutions to provided reference solutions
for halo_name in halo_names:
    
    # read in the test and reference solution for this halo
    sol_test = npz_test[halo_name].item()
    sol_ref = npz_ref[halo_name].item()
    
    
    # assert whether time series of state variables and associated derivatives are EXACTLY EQUAL in test output file vs. reference solution file
    # the reference solution file is the result of test_run.py during code development (and sanity checked by eye as needed)
    # if ANY of these fail, the whole test will fail (and it means something went wrong with generating the solution in the user's environment)
    
    test_vars = ['cosmic_age','redshift',
                 'M_cgm','Mdot_cgm',
                 'M_ism','Mdot_ism',
                 'M_star','Mdot_sfr',
                 'Eth_cgm','Edot_cgm_th',
                 'Ekin_cgm','Edot_cgm_kin',
                 'MZ_cgm','MZdot_cgm',
                 'MZ_ism','MZdot_ism',
                 'MZ_star','MZdot_star']
    
    for test_var in test_vars:
        # NOTE: may want to switch to np.testing.assert_array_allclose() with rtol set to machine precision
        np.testing.assert_array_equal(sol_test[test_var],sol_ref[test_var],err_msg='>>>>> test failed for comparing %s to reference solution'%test_var)
        
        
# do NOT close the output and reference test files in case user wants to interactively plot things in test_interactive.ipynb 
# npz_test.close()
# npz_ref.close()

