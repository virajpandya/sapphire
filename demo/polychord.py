%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pandas as pd
from astropy.table import Table
from astropy.cosmology import Planck15
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

import sapphire 

""" This is an optional feature we are using to downsample the # of halos we will model to speed up computation time """

# define a list of (Mlow,Mhigh,N) tuples with same bin definitions as Behroozi+19 relation
# we will only model N random halos within each of these z=0 Mvir mass bins to cut down on modeling time
# this list of tuples is an input runtime parameter for sapphire below (along with np.random.seed(#) for reproducibility) 
downsample_bins = np.arange(10.0,12.4,0.2) # note the masses are log10 mvir
print(downsample_bins)

downsample_defs = [(downsample_bins[i],downsample_bins[i+1],5) for i in range(len(downsample_bins)-1)]

# set up a general parameters dict 

# try running new thermal_general model on same 656 TNG trees 
parameters = {'output_path':'./', # path where any output files should be saved (irrelevant if return_or_write=='return' below)
              
              ### runtime-related related parameters (you only need to vary tree_path) 
              
              'tree_type':'tng_sapphire', # type of merger trees / MAHs -- this determines what read_trees module gets loaded
              'tree_path':'../data/tng_trees_sapphire.npz', # either directory containing tree files or path to single tree file itself in case of TNG experimentation
              'halo_names':[], # optional for certain simulations -- list of halo names/IDs to process (default=empty=process all trees for given tree_type and tree_dir)
              'subvolumes':None, # for testing speed purposes, only work with trees from a single TNG subvolume 
              'num_readers':5, # irrelevant for tree_type='tng_sapphire'
              'num_writers':5, # irrelevant when return_or_write = 'return' (otherwise how many output writer tasks to start in parallel)
              'return_or_write':'return', # if 'return', return pandas dataframe (you should assign to a variable below); if 'write' then save dataframe to hdf5
              'min_root_mass':1e10, # relevant for tree_type ='tng' and 'tng_sapphire', not 'fire2' or 'fire2_pandya22'
              'max_root_mass':3e12, # relevant for tree_type=tng_sapphire -- halos above this z=0 Mvir will not be modeled (set to None or 0 if you don't want)
              'downsample_defs':downsample_defs, # list of tuples (logMlow,logMhigh,N) such that N random halos will be chosen in each mass bin (None otherwise)
              'downsample_seed':999, # np.random.seed(#) to ensure reproducibility when downsampling to random N halos in different mass bins (None otherwise)
              'rtol':1e-5, # relative error tolerance for ODE solver (this controls error on exponent of state variables), should be 1e-3 or lower 
              'atol':1e-3, # absolute error tolerance for ODE solver (this controls early evolution when ICs close to 0), should be 1e-3 or lower 
              
              ### fixed physical model parameters (this just sets the overall physical model and associated fixed parameters) 
              
              'physical_model':'thermal_general', # name of physical model (and associated free parameter functional forms)
              'coolfunc':'wiersma09', # which cooling function to use (wiersma09, sd93, ploeckinger20)
              'alpha_n':-3/2., # slope of CGM density power law
              'alpha_T':0.0, # slope of CGM temperature power law 
              'tau_escape':1.0, # scales the halo outflow timescale tdyn=tau*Rvir/Vvir
              'f_recycle':0.4, # instantaneous stellar-->ISM recycling fraction 
              'e_wind_halo':1.0, # specific energy of halo wind relative to Ecgm/Mcgm                            
              'yZ':0.02, # metal yield of 1 SN per 100 Msun of stars formed (2 Msun / 100 Msun = 0.02 for 10 Msun of SN ejecta)
              'return_all':False, # whether to return only the ODE RHS for solver, or all properties at each time for outputting
              
              ### FREE PARAMETERS BELOW THIS LINE -- WE WILL VARY A SUBSET OF THESE 
              ### there are 4 physical parameters: mass, energy and metal loading factors, and the ISM depletion time
              ### each physical parameter has 4 hyperparameters describing a power law: normalization, slope, and the redshift dependences of those two
              ### the exact power law is X(Vvir,z) = A * (Vvir/125.)**(alpha0 + alphaz*(1+z)) * (1+z)**beta              
              
              'etaM_A':1.0, # normalization of power law for mass loading factor [prior range: 0.01 to 100]
              'etaM_alpha0':-0.5, # slope of power law for mass loading factor [prior range: -2 to 2]
              'etaM_alphaz':0.0, # this can be fixed to 0; redshift dependence of slope of power law for mass loading factor [prior range: -2 to 2]
              'etaM_beta':0.0, # this can be fixed to 0; redshift dependence of normalization of power law for mass loading factor [prior range: -2 to 2]
              'tdep_A':3.0, # [prior range: 0.01 to 10]
              'tdep_alpha0':-3.0, # [prior range: -2 to 2]
              'tdep_alphaz':0.0, # this can be fixed to 0 [prior range: -2 to 2]
              'tdep_beta':-0.7, # [prior range: -2 to 2]
              'etaE_A':0.1, # [prior range: 0.01 to 1]
              'etaE_alpha0':-0.5, # [prior range: -2 to 2]
              'etaE_alphaz':0.0, # this can be fixed to 0 [prior range: -2 to 2]
              'etaE_beta':0.0,   # this can be fixed to 0 [prior range: -2 to 2]
              'etaZ_A':0.5, # [prior range: 0.01 to 1]
              'etaZ_alpha0':0.0, # [prior range: -2 to 2]
              'etaZ_alphaz':0.0, # this can be fixed to 0 [prior range: -2 to 2]
              'etaZ_beta':0.0} # this can be fixed to 0 [prior range: -2 to 2]


names = ['etaM_A', 
         'etaM_alpha0', 
         'tdep_A', 
         'tdep_alpha0', 
         'tdep_beta', 
         'etaE_A',
         'etaE_alpha0',
         'etaZ_A',
         'etaZ_alpha0'
        ]

fid = np.array([1., -0.5, 3., -3., -0.7, 0.1, -0.5, 0.5, 0.])
low = np.array([0.01, -2, 0.01, -4, -2, 0.01, -2, 0.01, -2])
high = np.array([100., 2, 10., 4, 2, 1., 2, 1., 2])

""" 
Suppose you wanted to compute the goodness of fit 
One way is to first bin up the SAM halos in the same mass bins as Behroozi and then compute the median SMHM of all halos in each mass bin

NOTE: I am not sure this is the best way to compare the model to the data
It is probably better to compute the deviation of each individual halo from the Behroozi+19 median for the mass bin it belongs to
But for that, I need to fix a minor bug with the interpolator inside sapphire for df_results['Mvir'], so maybe use this median approach for now ...
"""

def median_smhm(df,Mmin,Mmax):
    """ 
    Filters input dataframe to only consider halos with z=0 Mvir between Mmin and Mmax
    Then returns median SMHM ratio for those halos
    """ 
    
    Mvir_bin = np.array([df.iloc[irow]['Mvir'][-1] for irow in range(len(df)) 
                            if df.iloc[irow]['Mvir'][-1]>=10**Mmin and df.iloc[irow]['Mvir'][-1]<=10**Mmax])
    Mstar_bin = np.array([df.iloc[irow]['M_star'][-1] for irow in range(len(df)) 
                             if df.iloc[irow]['Mvir'][-1]>=10**Mmin and df.iloc[irow]['Mvir'][-1]<=10**Mmax])    
    
    return np.median(Mstar_bin / Mvir_bin)

# read in Behroozi+19 UniverseMachine DR1 Median SMHM relation at z=0  
# and discard bins below logmvir<10.5 due to Bolshoi-Planck resolution limit (not sure why he included these)
tum = Table.read('smhm_a1.002312.dat',format='ascii')
tum = tum[(tum['HM(0)']>=10.0) & (tum['HM(0)']<=12.4)]

def log_like(theta):
    parameters.update({n: t for (n, t) in zip (names, theta)}) # before these were 1.0 and 0.1 respectively 
    
    df_results = sapphire.run(parameters)
    # we will store median smhm of predicted halos in the two models above
    m_smhm = np.array([])

    for Mmin,Mmax,N in downsample_defs: # our Mmin,Mmax bin edges for downsampling the # of halos was the same as the Behroozi bin definitions

        ### filter dataframe for both models to only contain halos that fall within current z=0 Mvir bin

        # first the weak feedback model (df_results)
        m_smhm = np.append(m_smhm, median_smhm(df_results,Mmin,Mmax))

    
    mask_high = np.log10(m_smhm) > tum['Med_Cen_SF(7)']
    mask_low = ~mask_high

    errors = np.abs(tum['Err-(9)']) * mask_low + np.abs(tum['Err+(8)']) * mask_high
    return - 0.5*np.sum((tum['Med_Cen_SF(7)'] - np.log10(m_smhm))**2/errors**2), []

def prior(hypercube):
    """ Uniform prior from [-1,1]^D. """
    return low + hypercube * (high - low)

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

nDims = len(names)
nDerived = 0
settings = PolyChordSettings(nDims, nDerived)
#settings.file_root = 'noisy_fullrange_temp099'
settings.file_root = 'sapphire'
settings.nlive = 50
settings.num_repeats = nDims
settings.precision_criterion = 0.1
settings.do_clustering = True
settings.read_resume = False

output = pypolychord.run_polychord(log_like, nDims, nDerived, settings, prior, dumper)