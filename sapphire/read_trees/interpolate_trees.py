"""
This module does a smooth interpolation of the relevant merger tree properties vs. time 
for all the halos read in by the other properties. The resulting smooth interpolator objects
will then be provided to the ODE integrator to ensure stability for the adaptive timestepping.

December 8, 2022: currently the model needs the following merger tree properties smoothly interpolated:
redshift, halo DM accretion rate, Mvir, Rvir, Vvir, NFW concentration
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from astropy.table import Table

def run(tree_tables):
    """
    Performs a smooth interpolation of redshift, DM accretion rate, Mvir, Rvir, Vvir, NFW concentration
    for each halo's most massive progenitor history provided in tables_halos -- these properties should
    already be made available by the relevant read_*.py module. The interpolation is done vs. cosmic age
    (time since Big Bang) in Gyr.
    
    NOTE: We generally interpolate in log-space since that is better behaved for many of our properties that
    change by orders of magnitude as a function of time. This should be fine for Mvir, Rvir, cNFW, etc. but 
    it should be checked for the DM accretion rate -- especially if it's a net rather than gross rate since
    the former can be <= 0 thus making log10 undefined. Net accretion rates must necessarily be used for 
    large-volume simulations by finite-differencing the Mvir time series, so those are particularly susceptible.
    
    Returns a list of N lists of size M where N is the number of halos and M is the number of requested
    property interpolator objects -- the lists are 1:1 mapped with the input tree_tables
    
    tree_tables is a dict of Astropy Tables giving the tree for each halo (the keys of this dict are the halo names/IDs)
    """
    
    # initialize an empty dict that will store lists of interpolators for each halo with the key being the halo_name
    tree_interpolators = {}
    
    # loop over each tree and create then store the needed interpolator objects
    # NOTE: if this is slow for large simulations, can do another optional round of multiprocessing/mpi4py here
    for tname in tree_tables.keys():
        t = tree_tables[tname]

        # faster to interpolate snapshot redshift than do cosmology root-finding calculation
        interp_redshift = UnivariateSpline(t['cosmic_age'],t['redshift'],k=5,s=0) 

        # log10 of DM accretion rate in Msun/yr
        interp_logMAR = UnivariateSpline(t['cosmic_age'],np.log10(gaussian_filter1d(t['dm_accretion_rate'],10,mode='nearest')),k=5,s=2) 

        # log10 of Mvir in Msun (no little-h)
        interp_logMvir = UnivariateSpline(t['cosmic_age'],gaussian_filter1d(t['log_mvir'],10,mode='nearest'),k=5,s=2) 

        # log10 of Rvir in proper kpc (no little-h)
        interp_logRvir = UnivariateSpline(t['cosmic_age'],gaussian_filter1d(t['log_rvir'],10,mode='nearest'),k=5,s=2)

        # log10 of Vvir := sqrt(GMvir/Rvir) in proper km/s 
        interp_logVvir = UnivariateSpline(t['cosmic_age'],gaussian_filter1d(t['log_vvir'],10,mode='nearest'),k=5,s=2) 

        # log10 of NFW halo concentration := Rvir / Rs_Klypin 
        interp_logcNFW = UnivariateSpline(t['cosmic_age'],gaussian_filter1d(t['log_cNFW'],10,mode='nearest'),k=5,s=2) 
        
        # finally add a list for this halo to tree_interpolators dict with this tree name being the key -- the parent module should respect the list order! 
        tree_interpolators[tname] = [interp_redshift,interp_logMAR,interp_logMvir,interp_logRvir,interp_logVvir,interp_logcNFW]
        
    return tree_interpolators 

