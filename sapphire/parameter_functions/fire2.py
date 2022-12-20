"""
This module implements the fiducial FIRE-2 functional forms and parameter values for the free parameters
of our model (from Pandya+22)

This file can be adapted for other use cases -- for example:
- defining another simulation.py module that contains different functional forms and parameter values (e.g., tng.py, eagle.py, camels_runXXX.py)
- defining a generic.py module that pre-defines general functional forms but with parameter values provided as function inputs from another method (e.g., MCMC)
"""

import numpy as np

def get_tdep(Vvir, redshift): 
    # Vvir at Rvir in km/s, redshift dimensionless, return value is t_dep in Gyr

    alphaz = -3.0 + 2.4*np.log10(1+redshift)
    return 10**0.46 * (Vvir/125.)**alphaz * (1+redshift)**-0.7

def get_etaM_ism(Vvir, redshift):
    # ISM mass loading at 0.1-0.2 Rvir using Bernoulli velocity cut to 0.5Rvir (or farther)
    # Vvir at Rvir in km/s, redshift dimensionless, returns dimensionless mass loading factor eta:=mdot_out/SFR 
    
    #### from single-snapshot particle tracking
    alphaz = -3.7 + 4.2*np.log10(1+redshift)
    return 10**-0.2 * (Vvir/125.)**alphaz * (1+redshift)**2.4

def get_vB_ism(Vvir,redshift):
    # Bernoulli velocity in km/s for ISM outflows 

    #### from single-snapshot particle tracking
    alphaz = 1.0 - 0.3*np.log10(1+redshift)
    return 10**2.4 * (Vvir/125.)**alphaz * (1+redshift)**-0.2

def get_Zin_halo(Vvir, redshift):
    # halo inflow metallicity (Zsun) vs. Vvir (km/s) and redshift
    # multiply again by by Zsun=0.02 to get Zdot/Mdot for the model
    # this is for single-adjacent-snapshot particle tracking
    
    alphaz = 2.2 + 0.1*np.log10(1+redshift)
    return 0.02 * 10**-0.6 * (Vvir/125.)**alphaz * (1+redshift)**-1.3

def get_fprev(Vvir):
    # generalized logistic function that depends only on Vvir (not redshift) -- fitted with curve_fit (see rescalings.ipynb)
    # i imposed a bound on fmin=0.2 during curve_fitting to prevent crazy low accretion rates for m10q 
    # return 0.2 + (0.99669179 - 0.2) / (1.0 + np.exp(-5.99220277*(np.log10(Vvir) - 1.86754933)))
    return 0.25 + (0.9641 - 0.25) / (1.0 + np.exp(-7.1348*(np.log10(Vvir) - 1.8775)))

def get_fthermal_accretion(Vvir,redshift):
    # power law with both slope and normalization depending on redshift, with parameters from curve_fit (see rescalings.ipynb)
    # this can exceed 1 so must take minimum wrt 1 otherwise model will crash    
    logx = np.log10(Vvir)
    logz = np.log10(1+redshift)
    return np.minimum(1,10**(-0.1666 + (-0.2757-0.8675*logz)*(logx-np.log10(125.)) - 0.6559*logz))

def get_fthermal_wind(Vvir):
    # generalized logistic function that depends only on Vvir (not redshift) -- fitted with curve_fit (see rescalings.ipynb)
    # i imposed a bound on fmax=1.0 during curve_fitting so np.minimum is not needed
    return 0.3524 + (1.0 - 0.3524) / (1.0 + np.exp(20.8774*(np.log10(Vvir) - 1.5012)))

def get():
    """
    This convenience function returns the other functions in a list for use by the integrator.
    The order of the functions in the returned list must be the same as expected by the physical_models.evolve_galaxy() function
    """
    
    return [get_tdep, get_etaM_ism, get_vB_ism, get_Zin_halo, get_fprev, get_fthermal_accretion, get_fthermal_wind]