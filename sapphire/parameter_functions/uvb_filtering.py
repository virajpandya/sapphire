"""
This submodule defines functions to compute the suppression of gas accretion into halos due to the UV background.

December 7, 2022: currently only the classic Kravtsov+04/Okamoto+08 model
"""

import numpy as np
from astropy.cosmology import Planck15 

def m_filt(z, z_overlap=9.0, z_reionize=3.5, z_squelch=8.0):
    """
    This computes the "filtering mass" below which halos accrete <50% of cosmic baryon fraction fb*Mdot_in_dm (Gnedin+00)
    This is the analytic model from Appendix B of Kravtsov+04 and basically copied from Rachel Somerville's Santa Cruz SAM code
    z1=z_overlap, z2=z_reionize, z_squelch = when suppression ('squelching') turned on
    NOTE:   Rachel tuned these parameters to reproduce Okamoto+08 so some of the parameter names have lost their meaning
            I should probably re-write this so the parameters reflect their intended meaning (eg z_reionize should be >> 3.5)    
    """
    
    a_o = 1.0/(1+z_overlap) # scale factor where multiple HII regions BEGIN TO overlap
    a_r = 1.0/(1+z_reionize) # scale factor of complete reionization
    a = 1.0 / (1+z) # current scale factor
    
    # Jeans mass in Msun; 0.59 is mean molecular weight
    m_jeans = 2.5E11 * Planck15.h**-1 * np.sqrt(Planck15.Om0)**-1 * (0.59)**-1.5 
    
    alpha = 6.0 # power that controls growth rate of UV background flux (Kravtsov+04 fixed alpha=6)
    
    if a < a_o:
        f_a = (3.0 * a / ((2.0+alpha)*(5.0+2.0*alpha))) * (a/a_o)**alpha
    elif a >= a_o and a <= a_r: 
        f_a = (3.0/a)*a_o**2*(1.0/(2+alpha) - (2*(a/a_o)**-0.5)/(5+2*alpha)) + a**2/10. - (a_o**2/10.)*(5-4*(a/a_o)**-0.5)
    elif a > a_r: 
        f1 = a_o**2*(1/(2+alpha) - 2*(a/a_o)**-0.5/(5+2*alpha))
        f2 = a_r**2/10.*(5-4*(a/a_r)**-0.5)
        f3 = a_o**2/10.*(5-4*(a/a_o)**-0.5)
        f4 = a*a_r/3. - a_r**2/3.*(3-2*(a/a_r)**-0.5)
        f_a = 3.0/a * (f1 + f2 - f3 + f4)
        
    mf = m_jeans * f_a**1.5
    
    # constant from Rachel's stars.cc implementation
    # this ensures Mfilt defined in this way matches the Okamoto+08 "M_characteristic" definition normalization
    Mfilt_factor = 0.0933 
    
    # return filtering mass in Msun
    return Mfilt_factor * mf 

def collapse_fraction(m,mfilt):
    """
    This returns the actual gas accretion fraction that should be multiplied by fb*Mdot_in_dm to get Mdot_in_gas_halo
    The rest of (1-fcoll) is heated to >Tvir by the UV background and does not accrete
    
    Halos with Mvir >> Mfilt will have f_UV ~ 1 (no suppression), whereas much lower mass halos tend to f_UV ~ 0
    This is eqn (1) of Okamoto+08 who studied the UV background in hydro simulations
    """
    
    alpha = 2.0
    
    fcoll = (1+(2**(alpha/3.)-1.)*(m/mfilt)**-alpha)**(-3/alpha)
    
    return fcoll

def get():
    """
    Convenience function to return list of the above functions [m_filt(),collapse_fraction()] for the integrator
    """
    
    return [m_filt,collapse_fraction]