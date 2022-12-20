"""
This submodule reads in the cooling function, interpolates it, and sends it back to parent module.

December 7, 2022: currently only Wiersma+09 and Sutherland & Dopita 1993 are allowed.
"""

import numpy as np
import h5py 
from scipy.interpolate import RegularGridInterpolator, interp2d 
from glob import glob 
import os 



def interpolate_wiersma09():
    """
    This function reads and interpolates the Wiersma09 cooling functions found in the ./data/wiersma09/ subdirectory
    
    The returned coolfunc object can be called as coolfunc(redshift,logZ/Zsun,logT,lognH) and the value is a number with units of erg*cm**3
    """
    
    # first get the absolute path to the wiersma09 data subdirectory
    path_abs = os.path.dirname(os.path.abspath(__file__)) # absolute path of the directory containing this file (read_coolfunc.py)
    path_coolfuncs = os.path.join(path_abs,'data/wiersma09') # absolute path to wiersma09 data subdirectory 
    
    ### glob the Wiersma+09 cooling tables for all redshifts
    files_W09 = glob('%s/z_*.hdf5'%path_coolfuncs)
    files_W09.sort() # sort from low-z to high-z
    files_W09 = files_W09[:-3] # remove z_collis, z_photodis and z_8nocompton (final 3 in list)

    redshifts_W09 = np.asarray([f[f.find('z_')+2:f.find('.hdf5')] for f in files_W09]) # 1-1 mapped list of redshifts
    
    ### Create a list of lists where each entry is a 2D matrix (so a 4D matrix)
    # each outer row = different redshift
    # each outer column = different metallicity (for a given redshift row)
    # value for a given outer (row,column) = a 2D matrix giving cooling function value for each Temp (inner row) and nH (inner column)
    matrices = []

    Zarr = np.logspace(-4,1,10) # initialize list of overall metallicities 

    for fnum,file in enumerate(files_W09):
        this_redshift = float(redshifts_W09[fnum])        
        f = h5py.File(file, 'r')
        
        # first pull the pure H+He cooling table for this redshift along with the nH and T bins (roughly uniform in log)
        # NOTE: need to assume a helium mass fraction (7 available values, see via Metal_free['Helium_mass_fraction_bins'].value)
        # I chose the Solar Helium mass fraction from Asplund+09 of Y=0.2485 (values range from Y=0.238 to 0.298)
        Metal_free = f.get('Metal_free')
        coolfunc_Metal_free = np.array(Metal_free['Net_Cooling'][1]) # 1st index assumes helium Y=0.248
        bins_logT = np.log10(np.array(Metal_free['Temperature_bins']))
        bins_lognH = np.log10(np.array(Metal_free['Hydrogen_density_bins']))
        
        # next get the total cooling from all metals (we will add this to the above after scaling total overall metallicity Z)
        # this assumes solar relative abundances for each of the elements (and hence solar metallicity overall)
        # NOTE: these Total_Metals coolfuncs are based on the exact same logT and lognH bins as for Metal_free above 
        Total_Metals = f.get('Total_Metals')
        coolfunc_Total_Metals = np.array(Total_Metals['Net_Cooling']) # 2D coolfunc as above; no indexing needed (unlike Y above)
        
        f.close()
        
        # Sum Metal_free and Total_Metals cooling functions assuming a range of overall metallicities
        # this is eqn 5 of Wiersma+09 assuming electron fraction n_e/n_H = solar n_e/n_H
        # that solar n_e/n_H assumption is true at all z within 5% (checked by dividing provided Metal_free and Solar ne/nH arrays)
        coolfunc_Zarr = [coolfunc_Metal_free + Zval*coolfunc_Total_Metals for Zval in Zarr]
        
        # Create a 4-dimensional array as above
        
        # initialize list of arrays to store at this redshift -- each array will correspond to one metallicity
        arrs_this_redshift = []
        
        for Znum,Zval in enumerate(Zarr): 
            # append the 2D coolfunc for this metallicity (at this redshift)
            arrs_this_redshift.append(coolfunc_Zarr[Znum])
            
        # append this redshift's metallicity-dependent list of 2D coolfuncs to overall list
        matrices.append(arrs_this_redshift)
        
    ### prepare inputs for interpolation -- don't use log since can have net heating (negative coolfunc)        
    values_4D = np.asarray(matrices) 
    
    # tuple the 1D coordinate arrays following the same order of dimensions as above (use log for metallicity coordinate)
    coords = (redshifts_W09.astype('float'),np.log10(Zarr),bins_logT,bins_lognH)    

    # generate interpolation function for this 4D coolfunc 
    # turn off bounds error and instead have the function return 0 if outside the domain (no extrapolation)
    coolfunc_W09 = RegularGridInterpolator(coords,values_4D,bounds_error=False,fill_value=0.0)
    
    # finally return this interpolator object for use in the parent module
    return coolfunc_W09

def interpolate_sd93():  
    
    # when implementing, remember to make sure the call signature (inputs) to resulting coolfunc interpolator object is same as fiducial coolfunc_W09
    
    raise NotImplementedError('viraj will implement the SD93 coolfunc soon')

def interpolate_ploeckinger20(): 
    
    # when implementing, remember to make sure the call signature (inputs) to resulting coolfunc interpolator object is same as fiducial coolfunc_W09    
    
    raise NotImplementedError('viraj will implement the Ploeckinger20 coolfunc soon')


def return_coolfunc(coolfunc_name):
    """
    This is the main function that will call the relevant interpolate_XXX() function from above 
    and return the coolfunc interpolator object to the parent module
    """
    
    if coolfunc_name == 'wiersma09': 
        coolfunc = interpolate_wiersma09()
        
    elif coolfunc_name == 'sd93': 
        coolfunc = interpolate_sd93()
        
    elif coolfunc_name == 'ploeckinger20':
        coolfunc = interpolate_ploeckinger20()
        
    else:
        raise ValueError('Must be one of wiersma09, sd93, or ploeckinger20')
    
    return coolfunc

# return_coolfunc('wiersma09')