# MARTA's VERSION!

"""
This module defines the ODE RHS function, initial conditions, and other associated specialized functions for the 
general purely thermal model that Viraj adapted for JAX. 

This also includes the ODE solver call including my early 2023 updates to logarithmic ODE solver. 

NOTE:   In the future I plan to move the ODE solver / numerics stuff to another module, and this module should just define
        a generic integrator function and initial condition function and any other associated specialized functions for this particular physical model.
        
        And possibly it may also make sense to have the user put the functional forms for any free parameters in here as well...

"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 
from scipy.integrate import simpson, solve_ivp, trapezoid
import scipy
import random

from astropy.io import fits
from scipy import interpolate
import os 

# define constants and unit conversions globally for this module 
const_mp = const.m_p.to('Msun').value 
const_mp_cgs = const.m_p.to('g').value 
cm_to_kpc = u.cm.to('kpc') # multiply something in cm by this to get to kpc 
kpc_to_cm = u.kpc.to('cm') # multiply something in kpc, it becomes units of cm
const_kB = const.k_B.to('erg/K').value # so that k*T = erg by default
yr_to_s = u.yr.to('s') # if you multiply something in yr by this, you get it in units of sec
s_to_yr = u.s.to('yr')
Msun_to_g = u.Msun.to('g') 
f_b = Planck15.Ob0 / Planck15.Om0 # ~15 %
G = const.G.to('cm**3 / (g * s**2)').value
c = const.c.to('cm/s').value

# # set the grid for BHAR-(M_star and redshift) dependence 
# z_bottom = 0.05
# z_top = 4.
# logmstar_bottom = 9.5
# logmstar_top = 12.
# gridsize_mstar = 50 # the M bin size is 49
# gridsize_z = 51 # the z bin size is 50
# logmgrid_bound = np.linspace(logmstar_bottom, logmstar_top, gridsize_mstar)
# logmgrid = (logmgrid_bound[:-1] + logmgrid_bound[1:]) / 2.
# log1pzgrid_bound = np.linspace(np.log10(1. + z_bottom), np.log10(1. + z_top), gridsize_z)
# log1pzgrid = (log1pzgrid_bound[:-1] + log1pzgrid_bound[1:]) / 2.

# # load the median logBHAR map
# # first get the absolute path to the wiersma09 data subdirectory
# path_abs = os.path.dirname(os.path.abspath(__file__)) # absolute path of the directory containing this file (read_coolfunc.py)
# path_bhar = os.path.join(path_abs,'main/maps_logbhar.fits') # absolute path to wiersma09 data subdirectory 
# medmap_logbhar = fits.open(path_bhar)[0].data
# interpfunc = interpolate.RegularGridInterpolator((logmgrid, log1pzgrid), medmap_logbhar.T, bounds_error = False, fill_value = None)

# define analytic functions to return n0 (density within Rvir) and T0 (temperature at Rvir) given state variables and assumed profile slopes
# NOTE: this is temporarily here -- eventually will be moved to the relevant subgrid_recipes CGM module 
def analytic_n0(Mcgm,Rvir,alpha_n):
    Rvir = Rvir * kpc_to_cm
    return (3+alpha_n)*Mcgm*Rvir**(alpha_n) / (4*np.pi * 0.59*const_mp * (Rvir**(3+alpha_n) - (0.1*Rvir)**(3+alpha_n))) # cm**-3

def analytic_T0(Ecgm,Rvir,n0,alpha_n,alpha_T):
    Rvir = Rvir * kpc_to_cm
    T0_est = (alpha_n+alpha_T+3)*Rvir**(alpha_n+alpha_T) * Ecgm / (6*np.pi*n0*const_kB*(Rvir**(alpha_n+alpha_T+3)-(0.1*Rvir)**(alpha_n+alpha_T+3))) # K
    return np.nan_to_num(T0_est,nan=1,posinf=1,neginf=1) # at early times when Mcgm=0, n0=0 so T0=nan; return 1K


"""
Set up the integrator function 
This should return the Mdot's at each timestep
"""

def evolve_galaxy(logt,logy,parameters,tree_interpolators,parameter_functions,uvb_model,coolfunc):
    
    # unpack the log10 state variables and raise to power 10 
    M_star = 10**logy[0]
    M_ism = 10**logy[1]
    M_cgm = 10**logy[2]
    Eth_cgm = 10**logy[3]
    MZ_star = 10**logy[4]
    MZ_ism = 10**logy[5]
    MZ_cgm = 10**logy[6]
    M_BH = 10**logy[7]    

    #adding bulge component
    # M_bulge_star = 10**logy[8]
    # M_bulge_gas = 10**logy[9]
    
    # also raise the log10(t/sec) to power of 10 and convert to Gyr for interpolation functions
    t_Gyr = 10**logt * s_to_yr * 1e-9 
    
    # unpack our SAM's free parameters
    alpha_n = parameters['alpha_n']
    alpha_T = parameters['alpha_T']
    tau_escape = parameters['tau_escape']
    f_recycle = parameters['f_recycle']
    e_wind_halo = parameters['e_wind_halo']
    yZ = parameters['yZ']
    etaM_A = parameters['etaM_A']
    etaM_alpha0 = parameters['etaM_alpha0']
    etaM_alphaz = parameters['etaM_alphaz']
    etaM_beta = parameters['etaM_beta']    
    tdep_A = parameters['tdep_A']
    tdep_alpha0 = parameters['tdep_alpha0']   
    tdep_alphaz = parameters['tdep_alphaz']   
    tdep_beta = parameters['tdep_beta']    
    etaE_A = parameters['etaE_A']
    etaE_alpha0 = parameters['etaE_alpha0']
    etaE_alphaz = parameters['etaE_alphaz']
    etaE_beta = parameters['etaE_beta']        
    etaZ_A = parameters['etaZ_A']
    etaZ_alpha0 = parameters['etaZ_alpha0']
    etaZ_alphaz = parameters['etaZ_alphaz']
    etaZ_beta = parameters['etaZ_beta']   
    f_bh = parameters['f_bh']
    kappa_bh = parameters['kappa_bh']    
    return_all = parameters['return_all']
    etaE_BH = parameters['etaE_BH']   # added by bry
    
    # unpack tree_interpolators (note the order of the returned list elements in read_trees/interpolate_trees.py module)
    interp_redshift, interp_logMAR, interp_logMvir, interp_logRvir, interp_logVvir, interp_logcNFW = tree_interpolators    
    
    # get Mdot_in_dm, Vvir and other quantities at this time using my interpolator functions 
    # I do .item() since scipy returns 0d array instead of scalar because of some convention ... 
    redshift = interp_redshift(t_Gyr).item() # dimensionless    
    Mdot_in_dm = 10**interp_logMAR(t_Gyr).item() # Msun/yr 
    Mvir = 10**interp_logMvir(t_Gyr).item() # Msun 
    Rvir = 10**interp_logRvir(t_Gyr).item() # proper kpc 
    Vvir = 10**interp_logVvir(t_Gyr).item() # proper km/s 
    NFW_c = 10**interp_logcNFW(t_Gyr).item() # dimensionless NFW halo concentration = Rvir/Rs_klypin 
    
    # unpack parameter_functions
    get_tdep, get_etaM_ism, get_etaE_ism, get_etaZ_ism, get_Zin_halo, get_fprev = parameter_functions     
    
    # unpack uvb_model 
    m_filt, collapse_fraction = uvb_model         
    
    # compute gas free-fall radius rff = radius of Vmax ~ 2.16*Rscale ~ 2.16 * Rvir/cNFW 
    rff = 2.16 * (Rvir / NFW_c) / Rvir # in units of Rvir
    
    # compute any other global halo quantities 
    Tvir = 35.9 * Vvir**2 # K 
    evir_halo = 3/2. * const_kB * Tvir / (0.59*const_mp) # erg/Msun  
    tdyn_halo = Rvir / (Vvir * 1e5*cm_to_kpc / (s_to_yr*1e-9)) # Gyr
    
    # analytic formula for NFW Vcirc(rff:=r/Rvir) rom Mo, van den Bosch & White eqn 11.26
    NFW_vcirc_num = np.log(1+NFW_c*rff) - NFW_c*rff/(1+NFW_c*rff)
    NFW_vcirc_den = rff*(np.log(1+NFW_c)-NFW_c/(1+NFW_c))
    NFW_vcirc = Vvir*np.sqrt(NFW_vcirc_num/NFW_vcirc_den) # km/s
    tdyn_ism = rff*Rvir / (NFW_vcirc * 1e5*cm_to_kpc / (s_to_yr*1e-9)) # Gyr
    
    # get SF and ISM wind scalings from FIRE 
    tdep_ism = get_tdep(Vvir,redshift)#,A=tdep_A,alpha0=tdep_alpha0,alphaz=tdep_alphaz,beta=tdep_beta) # Gyr 
    etaM_ism = get_etaM_ism(Vvir,redshift)#,A=etaM_A,alpha0=etaM_alpha0,alphaz=etaM_alphaz,beta=etaM_beta) # dimensionless
    etaE_ism = get_etaE_ism(Vvir,redshift)#,A=etaE_A,alpha0=etaE_alpha0,alphaz=etaE_alphaz,beta=etaE_beta) # dimensionless    
    etaZ_ism = get_etaZ_ism(Vvir,redshift)#,A=etaZ_A,alpha0=etaZ_alpha0,alphaz=etaZ_alphaz,beta=etaZ_beta) # dimensionless    

    # new FIRE Zdot/Mdot scalings for chemical evolution modeling
    Zin_halo = get_Zin_halo(Vvir,redshift) # dimensionless Zdot/Mdot (*not* normalized to Zsun)
    
    # compute log10 of CGM metallicity normalized to Zsun for cooling function
    log_Zcgm = np.log10(MZ_cgm / M_cgm / 0.02)
        
    ### compute Mdot_in_halo and Edot_in_halo terms 
    # first f_UV for UVB photoionization suppression of halo inflows
    Mfilt = m_filt(z=redshift)
    f_UV = collapse_fraction(Mvir,Mfilt)  
    
    # f_prev due to SN wind shock heating 
    # in the future will add zeta=Edot_out_halo_previous / (evir_halo*Mdot_in_halo) here 
    
    # assume this gas accretion brings in specific energy = evir_halo
    # f_prev will be computed and multiply both of these inflow rates AFTER we compute the halo energy outflow rate below
    Mdot_in_halo = f_UV * f_b * Mdot_in_dm # Msun/yr
    Edot_in_halo = Mdot_in_halo * evir_halo / yr_to_s # erg/s 
    
    ### compute Edot_cool and Mdot_cool terms
    # first get n0 for density profile and T0 for temperature profile (this can be done analytically)
    rarr = np.logspace(np.log10(0.1),np.log10(1.0),100)    

    n0_val = analytic_n0(M_cgm,Rvir,alpha_n) # cm**-3
    T0_val = analytic_T0(Eth_cgm,Rvir,n0_val,alpha_n,alpha_T) # K
    
    # now construct density and temperature profiles
    n_rarr = n0_val * rarr**alpha_n # cm**-3        
    T_rarr = T0_val * rarr**alpha_T # K

    # print(T0_val)

    # get cooling function value in each spherical shell
    #NOTE: this should be modularized depending on cooling function interpolator being used     
    halo_coolfunc = coolfunc([[redshift,
                               log_Zcgm,
                               np.log10(Tval),
                               np.log10(nval)] for (Tval,nval) in list(zip(T_rarr,n_rarr))]) # erg/s * cm**3  ###for Wiersma

    # halo_coolfunc = np.array([10**coolfunc([np.log10(Tval),log_Zcgm]).item() for Tval in T_rarr]) # erg/s * cm**3 ###for Sutherland & Dopita
    
    # compute # of shells with net cooling i.e., Lambda>0 just for bookkeeping
    ind_cool = np.where(halo_coolfunc>0)[0]     
    
    ### compute SFR term
    Mdot_sfr = M_ism / (tdep_ism*1e9) # Msun/yr

    ### compute Mdot_wind and Edot_wind terms
    Mdot_wind = etaM_ism * Mdot_sfr # Msun/yr
    
    # March 2023 -- switch from parameterized vB back to parameterized etaE for computing Edot_wind 
    Edot_wind = etaE_ism * ((1e51/100.) * (Mdot_sfr/yr_to_s)) # erg/s
        
    ####### At z>8.989, no cooling tables so Mdot_cool = 0 hence M_ism can become negative (SFR still gets calculated e Mism/tdep)
    ## set Mdot_sfr = 0 and Mdot_wind = 0 otherwise ODE solver will try to accommodate this with crazy small timesteps
    # another hack: also enforce this when Mism < 1e3 and Mdot_cool=0.0 which happens at early times in the dwarfs
    # if redshift > 8.989 or (M_ism < 1e3 and Mdot_cool == 0.0): 
    #     Mdot_sfr = 0.0
    #     Mdot_wind = 0.0
    #     etaE_ism = 0.0
    
    ############## BH accretion model: ################
    
    #choose quasar and radio accretion model
    quasar_accmodel = 'model_Y17'
    radio_accmodel = 'bondi_density_extrapolate'
    
    
    #Yang 2017 quasar accretion model
    if quasar_accmodel == 'model_Y17':
        # Sophie's model:
        Mdot_bh_quasar = 10**(0.22*np.log10(Mdot_sfr) + 1.16*np.log10(M_star) - 14.6)

    #maximum cooling flow radio accretion model
    if radio_accmodel == 'max_cool_flow':
        temp = T0_val
        cool_func = coolfunc([redshift,log_Zcgm,np.log10(T0_val),np.log10(n0_val*rff**alpha_n)])[0]
        if cool_func > 0:
            Mdot_bh_radio = kappa_bh * 15./16. * G * np.pi * const_mp_cgs * 0.59 * (const_kB * temp) / cool_func * M_BH * yr_to_s #Msun/yr
        else:
            Mdot_bh_radio = 0.0

    #bondi with extrapolating density to Bondi radius radio accretion model
    if radio_accmodel == 'bondi_density_extrapolate':
        temp = T0_val
        cs = np.sqrt((const_kB * temp) / (const_mp_cgs * 0.59)) #cm/s (speed of sound)
        r_bondi = 2.0 * G * M_BH * Msun_to_g / cs**2 #cm (bondi radius)
        n_bondi = n0_val * (r_bondi/(Rvir * kpc_to_cm))**alpha_n # cm**-3 (density at bondi radius)
        rho_bondi = n_bondi * const_mp_cgs * 0.59
        Mdot_bh_radio = 0.1 * np.pi * (G*M_BH * Msun_to_g)**2 * rho_bondi / cs**3 * yr_to_s / Msun_to_g
        # print("here")

    #bondi accretion with broken density law (density(Bondi)=density(ISM)) radio accretion model
    if radio_accmodel == 'bondi_density_flat':
        temp = T0_val
        cs = np.sqrt((const_kB * temp) / (const_mp_cgs * 0.59)) #cm/s (speed of sound)
        r_bondi = 2.0 * G * M_BH * Msun_to_g / cs**2 #cm (bondi radius)
        n_bondi = n0_val * 0.01**alpha_n # cm**-3 (density at ISM radius)
        rho_bondi = n_bondi * const_mp_cgs * 0.59
        Mdot_bh_radio = 4.0 * np.pi * (G*M_BH * Msun_to_g)**2 * rho_bondi / cs**3 * yr_to_s / Msun_to_g

    Mdot_bh = Mdot_bh_quasar + Mdot_bh_radio
    
    
    ############## BH feedback model (bry): ################
    
    Ledd = 1.26e38 * M_BH                     # erg/s, m_BH in Msun
    # eddington mass accretion rate of BHs
    eta = 0.1
    Mdot_bh_edd = (Ledd / (eta * c**2)) * (yr_to_s/Msun_to_g)        # Msun/yr
    f_edd = Mdot_bh / Mdot_bh_edd
    
    ### Select between 'model1' or 'model2' (others to be added)
    fbmodel = 'tng'
    
    # Model 1:
    if fbmodel == 'simple':
        Edot_bh = etaE_BH * Mdot_bh * (Msun_to_g/yr_to_s) * c**2
    
    # Model 2:
    if fbmodel == 'tng':
        # Replicating TNG Mbh-dependent threshold btwn thermal+kinetic mode
        chi_0 = 0.002
        Mpiv = 1e7 #Msun
        Beta = 2.0
        X_max = 0.1
        X_crit = min( chi_0*(M_BH/Mpiv)**Beta, X_max )        # m_BH in Msun

        L_bh_quasar = 0.0
        L_bh_radio = 0.0
        etaE_bh_jet = 0.0
        etaE_bh_wind = 0.0
        Edot_bh_quasar_fb = 0.0
        Mdot_bh_quasar_fb = 0.0
        Edot_bh_radio_fb = 0.0
        Mdot_bh_radio_fb = 0.0
        
        etaE_BH = 0.02

        # turn on kinetic winds if f_Edd dips below X_crit (TNG Mbh threshold for kinetic winds turning on)
        if (Mdot_bh > 0) & (f_edd < X_crit):
            ###############
            # try: 
            #Mdot_radioBH = 0. #Mdot_cool # == Mdot_cool from CGM to ISM
            Edot_bh_fb = etaE_BH * Mdot_bh * (Msun_to_g/yr_to_s)  * c**2
            Mdot_bh_fb = etaM_BH * Mdot_bh * (Msun_to_g/yr_to_s)  * c**2
            
        else:
            Edot_bh_fb = 0.0
            Mdot_bh_fb = 0.0
    
    if fbmodel == 'fiducial': #wind and jet in mass and energy + threshold
        # luminosity efficiency - assuming 0 for now for both modes.
        eps_wind = 0.0
        eps_jet = 0.0
        
        # luminosity of the disk from 2 modes
        L_bh_quasar = eps_wind * Mdot_bh_quasar * c**2 * (Msun_to_g/yr_to_s) #erg/s
        L_bh_radio = eps_jet * Mdot_bh_radio * c**2 * (Msun_to_g/yr_to_s) #erg/s
        L_bh = L_bh_quasar + L_bh_radio


        ##############################################
        #mass/energy feedback in wind mode
        
        # # Somerville SC-SAM version:
        # v_esc = (2 * G * M_star * Msun_to_g / (0.1*Rvir*kpc_to_cm))**0.5
        # Mdot_bh_quasar_fb = etaM_bh_wind * Mdot_bh_quasar * (eps_wind * c / v_esc)

        # Choi et al. version:
        esp_w = 5e-3
        v_w = 1e9
        # psi = 9.0
        psi = (2.0*eps_w*c**2)/(v_w**2) #link this to accretion rate
        etaE_bh_wind = 0.0
        
        Mdot_bh_quasar_fb = (psi/(1.0+psi)) * Mdot_bh_quasar
        Edot_bh_quasar_fb = eps_w * (1.0/(1.0+psi)) * Mdot_bh_quasar * c**2 * (Msun_to_g/yr_to_s) #erg/s


        ##############################################
        # mass/energy feedback in jet mode
        
        etaM_bh_jet = 0.0 # assuming no mass outflow from jets
        Mdot_bh_radio_fb = etaM_bh_jet * Mdot_bh_radio
        
        etaE_bh_jet =(10**(1.2*np.log10(M_BH) - 10.9))/3.
        Edot_bh_radio_fb = etaE_bh_jet * Mdot_bh_radio * c**2 * (Msun_to_g/yr_to_s) #erg/s

        # adding radio + quasar mode together
        Mdot_bh_fb = Mdot_bh_radio_fb + Mdot_bh_quasar_fb
        Edot_bh_fb = Edot_bh_radio_fb + Edot_bh_quasar_fb
        
    
    ############################################################

    # compute the Edot_cool integral, being careful with units (negative Edot_cool is OK, just means net heating)
    Edot_cool = trapezoid(y=4*np.pi*(rarr*Rvir*kpc_to_cm)**2*(n_rarr)**2*halo_coolfunc,x=rarr*Rvir*kpc_to_cm) # erg/s   

    ### a new piece of code (cooling energy is affected by feedback directly, which prevents accretion of cold gas)
    f_Mdotcool_prev = 0.1 #0.03  # ie, the fraction of Edot_bh_fb that prevents Mdot_cool. Maybe needs a better name...
    Edot_cool_eff = Edot_cool - f_Mdotcool_prev*Edot_bh_fb

    #print(Edot_cool, Edot_cool_eff, Eth_cgm/tdyn_halo)
  
    if Edot_cool_eff > Eth_cgm / (tdyn_halo * yr_to_s * 10**9): ### new rate limiter
        Edot_cool_eff = Eth_cgm / (tdyn_halo * yr_to_s * 10**9)
    
    # if Edot_cool<=0, set cooling/accretion timescales to infinity and Mdot_cool = 0
    if Edot_cool_eff <= 0:
        tcool_eff = np.inf
        tff_eff = np.inf
        t_accrete = np.inf        
        Mdot_cool = 0.0
    else: 
        # compute an effective tcool as Energy_cgm / Edot_cool
        tcool_eff = (Eth_cgm / Edot_cool_eff)*s_to_yr*1e-9 # Gyr
        
        # compute effective free-fall time at pre-defined radius rff 
        tff_eff = (rff*Rvir*kpc_to_cm*1e-5 / NFW_vcirc)*s_to_yr*1e-9 # Gyr        
        
        # compute Mdot_cool as M_cgm / t_accrete where t_accrete = tcool,eff + tff,eff
        t_accrete = tcool_eff + tff_eff # Gyr
        Mdot_cool = M_cgm / (t_accrete*1e9) # Msun/yr        

    # # #Viraj's addition (to fix sd93?)
    # e_wind_halo = T0_val / Tvir   
    
    ### compute Edot_out_halo and Mdot_out_halo terms 
    Ebind_cgm = evir_halo * M_cgm # erg
    Delta_Ecgm = Eth_cgm - Ebind_cgm # erg
    Edot_out_halo = np.max([0.0,Delta_Ecgm]) / (tau_escape*tdyn_halo*1e9*yr_to_s) # erg/s
    Mdot_out_halo = (Edot_out_halo/s_to_yr) / (e_wind_halo*(Eth_cgm/M_cgm)) # Msun/yr
    
    ### Go back and compute f_prev to reduce Mdot_in_halo and Edot_in_halo due to preventative feedback 
    # f_prev = np.min([1.0,kappa_heat * (Edot_out_halo / Edot_in_halo)])
    
    ### Nov 14 2022 -- parameterized fthermal_wind, fthermal_accretion, fprev
    f_prev = get_fprev(Vvir)#,redshift)
    
    # Carr+23/Pandya+20 style version where f_prev depends on ratio of Edot_out_halo and Edot_in_halo in the absence of preventative feedback
    # f_prev = np.min([alpha_prev * Edot_in_halo/Edot_out_halo, 1.0])
    
    Mdot_in_halo *= f_prev 
    Edot_in_halo *= f_prev 
        
    ##### New chemical evolution MZdot's 
    Zcgm = MZ_cgm / M_cgm # metal mass fraction of CGM (not normalized to solar and not log10)
    Zism = MZ_ism / M_ism # metal mass fraction of ISM     
    
    MZdot_sfr = Zism*(1.0-f_recycle)*Mdot_sfr # new long-lived stellar mass has same metallicity as ISM
    MZdot_cool = Zcgm*Mdot_cool # metal inflow rate from CGM to ISM
    MZdot_yield = yZ*Mdot_sfr # new metal mass produced by both short- and long-lived stars 
    MZdot_wind = Zism*Mdot_wind + etaZ_ism*MZdot_yield # metal outflow rate of ISM wind
    MZdot_in_halo = Zin_halo*Mdot_in_halo # metal inflow rate into CGM from cosmic accretion
    MZdot_out_halo = Zcgm*Mdot_out_halo # metal outflow rate from overpressurized halo
    
    ### Finally combine and return the total derivatives for the state variables (in the same order as y)
    # FIRST: since the time array (hence dt) will be in sec, convert all of these to Msun/s and erg/s
    
    #adding bulge component (subtract stars that go to bulge from disk)
    #HOWEVER!!! I'm not sure if stars should be subtracted here. Is Mdot_sfr_longlived for the whole galaxy? Or is it for disk only? We use it to get M_star, and then I use M_star as the mass of the stars in the disk only. So from the way I currently use it, it should be disk-only stars, and then we should be subtracting thouse stars that go to the bulge.
    Mdot_sfr_longlived = ((1.0-f_recycle)*Mdot_sfr) / yr_to_s # Msun/s
    #adding bulge component (subtract gas that goes to bulge from ISM)
    Mdot_ism = (Mdot_cool - (1.0-f_recycle)*Mdot_sfr - Mdot_wind - Mdot_bh_fb) / yr_to_s # Msun/s -
    Mdot_cgm = (Mdot_in_halo - Mdot_cool + Mdot_wind - Mdot_out_halo) / yr_to_s # Msun/s +
    Edot_cgm_th = Edot_in_halo - Edot_cool_eff + Edot_wind - Edot_out_halo + (1.0 - f_Mdotcool_prev) * Edot_bh_fb # erg/s
    MZdot_star = MZdot_sfr / yr_to_s # Msun/s
    MZdot_ism = (MZdot_cool + MZdot_yield - MZdot_sfr - MZdot_wind) / yr_to_s # Msun/s
    MZdot_cgm = (MZdot_in_halo - MZdot_cool + MZdot_wind - MZdot_out_halo) / yr_to_s # Msun/s
    Mbhdot = Mdot_bh / yr_to_s #Msun/s
    
    # SECOND: since we are integrating log(state_variable) AND log(time), multiply RHS by 10**t[sec] / 10**state_variable
    #         thus these all become dimensionless logarithmic derivatives dlogX/dlogt
    Mdot_sfr_longlived *= 10**logt / M_star 
    Mdot_ism *= 10**logt / M_ism 
    Mdot_cgm *= 10**logt / M_cgm
    Edot_cgm_th *= 10**logt / Eth_cgm
    MZdot_star *= 10**logt / MZ_star
    MZdot_ism *= 10**logt / MZ_ism
    MZdot_cgm *= 10**logt / MZ_cgm    
    Mbhdot *= 10**logt / M_BH    
    
    if return_all == False: # only return list of derivatives for solve_ivp
        
        return [Mdot_sfr_longlived, Mdot_ism, Mdot_cgm, Edot_cgm_th, MZdot_star, MZdot_ism, MZdot_cgm, Mbhdot] #adding bulge component

    elif return_all == True: # return a DICT of derivatives and all other properties for this time/state 

        # June 5, 2023: remove storing rarr and halo_coolfunc which are radial arrays instead of floats -- these making saving outputs harder
        return {'Mdot_sfr':Mdot_sfr,'Mdot_ism':Mdot_ism,'Mdot_cgm':Mdot_cgm,'Edot_cgm_th':Edot_cgm_th, 
                'redshift':redshift,'Mdot_in_dm':Mdot_in_dm,'Mvir':Mvir,'Rvir':Rvir,'Vvir':Vvir,'log_Zcgm':log_Zcgm,
                'NFW_c':NFW_c,'Tvir':Tvir,'evir_halo':evir_halo,'tdyn_halo':tdyn_halo,
                'tdyn_ism':tdyn_ism,'tdep_ism':tdep_ism,'etaM_ism':etaM_ism,'Mfilt':Mfilt,
                'f_UV':f_UV,'Mdot_in_halo':Mdot_in_halo,'Edot_in_halo':Edot_in_halo,
                'n0':n0_val,'T0':T0_val,'Ncoolingshells':len(ind_cool),
                'Edot_cool':Edot_cool,'Mdot_cool':Mdot_cool,'tcool_eff':tcool_eff,'tff_eff':tff_eff,
                't_accrete':t_accrete,'Mdot_wind':Mdot_wind,'Edot_wind':Edot_wind,'Ebind_cgm':Ebind_cgm,
                'Edot_out_halo':Edot_out_halo,'Mdot_out_halo':Mdot_out_halo,'cosmic_age':t_Gyr,
                'M_star':M_star,'M_ism':M_ism,'M_cgm':M_cgm,'Eth_cgm':Eth_cgm,'MZ_star':MZ_star,'MZ_ism':MZ_ism,'MZ_cgm':MZ_cgm,
                'Delta_Ecgm':Delta_Ecgm,'f_prev':f_prev,'NFW_vcirc':NFW_vcirc,'etaE_ism':etaE_ism,'etaZ_ism':etaZ_ism,
                'MZdot_star':MZdot_star,'MZdot_ism':MZdot_ism,'MZdot_cgm':MZdot_cgm,
                'MZdot_sfr':MZdot_sfr,'MZdot_yield':MZdot_yield,'MZdot_cool':MZdot_cool,'MZdot_wind':MZdot_wind,
                'MZdot_in_halo':MZdot_in_halo,'MZdot_out_halo':MZdot_out_halo,
                'Zcgm':Zcgm,'Zism':Zism,'Zin_halo':Zin_halo, 'Mbh_dot':Mbhdot*M_BH/10**logt,'M_BH':M_BH, 'etaE_BH':etaE_BH,
                'Mdot_bh_radio':Mdot_bh_radio, 'Mdot_bh_quasar':Mdot_bh_quasar, 
                'Mdot_bh_fb':Mdot_bh_fb, 'Edot_bh_fb':Edot_bh_fb, 'etaE_bh_jet':etaE_bh_jet, 'etaE_bh_wind':etaE_bh_wind,
                'L_bh_quasar':L_bh_quasar, 'L_bh_radio':L_bh_radio, 'Edot_bh_quasar':Edot_bh_quasar_fb, 'Edot_bh_radio':Edot_bh_radio_fb,
                'Mdot_bh':Mdot_bh, 'Mdot_bh_edd':Mdot_bh_edd, 'Mdot_bh_radio_fb':Mdot_bh_radio_fb, 'Mdot_bh_quasar_fb':Mdot_bh_quasar_fb,
                'f_edd':f_edd, 'Edot_cool_eff':Edot_cool_eff}    

        
        
# NOTE: this is a convenience function and should be moved to a utils module 
def combine_outputs(sol,parameters,inputs):
    """
    This function re-runs the integrator at each output solution time to store 
    ALL relevant properties during the integration and returns a combined dict
    
    inputs is the list of extra input args provided to integrator
    """    
    
    # first update parameters dict to return all requested properties computed by evolve_galaxy
    parameters.update(return_all=True)

    # store the output dict for each set of [time,state_variables] in the sol object
    out_dicts = [evolve_galaxy(sol.t[tsol],sol.y[:,tsol],parameters,*inputs) for tsol in range(len(sol.t))]

    # use dict comprehension to combine all dicts so I have a list of each property vs time
    # June 5, 2023: store as np.array instead of python list to ensure dtype and array manipulations and efficient saving
    comb_dict = {key: np.array([d.get(key) for d in out_dicts]) for key in set(out_dicts[0].keys())}

    # append a few other properties from sol object like number of RHS evaluations, etc.
    # June 5, 2023: to preserve same column shape as other time series columns, use np.array_like
    #               NOTE: this should instead be just saved as metadata somehow TBD 
    comb_dict['sol_nfev'] = np.full_like(comb_dict['cosmic_age'],sol.nfev) 
    comb_dict['sol_njev'] = np.full_like(comb_dict['cosmic_age'],sol.njev)
    comb_dict['sol_nlu'] = np.full_like(comb_dict['cosmic_age'],sol.nlu)
    comb_dict['sol_success'] = np.full_like(comb_dict['cosmic_age'],sol.success)
    comb_dict['sol_status'] = np.full_like(comb_dict['cosmic_age'],sol.status)
    
    # since dicts are mutable object, need to revert return_all to False else integrator will fail for next halo
    parameters.update(return_all=False) 
    
    return comb_dict 
    
               
def integrator(halo_data,parameters,parameter_functions,uvb_model,coolfunc,t_steps=1000):
    """
    This is the main function that initializes the model, calls the ODE integrator, and saves outputs. 
    
    halo_data is a tuple (halo_name, tree_interpolators) -- see driver.py for what the list tree_interpolators means
    
    return signature is always a dict with halo_name:result_dict key-value pair for this single halo
    
    NOTE: June 2023, I updated this for logarithmic ODE solver and to reflect my latest early 2023 initial conditions, etc.
    """
    
    halo_name, tree_interpolators = halo_data 
    interp_redshift, interp_logMAR, interp_logMvir, interp_logRvir, interp_logVvir, interp_logcNFW = tree_interpolators        
    
    # set up the timespan to integrate in seconds 
    # get the initial and final integration times from the get_knots() function for any of the interpolator functions
    # then sample t_steps number of points between t_init and t_final
    t_init = np.log10(tree_interpolators[0].get_knots()[0] * 1e9 * yr_to_s)
    t_final = np.log10(tree_interpolators[0].get_knots()[-1] * 1e9 * yr_to_s)
    t_eval = np.linspace(t_init, t_final, t_steps) 

    ### initial conditions for state variables -- LOGARITHMIC since we are integrating dlogX/dlogt 
    # order: Mstar [Msun], Mism [Msun], Mcgm [Msun], Eth_cgm [erg] MZ_star, MZ_ism, MZ_cgm 
    Mcgm_init = 0.5*f_b*10**interp_logMvir(10**t_init*s_to_yr*1e-9) # initial Mcgm (we don't have a model beyond z>8.98)
    Ecgm_init = (Mcgm_init*u.Msun * (interp_logVvir(10**t_init*s_to_yr*1e-9)*u.km/u.s)**2).to('erg').value 
    Mstar_init = 1. # arbitrary small number
    Mism_init = 1. # arbitrary small number
    Z_init = 1e-4 * 0.02 # assume initial metallicity is 1E-4/Zsun, multiply initial masses by this factor to get initial metal *mass*
    M_BH_init = 1.0e5   

    #adding_bulge_component
    # M_bulge_star_init = 1.
    # M_bulge_gas_init = 1.

    initial_conditions = np.log10([Mstar_init,Mism_init,Mcgm_init,Ecgm_init/2.,Z_init*Mstar_init,Z_init*Mism_init,Z_init*Mcgm_init, M_BH_init]) #adding bulge component

    #### tolerances
    # just choose reasonably small ones based on experimentation -- rtol gives requested error on exponent of variables, atol relevant for early evolution
    rtol = parameters['rtol'] 
    atol = parameters['atol'] 

    # call the integrator
    try: 
        sol = solve_ivp(evolve_galaxy,[t_init,t_final],initial_conditions,dense_output=True,method='BDF',
                        atol=atol,rtol=rtol,t_eval=t_eval,
                        args=(parameters,tree_interpolators,parameter_functions,uvb_model,coolfunc,))
    except: # if solve_ivp itself just exits with error, return normal dict structure for this halo but with sol_success = False
        print('<!----- WARNING: solve_ivp exited with error for halo_name=%s'%halo_name)
        # June 5, 2023: return time series of sol_success=False to enable dict comprehension later to filter these bad sols out
        return {str(halo_name):{'sol_success':np.full(t_steps,False),'sol_object':None}} # no sol object since solve_ivp exited with error

    
    # print warning if integrator failed for this halo and return the normal dict structure but with sol_success = False
    if sol.status == -1:
        print('<!----- WARNING: solve_ivp completed BUT sol.status code = -1 (failed to integrate) for halo_name=%s'%halo_name)
        # June 5, 2023: return time series of sol_success=False to enable dict comprehension later to filter these bad sols out        
        return {str(halo_name):{'sol_success':np.full(t_steps,False),'sol_object':sol}} # returning sol object if helpful for debugging
    # otherwise call my utility script to merge ALL outputs of ODE function and return normal dict structure for this halo
    else: 
        comb_dict = combine_outputs(sol,parameters,[tree_interpolators,parameter_functions,uvb_model,coolfunc])
        return {str(halo_name):comb_dict} 
    