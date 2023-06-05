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
from scipy.integrate import simps, solve_ivp, trapz

# define constants and unit conversions globally for this module 
const_mp = const.m_p.to('Msun').value 
cm_to_kpc = u.cm.to('kpc') # multiply something in cm by this to get to kpc 
kpc_to_cm = u.kpc.to('cm') # multiply something in kpc, it becomes units of cm
const_kB = const.k_B.to('erg/K').value # so that k*T = erg by default
yr_to_s = u.yr.to('s') # if you multiply something in yr by this, you get it in units of sec
s_to_yr = u.s.to('yr')
Msun_to_g = u.Msun.to('g') 
f_b = Planck15.Ob0 / Planck15.Om0 # ~15 %
G = const.G.to('cm**3 / (g * s**2)').value

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
    return_all = parameters['return_all']
    
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
    tdep_ism = get_tdep(Vvir,redshift,A=tdep_A,alpha0=tdep_alpha0,alphaz=tdep_alphaz,beta=tdep_beta) # Gyr 
    etaM_ism = get_etaM_ism(Vvir,redshift,A=etaM_A,alpha0=etaM_alpha0,alphaz=etaM_alphaz,beta=etaM_beta) # dimensionless
    etaE_ism = get_etaE_ism(Vvir,redshift,A=etaE_A,alpha0=etaE_alpha0,alphaz=etaE_alphaz,beta=etaE_beta) # dimensionless    
    etaZ_ism = get_etaZ_ism(Vvir,redshift,A=etaZ_A,alpha0=etaZ_alpha0,alphaz=etaZ_alphaz,beta=etaZ_beta) # dimensionless    

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
    
    # get cooling function value in each spherical shell
    # NOTE: this should be modularized depending on cooling function interpolator being used     
    halo_coolfunc = coolfunc([[redshift,
                               log_Zcgm,
                               np.log10(Tval),
                               np.log10(nval)] for (Tval,nval) in list(zip(T_rarr,n_rarr))]) # erg/s * cm**3

    # compute # of shells with net cooling i.e., Lambda>0 just for bookkeeping
    ind_cool = np.where(halo_coolfunc>0)[0]     
    
    # compute the Edot_cool integral, being careful with units (negative Edot_cool is OK, just means net heating)
    Edot_cool = trapz(y=4*np.pi*(rarr*Rvir*kpc_to_cm)**2*(n_rarr)**2*halo_coolfunc,x=rarr*Rvir*kpc_to_cm) # erg/s    
    
    # ### Prevent stiff early blow-up -- no net heating: 
    # if Edot_cool < 0:
    #     Edot_cool = 0.0
    
    # if Edot_cool<=0, set cooling/accretion timescales to infinity and Mdot_cool = 0
    if Edot_cool <= 0:
        tcool_eff = np.inf
        tff_eff = np.inf
        t_accrete = np.inf        
        Mdot_cool = 0.0
    else: 
        # compute an effective tcool as Energy_cgm / Edot_cool
        tcool_eff = (Eth_cgm / Edot_cool)*s_to_yr*1e-9 # Gyr    
        
        # compute effective free-fall time at pre-defined radius rff 
        tff_eff = (rff*Rvir*kpc_to_cm*1e-5 / NFW_vcirc)*s_to_yr*1e-9 # Gyr        
        
        # compute Mdot_cool as M_cgm / t_accrete where t_accrete = tcool,eff + tff,eff
        t_accrete = tcool_eff + tff_eff # Gyr
        Mdot_cool = M_cgm / (t_accrete*1e9) # Msun/yr        
    
    
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
    Mdot_sfr_longlived = (1.0-f_recycle)*Mdot_sfr / yr_to_s # Msun/s
    Mdot_ism = (Mdot_cool - (1.0-f_recycle)*Mdot_sfr - Mdot_wind) / yr_to_s # Msun/s 
    Mdot_cgm = (Mdot_in_halo - Mdot_cool + Mdot_wind - Mdot_out_halo) / yr_to_s # Msun/s 
    Edot_cgm_th = Edot_in_halo - Edot_cool + Edot_wind - Edot_out_halo # erg/s
    MZdot_star = MZdot_sfr / yr_to_s # Msun/s
    MZdot_ism = (MZdot_cool + MZdot_yield - MZdot_sfr - MZdot_wind) / yr_to_s # Msun/s
    MZdot_cgm = (MZdot_in_halo - MZdot_cool + MZdot_wind - MZdot_out_halo) / yr_to_s # Msun/s
    
    # SECOND: since we are integrating log(state_variable) AND log(time), multiply RHS by 10**t[sec] / 10**state_variable
    #         thus these all become dimensionless logarithmic derivatives dlogX/dlogt
    Mdot_sfr_longlived *= 10**logt / M_star 
    Mdot_ism *= 10**logt / M_ism 
    Mdot_cgm *= 10**logt / M_cgm
    Edot_cgm_th *= 10**logt / Eth_cgm
    MZdot_star *= 10**logt / MZ_star
    MZdot_ism *= 10**logt / MZ_ism
    MZdot_cgm *= 10**logt / MZ_cgm    
    
    if return_all == False: # only return list of derivatives for solve_ivp
        
        return [Mdot_sfr_longlived, Mdot_ism, Mdot_cgm, Edot_cgm_th, MZdot_star, MZdot_ism, MZdot_cgm]

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
                'Zcgm':Zcgm,'Zism':Zism,'Zin_halo':Zin_halo}    

        
        
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

    initial_conditions = np.log10([Mstar_init,Mism_init,Mcgm_init,Ecgm_init/2.,Z_init*Mstar_init,Z_init*Mism_init,Z_init*Mcgm_init])

    #### tolerances
    # just choose reasonably small ones based on experimentation -- rtol gives requested error on exponent of variables, atol relevant for early evolution
    rtol = 1e-5
    atol = 1e-3

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
    