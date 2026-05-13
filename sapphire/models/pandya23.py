"""
This module defines the ODE RHS function, initial conditions, and other associated specialized functions for the 
general purely thermal model that Viraj adapted for JAX. 

This also includes the ODE solver call including my early 2023 updates to logarithmic ODE solver.

NOTE:   In the future I plan to move the ODE solver / numerics stuff to another module, and this module should just define
        a generic integrator function and initial condition function and any other associated specialized functions for this particular physical model.
        
        And possibly it may also make sense to have the user put the functional forms for any free parameters in here as well...

"""

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 
from functools import partial 

# in case user loads module separately from sapphire.run()
from jax import config as jax_config
jax_config.update("jax_enable_x64", True) # required to accurately solve and take gradients through our diffeqs 

import jax.numpy as jnp
from jax._src.third_party.scipy.interpolate import RegularGridInterpolator as jax_RegularGridInterpolator
from jax import jit, grad, vmap, pmap, debug, jvp, vjp, jacrev, jacfwd, make_jaxpr, hessian, value_and_grad
# from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline    
from jax.experimental.ode import odeint
from jax.lax import fori_loop, while_loop
from jax.scipy.integrate import trapezoid
from jax.random import PRNGKey, key    
from diffrax import diffeqsolve, ODETerm, PIDController, SaveAt, Kvaerno3, Bosh3, Dopri5, Tsit5, DirectAdjoint, RecursiveCheckpointAdjoint, BacksolveAdjoint
from diffrax import backward_hermite_coefficients, CubicInterpolation    
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from interpax import Interpolator1D


""" Nov 4, 2025 -- wrap the entire thing, later will refactor to use classes with equinox """

def setup(config):

    ### Feb 16, 2026 -- for new forecasting tests 
    if 'clip_etaE_max' not in config.keys():
        config['clip_etaE_max'] = 1.0 # clip etaE at 1.0, unless higher max requested

    # set global fixed parameters 
    params_fixed_astro = config['params_fixed_astro']
    alpha_n = params_fixed_astro['alpha_n'] # slope of CGM density power law 
    alpha_T = params_fixed_astro['alpha_T'] # slope of CGM temperature power law 
    f_recycle = params_fixed_astro['f_recycle'] # instantaneous stellar-->ISM recycling fraction 
    yZ = params_fixed_astro['yZ'] # metal yield of 1 SN per 100 Msun of stars formed (2 Msun / 100 Msun = 0.02 for 10 Msun of SN ejecta) 
    A_Zin_halo = params_fixed_astro['A_Zin_halo'] # pre-enrichment metallicity that multiplies Mdot_in_halo, default 0.25 from Pandya+23 

    
    ### define constants and conversions 
    const_mp = const.m_p.to('Msun').value 
    cm_to_kpc = u.cm.to('kpc') # multiply something in cm by this to get to kpc 
    kpc_to_cm = u.kpc.to('cm') # multiply something in kpc, it becomes units of cm
    const_kB = const.k_B.to('erg/K').value # so that k*T = erg by default
    yr_to_s = u.yr.to('s') # if you multiply something in yr by this, you get it in units of sec
    s_to_yr = u.s.to('yr')
    Msun_to_g = u.Msun.to('g') 
    f_b = Planck15.Ob0 / Planck15.Om0 # ~15 %
    littleh = Planck15.h
    Om0 = Planck15.Om0
    G = const.G.to('cm**3 / (g * s**2)').value
    t0 = Planck15.age(0).value 
    
    ### jax-cosmo does not have rho_crit(z) so for now just interpolate (later can implement)
    # interpolate cosmology functions since its easier than analytic ones for now
    zarr = jnp.linspace(10.0,0.0,100)
    tarr = Planck15.age(zarr).value # Gyr
    rhocrit_arr = jnp.asarray(Planck15.critical_density(zarr).value)
    Om_arr = jnp.asarray(Planck15.Om(zarr))
    
    # interp_rhocrit_z = jit(InterpolatedUnivariateSpline(jnp.asarray(zarr),rhocrit_arr,k=3))
    # interp_Om_z = jit(InterpolatedUnivariateSpline(jnp.asarray(zarr),Om_arr,k=3))
    # interp_redshift = jit(InterpolatedUnivariateSpline(jnp.asarray(tarr),jnp.asarray(zarr),k=3))

    ### March 2026 -- note that interpax returns a function, unlike above, so need to do .__call__
    interp_rhocrit_z = jit(Interpolator1D(zarr[::-1],rhocrit_arr[::-1], method="cubic2",
                                          extrap=(rhocrit_arr[-1], rhocrit_arr[0])).__call__)
    
    interp_Om_z = jit(Interpolator1D(zarr[::-1],Om_arr[::-1], method="cubic2",
                                     extrap=(Om_arr[-1], Om_arr[0])).__call__)
    
    interp_redshift = jit(Interpolator1D(tarr,jnp.asarray(zarr), method="cubic2",
                                         extrap=(zarr[0], zarr[-1])).__call__)    
    
    ### coolfunc
    from sapphire.utils import read_coolfunc
    coolfunc = read_coolfunc.get(config) 
    
    @jit
    def get_tdep(Vvir,redshift,A=3.0,alpha0=-3.0,alphaz=0.0,beta=-0.7): 
        """
        Returns ISM gas depletion time in Gyr for halo with Vvir [km/s] and redshift [dimensionless]
        Power law whose normalization and slope both can depend on redshift
        """
        alpha = alpha0 + alphaz*(1+redshift)
        return A * (Vvir/125.)**alpha * (1+redshift)**beta
    
    @jit
    def get_etaM_ism(Vvir,redshift,A=1.0,alpha0=-3.7,alphaz=0.0,beta=2.4):
        """
        Returns ISM mass loading factor [dimensionless] for halo with Vvir [km/s] and redshift [dimensionless]
        Power law whose normalization and slope both can depend on redshift
        """    
        alpha = alpha0 + alphaz*(1+redshift)
        return A * (Vvir/125.)**alpha * (1+redshift)**beta
    
    @jit
    def get_etaE_ism(Vvir,redshift,A=0.1,alpha0=-0.5,alphaz=0.0,beta=0.0): 
        """
        Returns ISM energy loading factor [dimensionless] for halo with Vvir [km/s] and redshift [dimensionless]
        Power law whose normalization and slope can both depend on redshift
        
        We don't want etaE to exceed 1 (which it might for certain slopes and normalizations) so we use jnp.minimum
        """
        alpha = alpha0 + alphaz*(1+redshift)
        # Aug 14, 2024: jnp.max -> jnp.clip
        # Feb 16, 2026: allow this to exceed one for tests
        return jnp.clip(A * (Vvir/125.)**alpha * (1+redshift)**beta,None,config['clip_etaE_max']) 
    
    @jit 
    def get_etaZ_ism(Vvir,redshift,A=0.5,alpha0=0.0,alphaz=0.0,beta=0.0): # this is the Carr/Bryan approach for increasing Zwind > Zism 
        """
        Returns ISM metal loading factor [dimensionless] for halo with Vvir [km/s] and redshift [dimensionless] 
        Power law whose normalization and slope can both depend on redshift
        
        We don't want etaZ to exceed 1 (which it might for certain slopes and normalizations) so we use jnp.minimum
        """
        
        alpha = alpha0 + alphaz*(1+redshift)
        return jnp.clip(A * (Vvir/125.)**alpha * (1+redshift)**beta,None,1.0)     # Aug 14, 2024 - jnp.max -> jnp.clip
    
    @jit
    def get_Zin_halo(Vvir,redshift,A=0.25,alpha0=2.2,alphaz=0.0,beta=-1.3):
        """
        Returns metal mass fraction of inflowing gas at Rvir for halo with Vvir [km/s] and redshift [dimensionless]
        Power law whose normalization and slope both can depend on redshift
        NOTE: we multiply the return value by 0.02 since the fiducial FIRE-2 calibrated parameters give Zin_halo in units of Zsun=0.02
        """      
        alpha = alpha0 + alphaz*(1+redshift)
        return 0.02 * A * (Vvir/125.)**alpha * (1+redshift)**beta
    
    @jit
    def get_fprev(Vvir,fmin=0.25,fmax=0.9641,fscale=-7.1348,fpivot=1.8775):
        """
        Returns halo gas accretion suppression factor [dimensionless] for halo with Vvir [km/s]
        Generalized logistic function
        """      
        return fmin + (fmax - fmin) / (1.0 + jnp.exp(fscale*(jnp.log10(Vvir) - fpivot)))    
    
    ### Jan 6, 2026 -- put the parameterization functions into a list for visualization purposes
    ### TO DO: this will be improved/generalized later during viraj's equinox refactor
    list_parameterizations = [get_tdep,get_etaM_ism,get_etaE_ism,get_etaZ_ism]
    
    # unclear whether jit is necessary for these -- they're basically pure float operations, not jax arrays
    # might be useful for GPUs -- can benchmark later
    
    def analytic_n0(Mcgm,Rvir,alpha_n):
        """
        Returns the number density [cm**-3] within Rvir [kpc] given CGM mass [Msun] and density profile slope [dimensionless]
        """
        Rvir = Rvir * kpc_to_cm
        return (3+alpha_n)*Mcgm*Rvir**(alpha_n) / (4*jnp.pi * 0.59*const_mp * (Rvir**(3+alpha_n) - (0.1*Rvir)**(3+alpha_n))) # cm**-3
    
    
    def analytic_T0(Ecgm,Rvir,n0,alpha_n,alpha_T):
        """
        Returns the temperature [K] at Rvir [kpc] given CGM thermal energy [erg], density within Rvir [cm**-3'] and slopes of density and temperature profiles
        """    
        Rvir = Rvir * kpc_to_cm
        T0_est = (alpha_n+alpha_T+3)*Rvir**(alpha_n+alpha_T) * Ecgm / (6*jnp.pi*n0*const_kB*(Rvir**(alpha_n+alpha_T+3)-(0.1*Rvir)**(alpha_n+alpha_T+3))) # K
        return T0_est
        # return jnp.nan_to_num(T0_est,nan=1,posinf=1,neginf=1) 
    
    
    @partial(jit,static_argnums=(1,2,3))
    def m_filt(z, z_overlap=9.0, z_reionize=3.5, z_squelch=8.0):
        """
        Computes the "Filtering Mass" at which halos accrete only half of the incoming baryon fraction fb*Mdot_vir (Gnedin+00)
        Follows Rachel's implementation of Kravtsov+04 appendix B
        z1=z_overlap, z2=z_reionize, z_squelch = when suppression ('squelching') turned on
        Rachel tuned these parameters to reproduce Okamoto+08, so do not over-interpret parameter names or modify values
        NOTE: Must re-write so parameters reflect their intended meaning (eg z_reionize should be >> 3.5) and to switch to more modern replacement for Okamoto+08
        """
        
        # this constant ensures m_filt defined in this way matches the normalization and definition of the Okamoto+08 "M_characteristic" variable 
        Mfilt_factor = 0.0933 
        
        # Jeans mass in Msun; 0.59 is mean molecular weight
        # note that we are multiplying this by Mfilt_factor here instead of in the return statement below
        m_jeans = Mfilt_factor * 2.5E11 * littleh**-1 * jnp.sqrt(Om0)**-1 * (0.59)**-1.5 
        
        # power that controls growth rate of UV background flux (Kravtsov+04 fixed alpha=6)    
        alpha = 6.0 
        
        # convert the input redshifts into scale factors 
        a_o = 1.0/(1+z_overlap) # scale factor where multiple HII regions BEGIN TO overlap 
        a_r = 1.0/(1+z_reionize) # scale factor of complete reionization 
        a = 1.0 / (1+z) # current scale factor 
        
        # return m_filt = m_jeans * f_a**1.5 where f_a is computed differently for a<a_o, a_o<a<a_r, or a>a_r
        # use jax differentiable control flow function to return one of three expressions (instead of non-differentiable if/else statements)
        return jnp.select([a < a_o, (a >= a_o) & (a <= a_r), a > a_r],
                          [m_jeans*jnp.array((3.0 * a / ((2.0+alpha)*(5.0+2.0*alpha))) * (a/a_o)**alpha)**1.5,
                           m_jeans*jnp.array((3.0/a)*a_o**2*(1.0/(2+alpha) - (2*(a/a_o)**-0.5)/(5+2*alpha)) + a**2/10. - (a_o**2/10.)*(5-4*(a/a_o)**-0.5))**1.5,
                           m_jeans*jnp.array((3.0/a)*(a_o**2*(1/(2+alpha) - 2*(a/a_o)**-0.5/(5+2*alpha))
                                    +a_r**2/10.*(5-4*(a/a_r)**-0.5)
                                    -a_o**2/10.*(5-4*(a/a_o)**-0.5)
                                    +a*a_r/3. - a_r**2/3.*(3-2*(a/a_r)**-0.5)))**1.5],
                          default=0.0) # we don't expect all 3 conditionals to ever be false, but keep 0 as the default value in this case
        
    
    # unclear whether jit is necessary for this -- it's basically a pure float operation, not jax arrays
    def collapse_fraction(m,mfilt):
        """
        Computes the gas accretion suppression factor f_UV that depends on the halo mass relative to filtering mass 
        Halos with Mvir >> Mfilt will have f_UV ~ 1 (no suppression), whereas much lower mass halos tend to f_UV ~ 0
        This is eqn (1) of Okamoto+08 who studied the UV background in hydro simulations
        NOTE: I should update this to a more modern function based on Faucher-Giguere+18 or something
        """
        
        alpha = 2.0
        
        fcoll = (1+(2**(alpha/3.)-1.)*(m/mfilt)**-alpha)**(-3/alpha)
        
        return fcoll
    
    
    
    # global quantities that do not change within integrator are defined once here to prevent jit overhead
    rarr = jnp.logspace(jnp.log10(0.1),jnp.log10(1.0),100)  
    
    @jit
    def integrator(logt,logy,args): 
        """
        args = (parameters, interps) which are both lists
        """
        
        # raise the log10(t/sec) to power of 10 and convert to Gyr for interpolation functions
        t_Gyr = 10**logt * s_to_yr * 1e-9    
        
        # unpack the log10 state variables and raise to power 10 
        M_star = 10**logy[0]
        M_ism = 10**logy[1]
        M_cgm = 10**logy[2]
        Eth_cgm = 10**logy[3]
        MZ_star = 10**logy[4]
        MZ_ism = 10**logy[5]
        MZ_cgm = 10**logy[6]
        
        # unpack our model's parameters dict 
        # etaM_A = parameters['etaM_A']
        # etaM_alpha0 = parameters['etaM_alpha0']
        # etaM_alphaz = parameters['etaM_alphaz']
        # etaM_beta = parameters['etaM_beta']
        # etaE_A = parameters['etaE_A']
        # etaE_alpha0 = parameters['etaE_alpha0']
        # etaE_alphaz = parameters['etaE_alphaz']
        # etaE_beta = parameters['etaE_beta']    
        # tdep_A = parameters['tdep_A']
        # tdep_alpha0 = parameters['tdep_alpha0']
        # tdep_alphaz = parameters['tdep_alphaz']
        # tdep_beta = parameters['tdep_beta']   
        # etaZ_A = parameters['etaZ_A']
        # etaZ_alpha0 = parameters['etaZ_alpha0']
        # etaZ_alphaz = parameters['etaZ_alphaz']
        # etaZ_beta = parameters['etaZ_beta']    
        
        ### unpack parameters and interps lists
        parameters, interps = args
        
        (etaM_A,etaM_alpha0,etaM_alphaz,etaM_beta,etaE_A,etaE_alpha0,etaE_alphaz,etaE_beta,
         tdep_A,tdep_alpha0,tdep_alphaz,tdep_beta,etaZ_A,etaZ_alpha0,etaZ_alphaz,etaZ_beta,
         realization) = parameters 
    
        # use global interp_redshift to go from current time to redshift
        redshift = interp_redshift(t_Gyr) # dimensionless    
    
        # unpack cubic hermite interpolators for current halo, and evaluate at current time in physical units
        interp_logmdot, interp_logmvir, interp_logrvir, interp_logvvir, interp_logcnfw = interps
        Mdot_in_dm = 10**interp_logmdot.evaluate(t_Gyr) # Msun/yr
        Mvir = 10**interp_logmvir.evaluate(t_Gyr) # Msun
        Rvir = 10**interp_logrvir.evaluate(t_Gyr) # proper kpc
        Vvir = 10**interp_logvvir.evaluate(t_Gyr) # proper km/s
        NFW_c = 10**interp_logcnfw.evaluate(t_Gyr) # dimensionless NFW halo concentration = Rvir/rscale
        
        
        # debug.print('{logt}, {z}, {M0}, {alpha_early}, {alpha_late}, {k_mah}, {tau_mah}, {cmin}, {beta_late}, {tau_prof}, {k_prof}',logt=logt,z=redshift,
        #             M0=M0,alpha_early=alpha_early,alpha_late=alpha_late,k_mah=k_mah,tau_mah=tau_mah,cmin=cmin,beta_late=beta_late,tau_prof=tau_prof,k_prof=k_prof)        
        
        # Mvir = Mpeak(t_Gyr,M0=M0,alpha_early=alpha_early,alpha_late=alpha_late,k=k_mah,tau_c=tau_mah) # Msun
        # Mdot_in_dm = diffmah(t_Gyr,M0=M0,alpha_early=alpha_early,alpha_late=alpha_late,k=k_mah,tau_c=tau_mah) # Msun/yr
        # Rvir = Mpeak_to_Rvir(Mvir,redshift) # proper kpc [ requires differentiable Planck15.Om(z) and Planck15.critical_density(z)
        # Vvir = jnp.sqrt(G * Mvir*Msun_to_g / (Rvir*kpc_to_cm)) * 1e-5 # proper km/s
        # NFW_c = diffprof(t_Gyr,cmin=cmin,beta_late=beta_late,tau_c=tau_prof,k=k_prof) # dimensionless NFW halo concentration = Rvir/rscale
        
        # compute gas free-fall radius rff = radius of Vmax ~ 2.16*Rscale ~ 2.16 * Rvir/cNFW 
        rff = 2.16 * (Rvir / NFW_c) / Rvir # in units of Rvir
        
        # compute any other global halo quantities 
        Tvir = 35.9 * Vvir**2 # K 
        evir_halo = 3/2. * const_kB * Tvir / (0.59*const_mp) # erg/Msun  
        tdyn_halo = Rvir / (Vvir * 1e5*cm_to_kpc / (s_to_yr*1e-9)) # Gyr
        
        # analytic formula for NFW Vcirc(rff:=r/Rvir) rom Mo, van den Bosch & White eqn 11.26
        NFW_vcirc_num = jnp.log(1+NFW_c*rff) - NFW_c*rff/(1+NFW_c*rff)
        NFW_vcirc_den = rff*(jnp.log(1+NFW_c)-NFW_c/(1+NFW_c))
        NFW_vcirc = Vvir*jnp.sqrt(NFW_vcirc_num/NFW_vcirc_den) # km/s
        tdyn_ism = rff*Rvir / (NFW_vcirc * 1e5*cm_to_kpc / (s_to_yr*1e-9)) # Gyr
        
        # get SF and ISM wind scalings from FIRE 
        tdep_ism = get_tdep(Vvir,redshift,A=tdep_A,alpha0=tdep_alpha0,alphaz=tdep_alphaz,beta=tdep_beta) # Gyr 
        etaM_ism = get_etaM_ism(Vvir,redshift,A=etaM_A,alpha0=etaM_alpha0,alphaz=etaM_alphaz,beta=etaM_beta) # dimensionless 
        etaE_ism = get_etaE_ism(Vvir,redshift,A=etaE_A,alpha0=etaE_alpha0,alphaz=etaE_alphaz,beta=etaE_beta) # dimensionless    
        etaZ_ism = get_etaZ_ism(Vvir,redshift,A=etaZ_A,alpha0=etaZ_alpha0,alphaz=etaZ_alphaz,beta=etaZ_beta) # dimensionless        
    
        # get halo inflow metallicity from FIRE scaling because we don't yet have outer halo wind recycling
        # dimensionless Zdot/Mdot (*not* normalized to Zsun)
        Zin_halo = get_Zin_halo(Vvir,redshift,A=A_Zin_halo) 
        
        # compute log10 of CGM metallicity normalized to Zsun for cooling function
        log_Zcgm = jnp.log10(MZ_cgm / M_cgm / 0.02)
            
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
       
        n0_val = analytic_n0(M_cgm,Rvir,alpha_n) # cm**-3
        T0_val = analytic_T0(Eth_cgm,Rvir,n0_val,alpha_n,alpha_T) # K
        
        # now construct density and temperature profiles
        n_rarr = n0_val * rarr**alpha_n # cm**-3        
        T_rarr = T0_val * rarr**alpha_T # K
        
        # get cooling function value in each spherical shell
        # halo_coolfunc = jit_coolfunc_W09(jnp.array([[redshift,log_Zcgm,jnp.log10(Tval),jnp.log10(nval)] 
        #                                             for (Tval,nval) in list(zip(T_rarr,n_rarr))])) # erg/s * cm**3
        halo_coolfunc = 10**coolfunc((jnp.log10(T0_val),log_Zcgm)) # erg s**-1 cm**3
        
        # compute the Edot_cool integral, being careful with units (negative Edot_cool is OK, just means net heating)
        # Edot_cool = jnp.trapz(y=4*jnp.pi*(rarr*Rvir*kpc_to_cm)**2*(n_rarr)**2*halo_coolfunc,x=rarr*Rvir*kpc_to_cm) # erg/s
        # Edot_cool = trapezoid(y=4*jnp.pi*(rarr*Rvir*kpc_to_cm)**2*(n_rarr)**2*halo_coolfunc,x=rarr*Rvir*kpc_to_cm) # erg/s    
        
        # March 2024 -- switch to CIE SD93 means: (1) Edot_cool is analytic , (2) need to impose catastrophic cooling rate limiter
        Edot_cool = 4*jnp.pi*n0_val**2 * (Rvir*kpc_to_cm)**3 * halo_coolfunc * jnp.log(10) # erg/s
        Edot_cool = jnp.min(jnp.array([Edot_cool, Eth_cgm/(tdyn_halo*1e9*yr_to_s)])) # erg/s 
        
        # use differentiable jnp.select to return tcool_eff and tff_eff depending on if there is net heating (Edot_cool<0)
        # if there is net heating, tcool_eff and tff_eff are both inf, and Mcgm/(inf+inf) = 0.0 as expected 
        # otherwise tcool_eff and tff_eff are returned in Gyr 
        # WARNING: we need to decide on rff
        # tcool_eff, tff_eff = jnp.select([Edot_cool <= 0, Edot_cool > 0],
        #                                 [jnp.array([jnp.inf,jnp.inf]),
        #                                  jnp.array([(Eth_cgm / Edot_cool)*s_to_yr*1e-9,
        #                                             (rff*Rvir*kpc_to_cm*1e-5 / NFW_vcirc)*s_to_yr*1e-9])])
        
        # March 2024 -- with SD93 CIE coolfunc, Edot_cool can never be zero so remove overhead of jnp.select
        tcool_eff = (Eth_cgm / Edot_cool)*s_to_yr*1e-9
        tff_eff = (rff*Rvir*kpc_to_cm*1e-5 / NFW_vcirc)*s_to_yr*1e-9
        t_accrete = tcool_eff + tff_eff 
        Mdot_cool = M_cgm / (t_accrete*1e9) # Msun/yr 
        
        # debug.print('n0_val={n0_val},T0_val={T0_val},log_Zcgm={log_Zcgm},Edot_cool={Edot_cool},tcool_eff={tcool_eff},tff_eff={tff_eff},Mdot_cool={Mdot_cool},halo_coolfunc={halo_coolfunc}',
        #             n0_val=n0_val,T0_val=T0_val,log_Zcgm=log_Zcgm,Edot_cool=Edot_cool,tcool_eff=tcool_eff,tff_eff=tff_eff,Mdot_cool=Mdot_cool,halo_coolfunc=halo_coolfunc)      
        
        ### compute SFR term
        Mdot_sfr = M_ism / (tdep_ism*1e9) # Msun/yr
        
        ### compute Mdot_wind and Edot_wind terms
        Mdot_wind = etaM_ism * Mdot_sfr # Msun/yr
        Edot_wind = etaE_ism * ((1e51/100.) * (Mdot_sfr/yr_to_s)) # erg/s
        
    #     debug.print('{z}, {Mdot_cool}, {M_cgm}, {Mdot_sfr}, {M_ism}, {Edot_diss}, {Edot_cool}, {Eth_cgm}, {log_Zcgm}, {T0_val}, {n0_val}, {Rvir}, {halo_coolfunc}',z=redshift,
    #                 Mdot_cool=Mdot_cool,M_cgm=M_cgm,Mdot_sfr=Mdot_sfr,M_ism=M_ism,
    #                 Edot_diss=Edot_diss,Edot_cool=Edot_cool,Eth_cgm=Eth_cgm,log_Zcgm=log_Zcgm,T0_val=T0_val,n0_val=n0_val,Rvir=Rvir,
    #                 halo_coolfunc=halo_coolfunc)
        
        # debug.print('Mdot_sfr={Mdot_sfr},Mdot_wind={Mdot_wind},Edot_wind={Edot_wind}',Mdot_sfr=Mdot_sfr,Mdot_wind=Mdot_wind,Edot_wind=Edot_wind)
        
        ### compute Edot_out_halo and Mdot_out_halo terms 
        Ebind_cgm = evir_halo * M_cgm # erg
        Delta_Ecgm = Eth_cgm - Ebind_cgm # erg
        Edot_out_halo = jnp.max(jnp.array([0.0,Delta_Ecgm])) / (tdyn_halo*1e9*yr_to_s) # erg/s
        Mdot_out_halo = (Edot_out_halo/s_to_yr) / (Eth_cgm/M_cgm) # Msun/yr
        
        ### Nov 14 2022 -- parameterized fthermal_wind, fthermal_accretion, fprev
        f_prev = get_fprev(Vvir)#,redshift)
        
        Mdot_in_halo *= f_prev 
        Edot_in_halo *= f_prev
                
        # debug.print('Edot_out_halo={Edot_out_halo},Mdot_out_halo={Mdot_out_halo},f_prev={f_prev},fthermal_wind={fthermal_wind},fthermal_accretion={fthermal_accretion},'
        #             +'Edot_in_halo={Edot_in_halo},fthermal_cgm={fthermal_cgm}',Edot_out_halo=Edot_out_halo,Mdot_out_halo=Mdot_out_halo,f_prev=f_prev,
        #             fthermal_wind=fthermal_wind,fthermal_accretion=fthermal_accretion,Edot_in_halo=Edot_in_halo,fthermal_cgm=fthermal_cgm)
        
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
        
    #     debug.print('BEFORE LOG: Mdot_ism={Mdot_ism},10**logt={explogt},M_ism={M_ism}',Mdot_ism=Mdot_ism,explogt=10**logt,M_ism=M_ism)
        
        # debug.print('ISM CHANGE: {Mdot_ism}={Mdot_cool}-{Mdot_sfr}-{Mdot_wind}',Mdot_ism=Mdot_ism,Mdot_cool=Mdot_cool/yr_to_s,
        #             Mdot_sfr=(1.0-f_recycle)*Mdot_sfr/yr_to_s,Mdot_wind=Mdot_wind/yr_to_s)
        
        # SECOND: since we are integrating log(state_variable) AND log(time), multiply RHS by 10**t[sec] / 10**state_variable
        #         thus these all become dimensionless logarithmic derivatives dlogX/dlogt
        Mdot_sfr_longlived *= 10**logt / M_star 
        Mdot_ism *= 10**logt / M_ism 
        Mdot_cgm *= 10**logt / M_cgm
        Edot_cgm_th *= 10**logt / Eth_cgm
        MZdot_star *= 10**logt / MZ_star
        MZdot_ism *= 10**logt / MZ_ism
        MZdot_cgm *= 10**logt / MZ_cgm    
        
        # debug.print('logt={logt},z={redshift},logy={logy},halo_coolfunc={halo_coolfunc}',logt=logt,redshift=redshift,logy=logy,halo_coolfunc=halo_coolfunc)
        
    #     debug.print('M_star={M_star},M_ism={M_ism},M_cgm={M_cgm},Eth_cgm={Eth_cgm},Ekin_cgm={Ekin_cgm},MZ_star={MZ_star},MZ_ism={MZ_ism},MZ_cgm={MZ_cgm}',
    #                 M_star=logy[0],M_ism=logy[1],M_cgm=logy[2],Eth_cgm=logy[3],Ekin_cgm=logy[4],MZ_star=logy[5],MZ_ism=logy[6],MZ_cgm=logy[7])        
    
    #     debug.print('Mdot_sfr_longlived={Mdot_sfr_longlived},Mdot_ism={Mdot_ism},Mdot_cgm={Mdot_cgm},Edot_cgm_th={Edot_cgm_th},Edot_cgm_kin={Edot_cgm_kin}'
    #                 +'MZdot_star={MZdot_star},MZdot_ism={MZdot_ism},MZdot_cgm={MZdot_cgm}',Mdot_sfr_longlived=Mdot_sfr_longlived,Mdot_ism=Mdot_ism,
    #                 Mdot_cgm=Mdot_cgm,Edot_cgm_th=Edot_cgm_th,Edot_cgm_kin=Edot_cgm_kin,MZdot_star=MZdot_star,MZdot_ism=MZdot_ism,MZdot_cgm=MZdot_cgm)
        
    
        
        return jnp.array([Mdot_sfr_longlived, Mdot_ism, Mdot_cgm, Edot_cgm_th, MZdot_star, MZdot_ism, MZdot_cgm])    
    
    
    """
    saveat function that returns state and auxiliary info at each time for evaluating output
    this is not elegant -- shouldn't have to copy-paste integrator function
    """
    
    @jit
    def saveat_fn(logt,logy,args): 
        """
        args = (parameters, interps) which are both lists
        """   
        
        # raise the log10(t/sec) to power of 10 and convert to Gyr for interpolation functions
        t_Gyr = 10**logt * s_to_yr * 1e-9    
        
        # unpack the log10 state variables and raise to power 10 
        M_star = 10**logy[0]
        M_ism = 10**logy[1]
        M_cgm = 10**logy[2]
        Eth_cgm = 10**logy[3]
        MZ_star = 10**logy[4]
        MZ_ism = 10**logy[5]
        MZ_cgm = 10**logy[6]
        
        ### unpack parameters and interps lists
        parameters, interps = args    
        
        (etaM_A,etaM_alpha0,etaM_alphaz,etaM_beta,etaE_A,etaE_alpha0,etaE_alphaz,etaE_beta,
         tdep_A,tdep_alpha0,tdep_alphaz,tdep_beta,etaZ_A,etaZ_alpha0,etaZ_alphaz,etaZ_beta,
         realization) = parameters 
        
        
        # use global interp_redshift to go from current time to redshift
        redshift = interp_redshift(t_Gyr) # dimensionless    
    
        # unpack cubic hermite interpolators for current halo, and evaluate at current time in physical units
        interp_logmdot, interp_logmvir, interp_logrvir, interp_logvvir, interp_logcnfw = interps
        Mdot_in_dm = 10**interp_logmdot.evaluate(t_Gyr) # Msun/yr
        Mvir = 10**interp_logmvir.evaluate(t_Gyr) # Msun
        Rvir = 10**interp_logrvir.evaluate(t_Gyr) # proper kpc
        Vvir = 10**interp_logvvir.evaluate(t_Gyr) # proper km/s
        NFW_c = 10**interp_logcnfw.evaluate(t_Gyr) # dimensionless NFW halo concentration = Rvir/rscale
        
        # compute gas free-fall radius rff = radius of Vmax ~ 2.16*Rscale ~ 2.16 * Rvir/cNFW 
        rff = 2.16 * (Rvir / NFW_c) / Rvir # in units of Rvir
        
        # compute any other global halo quantities 
        Tvir = 35.9 * Vvir**2 # K 
        evir_halo = 3/2. * const_kB * Tvir / (0.59*const_mp) # erg/Msun  
        tdyn_halo = Rvir / (Vvir * 1e5*cm_to_kpc / (s_to_yr*1e-9)) # Gyr
        
        # analytic formula for NFW Vcirc(rff:=r/Rvir) rom Mo, van den Bosch & White eqn 11.26
        NFW_vcirc_num = jnp.log(1+NFW_c*rff) - NFW_c*rff/(1+NFW_c*rff)
        NFW_vcirc_den = rff*(jnp.log(1+NFW_c)-NFW_c/(1+NFW_c))
        NFW_vcirc = Vvir*jnp.sqrt(NFW_vcirc_num/NFW_vcirc_den) # km/s
        tdyn_ism = rff*Rvir / (NFW_vcirc * 1e5*cm_to_kpc / (s_to_yr*1e-9)) # Gyr
        
        # get SF and ISM wind scalings from FIRE 
        tdep_ism = get_tdep(Vvir,redshift,A=tdep_A,alpha0=tdep_alpha0,alphaz=tdep_alphaz,beta=tdep_beta) # Gyr 
        etaM_ism = get_etaM_ism(Vvir,redshift,A=etaM_A,alpha0=etaM_alpha0,alphaz=etaM_alphaz,beta=etaM_beta) # dimensionless 
        etaE_ism = get_etaE_ism(Vvir,redshift,A=etaE_A,alpha0=etaE_alpha0,alphaz=etaE_alphaz,beta=etaE_beta) # dimensionless    
        etaZ_ism = get_etaZ_ism(Vvir,redshift,A=etaZ_A,alpha0=etaZ_alpha0,alphaz=etaZ_alphaz,beta=etaZ_beta) # dimensionless        
    
        # get halo inflow metallicity from FIRE scaling because we don't yet have outer halo wind recycling
        Zin_halo = get_Zin_halo(Vvir,redshift) # dimensionless Zdot/Mdot (*not* normalized to Zsun)
        
        # compute log10 of CGM metallicity normalized to Zsun for cooling function
        log_Zcgm = jnp.log10(MZ_cgm / M_cgm / 0.02)
            
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
       
        n0_val = analytic_n0(M_cgm,Rvir,alpha_n) # cm**-3
        T0_val = analytic_T0(Eth_cgm,Rvir,n0_val,alpha_n,alpha_T) # K
        
        # now construct density and temperature profiles
        n_rarr = n0_val * rarr**alpha_n # cm**-3        
        T_rarr = T0_val * rarr**alpha_T # K
        
        halo_coolfunc = 10**coolfunc((jnp.log10(T0_val),log_Zcgm)) # erg s**-1 cm**3
        
        # March 2024 -- switch to CIE SD93 means: (1) Edot_cool is analytic , (2) need to impose catastrophic cooling rate limiter
        Edot_cool = 4*jnp.pi*n0_val**2 * (Rvir*kpc_to_cm)**3 * halo_coolfunc * jnp.log(10) # erg/s
        Edot_cool = jnp.min(jnp.array([Edot_cool, Eth_cgm/(tdyn_halo*1e9*yr_to_s)])) # erg/s 
        
        # March 2024 -- with SD93 CIE coolfunc, Edot_cool can never be zero so remove overhead of jnp.select
        tcool_eff = (Eth_cgm / Edot_cool)*s_to_yr*1e-9
        tff_eff = (rff*Rvir*kpc_to_cm*1e-5 / NFW_vcirc)*s_to_yr*1e-9
        t_accrete = tcool_eff + tff_eff 
        Mdot_cool = M_cgm / (t_accrete*1e9) # Msun/yr 
        
    
        ### compute SFR term
        Mdot_sfr = M_ism / (tdep_ism*1e9) # Msun/yr
        
        ### compute Mdot_wind and Edot_wind terms
        Mdot_wind = etaM_ism * Mdot_sfr # Msun/yr
        Edot_wind = etaE_ism * ((1e51/100.) * (Mdot_sfr/yr_to_s)) # erg/s
        
        ### compute Edot_out_halo and Mdot_out_halo terms 
        Ebind_cgm = evir_halo * M_cgm # erg
        Delta_Ecgm = Eth_cgm - Ebind_cgm # erg
        Edot_out_halo = jnp.max(jnp.array([0.0,Delta_Ecgm])) / (tdyn_halo*1e9*yr_to_s) # erg/s
        Mdot_out_halo = (Edot_out_halo/s_to_yr) / (Eth_cgm/M_cgm) # Msun/yr
        
        ### Nov 14 2022 -- parameterized fthermal_wind, fthermal_accretion, fprev
        f_prev = get_fprev(Vvir)#,redshift)
        
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
        
        """ 
        Unlike for integrator, here we want to return state and auxiliary info for saveAt 
        So we don't need derivs as dlogx/dlogt
        """
        
        aux_info = {'Mdot_in_halo':Mdot_in_halo,'Edot_in_halo':Edot_in_halo,'Edot_cool':Edot_cool,'Mdot_cool':Mdot_cool,
                    'Mdot_sfr':Mdot_sfr,'Mdot_wind':Mdot_wind,'Edot_wind':Edot_wind,'Edot_out_halo':Edot_out_halo,'Mdot_out_halo':Mdot_out_halo,
                    'MZdot_sfr':MZdot_sfr,'MZdot_cool':MZdot_cool,'MZdot_yield':MZdot_yield,'MZdot_wind':MZdot_wind,'MZdot_in_halo':MZdot_in_halo,
                    'MZdot_out_halo':MZdot_out_halo,'cosmic_age':t_Gyr,'redshift':redshift,'Mvir':Mvir,'Mdot_in_dm':Mdot_in_dm,'Rvir':Rvir,'Vvir':Vvir,
                    'NFW_c':NFW_c,'Tvir':Tvir,'tdyn_halo':tdyn_halo,'tdyn_ism':tdyn_ism,'tdep_ism':tdep_ism,'etaM_ism':etaM_ism,'etaE_ism':etaE_ism,
                    'etaZ_ism':etaZ_ism,'Zin_halo':Zin_halo,'Mfilt':Mfilt,'f_UV':f_UV,'n0':n0_val,'T0':T0_val,'tcool_eff':tcool_eff,'tff_eff':tff_eff,
                    'Ebind_cgm':Ebind_cgm,'Delta_Ecgm':Delta_Ecgm,'f_prev':f_prev,'Zcgm':Zcgm,'Zism':Zism,'realization':realization}
        
        return logy, aux_info
    

    # Jan 6, 2026 -- return list of parameterized functions (tdep, etaM, etaE, etaZ) for visualization on posterior corner plots
    return integrator, saveat_fn, list_parameterizations     

#