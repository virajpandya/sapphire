"""
this module takes the batch_solve() function defined for a physical model in sapphire/physical_models
and evolves the ODEs
"""

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15, FlatLambdaCDM 
from functools import partial 

# in case user loads module separately from sapphire.run()
from jax import config as jax_config
jax_config.update("jax_enable_x64", True) # required to accurately solve and take gradients through our diffeqs 

import jax
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


def setup(config,integrator,saveat_fn,rand_halo_matrix,rand_coeff_matrix,rand_halo_tinit,ts_interp):

    ### define constants and conversions 
    const_mp = const.m_p.to('Msun').value 
    cm_to_kpc = u.cm.to('kpc') # multiply something in cm by this to get to kpc 
    kpc_to_cm = u.kpc.to('cm') # multiply something in kpc, it becomes units of cm
    const_kB = const.k_B.to('erg/K').value # so that k*T = erg by default
    yr_to_s = u.yr.to('s') # if you multiply something in yr by this, you get it in units of sec
    s_to_yr = u.s.to('yr')
    Msun_to_g = u.Msun.to('g') 

    # April 2026 -- cosmo params are inputs now for Robinson+26 
    params_cosmo = config['params_cosmo'] 

    f_b = params_cosmo['Ob0'] / params_cosmo['Om0'] # ~15% for Planck15
    littleh = params_cosmo['h0']
    Om0 = params_cosmo['Om0']
    G = const.G.to('cm**3 / (g * s**2)').value

    cosmo = FlatLambdaCDM(H0=littleh*100, Om0=Om0, Tcmb0=params_cosmo['Tcmb0'], 
                          Ob0=params_cosmo['Ob0'], Neff=params_cosmo['Neff'])
    
    G = const.G.to('cm**3 / (g * s**2)').value
    t0 = cosmo.age(0).value 
    
    ### set up global final integration time in log(t/sec) corresponding to z=0
    logt_final = jnp.log10(cosmo.age(0.0).value * 1e9 * yr_to_s)
    
    ### set up inputs for diffrax ODE solver
    
    terms = ODETerm(integrator)
    t1 = logt_final 
    dt0 = 1e-10 # April 2024 -- this cannot be None otherwise Tsit5() cannot find dt0 even if max_steps=16**6 (it solves in only ~500 steps vs 10K for Bosh3)
    solver = Tsit5(scan_kind='bounded') #Dopri5() #Tsit5() #Bosh3() 
    # if this is None, reverse-mode checkpointed adjoint (autodiff) cannot be done; typically we only need ~3K steps for our tolerance choice
    max_steps = config['solver_config']['max_steps']
    
    ### set up output times at which to return solution
    
    if config['output_redshifts'] in [0.0, [0], [0.0]]: 
        # TO DO: generalize this in case logt_final != 0.0 and user wants to run only to some other final redshift and output that
        saveat = SaveAt(t1=True,fn=saveat_fn) 
    else: 
        # t_eval = jnp.log10(yr_to_s*1e9*jnp.asarray(Planck15.age(jnp.arange(0.0,3.5,0.5)).value)[::-1])
        # assume user input redshift array in monotonically increasing order, but ages then become decreasing, need [::-1] to reverse for diffrax
        t_eval = jnp.log10(yr_to_s*1e9*jnp.asarray(cosmo.age(jnp.asarray(config['output_redshifts'])).value)[::-1]) 
        saveat = SaveAt(ts=t_eval,fn=saveat_fn)
        
        # t_eval = jnp.linspace(logt_init,logt_final,100)
        # tsave = SaveAt(ts=t_eval,dense=True) # save solution on our finely spaced grid with dense interpolation
    
    # first need to trim # halos so it = integer # of devices
    Ndevices = jax.local_device_count()
    print('Ndevices for jax batching', Ndevices)
    total_size = rand_halo_matrix.shape[0] - (rand_halo_matrix.shape[0] % Ndevices) # subtracted off extra not divisible by 64
    batch_halo_matrix = rand_halo_matrix[:total_size] # this can be split evenly into Ndevices batches with total_size/Ndevices halos each
    batch_coeff_matrix = rand_coeff_matrix[:total_size]
    batch_halo_tinit = rand_halo_tinit[:total_size]
    batch_halo_index = jnp.arange(len(batch_halo_tinit)) # these are indices of rows of batch_coeff_matrix and batch_halo_tinit that will be mapped over

    print('diffrax shapes',batch_halo_matrix.shape,batch_coeff_matrix.shape,batch_halo_index.shape,flush=True)

    
    def single_solve(halo_index,batch_params):
        """
        halo_index = row # of batch_coeffs_matrix, batch_halo_tinit
        """
        
        ### INITIAL CONDITIONS
        
        ### get that halo's coeff_matrix, t_init
        this_coeff_matrix = batch_coeff_matrix[halo_index]
        this_tinit = batch_halo_tinit[halo_index]
    
        ### set up integration times -- LOGARITHMIC in sec since we are integrating dlogX/dlogt 
        # logt_final is always at z=0 and fixed outside as a global parameter
        logt_init = jnp.log10(this_tinit * 1e9 * yr_to_s) # Gyr to sec
    
        ### set up interpolators for usage within integrator and for initial conditions
        interp_logmdot = CubicInterpolation(ts_interp, this_coeff_matrix[0])
        interp_logmvir = CubicInterpolation(ts_interp, this_coeff_matrix[1])
        interp_logrvir = CubicInterpolation(ts_interp, this_coeff_matrix[2])
        interp_logvvir = CubicInterpolation(ts_interp, this_coeff_matrix[3])
        interp_logcnfw = CubicInterpolation(ts_interp, this_coeff_matrix[4])
        interps = [interp_logmdot,interp_logmvir,interp_logrvir,interp_logvvir,interp_logcnfw]
    
        ### initial conditions for state variables -- TAKE LOG since we are integrating dlogX/dlogt 
        # order: Mstar [Msun], Mism [Msun], Mcgm [Msun], Eth_cgm [erg] MZ_star, MZ_ism, MZ_cgm 
        Mvir_init = 10**interp_logmvir.evaluate(10**logt_init*s_to_yr*1e-9) # Msun
    
        Mcgm_init = 0.5*f_b*Mvir_init # initial Mcgm [Msun]
    
        Rvir_init = 10**interp_logrvir.evaluate(10**logt_init*s_to_yr*1e-9) # proper kpc
        Vvir_init = 10**interp_logvvir.evaluate(10**logt_init*s_to_yr*1e-9) # proper km/s
        Ecgm_init = Mcgm_init*Msun_to_g * (Vvir_init*1e5)**2 # erg
    
        Mstar_init = 1.0 # arbitrary small number
        Mism_init = 1.0 # arbitrary small number
        Z_init = 1e-4 * 0.02 # assume initial metallicity is 1E-4/Zsun, multiply initial masses by this factor to get initial metal *mass*
    
        initial_conditions = jnp.log10(jnp.array([Mstar_init,Mism_init,Mcgm_init,Ecgm_init,Z_init*Mstar_init,Z_init*Mism_init,Z_init*Mcgm_init]))    
        
        ### CALL SOLVER -- in this example, want only z=0 final state variables
        # set adjoint=DirectAdjoint() if forward mode autodiff desired
        """ May 12 -- try atol=rtol=1e-5 """ 
        """ July 14 -- fix to atol=rtol=1e-5, max_steps=16**4, t1=True """
        sol_diffrax = diffeqsolve(terms,solver,
                                  logt_init,logt_final,dt0,initial_conditions,(batch_params,interps),throw=False,adjoint=DirectAdjoint(),
                                  stepsize_controller=PIDController(rtol=config['solver_config']['rtol'],
                                                                    atol=config['solver_config']['atol']),
                                  max_steps=max_steps,
                                  # saveat=SaveAt(ts=t_eval,fn=saveat_fn))
                                  # saveat=SaveAt(t1=True,fn=saveat_fn))
                                  saveat=saveat)
        
        return sol_diffrax
    
    """ set up jacfwd of single_solve for (possibly batched) sensitivity analysis """

    # only want jacfwd wrt to specific parameters [this should be same as in utils/benchmark_runtime.py]
    ### TO DO: generalize and turn into dict for better readability 
    inds_wanted = jnp.array([0,1,4,5,8,9,12,13])
    
    """ July 7 -- first a wrapper to only autodiff wrt free parameters, not full """
    def autodiff_jac(params_free,halo_index,params_full):
    
        # need to use params_free somewhere (otherwise jac entries are 0) so just replace params_full with it 
        params_full = params_full.at[inds_wanted].set(params_free)
        
        return single_solve(halo_index,params_full).ys[0]    


    
    """ 
    wrap with vmap or shardmap+vmap depending on Ndevices and TSG vs on-the-fly inference 
    TO DO: maybe this could be further modularized
    """

    # first set up device mesh [so far this is universal to any runtype below but may be made more sophisticated in future]
    mesh = Mesh(mesh_utils.create_device_mesh((Ndevices,)), axis_names=('i',))  

    # only evaluating single parameter set at a time, so parallelization happens over halos, not parameter sets 
    if config['runtype'] in ['single','inference']: 
    
        if jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) > 1:
            print('parallelizing batch_solve with shard_map over multi-CPU/GPU for halos only',flush=True)
        
            batch_solve = jit(shard_map(vmap(single_solve,in_axes=(0,None)),
                                        mesh=mesh,in_specs=(PartitionSpec('i'),PartitionSpec(None)),
                                        out_specs=PartitionSpec('i'),check_rep=False))
        
        elif len(jax.devices('gpu')) == 1:
            print('vmapping batch_solve over single GPU for halos only',flush=True)
        
            batch_solve = jit(vmap(single_solve,in_axes=(0,None)))

        ### April 2026 -- port over batch_jacfwd as well from old ipynb
        ### note: this is really only for runtype='single' -- 'inference' mode computes grad(loss) separately sapphire/inference/
        if jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) > 1:
            print('autodiff over multi-CPU/GPU for runtype=inference',flush=True)
            batch_jacfwd = jit(shard_map(vmap(jacfwd(autodiff_jac),in_axes=(None,0,None)),
                                        mesh=mesh,
                                        in_specs=(PartitionSpec(None),PartitionSpec('i'),PartitionSpec(None)), # shard over halo_index, not params
                                        out_specs=PartitionSpec('i'),check_rep=False)) 
            
        elif len(jax.devices('gpu')) == 1:
            print('autodiff over single GPU for runtype=inference',flush=True)
            batch_jacfwd = jit(vmap(jacfwd(autodiff_jac),in_axes=(None,0,None)))
        
        return batch_solve, batch_jacfwd

    # here, we are evaluating multiple parameter combos for Nbatch_halos, so parallelization should happen over param sets (typically thousands)
    # nov 2025 -- enforce that on CPU Nsamples at minimum = Ndevices, otherwise integer multiple of Ndevices (on GPU doesn't matter)
    elif config['runtype'] in ['sampling']: 

        # now create a jit(shard_map(vmap()) function that batches over params dimension and then another nested vmap over halos
        # June 16 -- multi-CPU-core and multi-GPU is same 
        if jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) > 1:
            
            print('parallelizing batch_solve over multiple CPU/GPU for params and halos')
            
            # now shard_map over CPU/GPU devices available 
            batch_solve = jit(shard_map(vmap(vmap(single_solve,in_axes=(0,None)),in_axes=(None,0)),
                                        mesh=mesh,
                                        in_specs=(PartitionSpec(None),PartitionSpec('i')), # shard over params, not halo_index
                                        out_specs=PartitionSpec('i'),check_rep=False)) 
        
        elif len(jax.devices('gpu')) == 1: # single GPU case just involves a nested vmap over params, then over halos
            print('vmapping batch_solve over single GPU for params and halos',flush=True)
            batch_solve = jit(vmap(vmap(single_solve,in_axes=(0,None)),in_axes=(None,0))) 

        ### April 2026 -- port over batch_jacfwd as well from old ipynb
        if jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) > 1:
            
            print('autodiff over multi-CPU/GPU for runtype=sampling',flush=True)
            
            # now shard_map over CPU/GPU devices available 
            batch_jacfwd = jit(shard_map(vmap(vmap(jacfwd(autodiff_jac),in_axes=(None,0,None)),in_axes=(0,None,0)),
                                        mesh=mesh,
                                        in_specs=(PartitionSpec('i'),PartitionSpec(None),PartitionSpec('i')), 
                                        out_specs=PartitionSpec('i'),check_rep=False)) 
        
        elif len(jax.devices('gpu')) == 1: # single GPU case just involves a nested vmap over params, then over halos
            
            print('autodiff over single GPU for runtype=sampling',flush=True)
            batch_jacfwd = jit(vmap(vmap(jacfwd(autodiff_jac),in_axes=(None,0,None)),in_axes=(0,None,0)))
        
        return batch_solve, batch_jacfwd

    ##### april 2026 -- for runtime benchmarking 
    elif config['runtype'] == 'benchmark':

        #### first for just running [only need one inner vmap unlike for 'sampling' above]
        if jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) > 1:
            
            print('for runtype=benchmark: parallelizing batch_solve over multiple CPU/GPU for params and halos',flush=True)
            
            # now shard_map over CPU/GPU devices available 
            batch_solve = jit(shard_map(vmap(single_solve,in_axes=(0,0)),
                                        mesh=mesh,
                                        in_specs=(PartitionSpec('i'),PartitionSpec('i')), # shard over params, not halo_index
                                        out_specs=PartitionSpec('i'),check_rep=False)) 
        
        elif len(jax.devices('gpu')) == 1: # single GPU case just involves a nested vmap over params, then over halos 
            print('for runtype=benchmark: vmapping batch_solve over single GPU for params and halos',flush=True)
            batch_solve = jit(vmap(single_solve,in_axes=(0,0)))

        ### next for jacobian 

        # June 16 -- multi-CPU-core and multi-GPU are same call structure
        if jax.devices()[0].platform == 'cpu' or len(jax.devices('gpu')) > 1:
            print('for runtype=benchmark: parallelizing autodiff over multi-CPU/GPU',flush=True)
            
            batch_jacfwd = jit(shard_map(vmap(jacfwd(autodiff_jac),in_axes=(0,0,0)),
                                         mesh=mesh,
                                         in_specs=(PartitionSpec('i'),PartitionSpec('i'),PartitionSpec('i')), 
                                         out_specs=PartitionSpec('i'),check_rep=False))
            
        elif len(jax.devices('gpu')) == 1:
            print('for runtype=benchmark: autodiff over single GPU',flush=True)
            batch_jacfwd = jit(vmap(jacfwd(autodiff_jac),in_axes=(0,0,0)))        

        ### return batch_solve, batch_jacfwd, jit(single_solve), and jit(jacfwd(single_solve))
        return batch_solve, batch_jacfwd, jit(single_solve), jit(jacfwd(autodiff_jac))

    
    else:
        raise ValueError('runtype must be one of single, sampling, inference, benchmark')    


    return batch_solve
    

    
        
               