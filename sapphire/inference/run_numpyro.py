""" 
this module defines the numpyro model and runs inference
it is analogous to explicit_likelihood.py and run_adam.py 

TO DO: the numpyro model definition and running NUTS, SVI, etc. can be in separate modules
"""

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15
from functools import partial 
import os
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 120
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

# in case user loads module separately from sapphire.run()
from jax import config as jax_config
jax_config.update("jax_enable_x64", True) # required to accurately solve and take gradients through our diffeqs 

import jax
import jax.numpy as jnp
from jax._src.third_party.scipy.interpolate import RegularGridInterpolator as jax_RegularGridInterpolator
from jax import jit, grad, vmap, pmap, debug, jvp, vjp, jacrev, jacfwd, make_jaxpr, hessian, value_and_grad
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline    
from jax.experimental.ode import odeint
from jax.lax import fori_loop, while_loop
from jax.scipy.integrate import trapezoid
from jax.random import PRNGKey, key    
from diffrax import diffeqsolve, ODETerm, PIDController, SaveAt, Kvaerno3, Bosh3, Dopri5, Tsit5, DirectAdjoint, RecursiveCheckpointAdjoint, BacksolveAdjoint
from diffrax import backward_hermite_coefficients, CubicInterpolation    
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec, PositionalSharding, NamedSharding

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, AIES, ESS
from chainconsumer import Chain, ChainConsumer, make_sample, PlotConfig, Truth, ChainConfig

# this keeps it clean
import sapphire.summaries.gaussian_kernel_regression as gkr



def setup(config,halo_index,obs_stats,batch_solve):

    inference_config = config['inference_config']
    sampling_config = config['sampling_config']
    params_fixed_astro = config['params_fixed_astro']

    # required parameter order for sapphire
    # NOTE: change API to just use dicts throughout (care is needed for gradients)
    full_params_order = ['A_M','alpha0_M','alphaz_M','beta_M',
                         'A_E','alpha0_E','alphaz_E','beta_E',
                         'A_SF','alpha0_SF','alphaz_SF','beta_SF',
                         'A_Z','alpha0_Z','alphaz_Z','beta_Z'] 

    params_bounds = sampling_config['params_bounds']
    params_free = list(params_bounds.keys())
    Nfree = len(params_free)
    
    lower_bounds = jnp.array([params_bounds[k][0] for k in params_free])
    upper_bounds = jnp.array([params_bounds[k][1] for k in params_free])  

    Lflag_smhm = config['flag_smhm'] 
    Lflag_fgas = config['flag_fgas'] 
    Lflag_mzr = config['flag_mzr']   

    Nbatch = config['Nbatch']

    ### unpack input observed (or mock) summary statistics
    (obs_x0_smhm,obs_bw_smhm,obs_avg_smhm,obs_err_smhm,
     obs_x0_fgas,obs_bw_fgas,obs_avg_fgas,obs_err_fgas,
     obs_x0_mzr,obs_bw_mzr,obs_avg_mzr,obs_err_mzr) = obs_stats

    # print('raw mock obs_err',obs_err_smhm,obs_err_fgas,obs_err_mzr,flush=True)
    
    ### IF MOCK MODE -- change obs_err_* to the user supplied constant or scalar
    #   (or have user add x-dependent function in sapphire.summaries and import here)
    if inference_config['fit_mock'] is True and inference_config['fit_obs'] is False: # checking both as safeguard

        # unclear how to use "scale" since that needs to be multiplied by intrinsic stderr(y) inside 
        # might need to add that as static (non-jit-traced) input to logL functions below
        obs_err_smhm = config['obs_err_smhm']
        obs_err_fgas = config['obs_err_fgas']
        obs_err_mzr = config['obs_err_mzr']
        
        
    """ define the numpyro model """
    
    # note: these input obs can be different from the obs_ ones returned above
    def model(obs_avg_smhm,obs_err_smhm,obs_avg_fgas,obs_err_fgas,obs_avg_mzr,obs_err_mzr):
        
        ### sample the free parameters assuming their respective Uniform priors 
        # the user should change the prior sampling function as needed...
        params_dict = {}
        for i, name in enumerate(params_free):
            params_dict[name] = numpyro.sample(name,dist.Uniform(lower_bounds[i], upper_bounds[i]))

        # build full parameter dict and array 
        full_params_dict = {**params_fixed_astro, **params_dict}
        full_params = jnp.array([full_params_dict[k] for k in full_params_order])
        
        full_params = full_params.at[0].set(10**full_params[0]) # 10**A_M
        full_params = full_params.at[4].set(10**full_params[4]) # 10**A_E
        full_params = full_params.at[8].set(10**full_params[8]) # 10**A_SF
        full_params = full_params.at[12].set(10**full_params[12]) # 10**A_Z
        full_params = jnp.concatenate([full_params,jnp.array([0])]) # realization #

        ### for some reason with adam, Uniform hard-cutoff is not enforced
        ### so here add a soft quadratic penalty just like i was doing manually, using numpyro.factor
        parr = jnp.array([params_dict[name] for name in params_free])
        lower_penalty = jnp.sum(jnp.maximum(lower_bounds - parr, 0)**2)
        upper_penalty = jnp.sum(jnp.maximum(parr - upper_bounds, 0)**2)
        penalty = 1e8 * Nbatch * (lower_penalty + upper_penalty) # only gradient should matter, not normalization I think
        numpyro.factor('penalty', -penalty)
        
        # first call the batch solve 
        shsol = batch_solve(halo_index, full_params)
    
        """ July 14 -- more compact using functions above """
        z0_Mvir, z0_smhm, fail_flag, Nfail, z0_Mstar, z0_fgas, z0_mzr = gkr.extract_quantities(shsol)
    
        numpyro.deterministic('Nfail', Nfail)    
    
        ### July 24 -- evaluate now using observed x0 and bin(=band) widths
        pred_avg_smhm, pred_err_smhm = gkr.nadaraya_watson(z0_Mvir, z0_smhm, obs_x0_smhm, obs_bw_smhm)
        pred_avg_fgas, pred_err_fgas = gkr.nadaraya_watson(z0_Mstar, z0_fgas, obs_x0_fgas, obs_bw_fgas)
        pred_avg_mzr, pred_err_mzr = gkr.nadaraya_watson(z0_Mstar, z0_mzr, obs_x0_mzr, obs_bw_mzr) 
    
        """ July 24 -- quadrature sum intrinsic model standard error with observed uncertainty """ 
        ### NEED TO UPDATE THIS TO GENERALLY HANDLE MOCKED OR ACTUAL OBSERVED EXTRA ERROR 
        tot_err_smhm = jnp.sqrt(pred_err_smhm**2 + obs_err_smhm**2)
        tot_err_fgas = jnp.sqrt(pred_err_fgas**2 + obs_err_fgas**2)
        tot_err_mzr = jnp.sqrt(pred_err_mzr**2 + obs_err_mzr**2)

        # to save as part of outputs 
        numpyro.deterministic('pred_avg_smhm',pred_avg_smhm)
        numpyro.deterministic('pred_avg_fgas',pred_avg_fgas)
        numpyro.deterministic('pred_avg_mzr',pred_avg_mzr)
        numpyro.deterministic('pred_err_smhm',pred_err_smhm)
        numpyro.deterministic('pred_err_fgas',pred_err_fgas)
        numpyro.deterministic('pred_err_mzr',pred_err_mzr)    
        
        if obs_avg_smhm is None: # for prior and posterior predictive checks
    
            numpyro.sample('obs_smhm',dist.Normal(pred_avg_smhm,tot_err_smhm)) 
            numpyro.sample('obs_fgas',dist.Normal(pred_avg_fgas,tot_err_fgas)) 
            numpyro.sample('obs_mzr',dist.Normal(pred_avg_mzr,tot_err_mzr)) 
            
        else:
    
            # sample gaussian likelihoods independently (these will be internally summed by numpyro like we'd do manually)
            # May 15 -- only compute each logL if requested based on command line argument
            if Lflag_smhm:
                obs_smhm = numpyro.sample('obs_smhm',dist.Normal(pred_avg_smhm,tot_err_smhm), obs=obs_avg_smhm)
            
            if Lflag_fgas:
                obs_fgas = numpyro.sample('obs_fgas',dist.Normal(pred_avg_fgas,tot_err_fgas), obs=obs_avg_fgas)
    
            if Lflag_mzr: 
                obs_mzr = numpyro.sample('obs_mzr',dist.Normal(pred_avg_mzr,tot_err_mzr), obs=obs_avg_mzr)
                

    # in case we want to compute loss for adam based on numpyro model
    # this hard-codes the input obs constraint as inputs to numpyro model() function
    def numpyro_loss(params):
        # numpyro.infer.util.log_density returns (log_density, aux) tuple.
        logp, _ = numpyro.infer.util.log_density(model, 
                                                 (obs_avg_smhm,obs_err_smhm,
                                                  obs_avg_fgas,obs_err_fgas,
                                                  obs_avg_mzr,obs_err_mzr), 
                                                 {}, params)
        
        # take negative of numpyro log-posterior so it is a loss for adam
        return -logp    

    # if requested, just return numpyro loss function but don't do any inference
    if inference_config['engine'] == 'adam' and inference_config['adam_logL'] == 'numpyro':
        return numpyro_loss
    

    ##### Otherwise proceed with numpyro inference with a wrapper that takes input chain_num

    ### TO DO: push this output filename stuff down to utils/write_results or somewhere
    prefix = config['output_prefix'].format(**config) # typically numpyro_{obs_name}_{chain_num}
    suffix = config['output_suffix'].format(**config) # typically {flag}{flag}{flag}

    def run_chain(chain_num):

        ### start warmup
        
        mcmc = MCMC(NUTS(model,dense_mass=True,
                        init_strategy=numpyro.infer.init_to_uniform(),
                        # init_strategy=numpyro.infer.init_to_value(values=init_zparams),
                        forward_mode_differentiation=True,
                        target_accept_prob=0.8,max_tree_depth=10, # defaults 0.8 and 10
                        ),
                    num_warmup=inference_config['num_warmup'],
                    num_samples=inference_config['num_samples'],num_chains=1)
        
        # Feb 4 -- make RNG chain_num so every chain combo has different initialization... 
        # TO DO: for mocks, put back in dependence on mock_num+chain_num for 
        mcmc.warmup(key(chain_num+22), # random int offset just in case.. 
                    obs_avg_smhm,obs_err_smhm,obs_avg_fgas,obs_err_fgas,obs_avg_mzr,obs_err_mzr,
                    collect_warmup=True,
                    extra_fields=('i','z','z_grad','potential_energy','energy','r','num_steps',
                    'adapt_state.step_size','adapt_state.inverse_mass_matrix',))
        
        warmup_samples = mcmc.get_samples()    
        warmup_extra_fields = mcmc.get_extra_fields()
        
        
        print(mcmc.print_summary(),flush=True)
        
        
        warmup_samples_dict = {k: warmup_samples[k] for k in params_free}
        warmup_samples_arr = jnp.stack([warmup_samples[k] for k in params_free], axis=1)
        
        print(jnp.corrcoef(warmup_samples_arr.T),flush=True) # jnp.cov(warmup_samples_arr.T)
        
        
        warmup_samples_df = pd.DataFrame(warmup_samples_dict)
        # print(warmup_samples_df,flush=True)
        
        c = ChainConsumer()
        
        c.add_chain(Chain(samples=warmup_samples_df, name="An Example Contour"))

        # TO DO: change this to use user-provided prior limits
        c.set_plot_config(PlotConfig(extents={'A_M':(-2.1,1.1),'alpha0_M':(-2.1,0.1),
                                                    'A_E':(-2.1,0.1),'alpha0_E':(-2.1,0.1),
                                                    'A_SF':(-0.1,1.2),'alpha0_SF':(-2.1,0.1),
                                                    'A_Z':(-2.1,0.1),'alpha0_Z':(-2.1,0.1),},
                                            labels={'A_M':r'$A_M$','alpha0_M':r'$\alpha_M^0$',
                                                    'A_E':r'$A_E$','alpha0_E':r'$\alpha_E^0$',
                                                    'A_SF':r'$A_{\rm SF}$','alpha0_SF':r'$\alpha_{\rm SF}^0$',
                                                    'A_Z':r'$A_Z$','alpha0_Z':r'$\alpha_Z^0$'},
                                            label_font_size=16))
        
        fig = c.plotter.plot()

        # TO DO: move this fname (w/ prefix and suffix) to utils
        fname = os.path.join(config['output_path'],'figures','%s_%s_corner_warmup.png'%(prefix,suffix))
        plt.savefig(fname,bbox_inches='tight',dpi=300,facecolor='w')
        print('saved %s'%fname,flush=True)
        
        
        # now run actual sampling phase
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key,
                obs_avg_smhm,obs_err_smhm,obs_avg_fgas,obs_err_fgas,obs_avg_mzr,obs_err_mzr,
                extra_fields=('i','z','z_grad','potential_energy','energy','r','num_steps',
                            'adapt_state.step_size','adapt_state.inverse_mass_matrix',))    
        
        samples = mcmc.get_samples()  
        samples_extra_fields = mcmc.get_extra_fields()
        
        print(mcmc.print_summary(),flush=True)
        
        
        samples_dict = {k: samples[k] for k in params_free}
        samples_df = pd.DataFrame(samples_dict)
        
        
        c = ChainConsumer()
        
        c.add_chain(Chain(samples=samples_df, name="An Example Contour"))
        
        c.set_plot_config(PlotConfig(extents={'A_M':(-2.1,1.1),'alpha0_M':(-2.1,0.1),
                                                    'A_E':(-2.1,0.1),'alpha0_E':(-2.1,0.1),
                                                    'A_SF':(-0.1,1.2),'alpha0_SF':(-2.1,0.1),
                                                    'A_Z':(-2.1,0.1),'alpha0_Z':(-2.1,0.1),},
                                            labels={'A_M':r'$A_M$','alpha0_M':r'$\alpha_M^0$',
                                                    'A_E':r'$A_E$','alpha0_E':r'$\alpha_E^0$',
                                                    'A_SF':r'$A_{\rm SF}$','alpha0_SF':r'$\alpha_{\rm SF}^0$',
                                                    'A_Z':r'$A_Z$','alpha0_Z':r'$\alpha_Z^0$'},
                                            label_font_size=16))
        
        fig = c.plotter.plot()

        fname = os.path.join(config['output_path'],'figures','%s_%s_corner_samples.png'%(prefix,suffix))
        plt.savefig(fname,bbox_inches='tight',dpi=300,facecolor='w')
        print('saved %s'%fname,flush=True)
        
        
        # TO DO: bring back filename prefix/suffix (and push that to utils)
        fname = os.path.join(config['output_path'],'outputs','%s_%s.npz'%(prefix,suffix))

        # TO DO: move this to utils/write_results.py 
        jnp.savez(fname,
                  warmup_samples = warmup_samples,
                  warmup_extra_fields = warmup_extra_fields,
                  samples = samples,
                  samples_extra_fields = samples_extra_fields,
                  Lflag_smhm=Lflag_smhm,
                  Lflag_fgas=Lflag_fgas,
                  Lflag_mzr=Lflag_mzr,
                  # save obs_stats in case it was changed for forecasting -- for easier predictive plots
                  obs_x0_smhm=obs_x0_smhm,
                  obs_bw_smhm=obs_bw_smhm,
                  obs_avg_smhm=obs_avg_smhm,
                  obs_err_smhm=obs_err_smhm,
                  obs_x0_fgas=obs_x0_fgas,
                  obs_bw_fgas=obs_bw_fgas,
                  obs_avg_fgas=obs_avg_fgas,
                  obs_err_fgas=obs_err_fgas,
                  obs_x0_mzr=obs_x0_mzr,
                  obs_bw_mzr=obs_bw_mzr,
                  obs_avg_mzr=obs_avg_mzr,
                  obs_err_mzr=obs_err_mzr
                 )
        
        print('saved %s'%fname,flush=True)

        plt.close('all')

        return warmup_samples,warmup_extra_fields,samples,samples_extra_fields


    ### finally call for the user input chain_num
    ### TO DO: consider a fori_loop or vmap over GPUs, though HMC is slow for our problem
    out_numpyro = run_chain(config['chain_num'])

    return out_numpyro




    
###