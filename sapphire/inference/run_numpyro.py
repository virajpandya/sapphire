""" 
this module runs numpyro inference routines (NUTS, AIES, SVI, etc.) 
using a pre-defined numpyro model

see define_numpyro.py for an example from Pandya+26
"""

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15
from functools import partial 
import os
import pandas as pd
from timeit import default_timer as timer

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

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, AIES, ESS
from chainconsumer import Chain, ChainConsumer, make_sample, PlotConfig, Truth, ChainConfig

# this keeps it clean
import sapphire.summaries.gaussian_kernel_regression as gkr



def setup(config,model,model_args,params_free,savefigs=False,savenpz=False):

    inference_config = config['inference_config']    

    ##### Otherwise proceed with numpyro inference with a wrapper that takes input chain_num

    def run_nuts(chain_num):

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
                    *model_args,
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

        if savefigs is True:
            c = ChainConsumer()
            
            c.add_chain(Chain(samples=warmup_samples_df, name="An Example Contour"))
    
            # TO DO: change this to use user-provided prior limits
            c.set_plot_config(PlotConfig(extents={'A_M':(-2.1,1.1),'alpha0_M':(-4.1,0.1),
                                                        'A_E':(-2.1,0.1),'alpha0_E':(-4.1,0.1),
                                                        'A_SF':(-0.1,1.2),'alpha0_SF':(-4.1,0.1),
                                                        'A_Z':(-2.1,0.1),'alpha0_Z':(-4.1,0.1),},
                                                labels={'A_M':r'$A_M$','alpha0_M':r'$\alpha_M^0$',
                                                        'A_E':r'$A_E$','alpha0_E':r'$\alpha_E^0$',
                                                        'A_SF':r'$A_{\rm SF}$','alpha0_SF':r'$\alpha_{\rm SF}^0$',
                                                        'A_Z':r'$A_Z$','alpha0_Z':r'$\alpha_Z^0$'},
                                                label_font_size=16))
            
            fig = c.plotter.plot()

            ### TO DO: push this output filename stuff down to utils/write_results or somewhere
            prefix = config['output_prefix'].format(**config) # typically numpyro_{obs_name}_{chain_num}
            suffix = config['output_suffix'].format(**config) # typically {flag}{flag}{flag}
            
            # TO DO: move this fname (w/ prefix and suffix) to utils
            fname = os.path.join(config['output_path'],'figures','%s_%s_corner_warmup.png'%(prefix,suffix))
            plt.savefig(fname,bbox_inches='tight',dpi=300,facecolor='w')
            print('saved %s'%fname,flush=True)
            
        
        # now run actual sampling phase
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key,
                 *model_args,
                 extra_fields=('i','z','z_grad','potential_energy','energy','r','num_steps',
                               'adapt_state.step_size','adapt_state.inverse_mass_matrix',))    
        
        samples = mcmc.get_samples()  
        samples_extra_fields = mcmc.get_extra_fields()
        
        print(mcmc.print_summary(),flush=True)
        
        
        samples_dict = {k: samples[k] for k in params_free}
        samples_df = pd.DataFrame(samples_dict)
        
        if savefigs is True:
        
            c = ChainConsumer()
            
            c.add_chain(Chain(samples=samples_df, name="An Example Contour"))
            
            c.set_plot_config(PlotConfig(extents={'A_M':(-2.1,1.1),'alpha0_M':(-4.1,0.1),
                                                        'A_E':(-2.1,0.1),'alpha0_E':(-4.1,0.1),
                                                        'A_SF':(-0.1,1.2),'alpha0_SF':(-4.1,0.1),
                                                        'A_Z':(-2.1,0.1),'alpha0_Z':(-4.1,0.1),},
                                                labels={'A_M':r'$A_M$','alpha0_M':r'$\alpha_M^0$',
                                                        'A_E':r'$A_E$','alpha0_E':r'$\alpha_E^0$',
                                                        'A_SF':r'$A_{\rm SF}$','alpha0_SF':r'$\alpha_{\rm SF}^0$',
                                                        'A_Z':r'$A_Z$','alpha0_Z':r'$\alpha_Z^0$'},
                                                label_font_size=16))
            
            fig = c.plotter.plot()
    
            fname = os.path.join(config['output_path'],'figures','%s_%s_corner_samples.png'%(prefix,suffix))
            plt.savefig(fname,bbox_inches='tight',dpi=300,facecolor='w')
            print('saved %s'%fname,flush=True)
    
            plt.close('all')

        return warmup_samples,warmup_extra_fields,samples,samples_extra_fields


    def run_aies(num_walkers=50):

        ### start warmup
        
        mcmc = MCMC(AIES(model,randomize_split=True),
                    num_warmup=inference_config['num_warmup'],
                    num_samples=inference_config['num_samples'],num_chains=num_walkers,chain_method='vectorized')
        
        # Feb 4 -- make RNG chain_num so every chain combo has different initialization... 
        # TO DO: for mocks, put back in dependence on mock_num+chain_num for 
        mcmc.warmup(key(config['rng_init']), # random int offset just in case.. 
                    *model_args,
                    collect_warmup=True,)
        
        warmup_samples = mcmc.get_samples()    
        warmup_extra_fields = mcmc.get_extra_fields()
        
        
        print(mcmc.print_summary(),flush=True)
        
        
        warmup_samples_dict = {k: warmup_samples[k] for k in params_free}
        warmup_samples_arr = jnp.stack([warmup_samples[k] for k in params_free], axis=1)
        
        print(jnp.corrcoef(warmup_samples_arr.T),flush=True) # jnp.cov(warmup_samples_arr.T)
        
        
        warmup_samples_df = pd.DataFrame(warmup_samples_dict)
        # print(warmup_samples_df,flush=True)

        if savefigs is True:
        
            c = ChainConsumer()
            
            c.add_chain(Chain(samples=warmup_samples_df, name="An Example Contour"))
    
            # TO DO: change this to use user-provided prior limits
            c.set_plot_config(PlotConfig(extents={'A_M':(-2.1,1.1),'alpha0_M':(-4.1,0.1),
                                                        'A_E':(-2.1,0.1),'alpha0_E':(-4.1,0.1),
                                                        'A_SF':(-0.1,1.2),'alpha0_SF':(-4.1,0.1),
                                                        'A_Z':(-2.1,0.1),'alpha0_Z':(-4.1,0.1),},
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
                 *model_args,)
        
        samples = mcmc.get_samples()  
        samples_extra_fields = mcmc.get_extra_fields()
        
        print(mcmc.print_summary(),flush=True)
        
        
        samples_dict = {k: samples[k] for k in params_free}
        samples_df = pd.DataFrame(samples_dict)
        
        if savefigs is True:
            
            c = ChainConsumer()
            
            c.add_chain(Chain(samples=samples_df, name="An Example Contour"))
            
            c.set_plot_config(PlotConfig(extents={'A_M':(-2.1,1.1),'alpha0_M':(-4.1,0.1),
                                                        'A_E':(-2.1,0.1),'alpha0_E':(-4.1,0.1),
                                                        'A_SF':(-0.1,1.2),'alpha0_SF':(-4.1,0.1),
                                                        'A_Z':(-2.1,0.1),'alpha0_Z':(-4.1,0.1),},
                                                labels={'A_M':r'$A_M$','alpha0_M':r'$\alpha_M^0$',
                                                        'A_E':r'$A_E$','alpha0_E':r'$\alpha_E^0$',
                                                        'A_SF':r'$A_{\rm SF}$','alpha0_SF':r'$\alpha_{\rm SF}^0$',
                                                        'A_Z':r'$A_Z$','alpha0_Z':r'$\alpha_Z^0$'},
                                                label_font_size=16))
            
            fig = c.plotter.plot()
    
            fname = os.path.join(config['output_path'],'figures','%s_%s_corner_samples.png'%(prefix,suffix))
            plt.savefig(fname,bbox_inches='tight',dpi=300,facecolor='w')
            print('saved %s'%fname,flush=True)
    
            plt.close('all')

        return warmup_samples,warmup_extra_fields,samples,samples_extra_fields    


    ### finally call for the user input chain_num
    ### TO DO: consider a fori_loop or vmap over GPUs, though HMC is slow for our problem
    if inference_config['engine'] == 'nuts':
        out_numpyro = run_nuts(config['chain_num'])
    elif inference_config['engine'] == 'aies':
        out_numpyro = run_aies(inference_config['num_walkers'])


    if savenpz is True:
        
        warmup_samples,warmup_extra_fields,samples,samples_extra_fields = out_numpyro
    
        ##### Save output 
        # TO DO: bring back filename prefix/suffix (and push that to utils)
        fname = os.path.join(config['output_path'],'outputs','%s_%s.npz'%(prefix,suffix))
    
        # TO DO: move this to utils/write_results.py 
        jnp.savez(fname,
                  warmup_samples = warmup_samples,
                  warmup_extra_fields = warmup_extra_fields,
                  samples = samples,
                  samples_extra_fields = samples_extra_fields,
                  model_args = model_args, # this used to be called obs_stats (useful for predictive check plots) 
                  config=config)
        
        print('saved %s'%fname,flush=True)
    

    return out_numpyro




    
###