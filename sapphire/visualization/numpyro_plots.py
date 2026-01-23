"""
this module creates trace plot
"""

# plot-specific modules
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import arviz as az
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from chainconsumer import Chain, ChainConsumer, make_sample, PlotConfig, Truth, ChainConfig
plt.rcParams['figure.dpi'] = 120
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15
from functools import partial 
import multiprocess
from timeit import default_timer as timer
import os

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
from numpyro.diagnostics import gelman_rubin

import numpy as np # for np.ravel(axes), can probably be done away with

# for easy handling of numpyro chains 
import pandas as pd


def trace(output_path,basestr,savefig=False):
    ### output_path is the base sapphire output directory from config.yaml
    ### basestr should have %s in for chain num, e.g., '/path/sapphire_outputs/outputs/numpyro_manga_chain%s_111.npz'

    ### TO DO: automate this and/or have it become an input
    params_free = ["A_M","alpha0_M","A_E","alpha0_E","A_SF","alpha0_SF","A_Z","alpha0_Z"]
    parlabels = {'A_M':r'$A_M$','alpha0_M':r'$\alpha_M^0$',
                 'A_E':r'$A_E$','alpha0_E':r'$\alpha_E^0$',
                 'A_SF':r'$A_{\rm SF}$','alpha0_SF':r'$\alpha_{\rm SF}^0$',
                 'A_Z':r'$A_Z$','alpha0_Z':r'$\alpha_Z^0$'}
    parlims = {'A_M':(-2.1,1.1),'alpha0_M':(-2.1,0.1),
               'A_E':(-2.1,0.1),'alpha0_E':(-2.1,0.1),
               'A_SF':(-0.1,1.2),'alpha0_SF':(-2.1,0.1),
               'A_Z':(-2.1,0.1),'alpha0_Z':(-2.1,0.1)}
    

    # samples dataframes indexed by int for chain_num for Gelman-Rubin statistic 
    chain_samples = {}

    ### automate this for different numbers of free parameters
    fig, axes = plt.subplots(nrows=4,ncols=2,figsize=(12,6),constrained_layout=True)
    allax = np.ravel(axes)

    # TO DO: generalize depending on actual # of chains
    for cnum in range(0,4):

        fname = os.path.join(output_path,'outputs',basestr%cnum)
        npz_nuts = jnp.load(fname,allow_pickle=True)
    
        # print('cnum=%s, warmup init_params=%s'%(cnum,jnp.array([npz_nuts['warmup_samples'].item()[key][0] for key in params_free])))

        warmup_samples_dict = {k: npz_nuts['warmup_samples'].item()[k] for k in params_free}
        warmup_samples_df = pd.DataFrame(warmup_samples_dict)
        
        samples_dict = {k: npz_nuts['samples'].item()[k] for k in params_free}
        samples_df = pd.DataFrame(samples_dict)
        chain_samples[cnum] = samples_df

        ### merge warmup and then samples into single df 
        Nwarmup = len(warmup_samples_df) # for axvline divider (note: this is the same for every chain, so could be moved outside)
        combined_df = pd.concat([warmup_samples_df, samples_df],ignore_index=True)

        ### add chain for trace plot
        for i,k in enumerate(params_free):
            allax[i].plot(np.log10(1+np.arange(len(combined_df))),combined_df[k].values,
                          label=cnum,lw=0.5,alpha=0.5)
    
    ### compute gelman-rubin statistic per parameter across chains
    # first create stacked numpy array of shape (Nchain, Nsamples, Nparams) - format expected by numpyro's gelman_rubin
    chain_samples_stacked = np.stack([chain_samples[i].to_numpy() for i in range(0,4)],axis=0)
    # print(chain_samples_stacked.shape)

    # compute, print and plot gelman_rubin per parameter
    grstat = gelman_rubin(chain_samples_stacked)
    # print(grstat)
    
    for i,k in enumerate(params_free):
        allax[i].set_ylabel(parlabels[k],fontsize=16)
        allax[i].set_ylim(parlims[k])

        # divider between warmup and sampling
        allax[i].axvline(np.log10(1+Nwarmup),color='k',lw=3,alpha=0.5) 

        # plot gelman-rubin statistic per parameter in lower-left corner
        allax[i].text(0.01,0.03,r'$\hat{R}=%.4f$'%(grstat[i]),transform=allax[i].transAxes,fontsize=14)

    axes[3][0].set_xlabel('log iteration #',fontsize=16)
    axes[3][1].set_xlabel('log iteration #',fontsize=16)

    fig.suptitle(basestr.replace('_chain%s','').replace('.npz',''),fontsize=16)#,y=1.01)

    if savefig is True:
        outfile = os.path.join(output_path,'figures',
                               # since this is run dependent, for now just prepend "trace" to original filename 
                               # e.g., numpyro_manga_chain%s_111.npz becomes trace_numpyro_manga_111.png
                               # TO DO: how to improve/generalize this?
                              'trace_'+(basestr.replace('_chain%s','').replace('.npz','.png')))
        plt.savefig(outfile,bbox_inches='tight',dpi=300,facecolor='w')
        print('saved %s'%outfile,flush=True)


### helper for corner() and multicorner()
def add_parameterization_axes(fig,axes):
    ##### automatically create new subplots in top-right empty area of corner based on its existing subplots
    # TO DO: generalize this for different # of parameterizations and cornerplot size (depends on # free params)
    
    # bottom-right corner subplot
    pos_br = axes[7,7].get_position()
    
    x_right = pos_br.x1   # right edge of bottom-right subplot
    y_top   = axes[0,0].get_position().y1  # top edge of top-left subplot
    
    ax_width = 0.16
    ax_pad   = 0.05

    # set up a left and right column for the 4 subplots (for pandya23/26)
    x_col2 = x_right - ax_width # right column flush with corner plot
    x_col1 = x_col2 - ax_width - ax_pad # left column just left of col2
    
    ### add 2 rows, with one subplot in each column (this required fiddling...)
    # first row
    ax_etaM = fig.add_axes([x_col1, y_top - ax_width, ax_width, ax_width])
    ax_etaE = fig.add_axes([x_col2, y_top - ax_width, ax_width, ax_width])
    
    # second row
    ax_etaZ = fig.add_axes([x_col1, y_top - 2*ax_width - 0.8*ax_pad, ax_width, ax_width])
    ax_tdep = fig.add_axes([x_col2, y_top - 2*ax_width - 0.8*ax_pad, ax_width, ax_width])

    par_axes = [ax_etaM, ax_etaE, ax_tdep, ax_etaZ]

    return par_axes


# TO DO: make this standalone (create figure on its own)
# TO DO: when plotting on corner, generalize/automate this subplot sizing and positioning for different # of parameters 

def plot_parameterizations(par_axes,config,samples_df,params_free,color_nuts,Nrand=100,label='__none__'): 
    # config = dict should point to the same ODE model as used for inference (constraints, etc. doesn't matter)

    # TO DO: generalize
    ax_etaM, ax_etaE, ax_tdep, ax_etaZ = par_axes

    ##### first load the ODE's parameterized functions 
    # easiest to re-read in the config that determined ODE model setup
    # this should've been same regardless of inference constraints
    from sapphire.utils import read_config
    config = read_config.get({'path_config':'/mnt/ceph/users/vpandya/sapphire/scripts/config.yaml'})
    
    # now have the ODE model setup return list of parameterized functions
    # pandya23/26: every parameterized function has the same call signature: Vvir,redshift,A=3.0,alpha0=-3.0,alphaz=0.0,beta=-0.7    
    from sapphire.models import model_loader
    integrator, saveat_fn, list_parameterizations = model_loader.get(config,verbose=False)
    get_tdep, get_etaM, get_etaE, get_etaZ = list_parameterizations # user should check order and call signature for different models

    # get 100 random draws from posterior samples
    rand_samples_df = samples_df.sample(Nrand,random_state=1)
    rand_samples_arr = rand_samples_df.to_numpy()

    # define Vvir array
    # TO DO: change this to vmax after switching to vmax-based inference
    Vvir_arr = jnp.linspace(30,150)    

    ### use vmap over rand_samples_arr since each get_XXX function operates on a single set of parameters
    # TO DO: generalize this for different ODE models and numbers of parameterizations
    @jit
    @vmap
    def get_post_powerlaws(post_params):
        ### params = array of shape (Nrealizations, 8)
        
        post_etaM = get_etaM(Vvir_arr,redshift=0.0,A=10**post_params[0],alpha0=post_params[1],alphaz=0.0,beta=0.0)
        post_etaE = get_etaE(Vvir_arr,redshift=0.0,A=10**post_params[2],alpha0=post_params[3],alphaz=0.0,beta=0.0)
        post_tdep = get_tdep(Vvir_arr,redshift=0.0,A=10**post_params[4],alpha0=post_params[5],alphaz=0.0,beta=-0.7)
        post_etaZ = get_etaZ(Vvir_arr,redshift=0.0,A=10**post_params[6],alpha0=post_params[7],alphaz=0.0,beta=0.0)
    
        return post_etaM, post_etaE, post_tdep, post_etaZ

    # shape (Nsamples, len(Vvir_arr))
    post_etaM, post_etaE, post_tdep, post_etaZ = get_post_powerlaws(rand_samples_arr) 

    ## now plot random 'posterior' realizations in background as thin colored lines
    for i in range(post_etaM.shape[0]):
        
        ax_etaM.plot(Vvir_arr,post_etaM[i],alpha=0.3,lw=0.5,c=color_nuts,zorder=0,label=label if i==0 else '__none__')
        ax_etaE.plot(Vvir_arr,post_etaE[i],alpha=0.3,lw=0.5,c=color_nuts,zorder=0,label=label if i==0 else '__none__')
        ax_tdep.plot(Vvir_arr,post_tdep[i],alpha=0.3,lw=0.5,c=color_nuts,zorder=0,label=label if i==0 else '__none__')
        ax_etaZ.plot(Vvir_arr,post_etaZ[i],alpha=0.3,lw=0.5,c=color_nuts,zorder=0,label=label if i==0 else '__none__')
   

        
### function that combines all chains and makes cornerplot, using ChainConsumer
# TO DO: generalize for any # of chains, and allow plotting individual chains w/o combining for another convergence check
# TO DO: add optional Fisher MAP and covariance location (e.g., from adam)
def corner(output_path,basestr,config,savefig=False,title=False,Nrand=100):
    
    params_free = ["A_M","alpha0_M","A_E","alpha0_E","A_SF","alpha0_SF","A_Z","alpha0_Z"]
    chain_colors = ['#3b82f6','#10b981','#ef4444','#a855f7'] # blue, emerald, red, purple from chainconsumer default color_finder
    color_nuts = '#3b82f6'#'#6366f1' #'#3b82f6' #chain_colors[1]
    color_adam = '#ef4444' # '#a855f7' #'#f43f5e' # chain_colors[3]
    color_obs = '#f59e0b'
    
    # samples dataframes indexed by int for chain_num
    chains_nuts = {} 
    warmup_chains_nuts = {} # to assess convergence paths
    
    for cnum in range(0,4):
    
        fname = os.path.join(output_path,'outputs',basestr%cnum)
        npz_nuts = jnp.load(fname,allow_pickle=True)
    
        # print('cnum=%s, warmup init_params=%s'%(cnum,jnp.array([npz_nuts['warmup_samples'].item()[key][0] for key in params_free])))
    
        samples_dict = {k: npz_nuts['samples'].item()[k] for k in params_free}
        samples_df = pd.DataFrame(samples_dict)
        chains_nuts[cnum] = samples_df
    
        warmup_samples_dict = {k: npz_nuts['warmup_samples'].item()[k] for k in params_free}
        warmup_samples_df = pd.DataFrame(warmup_samples_dict)
        warmup_chains_nuts[cnum] = warmup_samples_df

    samples_df = pd.concat(chains_nuts.values(), ignore_index=True)    

    # initialize chainconsumer
    consumer = ChainConsumer()
    
    consumer.add_chain(Chain(samples=samples_df, name="NUTS chain 3",color=color_nuts,shade_alpha=0.3))

    # TO DO: generalize with user inputting these
    consumer.set_plot_config(PlotConfig(extents={'A_M':(-2.1,1.1),'alpha0_M':(-2.1,0.1),
                                                 'A_E':(-2.1,0.1),'alpha0_E':(-2.1,0.1),
                                                 'A_SF':(-0.1,1.2),'alpha0_SF':(-2.1,0.1),
                                                 'A_Z':(-2.1,0.1),'alpha0_Z':(-2.1,0.1),},
                                        labels={'A_M':r'$A_M$','alpha0_M':r'$\alpha_M^0$',
                                                'A_E':r'$A_E$','alpha0_E':r'$\alpha_E^0$',
                                                'A_SF':r'$A_{\rm SF}$','alpha0_SF':r'$\alpha_{\rm SF}^0$',
                                                'A_Z':r'$A_Z$','alpha0_Z':r'$\alpha_Z^0$'},
                                        label_font_size=16))

    # create plot
    fig = consumer.plotter.plot()  
    axes = np.reshape(fig.axes,(8,8))

    # add parameterization axes in top-right
    par_axes = add_parameterization_axes(fig,axes)
    ax_etaM, ax_etaE, ax_tdep, ax_etaZ = par_axes    

    ### add subplots in top-right empty area for parameterizations
    plot_parameterizations(par_axes,config,samples_df,params_free,color_nuts,Nrand)    

    for ax in [ax_etaM,ax_etaE,ax_tdep,ax_etaZ]:
        ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.set_xlabel(r'$V_{\rm vir}$ [km/s]',fontsize=20)
    
    ax_etaM.set_ylabel(r'$\eta_M$',fontsize=20)
    ax_etaE.set_ylabel(r'$\eta_E$',fontsize=20)
    ax_tdep.set_ylabel(r'$t_{\rm dep}$ [Gyr]',fontsize=20)
    ax_etaZ.set_ylabel(r'$\eta_Z$',fontsize=20)
    
    ax_etaM.set_ylim(1e-2,1e2)
    ax_etaE.set_ylim(1e-1,1.1)
    ax_etaZ.set_ylim(1e-2,1.1)
    ax_tdep.set_ylim(1e0,2e2)    
    
    if title is True:
        fig.suptitle(basestr.replace('_chain%s','').replace('.npz',''),fontsize=16)

    axes[0,0].text(1.6,0.6,'NUTS (HMC)',fontsize=42,color=color_nuts,transform=axes[0,0].transAxes,weight='heavy')
    # axes[0,0].text(1.6,0.25,'MAP (Fisher)',fontsize=42,color=color_adam,transform=axes[0,0].transAxes,weight='heavy')
    
    if savefig is True:
        outfile = os.path.join(output_path,'figures',
                               # since this is run dependent, for now just prepend "corner" to original filename 
                               # e.g., numpyro_manga_chain%s_111.npz becomes corner_numpyro_manga_111.png
                               # TO DO: how to improve/generalize this?
                              'corner_'+(basestr.replace('_chain%s','').replace('.npz','.png')))
        plt.savefig(outfile,bbox_inches='tight',dpi=300,facecolor='w')
        print('saved %s'%outfile,flush=True)


### TO DO: merge this with corner() by adding more arguments 
def multicorner(output_path,basestr,config,savefig=False,title=False,Nrand=100):
    ### basestr should have %s for chain num AND %s for flag (111,110,etc.), e.g., 'numpyro_manga_chain%s_%s.npz'
    
    params_free = ["A_M","alpha0_M","A_E","alpha0_E","A_SF","alpha0_SF","A_Z","alpha0_Z"]

    # initialize chainconsumer
    consumer = ChainConsumer()
    
    flag_colors = {'100':'teal','110':'indigo','111':'orange'} # amber instead of orange looks nicer in chainconsumer
    # flag_labels = {'100':r'SMHM','110':r'SMHM+f$_{\rm gas}$','111':r'SMHM+f$_{\rm gas}$+MZR'}
    flag_labels = {'100':r'SMHM','110':r'SMHM+f$_{\rm gas}$','111':r'SMHM+f$_{\rm gas}$+MZR'}
    
    ### loop over each realization flag 

    flag_samples = {}

    for flag in ['100','110','111']:

        flag_color = flag_colors[flag]
        flag_label = flag_labels[flag]

        # samples dataframes indexed by int for chain_num
        chains_nuts = {} 
        
        for cnum in range(0,4):
        
            fname = os.path.join(output_path,'outputs',basestr%(cnum,flag))
            npz_nuts = jnp.load(fname,allow_pickle=True)
        
            samples_dict = {k: npz_nuts['samples'].item()[k] for k in params_free}
            samples_df = pd.DataFrame(samples_dict)
            chains_nuts[cnum] = samples_df

        samples_df = pd.concat(chains_nuts.values(), ignore_index=True)    
        flag_samples[flag] = samples_df # for plotting parametrizations below
    
        consumer.add_chain(Chain(samples=samples_df, name=flag_label,color=flag_color,shade_alpha=0.3))

    # TO DO: generalize with user inputting these
    consumer.set_plot_config(PlotConfig(extents={'A_M':(-2.1,1.1),'alpha0_M':(-2.1,0.1),
                                                 'A_E':(-2.1,0.1),'alpha0_E':(-2.1,0.1),
                                                 'A_SF':(-0.1,1.2),'alpha0_SF':(-2.1,0.1),
                                                 'A_Z':(-2.1,0.1),'alpha0_Z':(-2.1,0.1),},
                                        labels={'A_M':r'$A_M$','alpha0_M':r'$\alpha_M^0$',
                                                'A_E':r'$A_E$','alpha0_E':r'$\alpha_E^0$',
                                                'A_SF':r'$A_{\rm SF}$','alpha0_SF':r'$\alpha_{\rm SF}^0$',
                                                'A_Z':r'$A_Z$','alpha0_Z':r'$\alpha_Z^0$'},
                                        label_font_size=16,
                                        legend_location=(0,0),legend_kwargs={'bbox_to_anchor':(3.5,1.0),'fontsize':30}))

    # create plot
    fig = consumer.plotter.plot()  
    axes = np.reshape(fig.axes,(8,8))

    # add parameterization axes in top-right
    par_axes = add_parameterization_axes(fig,axes)
    ax_etaM, ax_etaE, ax_tdep, ax_etaZ = par_axes

    ### add subplots in top-right empty area for parameterizations
    for flag_num,flag in enumerate(['100','110','111']):
        plot_parameterizations(par_axes,config,flag_samples[flag],params_free,flag_colors[flag],Nrand,flag_labels[flag])

    for ax in [ax_etaM,ax_etaE,ax_tdep,ax_etaZ]:
        ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.set_xlabel(r'$V_{\rm vir}$ [km/s]',fontsize=20)
    
    ax_etaM.set_ylabel(r'$\eta_M$',fontsize=20)
    ax_etaE.set_ylabel(r'$\eta_E$',fontsize=20)
    ax_tdep.set_ylabel(r'$t_{\rm dep}$ [Gyr]',fontsize=20)
    ax_etaZ.set_ylabel(r'$\eta_Z$',fontsize=20)
    
    ax_etaM.set_ylim(1e-2,1e2)
    ax_etaE.set_ylim(1e-1,1.1)
    ax_etaZ.set_ylim(1e-2,1.1)
    ax_tdep.set_ylim(1e0,2e2)
    
    # leg = ax_tdep.legend(loc='best',fancybox=True,framealpha=0,fontsize=14)
    # leg.get_lines()[0].set_alpha(1.0)
    
    if title is True:
        fig.suptitle(basestr.replace('_chain%s','').replace('_%s','').replace('.npz',''),fontsize=16)

    # axes[0,0].text(1.6,0.6,'NUTS (HMC)',fontsize=42,color=color_nuts,transform=axes[0,0].transAxes,weight='heavy')
    # axes[0,0].text(1.6,0.25,'MAP (Fisher)',fontsize=42,color=color_adam,transform=axes[0,0].transAxes,weight='heavy')
    
    if savefig is True:
        outfile = os.path.join(output_path,'figures',
                               # since this is run dependent, for now just prepend "corner" to original filename 
                               # e.g., numpyro_manga_chain%s_111.npz becomes corner_numpyro_manga_111.png
                               # TO DO: how to improve/generalize this?
                              'multicorner_'+(basestr.replace('_chain%s','').replace('_%s','').replace('.npz','.png')))
        plt.savefig(outfile,bbox_inches='tight',dpi=300,facecolor='w')
        print('saved %s'%outfile,flush=True)




### function that overplots marginal posteriors from multi-constraint runs 
# TO DO: generalize this for any number of free parameters, multi-constraint runtypes, number of chains, etc.
def marginals(output_path,basestr,savefig=False):
    ### output_path is the base sapphire output directory from config.yaml
    ### basestr should have %s for chain num AND %s for flag (111,110,etc.), e.g., 'numpyro_manga_chain%s_%s.npz'

    ### TO DO: automate this and/or have it become an input
    params_free = ["A_M","alpha0_M","A_E","alpha0_E","A_SF","alpha0_SF","A_Z","alpha0_Z"]
    parlabels = {'A_M':r'$A_M$','alpha0_M':r'$\alpha_M^0$',
                 'A_E':r'$A_E$','alpha0_E':r'$\alpha_E^0$',
                 'A_SF':r'$A_{\rm SF}$','alpha0_SF':r'$\alpha_{\rm SF}^0$',
                 'A_Z':r'$A_Z$','alpha0_Z':r'$\alpha_Z^0$'}
    parlims = {'A_M':(-2.1,1.1),'alpha0_M':(-2.1,0.1),
               'A_E':(-2.1,0.1),'alpha0_E':(-2.1,0.1),
               'A_SF':(-0.1,1.2),'alpha0_SF':(-2.1,0.1),
               'A_Z':(-2.1,0.1),'alpha0_Z':(-2.1,0.1)}

    # par_labels = [r'$A_M$',r'$\alpha_M^0$',r'$A_E$',r'$\alpha_E^0$',r'$A_{\rm SF}$',r'$\alpha_{\rm SF}^0$',r'$A_Z$',r'$\alpha_Z^0$']

    # initialize 2-row figure for the 8 parameters [TO DO: GENERALIZE THIS]
    fig, axes = plt.subplots(nrows=2,ncols=4,figsize=(13,5),constrained_layout=True)
    allax = np.ravel(axes)
    
    flag_colors = {'100':'teal','110':'indigo','111':'orange'}
    # flag_labels = {'100':r'SMHM','110':r'SMHM+f$_{\rm gas}$','111':r'SMHM+f$_{\rm gas}$+MZR'}
    flag_labels = {'100':r'SMHM','110':r'SMHM+f$_{\rm gas}$','111':r'SMHM+f$_{\rm gas}$+MZR'}
    
    ### loop over each realization flag 

    for flag in ['100','110','111']:

        flag_color = flag_colors[flag]
        flag_label = flag_labels[flag]

        ### loop over and store posterior samples from all 4 chains    
        # samples dataframes indexed by int for chain_num for Gelman-Rubin statistic 
        chain_samples = {}

        # TO DO: generalize depending on actual # of chains
        for cnum in range(0,4):

            # TO DO: generalize this beyond numpyro_manga_chain%s_%s.npz
            fname = os.path.join(output_path,'outputs',basestr%(cnum,flag))
            npz_nuts = jnp.load(fname,allow_pickle=True)
        
            samples_dict = {k: npz_nuts['samples'].item()[k] for k in params_free}
            samples_df = pd.DataFrame(samples_dict)
            chain_samples[cnum] = samples_df

        # combine the different chains' posterior samples
        samples_df = pd.concat(chain_samples.values(), ignore_index=True)  

        # plot this realization's posterior for each parameter in its respective subplot
        for i,k in enumerate(parlabels.keys()):

            sns.kdeplot(samples_df[k].values,color=flag_color,ax=allax[i],lw=3,alpha=0.7,label=flag_label)
            allax[i].set_xlim(parlims[k])
            allax[i].set_xlabel(parlabels[k],fontsize=16)
            allax[i].set_ylabel('Posterior',fontsize=14)

    leg = allax[2].legend(fancybox=True)
    
    
    if savefig is True:
        outfile = os.path.join(output_path,'figures',
                               # since this is run dependent, for now just prepend "trace" to original filename 
                               # e.g., numpyro_manga_chain%s_111.npz becomes trace_numpyro_manga_111.png
                               # TO DO: how to improve/generalize this?
                              'marginals_'+(basestr.replace('_chain%s','').replace('_%s','').replace('.npz','.png')))
        plt.savefig(outfile,bbox_inches='tight',dpi=300,facecolor='w')
        print('saved %s'%outfile,flush=True)


### function that overplots random realizations 
### TO DO: generalize this, and maybe move to another module (which calls sapphire and extracts arbitrary params/outputs)
def posterior_predictive_checks(output_path,basestr,obs_stats=None,savefig=False):
    ### output_path is the base sapphire output directory from config.yaml
    ### basestr should have %s for chain num AND %s for flag (111,110,etc.), e.g., 'numpyro_manga_chain%s_%s.npz'
    ### obs_stats -- either passed or (if None) extracted from numpyro output npz file
    ######### TO DO: make obs_stats a dict
    
    ### TO DO: generalize this for future more complicated flags or other constraint dtypes
    def get_predictives(flag,obs_stats):
    
        chain_preds_smhm, chain_preds_fgas, chain_preds_mzr = [],[],[]
    
        # TO DO: generalize depending on actual # of chains
        for cnum in range(0,4):
        
            # TO DO: generalize this beyond numpyro_manga_chain%s_%s.npz
            fname = os.path.join(output_path,'outputs',basestr%(cnum,flag))
            npz_nuts = jnp.load(fname,allow_pickle=True)
    
            # just take the random posterior realizations without errorbars 
            pred_avg_smhm = npz_nuts['samples'].item()['pred_avg_smhm']
            pred_avg_fgas = npz_nuts['samples'].item()['pred_avg_fgas']
            pred_avg_mzr = npz_nuts['samples'].item()['pred_avg_mzr']
        
            chain_preds_smhm.append(pred_avg_smhm)
            chain_preds_fgas.append(pred_avg_fgas)
            chain_preds_mzr.append(pred_avg_mzr)

        ### if obs_stats is None, use the (already-open) last npz file to extract obs_stats
        # TO DO: generalize this for arbitrary summary statistics, and perhaps move to another function outside
        if obs_stats == None:
            # consider saving as dict['obs_stats'] = tuple to prevent npz error, and make it easier to load here/elsewhere
            obs_stats = (npz_nuts['obs_x0_smhm'],npz_nuts['obs_bw_smhm'],npz_nuts['obs_avg_smhm'],npz_nuts['obs_err_smhm'],
                         npz_nuts['obs_x0_fgas'],npz_nuts['obs_bw_fgas'],npz_nuts['obs_avg_fgas'],npz_nuts['obs_err_fgas'],
                         npz_nuts['obs_x0_mzr'],npz_nuts['obs_bw_mzr'],npz_nuts['obs_avg_mzr'],npz_nuts['obs_err_mzr'])

        # if we didn't use obs_stats from file, then the input one will be returned
        return jnp.vstack(chain_preds_smhm), jnp.vstack(chain_preds_fgas), jnp.vstack(chain_preds_mzr), obs_stats
    
    preds_smhm_111, preds_fgas_111, preds_mzr_111, obs_stats = get_predictives('111',obs_stats)
    preds_smhm_110, preds_fgas_110, preds_mzr_110, obs_stats = get_predictives('110',obs_stats)
    preds_smhm_100, preds_fgas_100, preds_mzr_100, obs_stats = get_predictives('100',obs_stats)

    ##### PLOT
    ### TO DO: generalize this for arbitrary predictives 
    color_obs = 'k'
    color_111, color_110, color_100 = 'orange', 'indigo', 'teal'
    
    fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(11,3),constrained_layout=True)
    ax_smhm, ax_fgas, ax_mzr = axes
    
    ##### first the data/constraints
    ax_smhm.errorbar(obs_stats[0],obs_stats[2],yerr=obs_stats[3],
                 fmt='-',color=color_obs,capsize=3,zorder=np.inf)

        
    # axes[0].set_xscale('log')
    ax_smhm.set_xlabel(r'$\log M_{\rm vir}/M_{\odot}$ (z=0)',fontsize=14)
    ax_smhm.set_ylabel(r'$\log M_*/M_{\rm vir}$ (z=0)',fontsize=14)
    
    ax_fgas.errorbar(obs_stats[4],obs_stats[6],yerr=obs_stats[7],
                     fmt='-',capsize=3,zorder=np.inf,color=color_obs,label='data')
    
    # plt.xscale('log')
    ax_fgas.set_xlabel(r'$\log M_*/M_{\odot}$ (z=0)',fontsize=14)
    ax_fgas.set_ylabel(r'$\log M_{\rm ISM}/M_*$ (z=0)',fontsize=14)
    
    ax_mzr.errorbar(obs_stats[8],obs_stats[10],yerr=obs_stats[11],
                     fmt='-',capsize=3,zorder=np.inf,color=color_obs)
    
    ax_mzr.set_xlabel(r'$\log M_*/M_{\odot}$ (z=0)',fontsize=14)
    ax_mzr.set_ylabel(r'$\log Z_*/Z_{\odot}$ (z=0)',fontsize=14)
    
    ##### plot random draws from posterior
    drawnums = jax.random.choice(key(1),jnp.arange(preds_smhm_111.shape[0]),(100,),replace=False)
    # drawnums = np.arange(100)
    
    for inum,ival in enumerate(drawnums):
        
        ax_smhm.plot(obs_stats[0],preds_smhm_111[ival],'-',color=color_111,alpha=0.2,lw=1)
        ax_smhm.plot(obs_stats[0],preds_smhm_110[ival],'-',color=color_110,alpha=0.2,lw=1)
        ax_smhm.plot(obs_stats[0],preds_smhm_100[ival],'-',color=color_100,alpha=0.2,lw=1)
    
        ax_fgas.plot(obs_stats[4],preds_fgas_111[ival],'-',color=color_111,alpha=0.2,lw=1,
                     label=r'SMHM+f$_{\rm gas}$+MZR' if inum==0 else '__none__')
        ax_fgas.plot(obs_stats[4],preds_fgas_110[ival],'-',color=color_110,alpha=0.2,lw=1,label=r'SMHM+f$_{\rm gas}$' if inum==0 else '__none__')
        ax_fgas.plot(obs_stats[4],preds_fgas_100[ival],'-',color=color_100,alpha=0.2,lw=1,label=r'SMHM' if inum==0 else '__none__')
    
        ax_mzr.plot(obs_stats[8],preds_mzr_111[ival],'-',color=color_111,alpha=0.2,lw=1)
        ax_mzr.plot(obs_stats[8],preds_mzr_110[ival],'-',color=color_110,alpha=0.2,lw=1)
        ax_mzr.plot(obs_stats[8],preds_mzr_100[ival],'-',color=color_100,alpha=0.2,lw=1)
    
    leg = ax_fgas.legend(fontsize=8.5,fancybox=True,framealpha=1,ncol=2,loc='lower center')
    leg.get_lines()[0].set_alpha(1.0)
    

    ### if requested, save figure
    if savefig is True:
        outfile = os.path.join(output_path,'figures',
                               # since this is run dependent, for now just prepend "trace" to original filename 
                               # e.g., numpyro_manga_chain%s_111.npz becomes trace_numpyro_manga_111.png
                               # TO DO: how to improve/generalize this?
                              'posterior_predictives_'+(basestr.replace('_chain%s','').replace('_%s','').replace('.npz','.png')))
        plt.savefig(outfile,bbox_inches='tight',dpi=300,facecolor='w')
        print('saved %s'%outfile,flush=True)
        
    return fig, axes
            
    ### 