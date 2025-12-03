"""
this module plots the results of adam map+fisher mock coverage tests
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

import numpy as np # for np.ravel(axes), can probably be done away with


# TO DO: generalize this to operate over arbitrary number of params (columns) and constraints (rows)
def predicted_vs_true_parameters(constraint_results,constraint_labels,constraint_colors,par_labels,output_path):

    # adapted from https://github.com/maho3/ltu-ili/blob/44522f47c37f69b74313e80c7f1f04600472d116/ili/validation/metrics.py#L471
    
    # TO DO: generalize this based on params_bounds from config.yaml
    axlims8 = {0:(-2.1,1.1),1:(-2.1,0.1),2:(-2.1,0.1),3:(-2.1,0.1),4:(-0.1,1.2),5:(-2.1,0.1),6:(-2.1,0.1),7:(-2.1,0.1)}
    
    fig, axes = plt.subplots(nrows=3,ncols=8, figsize=(24, 9),constrained_layout=True)
    
    allax = np.ravel(axes)
    
    out100 = constraint_results[0] # only smhm
    out110 = constraint_results[1] # with fgas
    out111 = constraint_results[2] # with mzr

    # TO DO: generalize this for arbitrary number of parameters
    for j in range(8):
    
        # unpack into mus, truths, sigmas
        mus100 = out100[:, 0, j]
        mus110 = out110[:, 0, j]
        mus111 = out111[:, 0, j]
        
        sigmas100 = out100[:, 1, j]
        sigmas110 = out110[:, 1, j]
        sigmas111 = out111[:, 1, j]
    
        truths100 = out100[:, 2, j]
        truths110 = out110[:, 2, j]
        truths111 = out111[:, 2, j]    
        
        # for points that didn't converge to minimum (non-invertible fisher = nan errorbar) , plot as unfilled marker w/o errorbar
        # this identifies such saddle points by checking whether sigmas is nan (since Fisher = inverse Hessian undefined)
        mask100 = jnp.isfinite(sigmas100) 
        mask110 = jnp.isfinite(sigmas110) 
        mask111 = jnp.isfinite(sigmas111)
    
        # minima w/ fisher errors
        axes[0][j].errorbar(truths100[mask100], mus100[mask100], sigmas100[mask100],
                        fmt="o", elinewidth=0.5, alpha=0.8,color='teal',mfc='none',capsize=2,mew=0.5)
        axes[1][j].errorbar(truths110[mask110], mus110[mask110], sigmas110[mask110],
                            fmt="o", elinewidth=0.5, alpha=0.8,color='indigo',mfc='none',capsize=2,mew=0.5)
        axes[2][j].errorbar(truths111[mask111], mus111[mask111], sigmas111[mask111],
                            fmt="o", elinewidth=0.5, alpha=0.8,color='orange',mfc='none',capsize=2,mew=0.5)
    
        ### non-minima 
        axes[0][j].plot(truths100[~mask100], mus100[~mask100],'x',color='teal',mew=0.5,alpha=0.8)
        axes[1][j].plot(truths110[~mask110], mus110[~mask110],'x',color='indigo',mew=0.5,alpha=0.8)
        axes[2][j].plot(truths111[~mask111], mus111[~mask111],'x',color='orange',mew=0.5,alpha=0.8)
    
    for j in range(8):
    
        for irow in range(3):
            
            axes[irow][j].plot(
                *(2 * [np.linspace(axlims8[j][0], axlims8[j][1], 10)]),
                'k--', ms=0.2, lw=2,zorder=np.inf)
            axes[irow][j].set_xlabel('True',fontsize=16)
        
            axes[irow][j].set_xlim(axlims8[j])
            axes[irow][j].set_ylim(axlims8[j])
    
        axes[0][j].set_title(par_labels[j], fontsize=24,y=1.03)
    
    axes[0][0].set_ylabel('Predicted',fontsize=16)
    axes[1][0].set_ylabel('Predicted',fontsize=16)
    axes[2][0].set_ylabel('Predicted',fontsize=16)
    
    for irow in range(3):
    
        axes[irow][-1].yaxis.set_label_position('right')
        axes[irow][-1].set_ylabel(constraint_labels[irow],fontsize=24,color=constraint_colors[irow])

    outfile = os.path.join(output_path,'figures','coverage_adam_predicted_vs_true_parameters.png')
    plt.savefig(outfile,bbox_inches='tight',dpi=300,facecolor='w')

    print('saved %s'%outfile,flush=True)


def percentile_fishers(sigmas):

    # apply mask 
    mask = jnp.isfinite(sigmas)
    masked_sigmas = sigmas[mask]

    # compute percentiles of maskd sigmas 
    p16_sigmas = jnp.nanpercentile(masked_sigmas,16,axis=0)
    p50_sigmas = jnp.nanpercentile(masked_sigmas,50,axis=0)
    p84_sigmas = jnp.nanpercentile(masked_sigmas,84,axis=0)
    
    return p16_sigmas, p50_sigmas, p84_sigmas   

def percentile_biases(mus,truths,sigmas_true):

    # accuracy/bias statistic 
    biases = (mus-truths)/(sigmas_true)
    
    # apply mask 
    mask = jnp.isfinite(sigmas_true) # should always be defined for truth but just in case
    masked_sigmas = sigmas_true[mask]
    masked_biases = biases[mask]
    
    # compute percentiles of masked sigmas 
    p16_biases = jnp.nanpercentile(masked_biases,16,axis=0)
    p50_biases = jnp.nanpercentile(masked_biases,50,axis=0)
    p84_biases = jnp.nanpercentile(masked_biases,84,axis=0)
    
    return p16_biases, p50_biases, p84_biases


def compute_f_fail(out_run):
    sigmas = out_run[:,1,:]
    Nfail = jnp.sum(~jnp.all(jnp.isfinite(sigmas),axis=1))
    return Nfail/len(sigmas)


def fisher_cond(corr_matrices):
    conds = jnp.asarray([jnp.linalg.cond(corr_matrices[i]) for i in range(len(corr_matrices))])
    return jnp.nanpercentile(conds,16),jnp.nanpercentile(conds,50),jnp.nanpercentile(conds,84)


def summaries(constraint_results,constraint_labels,constraint_colors,constraint_flags,constraint_corrs,par_labels,output_path):

    # Define x positions for the labels (parameters)
    xvals = jnp.arange(len(par_labels))
    
    ### TO DO: automate the figsize for different numbers of constraints/parameters
    fig, axes = plt.subplots(nrows=4,ncols=1,figsize=(6,8),constrained_layout=True,dpi=120)
    
    ax_precision = axes[0]
    ax_accuracy = axes[1]
    ax_fail = axes[2]
    ax_cond = axes[3]
    
    # TO DO: automate this for different # of constraints/parameters (or let user adapt on their own...)
    constraint_shift = {0: -0.2, 1: 0.0, 2: 0.2}  # SMHM, +fgas, +mzr
    
    ### precision and accuracy -- loop over each constraint type
    
    for constraint_num, constraint_flag in enumerate(constraint_flags):
    
        # extract this constraint's stuff
        constraint_result = constraint_results[constraint_num]
        constraint_label = constraint_labels[constraint_num]
        constraint_color = constraint_colors[constraint_num]
    
        # general x-shift between parameters for precision and accuracy subplots
        shift = constraint_shift[constraint_num] + 0.1
    
        ### loop over each parameter for precision and accuracy
        for j in range(constraint_result.shape[2]):
    
            ### precision (sigma_fisher)
            p16, p50, p84 = percentile_fishers(constraint_result[:, 1, j])
            ax_precision.errorbar(xvals[j] + shift, float(p50),
                                  yerr=[[p50 - p16], [p84 - p50]],
                                  fmt='-8',color=constraint_color,
                                  capsize=4, lw=1.5, alpha=0.8, mfc='none',
                                  ms=4, mew=0.5,label=constraint_label if j == 0 else '__none__')
    
            ### accuracy (bias) 
            p16, p50, p84 = percentile_biases(constraint_result[:, 0, j],constraint_result[:, 2, j],constraint_result[:, 3, j])
            ax_accuracy.errorbar(xvals[j] + shift, float(p50),
                                 yerr=[[p50 - p16], [p84 - p50]],
                                 fmt='-8',color=constraint_color,
                                 capsize=4, lw=1.5, alpha=0.8, mfc='none',
                                 ms=4, mew=0.5,label='__none__')
    
    ### precision and accuracy subplot labels, limits, etc. 
    
    # first common aesthetics for precision and accuracy
    for ax in [ax_precision,ax_accuracy]:
    
        ax.errorbar(par_labels,np.full(len(par_labels),0),alpha=0)
        ax.set_xticklabels(par_labels, rotation=0,fontsize=12)
        ax.set_xlabel('Astrophysical Parameter',fontsize=12)
    
    ax_precision.legend(loc='lower left',fontsize=7,fancybox=True,framealpha=0,ncol=3,bbox_to_anchor=(0,-0.05))
    ax_precision.set_ylabel(r'$\mathcal{P}\;\left(\sigma_{\rm Fisher}\right)$',fontsize=14)
    ax_precision.set_yscale('log')
    ax_precision.set_ylim(1e-3,1e1)
    
    ax_accuracy.set_ylabel(r'$\mathcal{P}\;\left(\frac{\theta_{\rm MAP}-\theta_{\rm true}}{\sigma_{\rm Fisher}^{\rm true}}\right)$',
                           fontsize=14)
    ax_accuracy.set_yscale('symlog',linthresh=1e-1)
    ax_accuracy.set_ylim(-10, 10)
    ax_accuracy.axhline(0.0, color='k', ls='-', alpha=0.1, zorder=0, lw=3,label='__none__')
    ax_accuracy.axhspan(-1,1, facecolor='k', ls='-', alpha=0.1, zorder=0, lw=3,edgecolor='none',
                        label=r'$\pm1\sigma_{\rm Fisher}^{\rm true}$',)
    ax_accuracy.legend(loc='upper center',fontsize=10,fancybox=True,framealpha=0,bbox_to_anchor=(0.5,1.07))
    
    
    ### failure (saddle) rate
    constraint_f_fail = [compute_f_fail(constraint_results[i]) for i in range(len(constraint_results))]
    
    for constraint_num,constraint_color in enumerate(constraint_colors):
    
        ax_fail.plot(constraint_labels[constraint_num],constraint_f_fail[constraint_num],'8',
                     color=constraint_color,mfc='none',ms=8,mew=2)
    
    ax_fail.plot(list(constraint_labels),constraint_f_fail,'k--',lw=1,alpha=0.5) 
    
    ax_fail.set_ylim(-0.05,1.05)
    ax_fail.set_ylabel(r'$\mathscr{F}_{\rm \;saddle}^{\rm \;fail}$',fontsize=14)
    ax_fail.set_xlabel('Constraints',fontsize=12)
    ax_fail.tick_params(axis='x',labelsize=12)
    
    ### degeneracy strength (fisher matrix condition number)
    constraint_conds = [fisher_cond(constraint_corrs[i]) for i in range(len(constraint_corrs))]
    
    for constraint_num, constraint_color in enumerate(constraint_colors):
    
        cond = constraint_conds[constraint_num]
    
        ax_cond.errorbar(constraint_labels[constraint_num], cond[1],
                         yerr=[[cond[1]-cond[0]], [cond[2]-cond[1]]],
                         fmt='8', color=constraint_color, mfc='none', ms=8, mew=2)
    
    ax_cond.plot(constraint_labels,[constraint_conds[i][1] for i in range(3)],'k--',lw=1,alpha=0.5) # July 8
    ax_cond.tick_params(axis='x',labelsize=12)
    
    ax_cond.set_ylabel(r'$\kappa_{\rm Fisher}$', fontsize=14)
    ax_cond.set_xlabel('Constraints', fontsize=12)
    ax_cond.set_yscale('log')
    ax_cond.set_ylim(1e2,None)
    
    # Add a thick downward arrow using ax.arrow
    ax_cond.annotate('', xy=(0.05, 0.2), xytext=(0.05, 0.5), xycoords='axes fraction',
                     arrowprops=dict(arrowstyle='simple', color='gray', lw=2,))
    
    # Add the text to the right of the arrow
    ax_cond.text(0.1, 0.35,r'lower $\kappa_{\rm Fisher}$' + '\n' + r'$\Rightarrow$ weaker' + '\n' + r'degeneracies',
                 transform=ax_cond.transAxes,fontsize=11, color='gray', va='center')
    
    ax_cond.text(0.99,0.8,'16-50-84 percentiles of\nthe condition number of Fisher matrices',
                 transform=ax_cond.transAxes,fontsize=8,ha='right')

    outfile = os.path.join(output_path,'figures','coverage_adam_summaries.png')
    plt.savefig(outfile,bbox_inches='tight',dpi=300,facecolor='w')

    print('saved %s'%outfile,flush=True)


# TO DO: generalize to operate over arbitrary number of parameters/constraints
# currently this is hard-coded for just the 3-constraint, 100 mocks of Pandya+26
def plot(output_path,mocknums,constraint_flags,constraint_labels,constraint_colors,par_labels):

    Npars = len(par_labels)
    
    ### nested function that returns trace_df and final point estimate +/- fisher sigma for each of the 3 constraints
    def read_mocks(mocknum):
    
        # TO DO: have constraint_flag (and par_labels) be an input (compatible with multiprocess.map)
        fname = os.path.join(output_path,'outputs','mock%s_%s.npz'%(mocknum,constraint_flag))
    
        npz = jnp.load(fname,allow_pickle=True)
    
        # best adam parameters
        mus = npz['theta_map']
        
        ### sqrt of inverse Fisher matrix diagonals
        Finv = npz['Finv_adam']
    
        # May 22: deal with when Finv_adam = nan due to saddle
        ### June 3 -- add correlation coefficient for off-diagonal degeneracies of Fisher
        try:
            sigmas = jnp.sqrt(jnp.diag(Finv))
            corr_matrix = Finv / jnp.outer(sigmas,sigmas)
        except:
            sigmas = jnp.full((Npars,), jnp.nan)
            corr_matrix = jnp.full((Npars, Npars), jnp.nan)        
    
        # mock truth parameter values
        truths = npz['free_params_arr']
    
        ### sqrt diagonals of TRUE Fisher -- this is what we normalize theta_map-theta_true by 
        Finv_true = npz['Finv_true']
    
        try: 
            sigmas_true = jnp.sqrt(jnp.diag(Finv_true))
        except:
            sigmas_true = jnp.full((Npars,), jnp.nan)
    
        # July 8 -- ratio of best loss from adam / true loss, note this includes saddles but we can filter later 
        # actually since these are negative log-posteriors, makes more sense to subtract best minus true 
        loss_ratio = float(npz['best_adam_loss']) - float(npz['true_loss'])
    
        # filter here -- set loss ratio to nan for saddles for plotting/nan-averaging purposes
        if not jnp.all(jnp.isfinite(sigmas)):
            loss_ratio = jnp.nan
        
        # return trace_dfs, mus, sigmas, truths
        return mus, sigmas, truths, sigmas_true, corr_matrix, loss_ratio

    
    ### loop over and store outputs with multiprocess
    constraint_results = []
    constraint_corrs = []
    constraint_Lratios = [] # loss ratios best_adam/true_loss
    
    tstart0 = timer()
    
    for constraint_flag in constraint_flags:
        tstart = timer()
        with multiprocess.Pool(40) as pool:
            # shape (Nmocks, 3 [mus/sigmas/truths], Nparams)
            
            out_constraint = pool.map(read_mocks, mocknums)
            constraint_results.append(jnp.asarray([out_constraint[i][:4] for i in mocknums]))
            constraint_corrs.append(jnp.asarray([out_constraint[i][4] for i in mocknums]))
            constraint_Lratios.append(jnp.asarray([out_constraint[i][5] for i in mocknums]))
            
        print('finished constraint_flag=%s in %.3f sec'%(constraint_flag,timer()-tstart))
    
    print('finished all runs in %.3f sec'%(timer()-tstart0))
    
    ### save predicted vs true plot
    predicted_vs_true_parameters(constraint_results,constraint_labels,constraint_colors,par_labels,output_path)

    ### save summary plot
    summaries(constraint_results,constraint_labels,constraint_colors,constraint_flags,constraint_corrs,par_labels,output_path)

    

### 