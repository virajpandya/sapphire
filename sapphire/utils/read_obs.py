"""
this module reads pre-summarized observational constraints from sapphire/data
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
from astropy.table import Table
from functools import partial 
import multiprocess
from timeit import default_timer as timer
import os, sys

# in case user loads module separately from sapphire.run()
from jax import config as jax_config
jax_config.update("jax_enable_x64", True) # required to accurately solve and take gradients through our diffeqs 

import jax
import jax.numpy as jnp


### gaussian kernel regression summary stats npz file for Pandya+26
def read_manga(config): # Jan 16 - shift is to test systematic shifts for inference

    # Jan 22 - for backward compatibility before I added these forecasting keys
    for shiftkey in ['shift_smhm','shift_fgas','shift_mzr']:
        if shiftkey not in config:
            config[shiftkey] = 0.0
    
    npz = jnp.load(os.path.join(config['data_path'],'obs/manga/obs_stats_manga.npz'),allow_pickle=True)

    ### Jan 21 2026 -- test impact of systematic shift up or down for SMHM using config['shift_XXX']

    # NOTE: the return order should be same as expected by sapphire.inference modules (for the "obs_stats" collection)
    return (jnp.asarray(npz['obs_x0_smhm']),
            npz['obs_bw_smhm'].item(),
            jnp.asarray(npz['obs_avg_smhm'])+config['shift_smhm'], # shift_XXX=0.0 is default
            jnp.asarray(npz['obs_err_smhm']),
            jnp.asarray(npz['obs_x0_Rgas']),
            npz['obs_bw_Rgas'].item(),
            jnp.asarray(npz['obs_avg_Rgas'])+config['shift_fgas'],
            jnp.asarray(npz['obs_err_Rgas']),
            jnp.asarray(npz['obs_x0_mzr']),
            npz['obs_bw_mzr'].item(),
            jnp.asarray(npz['obs_avg_mzr'])+config['shift_mzr'],
            jnp.asarray(npz['obs_err_mzr']))


### literature tables (these are unclear about uncertainties vs. scatter)
def read_lit(config):

    ### SMHM: behroozi+19
    tum = Table.read(os.path.join(config['data_path'],'obs/lit/smhm_a1.002312.dat'),format='ascii')
    tum = tum[(tum['HM(0)']>=10.0) & (tum['HM(0)']<=12.1)]    
    # tum = tum[tum['HM(0)']>=11.3] # optional to be same as manga 
    
    ### ISM fgas: peeples+14 (this is scatter, not standard on mean)
    tfgas = Table.read(os.path.join(config['data_path'],'obs/lit/peeples14.fgas.dat'),format='ascii')
    tfgas = tfgas[tfgas['logmstar']<11] # July 28 since we're not going so high anyway
    # tfgas = tfgas[tfgas['logmstar']>=9.0] # optional to be same as manga
    
    ### stellar MZR: Gallazzi+05 (this is scatter, not standard on mean)
    tmzr = Table.read(os.path.join(config['data_path'],'obs/lit/stellarmet.gallazzi.dat'),format='ascii')
    tmzr = tmzr[tmzr['logmstar']<11] # July 28 since we're not going so high anyway    
    
    ### compute x0 and bw for each
    x0_smhm = jnp.array(tum['HM(0)'])
    bw_smhm = 0.2 # jnp.diff(x0_smhm)[0] # all 0.2 dex for logmvir 
    
    x0_fgas = jnp.array(tfgas['logmstar'])
    bw_fgas = jnp.mean(jnp.diff(jnp.asarray(tfgas['logmstar']))) # take mean for constant bin width for simplicity
    
    x0_mzr = jnp.array(tmzr['logmstar'])
    bw_mzr = jnp.mean(jnp.diff(jnp.asarray(tmzr['logmstar']))) # again take mean for constant bw for simplicity -- here its always 0.2

    ### extract average relation and uncertainties 
    # this is why I switched to manga -- because hard to find standard errors on fgas and mzr (these are scatter)
    obs_smhm = jnp.asarray(tum['Med_Cen_SF(7)'])
    obs_smhm_low = jnp.asarray(tum['Err-(9)'])
    obs_smhm_high = jnp.asarray(tum['Err+(8)'])
    
    obs_mzr = jnp.asarray(tmzr['logZ_50'])
    obs_mzr_low = obs_mzr - jnp.asarray(tmzr['logZ_15'])
    obs_mzr_high = jnp.asarray(tmzr['logZ_84']) - obs_mzr
    
    ### for behroozi, take lower as symmetric since its larger than upper (and upper has weird insensible negatives)
    obs_smhm_err = obs_smhm_low

    ### Jan 5 2026 -- testing impact of smaller/larger low-mass SMHM errors (below manga mvir range)
    # obs_smhm_err = jnp.clip(obs_smhm_err,None,jnp.max(obs_smhm_err[x0_smhm>11]))
    # obs_smhm_err = jnp.hstack((jnp.full(5,0.01),obs_smhm_err[5:]))
    # obs_smhm_err = jnp.hstack((jnp.full(5,10.0),obs_smhm_err[5:]))

    ### Jan 15 2026 -- test impact of systematic shift up or down for SMHM
    # obs_smhm -= 0.3
    
    
    ### Calette+18 Appendix C -- constant w/ mstar for now
    obs_fgas = jnp.array(tfgas['fg50'])
    obs_fgas_err = 0.34 
    
    ### take max at each x0 for symmetric upper limit on error
    obs_mzr_err = jnp.maximum(obs_mzr_low,obs_mzr_high)

    # NOTE: the return order should be same as expected by sapphire.inference modules (for the "obs_stats" collection)
    return (x0_smhm,bw_smhm,obs_smhm,obs_smhm_err,
            x0_fgas,bw_fgas,obs_fgas,obs_fgas_err,
            x0_mzr,bw_mzr,obs_mzr,obs_mzr_err)


# Jan 29 -- behroozi for smhm, manga for fgas and mzr
def read_mix(config):

    obs_manga = read_manga(config)
    obs_lit = read_lit(config)
    obs_mix = (*obs_lit[:4], *obs_manga[4:])
    
    return obs_mix


# convenience wrapper
# TO DO: switch to importlib.load_module like in sapphire/models
def get(config):
    
    if config['obs_name'] == 'manga':
        return read_manga(config)

    elif config['obs_name'] == 'lit':
        return read_lit(config)

    elif config['obs_name'] == 'mix':
        return read_mix(config)

    else:
        sys.exit('obs_name must be one of manga, lit')
    
    

    

##
