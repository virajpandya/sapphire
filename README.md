# sapphire
sapphire is a JAX-based dynamical model for the phase space evolution of galaxy popoulations

<img width="1080" alt="sapphire_jax_github" src="https://user-images.githubusercontent.com/4482189/219704156-47d53e16-34b3-4937-863c-e05570c685ea.png">
logo/banner is a work in progress ... 

# Installation instructions 
### <<< add instructions to install JAX w/ GPU support on cluster >>>
### <<< add ```mamba env create -f environment.yml``` option >>>

1. download and install mamba/conda-forge: https://conda-forge.org/download/
2. CONDA_SUBDIR=osx-arm64 mamba create -n YOURENVNAME python=3.12 (skip the subdir part if not on macbook)
3. mamba activate YOURENVNAME
4. pip install jax jupyter matplotlib numpyro optax equinox flax pandas astropy chainconsumer pandas seaborn multiprocess arviz jax-cosmo diffrax
5. python -m ipykernel install --user --name YOURENVNAME
6. clone the sapphire repo and keep note of the path to the directory
7. download the latest/corresponding sapphire/data tarball from GitHub Releases, and do something like "tar -xzf sapphire-data-v0.1.tar.gz -C /Users/viraj/sapphire/" 
8. until we add a pip install option, in your .bashrc or .zshrc, add this line: ```export PYTHONPATH=$PYTHONPATH:"/PATH/TO/sapphire"```
9. jupyter lab & (this will open jupyterlab in a browser window)
10. try running/adapting one of the notebooks in demos/ or tests/

# Overview of code layout and philosophy 
sapphire is designed to be modular to maximize flexibility for assumed subgrid physics and for the numerics of how the ODEs are defined and solved.

### <<< add a basic flowchart/diagram showing the different code modules >>>

# Basic usage 
There are two main ways to run sapphire -- on the command line or via "import sapphire". See sapphire/scripts and sapphire/demo for examples. 

