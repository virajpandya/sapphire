# sapphire
sapphire is a JAX-based dynamical model for the phase space evolution of galaxy popoulations

<img width="1080" alt="sapphire_jax_github" src="https://user-images.githubusercontent.com/4482189/219704156-47d53e16-34b3-4937-863c-e05570c685ea.png">
logo/banner is a work in progress ... 

# Installation instructions [add JAX CPU/GPU install instructions]
Until we have a public release that can be pip/conda installed, please do the following to install the private development version: 
1. make sure you have a python=3.10 environment with the required dependencies given in environment.yml 
2. ```git clone git@github.com:virajpandya/sapphire.git```  (NOTE: this is a private repo so you'll probably need your git login)
3. in your .bashrc or .zshrc or .profile, add this line: ```export PYTHONPATH=$PYTHONPATH:"/PATH/TO/sapphire"```
4. download the latest sapphire/data tarball from GitHub Releases, and do something like "tar -xzf sapphire-data-v1.0.tar.gz -C sapphire/" 

#1 can easily be done by downloading miniconda, navigating to the cloned sapphire git directory and doing ```conda env create -f environment.yml```
This will create a new environment called sapphire which you can activate by doing ```conda activate sapphire```. 

#3 will enable you to do ```import sapphire``` in Python from anywhere on your system without needing to be in the cloned sapphire git directory. 
Once we allow pip install, #2 and #3 will no longer be necessary except for maybe the private development branch.

# Basic usage 
There are two main ways to run sapphire -- on the command line or via "import sapphire". See sapphire/scripts and sapphire/demo for examples. 

# Overview of code layout and philosophy 
sapphire is designed to be modular to maximize flexibility for assumed subgrid physics and for the numerics of how the ODEs are defined and solved.

### <<< add a basic flowchart/diagram showing the different code modules >>>

