# sapphire
sapphire is a next-generation multi-zone model of galaxy formation [JAX version not yet ported] 

<img width="1080" alt="sapphire_jax_github" src="https://user-images.githubusercontent.com/4482189/219704156-47d53e16-34b3-4937-863c-e05570c685ea.png">


# Installation instructions 
Until we have a public release that can be pip/conda installed, please do the following to install the private development version: 
1. ```git clone git@github.com:virajpandya/sapphire.git```  (NOTE: this is a private repo so you'll probably need your git login)
2. in your .bashrc or .zshrc or .profile, add this line: export PYTHONPATH=$PYTHONPATH:"/PATH/TO/sapphire" 
3. use miniconda to create a new python=3.10 environment with the required dependencies given in environment.yml 

#2 will enable you to do ```import sapphire``` in Python from anywhere on your system without needing to be in the cloned sapphire git firectory. 
Once we allow pip install, #2 will no longer be necessary. Right now it is useful during development. 

#3 can easily be done with miniconda by navigating to the cloned sapphire git directory and doing "conda env create -f environment.yml".
This will create a new environment called sapphire which you can activate by doing ```conda activate sapphire```. 

# Basic usage 
You can run sapphire with effectively only 2 lines, along with lines that define a dictionary of model/runtime parameters: 

```python
import sapphire
sapphire.run(parameters)
# where parameters is a dictionary of model/runtime parameters -- see example Jupyter notebooks under demo/ subdirectory
```

Eventually you will also be able to just run sapphire on the command line with an argument specifying a JSON parameter input file
```
python -m sapphire parameters.json 
```

# Try running the demo Jupyter notebook 
Under the demo/ subdirectory there are Jupyter notebooks that show simple use cases for sapphire. 

As of Feb 17, we only have a single notebook for use on rusty where we run sapphire on TNG100 and FIRE-2 DM-only trees. 
Will add more notebooks for offline testing later (probably with sample SatGen EPS merger trees bundled with sapphire).


# Overview of code layout and philosophy 
sapphire is designed to be modular to maximize flexibility for assumed subgrid physics and for the numerics of how the ODEs are defined and solved.

### NOTE: add a basic flowchart / diagram showing the different code pieces 

# Action items still being worked on
- Modularize the integrator function by creating file(s) in subgrid_recipes that have functions that return individual RHS ODE terms
- Switch everything to the new logarithmic JAX backend from other example ipynb repo 
- Add a highly generalized set of parameter functions for power laws, etc. with slopes/normalizations as 
- Figure out and implement the optimal strategy for mpi4py over nodes (currently only multiprocessing) -- take care of GPUs for JAX
- Implement smarter, faster parallelized read/write with hdf5/msgpack output files
- Allow for a JSON parameter input file to be specified as a command line argument (e.g., python -m sapphire parameters.json) 
- Go back and re-enable automated testing (Continuous Integration) via Github Actions 
- Create a rusty Binder example and include link above for people that want to test sapphire without downloading
- Implement satellites and BHs (tricky bit will be adding in Nsat * Msat_properties ODEs and mergers for both galaxies and BHs) 
- Impement self-consistent multi-phase ISM and SF model 
- Allow option to output 2D SFHs for predicting stellar pops / galaxy observables
