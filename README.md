# sapphire
sapphire is a JAX-based framework for modeling the dynamical phase space evolution of galaxy populations. 

<img width="3200" height="1800" alt="pandya26_sapphire" src="https://github.com/user-attachments/assets/38b18df2-c097-4518-add2-f3493a2d948b" />
Figure Credit: Viraj Pandya and Pandya et al. (2026).

## Installation instructions 

### pip install [CPU-only]

Until we put sapphire on PyPI, download/clone the sapphire repo and do

```
mamba activate yourenvname [highly recommended, see below]
cd /path/to/sapphire/download
pip install . 
```
This will put sapphire in your python path, so you can do ```import sapphire``` from anywhere. 

Then, download the latest corresponding data tarball from GitHub Releases and do something like

```
tar -xvzf /path/to/downloaded/sapphire/data/tarball/ -C /path/to/sapphire
```
which will extract the required data files into the ```sapphire/data``` subdirectory (ideally the pip installation path). Alternatively, you can extract this tarball anywhere you want and then feed it as the runtime ```data_path``` argument (e.g., sapphire/scripts/config.yaml -- see below).

We strongly recommend installing mamba from https://conda-forge.org/download/ to install a virtual environment with the necessary package dependencies listed in ```environment.yml```. You can automate the creation of a new mamba environment called "sapphire" by doing

``` 
CONDA_SUBDIR=osx-arm64 # this line is only if installing on MacOS
mamba create -f environment.yml 
```

If you want to use sapphire in Jupyter notebooks, you may then also need to do something like

```
mamba activate yourenvname
pip install jupyter jupyterlab
python -m ipykernel install --user --name yourenvname
```

### Installation with GPU support
<<< instructions forthcoming >>>

## Quick example
Go to https://binder.flatironinstitute.org/, enter "vpandya" for host user and "sapphire" for project name. This will automatically create an environment with the required python packages, so you only need to clone/upload and follow the pip installation instructions for sapphire and its data tarball.

Then try running ```sapphire/demo/prior_predictive_checks.ipynb```. You will need to modify the paths in that ipynb and in ```sapphire/scripts/config.yaml``` to start with ```/home/jovyan/```

## Documentation
Link to GitHub Wiki or MkDocs/ReadTheDocs forthcoming



