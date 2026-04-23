# sapphire
sapphire is a modular, automatically differentiable, multi-GPU-parallelized, open-source framework for evolving and understanding galaxy populations as dynamical systems. The code is written from scratch entirely in JAX (Bradbury et al. 2018) using the diffrax differential equation solver package (Kidger 2022). sapphire bridges astrophysics, cosmology, numerics, dynamics and statistics in new ways to enable:
- sensitivity analysis for galaxy astrophysics with Jacobians
- gradient descent with adam for efficient parameter optimization
- fast, exact Fisher uncertainty forecasts
- Bayesian inference with Hamiltonian Monte Carlo 
- multi-GPU training set generation for implicit likelihood inference
- interpretable emulation of cosmological simulations
- hybrid physics-informed, data-driven galaxy formation modeling
- ... and more coming soon

Figure 1 from [Pandya et al. (2026)](https://ui.adsabs.harvard.edu/abs/2026arXiv260406318P/abstract) provides a schematic overview of the interdisciplinary dynamical systems approach to galaxy evolution taken by sapphire. 

<img width="3200" height="1800" alt="pandya26_sapphire" src="https://github.com/user-attachments/assets/38b18df2-c097-4518-add2-f3493a2d948b" />

<br>

Figure 1 from [Pandya et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJ...956..118P/abstract) illustrates our new baseline physical model, which comprises eight nonlinearly coupled ordinary differential equations that describe how mass, metals and energy flow between galaxies and their gaseous atmospheres.

<img width="917" height="664" alt="image" src="https://github.com/user-attachments/assets/e8a57ffa-c78c-40b6-91db-e6af51efff05" />

## Installation instructions 

### pip install [CPU-only]

sapphire is on PyPI so you could do
```
mamba activate yourenvname [highly recommended, see below]
pip install --upgrade sapphire-jax
```

However, since sapphire is under active development, we instead recommend cloning the latest git repo:
```
mamba activate yourenvname [highly recommended, see below]
git clone https://github.com/virajpandya/sapphire.git
cd /path/to/sapphire/download
pip install -e .
```
This will put the sapphire directory in your python path, so you can do ```import sapphire``` from anywhere. 

Then, download the latest corresponding data tarball from GitHub Releases and do something like

```
tar -xvzf /path/to/downloaded/sapphire/data/tarball/ -C /path/to/sapphire
```
which will extract the required data files into the ```sapphire/data``` subdirectory (ideally the pip installation path). Alternatively, you can extract this tarball anywhere you want and then feed it as the ```data_path``` argument in the config.yaml file and/or as a command-line argument (see below).

We strongly recommend installing mamba from https://conda-forge.org/download/ to create a virtual environment with the necessary package dependencies listed in ```environment.yml```. You can automate the creation of a new mamba environment called "sapphire" by doing

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
This is pretty much similar to above except the mamba environment you create must install jax and jaxlib with GPU-support. Currently sapphire is only tested on Nvidia GPUs -- best to follow the instructions on the Nvidia website for installing the relevant GPU drivers beforehand. Afterwards, in principle it's as simple as: 

```
mamba create -n yourenvname python=3.12 
mamba activate yourenvname
pip install --upgrade "jax[cuda12]" # or [cuda13] depending on what Nvidia drivers you have installed
# then pip install the other sapphire dependencies in environment.yml
```

We recommend installing a single environment with GPU support since that automatically comes with CPU support (if/when no GPUs are auto-detected by JAX).

## Quick example
Note: We are working on adding Colab and non-Flatiron Binder options. For Flatiron Binder access, send Viraj Pandya (vgp2108@columbia.edu) your email address.

Go to https://binder.flatironinstitute.org/, enter "vpandya" for host user and "sapphire" for project name. This will automatically create an environment with the required python packages, so you only need to clone/upload and follow the pip installation instructions for sapphire and its data tarball. Note that there is a limit of 16 CPU cores and 256GB memory which may not be enough for many sapphire applications. 

Then try running ```sapphire/demo/pandya26/prior_predictive_checks.ipynb``` or any of the other notebooks to reproduce figures from [Pandya+26](https://ui.adsabs.harvard.edu/abs/2026arXiv260406318P/abstract). You will need to modify the paths in that ipynb and in ```sapphire/scripts/config.yaml``` to start with ```/home/jovyan/```. 

## Documentation
Link to GitHub Wiki or MkDocs/ReadTheDocs forthcoming


## Citation and Acknowledgement Policy
sapphire is made publicly available under an MIT License with original copyright held by Viraj Pandya. The LICENSE file in the root repository must be preserved if you use, adapt or fork the code in this repository. You are welcome to add your own license to that baseline MIT License.

If you use sapphire in your work, you must include the following line in the acknowledgements: 

> We used the \texttt{sapphire} code \citep{pandya23,pandya26} whose origins trace back to the PhD thesis of \citet{pandya21thesis}, which grew out of the Simulating Multi-scale Astrophysics to Understand Galaxies (SMAUG) project and was later integrated into the Simons Collaboration on Learning the Universe, with funding provided by NASA Hubble Fellowship HST-HF2-51489, NSF Astronomy and Astrophysics Grant 2307419 and NSF Graduate Research Fellowship 1339067.

If you did not use the code or outputs as-is but found the approach useful, we ask that you still cite these works:

```
<<<<<<<<<<
- expand this to include subsequent papers by other sapphire developers
>>>>>>>>>>

@ARTICLE{pandy26,
       author = {{Pandya}, Viraj and {Bryan}, Greg L. and {Makinen}, T. Lucas and {Gabrielpillai}, Austen and {Carr}, Christopher and {Fielding}, Drummond B. and {Hernquist}, Lars and {Ho}, Matthew and {Iyer}, Kartheik and {Jespersen}, Christian Kragh and {Koudmani}, Sophie and {Laska}, Marta and {Lemos}, Pablo and {Lovell}, Christopher C. and {Perez}, Lucia A. and {Robinson}, Jr., William F. and {Somerville}, Rachel S. and {Starkenburg}, Tjitske K. and {Stiskalek}, Richard and {Terrazas}, Bryan and {Voit}, G. Mark},
        title = "{Introducing sapphire: Towards Hybrid Physics-Informed, Data-Driven Modeling of Galaxy Formation}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies},
         year = 2026,
        month = apr,
          eid = {arXiv:2604.06318},
        pages = {arXiv:2604.06318},
          doi = {10.48550/arXiv.2604.06318},
archivePrefix = {arXiv},
       eprint = {2604.06318},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260406318P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{pandya23,
       author = {{Pandya}, Viraj and {Fielding}, Drummond B. and {Bryan}, Greg L. and {Carr}, Christopher and {Somerville}, Rachel S. and {Stern}, Jonathan and {Faucher-Gigu{\`e}re}, Claude-Andr{\'e} and {Hafen}, Zachary and {Angl{\'e}s-Alc{\'a}zar}, Daniel and {Forbes}, John C.},
        title = "{A Unified Model for the Coevolution of Galaxies and Their Circumgalactic Medium: The Relative Roles of Turbulence and Atomic Cooling Physics}",
      journal = {\apj},
     keywords = {Circumgalactic medium, Galactic winds, Galaxy evolution, Galaxy accretion, Cooling flows, Hydrodynamical simulations, Analytical mathematics, 1879, 572, 594, 575, 2028, 767, 38, Astrophysics - Astrophysics of Galaxies},
         year = 2023,
        month = oct,
       volume = {956},
       number = {2},
          eid = {118},
        pages = {118},
          doi = {10.3847/1538-4357/acf3ea},
archivePrefix = {arXiv},
       eprint = {2211.09755},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023ApJ...956..118P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@PHDTHESIS{pandya21thesis,
       author = {{Pandya}, Viraj.},
        title = "{Semi-Analytic Modeling of Galaxy Formation for the Future: The Challenge of Emulating Cosmological Hydrodynamical Simulations}",
       school = {University of California, Santa Cruz},
         year = 2021,
        month = aug
}

```
