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


## Citation and Acknowledgement Policy
sapphire is made publicly available under an MIT License with original copyright held by Viraj Pandya. The LICENSE file in the root repository must be preserved if you use, adapt or fork the code in this repository. You are welcome to add your own license to that baseline MIT License.

If you use sapphire in your work, you must include the following line in the acknowledgements: 

> We used the \texttt{sapphire} code \citep{pandya23,pandya26} whose origins trace back to the PhD thesis of \citet{pandya21thesis}, which grew out of the Simulating Multi-scale Astrophysics to Understand Galaxies (SMAUG) project and was later integrated into the Simons Collaboration on Learning the Universe, with funding provided by NASA Hubble Fellowship HST-HF2-51489, NSF Astronomy and Astrophysics Grant 2307419 and NSF Graduate Research Fellowship 1339067.

If you did not use the code or outputs as-is but found the approach useful, we ask that you still cite these works:

```
<<<<<<<<<<
- add Pandya+26 when submitted
- expand this to include subsequent papers by other sapphire developers
>>>>>>>>>>

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
