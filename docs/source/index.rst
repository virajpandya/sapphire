.. sapphire-jax documentation master file, created by
   sphinx-quickstart on Wed Jul 22 12:33:45 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

sapphire
==========================

`sapphire <https://github.com/virajpandya/sapphire>`_ is a modular, automatically differentiable, multi-GPU-parallelized, open-source framework for evolving and understanding galaxy populations as dynamical systems. The code is written entirely in `JAX <https://github.com/jax-ml/jax>`_ with a custom-built adaptive differentiable RK23 solver for pedagogical purposes. For production-level science, sapphire leverages the growing JAX ecosystem including `diffrax <https://github.com/patrick-kidger/diffrax>`_, `optax <https://github.com/google-deepmind/optax>`_ and `numpyro <https://github.com/pyro-ppl/numpyro>`_. 

sapphire bridges astrophysics, cosmology, numerics, dynamics and statistics in new ways to enable:

* sensitivity analysis for galaxy astrophysics with Jacobians
* gradient descent with adam for efficient parameter optimization
* fast, exact Fisher uncertainty forecasts
* Bayesian inference with Hamiltonian Monte Carlo
* multi-GPU training set generation for implicit likelihood inference
* interpretable emulation of cosmological simulations
* hybrid physics-informed, data-driven galaxy formation modeling
* ... and more coming soon

Early applications for modeling galactic atmospheres and galaxy population evolution are described in `Pandya et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023ApJ...956..118P/abstract>`_ and `Pandya et al. (2026) <https://ui.adsabs.harvard.edu/abs/2026arXiv260406318P/abstract>`_.

.. warning::
   This documentation is in the process of being auto-generated.
   
   In the meantime, see the `GitHub code <https://github.com/virajpandya/sapphire/tree/main/demo/pandya26>`_ for reproducing the analysis and figures of Pandya et al. (2026) -- this currently serves as a demo/tutorial.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getstarted/installation 

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/pandya26 


