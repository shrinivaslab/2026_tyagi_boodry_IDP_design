.. theory_idp_design documentation master file

================================
theory_idp_design documentation
================================

**theory_idp_design** is a Python library for designing intrinsically
disordered protein (IDP) sequences via differentiable optimization over
continuous sequence representations. It combines coarse-grained physics
models of IDP interactions with JAX-based gradient descent to produce
sequences that satisfy target thermodynamic properties.

The library supports multiple force fields (MPIPI, HPS, CALVADOS, KH-D)
and physics models (dimer virial, RPA, FINCHES), and can optimize for a
variety of design objectives including target interaction parameters,
stimulus-responsive switching, multi-condition behavior, and inter-chain
interactions.

.. _usage-section:

Getting Started
---------------

.. toctree::
   :maxdepth: 2

   usage

.. _api-section:

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api/designer
   api/predictor
   api/prediction_model
   api/ff_models
   api/metapredict_jax


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
