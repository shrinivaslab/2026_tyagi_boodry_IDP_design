.. _api-ff-models:

====================================
``models.ff_models`` --- Force Fields
====================================

.. module:: models.ff_models
   :synopsis: Coarse-grained force field implementations.

The :mod:`models.ff_models` module implements the coarse-grained force
fields used to compute pairwise residue interaction potentials. Each
force field class provides methods for computing unbonded energies
(short-range + electrostatic), charges, and pair-charge arrays.

Module-Level Constants
======================

.. data:: RES_ALPHA

   The 23-character residue alphabet:
   ``"MGKTRADEYVLQWFSHNPCIXZB"``.

.. data:: NUM_RESIDUES

   Number of residue types (23).

.. data:: DEBYE_RC

   Default Debye-Hückel electrostatic cutoff distance (90.0 Å).

.. data:: RES_CODE

   Dictionary mapping single-letter residue codes to three-letter names.


File I/O Functions
==================

.. autofunction:: read_MPIPI_ff_params

.. autofunction:: read_HPS

.. autofunction:: read_calvados

.. autofunction:: read_KH_D


Utility Functions
=================

.. autofunction:: seq_to_one_hot


MPIPI Model
============

.. autoclass:: MPIPI_model
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Initialization

   .. automethod:: __init__

   .. rubric:: Charge Computation

   .. automethod:: calculate_charges
   .. automethod:: create_pair_charges

   .. rubric:: Potential Functions

   .. automethod:: wang_frenkel_internal_func
   .. automethod:: wang_frenkel_internal_func_FINCHES
   .. automethod:: coulomb_internal_func
   .. automethod:: coulomb_internal_func_FINCHES

   .. rubric:: Variant Swapping

   .. automethod:: swap_wf_func
   .. automethod:: swap_coul_func

   .. rubric:: Combined Potential

   .. automethod:: calculate_unbonded
   .. automethod:: construct_eps_matrix


HPS Model
==========

.. autoclass:: HPS_model
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Initialization

   .. automethod:: __init__

   .. rubric:: Environment Setup

   .. automethod:: get_relative_perm
   .. automethod:: get_debye_kappa

   .. rubric:: Charge Computation

   .. automethod:: calculate_charges
   .. automethod:: create_pair_charges

   .. rubric:: Potential Functions

   .. automethod:: ashbaugh_hatch_internal_func
   .. automethod:: coulomb_internal_func

   .. rubric:: Combined Potential

   .. automethod:: calculate_unbonded
   .. automethod:: construct_eps_matrix


CALVADOS Model
===============

.. autoclass:: Calvados_model
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Initialization

   .. automethod:: __init__

   .. rubric:: Charge Computation

   .. automethod:: calculate_charges
   .. automethod:: create_pair_charges

   .. rubric:: Potential Functions

   .. automethod:: hps_lj_internal_func
   .. automethod:: calvados_coul_internal_func
   .. automethod:: calvados_coul_internal_func_FINCHES

   .. rubric:: Variant Swapping

   .. automethod:: swap_coul_func

   .. rubric:: Combined Potential

   .. automethod:: calculate_unbonded
   .. automethod:: construct_eps_matrix


KH-D Model
============

.. autoclass:: KHD_model
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Initialization

   .. automethod:: __init__

   .. rubric:: Charge Computation

   .. automethod:: calculate_charges
   .. automethod:: create_pair_charges

   .. rubric:: Potential Functions

   .. automethod:: ashbaugh_hatch_internal_func
   .. automethod:: coulomb_internal_func

   .. rubric:: Combined Potential

   .. automethod:: calculate_unbonded
   .. automethod:: construct_eps_matrix
