.. _api-prediction-model:

=============================================
``models.prediction_model`` --- Physics Models
=============================================

.. module:: models.prediction_model
   :synopsis: Physics models for IDP interaction parameters.

The :mod:`models.prediction_model` module implements the three physics
models that compute interaction parameters from amino acid sequences:
:class:`Dimer_model`, :class:`RPA_model`, and :class:`FINCHES_model`.
Each model wraps a force field (from :mod:`models.ff_models`) and provides
methods for both discrete sequence prediction and differentiable
(continuous/probabilistic) prediction suitable for gradient-based optimization.

Module-Level Constants
======================

.. data:: kb

   Boltzmann constant in kcal/mol/K (0.0019872041).

.. data:: RES_ALPHA

   The 23-character residue alphabet:
   ``"MGKTRADEYVLQWFSHNPCIXZB"``.

.. data:: NUM_RESIDUES

   Number of residue types (23).


Dimer Model
===========

.. autoclass:: Dimer_model
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Initialization

   .. automethod:: __init__

   .. rubric:: Interaction Tensor Computation

   .. automethod:: calculate_aa_interactions
   .. automethod:: set_aa_interactions

   .. rubric:: Sequence Transformations

   .. automethod:: pseq_to_dseq

   .. rubric:: Virial Coefficient Computation

   .. automethod:: dimer_virial_coefficients_internal
   .. automethod:: pairwise_interaction_matrix_internal

   .. rubric:: Prediction Methods (Discrete Sequences)

   .. automethod:: return_interaction_parameters
   .. automethod:: return_interaction_matrix

   .. rubric:: Prediction Methods (Differentiable / Continuous)

   .. automethod:: return_interaction_parameters_continuous
   .. automethod:: return_interaction_matrix_continuous


RPA Model
=========

.. autoclass:: RPA_model
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Initialization

   .. automethod:: __init__

   .. rubric:: Setup

   .. automethod:: set_aa_interactions

   .. rubric:: Chi Components

   .. automethod:: calc_chi_eff_MFT
   .. automethod:: calc_chi_eff_RPA
   .. automethod:: calc_chi_h
   .. automethod:: calc_chi_eff

   .. rubric:: Charge Utilities

   .. automethod:: get_pseq_charges
   .. automethod:: get_pseq_charge_locs

   .. rubric:: Prediction Methods

   .. automethod:: return_interaction_parameters
   .. automethod:: return_interaction_parameters_continuous


FINCHES Model
=============

.. autoclass:: FINCHES_model
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Initialization

   .. automethod:: __init__

   .. rubric:: Setup

   .. automethod:: set_aa_interactions
   .. automethod:: get_null_interaction_baseline

   .. rubric:: Aliphatic Weighting

   .. automethod:: get_aliphatic_residues
   .. automethod:: build_gap_merged_fragments
   .. automethod:: get_aliphatic_weighting
   .. automethod:: get_multiplier_weight_vectorized
   .. automethod:: get_aliphatic_interaction_weighting

   .. rubric:: Charge Weighting

   .. automethod:: get_neighbors_window_of3
   .. automethod:: calculacte_FCR_NCPR
   .. automethod:: get_charge_interaction_weighting

   .. rubric:: Prediction Methods

   .. automethod:: return_interaction_parameters
   .. automethod:: return_interaction_parameters_pseqs
