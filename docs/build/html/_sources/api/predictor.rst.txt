.. _api-predictor:

===================================
``models.predictor`` --- Predictor
===================================

.. module:: models.predictor
   :synopsis: Interface between sequences and physics models.

The :mod:`models.predictor` module provides the :class:`Predictor` class,
which serves as the primary interface between amino acid sequences (or
probabilistic sequence arrays) and the underlying physics models. It handles
condition generation, force field initialization, and dispatches to the
appropriate model for computing interaction parameters.

Module-Level Constants
======================

.. data:: MODEL_REGISTRY

   Dictionary mapping model name strings (``'dimer'``, ``'rpa'``,
   ``'finches'``) to their corresponding model classes.

.. data:: FF_REGISTRY

   List of supported force field identifiers.

Module-Level Functions
======================

.. autofunction:: generate_all_conditions


Predictor Class
===============

.. autoclass:: Predictor
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Initialization

   .. automethod:: __init__

   .. rubric:: Prediction Methods

   .. automethod:: predict_pair
   .. automethod:: predict_batch
   .. automethod:: predict_pseqs
   .. automethod:: predict_interaction_matrix_pseqs

   .. rubric:: Utilities

   .. automethod:: generate_pseqs
