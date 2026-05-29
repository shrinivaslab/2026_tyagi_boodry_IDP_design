.. _api-designer:

=================================
``models.designer`` --- Designer
=================================

.. module:: models.designer
   :synopsis: Sequence design via differentiable optimization.

The :mod:`models.designer` module contains the main :class:`Designer` class
that orchestrates IDP sequence optimization. It defines the loss functions
for each design objective, manages the Gumbel-softmax annealing schedule,
and runs the Adam optimization loop (with optional ``jax.lax.scan``
compilation on GPU).

Module-Level Functions
======================

.. autofunction:: set_pseq_residues

.. autofunction:: generate_all_conditions


Designer Class
==============

.. autoclass:: Designer
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Initialization

   .. automethod:: __init__

   .. rubric:: Value Conversions

   .. automethod:: convert_to_normalized_chi
   .. automethod:: convert_to_chi
   .. automethod:: normalize_logits

   .. rubric:: Metapredict Constraint

   .. automethod:: get_metapredict_function

   .. rubric:: Top-Level Dispatch

   .. automethod:: design_sequence
   .. automethod:: design_batch

   .. rubric:: Design-Type-Specific Optimizers

   .. automethod:: run_value_optimization
   .. automethod:: run_1d_switch_optimization
   .. automethod:: run_designed_response_optimization
   .. automethod:: run_multi_dimensional_switch_optimization
   .. automethod:: run_reference_optimization
   .. automethod:: run_interaction_matrix_optimization
   .. automethod:: run_two_sequence_optimization

   .. rubric:: Optimization Engine

   .. automethod:: optimization_loop
