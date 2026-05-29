.. _api-metapredict-jax:

================================================
``models.metapredict_jax`` --- Disorder Predictor
================================================

.. module:: models.metapredict_jax
   :synopsis: JAX implementation of MetapredictV3 for intrinsic disorder prediction.

The :mod:`models.metapredict_jax` module provides a pure-JAX
reimplementation of the MetapredictV3 disorder predictor. This enables
the disorder constraint to be fully differentiable and compatible with
JAX's JIT compilation and ``jax.lax.scan`` during sequence optimization.

The model uses a bidirectional LSTM architecture that predicts per-residue
disorder scores from a one-hot encoded amino acid sequence. The
implementation uses ``jax.lax.scan`` for the LSTM recurrence to enable
efficient GPU compilation.

Key Components
==============

.. autofunction:: pseq_to_mp_pseq

.. autofunction:: forward_pass

.. autofunction:: lstm_cell_forward

.. autofunction:: process_sequence_direction


MetapredictJAX Class
====================

.. autoclass:: MetapredictJAX
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Loading

   .. automethod:: load_params

   .. rubric:: Prediction

   .. automethod:: return_metapredict_fn
