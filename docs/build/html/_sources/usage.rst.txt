.. _usage-section:

=====
Usage
=====

.. _installation-section:

Installation
============

Prerequisites
-------------

All code was developed and tested with **Python 3.11**. We recommend
creating a fresh conda or mamba environment:

.. code-block:: bash

   conda create -n idp_design python=3.11
   conda activate idp_design

Clone the repository
--------------------

.. code-block:: bash

   git clone <repository-url>
   cd theory_idp_design

Install dependencies (CPU)
--------------------------

.. code-block:: bash

   pip install -r cpu_requirements.txt

Install dependencies (GPU)
--------------------------

For GPU-accelerated optimization with CUDA support:

.. code-block:: bash

   pip install -r gpu_requirements.txt

.. note::

   The GPU requirements install ``jax[cuda12]``. Make sure your system has
   a compatible CUDA toolkit and cuDNN installed.


Core Concepts
=============

Sequence Representation
-----------------------

Sequences are represented as **probabilistic sequences** (``pseq``):
continuous relaxations of one-hot encoded amino acid sequences. Each
position holds a probability distribution over the 23-character residue
alphabet:

.. code-block:: text

   RES_ALPHA = "MGKTRADEYVLQWFSHNPCIXZB"

The last three characters (``X``, ``Z``, ``B``) represent modified
residues: methylated arginine, phosphorylated serine, and phosphorylated
threonine, respectively.

During optimization, raw logits are passed through a temperature-scaled
softmax (Gumbel-softmax annealing) to produce ``pseq`` values. The
temperature is annealed from 1.0 to 0.001 over the course of
optimization, gradually sharpening the distribution toward a discrete
sequence.

Physics Models
--------------

Three physics models are available, each computing interaction parameters
from probabilistic sequences:

**Dimer model** (``model_name='dimer'``)
   Computes second virial coefficients (:math:`B_2`) by contracting
   dimer frequency vectors with a precomputed 529×529 interaction
   tensor derived from numerical integration of the Mayer f-function.
   The result is normalized to a :math:`\chi` parameter.

**RPA model** (``model_name='rpa'``)
   Computes an effective Flory :math:`\chi` as the sum of mean-field
   (Debye-Hückel), fluctuation (Random Phase Approximation), and
   short-range hydrophobic contributions. Accounts for sequence-level
   charge patterning.

**FINCHES model** (``model_name='finches'``)
   Computes pairwise residue interaction energies with optional
   charge-context and aliphatic-cluster weighting corrections, then
   decomposes into attractive/repulsive contributions relative to a
   null baseline.

Force Fields
------------

Each physics model uses a coarse-grained force field that defines the
pairwise interaction potentials:

- **MPIPI** (``ff_type='mpipi'`` or ``'mpipi_gg'``): Wang-Frenkel + Debye-Hückel
- **HPS** (``ff_type='hps_dignon'``, ``'hps_tesei'``, ``'hps_urry'``, ``'hps_fb'``): Ashbaugh-Hatch + Debye-Hückel
- **CALVADOS** (``ff_type='calvados1'`` or ``'calvados2'``): HPS-LJ + Yukawa
- **KH-D** (``ff_type='kh_d'``): Ashbaugh-Hatch + Debye-Hückel (Kim-Hummer)

Intrinsic Disorder Constraint
------------------------------

An optional **MetapredictV3** constraint ensures designed sequences
remain intrinsically disordered. When enabled, a differentiable loss
multiplier penalizes sequences whose predicted disorder score falls
below an annealed threshold (ramped from 0.2 to 0.8 over 80% of
iterations).


Design Types
============

The :class:`~models.designer.Designer` class supports several design
objectives, each accessed through :meth:`~models.designer.Designer.design_sequence`:

Value Optimization (``design_type='value'``)
--------------------------------------------

Optimize a single sequence to match a target :math:`\chi` value under
one thermodynamic condition.

.. code-block:: python

   from models.designer import Designer

   designer = Designer(model_name='dimer', ff_type='calvados2')
   seq, aux = designer.design_sequence(
       sequence_length=50,
       design_type='value',
       num_iterations=500,
       temps=[300], salts=[150], pHs=[7.4],
       phosphorylation=[0], methylation=[0],
       target_value=0.25,
   )

1D Switch Optimization (``design_type='1d_switch'``)
----------------------------------------------------

Design a sequence that transitions between condensed and expanded states
across two conditions (e.g., different temperatures or salt concentrations).

.. code-block:: python

   seq, aux = designer.design_sequence(
       sequence_length=50,
       design_type='1d_switch',
       num_iterations=500,
       temps=[300], salts=[150, 450], pHs=[7.4],
       phosphorylation=[0], methylation=[0],
       response_type='contractor',
   )

Designed Response (``design_type='designed_response'``)
-------------------------------------------------------

Optimize a sequence whose :math:`\chi` follows a prescribed profile
(linear, early step, late step, or bandpass) as a condition variable is
swept between two endpoints.

.. code-block:: python

   seq, aux = designer.design_sequence(
       sequence_length=50,
       design_type='designed_response',
       num_iterations=500,
       temps=[300], salts=[150], pHs=[7.4],
       phosphorylation=[0, 1], methylation=[0],
       response_type='contractor',
       response_profile='late_step',
       phos_locs=[10, 20, 25, 30, 35, 40],
   )

Multi-Dimensional Switch (``design_type='multi_dimensional_switch'``)
---------------------------------------------------------------------

Generalization of 1D switching to N conditions, each assigned a binary
target state (condensed or expanded).

.. code-block:: python

   seq, aux = designer.design_sequence(
       sequence_length=50,
       design_type='multi_dimensional_switch',
       num_iterations=500,
       temps=[300], salts=[150, 450], pHs=[7.4],
       phosphorylation=[0, 1], methylation=[0],
       targets=[1, 0, 0, 0],
       phos_locs=[10, 20, 25, 30, 35, 40],
   )

Reference Optimization (``design_type='reference'``)
----------------------------------------------------

Design a sequence that maximizes attraction (client) or repulsion with a
fixed reference sequence.

.. code-block:: python

   seq, aux = designer.design_sequence(
       sequence_length=50,
       design_type='reference',
       num_iterations=500,
       temps=[300], salts=[150], pHs=[7.4],
       phosphorylation=[0], methylation=[0],
       reference_sequence='GSGSGSGSGS',
       opt_type='client',
   )

Two-Sequence Optimization (``design_type='two_sequence'``)
----------------------------------------------------------

Simultaneously co-optimize two sequences for joint phase behavior
(co-condensation or demixing).

.. code-block:: python

   seq, aux = designer.design_sequence(
       sequence_length=50,
       design_type='two_sequence',
       num_iterations=500,
       temps=[300], salts=[150], pHs=[7.4],
       phosphorylation=[0], methylation=[0],
       behavior='condensed_mix',
   )

Interaction Matrix Optimization (``design_type='interaction_matrix'``)
----------------------------------------------------------------------

Optimize a sequence to produce a target spatial pattern of residue-residue
interactions (e.g., attractive ends with repulsive middle).

.. code-block:: python

   seq, aux = designer.design_sequence(
       sequence_length=50,
       design_type='interaction_matrix',
       num_iterations=500,
       temps=[300], salts=[150], pHs=[7.4],
       phosphorylation=[0], methylation=[0],
   ) 
Batch Optimization
==================

For running many independent value optimizations in parallel (particularly
useful on GPU), use :meth:`~models.designer.Designer.design_batch`:

.. code-block:: python

   seqs, results = designer.design_batch(
       sequence_length=50,
       num_sequences=32,
       num_iterations=500,
       temps=[300], salts=[150], pHs=[7.4],
       phosphorylation=[0], methylation=[0],
       target_values=0.25,
   )

This uses ``jax.vmap`` to parallelize across sequences and
``jax.lax.scan`` to compile the iteration loop, yielding significant
speedups on GPU hardware.


Prediction Only
===============

To compute interaction parameters for existing sequences without
optimization, use the :class:`~models.predictor.Predictor` class
directly:

.. code-block:: python

   from models.predictor import Predictor

   predictor = Predictor(
       model_name='dimer',
       ff_type='calvados2',
       temps=[300], salts=[150], pHs=[7.4],
   )

   # Single pair
   result = predictor.predict_pair("GSGSGSGSGS", "GSGSGSGSGS")

   # Batch
   results = predictor.predict_batch(
       seqs1=["GSKRGSGSGS", "DDDDDKKKKK"],
       seqs2=["GSKRGSGSGS", "DDDDDKKKKK"],
   )
