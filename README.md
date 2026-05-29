# theory-idp-design

This repository contains code corresponding to the paper titled "Physics-guided de novo design of intrinsically disordered proteins". We provide worked examples in `example_usage.ipynb`for the following example optimizations:

- Interaction parameter optimization
- Salt sensor optimization
- Early- and late-step phosphorylation designed response
- Salt-phosphorylation multidimensional sensor (OR-like response)
- Client/excluder optimization
- Multiple-sequence optimization
- Batched interaction parameter optimization

Within the same `example_usage.ipynb` file we additionally provide usage examples for interaction parameter calculation from IDP sequence. ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shrinivaslab/2026_tyagi_boodry_idp_design/blob/master/example_usage.ipynb))

Additional code is available upon request.

---

# Installation

## Create a new environment

We recommend starting with a fresh environment (via `mamba` or `conda`). All code was tested using Python version `3.11`:

```bash
conda create -n idp_design python=3.11
conda activate idp_design
```

## Clone the repository

```bash
git clone <repository-url>
cd 2026_tyagi_boodry_idp_design
```

## Install dependencies (CPU)

```bash
pip install -r cpu_requirements.txt
```

## Install dependencies (GPU)

For GPU-accelerated optimization with CUDA support:

```bash
pip install -r gpu_requirements.txt
```

> **Note:** The GPU requirements install `jax[cuda12]`. Make sure your system has a compatible CUDA toolkit and cuDNN installed.

---

# Documentation

Full API documentation is written in Sphinx RST format and lives in `docs/`.

## Build the docs

```bash
cd docs
make html
```

The built documentation will be output to `docs/build/html/`. Open `docs/build/html/index.html` in a browser to view it.

> **Note:** Sphinx must be installed (`pip install sphinx`) before building.

---

# Core Concepts

## Sequence Representation

Sequences are represented as **probabilistic sequences** (`pseq`): continuous relaxations of one-hot encoded amino acid sequences. Each position holds a probability distribution over the 23-character residue alphabet:

```
RES_ALPHA = "MGKTRADEYVLQWFSHNPCIXZB"
```

The last three characters (`X`, `Z`, `B`) represent modified residues: methylated arginine, phosphorylated serine, and phosphorylated threonine, respectively.

During optimization, raw logits are passed through a temperature-scaled softmax (Gumbel-softmax annealing) to produce `pseq` values. The temperature is annealed from 1.0 to 0.001 over the course of optimization, gradually sharpening the distribution toward a discrete sequence.

## Physics Models

Three physics models are available, each computing interaction parameters from probabilistic sequences:

- **Dimer model** (`model_name='dimer'`): Computes second virial coefficients (B₂) by contracting dimer frequency vectors with a precomputed 529×529 interaction tensor derived from numerical integration of the Mayer f-function. The result is normalized to a χ parameter.
- **RPA model** (`model_name='rpa'`): Computes an effective Flory χ as the sum of mean-field (Debye-Hückel), fluctuation (Random Phase Approximation), and short-range hydrophobic contributions. Accounts for sequence-level charge patterning.
- **FINCHES model** (`model_name='finches'`): Computes pairwise residue interaction energies with optional charge-context and aliphatic-cluster weighting corrections, then decomposes into attractive/repulsive contributions relative to a null baseline. Returned interaction parameter value is denoted as ε.

## Force Fields

Each physics model uses a coarse-grained force field that defines pairwise interaction potentials:


| `ff_type`                                       | Potential      | Electrostatics            |
| ----------------------------------------------- | -------------- | ------------------------- |
| `mpipi`, `mpipi_gg`                             | Wang-Frenkel   | Debye-Hückel              |
| `hps_dignon`, `hps_tesei`, `hps_urry`, `hps_fb` | Ashbaugh-Hatch | Debye-Hückel              |
| `calvados1`, `calvados2`                        | HPS-LJ         | Yukawa                    |
| `kh_d`                                          | Ashbaugh-Hatch | Debye-Hückel (Kim-Hummer) |


## Intrinsic Disorder Constraint

An optional **MetapredictV3** constraint ensures designed sequences remain intrinsically disordered. When enabled, a differentiable loss multiplier penalizes sequences whose predicted disorder score falls below an annealed threshold (ramped from 0.2 to 0.8 over 80% of iterations).

---

# Usage

## Value Optimization

Optimize a single sequence to match a target χ value under one thermodynamic condition:

```python
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
```

## 1D Switch Optimization

Design a sequence that transitions between condensed and expanded states across two conditions (e.g., two salt concentrations):

```python
seq, aux = designer.design_sequence(
    sequence_length=50,
    design_type='1d_switch',
    num_iterations=500,
    temps=[300], salts=[150, 450], pHs=[7.4],
    phosphorylation=[0], methylation=[0],
    response_type='contractor',
)
```

## Designed Response

Optimize a sequence whose χ follows a prescribed profile (linear, early step, late step, or bandpass) as a condition variable is swept between two endpoints:

```python
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
```

## Multi-Dimensional Switch

Generalization of 1D switching to N conditions, each assigned a binary target state (condensed or expanded):

```python
seq, aux = designer.design_sequence(
    sequence_length=50,
    design_type='multi_dimensional_switch',
    num_iterations=500,
    temps=[300], salts=[150, 450], pHs=[7.4],
    phosphorylation=[0, 1], methylation=[0],
    targets=[1, 0, 0, 0],
    phos_locs=[10, 20, 25, 30, 35, 40],
)
```

## Reference Optimization

Design a sequence that maximizes attraction (`client`) or repulsion with a fixed reference sequence:

```python
seq, aux = designer.design_sequence(
    sequence_length=50,
    design_type='reference',
    num_iterations=500,
    temps=[300], salts=[150], pHs=[7.4],
    phosphorylation=[0], methylation=[0],
    reference_sequence='GSGSGSGSGS',
    opt_type='client',
)
```

## Two-Sequence Optimization

Simultaneously co-optimize two sequences for joint phase behavior (co-condensation or demixing):

```python
seq1, seq2, aux = designer.design_sequence(
    sequence_length=50,
    design_type='two_sequence',
    num_iterations=500,
    temps=[300], salts=[150], pHs=[7.4],
    phosphorylation=[0], methylation=[0],
    behavior='condensed_mix',
)
```

## Interaction Matrix Optimization

Optimize a sequence to produce a target spatial pattern of residue-residue interactions (e.g., attractive termini with repulsive middle) (very rudimentary implementation as of right now):

```python
seq, aux = designer.design_sequence(
    sequence_length=50,
    design_type='interaction_matrix',
    num_iterations=500,
    temps=[300], salts=[150], pHs=[7.4],
    phosphorylation=[0], methylation=[0],
)
```

## Batch Optimization

Run many independent value optimizations in parallel (particularly useful on GPU). Uses `jax.vmap` to parallelize across sequences and `jax.lax.scan` to compile the iteration loop. target_values can be either a single value to optimize all sequences toward or a list/array of length (num_sequences) where seqs[i] is optimized toward target_values[i]:

```python
seqs, results = designer.design_batch(
    sequence_length=50,
    num_sequences=32,
    num_iterations=500,
    temps=[300], salts=[150], pHs=[7.4],
    phosphorylation=[0], methylation=[0],
    target_values=0.25,
)
```

## Prediction Only

To compute interaction parameters for existing sequences without optimization, use the `Predictor` class directly:

```python
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
```

---