from typing import Sequence, Tuple, Optional, List
import jax
import jax.numpy as jnp
import itertools
import functools
from tqdm import tqdm
from pathlib import Path


import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, lax
import functools
import os
import importlib
jax.config.update("jax_enable_x64", False)

from jax.lib import xla_bridge
# print(xla_bridge.get_backend().platform)
import numpy as onp
import optax
from models.predictor import Predictor, FF_REGISTRY, MODEL_REGISTRY
from models.ff_models import RES_ALPHA, seq_to_one_hot

# from models.metapredict_utils import pseq_to_mp_pseq, get_metapredict_fn
# import models.metapredict_utils as metapredict
from models.metapredict_jax import pseq_to_mp_pseq, MetapredictJAX

ROOT = Path(__file__).resolve().parents[1]


bond_factors = {
    'mpipi': 3.6,
    'hps_tesei': 3.0,
    'hps_dignon': 3.2
}

# for mutation-type conditions (e.g. phosphorylation, methylation)
def set_pseq_residues(pseq: jnp.ndarray, locs: List[int], residue: str) -> jnp.ndarray:
    """Fix specific positions of a probabilistic sequence to a given residue.

    Replaces the distribution at each position in locs with a hard one-hot
    vector for the specified residue. Used to encode post-translational
    modifications (phosphorylation, methylation).

    Args:
        pseq: Probabilistic sequence array of shape (L, len(RES_ALPHA)).
        locs: Sequence positions (0-indexed) to override.
        residue: Single-character residue code from RES_ALPHA.

    Returns:
        Modified pseq with hard one-hot vectors at the specified positions.
    """
    base_vector = jnp.zeros(len(RES_ALPHA)).at[RES_ALPHA.index(residue)].set(1.0)
    for loc in locs:
        pseq = pseq.at[loc].set(base_vector)
    
    return pseq

def generate_all_conditions(phosphorylation: List[int], methylation: List[int], temps: List[float | int], salts: List[float | int], pHs: List[float | int]) -> List[Tuple[int, int, float, float, float]]:
    """Return the Cartesian product of all condition variables.

    Each returned tuple is (phosphorylation, methylation, temp, salt, pH).

    Args:
        phosphorylation: Binary flags (0 or 1) for phosphorylation state.
        methylation: Binary flags (0 or 1) for methylation state.
        temps: Temperatures in Kelvin.
        salts: Salt concentrations in mM.
        pHs: pH values.

    Returns:
        List of all condition tuples.
    """
    return list(itertools.product(phosphorylation, methylation, temps, salts, pHs))

class Designer:

    def __init__(self, model_name: str, ff_type: str, pH_mode: str = 'normal', **model_kwargs):
        """Initialize a Designer for IDP sequence optimization.

        Args:
            model_name: Physics model to use ('dimer', 'rpa', 'finches', etc.).
            ff_type: Force field type ('mpipi', 'hps_tesei', 'hps_dignon',
                'calvados2', etc.). Determines bond length factor.
            pH_mode: pH handling mode passed to the Predictor.
            **model_kwargs: Additional keyword arguments forwarded to the
                Predictor and underlying physics model.
        """
        self.model_name = model_name
        self.ff_type = ff_type
        self.model_kwargs = model_kwargs
        self.pH_mode = pH_mode
        if ff_type not in bond_factors.keys():
            self.bond_factor = 3.6
        else:
            self.bond_factor = bond_factors[ff_type]

        if self.model_name == 'rpa':
            self.rpa_check = True
        else:
            self.rpa_check = False


    def convert_to_normalized_chi(self, value: float, seq_length: int) -> float:
        """Convert a raw interaction parameter to a normalized chi value.

        For the dimer model, divides the raw virial coefficient by
        N^2 * l^3 where l = 0.38 * bond_factor. For self-interactions
        this yields 0.5 - chi (normalized virial); for cross-interactions
        it yields chi directly. For non-dimer models, returns the value
        as-is cast to float32.

        Args:
            value: Raw interaction parameter from the physics model.
            seq_length: Length of the sequence (N).

        Returns:
            Normalized interaction parameter as float32.
        """
        if self.model_name == 'dimer':
            l = 0.38 * self.bond_factor
            return value / (jnp.float32(seq_length)**2 * jnp.float32(l)**3)
        else:
            return jnp.float32(value)

    def convert_to_chi(self, value: float) -> float:
        """Convert a normalized virial value to the chi parameter.

        For the dimer model, computes 0.5 - value (inverting the
        normalized virial convention). For other models, returns the
        value unchanged.

        Args:
            value: Normalized virial (dimer) or raw chi (other models).

        Returns:
            Chi parameter as float32.
        """
        return jnp.float32(0.5 - value if self.model_name == 'dimer' else value)


    def get_metapredict_function(self, num_iterations: int, sequence_length: int) -> Tuple[callable, jnp.ndarray]:
        """Build the intrinsic disorder constraint function and annealing schedule.

        Loads MetapredictV3 and constructs a differentiable loss multiplier
        that penalizes sequences with predicted disorder below an annealed
        minimum threshold. The threshold ramps from 0.2 to 0.8 over the
        first 80% of iterations, then holds at 0.8.

        Args:
            num_iterations: Total number of optimization iterations.
            sequence_length: Length of the target sequence.

        Returns:
            disorder_loss_multiplier: Callable (pseq, min_disorder) ->
                (scalar, avg_disorder) that returns a loss scaling factor
                and the mean predicted disorder.
            min_disorders: Array of shape (num_iterations,) with the
                per-iteration minimum disorder thresholds.
        """
        relu_steep_slope = jnp.float32(-10.0)
        relu_slope_scale = jnp.float32(100.0)
        start_min_disorder = jnp.float32(0.2)
        end_min_disorder = jnp.float32(0.8)

        relu_flattened_slope = relu_steep_slope / relu_slope_scale
        disorder_anneal_frac = jnp.float32(0.8)

        n_disorder_anneal_iters = int(num_iterations * disorder_anneal_frac)

        min_disorders = onp.concatenate([onp.linspace(start_min_disorder, end_min_disorder, n_disorder_anneal_iters), onp.full((num_iterations - n_disorder_anneal_iters,), end_min_disorder)])
        jax_mp = MetapredictJAX.load_params(ROOT / 'ff_params' / 'metapredict_V3_params.pkl')

        jax_mp_fn = jax_mp.return_metapredict_fn()

        def disorder_loss_multiplier_helper(avg_disorder: float, min_disorder: float):
            offset = 1 - relu_steep_slope*min_disorder
            offset_small = 1- relu_flattened_slope * min_disorder

            return jnp.where(
                avg_disorder < min_disorder,
                relu_steep_slope * avg_disorder + offset,
                relu_flattened_slope * avg_disorder + offset_small
            )

        def disorder_loss_multiplier(pseq: jnp.ndarray, min_disorder: float):
            mp_pseq = pseq_to_mp_pseq(pseq)
            _, avg_disorder = jax_mp_fn(jnp.expand_dims(mp_pseq, axis = 0).astype(jnp.float32))

            return disorder_loss_multiplier_helper(avg_disorder, min_disorder), avg_disorder

        return disorder_loss_multiplier, min_disorders
    
    def design_batch(self,
    sequence_length: int,
    num_sequences: int,
    num_iterations: int,
    temps: List[float | int],
    salts: List[float | int],
    pHs: List[float | int],
    phosphorylation: List[int],
    methylation: List[int],
    target_values,
    metapredict_constraint: bool = True,
    **kwargs):
        """
        Batch-parallel value optimization. Runs num_sequences independent
        optimizations simultaneously via vmap, compiled into a single
        jax.lax.scan dispatch.

        Args:
            target_values: single float (same target for all sequences)
                           or array-like of shape (num_sequences,) for
                           per-sequence targets. Values are in chi units
                           for the dimer model.
        Returns:
            final_seqs: list of num_sequences optimized sequence strings
            results: dict with 'all_losses', 'all_chi', 'final_chi', 'final_pseqs'
        """

        all_conditions = generate_all_conditions(phosphorylation, methylation, temps, salts, pHs)
        if len(all_conditions) != 1:
            raise ValueError("design_batch requires exactly 1 condition")

        predictor = Predictor(
            model_name=self.model_name, ff_type=self.ff_type,
            temps=temps, salts=salts, pHs=pHs, pH_mode=self.pH_mode,
            **self.model_kwargs,
        )

        if isinstance(target_values, (int, float)):
            targets_arr = jnp.full(num_sequences, target_values, dtype=jnp.float32)
        else:
            targets_arr = jnp.asarray(target_values, dtype=jnp.float32)
        if self.model_name == 'dimer':
            targets_arr = jnp.float32(0.5) - targets_arr

        if self.rpa_check:
            all_pHs = [c[-1] for c in all_conditions]
            aa_interactions = predictor.model.set_aa_interactions(
                sequence_length, sequence_length, all_pHs)
        else:
            aa_interactions = predictor.aa_interactions.astype(jnp.float32)

        l = 0.38 * self.bond_factor
        chi_divisor = jnp.float32(sequence_length**2 * l**3) if self.model_name == 'dimer' else jnp.float32(1.0)

        if metapredict_constraint:
            disorder_loss_fn, min_disorders_np = self.get_metapredict_function(
                num_iterations, sequence_length)
            min_disorders = jnp.asarray(min_disorders_np, dtype=jnp.float32)
        else:
            disorder_loss_fn = None
            min_disorders = jnp.zeros(num_iterations, dtype=jnp.float32)

        def single_loss(logits, target, temperature, min_disorder):
            logits = logits.at[:, 20].set(0.0)
            logits = logits.at[:, 21].set(0.0)
            logits = logits.at[:, 22].set(0.0)
            pseq = jax.nn.softmax(logits / temperature, axis=-1)

            raw_out = predictor.predict_pseqs(pseq, pseq, aa_interactions)
            normalized = raw_out[0] / chi_divisor

            loss = jnp.sqrt((normalized - target) ** 2 + jnp.float32(1e-8))

            if metapredict_constraint:
                scalar, _ = disorder_loss_fn(pseq[:, :20], min_disorder=min_disorder)
                loss = loss * scalar

            return loss, normalized

        batched_val_grad = jax.vmap(
            jax.value_and_grad(single_loss, has_aux=True),
            in_axes=(0, 0, None, None),
        )

        keys = jax.random.split(jax.random.PRNGKey(42), num_sequences)
        init_logits = jax.vmap(
            lambda k: jax.random.normal(k, (sequence_length, len(RES_ALPHA)), dtype=jnp.float32)
        )(keys)
        init_logits = init_logits.at[:, :, 20].set(0.0)
        init_logits = init_logits.at[:, :, 21].set(0.0)
        init_logits = init_logits.at[:, :, 22].set(0.0)

        batch_params = {'logits': init_logits}
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=0.1),
        )
        opt_state = optimizer.init(batch_params)

        gumbel_temps = jnp.linspace(1.0, 0.001, num_iterations, dtype=jnp.float32)

        def scan_body(carry, xs):
            params, opt_state = carry
            temperature, min_disorder = xs

            (losses, chi_vals), grads_per_seq = batched_val_grad(
                params['logits'], targets_arr, temperature, min_disorder
            )

            grads = {'logits': grads_per_seq}
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            return (params, opt_state), (losses, chi_vals)

        def run_opt(carry, xs):
            return jax.lax.scan(scan_body, carry, xs)

        print(f"Compiling batch of {num_sequences} sequences x {num_iterations} iterations...")
        (final_params, _), (all_losses, all_chi_raw) = jax.jit(run_opt)(
            (batch_params, opt_state), (gumbel_temps, min_disorders)
        )

        if self.model_name == 'dimer':
            all_chi = jnp.float32(0.5) - all_chi_raw
        else:
            all_chi = all_chi_raw

        final_logits = final_params['logits']
        final_pseqs = jax.nn.softmax(final_logits / 0.001, axis=-1)
        final_indices = jnp.argmax(final_pseqs, axis=-1)

        final_seqs = []
        for b in range(num_sequences):
            seq = ''.join([RES_ALPHA[int(idx)] for idx in final_indices[b]])
            final_seqs.append(seq)

        final_chi = all_chi[-1]
        final_losses = all_losses[-1]

        print('Optimization complete!')
        n_print = min(num_sequences, 10)
        for b in range(n_print):
            seq_preview = final_seqs[b] if len(final_seqs[b]) <= 60 else final_seqs[b][:57] + '...'
            print(f'  Seq {b}: chi={float(final_chi[b]):.6f}  loss={float(final_losses[b]):.6f}  {seq_preview}')
        if num_sequences > n_print:
            print(f'  ... and {num_sequences - n_print} more sequences')

        return final_seqs, {
            'all_losses': all_losses,
            'all_chi': all_chi,
            'final_chi': final_chi,
            'final_pseqs': final_pseqs,
        }

    def design_sequence(self, 
    sequence_length: int, 
    design_type: str, 
    num_iterations: int, 
    temps: List[float | int], 
    salts: List[float | int], 
    pHs: List[float | int], 
    phosphorylation: List[int],
    methylation: List[int], 
    metapredict_constraint: bool = True, 
    meth_locs: Optional[List[int]] = None, 
    phos_locs: Optional[List[int]] = None, 
    **kwargs):
        """Top-level dispatcher for single-sequence optimization.

        Validates arguments and routes to the appropriate run_* method
        based on design_type.

        Args:
            sequence_length: Length of the designed sequence.
            design_type: One of 'value', '1d_switch', 'reference',
                'multi_dimensional_switch', 'designed_response',
                'interaction_matrix', or 'two_sequence'.
            num_iterations: Number of gradient descent iterations.
            temps: Temperatures in Kelvin.
            salts: Salt concentrations in mM.
            pHs: pH values.
            phosphorylation: Binary flags for phosphorylation states.
            methylation: Binary flags for methylation states.
            metapredict_constraint: If True, enforce intrinsic disorder
                constraint via MetapredictV3.
            meth_locs: Sequence positions for methylation modifications.
            phos_locs: Sequence positions for phosphorylation modifications.
            **kwargs: Design-type-specific arguments (target_value, threshold,
                response_type, reference_sequence, targets, opt_type,
                response_profile, behavior, etc.).

        Returns:
            For most design types: (final_sequence, aux_data).
            For 'two_sequence': (seq1, seq2, aux_data).

        Raises:
            ValueError: If design_type is invalid or required kwargs are missing.
        """
        all_conditions = generate_all_conditions(phosphorylation, methylation, temps, salts, pHs)

        if design_type not in ['value', '1d_switch', 'reference', 'multi_dimensional_switch', 'designed_response', 'interaction_matrix', 'two_sequence']:
            raise ValueError("design_type must be either 'value', '1d_switch', 'reference', 'multi_dimensional_switch', 'designed_response', 'interaction_matrix', or 'two_sequence'")

        if design_type == 'value':
            if kwargs['target_value'] is None:
                raise ValueError("target_value must be provided for design_type = 'value'")

            if len(all_conditions) != 1:
                raise ValueError("Must have exactly 1 condition for design_type = 'value'")

            target_value = kwargs.pop('target_value')

            predictor = Predictor(model_name = self.model_name, ff_type = self.ff_type, temps = temps, salts = salts, pHs = pHs, pH_mode = self.pH_mode, **self.model_kwargs)

            final_seq, aux = self.run_value_optimization(predictor = predictor, 
                                    num_iterations = num_iterations, 
                                    all_conditions = all_conditions, 
                                    target_value = target_value, 
                                    sequence_length = sequence_length, 
                                    metapredict_constraint = metapredict_constraint, 
                                    **kwargs)

            return final_seq, aux
        
        elif design_type == '1d_switch':
            if kwargs.get('threshold') is None:
                # set to default value of 1/sqrt(seq_length)
                threshold = 1.0 / jnp.sqrt(sequence_length)
            else:
                threshold = kwargs.pop('threshold')

            if kwargs.get('response_type') is None or kwargs.get('response_type') not in ['expander', 'contractor']:
                raise ValueError('Must specify response_type (expander or contractor) for design_type = "1d_switch"')
            else:
                response_type = kwargs.pop('response_type')

            if len(all_conditions) != 2:
                raise ValueError("Must have exactly 2 conditions for design_type = '1d_switch'")

            predictor = Predictor(model_name = self.model_name, ff_type = self.ff_type, temps = temps, salts = salts, pHs = pHs, pH_mode = self.pH_mode, **self.model_kwargs)

            final_seq, aux = self.run_1d_switch_optimization(predictor, 
                            all_conditions, 
                            threshold, 
                            num_iterations, 
                            response_type, 
                            sequence_length, 
                            phos_locs, 
                            meth_locs, metapredict_constraint, **kwargs)

            return final_seq, aux

        elif design_type == 'reference':
            if kwargs.get('reference_sequence') is None:
                raise ValueError("reference_sequence must be provided for design_type = 'reference'")

            if len(all_conditions) != 1:
                raise ValueError("Must have exactly 1 condition for design_type = 'reference'")

            if kwargs.get('opt_type') is None:
                raise ValueError('Must specify opt_type for inter-chain interactions with design_type = "reference"')

            opt_type = kwargs.pop('opt_type')

            predictor = Predictor(model_name = self.model_name, ff_type = self.ff_type, temps = temps, salts = salts, pHs = pHs, pH_mode = self.pH_mode, **self.model_kwargs)

            final_seq, aux = self.run_reference_optimization(predictor, 
                                all_conditions, 
                                kwargs.pop('reference_sequence'), 
                                num_iterations, 
                                sequence_length, 
                                metapredict_constraint, 
                                opt_type,
                                **kwargs)

            return final_seq, aux

        elif design_type == 'multi_dimensional_switch':
            if kwargs.get('targets') is None:
                raise ValueError('Must specify targets for design_type = "multi_dimensional_switch"')

            if len(all_conditions) != len(kwargs['targets']):
                raise ValueError('Must have exactly as many targets as conditions for design_type = "multi_dimensional_switch"')

            if kwargs.get('threshold') is None:
                threshold = 1.0 / jnp.sqrt(sequence_length)
            else:
                threshold = kwargs.pop('threshold')

            targets = kwargs.pop('targets')
            targets = jnp.array(targets)

            predictor = Predictor(model_name = self.model_name, ff_type = self.ff_type, temps = temps, salts = salts, pHs = pHs, pH_mode = self.pH_mode, **self.model_kwargs)

            final_seq, aux = self.run_multi_dimensional_switch_optimization(predictor = predictor, 
                                    all_conditions = all_conditions, 
                                    num_iterations = num_iterations, 
                                    threshold = threshold, 
                                    targets = targets, 
                                    sequence_length = sequence_length, 
                                    metapredict_constraint = metapredict_constraint, 
                                    phos_locs = phos_locs, 
                                    meth_locs = meth_locs, 
                                    **kwargs)

            return final_seq, aux

        elif design_type == 'designed_response':
            if kwargs.get('response_type') is None or kwargs.get('response_type') not in ['contractor', 'expander']:
                raise ValueError('Must specify response_type (contractor or expander) for design_type = "designed_response"')

            response_type = kwargs.pop('response_type')

            if kwargs.get('response_profile') is None or kwargs.get('response_profile') not in ['linear', 'early_step', 'late_step', 'bandpass']:
                raise ValueError('Must specify response_profile (linear, early_step, late_step, bandpass) for design_type = "designed_response"')

            response_profile = kwargs.pop('response_profile')

            if kwargs.get('threshold') is None:
                threshold = 1.0 / jnp.sqrt(sequence_length)
            else:
                threshold = kwargs.pop('threshold')

            if kwargs.get('num_subdivisions') is None:
                num_subdivisions = 7
            else:
                num_subdivisions = kwargs.pop('num_subdivisions')

            if len(all_conditions) != 2:
                raise ValueError('Must have exactly 2 conditions (endpoints) for design_type = "designed_response"')

            all_temps = list(set([condition[2] for condition in all_conditions]))
            all_salts = list(set([condition[3] for condition in all_conditions]))
            all_pHs = list(set([condition[4] for condition in all_conditions]))
            all_phosphorylation = list(set([condition[0] for condition in all_conditions]))
            all_methylation = list(set([condition[1] for condition in all_conditions]))

            lens_is_two = {name: len(vals) == 2 for name, vals in [
                        ("temps", all_temps),
                        ("salts", all_salts),
                        ("pHs", all_pHs),
                        ("phosphorylation", all_phosphorylation),
                        ("methylation", all_methylation),
                    ]}

            condition_dict = {name: value for name, value in [('temps', all_temps), 
                ('salts', all_salts), 
                ('pHs', all_pHs), 
                ('phosphorylation', all_phosphorylation), 
                ('methylation', all_methylation)]}

            critical_condition = list({k for k, v in lens_is_two.items() if v})[0]

            if critical_condition in ['phosphorylation', 'methylation']:
                predictor = Predictor(model_name = self.model_name, ff_type = self.ff_type, temps = temps, salts = salts, pHs = pHs, pH_mode = self.pH_mode, **self.model_kwargs)
            elif critical_condition in ['temps', 'salts', 'pHs']:
                ends = [condition_dict[critical_condition][0], condition_dict[critical_condition][1]]
                if critical_condition in ['temps']:
                    temps = list(onp.linspace(ends[0], ends[1], num_subdivisions))
                elif critical_condition in ['salts']:
                    salts = list(onp.linspace(ends[0], ends[1], num_subdivisions))
                elif critical_condition in ['pHs']:
                    pHs = list(onp.linspace(ends[0], ends[1], num_subdivisions))

                predictor = Predictor(model_name = self.model_name, ff_type = self.ff_type, temps = temps, salts = salts, pHs = pHs, pH_mode = self.pH_mode, **self.model_kwargs)

            all_conditions = generate_all_conditions(phosphorylation, methylation, temps, salts, pHs)


            argmax_seq, aux = self.run_designed_response_optimization(predictor = predictor, 
                                    all_conditions = all_conditions, 
                                    num_iterations = num_iterations, 
                                    sequence_length = sequence_length, 
                                    metapredict_constraint = metapredict_constraint, 
                                    phos_locs = phos_locs, 
                                    meth_locs = meth_locs, 
                                    response_type = response_type, 
                                    response_profile = response_profile, 
                                    threshold = threshold,
                                    critical_condition = critical_condition,
                                    num_subdivisions = num_subdivisions,
                                    **kwargs)

            return argmax_seq, aux

        # for initial testing - just make a loop
        elif design_type == 'interaction_matrix':
            predictor = Predictor(model_name = self.model_name, ff_type = self.ff_type, temps = temps, salts = salts, pHs = pHs, pH_mode = self.pH_mode, **self.model_kwargs)

            argmax_seq, aux = self.run_interaction_matrix_optimization(predictor = predictor,
                                    all_conditions = all_conditions,
                                    num_iterations = num_iterations,
                                    sequence_length = sequence_length,
                                    metapredict_constraint = metapredict_constraint,
                                    **kwargs)

            return argmax_seq, aux

        elif design_type == 'two_sequence':
            if kwargs.get('behavior') is None or kwargs.get('behavior') not in ['condensed_mix', 'condensed_demix']:
                raise ValueError('Must specify behavior (condensed_mix or condensed_demix) for design_type = "two_sequence"')

            if len(all_conditions) != 1:
                raise ValueError('Must have exactly 1 condition for design_type = "two_sequence"')

            behavior = kwargs.pop('behavior')

            predictor = Predictor(model_name = self.model_name, ff_type = self.ff_type, temps = temps, salts = salts, pHs = pHs, pH_mode = self.pH_mode, **self.model_kwargs)

            seq1, seq2, aux = self.run_two_sequence_optimization(predictor = predictor,
                                all_conditions = all_conditions,
                                num_iterations = num_iterations,
                                sequence_length = sequence_length,
                                metapredict_constraint = metapredict_constraint,
                                behavior = behavior,
                                **kwargs)

            return seq1, seq2, aux

    def run_interaction_matrix_optimization(self,
    predictor: Predictor,
    all_conditions: List[Tuple[int, int, float, float, float]],
    num_iterations: int,
    sequence_length: int,
    metapredict_constraint: bool = True,
    **kwargs):
        """Optimize a sequence to produce a target residue-residue interaction matrix.

        Designs a sequence whose pairwise interaction matrix has attractive
        (negative) interactions between the N- and C-terminal regions and
        repulsive (positive) interactions in the middle.

        Args:
            predictor: Initialized Predictor instance.
            all_conditions: List of condition tuples.
            num_iterations: Number of optimization iterations.
            sequence_length: Length of the designed sequence.
            metapredict_constraint: Whether to apply disorder constraint.
            **kwargs: Forwarded to optimization_loop.

        Returns:
            argmax_seq: Optimized amino acid sequence string.
            aux: Tuple of (all_losses, all_values, all_pseqs).
        """
        if metapredict_constraint:
            disorder_loss_multiplier, min_disorders = self.get_metapredict_function(num_iterations, sequence_length)
        else:
            disorder_loss_multiplier = None
            min_disorders = None

        if self.rpa_check:
            all_pHs = [condition[-1] for condition in all_conditions]
            aa_interactions = predictor.model.set_aa_interactions(sequence_length, sequence_length, all_pHs)
        else:
            aa_interactions = predictor.aa_interactions.astype(jnp.float32)

        def loss_fn(params: dict, temperature: float, disorder_loss_fn: callable, min_disorder: float = 0.2) -> float:
            logits = params['logits']
            logits = logits.at[:, 20].set(0.0) #to remove the dummy X residue
            logits = logits.at[:, 21].set(0.0) #to remove dummy Z residue (charged Ser)
            logits = logits.at[:, 22].set(0.0) #to remove dummy B residue (charged Tyr)
            pseq = self.normalize_logits(logits, temperature)

            matrix_out = predictor.predict_interaction_matrix_pseqs(pseq, pseq, aa_interactions) #shape : (sequence_length, sequence_length)

            # for loop - want the two ends of the sequence to attract one another, the middle to repel itself
            loss = 0.0


            top_right_corner = matrix_out[:5 , -5:]
            bottom_left_corner = matrix_out[-5:, :5]
            middle = matrix_out[5:-5, 5:-5]


            # want top_right_corner and bottom_left_corner to be negative, middle to be positive
            # want these quantities to be as large as possible
            loss += jnp.sum(top_right_corner) + jnp.sum(bottom_left_corner) - jnp.sum(middle)

            if metapredict_constraint:
                disorder_scalar, avg_disorder = disorder_loss_fn(pseq[:, :20], min_disorder = min_disorder)
                loss *= disorder_scalar

                return loss, (matrix_out, pseq, avg_disorder, disorder_scalar)
            
            return loss, (matrix_out, pseq)

        argmax_seq, aux = self.optimization_loop(predictor, loss_fn, sequence_length = sequence_length, num_iterations = num_iterations, disorder_loss_fn = disorder_loss_multiplier, metapredict_constraint = metapredict_constraint, min_disorders = min_disorders, **kwargs)

        return argmax_seq, aux

    def run_value_optimization(self, 
    predictor: Predictor, 
    num_iterations: int, 
    all_conditions: List[Tuple[int, int, float, float, float]], 
    target_value: float, 
    sequence_length: int, 
    metapredict_constraint: bool = True, 
    **kwargs):
        """Optimize a sequence to match a target chi (interaction parameter) value.

        Minimizes |chi_predicted - chi_target| under a single thermodynamic
        condition, optionally subject to an intrinsic disorder constraint.

        Args:
            predictor: Initialized Predictor instance.
            num_iterations: Number of optimization iterations.
            all_conditions: Single-element list of the condition tuple.
            target_value: Desired chi value. Internally converted to
                normalized virial (0.5 - target) for the dimer model.
            sequence_length: Length of the designed sequence.
            metapredict_constraint: Whether to apply disorder constraint.
            **kwargs: Forwarded to optimization_loop.

        Returns:
            argmax_seq: Optimized amino acid sequence string.
            aux: Tuple of (all_losses, all_values, all_pseqs).
        """
        target_value = 0.5 - target_value if self.model_name == 'dimer' else target_value
        # define metapredict parameters if needed
        if metapredict_constraint:
            disorder_loss_multiplier, min_disorders = self.get_metapredict_function(num_iterations, sequence_length)
        else:
            disorder_loss_multiplier = None
            min_disorders = None

        if self.rpa_check:
            all_pHs = [condition[-1] for condition in all_conditions]
            aa_interactions = predictor.model.set_aa_interactions(sequence_length, sequence_length, all_pHs)
        else:
            aa_interactions = predictor.aa_interactions.astype(jnp.float32)

        # define loss function
        def loss_fn(params: dict, temperature: float, disorder_loss_fn: callable, min_disorder: float = 0.2) -> float:
            logits = params['logits']
            logits = logits.at[:, 20].set(0.0) #to remove the dummy X residue
            logits = logits.at[:, 21].set(0.0) #to remove dummy Z residue (charged Ser)
            logits = logits.at[:, 22].set(0.0) #to remove dummy B residue (charged Tyr)
            pseq = self.normalize_logits(logits, temperature)

            out = self.convert_to_normalized_chi(predictor.predict_pseqs(pseq, pseq, aa_interactions), sequence_length)

            # loss = jnp.sum((jnp.sqrt(out[0] - target_value))** 2)
            loss = jnp.sum(jnp.sqrt((out[0] - target_value)**2) )

            if metapredict_constraint:
                disorder_scalar, avg_disorder = disorder_loss_fn(pseq[:, :20], min_disorder = min_disorder)
                loss *= disorder_scalar

                return loss, (out, pseq, avg_disorder, disorder_scalar)
            
            return loss, (out, pseq)


        argmax_seq, aux = self.optimization_loop(predictor, loss_fn, sequence_length = sequence_length, num_iterations = num_iterations, disorder_loss_fn = disorder_loss_multiplier, metapredict_constraint = metapredict_constraint, min_disorders = min_disorders, **kwargs)

        return argmax_seq, aux
             

    def run_1d_switch_optimization(self, 
    predictor: Predictor, 
    all_conditions: List[Tuple[int, int, float, float, float]], 
    threshold: float, 
    num_iterations: int,
    response_type: str,
    sequence_length: int, 
    phos_locs: Optional[List[int]] = None, 
    meth_locs: Optional[List[int]] = None, 
    metapredict_constraint: bool = True, 
    **kwargs):
        """Optimize a sequence that switches between expanded and condensed states.

        Designs a sequence whose chi value crosses a threshold between two
        conditions — e.g., condensed under condition A and expanded under
        condition B (contractor), or vice versa (expander).

        Args:
            predictor: Initialized Predictor instance.
            all_conditions: Exactly 2 condition tuples (the two switch states).
            threshold: Chi separation threshold; defaults to 1/sqrt(N).
            num_iterations: Number of optimization iterations.
            response_type: 'contractor' (high chi -> low chi) or
                'expander' (low chi -> high chi).
            sequence_length: Length of the designed sequence.
            phos_locs: Positions for phosphorylation modification.
            meth_locs: Positions for methylation modification.
            metapredict_constraint: Whether to apply disorder constraint.
            **kwargs: Forwarded to optimization_loop.

        Returns:
            argmax_seq: Optimized amino acid sequence string.
            aux: Tuple of (all_losses, all_values, all_pseqs).
        """
        # define metapredict parameters if needed
        if metapredict_constraint:
            disorder_loss_multiplier, min_disorders = self.get_metapredict_function(num_iterations, sequence_length)
        else:
            disorder_loss_multiplier = None
            min_disorders = None

        if self.rpa_check:
            all_pHs = [condition[-1] for condition in all_conditions]
            aa_interactions = predictor.model.set_aa_interactions(sequence_length, sequence_length, all_pHs)
        else:
            aa_interactions = predictor.aa_interactions.astype(jnp.float32)

        targets = jnp.array([1,0]) if response_type == 'contractor' else jnp.array([0,1])

        phos_check = phos_locs is not None
        methyl_check = meth_locs is not None
            

        def loss_fn(params: dict, temperature: float, disorder_loss_fn: callable, min_disorder: float = 0.2) -> float:
            logits = params['logits']
            logits = logits.at[:, 20].set(0.0) #to remove the dummy X residue
            logits = logits.at[:, 21].set(0.0) #to remove dummy Z residue (charged Ser)
            logits = logits.at[:, 22].set(0.0) #to remove dummy B residue (charged Tyr)
            pseq = self.normalize_logits(logits, temperature)

            if phos_check:
                pseq = set_pseq_residues(pseq, phos_locs, 'S')
                mod_pseq = pseq.copy()
                mod_pseq = set_pseq_residues(mod_pseq, phos_locs, 'Z')

            if methyl_check:
                pseq = set_pseq_residues(pseq, meth_locs, 'R')
                mod_pseq = pseq.copy()
                mod_pseq = set_pseq_residues(mod_pseq, meth_locs, 'X')

            out = self.convert_to_normalized_chi(predictor.predict_pseqs(pseq, pseq, aa_interactions), sequence_length)

            if phos_check or methyl_check:
                out = jnp.array([out[0], self.convert_to_normalized_chi(predictor.predict_pseqs(mod_pseq, mod_pseq, aa_interactions)[0], sequence_length)])
            

            penalty = jnp.where(targets == 1, - out + threshold, out + threshold)

            loss = jnp.sum(jnp.maximum(0.0, penalty))

            if metapredict_constraint:
                disorder_scalar, avg_disorder = disorder_loss_fn(pseq[:, :20], min_disorder = min_disorder)
                loss *= disorder_scalar

                return loss, (out, pseq, avg_disorder, disorder_scalar)

            return loss, (out, pseq)

        argmax_seq, aux = self.optimization_loop(predictor, loss_fn, sequence_length = sequence_length, num_iterations = num_iterations, disorder_loss_fn = disorder_loss_multiplier, metapredict_constraint = metapredict_constraint, min_disorders = min_disorders, **kwargs)

        return argmax_seq, aux


    def run_designed_response_optimization(self,
    predictor: Predictor,
    all_conditions: List[Tuple[int, int, float, float, float]],
    num_iterations: int,
    sequence_length: int,
    response_type: str,
    response_profile: str,
    threshold: float,
    critical_condition: str,
    metapredict_constraint: bool = True,
    num_subdivisions: int = 7,
    phos_locs: Optional[List[int]] = None,
    meth_locs: Optional[List[int]] = None,
    **kwargs):
        """Optimize a sequence to follow a prescribed response profile across conditions.

        Designs a sequence whose chi varies according to a target shape
        (linear, early_step, late_step, or bandpass) as a single condition
        variable is swept between two endpoints.

        Args:
            predictor: Initialized Predictor instance.
            all_conditions: Condition tuples spanning the sweep range.
            num_iterations: Number of optimization iterations.
            sequence_length: Length of the designed sequence.
            response_type: 'contractor' or 'expander' — sets the sign
                of the target response direction.
            response_profile: Shape of the desired response curve:
                'linear', 'early_step', 'late_step', or 'bandpass'.
            threshold: Magnitude of the target chi change.
            critical_condition: The condition variable being swept
                ('temps', 'salts', 'pHs', 'phosphorylation', or 'methylation').
            metapredict_constraint: Whether to apply disorder constraint.
            num_subdivisions: Number of evenly spaced evaluation points
                along the swept condition axis.
            phos_locs: Positions for phosphorylation modification.
            meth_locs: Positions for methylation modification.
            **kwargs: Forwarded to optimization_loop.

        Returns:
            argmax_seq: Optimized amino acid sequence string.
            aux: Tuple of (all_losses, all_values, all_pseqs).
        """
        if metapredict_constraint:
            disorder_loss_multiplier, min_disorders = self.get_metapredict_function(num_iterations, sequence_length)
        else:
            disorder_loss_multiplier = None
            min_disorders = None

        if self.rpa_check:
            all_pHs = [condition[-1] for condition in all_conditions]
            aa_interactions = predictor.model.set_aa_interactions(sequence_length, sequence_length, all_pHs)
        else:
            aa_interactions = predictor.aa_interactions

        if critical_condition in ['phosphorylation', 'methylation']:
            if critical_condition == 'phosphorylation' and len(phos_locs) + 1 !=  num_subdivisions:
                raise ValueError(f'Must have exactly {num_subdivisions - 1} phosphorylation locations for phosphorylation designed response')
            if critical_condition == 'methylation' and len(meth_locs) + 1 !=  num_subdivisions:
                raise ValueError(f'Must have exactly {num_subdivisions} methylation locations for methylation designed response')

        direction = -1 if response_type == 'contractor' else 1
        orig_residue = 'S' if critical_condition == 'phosphorylation' else 'R'
        mod_residue = 'Z' if critical_condition == 'phosphorylation' else 'X'

        def loss_fn(params: dict, temperature: float, disorder_loss_fn: callable, min_disorder: float = 0.2) -> float:
            logits = params['logits']
            logits = logits.at[:, 20].set(0.0) #to remove the dummy X residue
            logits = logits.at[:, 21].set(0.0) #to remove dummy Z residue (charged Ser)
            logits = logits.at[:, 22].set(0.0) #to remove dummy B residue (charged Tyr)
            pseq = self.normalize_logits(logits, temperature)

            ### evaluate pseqs for all incremental conditions - need these annoying loops for phos/methyl but temps/salts/pHs can be parallelized
            out = []
            all_mod_pseqs = []
            if critical_condition in ['phosphorylation', 'methylation']:
                locations = phos_locs if critical_condition == 'phosphorylation' else meth_locs
                pseq = set_pseq_residues(pseq, locations, orig_residue)
                all_mod_pseqs.append(pseq)
                mod_pseq = pseq.copy()
                out.append(self.convert_to_normalized_chi(predictor.predict_pseqs(mod_pseq, mod_pseq, aa_interactions)[0], sequence_length))
                for loc in locations:
                    mod_pseq = set_pseq_residues(mod_pseq, [loc], mod_residue)
                    out.append(self.convert_to_normalized_chi(predictor.predict_pseqs(mod_pseq, mod_pseq, aa_interactions)[0], sequence_length))
                    all_mod_pseqs.append(mod_pseq)
            else:
                out = self.convert_to_normalized_chi(predictor.predict_pseqs(pseq, pseq, aa_interactions), sequence_length)

            out = jnp.array(out)

            loss = 0.0
            
            if response_profile == 'bandpass':
                first_half = jnp.linspace(0, direction*threshold, num_subdivisions//2 + num_subdivisions%2)
                second_half = jnp.linspace(0, direction*threshold, num_subdivisions//2)
                ideal_out = jnp.concatenate([first_half, second_half[::-1]])
                loss += jnp.sum(jnp.sqrt((out - ideal_out)**2))
            elif response_profile == 'linear':
                loss += jnp.sqrt((direction*out[0] + threshold)**2)
                loss += jnp.sqrt((-direction*out[-1] + threshold)**2)
            elif response_profile == 'early_step':
                diffs = jnp.diff(out)

                low_cond_val = -direction*threshold
                high_cond_val = direction*threshold

                switch_point = num_subdivisions//2

                # ideal_out = jnp.array([low_cond_val for _ in range(len(out[:switch_point]))] +  [(low_cond_val + high_cond_val)/2] + [high_cond_val for _ in range(len(out[switch_point + 1:]))])
                # loss += jnp.sum(jnp.sqrt((out - ideal_out)**2))

                loss += jnp.sqrt((low_cond_val - out[0])**2) + jnp.sqrt((high_cond_val - out[-1])**2) + jnp.sqrt((low_cond_val - out[switch_point])**2)
            elif response_profile == 'late_step':
                diffs = jnp.diff(out)
                low_cond_val = -direction*threshold
                high_cond_val = direction*threshold

                switch_point = num_subdivisions//2

                # ideal_out = jnp.array([low_cond_val for _ in range(len(out[:switch_point]))] +  [(low_cond_val + high_cond_val)/2] + [high_cond_val for _ in range(len(out[switch_point + 1:]))])
                # loss += jnp.sum(jnp.sqrt((out - ideal_out)**2))
                loss += jnp.sqrt((low_cond_val - out[0])**2) + jnp.sqrt((high_cond_val - out[-1])**2) + jnp.sqrt((high_cond_val - out[switch_point])**2)


            if metapredict_constraint:
                disorder_scalar, avg_disorder = disorder_loss_fn(pseq[:, :20], min_disorder = min_disorder)
                loss *= disorder_scalar

                return loss, (out, pseq, avg_disorder, disorder_scalar)

            return loss, (out, pseq)
        

        argmax_seq, aux = self.optimization_loop(predictor, loss_fn, sequence_length = sequence_length, num_iterations = num_iterations, disorder_loss_fn = disorder_loss_multiplier, metapredict_constraint = metapredict_constraint, min_disorders = min_disorders, **kwargs)

        return argmax_seq, aux


    def run_multi_dimensional_switch_optimization(self,
    predictor: Predictor,
    all_conditions: List[Tuple[int, int, float, float, float]],
    num_iterations: int,
    threshold: float,
    targets: jnp.ndarray,
    sequence_length: int,
    phos_locs: Optional[List[int]] = None, 
    meth_locs: Optional[List[int]] = None, 
    metapredict_constraint: bool = True,
    **kwargs):
        """Optimize a sequence that switches across multiple conditions simultaneously.

        Generalizes 1D switching to N conditions. Each condition is assigned
        a binary target (1 = condensed, 0 = expanded) and the optimizer
        drives chi above or below the threshold accordingly.

        Args:
            predictor: Initialized Predictor instance.
            all_conditions: One condition tuple per target.
            num_iterations: Number of optimization iterations.
            threshold: Chi separation threshold.
            targets: Array of shape (N,) with binary values (0 or 1)
                indicating desired state per condition.
            sequence_length: Length of the designed sequence.
            phos_locs: Positions for phosphorylation modification.
            meth_locs: Positions for methylation modification.
            metapredict_constraint: Whether to apply disorder constraint.
            **kwargs: Forwarded to optimization_loop.

        Returns:
            argmax_seq: Optimized amino acid sequence string.
            aux: Tuple of (all_losses, all_values, all_pseqs).
        """
        # define metapredict parameters if needed
        if metapredict_constraint:
            disorder_loss_multiplier, min_disorders = self.get_metapredict_function(num_iterations, sequence_length)
        else:
            disorder_loss_multiplier = None
            min_disorders = None

        if self.rpa_check:
            all_pHs = [condition[-1] for condition in all_conditions]
            aa_interactions = predictor.model.set_aa_interactions(sequence_length, sequence_length, all_pHs)
        else:
            aa_interactions = predictor.aa_interactions.astype(jnp.float32) 


        all_phos_methyl = [condition[:2] for condition in all_conditions]

        phos_check = phos_locs is not None
        methyl_check = meth_locs is not None

        # print(all_conditions)



        def loss_fn(params: dict, temperature:float, disorder_loss_fn: callable, min_disorder: float = 0.2) -> float:
            logits = params['logits']
            logits = logits.at[:, 20].set(0.0) #to remove the dummy X residue
            logits = logits.at[:, 21].set(0.0) #to remove dummy Z residue (charged Ser)
            logits = logits.at[:, 22].set(0.0) #to remove dummy B residue (charged Tyr)

            pseq = self.normalize_logits(logits, temperature)

            phos_pseq = pseq.copy()
            meth_pseq = pseq.copy()
            phos_meth_pseq = pseq.copy()

            if phos_check:
                pseq = set_pseq_residues(pseq, phos_locs, 'S')
                phos_pseq = set_pseq_residues(phos_pseq, phos_locs, 'Z')

            if methyl_check:
                pseq = set_pseq_residues(pseq, meth_locs, 'R')
                meth_pseq = set_pseq_residues(meth_pseq, meth_locs, 'X')

            if phos_check and methyl_check:
                phos_meth_pseq = set_pseq_residues(phos_meth_pseq, phos_locs, 'Z')
                phos_meth_pseq = set_pseq_residues(phos_meth_pseq, meth_locs, 'X')

            out = []
            # print(phos_check, methyl_check)
            if phos_check or methyl_check:
                for i, phos_methyl in enumerate(all_phos_methyl[::2]):
                    p, m = phos_methyl
                    calc_pseq = (1-m)*((1-p)*pseq + p*phos_pseq) + m*((1-p)*meth_pseq + p*phos_meth_pseq)
                    intm_result = self.convert_to_normalized_chi(predictor.predict_pseqs(calc_pseq, calc_pseq), sequence_length)
                    for i, val in enumerate(intm_result):
                        out.append(val)
            else:
                out = self.convert_to_normalized_chi(predictor.predict_pseqs(pseq, pseq), sequence_length)
                # print('asdfasdf', len(out))

            out = jnp.array(out)

            # print(len(out))
            # print(threshold)

            penalty = jnp.where(targets == 1, - out + threshold, out + threshold)

            loss = jnp.sum(jnp.maximum(0.0, penalty))

            if metapredict_constraint:
                disorder_scalar, avg_disorder = disorder_loss_fn(pseq[:, :20], min_disorder = min_disorder)
                loss *= disorder_scalar

                return loss, (out, pseq, avg_disorder, disorder_scalar)

            return loss, (out, pseq)

        argmax_seq, aux = self.optimization_loop(predictor, loss_fn, sequence_length, num_iterations, disorder_loss_fn = disorder_loss_multiplier, metapredict_constraint = metapredict_constraint, min_disorders = min_disorders, **kwargs)

        return argmax_seq, aux
            

    def run_reference_optimization(self,
    predictor: Predictor,
    all_conditions: List[Tuple[float | int, float | int, float | int]],
    reference_sequence: str,
    num_iterations: int,
    sequence_length: int,
    opt_type: str,
    metapredict_constraint: bool = True,
    **kwargs):
        """Optimize a sequence for inter-chain interaction with a reference sequence.

        Designs a sequence that either strongly attracts ('client') or
        repels a fixed reference sequence, as measured by the
        cross-interaction chi parameter.

        Args:
            predictor: Initialized Predictor instance.
            all_conditions: Single-element list of the condition tuple.
            reference_sequence: Fixed partner amino acid sequence string.
            num_iterations: Number of optimization iterations.
            sequence_length: Length of the designed sequence.
            opt_type: 'client' (maximize attraction, target = -6) or
                other (maximize repulsion, target = 6).
            metapredict_constraint: Whether to apply disorder constraint.
            **kwargs: Forwarded to optimization_loop.

        Returns:
            argmax_seq: Optimized amino acid sequence string.
            aux: Tuple of (all_losses, all_values, all_pseqs).
        """
        target_value = -6 if opt_type == 'client' else 6

        # define metapredict parameters if needed
        if metapredict_constraint:
            disorder_loss_multiplier, min_disorders = self.get_metapredict_function(num_iterations, sequence_length)
        else:
            disorder_loss_multiplier = None
            min_disorders = None

        if self.rpa_check:
            all_pHs = [condition[-1] for condition in all_conditions]
            aa_interactions = predictor.model.set_aa_interactions(sequence_length, sequence_length, all_pHs)
        else:
            aa_interactions = predictor.aa_interactions.astype(jnp.float32) 

        reference_pseq = seq_to_one_hot(reference_sequence)

        def loss_fn(params: dict, temperature: float, disorder_loss_fn: callable, min_disorder: float = 0.2) -> float:
            logits = params['logits']
            logits = logits.at[:, 20].set(0.0) #to remove the dummy X residue
            logits = logits.at[:, 21].set(0.0) #to remove dummy Z residue (charged Ser)
            logits = logits.at[:, 22].set(0.0) #to remove dummy B residue (charged Tyr)
            pseq = self.normalize_logits(logits, temperature)

            out = sequence_length * self.convert_to_normalized_chi(predictor.predict_pseqs(pseq, reference_pseq, aa_interactions), sequence_length)
            # out = predictor.predict_pseqs(pseq, reference_pseq, aa_interactions)

            loss = jnp.sum(jnp.sqrt((out[0] - target_value)**2))

            out += 0.5

            if metapredict_constraint:
                disorder_scalar, avg_disorder = disorder_loss_fn(pseq[:, :20], min_disorder = min_disorder)
                loss *= disorder_scalar

                return loss, (out, pseq, avg_disorder, disorder_scalar)

            return loss, (out, pseq)

        argmax_seq, aux = self.optimization_loop(predictor, loss_fn, sequence_length = sequence_length, num_iterations = num_iterations, disorder_loss_fn = disorder_loss_multiplier, metapredict_constraint = metapredict_constraint, min_disorders = min_disorders, **kwargs)

        return argmax_seq, aux

    def run_two_sequence_optimization(self,
    predictor: Predictor,
    all_conditions: List[Tuple[int, int, float, float, float]],
    num_iterations: int,
    sequence_length: int,
    metapredict_constraint: bool = True,
    behavior: str = 'condensed_mix',
    **kwargs):
        """Co-optimize two sequences for joint phase behavior.

        Simultaneously designs two sequences that are each individually
        condensed (self-interaction chi below threshold) and either
        co-condense ('condensed_mix': cross-interaction also attractive)
        or demix ('condensed_demix': cross-interaction repulsive relative
        to self-interactions).

        Logits for both sequences are concatenated into a single
        (2*sequence_length, 23) array and optimized jointly.

        Args:
            predictor: Initialized Predictor instance.
            all_conditions: Single-element list of the condition tuple.
            num_iterations: Number of optimization iterations.
            sequence_length: Length of each designed sequence.
            metapredict_constraint: Whether to apply disorder constraint.
            behavior: 'condensed_mix' or 'condensed_demix'.
            **kwargs: Forwarded to optimization_loop.

        Returns:
            seq1: First optimized amino acid sequence string.
            seq2: Second optimized amino acid sequence string.
            aux: Tuple of (all_losses, all_values, all_pseqs).
        """
        # define metapredict parameters if needed
        if metapredict_constraint:
            disorder_loss_multiplier, min_disorders = self.get_metapredict_function(num_iterations, sequence_length)
        else:
            disorder_loss_multiplier = None
            min_disorders = None

        if self.rpa_check:
            all_pHs = [condition[-1] for condition in all_conditions]
            aa_interactions = predictor.model.set_aa_interactions(sequence_length, sequence_length, all_pHs)
        else:
            aa_interactions = predictor.aa_interactions.astype(jnp.float32)

        
            

        def loss_fn(params: dict, temperature: float, disorder_loss_fn: callable, min_disorder: float = 0.2) -> float:
            logits = params['logits']
            logits1 = logits[:sequence_length]
            logits2 = logits[sequence_length:]
            logits1 = logits1.at[:, 20].set(0.0) #to remove the dummy X residue
            logits1 = logits1.at[:, 21].set(0.0) #to remove dummy Z residue (charged Ser)
            logits1 = logits1.at[:, 22].set(0.0) #to remove dummy B residue (charged Tyr)
            pseq1 = self.normalize_logits(logits1, temperature)
            logits2 = logits2.at[:, 20].set(0.0) #to remove the dummy X residue
            logits2 = logits2.at[:, 21].set(0.0) #to remove dummy Z residue (charged Ser)
            logits2 = logits2.at[:, 22].set(0.0) #to remove dummy B residue (charged Tyr)
            pseq2 = self.normalize_logits(logits2, temperature)


            out1 = self.convert_to_normalized_chi(predictor.predict_pseqs(pseq1, pseq1, aa_interactions), sequence_length)[0]
            out2 = self.convert_to_normalized_chi(predictor.predict_pseqs(pseq2, pseq2, aa_interactions), sequence_length)[0]
            out3 = self.convert_to_normalized_chi(predictor.predict_pseqs(pseq1, pseq2, aa_interactions), sequence_length)[0]

            all_outs = jnp.array([out1, out2, 0.5 - out3])

            if behavior == 'condensed_mix':
                threshold_condense = -0.15
                penalty1 = jnp.maximum(0.0, out1 - threshold_condense)
                penalty2 = jnp.maximum(0.0, out2 - threshold_condense)
                penalty3 = jnp.maximum(0.0, out3 - threshold_condense)
                # loss = penalty1 + penalty2 + penalty3
            else: # behavior == 'condensed_demix'
                threshold_condense = -0.2
                threshold_demix = 0.15
                penalty1 = jnp.maximum(0.0, out1 - threshold_condense)
                penalty2 = jnp.maximum(0.0, out2 - threshold_condense)
                penalty3 = jnp.maximum(0.0, (out1 - out3) + threshold_demix) + jnp.maximum(0, (out2 - out3) + threshold_demix)
                # loss = penalty1 + penalty2 + penalty

            if metapredict_constraint:
                disorder_scalar1, avg_disorder1 = disorder_loss_fn(pseq1[:, :20], min_disorder = min_disorder)
                disorder_scalar2, avg_disorder2 = disorder_loss_fn(pseq2[:, :20], min_disorder = min_disorder)

                penalty_disorder1 = jnp.maximum(0, min_disorder - avg_disorder1)
                penalty_disorder2 = jnp.maximum(0, min_disorder - avg_disorder2)
                loss = jnp.maximum(0, penalty1 + penalty2 + penalty3 + penalty_disorder1 + penalty_disorder2)

                return loss, (all_outs, jnp.concatenate([pseq1, pseq2]), [avg_disorder1, avg_disorder2], [disorder_scalar1, disorder_scalar2])

            return loss, (all_outs, jnp.concatenate([pseq1, pseq2]))

        init_logits = onp.random.normal(len(RES_ALPHA), 1.0, (2*sequence_length, len(RES_ALPHA)))
        init_logits = jnp.array(init_logits)
        init_logits = init_logits.at[:, 20].set(0.0) #to remove the dummy X residue
        init_logits = init_logits.at[:, 21].set(0.0) #to remove dummy Z residue (charged Ser)
        init_logits = init_logits.at[:, 22].set(0.0) #to remove dummy B residue (charged Tyr)

        argmax_seq, aux = self.optimization_loop(predictor, loss_fn, sequence_length = sequence_length, num_iterations = num_iterations, disorder_loss_fn = disorder_loss_multiplier, metapredict_constraint = metapredict_constraint, min_disorders = min_disorders, init_logits = init_logits, **kwargs)

        seq1, seq2 = argmax_seq[:sequence_length], argmax_seq[sequence_length:]

        return seq1, seq2, aux
        
    def optimization_loop(self, 
    predictor: Predictor, 
    loss_fn: callable, 
    sequence_length: int,
    num_iterations: int = 1000, 
    disorder_loss_fn: Optional[callable] = None, 
    metapredict_constraint: bool = True, 
    min_disorders: Optional[jnp.ndarray] = None, 
    init_logits: Optional[jnp.ndarray] = None,
    **kwargs):
        """Run the gradient-descent optimization loop shared by all design types.

        On GPU (and non-finches models), compiles the entire iteration loop
        into a single XLA dispatch via jax.lax.scan to eliminate per-iteration
        Python overhead. On CPU or for the finches model, falls back to a
        Python for-loop with tqdm progress reporting.

        Uses Gumbel-softmax temperature annealing from 1.0 to 0.001 and
        the Adam optimizer with lr=0.1.

        Args:
            predictor: Initialized Predictor instance.
            loss_fn: Differentiable loss function with signature
                (params, temperature, [disorder_loss_fn, min_disorder]) ->
                (loss, aux_tuple).
            sequence_length: Length of the designed sequence.
            num_iterations: Number of Adam steps to run.
            disorder_loss_fn: Callable for the disorder constraint, or None.
            metapredict_constraint: Whether the loss_fn uses disorder scaling.
            min_disorders: Per-iteration disorder thresholds of shape
                (num_iterations,), or None.
            init_logits: Optional initial logits array. If None, sampled
                from a normal distribution with shape (sequence_length, 23).
            **kwargs: Unused.

        Returns:
            argmax_seq: Final optimized amino acid sequence string.
            aux: Tuple of (all_losses, all_values, all_pseqs).
        """
        lr = 0.1
        gumbel_start = 1.0
        gumbel_end = 0.001
        gumbel_temps = jnp.linspace(gumbel_start, gumbel_end, num_iterations, dtype=jnp.float32)

        if init_logits is None:
            init_logits = jnp.array(
                onp.random.normal(len(RES_ALPHA), 1.0, (sequence_length, len(RES_ALPHA))),
                dtype=jnp.float32,
            )
            init_logits = init_logits.at[:, 20].set(0.0)
            init_logits = init_logits.at[:, 21].set(0.0)
            init_logits = init_logits.at[:, 22].set(0.0)
        else:
            init_logits = jnp.array(init_logits, dtype=jnp.float32)

        params = {'logits': init_logits}
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(params)

        backend = xla_bridge.get_backend().platform
        use_scan = (backend == 'gpu') and (self.model_name != 'finches')

        if use_scan:
            # --- GPU path: compile full loop into one dispatch via jax.lax.scan ---
            if min_disorders is not None:
                min_disorders_jax = jnp.asarray(min_disorders, dtype=jnp.float32)
            else:
                min_disorders_jax = jnp.zeros(num_iterations, dtype=jnp.float32)

            if metapredict_constraint:
                def step_loss(params, temperature, min_disorder):
                    return loss_fn(params, temperature, disorder_loss_fn, min_disorder)
            else:
                def step_loss(params, temperature, min_disorder):
                    return loss_fn(params, temperature)

            grad_fn = jax.value_and_grad(step_loss, has_aux=True)

            def scan_body(carry, xs):
                params, opt_state = carry
                temperature, min_disorder = xs
                (loss, aux), grads = grad_fn(params, temperature, min_disorder)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), (loss, aux[0], aux[1])

            def run_opt(carry, xs):
                return jax.lax.scan(scan_body, carry, xs)

            print(f"Compiling and running {num_iterations} iterations...")
            (final_params, _), (all_losses, all_values_raw, all_pseqs) = jax.jit(run_opt)(
                (params, opt_state), (gumbel_temps, min_disorders_jax)
            )

            all_values = self.convert_to_chi(all_values_raw)

        else:
            # --- CPU / finches path: Python loop with tqdm progress ---
            if self.model_name == 'finches':
                grad_loss_fn = jax.value_and_grad(loss_fn, has_aux=True)
            else:
                grad_loss_fn = jax.jit(
                    jax.value_and_grad(loss_fn, has_aux=True),
                    static_argnums=(2,)
                )

            all_losses = list()
            all_pseqs = list()
            all_values = list()
            lot_every = num_iterations // 1

            for i in tqdm(range(num_iterations), desc='Iteration'):
                if metapredict_constraint:
                    (loss, aux), grads = grad_loss_fn(params, gumbel_temps[i], disorder_loss_fn, min_disorders[i])
                else:
                    (loss, aux), grads = grad_loss_fn(params, gumbel_temps[i])

                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                all_losses.append(loss)
                all_values.append(self.convert_to_chi(aux[0]))
                all_pseqs.append(aux[1])

                if (i + 1) % lot_every == 0:
                    argmax_seq = ''.join([RES_ALPHA[j] for j in jnp.argmax(all_pseqs[-1], axis=1)])
                    print(f'Iteration {i+1} loss: {float(loss)}')
                    print(f'Iteration {i+1} value: {all_values[-1]}')
                    print(f'Iteration {i+1} sequence: {argmax_seq}')

        argmax_seq = ''.join([RES_ALPHA[i] for i in jnp.argmax(
            all_pseqs[-1] if isinstance(all_pseqs, list) else all_pseqs[-1],
            axis=1
        )])
        print('Optimization complete!')
        print(f'Final sequence: {argmax_seq}')
        print(f'Final loss: {float(all_losses[-1]) if isinstance(all_losses, list) else float(all_losses[-1])}')

        return argmax_seq, (all_losses, all_values, all_pseqs)



    def normalize_logits(self, logits: jnp.ndarray, temperature: float = 0.01) -> jnp.ndarray:
        """Convert raw logits to a probabilistic sequence via temperature-scaled softmax.

        Args:
            logits: Raw logit array of shape (L, len(RES_ALPHA)).
            temperature: Softmax temperature. Lower values produce sharper
                (more one-hot-like) distributions.

        Returns:
            Probabilistic sequence array of same shape, rows summing to 1.
        """
        return jax.nn.softmax(logits / temperature, axis = -1)