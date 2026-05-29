from typing import Sequence, Tuple, Optional, List
import jax
import jax.numpy as jnp
import itertools
from tqdm import tqdm

from models.prediction_model import Dimer_model, RPA_model, FINCHES_model
from models.ff_models import seq_to_one_hot, MPIPI_model, HPS_model, Calvados_model, KHD_model
from models.ff_models import RES_ALPHA

MODEL_REGISTRY = {
    "dimer": Dimer_model,
    "rpa": RPA_model,
    "finches": FINCHES_model,
}

FF_REGISTRY = ['mpipi', 'mpipi_gg', 'hps_dignon', 'hps_tesei', 'hps_urry', 'hps_fb', 'kh_d', 'calvados1', 'calvados2']

def generate_all_conditions(temps: List[float], salts: List[float], phs: List[float]) -> List[Tuple[float, float, float]]:
    """Return the Cartesian product of temperature, salt, and pH conditions.

    Args:
        temps: Temperatures in Kelvin.
        salts: Salt concentrations in mM.
        phs: pH values.

    Returns:
        List of (temp, salt, pH) tuples.
    """
    return list(itertools.product(temps, salts, phs))

class Predictor:

    def __init__(self, model_name: str, ff_type: str, temps: Optional[List[float | int]] = [300], salts: Optional[List[float | int]] = [150], pHs: Optional[List[float | int]] = [7.4], pH_mode: str = 'normal', **model_kwargs):
        """Initialize a Predictor wrapping a physics model and force field.

        Instantiates the chosen physics model (Dimer, RPA, or FINCHES) with
        the specified force field and precomputes amino acid interaction
        parameters for all condition combinations.

        Args:
            model_name: Physics model key ('dimer', 'rpa', or 'finches').
            ff_type: Force field identifier (e.g. 'mpipi', 'calvados2').
            temps: Temperatures in Kelvin.
            salts: Salt concentrations in mM.
            pHs: pH values.
            pH_mode: 'normal' (fixed charges) or 'pH' (Henderson-Hasselbalch).
            **model_kwargs: For RPA: dict of rho0, lB, kappa, a, Vh0.
                For FINCHES: use_charge, use_aliphatic flags.
        """
        key = model_name.lower()
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Options: {list(MODEL_REGISTRY)}")
        key_ff = ff_type.lower()
        if key_ff not in FF_REGISTRY:
            raise ValueError(f"Unknown force field '{ff_type}'. Options: {list(FF_REGISTRY)}")
        
        self.rpa_check = True if model_name == 'rpa' else False


        # MUST have some definition of rho0, lb, kappa, a, Vh0 for RPA model
        # print(model_kwargs)
        if self.rpa_check and len(model_kwargs) > 0:
            rpa_params = model_kwargs
        elif self.rpa_check and len(model_kwargs) == 0: 
            rpa_params = {
                'rho0': 1.0,
                'lB': 1.7,
                'kappa': 0.75,
                'a': 0.1,
                'Vh0': 3.0
            }
                
        if self.rpa_check:
            self.model = MODEL_REGISTRY[key](ff_type = ff_type, pH_mode = pH_mode, params = rpa_params)
        else: # model_kwargs only useful outside of RPA for finches model - can pass in use_charge, use_aliphatic
            self.model = MODEL_REGISTRY[key](ff_type = ff_type, pH_mode = pH_mode, **model_kwargs)

        self.model_name = key
        self.ff_type = key_ff
        self.pH_mode = pH_mode
        self.pHs = pHs

        self.all_conditions = generate_all_conditions(temps, salts, pHs)
        # for models other than RPA, precompute interaction parameters for all conditions
        if self.rpa_check == False:
            self.aa_interactions = self.model.set_aa_interactions(self.all_conditions)

    def predict_pair(self, seq1: str, seq2: str) -> float:
        """Predict interaction parameters for a single pair of sequences.

        Args:
            seq1: First amino acid sequence string.
            seq2: Second amino acid sequence string.

        Returns:
            Array of interaction parameter values, one per condition.
        """
        if self.rpa_check:
            N1, N2 = len(seq1), len(seq2)
            aa_interactions = self.model.set_aa_interactions(N1, N2, self.pHs)
        else:
            aa_interactions = self.aa_interactions

        out = self.model.return_interaction_parameters(seq1, seq2, aa_interactions)
        return out

    def generate_pseqs(self, seqs1: List[str], seqs2: List[str], same_len: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Convert lists of sequence strings to stacked one-hot arrays.

        If same_len is True, sequences are directly stacked. Otherwise
        they are zero-padded to the maximum length across both lists.

        Args:
            seqs1: First list of amino acid sequence strings.
            seqs2: Second list of amino acid sequence strings.
            same_len: Whether all sequences in each list share the same length.

        Returns:
            Tuple of (pseqs1, pseqs2), each of shape (batch, max_len, 23).
        """
        def to_one_hot(seq):
            return seq_to_one_hot(seq)  # shape (len(seq), 23)

        if same_len:
            pseqs1 = jnp.stack([to_one_hot(s) for s in seqs1], axis=0)
            pseqs2 = jnp.stack([to_one_hot(s) for s in seqs2], axis=0)
            return pseqs1, pseqs2

        max_len = max(max(len(s) for s in seqs1), max(len(s) for s in seqs2))

        def pad_to_max(arr):
            pad_len = max_len - arr.shape[0]
            padding = jnp.zeros((pad_len, len(RES_ALPHA)), dtype=arr.dtype)
            return jnp.concatenate([arr, padding], axis=0)

        pseqs1 = jnp.stack([pad_to_max(to_one_hot(s)) for s in seqs1], axis=0)
        pseqs2 = jnp.stack([pad_to_max(to_one_hot(s)) for s in seqs2], axis=0)
        
        return pseqs1, pseqs2

    def predict_batch(
        self,
        seqs1: List[str],
        seqs2: Optional[List[str]] = None,
        print_check: bool = False,
    ) -> List[float]:
        """Predict interaction parameters for a batch of sequence pairs.

        Uses vmap-based parallelism when the model supports continuous
        (differentiable) predictions. Falls back to a sequential loop
        for models like FINCHES that don't support JIT/vmap.

        Args:
            seqs1: List of first sequences in each pair.
            seqs2: List of second sequences. If None, uses seqs1 (self-interactions).
            print_check: If True, print per-pair results.

        Returns:
            Array or list of interaction parameters, shape (batch, N_conditions).

        Raises:
            ValueError: If seqs1 and seqs2 have different lengths.
        """
        if seqs2 is None:
            seqs2 = seqs1
        if len(seqs1) != len(seqs2):
            raise ValueError("seqs1 and seqs2 must have the same length")

        same_len = len({len(s) for s in seqs1}) == 1 and len({len(s) for s in seqs2}) == 1
        supports_cont = hasattr(self.model, "return_interaction_parameters_continuous")

        pseqs1, pseqs2 = self.generate_pseqs(seqs1, seqs2, same_len)
        # aa_interactions = self.aa_interactions

        if same_len:
            seq_len = len(seqs1[0])

        if self.rpa_check:
            aa_interactions = self.model.set_aa_interactions(seq_len, seq_len, self.pHs)
        else:
            aa_interactions = self.aa_interactions

        if supports_cont:
            vmapped = jax.vmap(
                lambda a, b, aa: self.model.return_interaction_parameters_continuous(a, b, aa),
                in_axes=(0, 0, None),
            )

            outs = vmapped(pseqs1, pseqs2, aa_interactions)

            out = outs[0] if len(outs) == 1 else outs

            if print_check:
                print("Batch predictions:")
                for j, seq_pair in enumerate(zip(seqs1, seqs2)):
                    seq1, seq2 = seq_pair
                    print("-" * 25)
                    print(f"Sequence Pair {j}: {seq1} vs {seq2}")
                    if not self.rpa_check:
                        for i, cond in enumerate(self.all_conditions):
                            temp, salt, pH = cond
                            print("-" * 25)
                            print(f"Temperature: {temp} K, Salt concentration: {salt} mM, pH: {pH} -> {out[j][i]}")
                    else:
                        for i, pH in enumerate(self.pHs):
                            print("-" * 25)
                            print(f"pH: {pH} -> {out[j][i]}")

            return out

        

        aa_interactions = self.aa_interactions

        out = [
            self.model.return_interaction_parameters_pseqs(pseq1, pseq2, aa_interactions) 
            for pseq1, pseq2 in tqdm(zip(pseqs1, pseqs2), total=len(pseqs1), desc='Predicting batch')
        ]
        
        # out = jnp.concatenate(results, axis=0)
        if print_check:
            print("Batch predictions:")
            for j, seq_pair in enumerate(zip(seqs1, seqs2)):
                seq1, seq2 = seq_pair
                print("-" * 50)
                print(f"Sequence Pair {j}: {seq1} vs {seq2}")
                for i, cond in enumerate(self.all_conditions):
                    temp, salt, pH = cond
                    print("-" * 25)
                    print(f"Temperature: {temp} K, Salt concentration: {salt} mM, pH: {pH} -> {out[j][i]}")
        return out
 
        # fallback: loop over sequence pairs one-by-one
        # note - this is the only way to get predictions for FINCHES model since it doesn't support vmapping/JIT-ing
        aa_interactions = self.aa_interactions
        out = [self.model.return_interaction_parameters_pseqs(seq_to_one_hot(seq1), seq_to_one_hot(seq2), aa_interactions) for seq1, seq2 in zip(seqs1, seqs2)]
        if print_check:
            print('Batch predictions:')
            for j, seq_pair in enumerate(zip(seqs1, seqs2)):
                seq1, seq2 = seq_pair
                print('-'*50)
                print(f'Sequence Pair {j}: {seq1} vs {seq2}')
                for i, cond in enumerate(self.all_conditions):
                    temp, salt, pH = cond
                    print('-'*25)
                    print(f'Temperature: {temp} K, Salt concentration: {salt} mM, pH: {pH} -> {out[j][i]}')
        return out

    def predict_pseqs(self, pseq1: jnp.ndarray, pseq2: jnp.ndarray, res_diffs: Optional[jnp.ndarray] = None) -> List[float]:
        """Predict interaction parameters from probabilistic sequence arrays.

        Differentiable path used during gradient-based sequence optimization.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).
            res_diffs: Precomputed AA interaction data. For RPA, a list of
                [res_diffs, all_charges]. For dimer/FINCHES, the interaction
                tensor. If None, computed on the fly (RPA only).

        Returns:
            Array of interaction parameters, shape (N_conditions,).
        """
        if self.rpa_check:
            N1, N2 = pseq1.shape[0], pseq2.shape[0]
            if res_diffs is None:
                res_diffs = self.model.set_aa_interactions(N1, N2, self.pHs)
            aa_interactions = res_diffs
        else:
            aa_interactions = self.aa_interactions

        # print(len(aa_interactions))

        if self.model_name == 'finches':
            out = self.model.return_interaction_parameters_pseqs(pseq1, pseq2, aa_interactions)
        else:
            out = self.model.return_interaction_parameters_continuous(pseq1, pseq2, aa_interactions)

        return out # should be shape (N_conditions,)


    def predict_interaction_matrix_pseqs(self, pseq1: jnp.ndarray, pseq2: jnp.ndarray, res_diffs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Predict the full residue-residue interaction matrix from probabilistic sequences.

        Differentiable path for optimization of spatially patterned interactions.
        Currently supported for the dimer model only.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).
            res_diffs: Precomputed AA interaction data, or None.

        Returns:
            Interaction matrix of shape (L1, L2).
        """
        if self.rpa_check:
            N1, N2 = pseq1.shape[0], pseq2.shape[0]
            if res_diffs is None:
                res_diffs = self.model.set_aa_interactions(N1, N2, self.pHs)
            aa_interactions = res_diffs
        else:
            aa_interactions = self.aa_interactions
        return self.model.return_interaction_matrix_continuous(pseq1, pseq2, aa_interactions)

