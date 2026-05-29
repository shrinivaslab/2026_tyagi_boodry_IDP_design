import sys
import numpy as onp
import random
import itertools
import unittest
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import optax
import pandas as pd
import jax

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, lax
import functools
import os
import importlib
jax.config.update("jax_enable_x64", False)
from jax.lib import xla_bridge
# print(xla_bridge.get_backend().platform)

import models.ff_models as ff_models
importlib.reload(ff_models)
from models.ff_models import MPIPI_model, HPS_model, Calvados_model, KHD_model
from models.ff_models import seq_to_one_hot

sys.path.append(str(Path(__file__).resolve().parent))
kb = 0.0019872041
RES_ALPHA = "MGKTRADEYVLQWFSHNPCIXZB"
NUM_RESIDUES = len(RES_ALPHA)
assert(NUM_RESIDUES == 23)




class Dimer_model:
    """Second-virial-coefficient model for IDP interaction parameters.

    Computes pairwise dimer-dimer interaction energies by numerically
    integrating the Mayer f-function over radial distance, then contracts
    the resulting 529x529 interaction tensor with dimer frequency vectors
    derived from probabilistic sequences.
    """

    def __init__(self, ff_type, pH_mode = 'normal'):
        """Initialize the dimer model with a specific force field.

        Args:
            ff_type: Force field identifier (e.g. 'mpipi', 'calvados2').
            pH_mode: 'normal' or 'pH' for charge calculation.
        """
        self.ff_type = ff_type
        self.pH_mode = pH_mode
        if ff_type == 'mpipi_gg':
            self.model = MPIPI_model(version = 'GG')
        elif ff_type == 'mpipi':
            self.model = MPIPI_model(version = 'normal')
        elif ff_type == 'hps_dignon':
            self.model = HPS_model(version = 'Dignon')
        elif ff_type == 'hps_tesei':
            self.model = HPS_model(version = 'Tesei')
        elif ff_type == 'hps_urry':
            self.model = HPS_model(version = 'Urry')
        elif ff_type == 'hps_fb':
            self.model = HPS_model(version = 'FB')
        elif ff_type == 'calvados1':
            self.model = Calvados_model(version = '1')
        elif ff_type == 'calvados2':
            self.model = Calvados_model(version = '2')
        elif ff_type == 'kh-d':
            self.model = KHD_model()
        else:
            raise ValueError(f"Invalid ff type: {ff_type}, choose from mpipi, mpipi_gg, hps_dignon, hps_tesei, hps_urry, hps_fb, calvados1, calvados2, kh-d")

    @functools.partial(jit, static_argnums=(0,))
    def calculate_aa_interactions(self, r, dr, kT, temp, salt_conc, pair_charges):
        """Compute the 23^2 x 23^2 dimer-dimer interaction energy tensor.

        For each quadruplet of residue types (i,j,k,l), numerically integrates
        r^2 * (1 - exp(-alpha * sum_of_unbonded / kT)) over the radial grid
        to obtain the Mayer f-function integral.

        Args:
            r: Radial distance grid, shape (num_steps,).
            dr: Grid spacing.
            kT: Thermal energy (kb * temp).
            temp: Temperature in Kelvin.
            salt_conc: Salt concentration in mM.
            pair_charges: Charge pairs, shape (529, 2).

        Returns:
            Interaction tensor of shape (23, 23, 23, 23).
        """
        alpha = 0.75
        num_steps = r.shape[0]
        unbonded_values = vmap(lambda r: self.model.calculate_unbonded(r, pair_charges, salt_conc, temp))(r)
        unbonded_values = unbonded_values.reshape(num_steps, NUM_RESIDUES, NUM_RESIDUES) # to include temp and  salt_conc
        def compute_interaction_sum(i, j, k, l):
            interaction_sum = 0
            for x, y in [(i, k), (i, l), (j, k), (j, l)]:
                interaction_sum += unbonded_values[:, x, y]
            pairs = 1 - jnp.exp(-alpha * (interaction_sum / kT))
            y_vals = (r**2) * pairs
            integrals = jnp.sum((y_vals[:-1] + y_vals[1:]) * dr / 2, axis=0)
            return integrals
        compute_interactions_vmap = vmap(
            vmap(
                vmap(
                    vmap(compute_interaction_sum, in_axes=(None, None, None, 0)),
                    in_axes=(None, None, 0, None)
                ),
                in_axes=(None, 0, None, None)
            ),
            in_axes=(0, None, None, None)
        )
        indices = jnp.arange(NUM_RESIDUES)
        interactions = compute_interactions_vmap(indices, indices, indices, indices)

        return interactions

    @functools.partial(jit, static_argnums=(0,))
    def pseq_to_dseq(self, pseq):
        """Convert a probabilistic sequence to dimer probability matrices.

        Computes the outer product of adjacent residue distributions to
        obtain the probability of each dimer type at each position.

        Args:
            pseq: Probabilistic sequence, shape (L, 23).

        Returns:
            Dimer probabilities, shape (L-1, 23, 23).
        """
        num_dimers = pseq.shape[0] - 1
        dimer_prob = jnp.zeros((num_dimers, NUM_RESIDUES, NUM_RESIDUES))
        def compute_dimer(i):
            return jnp.outer(pseq[i], pseq[i + 1])
        dimer_prob = vmap(compute_dimer)(jnp.arange(num_dimers))
        return dimer_prob

    @functools.partial(jit, static_argnums=(0,))
    def dimer_virial_coefficients_internal(self, dseq1, dseq2, aa_interactions):
        """Compute the second virial coefficient from dimer frequency vectors.

        Contracts the dimer frequency distributions of two sequences with
        the precomputed 529x529 interaction tensor to yield a scalar virial
        coefficient.

        Args:
            dseq1: Dimer probabilities for sequence 1, shape (N1-1, 23, 23).
            dseq2: Dimer probabilities for sequence 2, shape (N2-1, 23, 23).
            aa_interactions: Interaction tensor, shape (529, 529).

        Returns:
            Scalar virial coefficient B2.
        """
        freq_dseq1 = jnp.sum(dseq1, axis=0) # (23, 23)
        freq_dseq2 = jnp.sum(dseq2, axis=0) # (23, 23)
        freq_dseq1_flat = freq_dseq1.flatten() # (529)
        freq_dseq2_flat = freq_dseq2.flatten() # (529)
        intm_result = jnp.dot(freq_dseq1_flat, aa_interactions) # dot product of (529) and (529, 529) -> (529)
        virialB = jnp.float32(2 * jnp.pi / 1e3) * jnp.dot(intm_result, freq_dseq2_flat) # dot product of (529) and (529) -> (1)
        return virialB.squeeze()

    @functools.partial(jit, static_argnums=(0,))
    def pairwise_interaction_matrix_internal(self, dseq1, dseq2, aa_interactions):
        """
        dseq1: (N1-1, 23, 23) dimer probabilities for seq1
        dseq2: (N2-1, 23, 23) dimer probabilities for seq2
        aa_interactions: (529, 529) dimer–dimer interaction energies
        Returns: (N1, N2) residue–residue interaction matrix
        """
        n1 = dseq1.shape[0]          # number of dimers in seq1
        n2 = dseq2.shape[0]          # number of dimers in seq2

        d1 = dseq1.reshape(n1, NUM_RESIDUES**2)  # (n1, 529)
        d2 = dseq2.reshape(n2, NUM_RESIDUES**2)  # (n2, 529)

        # dimer–dimer energy block
        dim_mat = d1 @ aa_interactions @ d2.T    # (1, n1, n2)

        dim_mat = dim_mat.squeeze() # (n1, n2)

        # distribute each dimer–dimer energy equally to its four endpoint residue pairs
        contrib = 0.25 * dim_mat
        res_mat = jnp.zeros((n1 + 1, n2 + 1))
        res_mat = res_mat.at[:-1, :-1].add(contrib)
        res_mat = res_mat.at[1:,  :-1].add(contrib)
        res_mat = res_mat.at[:-1, 1: ].add(contrib)
        res_mat = res_mat.at[1:,  1: ].add(contrib)

        # match virial scaling
        res_mat = res_mat * (2 * jnp.pi / 1e3)

        return res_mat

    @functools.partial(jit, static_argnums=(0,))
    def set_aa_interactions(self, all_conditions, r0 = 0.01, r1 = 100.0, num_steps = 100):
        """Precompute the 529x529 interaction tensor for each thermodynamic condition.

        Args:
            all_conditions: List of (temp, salt, pH) tuples.
            r0: Radial integration lower bound (Angstroms).
            r1: Radial integration upper bound (Angstroms).
            num_steps: Number of radial grid points.

        Returns:
            Array of shape (N_conditions, 529, 529).
        """
        r = jnp.linspace(r0, r1, num_steps)
        dr = (r1 - r0)/(num_steps-1)
        all_virials = []
        for conditions in all_conditions:
            temp, salt, pH = conditions
            charges = self.model.calculate_charges(pH = pH, pH_mode = self.pH_mode)
            pair_charges = self.model.create_pair_charges(charges)
            aa_interactions = self.calculate_aa_interactions(r, dr, kb * temp, temp, salt, pair_charges)
            aa_interactions = aa_interactions.reshape(NUM_RESIDUES**2, NUM_RESIDUES**2)
            all_virials.append(aa_interactions)
        all_virials = jnp.array(all_virials)
        
        return all_virials

    def return_interaction_matrix(self, seq1, seq2, aa_interactions):
        """Compute the residue-residue interaction matrix from sequence strings.

        Args:
            seq1: First amino acid sequence string.
            seq2: Second amino acid sequence string.
            aa_interactions: Precomputed interaction tensor(s).

        Returns:
            Interaction matrix of shape (len(seq1), len(seq2)).
        """
        pseq1 = seq_to_one_hot(seq1)
        pseq2 = seq_to_one_hot(seq2)
        dseq1 = self.pseq_to_dseq(pseq1)
        dseq2 = self.pseq_to_dseq(pseq2)
        return self.pairwise_interaction_matrix_internal(dseq1, dseq2, aa_interactions)

    def return_interaction_parameters(self, seq1, seq2, aa_interactions):
        """Compute virial coefficients from sequence strings.

        Args:
            seq1: First amino acid sequence string.
            seq2: Second amino acid sequence string.
            aa_interactions: Precomputed interaction tensors, shape (N_conditions, 529, 529).

        Returns:
            Array of virial coefficients, shape (N_conditions,).
        """
        pseq1 = seq_to_one_hot(seq1)
        pseq2 = seq_to_one_hot(seq2)
        dseq1 = self.pseq_to_dseq(pseq1)
        dseq2 = self.pseq_to_dseq(pseq2)

        all_virials = []
        for aa_interaction in aa_interactions:
            virialB = self.dimer_virial_coefficients_internal(dseq1, dseq2, aa_interaction)
            all_virials.append(virialB)
        all_virials = jnp.array(all_virials)
        return all_virials

    @functools.partial(jit, static_argnums=(0,))
    def return_interaction_parameters_continuous(self, pseq1, pseq2, aa_interactions):
        """Compute virial coefficients from probabilistic sequences (differentiable).

        JIT-compiled path used during gradient-based optimization.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).
            aa_interactions: Precomputed interaction tensors, shape (N_conditions, 529, 529).

        Returns:
            Array of virial coefficients, shape (N_conditions,).
        """
        dseq1 = self.pseq_to_dseq(pseq1)
        dseq2 = self.pseq_to_dseq(pseq2)
        all_virials = []
        for aa_interaction in aa_interactions:
            virialB = self.dimer_virial_coefficients_internal(dseq1, dseq2, aa_interaction)
            all_virials.append(virialB)
        all_virials = jnp.array(all_virials)
        return all_virials

    @functools.partial(jit, static_argnums=(0,))
    def return_interaction_matrix_continuous(self, pseq1, pseq2, aa_interactions):
        """Compute the residue-residue interaction matrix from probabilistic sequences.

        JIT-compiled, differentiable path for optimization.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).
            aa_interactions: Precomputed interaction tensors.

        Returns:
            Interaction matrix of shape (L1, L2).
        """
        dseq1 = self.pseq_to_dseq(pseq1)
        dseq2 = self.pseq_to_dseq(pseq2)
        return self.pairwise_interaction_matrix_internal(dseq1, dseq2, aa_interactions)

    
class RPA_model:
    """Random Phase Approximation model for effective chi parameters.

    Combines mean-field (MFT), fluctuation (RPA), and short-range
    hydrophobic (chi_h) contributions to compute an effective Flory
    chi parameter for IDP pairs.
    """

    def __init__(self, ff_type, pH_mode = 'normal', params = None):
        """Initialize the RPA model with a force field and physical parameters.

        Args:
            ff_type: Force field identifier.
            pH_mode: 'normal' or 'pH' for charge calculation.
            params: Dict with RPA parameters: rho0, lB, kappa, a, Vh0.
        """
        self.ff_type = ff_type
        self.pH_mode = pH_mode
        if ff_type == 'mpipi_gg':
            self.model = MPIPI_model(version = 'GG')
        elif ff_type == 'mpipi':
            self.model = MPIPI_model(version = 'normal')
        elif ff_type == 'hps_dignon':
            self.model = HPS_model(version = 'Dignon')
        elif ff_type == 'hps_tesei':
            self.model = HPS_model(version = 'Tesei')
        elif ff_type == 'hps_urry':
            self.model = HPS_model(version = 'Urry')
        elif ff_type == 'hps_fb':
            self.model = HPS_model(version = 'FB')
        elif ff_type == 'calvados1':
            self.model = Calvados_model(version = '1')
        elif ff_type == 'calvados2':
            self.model = Calvados_model(version = '2')
        elif ff_type == 'kh-d':
            self.model = KHD_model()
        else:
            raise ValueError(f"Invalid ff type: {ff_type}, choose from mpipi, mpipi_gg, hps_dignon, hps_tesei, hps_urry, hps_fb, calvados1, calvados2, kh-d")

        self.eps_matrix = self.model.construct_eps_matrix()
        # self.charges = self.model.calculate_charges(pH_mode = self.pH_mode)
        self.params = params

    def return_interaction_parameters(self, seq1, seq2, aa_interactions):
        """Compute effective chi from sequence strings.

        Args:
            seq1: First amino acid sequence string.
            seq2: Second amino acid sequence string.
            aa_interactions: Precomputed [res_diffs, all_charges] from set_aa_interactions.

        Returns:
            Array of effective chi values, one per pH condition.
        """
        pseq1 = seq_to_one_hot(seq1)
        pseq2 = seq_to_one_hot(seq2)
        self.N1 = pseq1.shape[0]
        self.N2 = pseq2.shape[0]

        rho0 = self.params['rho0']
        lB = self.params['lB']
        kappa = self.params['kappa']
        a = self.params['a']
        Vh0 = self.params['Vh0']

        chi_eff = self.calc_chi_eff(pseq1, pseq2, rho0, lB, kappa, a, Vh0, aa_interactions)
        return chi_eff

    def set_aa_interactions(self, N1, N2, pHs):
        """Precompute residue separation matrices and per-pH charge vectors.

        Args:
            N1: Length of sequence 1.
            N2: Length of sequence 2.
            pHs: List of pH values.

        Returns:
            List of [res_diffs_array, all_charges_array].
        """
        indices_1 = jnp.arange(N1)
        indices_2 = jnp.arange(N2)
        res_diffs_1 = jnp.abs(indices_1[:, None] - indices_1[None, :])
        res_diffs_2 = jnp.abs(indices_2[:, None] - indices_2[None, :])
        res_diffs = [res_diffs_1, res_diffs_2]

        all_charges = jnp.array([self.model.calculate_charges(pH = pH, pH_mode = self.pH_mode) for pH in pHs])

        return [jnp.array(res_diffs), all_charges] 

    @functools.partial(jit, static_argnums=(0,))
    def return_interaction_parameters_continuous(self, pseq1, pseq2, aa_interactions):
        """Compute effective chi from probabilistic sequences (differentiable).

        Requires set_aa_interactions to have been called first to provide
        the res_diffs and charge data in aa_interactions.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).
            aa_interactions: [res_diffs, all_charges] from set_aa_interactions.

        Returns:
            Array of effective chi values, one per pH condition.
        """
        rho0 = self.params['rho0']
        lB = self.params['lB']
        kappa = self.params['kappa']
        a = self.params['a']
        Vh0 = self.params['Vh0']

        chi_eff = self.calc_chi_eff(pseq1, pseq2, rho0, lB, kappa, a, Vh0, aa_interactions)
        return chi_eff

    @functools.partial(jit, static_argnums=(0,))
    def get_pseq_charges(self, pseq, charge):
        """Compute total net charge of a probabilistic sequence.

        Args:
            pseq: Probabilistic sequence, shape (L, 23).
            charge: Per-residue-type charges, shape (23,).

        Returns:
            Scalar total charge.
        """
        return jnp.sum(pseq * charge)

    @functools.partial(jit, static_argnums=(0,))
    def get_pseq_charge_locs(self, pseq, charge):
        """Compute per-position charge of a probabilistic sequence.

        Args:
            pseq: Probabilistic sequence, shape (L, 23).
            charge: Per-residue-type charges, shape (23,).

        Returns:
            Per-position charges, shape (L,).
        """
        return jnp.sum(pseq * charge, axis = 1)

    @functools.partial(jit, static_argnums=(0,))
    def calc_chi_eff_MFT(self, pseq1, pseq2, rho0, lB, kappa, charge):
        """Compute the mean-field theory electrostatic contribution to chi.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).
            rho0: Reference density.
            lB: Bjerrum length.
            kappa: Inverse Debye length.
            charge: Per-residue-type charges, shape (23,).

        Returns:
            Scalar mean-field chi contribution.
        """
        N1 = pseq1.shape[0]
        N2 = pseq2.shape[0]
        sig_1 = self.get_pseq_charges(pseq1, charge)
        sig_2 = self.get_pseq_charges(pseq2, charge)
        chi = - 2. * jnp.pi * lB / kappa**2 * sig_1 * sig_2 / N1 / N2
        return chi * rho0

    @functools.partial(jit, static_argnums=(0,))
    def calc_chi_eff_RPA(self, pseq1, pseq2, rho0, lB, kappa, a, res_diffs, charge):
        """Compute the RPA fluctuation correction to chi.

        Integrates the charge structure factor product over wavevector
        space, accounting for chain connectivity through the Gaussian
        connection tensor.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).
            rho0: Reference density.
            lB: Bjerrum length.
            kappa: Inverse Debye length.
            a: Smearing length scale.
            res_diffs: [res_diffs_1, res_diffs_2] separation matrices.
            charge: Per-residue-type charges, shape (23,).

        Returns:
            Scalar RPA chi contribution.
        """
        res_diffs_1 = res_diffs[0]
        res_diffs_2 = res_diffs[1]
        kmax = 2./a
        k = jnp.linspace(1e-15, kmax, int(3e2))
        k2 = k**2
        dk = k[1] - k[0]
        Gamma4 = jnp.exp(-2. * a**2 * k2)
        b2k2_over_6 = k2/6.

        N1 = pseq1.shape[0]
        sig_1 = self.get_pseq_charge_locs(pseq1, charge)
        connection_tensor_1 = jnp.exp(- jnp.einsum('ab,i->abi', res_diffs_1, b2k2_over_6))
        g0_1 = jnp.einsum('a,b,abi->i',sig_1, sig_1, connection_tensor_1) / N1


        N2 = pseq2.shape[0]
        sig_2 = self.get_pseq_charge_locs(pseq2, charge)
        connection_tensor_2 = jnp.exp(- jnp.einsum('ab,i->abi', res_diffs_2, b2k2_over_6))
        g0_2 = jnp.einsum('a,b,abi->i',sig_2, sig_2, connection_tensor_2) / N2
        integrand = k2 / (k2 + kappa**2)**2 * Gamma4 * g0_1 * g0_2
        chi = 2. * jnp.pi * lB**2 * jnp.sum(integrand) * dk
        return chi * rho0

    @functools.partial(jit, static_argnums=(0,))
    def calc_chi_h(self, pseq1, pseq2, Vh0, rho0):
        """Compute the short-range hydrophobic contribution to chi.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).
            Vh0: Hydrophobic interaction strength.
            rho0: Reference density.

        Returns:
            Scalar hydrophobic chi contribution.
        """
        interaction_matrix = pseq1 @ self.eps_matrix @ pseq2.T
        chi = -0.5 * Vh0 * jnp.mean(interaction_matrix)
        return chi * rho0

    @functools.partial(jit, static_argnums=(0,))
    def calc_chi_eff(self, pseq1, pseq2, rho0, lB, kappa, a, Vh0, aa_interactions):
        """Compute the total effective chi as MFT + RPA + hydrophobic.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).
            rho0: Reference density.
            lB: Bjerrum length.
            kappa: Inverse Debye length.
            a: Smearing length scale.
            Vh0: Hydrophobic interaction strength.
            aa_interactions: [res_diffs, all_charges] from set_aa_interactions.

        Returns:
            Array of effective chi values, one per pH condition.
        """
        res_diffs = aa_interactions[0]
        charges = aa_interactions[1]
        all_chi_effs = []
        for charge in charges:
            chi_RPA = self.calc_chi_eff_RPA(pseq1, pseq2, rho0, lB, kappa, a, res_diffs, charge)
            chi_MFT = self.calc_chi_eff_MFT(pseq1, pseq2, rho0, lB, kappa, charge)
            chi_h = self.calc_chi_h(pseq1, pseq2, Vh0, rho0)
            all_chi_effs.append(chi_RPA + chi_MFT + chi_h)
            
        return jnp.array(all_chi_effs)

aliphatic_aas = ['A', 'I', 'L', 'M', 'V']
charged_aas = ['D', 'E', 'K', 'R']

from scipy.optimize import root_scalar
class FINCHES_model:
    """FINCHES-style interaction parameter model.

    Computes pairwise residue interaction energies with optional
    charge-context and aliphatic-cluster weighting corrections, then
    decomposes the interaction matrix into attractive/repulsive
    contributions relative to a null baseline.
    """

    def __init__(self, ff_type, pH_mode = 'normal', use_charge = True, use_aliphatic = True):
        """Initialize the FINCHES model.

        Args:
            ff_type: Force field identifier.
            pH_mode: 'normal' or 'pH' for charge calculation.
            use_charge: If True, apply charge-context interaction weighting.
            use_aliphatic: If True, apply aliphatic cluster weighting.
        """
        self.ff_type = ff_type
        self.pH_mode = pH_mode
        self.use_charge = use_charge
        self.use_aliphatic = use_aliphatic
        self.unit_conv = 1.0

        self.charge_prefactor = None
        self.null_interaction_baseline = None
        if ff_type == 'mpipi_gg':
            self.model = MPIPI_model(version = 'GG_FINCHES')
            self.model.swap_coul_func()
            self.model.swap_wf_func()
            self.charge_prefactor = 0.2
            self.unit_conv = 1.0
            self.null_interaction_baseline = -0.128533
        elif ff_type == 'mpipi':
            self.model = MPIPI_model(version = 'normal')
            self.model.swap_coul_func()
            self.model.swap_wf_func()
            self.charge_prefactor = 0.2
            self.unit_conv = 1.0
            self.null_interaction_baseline = -0.066265
        elif ff_type == 'hps_dignon':
            self.model = HPS_model(version = 'Dignon')
        elif ff_type == 'hps_tesei':
            self.model = HPS_model(version = 'Tesei')
        elif ff_type == 'hps_urry':
            self.model = HPS_model(version = 'Urry')
        elif ff_type == 'hps_fb':
            self.model = HPS_model(version = 'FB')
        elif ff_type == 'calvados1':
            self.model = Calvados_model(version = '1')
            self.model.swap_coul_func()
            self.unit_conv = 4.184
        elif ff_type == 'calvados2':
            self.model = Calvados_model(version = '2')
            self.charge_prefactor = 0.7
            self.null_interaction_baseline = -0.45
            self.unit_conv = 4.184
            self.model.swap_coul_func()
        elif ff_type == 'kh-d':
            self.model = KHD_model()
        else:
            raise ValueError(f"Invalid ff type: {ff_type}, choose from mpipi, mpipi_gg, hps_dignon, hps_tesei, hps_urry, hps_fb, calvados1, calvados2, kh-d")

        if self.use_charge and self.charge_prefactor is None:
            raise ValueError("Charge prefactor is not set for this ff type. Set use_charge to False or manually set the charge prefactor.")

        base_conditions = [(288, 150, 7.0)]
        if self.null_interaction_baseline is None:
            self.set_aa_interactions(base_conditions)
            self.get_null_interaction_baseline()

    def get_aliphatic_residues(self, pseq):
        """Compute per-position aliphatic probability from a probabilistic sequence.

        Args:
            pseq: Probabilistic sequence, shape (L, 23).

        Returns:
            Per-position aliphatic content, shape (L,).
        """
        aliphatic_inds = [RES_ALPHA.index(aa) for aa in aliphatic_aas]
        aliphatic_residues = pseq[:, aliphatic_inds].sum(axis = 1)

        return aliphatic_residues

    def build_gap_merged_fragments(self, ali_mask, max_separation=1):
        """Identify contiguous aliphatic fragments, allowing small gaps.

        Args:
            ali_mask: Binary mask of aliphatic positions, shape (L,).
            max_separation: Maximum number of non-aliphatic residues
                allowed within a fragment before splitting.

        Returns:
            List of (start, end) tuples for each merged fragment.
        """
        fragments = []
        seq_len = len(ali_mask)
        i = 0
        while i < seq_len:
            if ali_mask[i] == 1:
                start = i
                zeros_allowed = max_separation
                j = i + 1
                while j < seq_len:
                    if ali_mask[j] == 1:
                        j += 1
                        zeros_allowed = max_separation
                    else:
                        if zeros_allowed > 0:
                            zeros_allowed -= 1
                            j += 1
                        else:
                            break
                # trim trailing zeros from [start, j)
                end = j
                while end > start and ali_mask[end - 1] == 0:
                    end -= 1
                fragments.append((start, end))
                i = j
            else:
                i += 1
        return fragments
    
    def get_aliphatic_weighting(self, pseq, max_separation=1):
        """Compute per-position aliphatic cluster strength.

        Assigns each aliphatic position a windowed count (clipped to 3)
        reflecting the local density of aliphatic residues within its
        merged fragment.

        Args:
            pseq: Probabilistic sequence, shape (L, 23).
            max_separation: Gap tolerance for fragment merging.

        Returns:
            Per-position cluster weights, shape (L,), values in [0, 3].
        """
        ali = self.get_aliphatic_residues(pseq)               # shape (L,), float
        ali_mask = (ali > 0.1).astype(int)               # shape (L,), int {0,1}
        L = len(ali_mask)

        # 2) Gap-merged fragments (allow a single 0 inside)
        fragments = self.build_gap_merged_fragments(ali_mask, max_separation=max_separation)

        # 3) Per-position windowed counts clipped to fragment bounds
        groups = jnp.zeros_like(ali, dtype=jnp.float32)
        w_half = 4                                           # original uses max_distance=4
        for f_start, f_end in fragments:
            frag = jnp.array(ali_mask[f_start:f_end])   # local fragment as 0/1 ints
            l = int(frag.shape[0])

            # iterate local indices j where frag[j] == 1
            for j_local in range(l):
                if int(frag[j_local]) != 1:
                    continue

                if l <= w_half:
                    # small fragment: whole fragment window
                    count = jnp.sum(frag)
                else:
                    # branchy window selection (matches original)
                    # left-edge: window starts at 0
                    if (w_half >= j_local) and ((j_local + w_half) <= l):
                        left = 0
                        right = j_local + w_half + 1
                    # right-edge: window ends at l
                    elif (j_local + w_half) > l:
                        left = j_local - w_half
                        right = l
                    # middle: symmetric window
                    else:
                        left = j_local - w_half
                        right = j_local + w_half + 1

                    # count 1s in local window
                    count = jnp.sum(frag[left:right])

                groups = groups.at[f_start + j_local].set(jnp.minimum(count, 3.0))

        return groups

    def get_multiplier_weight_vectorized(self, w1, w2):
        """Compute pairwise interaction multiplier from aliphatic cluster weights.

        Maps the minimum of two cluster weights to a multiplier in [1, 3]
        with a linear-then-quadratic profile.

        Args:
            w1: Cluster weights for sequence 1 (broadcastable).
            w2: Cluster weights for sequence 2 (broadcastable).

        Returns:
            Pairwise multiplier array.
        """
        w1_clipped = jnp.clip(w1, 1.0, 3.0)
        w2_clipped = jnp.clip(w2, 1.0, 3.0)
        
        excess_1 = w1_clipped - 1.0
        excess_2 = w2_clipped - 1.0
        min_excess = jnp.minimum(excess_1, excess_2)
        
        quadratic_boost = jnp.maximum(0.0, min_excess - 1.0)
        
        multiplier = 1.0 + 0.5 * min_excess + 1.0 * quadratic_boost
        
        return multiplier

    def get_aliphatic_interaction_weighting(self, pseq1, pseq2):
        """Compute the L1 x L2 aliphatic interaction multiplier matrix.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).

        Returns:
            Multiplier matrix, shape (L1, L2).
        """
        weighting_1 = self.get_aliphatic_weighting(pseq1)
        weighting_2 = self.get_aliphatic_weighting(pseq2)
        
        weighting_1_expanded = weighting_1[:, None]  
        weighting_2_expanded = weighting_2[None, :] 
        
        weighting_overall = self.get_multiplier_weight_vectorized(weighting_1_expanded, weighting_2_expanded)
        
        return weighting_overall

    def get_neighbors_window_of3(self, i, pseq):
        """Extract a 3-residue window centered at position i (clamped at edges).

        Args:
            i: Center position index.
            pseq: Probabilistic sequence, shape (L, 23).

        Returns:
            Window slice of pseq, shape (2 or 3, 23).
        """
        if i == 0:
            return pseq[:i+2]
        elif i == pseq.shape[0] - 1:
            return pseq[i-1:]
        else:
            return pseq[i-1:i+2]

    def calculacte_FCR_NCPR(self, window):
        """Compute fraction of charged residues and net charge per residue.

        Args:
            window: Slice of probabilistic sequence, shape (W, 23).

        Returns:
            Tuple of (FCR, NCPR) scalars.
        """
        pos_indices = [RES_ALPHA.index(aa) for aa in ['K', 'R']]
        neg_indices = [RES_ALPHA.index(aa) for aa in ['D', 'E']]

        pos_count = jnp.sum(jnp.sum(window[:, pos_indices], axis = 1))
        neg_count = jnp.sum(jnp.sum(window[:, neg_indices], axis = 1))

        total_charged = pos_count + neg_count
        fcr = total_charged / window.shape[0]
        ncpr = (pos_count - neg_count) / window.shape[0]

        return fcr, ncpr

    def get_charge_interaction_weighting(self, pseq1, pseq2):
        """Compute an L1 x L2 charge-context weighting mask.

        For each pair of charged residue positions, computes |NCPR/FCR|
        of their combined 3-residue windows to modulate the interaction.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).

        Returns:
            Charge weighting matrix, shape (L1, L2).
        """
        seq1 = ''.join(RES_ALPHA[jnp.argmax(pseq1[i])] for i in range(pseq1.shape[0]))
        seq2 = ''.join(RES_ALPHA[jnp.argmax(pseq2[i])] for i in range(pseq2.shape[0]))

        result = jnp.zeros((pseq1.shape[0], pseq2.shape[0]))

        for i in range(pseq1.shape[0]):
            if seq1[i] in charged_aas:
                for j in range(pseq2.shape[0]):
                    if seq2[j] in charged_aas:
                        window1 = self.get_neighbors_window_of3(i, pseq1)
                        window2 = self.get_neighbors_window_of3(j, pseq2)

                        combined_window = jnp.concatenate([window1, window2])

                        fcr, ncpr = self.calculacte_FCR_NCPR(combined_window)

                        if fcr > 0.0:
                            charge_weight = jnp.abs(ncpr / fcr)
                        else:
                            charge_weight = 0.0

                        result = result.at[i, j].set(charge_weight)

        return result

    def get_null_interaction_baseline(self, lower_end = -10.0, upper_end = 10.0, alternative_sequence = None):
        """Calibrate the null interaction baseline using a GS-repeat sequence.

        Finds the baseline value such that a (GS)_200 sequence yields
        zero net interaction (root-finding on the interaction parameter).

        Args:
            lower_end: Lower bracket for root finding.
            upper_end: Upper bracket for root finding.
            alternative_sequence: Use this sequence instead of 'GS'*200.
        """
        self.null_interaction_baseline = 0.0

        if alternative_sequence is not None:
            seq = alternative_sequence
        else:
            seq = 'GS' * 200

        def f(null_interaction_baseline):
            return self.return_interaction_parameters(seq, seq, null_interaction_baseline = null_interaction_baseline, use_charge = False, use_aliphatic = False)

        result = root_scalar(f, bracket = (lower_end, upper_end))

        self.null_interaction_baseline = result.root / self.unit_conv

    @functools.partial(jit, static_argnums=(0,))
    def set_aa_interactions(self, all_conditions, r0=0.1, r1=30.0, num_steps=2991):
        """Precompute the 23x23 pairwise residue interaction matrix per condition.

        Integrates unbonded potentials over [sigma, 3*sigma] for each
        residue-type pair, yielding a condition-dependent interaction matrix.

        Args:
            all_conditions: List of (temp, salt, pH) tuples.
            r0: Radial grid lower bound (Angstroms).
            r1: Radial grid upper bound (Angstroms).
            num_steps: Number of radial grid points.

        Returns:
            Array of shape (N_conditions, 23, 23).
        """
        r = jnp.linspace(r0, r1, num_steps)
        dr = jnp.diff(r)
        all_unbonded_interactions = []
        for condition in all_conditions:
            pair_interactions_itr = jnp.zeros((NUM_RESIDUES, NUM_RESIDUES))
            temp, salt, pH = condition
            charges = self.model.calculate_charges(pH=pH, pH_mode=self.pH_mode)

            # how FINCHES handles histidines with calvados
            if self.model.ff_name == 'Calvados1' or self.model.ff_name == 'Calvados2':
                charges = charges.at[RES_ALPHA.index('H')].set(1. / (1 + 10**(pH - 6.0)))


            pair_charges = self.model.create_pair_charges(charges)
            unbonded_interactions = self.model.calculate_unbonded(r, pair_charges, salt, temp)  * self.unit_conv # (529, len(r))

            s1 = self.model.sigma_flattened             # (529,)
            s3 = self.model.sigma_flattened * 3.0       # (529,)

            # nearest r indices to each s1 and s3 entry
            s1_idx = jnp.abs(r[None, :] - s1[:, None]).argmin(axis=1)
            s3_idx = jnp.abs(r[None, :] - s3[:, None]).argmin(axis=1)

            s1_idx_reshaped = s1_idx.reshape(NUM_RESIDUES, NUM_RESIDUES)
            s3_idx_reshaped = s3_idx.reshape(NUM_RESIDUES, NUM_RESIDUES)

            r_idx = jnp.arange(unbonded_interactions.shape[1])[None, :]  # (1, N)


            mask = (r_idx >= s1_idx[:, None]) & (r_idx < (s3_idx[:, None]))  # (529, N)
            segment_mask = mask[:, :-1] & mask[:, 1:]

            segments = 0.5 * (unbonded_interactions[:, :-1] + unbonded_interactions[:, 1:]) * dr
            segments = segments * dr[None, :] * segment_mask

            pair_integrals = 100 * jnp.sum(segments, axis=1)

            reshaped_pair_integrals = pair_integrals.reshape(NUM_RESIDUES, NUM_RESIDUES)

            all_unbonded_interactions.append(reshaped_pair_integrals)

        return jnp.array(all_unbonded_interactions) #N_conditions x NUM_RESIDUES**2

    def return_interaction_parameters(self, seq1, seq2, aa_interactions = None, null_interaction_baseline = None, use_charge = None, use_aliphatic = None):
        """Compute FINCHES interaction parameters from sequence strings.

        Builds the pairwise interaction matrix W, applies charge and
        aliphatic weightings, then decomposes into attractive/repulsive
        components relative to the null baseline.

        Args:
            seq1: First amino acid sequence string.
            seq2: Second amino acid sequence string.
            aa_interactions: Precomputed 23x23 matrices. If None, uses stored values.
            null_interaction_baseline: Override the calibrated baseline.
            use_charge: Override the charge weighting flag.
            use_aliphatic: Override the aliphatic weighting flag.

        Returns:
            Array of epsilon values, one per condition.
        """
        pseq1 = seq_to_one_hot(seq1)
        pseq2 = seq_to_one_hot(seq2)

        if null_interaction_baseline is None:
            null_interaction_baseline = self.null_interaction_baseline
        if use_charge is None:
            use_charge = self.use_charge
        if use_aliphatic is None:
            use_aliphatic = self.use_aliphatic

        all_eps = []
        for aa_interaction in aa_interactions:
            W = pseq1 @ aa_interaction @ pseq2.T

            if self.use_charge:
                charge_mask = jnp.array(self.get_charge_interaction_weighting(pseq1, pseq2), dtype = jnp.float32)
                charge_pref = float(self.charge_prefactor)
                W = W * (1 - charge_mask * charge_pref)

            if self.use_aliphatic:
                aliph_mask = jnp.array(self.get_aliphatic_interaction_weighting(pseq1, pseq2), dtype = jnp.float32)
                W = W * aliph_mask

            baseline = null_interaction_baseline
            attractive_matrix = jnp.where(W < baseline, W, 0.0)
            repulsive_matrix = jnp.where(W > baseline, W, 0.0)

            attractive_matrix = attractive_matrix - baseline
            repulsive_matrix = repulsive_matrix - baseline

            attractive_vector = jnp.mean(attractive_matrix, axis = 1)
            repulsive_vector = jnp.mean(repulsive_matrix, axis = 1)

            epsilon = jnp.sum(attractive_vector) + jnp.sum(repulsive_vector)
            all_eps.append(epsilon)

        return jnp.array(all_eps)

    def return_interaction_parameters_pseqs(self, pseq1, pseq2, aa_interactions, null_interaction_baseline = None, use_charge = None, use_aliphatic = None):
        """Compute FINCHES interaction parameters from probabilistic sequences.

        Same as return_interaction_parameters but operates on pseq arrays
        directly. Not JIT-compiled due to Python-level branching in the
        charge and aliphatic weighting logic.

        Args:
            pseq1: Probabilistic sequence 1, shape (L1, 23).
            pseq2: Probabilistic sequence 2, shape (L2, 23).
            aa_interactions: Precomputed 23x23 matrices per condition.
            null_interaction_baseline: Override the calibrated baseline.
            use_charge: Override the charge weighting flag.
            use_aliphatic: Override the aliphatic weighting flag.

        Returns:
            Array of epsilon values, one per condition.
        """
        if null_interaction_baseline is None:
            null_interaction_baseline = self.null_interaction_baseline
        if use_charge is None:
            use_charge = self.use_charge
        if use_aliphatic is None:
            use_aliphatic = self.use_aliphatic

        all_eps = []
        for aa_interaction in aa_interactions:
            W = pseq1 @ aa_interaction @ pseq2.T

            charge_mask = jnp.array(self.get_charge_interaction_weighting(pseq1, pseq2), dtype = jnp.float32)
            charge_pref = float(self.charge_prefactor)
            W = W * (1 - charge_mask * charge_pref)

            aliph_mask = jnp.array(self.get_aliphatic_interaction_weighting(pseq1, pseq2), dtype = jnp.float32)
            W = W * aliph_mask

            baseline = null_interaction_baseline
            attractive_matrix = jnp.where(W < baseline, W, 0.0)
            repulsive_matrix = jnp.where(W > baseline, W, 0.0)

            attractive_matrix = attractive_matrix - baseline
            repulsive_matrix = repulsive_matrix - baseline

            attractive_vector = jnp.mean(attractive_matrix, axis = 1)
            repulsive_vector = jnp.mean(repulsive_matrix, axis = 1)

            epsilon = jnp.sum(attractive_vector) + jnp.sum(repulsive_vector)
            all_eps.append(epsilon)

        return jnp.array(all_eps)