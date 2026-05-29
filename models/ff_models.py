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


ROOT = Path(__file__).resolve().parents[1]
kb = 0.0019872036 # kcal/mol/K
RES_ALPHA = "MGKTRADEYVLQWFSHNPCIXZB"
NUM_RESIDUES = len(RES_ALPHA)
assert(NUM_RESIDUES == 23)

DEBYE_RC = 90.0

RES_CODE = {
    "A": "Ala",
    "R": "Arg",
    "D": "Asp",
    "N": "Asn",
    "C": "Cys",
    "E": "Glu",
    "Q": "Gln",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "L": "Leu",
    "K": "Lys",
    "M": "Met",
    "F": "Phe",
    "P": "Pro",
    "S": "Ser",
    "T": "Thr",
    "W": "Trp",
    "Y": "Tyr",
    "V": "Val",
    "X": "mArg",
    "Z": "pSer",
    "B": "pThr"
}

RES_3_TO_1 = {value.upper(): key for key, value in RES_CODE.items()}


def read_MPIPI_ff_params(fpath):
    """Read MPIPI force field parameters from a whitespace-delimited file.

    Args:
        fpath: Path to the MPIPI parameter file.

    Returns:
        Tuple of (eps_table, eps_flattened, sigma_flattened, nu_flattened,
        mu_flattened, rc_flattened) — the 23x23 epsilon table and flattened
        (529,) arrays for each parameter.
    """
    mpipi_df = pd.read_csv(fpath, delim_whitespace=True, header=None, 
                         names = ['res1', 'res2', 'eps', 'sigma', 'nu', 'mu', 'rc'])
    eps_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    sigma_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    nu_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    mu_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    rc_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    for i in range(NUM_RESIDUES):
        for j in range(i, NUM_RESIDUES):
            df_row = mpipi_df[(mpipi_df['res1'] == i+1) & (mpipi_df['res2'] == j+1)]
            assert(len(df_row) == 1)
            df_row = df_row.iloc[0]
            eps_table[i, j] = df_row.eps
            eps_table[j, i] = df_row.eps

            sigma_table[i, j] = df_row.sigma
            sigma_table[j, i] = df_row.sigma

            nu_table[i, j] = df_row.nu
            nu_table[j, i] = df_row.nu

            mu_table[i, j] = df_row.mu
            mu_table[j, i] = df_row.mu
            
            rc_table[i, j] = df_row.rc
            rc_table[j, i] = df_row.rc

    eps_table = jnp.array(eps_table, dtype=jnp.float32)
    sigma_table = jnp.array(sigma_table, dtype=jnp.float32)
    nu_table = jnp.array(nu_table, dtype=jnp.float32)
    mu_table = jnp.array(mu_table, dtype=jnp.float32)
    rc_table = jnp.array(rc_table, dtype=jnp.float32)

    eps_flattened = list()
    sigma_flattened = list()
    nu_flattened = list()
    mu_flattened = list()
    rc_flattened = list()
    for i in range(NUM_RESIDUES):
        for j in range(NUM_RESIDUES):
            eps_flattened.append(eps_table[i][j])
            sigma_flattened.append(sigma_table[i][j])
            nu_flattened.append(nu_table[i][j])
            mu_flattened.append(mu_table[i][j])
            rc_flattened.append(rc_table[i][j])
    eps_flattened = jnp.array(eps_flattened, dtype=jnp.float32)
    sigma_flattened = jnp.array(sigma_flattened, dtype=jnp.float32)
    nu_flattened = jnp.array(nu_flattened, dtype=jnp.float32)
    mu_flattened = jnp.array(mu_flattened, dtype=jnp.float32)
    rc_flattened = jnp.array(rc_flattened, dtype=jnp.float32)
    return eps_table, eps_flattened, sigma_flattened, nu_flattened, mu_flattened, rc_flattened

def read_HPS(fpath):
    """Read HPS force field parameters from a whitespace-delimited file.

    Args:
        fpath: Path to the HPS parameter file.

    Returns:
        Tuple of (lambda_table, eps_flattened, sigma_flattened,
        lambda_flattened) — the 23x23 lambda table and flattened (529,) arrays.
    """
    hps_df = pd.read_csv(fpath, delim_whitespace=True, header=None, 
                         names = ['res1', 'res2', 'eps', 'sigma', 'lambda_'])
    
    eps_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    sigma_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    lambda_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))

    for i in range(NUM_RESIDUES):
        for j in range(i, NUM_RESIDUES):
            df_row = hps_df[(hps_df['res1'] == i+1) & (hps_df['res2'] == j+1)]
            assert(len(df_row) == 1)
            df_row = df_row.iloc[0]

            eps_table[i, j] = df_row.eps
            eps_table[j, i] = df_row.eps

            sigma_table[i, j] = df_row.sigma
            sigma_table[j, i] = df_row.sigma

            lambda_table[i, j] = df_row.lambda_
            lambda_table[j, i] = df_row.lambda_

    eps_table = jnp.array(eps_table, dtype=jnp.float32)
    sigma_table = jnp.array(sigma_table, dtype=jnp.float32)
    lambda_table = jnp.array(lambda_table, dtype=jnp.float32)

    eps_flattened = list()
    sigma_flattened = list()
    lambda_flattened = list()
    for i in range(NUM_RESIDUES):
        for j in range(NUM_RESIDUES):
            eps_flattened.append(eps_table[i][j])
            sigma_flattened.append(sigma_table[i][j])
            lambda_flattened.append(lambda_table[i][j])
    eps_flattened = jnp.array(eps_flattened, dtype=jnp.float32)
    sigma_flattened = jnp.array(sigma_flattened, dtype=jnp.float32)
    lambda_flattened = jnp.array(lambda_flattened, dtype=jnp.float32)

    return lambda_table, eps_flattened, sigma_flattened, lambda_flattened

def read_calvados(fpath):
    """Read CALVADOS force field parameters from a text file.

    Args:
        fpath: Path to the CALVADOS parameter file.

    Returns:
        Tuple of (lambdas, lambdas_flattened, sigmas_flattened) —
        per-residue lambdas and pairwise-averaged flattened (529,) arrays.
    """
    lambdas = list()
    sigmas = list()
    with open(fpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            idx, mass, charge, sigma, lambda_, aa_name = parts
            lambdas.append(float(lambda_))
            sigmas.append(float(sigma))
    lambdas = jnp.array(lambdas, dtype=jnp.float32)
    sigmas = jnp.array(sigmas, dtype=jnp.float32)

    lambdas_flattened = list()
    sigmas_flattened = list()
    for i in range(NUM_RESIDUES):
        for j in range(NUM_RESIDUES):
            lambda_ij = (lambdas[i] + lambdas[j]) / 2
            lambdas_flattened.append(lambda_ij)
            sigma_ij = (sigmas[i] + sigmas[j]) / 2
            sigmas_flattened.append(sigma_ij)
    lambdas_flattened = jnp.array(lambdas_flattened, dtype=jnp.float32)
    sigmas_flattened = jnp.array(sigmas_flattened, dtype=jnp.float32)
    return lambdas, lambdas_flattened, sigmas_flattened

def read_KH_D(fpath):
    """Read Kim-Hummer-D force field parameters from a text file.

    Args:
        fpath: Path to the KH-D parameter file.

    Returns:
        Tuple of (eps_table, eps_flattened, sigma_flattened, lambda_flattened) —
        the 23x23 epsilon table and flattened (529,) arrays. Lambda is +1
        for repulsive pairs and -1 for attractive.
    """
    eps_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    sigma_table = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))
    with open(fpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            res1, res2, sigma, eps = parts
            res1 = int(res1) - 1
            res2 = int(res2) - 1
            sigma = float(sigma)
            eps = float(eps)
            eps_table[res1, res2] = eps
            eps_table[res2, res1] = eps
            sigma_table[res1, res2] = sigma
            sigma_table[res2, res1] = sigma
    eps_table = jnp.array(eps_table, dtype=jnp.float32)
    sigma_table = jnp.array(sigma_table, dtype=jnp.float32)


    eps_flattened = list()
    sigma_flattened = list()
    lambda_flattened = list()
    for i in range(NUM_RESIDUES):
        for j in range(NUM_RESIDUES):
            eps_flattened.append(eps_table[i][j])
            sigma_flattened.append(sigma_table[i][j])
            lambda_flattened.append(jnp.where(eps_table[i][j] <= 0.0, 1.0, -1.0))
    eps_flattened = jnp.array(eps_flattened, dtype=jnp.float32)
    sigma_flattened = jnp.array(sigma_flattened, dtype=jnp.float32)
    lambda_flattened = jnp.array(lambda_flattened, dtype=jnp.float32)
    return eps_table, eps_flattened, sigma_flattened, lambda_flattened
    

class MPIPI_model:
    """Mpipi coarse-grained force field for IDP interactions.

    Combines Wang-Frenkel short-range potentials with Debye-Huckel
    electrostatics. Supports normal, GG, and FINCHES variants.
    """

    def __init__(self, version = 'GG'):
        """Initialize the MPIPI model.

        Args:
            version: 'GG', 'normal', or 'GG_FINCHES'.
        """
        self.version = version

        if version == 'GG':
            self.ff_name = 'MPIPI_GG'
            self.fpath = ROOT / 'ff_params' / 'MPIPI_GG.txt'
            self.debye_kappa = 0.12578
        elif version == 'normal':
            self.ff_name = 'MPIPI'
            self.fpath = ROOT / 'ff_params' / 'MPIPI.txt'
            self.debye_kappa = 0.12578
        elif version == 'GG_FINCHES':
            self.ff_name = 'MPIPI_GG_FINCHES'
            self.fpath = ROOT / 'ff_params' / 'MPIPI_GG_FINCHES.txt'
            self.debye_kappa = 0.12578
        else:
            raise ValueError(f"Invalid version: {version}")

        self.eps_table, self.eps_flattened, self.sigma_flattened, self.nu_flattened, self.mu_flattened, self.rc_flattened = read_MPIPI_ff_params(self.fpath)
        self._mapped_wang_frenkel = vmap(self.wang_frenkel_internal_func, in_axes = (None, 0, 0, 0, 0, 0))
        self._mapped_coulomb = vmap(self.coulomb_internal_func, in_axes = (None, 0, None, None, None))
    

    def construct_eps_matrix(self):
        """Construct the 23x23 epsilon matrix in kJ/mol units.

        Returns:
            Negated epsilon table scaled by 4.184 (kcal->kJ conversion).
        """
        return -self.eps_table * 4.184

    def calculate_charges(self, pH = 7, pH_mode = 'normal'):
        """Compute per-residue-type charges.

        In 'normal' mode, uses fixed partial charges. In 'pH' mode,
        computes Henderson-Hasselbalch titration charges.

        Args:
            pH: Solution pH (used only in 'pH' mode).
            pH_mode: 'normal' or 'pH'.

        Returns:
            Charge array of shape (23,), ordered by RES_ALPHA.
        """
        if pH_mode == 'normal':
            charges_dict = {
                'K': 0.75,
                'R': 0.75,
                'D': -0.75,
                'E': -0.75,
                'H': 0.375,
                'B': -2.0,
                'Z': -2.0,
                'X': 0.0
            }
        elif pH_mode == 'pH':
            pKa_values = {
                'N-terminus': 9.69,
                'C-terminus': 2.34,
                'D': 3.86,
                'E': 4.25,
                'C': 8.33,
                'Y': 10.07,
                'H': 6.00,
                'K': 10.54,
                'R': 12.48,
                }

            charges_dict = {
                'N-terminus': 1.0 / (1.0 + 10**(pH - pKa_values['N-terminus'])),
                'C-terminus': -1.0 / (1.0 + 10**(pKa_values['C-terminus'] - pH)),
                'D': -1.0 / (1.0 + 10**(pKa_values['D'] - pH)),
                'E': -1.0 / (1.0 + 10**(pKa_values['E'] - pH)),
                'C': -1.0 / (1.0 + 10**(pKa_values['C'] - pH)),
                'Y': -1.0 / (1.0 + 10**(pKa_values['Y'] - pH)),
                'H': 1.0 / (1.0 + 10**(pH - pKa_values['H'])),
                'K': 1.0 / (1.0 + 10**(pH - pKa_values['K'])),
                'R': 1.0 / (1.0 + 10**(pH - pKa_values['R'])),
                'B': -2.0,
                'Z': -2.0,
                'X': 0.0
            }

        else:
            raise ValueError(f"Invalid pH mode: {pH_mode}")

        return jnp.array([charges_dict.get(res, 0.0) for res in RES_ALPHA])

    
    def create_pair_charges(self, charges):
        """Build all 529 charge pairs from per-residue charges.

        Args:
            charges: Per-residue-type charges, shape (23,).

        Returns:
            Array of shape (529, 2) with [qi, qj] for each pair.
        """
        pair_charges = list()
        for i in range(NUM_RESIDUES):
            for j in range(NUM_RESIDUES):
                pair_charges.append([charges[i], charges[j]])
        return jnp.array(pair_charges)

    @functools.partial(jit, static_argnums = (0,))
    def wang_frenkel_internal_func(self, r, r_c, sigma, nu, mu, eps):
        """Evaluate the Wang-Frenkel potential at distance r.

        Args:
            r: Interparticle distance (scalar or array).
            r_c: Cutoff distance.
            sigma: Length scale parameter.
            nu: Exponent parameter.
            mu: Exponent parameter.
            eps: Well depth (energy scale).

        Returns:
            Potential energy at r, zero beyond r_c.
        """
        alpha = 2*nu * (r_c/sigma)**(2*mu)
        alpha *= ((1+2*nu) / (2*nu * ((r_c/sigma)**(2*mu)-1)))**(2*nu+1)
        val = eps*alpha * ((sigma/r)**(2*mu)-1) * ((r_c/r)**(2*mu)-1)**(2*nu)
        r_c = 3*sigma

        
        return jnp.where(r < r_c, val, 0.0)


    @functools.partial(jit, static_argnums = (0,))
    def wang_frenkel_internal_func_FINCHES(self, r, r_c, sigma, nu, mu, eps):
        """Wang-Frenkel potential variant used by the FINCHES model.

        Uses Rij = 3*sigma as the cutoff regardless of r_c.

        Args:
            r: Interparticle distance.
            r_c: Cutoff distance (unused; Rij = 3*sigma is used instead).
            sigma: Length scale parameter.
            nu: Exponent parameter.
            mu: Exponent parameter.
            eps: Well depth.

        Returns:
            Potential energy at r.
        """
        Rij = 3 *sigma

        alpha_ij_term1 = 2 * nu * jnp.power(Rij/sigma, 2*mu)
        alpha_ij_term2 = (2 * nu + 1) / (2 * nu * (jnp.power(Rij/sigma, 2*mu) - 1))

        alpha_ij = alpha_ij_term1 * jnp.power(alpha_ij_term2, 2*nu + 1)

        main_term1 = eps * alpha_ij
        main_term2 = jnp.power(sigma/r, 2 * mu) - 1
        main_term3 = jnp.power(jnp.power(Rij/r, 2 * mu) - 1, 2*nu)

        return main_term1 * main_term2 * main_term3

    def swap_wf_func(self):
        """Replace the Wang-Frenkel potential with the FINCHES variant."""
        self._mapped_wang_frenkel = vmap(self.wang_frenkel_internal_func_FINCHES, in_axes = (None, 0, 0, 0, 0, 0))

    @functools.partial(jit, static_argnums = (0,))
    def coulomb_internal_func(self, r, qij, r_c, temp, salt_conc):
        """Evaluate screened Coulomb (Debye-Huckel) potential.

        Uses a fixed dielectric constant of 80 and salt-dependent screening.

        Args:
            r: Interparticle distance.
            qij: Charge pair [qi, qj].
            r_c: Electrostatic cutoff distance.
            temp: Temperature in Kelvin (unused in this variant).
            salt_conc: Salt concentration in mM.

        Returns:
            Electrostatic energy at r, zero beyond r_c.
        """
        qi = qij[0]
        qj = qij[1]
        k = self.debye_kappa * jnp.sqrt(salt_conc/150.0)
        C = 5.513725184e-22 # (1/(4*pi*eps0)  in units of kcal* (Armstrong)/(electron^^2)
        fac1 =  249.4 - 0.788 * temp + 0.00072 * temp**2
        fac2 = 1 - 0.2551 * (salt_conc/1000) + 0.05151 * (salt_conc/1000)**2 - 0.006889 * (salt_conc/1000)**3
        epsilon = fac1 * fac2
        epsilon = 80.0

        val = (C * qi * qj) / (epsilon * r) * jnp.exp(-k * r)
        val *= 6.022e23
        return jnp.where(r < r_c, val, 0.0)

    @functools.partial(jit, static_argnums = (0,))
    def coulomb_internal_func_FINCHES(self, r, qij, r_c, temp, salt_conc):
        """Screened Coulomb potential variant used by the FINCHES model.

        Computes Debye-Huckel electrostatics with salt-dependent screening
        using SI-based constants and returns energy in kJ/mol.

        Args:
            r: Interparticle distance in Angstroms.
            qij: Charge pair [qi, qj].
            r_c: Cutoff distance (unused).
            temp: Temperature in Kelvin.
            salt_conc: Salt concentration in mM.

        Returns:
            Electrostatic energy at r in kJ/mol.
        """
        qi = qij[0]
        qj = qij[1]

        salt_conc /= 1000
        sqrt_salt = jnp.sqrt(salt_conc)
        kappa = sqrt_salt / 3.06

        C = 1.602176634e-19  # elementary charge constant (in Coulombs)
        Na = 6.023e23        # Avogadro's number (unitless - just a big ol number)
        conversion_constant = (C*C*Na) / (jnp.pi * 4 * 8.854187812799999e-12)
        r_meters = r * 1e-10 # convert from Angstroms to meters
        DH = jnp.exp(-kappa*r)
        dieletric = 80.0

        energy_in_J = (conversion_constant * qi * qj) / (dieletric * r_meters) * DH

        return energy_in_J / 1000

    def swap_coul_func(self):
        """Replace the Coulomb potential with the FINCHES variant."""
        self._mapped_coulomb = vmap(self.coulomb_internal_func_FINCHES, in_axes = (None, 0, None, None, None))

    @functools.partial(jit, static_argnums = (0,))
    def calculate_unbonded(self, r, pair_charges, salt_conc, temp):
        """Compute total unbonded energy (WF + Coulomb) for all 529 pairs.

        Args:
            r: Radial distance(s).
            pair_charges: Charge pairs, shape (529, 2).
            salt_conc: Salt concentration in mM.
            temp: Temperature in Kelvin.

        Returns:
            Total unbonded energies, shape (529,) or (529, len(r)).
        """
        wf_vals = self._mapped_wang_frenkel(r, self.rc_flattened, self.sigma_flattened, self.nu_flattened, self.mu_flattened, self.eps_flattened)
        coul_vals = self._mapped_coulomb(r, pair_charges, DEBYE_RC, temp, salt_conc)
        return wf_vals + coul_vals

class HPS_model:
    """HPS (hydrophobicity scale) coarse-grained force field.

    Combines Ashbaugh-Hatch short-range potentials with Debye-Huckel
    electrostatics. Supports Dignon, Tesei, Urry, and FB parameterizations.
    """

    def __init__(self, version = 'Dignon'):
        """Initialize the HPS model.

        Args:
            version: 'Dignon', 'Tesei', 'Urry', or 'FB'.
        """
        self.version = version
        if version == 'Dignon':
            self.ff_name = 'HPS_Dignon'
            self.fpath = ROOT / 'ff_params' / 'HPS_dignon_check.txt'
        elif version == 'Tesei':
            self.ff_name = 'HPS_Tesei'
            self.fpath = ROOT / 'ff_params' / 'HPS_tesei.txt'
        elif version == 'Urry':
            self.ff_name = 'HPS_Urry'
            self.fpath = ROOT / 'ff_params' / 'HPS_urry.txt'
        elif version == 'FB':
            self.ff_name = 'HPS_FB'
            self.fpath = ROOT / 'ff_params' / 'HPS_FB.txt'
        else:
            raise ValueError(f"Invalid version: {version}, choose from Dignon, Tesei, Urry, FB")

        self.lambda_table, self.eps_flattened, self.sigma_flattened, self.lambda_flattened = read_HPS(self.fpath)
        self._mapped_ashbaugh_hatch = vmap(self.ashbaugh_hatch_internal_func, in_axes = (None, 0, 0, 0))
        self._mapped_coulomb = vmap(self.coulomb_internal_func, in_axes = (None, 0, None, None))


    def get_debye_kappa(self, temp, epsilon_r, salt_conc):
        """Compute and store the Debye screening parameter kappa.

        Args:
            temp: Temperature in Kelvin.
            epsilon_r: Relative permittivity.
            salt_conc: Salt concentration in mM.
        """
        salt_conc /= 1000
        lambda_D = jnp.sqrt(3.953940922571158e-6*temp*epsilon_r/salt_conc)*10
        self.debye_kappa = 1/lambda_D

    def get_relative_perm(self, temp, salt_conc):
        """Compute and store the temperature/salt-dependent relative permittivity.

        Args:
            temp: Temperature in Kelvin.
            salt_conc: Salt concentration in mM.
        """
        fac1 = 249.4 - 0.788 * temp + 0.00072 * temp**2
        fac2 = 1 - 0.2551 * (salt_conc/1000) + 0.05151 * (salt_conc/1000)**2 - 0.006889 * (salt_conc/1000)**3
        epsilon_r = fac1 * fac2
        self.epsilon_r = epsilon_r

    def construct_eps_matrix(self):
        """Construct the 23x23 effective epsilon matrix from lambda values.

        Returns:
            Epsilon matrix of shape (23, 23).
        """
        if self.version == 'Dignon':
            mu, Delta = 1, 0
            lambda_file = ROOT / 'ff_params' / 'lambda_dignon.txt'
        elif self.version == 'FB':
            mu, Delta = 1, 0
            lambda_file = ROOT / 'ff_params' / 'lambda_FB.txt'
        elif self.version == 'Urry':
            mu, Delta = 1, 0.08
            lambda_file = ROOT / 'ff_params' / 'lambda_urry.txt'
        elif self.version == 'Tesei':
            mu, Delta = 1, 0


        # eps_matrix = onp.zeros((NUM_RESIDUES, NUM_RESIDUES))

        # q, lambd = onp.loadtxt(lambda_file, dtype = str).T

        # val = onp.array([float(l) for l in lambd])
        # for i in range(20):
        #     for j in range(20):
        #         r1 = RES_ALPHA.index(res_3_to_1[ q[i] ])
        #         r2 = RES_ALPHA.index(res_3_to_1[ q[j] ])
        #         eps_matrix[r1, r2] = - ( 0.5 * (val[i] + val[j]) * mu - Delta)
        
        eps_matrix = -(self.lambda_table * mu - Delta)
        return eps_matrix

    def calculate_charges(self, pH = 7, pH_mode = 'normal'):
        """Compute per-residue-type charges for the HPS model.

        Args:
            pH: Solution pH (used only in 'pH' mode).
            pH_mode: 'normal' or 'pH'.

        Returns:
            Charge array of shape (23,), ordered by RES_ALPHA.
        """
        if pH_mode == 'normal' and (self.version == 'FB' or self.version == 'Urry' or self.version == 'Dignon'):
            charges_dict = {
                'D': -1.0,
                'E': -1.0,
                'K': 1.0,
                'R': 1.0,
                'H': 0.5,
                'X': 0.0,
                'Z': -2.0,
                'B': -2.0
            }
            return jnp.array([charges_dict.get(res, 0.0) for res in RES_ALPHA])
        elif pH_mode == 'normal' and (self.version == 'Tesei'):
            charges_dict = {
                'D': -1.0,
                'E': -1.0,
                'K': 1.0,
                'R': 1.0,
                'H': 0.0,
                'X': 0.0,
                'Z': -2.0,
                'B': -2.0
            }
            return jnp.array([charges_dict.get(res, 0.0) for res in RES_ALPHA])
        elif pH_mode == 'pH':
            pKa_values = {
            'N-terminus': 9.69,
            'C-terminus': 2.34,
            'D': 3.86,
            'E': 4.25,
            'C': 8.33,
            'Y': 10.07,
            'H': 6.00,
            'K': 10.54,
            'R': 12.48,
            }
            charges = {
                'N-terminus': 1.0 / (1.0 + 10**(pH - pKa_values['N-terminus'])),
                'C-terminus': -1.0 / (1.0 + 10**(pKa_values['C-terminus'] - pH)),
                'D': -1.0 / (1.0 + 10**(pKa_values['D'] - pH)),
                'E': -1.0 / (1.0 + 10**(pKa_values['E'] - pH)),
                'C': -1.0 / (1.0 + 10**(pKa_values['C'] - pH)),
                'Y': -1.0 / (1.0 + 10**(pKa_values['Y'] - pH)),
                'H': 1.0 / (1.0 + 10**(pH - pKa_values['H'])),
                'K': 1.0 / (1.0 + 10**(pH - pKa_values['K'])),
                'R': 1.0 / (1.0 + 10**(pH - pKa_values['R'])),
                'B': -2.0,
                'Z': -2.0,
                'X': 0.0
            }
            return jnp.array([charges.get(res, 0.0) for res in RES_ALPHA])
        else:
            raise ValueError(f"Invalid pH mode: {pH_mode}")

    def create_pair_charges(self, charges):
        """Build all 529 charge pairs from per-residue charges.

        Args:
            charges: Per-residue-type charges, shape (23,).

        Returns:
            Array of shape (529, 2).
        """
        pair_charges = list()
        for i in range(NUM_RESIDUES):
            for j in range(NUM_RESIDUES):
                pair_charges.append([charges[i], charges[j]])
        return jnp.array(pair_charges)
    

    @functools.partial(jit, static_argnums = (0,))
    def ashbaugh_hatch_internal_func(self, r, eps, sigma, lambda_):
        """Evaluate the Ashbaugh-Hatch potential at distance r.

        Combines a repulsive WCA core with a lambda-scaled attractive tail.

        Args:
            r: Interparticle distance.
            eps: Well depth.
            sigma: Length scale.
            lambda_: Hydrophobicity parameter (0 = repulsive, 1 = attractive).

        Returns:
            Potential energy at r.
        """
        potential_close = eps*(4*((sigma/r)**12-(sigma/r)**6)+1-lambda_)
        potential_far = lambda_*eps*4*((sigma/r)**12-(sigma/r)**6)
        return jnp.where(r <= 2**(1/6) * sigma, potential_close, potential_far)

    @functools.partial(jit, static_argnums = (0,))
    def coulomb_internal_func(self, r, qij, r_c, salt_conc):
        """Evaluate screened Coulomb potential for HPS model.

        Uses precomputed epsilon_r and debye_kappa from get_relative_perm
        and get_debye_kappa.

        Args:
            r: Interparticle distance.
            qij: Charge pair [qi, qj].
            r_c: Electrostatic cutoff distance.
            salt_conc: Salt concentration in mM (unused; kappa precomputed).

        Returns:
            Electrostatic energy at r, zero beyond r_c.
        """
        qi = qij[0]
        qj = qij[1]
        C = 5.513725184e-22 # (1/(4*pi*eps0)  in units of kcal* (Armstrong)/(electron^^2)
        val = (C * qi * qj) / (self.epsilon_r * r) * jnp.exp(-self.debye_kappa * r)
        val *= 6.022e23
        return jnp.where(r < r_c, val, 0.0)

    @functools.partial(jit, static_argnums = (0,))
    def calculate_unbonded(self, r, pair_charges, salt_conc, temp):
        """Compute total unbonded energy (Ashbaugh-Hatch + Coulomb) for all 529 pairs.

        Also updates internal epsilon_r and debye_kappa for the given conditions.

        Args:
            r: Radial distance(s).
            pair_charges: Charge pairs, shape (529, 2).
            salt_conc: Salt concentration in mM.
            temp: Temperature in Kelvin.

        Returns:
            Total unbonded energies, shape (529,) or (529, len(r)).
        """
        self.get_relative_perm(temp = temp, salt_conc = salt_conc)
        self.get_debye_kappa(temp = temp, epsilon_r = self.epsilon_r, salt_conc = salt_conc)
        ah_vals = self._mapped_ashbaugh_hatch(r, self.eps_flattened, self.sigma_flattened, self.lambda_flattened)
        coul_vals = self._mapped_coulomb(r, pair_charges, DEBYE_RC, salt_conc)
        return ah_vals + coul_vals

class Calvados_model:
    """CALVADOS coarse-grained force field for IDP interactions.

    Combines HPS-style Lennard-Jones short-range potentials with
    Yukawa electrostatics. Supports versions 1 and 2.
    """

    def __init__(self, version = '2'):
        """Initialize the CALVADOS model.

        Args:
            version: '1' or '2'.
        """
        ff_string = 'calvados_' + str(version) + '.txt'
        self.fpath = ROOT / 'ff_params' / ff_string
        self.lambdas, self.lambdas_flattened, self.sigma_flattened = read_calvados(self.fpath)
        self.debye_rc = 40.0
        self._mapped_hps_lj = vmap(self.hps_lj_internal_func, in_axes = (None, 0, 0))
        self._mapped_calvados_coul = vmap(self.calvados_coul_internal_func, in_axes = (None, 0, None, None, None))
        self.ff_name = 'Calvados' + str(version)

    def construct_eps_matrix(self):
        """Construct the 23x23 effective epsilon matrix from lambda values.

        Returns:
            Epsilon matrix of shape (23, 23).
        """
        mu, Delta = 1, 0

        sums = jnp.expand_dims(self.lambdas, 0) + jnp.expand_dims(self.lambdas, 1)
        eps_matrix = Delta - 0.5 * mu * sums
        # eps_matrix = -(0.5 * jnp.add.outer(self.lambdas, self.lambdas) * mu - Delta)
        return eps_matrix

    def calculate_charges(self, pH = 7, pH_mode = 'normal'):
        """Compute per-residue-type charges for the CALVADOS model.

        Args:
            pH: Solution pH (used only in 'pH' mode).
            pH_mode: 'normal' or 'pH'.

        Returns:
            Charge array of shape (23,), ordered by RES_ALPHA.
        """
        charges_dict = {
            'D': -1.0,
            'E': -1.0,
            'K': 1.0,
            'R': 1.0,
            'X': 0.0,
            'Z': -2.0,
            'B': -2.0
        }
        if pH_mode == 'normal':
            return jnp.array([charges_dict.get(res, 0.0) for res in RES_ALPHA])
        elif pH_mode == 'pH':
            pKa_values = {
            'N-terminus': 9.69,
            'C-terminus': 2.34,
            'D': 3.86,
            'E': 4.25,
            'C': 8.33,
            'Y': 10.07,
            'H': 6.00,
            'K': 10.54,
            'R': 12.48,
            }
            charges = {
                'N-terminus': 1.0 / (1.0 + 10**(pH - pKa_values['N-terminus'])),
                'C-terminus': -1.0 / (1.0 + 10**(pKa_values['C-terminus'] - pH)),
                'D': -1.0 / (1.0 + 10**(pKa_values['D'] - pH)),
                'E': -1.0 / (1.0 + 10**(pKa_values['E'] - pH)),
                'C': -1.0 / (1.0 + 10**(pKa_values['C'] - pH)),
                'Y': -1.0 / (1.0 + 10**(pKa_values['Y'] - pH)),
                'H': 1.0 / (1.0 + 10**(pH - pKa_values['H'])),
                'K': 1.0 / (1.0 + 10**(pH - pKa_values['K'])),
                'R': 1.0 / (1.0 + 10**(pH - pKa_values['R'])),
                'B': -2.0,
                'Z': -2.0,
                'X': 0.0
            }
            return jnp.array([charges.get(res, 0.0) for res in RES_ALPHA])
        else:
            raise ValueError(f"Invalid pH mode: {pH_mode}")

    def create_pair_charges(self, charges):
        """Build all 529 charge pairs from per-residue charges.

        Args:
            charges: Per-residue-type charges, shape (23,).

        Returns:
            Array of shape (529, 2).
        """
        pair_charges = list()
        for i in range(NUM_RESIDUES):
            for j in range(NUM_RESIDUES):
                pair_charges.append([charges[i], charges[j]])
        return jnp.array(pair_charges)

    @functools.partial(jit, static_argnums = (0,))
    def hps_lj_internal_func(self, r, sigma, lam):
        """Evaluate the CALVADOS HPS Lennard-Jones potential.

        Shifted Ashbaugh-Hatch form with eps = 0.2 kJ/mol and cutoff at 20 Angstroms.

        Args:
            r: Interparticle distance.
            sigma: Length scale.
            lam: Hydrophobicity parameter.

        Returns:
            Potential energy at r.
        """
        hps_lj_eps = 0.2


        lj_r_cuts = 20.0
        shift = (sigma / lj_r_cuts)**12 - (sigma / lj_r_cuts)**6
        r_switch = 2**(1/6) * sigma
        lj_core = 4 * ((sigma / r)**12 - (sigma / r)**6)
        cond1 = (r <= r_switch) 
        cond2 = (r_switch < r) & (r < lj_r_cuts)
        ah_long = lam * hps_lj_eps*(lj_core - 4 * shift)
        ah_short = (lj_core* hps_lj_eps) - (4 * lam * shift * hps_lj_eps) + ((1 - lam) * hps_lj_eps)
        val = jnp.where(cond1, ah_short, jnp.where(cond2, ah_long, 0.0))
        return val

    def swap_coul_func(self):
        """Replace the Coulomb potential with the FINCHES-compatible variant."""
        self._mapped_calvados_coul = vmap(self.calvados_coul_internal_func_FINCHES, in_axes = (None, 0, None, None, None))

    @functools.partial(jit, static_argnums = (0,))
    def calvados_coul_internal_func_FINCHES(self, r, qij, r_c, temp, salt_conc):
        """FINCHES-variant Yukawa electrostatics for CALVADOS.

        Uses temperature-dependent Bjerrum length and returns energy
        in kcal/mol (divided by 4.184 from kJ).

        Args:
            r: Interparticle distance in Angstroms.
            qij: Charge pair [qi, qj].
            r_c: Cutoff distance (unused; internal cutoff = 4 nm).
            temp: Temperature in Kelvin.
            salt_conc: Salt concentration in mM.

        Returns:
            Yukawa electrostatic energy at r in kcal/mol.
        """
        RT = 8.3145*temp*1e-3
        fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
        epsw = fepsw(temp)

        lB = 1.6021766**2/(4*jnp.pi*8.854188*epsw)*6.022*1000/RT

        qi = qij[0]
        qj = qij[1]
        qq = qi*qj

        ionic_strength = salt_conc/1000
        yukawa_kappa = jnp.sqrt(8*jnp.pi*lB*ionic_strength*6.022/10)
        yukawa_eps = qq*lB*RT
        yukawa_r_cut = 4.0
        r /= 10.0

        kappa = yukawa_kappa
        shift = jnp.exp(-yukawa_kappa*yukawa_r_cut)/yukawa_r_cut
        yu = yukawa_eps*(jnp.exp(-kappa*r)/r-shift)

        return yu/4.184
        
    @functools.partial(jit, static_argnums = (0,))
    def calvados_coul_internal_func(self, r, qij, r_c, temp, salt_conc):
        """Standard Yukawa electrostatics for CALVADOS.

        Args:
            r: Interparticle distance.
            qij: Charge pair [qi, qj].
            r_c: Electrostatic cutoff distance.
            temp: Temperature in Kelvin.
            salt_conc: Salt concentration in mM.

        Returns:
            Electrostatic energy at r, zero beyond r_c.
        """
        kT = 0.0019872041*temp
        # Calculate the prefactor for the Yukawa potential
        # epsw = 5321/temp+233.76-0.9297*temp+0.1417*1e-2*temp*temp-0.8292*1e-6*temp**3
        epsw = 80.0
        C = 5.513725184e-22 # (1/(4*pi*eps0)  in units of kcal* (Armstrong)/(electron^^2)
        lB =  C*6.022e23/(epsw*kT)
        eps_yu = (lB*kT)/6.022e23
        k_yu = jnp.sqrt(8 * jnp.pi * lB * (salt_conc/1000) * 6.022e23 * 1e-27)
        qi = qij[0]
        qj = qij[1]        
        

        shift = jnp.exp(-k_yu * r_c) / r_c
        val = qi * qj * eps_yu * (jnp.exp(-k_yu * r) / r - shift)
        val *= 6.022e23
        return jnp.where(r < r_c, val, 0.0)

    @functools.partial(jit, static_argnums = (0,))
    def calculate_unbonded(self, r, pair_charges, salt_conc, temp):
        """Compute total unbonded energy (HPS LJ + Coulomb) for all 529 pairs.

        Args:
            r: Radial distance(s).
            pair_charges: Charge pairs, shape (529, 2).
            salt_conc: Salt concentration in mM.
            temp: Temperature in Kelvin.

        Returns:
            Total unbonded energies, shape (529,) or (529, len(r)).
        """
        hps_lj_vals = self._mapped_hps_lj(r, self.sigma_flattened, self.lambdas_flattened)
        coul_vals = self._mapped_calvados_coul(r, pair_charges, self.debye_rc, temp, salt_conc)
        return hps_lj_vals + coul_vals

class KHD_model:
    """Kim-Hummer-D coarse-grained force field.

    Combines Ashbaugh-Hatch short-range potentials (with sign-dependent
    lambda) and Debye-Huckel electrostatics.
    """

    def __init__(self):
        """Initialize the KH-D model, loading parameters from file."""
        self.fpath = ROOT / 'ff_params' / 'KH-D.txt'
        self.eps_table, self.eps_flattened, self.sigma_flattened, self.lambda_flattened = read_KH_D(self.fpath)
        self.eps_flattened_positive = jnp.where(self.eps_flattened < 0.0, -self.eps_flattened, self.eps_flattened)
        self._mapped_coulomb = vmap(self.coulomb_internal_func, in_axes = (None, 0, None, None, None))
        self._mapped_ashbaugh_hatch = vmap(self.ashbaugh_hatch_internal_func, in_axes = (None, 0, 0, 0))
        self.debye_kappa = 0.131


    def construct_eps_matrix(self):
        """Return the raw 23x23 epsilon table.

        Returns:
            Epsilon matrix of shape (23, 23).
        """
        return self.eps_table

    def calculate_charges(self, pH = 7, pH_mode = 'normal'):
        """Compute per-residue-type charges for the KH-D model.

        Args:
            pH: Solution pH (used only in 'pH' mode).
            pH_mode: 'normal' or 'pH'.

        Returns:
            Charge array of shape (23,), ordered by RES_ALPHA.
        """
        charges_dict = {
            'D': -1.0,
            'E': -1.0,
            'K': 1.0,
            'R': 1.0,
            'X': 0.0,
            'Z': -2.0,
            'B': -2.0
        }
        if pH_mode == 'normal':
            return jnp.array([charges_dict.get(res, 0.0) for res in RES_ALPHA])
        elif pH_mode == 'pH':
            pKa_values = {
            'N-terminus': 9.69,
            'C-terminus': 2.34,
            'D': 3.86,
            'E': 4.25,
            'C': 8.33,
            'Y': 10.07,
            'H': 6.00,
            'K': 10.54,
            'R': 12.48,
            }

            charges = {
                'N-terminus': 1.0 / (1.0 + 10**(pH - pKa_values['N-terminus'])),
                'C-terminus': -1.0 / (1.0 + 10**(pKa_values['C-terminus'] - pH)),
                'D': -1.0 / (1.0 + 10**(pKa_values['D'] - pH)),
                'E': -1.0 / (1.0 + 10**(pKa_values['E'] - pH)),
                'C': -1.0 / (1.0 + 10**(pKa_values['C'] - pH)),
                'Y': -1.0 / (1.0 + 10**(pKa_values['Y'] - pH)),
                'H': 1.0 / (1.0 + 10**(pH - pKa_values['H'])),
                'K': 1.0 / (1.0 + 10**(pH - pKa_values['K'])),
                'R': 1.0 / (1.0 + 10**(pH - pKa_values['R'])),
                'B': -2.0,
                'Z': -2.0,
                'X': 0.0
            }
            return jnp.array([charges.get(res, 0.0) for res in RES_ALPHA])
        else:
            raise ValueError(f"Invalid pH mode: {pH_mode}")
    
    def create_pair_charges(self, charges):
        """Build all 529 charge pairs from per-residue charges.

        Args:
            charges: Per-residue-type charges, shape (23,).

        Returns:
            Array of shape (529, 2).
        """
        pair_charges = list()
        for i in range(NUM_RESIDUES):
            for j in range(NUM_RESIDUES):
                pair_charges.append([charges[i], charges[j]])
        return jnp.array(pair_charges)

    @functools.partial(jit, static_argnums = (0,))
    def coulomb_internal_func(self, r, qij, r_c, temp, salt_conc):
        """Evaluate screened Coulomb potential for KH-D model.

        Uses a fixed dielectric of 80 and salt-dependent Debye screening.

        Args:
            r: Interparticle distance.
            qij: Charge pair [qi, qj].
            r_c: Electrostatic cutoff distance.
            temp: Temperature in Kelvin.
            salt_conc: Salt concentration in mM.

        Returns:
            Electrostatic energy at r, zero beyond r_c.
        """
        qi = qij[0]
        qj = qij[1]
        k = self.debye_kappa * jnp.sqrt(salt_conc/150)
        C = 5.513725184e-22 # (1/(4*pi*eps0)  in units of kcal* (Armstrong)/(electron^^2)
        fac1 =  249.4 - 0.788 * temp + 0.00072 * temp**2
        fac2 = 1 - 0.2551 * (salt_conc/1000) + 0.05151 * (salt_conc/1000)**2 - 0.006889 * (salt_conc/1000)**3
        epsilon = fac1 * fac2
        epsilon = 80.0
        val = (C * qi * qj) / (epsilon * r) * jnp.exp(-k * r)
        val *= 6.022e23
        return jnp.where(r < r_c, val, 0.0)

    @functools.partial(jit, static_argnums = (0,))
    def ashbaugh_hatch_internal_func(self, r, eps, sigma, lambda_):
        """Evaluate the Ashbaugh-Hatch potential for KH-D.

        Args:
            r: Interparticle distance.
            eps: Well depth (absolute value).
            sigma: Length scale.
            lambda_: Sign-dependent hydrophobicity (+1 repulsive, -1 attractive).

        Returns:
            Potential energy at r.
        """
        potential_close = eps*(4*((sigma/r)**12-(sigma/r)**6)+1-lambda_)
        potential_far = lambda_*eps*4*((sigma/r)**12-(sigma/r)**6)
        return jnp.where(r <= 2**(1/6) * sigma, potential_close, potential_far)

    @functools.partial(jit, static_argnums = (0,))
    def calculate_unbonded(self, r, pair_charges, salt_conc, temp):
        """Compute total unbonded energy (Ashbaugh-Hatch + Coulomb) for all 529 pairs.

        Args:
            r: Radial distance(s).
            pair_charges: Charge pairs, shape (529, 2).
            salt_conc: Salt concentration in mM.
            temp: Temperature in Kelvin.

        Returns:
            Total unbonded energies, shape (529,) or (529, len(r)).
        """
        ah_vals = self._mapped_ashbaugh_hatch(r, self.eps_flattened_positive, self.sigma_flattened, self.lambda_flattened)
        coul_vals = self._mapped_coulomb(r, pair_charges, DEBYE_RC, temp, salt_conc)
        return ah_vals + coul_vals

def seq_to_one_hot(seq):
    """Convert an amino acid sequence string to a one-hot encoded array.

    Args:
        seq: Amino acid sequence string using RES_ALPHA characters.

    Returns:
        One-hot array of shape (len(seq), 23).
    """
    all_vecs = list()
    for res in seq:
        res_idx = RES_ALPHA.index(res)
        res_vec = onp.zeros(NUM_RESIDUES)
        res_vec[res_idx] = 1.0
        all_vecs.append(res_vec)
    return jnp.array(all_vecs)
