"""
MetaPredict V3 - JAX Implementation
====================================

Standalone JAX implementation of MetaPredict V3 for intrinsic disorder prediction.

This is a pure JAX implementation with no PyTorch dependencies. It provides:
- Fast predictions with JIT compilation
- Batch processing support  
- Gradient computation capability for optimization
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List, NamedTuple, Optional
from functools import partial
import pickle
import os


# V3 Network Configuration
class NetworkConfig(NamedTuple):
    """Network architecture configuration"""
    input_size: int
    hidden_size: int
    num_layers: int
    num_classes: int
    use_layer_norm: bool
    num_linear_layers: int


# V3 is the most advanced network with layer normalization
V3_CONFIG = NetworkConfig(
    input_size=20,
    hidden_size=52,
    num_layers=2,
    num_classes=1,
    use_layer_norm=True,
    num_linear_layers=1,
)


# Amino acid to index mapping
AA_TO_INDEX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

RES_ALPHA_MP = "ACDEFGHIKLMNPQRSTVWY"
RES_ALPHA = "MGKTRADEYVLQWFSHNPCIXZB"
res_idx_mapper = list()
for res_idx, res in enumerate(RES_ALPHA_MP):
    orig_idx = RES_ALPHA.index(res)
    res_idx_mapper.append(orig_idx)
res_idx_mapper = jnp.array(res_idx_mapper)


def pseq_to_mp_pseq(pseq: jnp.ndarray) -> jnp.ndarray:
    n_cols = pseq.shape[1]
    return pseq.at[:, jnp.arange(n_cols)].set(pseq[:, res_idx_mapper])

def one_hot_encode_sequence(sequence: str) -> jnp.ndarray:
    """
    One-hot encode an amino acid sequence.
    
    Args:
        sequence: Amino acid sequence string
        
    Returns:
        One-hot encoded array of shape (seq_length, 20)
    """
    seq_length = len(sequence)
    encoding = np.zeros((seq_length, 20), dtype=np.float32)
    
    for i, aa in enumerate(sequence.upper()):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1.0
    
    return jnp.array(encoding)


def lstm_cell_forward(x, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh):
    """
    Forward pass through a single LSTM cell.
    
    Args:
        x: Input of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        c_prev: Previous cell state of shape (hidden_size,)
        weight_ih: Input-to-hidden weight matrix
        weight_hh: Hidden-to-hidden weight matrix
        bias_ih: Input-to-hidden bias
        bias_hh: Hidden-to-hidden bias
        
    Returns:
        Tuple of (new_hidden_state, new_cell_state)
    """
    # Compute gates
    gates = jnp.matmul(weight_ih, x) + bias_ih + jnp.matmul(weight_hh, h_prev) + bias_hh
    
    hidden_size = h_prev.shape[0]
    
    # Split into individual gates
    i_gate = jax.nn.sigmoid(gates[0:hidden_size])
    f_gate = jax.nn.sigmoid(gates[hidden_size:2*hidden_size])
    g_gate = jnp.tanh(gates[2*hidden_size:3*hidden_size])
    o_gate = jax.nn.sigmoid(gates[3*hidden_size:4*hidden_size])
    
    # Update cell state and hidden state
    c_new = f_gate * c_prev + i_gate * g_gate
    h_new = o_gate * jnp.tanh(c_new)
    
    return h_new, c_new

def process_sequence_direction(sequence, params, hidden_size):
    h0 = jnp.zeros(hidden_size)
    c0 = jnp.zeros(hidden_size)

    def step_fn(carry, x_t):
        h, c = carry
        h_new, c_new = lstm_cell_forward(
            x_t, h, c,
            params['weight_ih'],
            params['weight_hh'],
            params['bias_ih'],
            params['bias_hh']
        )
        return (h_new, c_new), h_new

    (_, _), outputs = jax.lax.scan(step_fn, (h0, c0), sequence)
    return outputs

# def process_sequence_direction(sequence, params, hidden_size):
#     """
#     Process sequence in one direction (forward or backward).
    
#     Args:
#         sequence: Input sequence of shape (seq_length, input_size)
#         params: Dictionary with weight_ih, weight_hh, bias_ih, bias_hh
#         hidden_size: Number of hidden units
        
#     Returns:
#         Hidden states of shape (seq_length, hidden_size)
#     """
#     seq_length = sequence.shape[0]
    
#     # Initialize hidden and cell states
#     h = jnp.zeros(hidden_size)
#     c = jnp.zeros(hidden_size)
    
#     outputs = []
    
#     for t in range(seq_length):
#         h, c = lstm_cell_forward(
#             sequence[t],
#             h, c,
#             params['weight_ih'],
#             params['weight_hh'],
#             params['bias_ih'],
#             params['bias_hh']
#         )
#         outputs.append(h)
    
#     return jnp.stack(outputs)


def bidirectional_lstm_layer(x, layer_params, hidden_size):
    """
    Bidirectional LSTM layer processing.
    
    Args:
        x: Input of shape (seq_length, input_size)
        layer_params: Dictionary with 'forward' and 'backward' parameters
        hidden_size: Number of hidden units
        
    Returns:
        Concatenated forward and backward outputs of shape (seq_length, 2*hidden_size)
    """
    # Forward pass
    forward_out = process_sequence_direction(
        x, layer_params['forward'], hidden_size
    )
    
    # Backward pass (reverse the sequence)
    x_reversed = jnp.flip(x, axis=0)
    backward_out = process_sequence_direction(
        x_reversed, layer_params['backward'], hidden_size
    )
    backward_out = jnp.flip(backward_out, axis=0)
    
    # Concatenate forward and backward
    return jnp.concatenate([forward_out, backward_out], axis=1)


def apply_layer_norm(x, gamma, beta):
    """
    Apply layer normalization.
    
    Args:
        x: Input of shape (seq_length, features)
        gamma: Scale parameter
        beta: Shift parameter
        
    Returns:
        Normalized output
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
    return gamma * x_norm + beta


@partial(jax.jit, static_argnames=['config'])
def forward_pass(x, params, config):
    """
    Forward pass through the network.
    
    Args:
        x: One-hot encoded sequence of shape (seq_length, 20)
        params: Network parameters dictionary
        config: NetworkConfig (must be hashable for JIT)
        
    Returns:
        Disorder predictions of shape (seq_length,)
    """
    # Process through LSTM layers
    lstm_out = x
    
    for layer_idx in range(config.num_layers):
        layer_params = params['lstm'][layer_idx]
        
        if layer_idx == 0:
            input_size = config.input_size
        else:
            input_size = config.hidden_size * 2
        
        lstm_out = bidirectional_lstm_layer(
            lstm_out,
            layer_params,
            config.hidden_size
        )
    
    # Apply layer normalization if specified
    if config.use_layer_norm:
        lstm_out = apply_layer_norm(
            lstm_out,
            params['layer_norm']['gamma'],
            params['layer_norm']['beta']
        )
    
    # Process through linear layers
    output = lstm_out
    for i in range(config.num_linear_layers):
        linear_params = params['linear_layers'][i]
        output = jnp.matmul(output, linear_params['weight'].T) + linear_params['bias']
    
    # Squeeze to get shape (seq_length,)
    return output.squeeze(-1)


@partial(jax.jit, static_argnames=['config'])
def forward_pass_batch(x_batch, params, config):
    """
    Batched forward pass through the network using vmap.

    Args:
        x_batch: One-hot encoded batch of shape (batch_size, seq_length, 20)
        params: Network parameters dictionary
        config: NetworkConfig (must be hashable for JIT)

    Returns:
        Disorder predictions of shape (batch_size, seq_length)
    """
    return jax.vmap(
        lambda x: forward_pass(x, params, config),
        in_axes=0,
        out_axes=0
    )(x_batch)


class MetapredictJAX:
    """
    MetaPredict V3 disorder predictor in JAX.
    
    This class provides a clean interface for disorder prediction
    with JIT-compiled operations for fast inference.
    """
    
    def __init__(self, params: dict, config: NetworkConfig = V3_CONFIG):
        """
        Initialize the predictor.
        
        Args:
            params: Network parameters dictionary
            config: Network configuration (default: V3_CONFIG)
        """
        self.params = params
        self.config = config
        self.version = 'V3'
    
    def predict(self, pseq: jnp.ndarray, normalized: bool = True, 
                round_values: bool = True) -> Tuple[jnp.ndarray, float]:
        """
        Predict disorder for a protein sequence.
        
        Args:
            pseq: Protein sequence array of shape (seq_length, 20)
            normalized: Whether to clip scores to [0, 1] range
            round_values: Whether to round values to 4 decimal places
            
        Returns:
            Tuple of (disorder_profile, average_score)
            - disorder_profile: Per-residue disorder scores of shape (seq_length,)
            - average_score: Mean disorder score (scalar)
        """
        # Encode sequence
        x = pseq.squeeze(0)        
        # Forward pass
        disorder_profile = forward_pass(x, self.params, self.config)
        
        # Post-processing
        if normalized:
            disorder_profile = jnp.clip(disorder_profile, 0.0, 1.0)
        
        if round_values:
            disorder_profile = jnp.round(disorder_profile, 4)
        
        # Calculate average score
        avg_score = jnp.mean(disorder_profile).astype(jnp.float32)
        
        if round_values:
            avg_score = round(avg_score, 4)
        
        return disorder_profile, avg_score
    
    def predict_for_optimization(self, pseq: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.float32]:
        x = pseq.squeeze(0)
        disorder_profile = forward_pass(x, self.params, self.config)
        disorder_profile = jnp.clip(disorder_profile, 0.0, 1.0)
        avg_score = jnp.mean(disorder_profile)
        return disorder_profile, avg_score.astype(jnp.float32)

    def return_metapredict_fn(self):
        return self.predict_for_optimization
    
    def predict_batch(self, sequences: List[str], normalized: bool = True,
                      round_values: bool = True,
                      use_vmap: bool = True) -> List[Tuple[jnp.ndarray, float]]:
        """
        Predict disorder for multiple sequences.
        
        Args:
            sequences: List of amino acid sequence strings
            normalized: Whether to clip scores to [0, 1] range
            round_values: Whether to round values to 4 decimal places
            use_vmap: If True, use vmap-based batch inference for same-length
                sequence groups. If False, use per-sequence loop inference.
            
        Returns:
            List of (disorder_profile, average_score) tuples
        """
        if not sequences:
            return []

        if not use_vmap:
            return [self.predict(seq, normalized, round_values) for seq in sequences]

        # Group by sequence length so each vmap call has a single static shape.
        length_groups = {}
        for idx, seq in enumerate(sequences):
            length_groups.setdefault(len(seq), []).append((idx, seq))

        batch_results: List[Optional[Tuple[jnp.ndarray, float]]] = [None] * len(sequences)

        for _, indexed_seqs in length_groups.items():
            x_batch = jnp.stack(
                [one_hot_encode_sequence(seq) for _, seq in indexed_seqs],
                axis=0
            )

            disorder_batch = forward_pass_batch(x_batch, self.params, self.config)

            if normalized:
                disorder_batch = jnp.clip(disorder_batch, 0.0, 1.0)

            if round_values:
                disorder_batch = jnp.round(disorder_batch, 4)

            avg_batch = jnp.mean(disorder_batch, axis=1)

            if round_values:
                avg_batch = jnp.round(avg_batch, 4)

            for local_i, (orig_idx, _) in enumerate(indexed_seqs):
                batch_results[orig_idx] = (
                    disorder_batch[local_i],
                    float(avg_batch[local_i])
                )

        if any(x is None for x in batch_results):
            raise RuntimeError("Batch prediction failed to populate all outputs")

        return batch_results  # type: ignore[return-value]
    
    def save_params(self, filepath: str):
        """
        Save parameters to a pickle file.
        
        Args:
            filepath: Path to save the parameters
        """
        save_data = {
            'params': self.params,
            'config': self.config,
            'version': self.version
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load_params(cls, filepath: Optional[str] = None):
        """
        Load parameters from a pickle file.
        
        Args:
            filepath: Path to the parameter file. If None, searches for 
                     metapredict_V3_params.pkl in the data directory.
            
        Returns:
            MetapredictJAX instance
        """
        if filepath is None:
            # Auto-locate the weights file
            package_dir = os.path.dirname(__file__)
            default_paths = [
                os.path.join(package_dir, 'data', 'metapredict_V3_params.pkl'),
                os.path.join(package_dir, '..', 'metapredict_V3_params.pkl'),
                'metapredict_V3_params.pkl',
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    filepath = path
                    break
            
            if filepath is None:
                raise FileNotFoundError(
                    "Could not find metapredict_V3_params.pkl. "
                    f"Searched: {default_paths}"
                )
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        return cls(
            params=save_data['params'],
            config=save_data.get('config', V3_CONFIG)
        )


# Convenience function for quick predictions
def predict_disorder(sequence: str, model_path: Optional[str] = None) -> Tuple[jnp.ndarray, float]:
    """
    Quick disorder prediction for a sequence.
    
    Args:
        sequence: Amino acid sequence string
        model_path: Path to the model parameters file. If None, auto-locates.
        
    Returns:
        Tuple of (disorder_profile, average_score)
    """
    model = MetapredictJAX.load_params(model_path)
    return model.predict(sequence)


if __name__ == '__main__':
    print("MetaPredict V3 - JAX Implementation")
    print("=" * 60)
    print("\nThis is a standalone JAX implementation of MetaPredict V3.")
    print("To use it, load the model and make predictions:")
    print("\n  from metapredict_jax import MetapredictJAX")
    print("  model = MetapredictJAX.load_params('metapredict_V3_params.pkl')")
    print("  disorder, avg = model.predict('MKLAVLGVAGIAGIASAALGGKQ')")
    print("\n" + "=" * 60)
