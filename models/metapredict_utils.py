import jax
import jax.numpy as jnp
import numpy as onp
import random
from models.ff_models import RES_ALPHA
import torch

import metapredict
from metapredict.backend import predictor, encode_sequence
from wrap_torch2jax import torch2jax_with_vjp

RES_ALPHA = RES_ALPHA[:20]

def get_rand_seq(n):
    return ''.join(random.choices(RES_ALPHA, k=n))

RES_ALPHA_MP = "ACDEFGHIKLMNPQRSTVWY"
res_idx_mapper = list()
for res_idx, res in enumerate(RES_ALPHA_MP):
    orig_idx = RES_ALPHA.index(res)
    res_idx_mapper.append(orig_idx)
res_idx_mapper = jnp.array(res_idx_mapper)


def pseq_to_mp_pseq(pseq: jnp.ndarray) -> jnp.ndarray:
    n_cols = pseq.shape[1]
    return pseq.at[:, jnp.arange(n_cols)].set(pseq[:, res_idx_mapper])

def get_metapredict_fn(n):
   
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_jax = jax.devices(device.type)[0]
    ex_seq = get_rand_seq(n).upper()
    
    # Initialize predictor
    disorder_obj = predictor.predict(ex_seq)
    model_name = list(predictor.loaded_models.keys())[0]
    
    network = predictor.loaded_models[model_name]
    network = network.to(device)
    network.eval() 

    ex_seq_tensor = encode_sequence.one_hot(ex_seq).view(1, n, -1)
    ex_seq_tensor = ex_seq_tensor.to(device)

    jax_mp_fn = torch2jax_with_vjp(network, ex_seq_tensor.float(), use_torch_vjp=False)

    ex_jax_input = jnp.array(ex_seq_tensor.cpu().numpy())
    try:
        jax_mp_fn(ex_jax_input.astype(jnp.float32))
    except:
        raise RuntimeError("Failed to call JAX metapredict function")

    return jax_mp_fn


