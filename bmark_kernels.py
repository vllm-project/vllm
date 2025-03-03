import itertools
import pytest

import jax
from jax import numpy as jnp
from vllm.lora.ops.xla_ops.pallas import bgmv

def create_tensors(T, D, L, N):
    """
    Inputs: (All integers)
        T: Total number of tokens
        D: Input dim
        L: LoRA Dim
        N: N LoRAs
    
    Outputs:
        inputs:     jax.Array - shape (T, D)
        loras:      jax.Array - shape (N, 1, L, D)
        idxs:       jax.Array - shape (T, ) - all values must be in [0, N)
        
        ref_output: jax.Array - shape (T, L) - inputs @ loras[idxs].T
    """
    inputs = jax.random.normal(jax.random.PRNGKey(0), (T, D))
    loras = jax.random.normal(jax.random.PRNGKey(0), (N, 1, L, D))
    idxs = jax.random.randint(jax.random.PRNGKey(0),
                              shape=(T, ),
                              minval=0,
                              maxval=N)


    return inputs, loras, idxs

def ref_bgmv(inputs, loras, idxs):
    return jnp.einsum("td,__ld->tl", inputs, loras[idxs])

SEQ_LENS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
HIDDEN_DIM = [1024, 2048, 3072, 4096]
LORA_RANKS = [8, 16, 32, 64, 128, 256]
N_LORAS = [1, 2, 4, 8, 16, 32]


@pytest.mark.parametrize("T,D,L,N", itertools.product(SEQ_LENS, HIDDEN_DIM, LORA_RANKS, N_LORAS))
@pytest.mark.parametrize("func", [bgmv, ref_bgmv])
def test_bgmv_benchmark(benchmark, T, D, L, N):
    inputs, loras, idxs = create_tensors(T, D, L, N)

    benchmark.pedantic(ref_bgmv, args=(inputs, loras, idxs), rounds=10, warmup_rounds=5, iterations=10)
