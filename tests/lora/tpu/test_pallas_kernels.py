# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bgmv import bgmv

N_TOKENS = [
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072
]
HIDDEN_SIZES = [128, 256, 512, 896, 1024, 2048, 4096, 8192, 8320]

DTYPES = [jnp.float16, jnp.bfloat16]
NUM_LORA = [1, 2, 4, 8, 16, 32]
RANKS = [8, 16, 32, 64, 128]


def generate_test_data(T, D, L, N, seed, dtype=jnp.float32):
    """
    Generates debug tensors for testing.
    """
    inputs = jax.random.normal(jax.random.PRNGKey(seed), (T, D))
    loras = jax.random.normal(jax.random.PRNGKey(seed), (N, 1, L, D))
    idxs = jax.random.randint(jax.random.PRNGKey(seed),
                              shape=(T, ),
                              minval=0,
                              maxval=N)

    ref_output = jnp.einsum("td,t_ld->tl", inputs, loras[idxs])
    return inputs, loras, idxs, ref_output


# Parameterize tests with various shapes and dtypes
@pytest.mark.parametrize("T", N_TOKENS)
@pytest.mark.parametrize("D", HIDDEN_SIZES)
@pytest.mark.parametrize("L", RANKS)
@pytest.mark.parametrize("N", NUM_LORA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.parametrize("seed", [0])
def test_bgmv(T, D, L, N, dtype, op_type, seed):
    inputs, loras, idxs, ref_output = generate_test_data(
        T, D, L, N, seed, dtype)

    # Run bgmv
    match op_type:
        case "expand":
            output = bgmv(inputs, loras, idxs)  # TODO: Specialise
        case "shrink":
            output = bgmv(inputs, loras, idxs)

    # Make sure we have no NaNs
    assert jnp.isnan(output).sum() == 0

    # Compare with reference output
    np.testing.assert_allclose(output, ref_output, rtol=1e-3, atol=1e-3)
