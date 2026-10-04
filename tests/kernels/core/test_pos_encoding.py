# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from itertools import product
from unittest.mock import patch

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.utils.torch_utils import set_random_seed

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.bfloat16, torch.float]
HEAD_SIZES = [64, 80, 120, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [17]  # Arbitrary values for testing
BATCH_SIZES = [5]  # Arbitrary values for testing
SEQ_LENS = [11, 8192]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]
USE_KEY = [True, False]


def _get_flat_tensor_shape(
    batch_size: int, seq_len: int, num_heads: int, head_size: int
) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads * head_size)


# For testing sliced tensors
def _get_padded_tensor_shape(
    batch_size: int, seq_len: int, num_heads: int, head_size: int
) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads, head_size + 64)


def _get_batch_tensor_shape(
    batch_size: int, seq_len: int, num_heads: int, head_size: int
) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads, head_size)


TENSORS_SHAPES_FN = [
    _get_batch_tensor_shape,
    _get_flat_tensor_shape,
    _get_padded_tensor_shape,
]


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("tensor_shape_fn", TENSORS_SHAPES_FN)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_key", USE_KEY)
@torch.inference_mode()
def test_rotary_embedding(
    default_vllm_config,
    is_neox_style: bool,
    tensor_shape_fn: Callable[[int, int, int, int], tuple[int, ...]],
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: int | None,
    dtype: torch.dtype,
    seed: int,
    device: str,
    use_key: bool,
    max_position: int = 8192,
    rope_theta: float = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size

    set_random_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": rope_theta,
        "partial_rotary_factor": rotary_dim / head_size,
    }
    rope = get_rope(head_size, max_position, is_neox_style, rope_parameters)
    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query_shape = tensor_shape_fn(batch_size, seq_len, num_heads, head_size)
    # slice tensor if required, noop otherwise
    query = torch.randn(query_shape, dtype=dtype)[..., :head_size]
    key = torch.randn_like(query)[..., :head_size] if use_key else None

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope.forward_native(positions, query, key)
    out_query, out_key = rope.forward(positions, query, key)
    # Compare the results.
    torch.testing.assert_close(
        out_query,
        ref_query,
        atol=get_default_atol(out_query),
        rtol=get_default_rtol(out_query),
    )
    if use_key:
        torch.testing.assert_close(
            out_key,
            ref_key,
            atol=get_default_atol(out_key),
            rtol=get_default_rtol(out_key),
        )
    else:
        assert ref_key is None and out_key is None, "expected returned key to be None"


@torch.inference_mode()
def test_rope_module_cache(default_vllm_config):
    MAX_POSITIONS = [123, 1234]
    ROPE_THETAS = [10000, 1000000]
    ROPE_PARAMETERS = (
        {"rope_type": "default"},
        {"rope_type": "linear", "factor": (1,)},
        {"rope_type": "dynamic", "factor": 1},
    )
    settings = (
        HEAD_SIZES,
        ROTARY_DIMS,
        MAX_POSITIONS,
        ROPE_THETAS,
        IS_NEOX_STYLE,
        ROPE_PARAMETERS,
        DTYPES,
    )
    rope_setting_id_map: dict[str, int] = {}
    for setting in product(*settings):
        (
            head_size,
            rotary_dim,
            max_position,
            rope_theta,
            is_neox_style,
            rope_parameters,
            dtype,
        ) = setting
        if rotary_dim is None:
            rotary_dim = head_size
        rope_parameters["rope_theta"] = rope_theta
        rope_parameters["partial_rotary_factor"] = rotary_dim / head_size
        rope = get_rope(
            head_size,
            max_position,
            is_neox_style,
            rope_parameters,
            dtype,
        )
        # different settings cannot share the same rope module
        assert id(rope) not in rope_setting_id_map.values()
        assert all(x.dtype == dtype for x in rope.buffers())
        assert all(x.dtype == dtype for x in rope.parameters())
        rope_setting_id_map[str(setting)] = id(rope)

    for setting in product(*settings):
        (
            head_size,
            rotary_dim,
            max_position,
            rope_theta,
            is_neox_style,
            rope_parameters,
            dtype,
        ) = setting
        if rotary_dim is None:
            rotary_dim = head_size
        rope_parameters["rope_theta"] = rope_theta
        rope_parameters["partial_rotary_factor"] = rotary_dim / head_size
        rope = get_rope(
            head_size,
            max_position,
            is_neox_style,
            rope_parameters,
            dtype,
        )
        # check if cache take effect
        assert id(rope) == rope_setting_id_map[str(setting)]


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.half])
@pytest.mark.parametrize("use_key", USE_KEY)
@torch.inference_mode()
def test_deepseek_scaling_rotary_embedding_cuda_kernel(
    default_vllm_config,
    is_neox_style: bool,
    dtype: torch.dtype,
    use_key: bool,
) -> None:
    """The fused-kernel path of DeepseekScalingRotaryEmbedding must match
    forward_native (it previously fell back to eager without FlashInfer).

    float32 compares the two directly. For half dtypes the kernel computes
    in fp32 while the reference rounds through the half dtype, so both are
    compared against an fp32-reference ground truth instead, asserting the
    kernel is at least as accurate as the eager fallback it replaces.
    """
    import vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope as dsr

    set_random_seed(0)
    torch.set_default_device("cuda")

    # DeepSeek-V2 rope_scaling: qk_rope_head_dim=64, YaRN factor 40.
    kwargs: dict = dict(
        head_size=64,
        rotary_dim=64,
        max_position_embeddings=4096,
        base=10000,
        is_neox_style=is_neox_style,
        scaling_factor=40.0,
        extrapolation_factor=1,
        attn_factor=1,
        beta_fast=32,
        beta_slow=1,
        mscale=0.707,
        mscale_all_dim=0.707,
    )

    def make_rope(rope_dtype: torch.dtype) -> dsr.DeepseekScalingRotaryEmbedding:
        # Force the ops.rotary_embedding branch: without FlashInfer the
        # cos/sin cache is kept in the model dtype, matching a
        # FlashInfer-less deployment.
        with patch.object(dsr, "has_flashinfer", lambda: False):
            rope = dsr.DeepseekScalingRotaryEmbedding(dtype=rope_dtype, **kwargs)
        assert not rope.use_flashinfer
        return rope

    num_tokens, num_heads, num_kv_heads = 64, 16, 1
    positions = torch.randint(0, int(4096 * 40) - 1, (num_tokens,))
    query32 = torch.randn(num_tokens, num_heads, 64, dtype=torch.float32)
    key32 = (
        torch.randn(num_tokens, num_kv_heads, 64, dtype=torch.float32)
        if use_key
        else None
    )

    rope = make_rope(dtype)
    query = query32.to(dtype)
    key = key32.to(dtype) if key32 is not None else None

    # forward_native asserts key is not None, so feed it a dummy key when
    # exercising the kernel's key=None support and compare queries only.
    ref_key_in = key.clone() if key is not None else torch.zeros_like(query[:, :1])
    ref_query, ref_key = rope.forward_native(positions, query.clone(), ref_key_in)
    out_query, out_key = rope.forward_cuda(
        positions, query.clone(), key.clone() if key is not None else None
    )
    if not use_key:
        assert out_key is None
        ref_key = None

    if dtype == torch.float32:
        torch.testing.assert_close(out_query, ref_query, atol=1e-5, rtol=1e-5)
        if use_key:
            torch.testing.assert_close(out_key, ref_key, atol=1e-5, rtol=1e-5)
        return

    # Half dtypes: compare both paths against the fp32 ground truth.
    rope32 = make_rope(torch.float32)
    truth_key_in = (
        key32.clone() if key32 is not None else torch.zeros_like(query32[:, :1])
    )
    truth_query, truth_key = rope32.forward_native(
        positions, query32.clone(), truth_key_in
    )

    kernel_err = (out_query.float() - truth_query).abs().mean()
    native_err = (ref_query.float() - truth_query).abs().mean()
    assert kernel_err <= native_err * 1.05, (
        f"fused kernel mean error {kernel_err} vs native {native_err}"
    )
    if out_key is not None and ref_key is not None:
        kernel_err_k = (out_key.float() - truth_key).abs().mean()
        native_err_k = (ref_key.float() - truth_key).abs().mean()
        assert kernel_err_k <= native_err_k * 1.05
