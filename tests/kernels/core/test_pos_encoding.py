# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable
from itertools import product

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


def _get_yarn_rope(factor: float, **rope_parameter_overrides):
    rope_parameters = {
        "rope_type": "yarn",
        "rope_theta": 10000,
        "factor": factor,
        "original_max_position_embeddings": 512,
    }
    rope_parameters.update(rope_parameter_overrides)
    return get_rope(
        head_size=64,
        max_position=int(512 * factor),
        is_neox_style=True,
        rope_parameters=rope_parameters,
        dtype=torch.float32,
    )


def _yarn_output_at_position_zero(rope) -> torch.Tensor:
    positions = torch.zeros(1, dtype=torch.long)
    query = torch.ones(1, 64, dtype=torch.float32)
    out_query, _ = rope.forward_native(positions, query)
    return out_query


def _assert_yarn_magnitude(expected: float, **rope_parameter_overrides) -> None:
    rope = _get_yarn_rope(factor=16.0, **rope_parameter_overrides)
    actual = _yarn_output_at_position_zero(rope)
    torch.testing.assert_close(actual, torch.full_like(actual, expected))


@pytest.mark.parametrize("attention_factor", [1.0, 0.5])
@torch.inference_mode()
def test_yarn_explicit_attention_factor(default_vllm_config, attention_factor):
    _assert_yarn_magnitude(attention_factor, attention_factor=attention_factor)


@torch.inference_mode()
def test_yarn_default_mscale(default_vllm_config):
    _assert_yarn_magnitude(0.1 * math.log(16.0) + 1.0)


@torch.inference_mode()
def test_yarn_mscale_ratio(default_vllm_config):
    expected = (0.2 * math.log(16.0) + 1.0) / (0.1 * math.log(16.0) + 1.0)
    _assert_yarn_magnitude(expected, mscale=2.0, mscale_all_dim=1.0)


@torch.inference_mode()
def test_yarn_equal_mscale_ratio(default_vllm_config):
    _assert_yarn_magnitude(1.0, mscale=1.0, mscale_all_dim=1.0)


@torch.inference_mode()
def test_yarn_mscale_ratio_preserves_attn_factor(default_vllm_config):
    _assert_yarn_magnitude(
        0.5,
        mscale=1.0,
        mscale_all_dim=1.0,
        attn_factor=0.5,
    )


@torch.inference_mode()
def test_yarn_attention_factor_takes_precedence(default_vllm_config):
    _assert_yarn_magnitude(
        0.5,
        attention_factor=0.5,
        mscale=2.0,
        mscale_all_dim=1.0,
        attn_factor=0.25,
    )


@torch.inference_mode()
def test_yarn_disabled_scaling_ignores_mscale_ratio(default_vllm_config):
    _assert_yarn_magnitude(
        0.5,
        apply_yarn_scaling=False,
        mscale=2.0,
        mscale_all_dim=1.0,
        attn_factor=0.5,
    )


@torch.inference_mode()
def test_yarn_explicit_attention_factor_overrides_disabled_scaling(
    default_vllm_config,
):
    _assert_yarn_magnitude(
        0.25,
        attention_factor=0.25,
        apply_yarn_scaling=False,
        attn_factor=0.5,
    )
