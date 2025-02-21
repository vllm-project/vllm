# SPDX-License-Identifier: Apache-2.0

from itertools import accumulate, product
from typing import Callable, Dict, List, Optional

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform

from .allclose_default import get_default_atol, get_default_rtol

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.half, torch.bfloat16, torch.float]
HEAD_SIZES = [64, 80, 112, 120, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [17]  # Arbitrary values for testing
BATCH_SIZES = [5]  # Arbitrary values for testing
SEQ_LENS = [11, 8192]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


def _get_flat_tensor_shape(batch_size: int, seq_len: int, num_heads: int,
                           head_size: int) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads * head_size)


def _get_batch_tensor_shape(batch_size: int, seq_len: int, num_heads: int,
                            head_size: int) -> tuple[int, ...]:
    return (batch_size, seq_len, num_heads, head_size)


TENSORS_SHAPES_FN = [_get_batch_tensor_shape, _get_flat_tensor_shape]


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
@torch.inference_mode()
def test_rotary_embedding(
    is_neox_style: bool,
    tensor_shape_fn: Callable[[int, int, int, int], tuple[int]],
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size

    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query_shape = tensor_shape_fn(batch_size, seq_len, num_heads, head_size)
    query = torch.randn(query_shape, dtype=dtype)
    key = torch.randn_like(query)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope.forward_native(positions, query, key)
    out_query, out_key = rope.forward(positions, query, key)
    # Compare the results.
    torch.testing.assert_close(out_query,
                               ref_query,
                               atol=get_default_atol(out_query),
                               rtol=get_default_rtol(out_query))
    torch.testing.assert_close(out_key,
                               ref_key,
                               atol=get_default_atol(out_key),
                               rtol=get_default_rtol(out_key))


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
@torch.inference_mode()
def test_batched_rotary_embedding(
    is_neox_style: bool,
    tensor_shape_fn: Callable[[int, int, int, int], tuple[int]],
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style, {
        "rope_type": "linear",
        "factor": (1, )
    })
    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query_shape = tensor_shape_fn(batch_size, seq_len, num_heads, head_size)
    query = torch.randn(query_shape, dtype=dtype)
    key = torch.randn_like(query)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope.forward_native(positions, query, key)
    out_query, out_key = rope.forward(positions,
                                      query,
                                      key,
                                      offsets=torch.zeros(batch_size * seq_len,
                                                          dtype=torch.long,
                                                          device=device))
    # Compare the results.
    torch.testing.assert_close(out_query,
                               ref_query,
                               atol=get_default_atol(out_query),
                               rtol=get_default_rtol(out_query))
    torch.testing.assert_close(out_key,
                               ref_key,
                               atol=get_default_atol(out_key),
                               rtol=get_default_rtol(out_key))


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_batched_rotary_embedding_multi_lora(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    device: str,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    scaling_factors: List[int] = [1, 2, 4]
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style, {
        "rope_type": "linear",
        "factor": tuple(scaling_factors)
    })
    rope = rope.to(dtype=dtype, device=torch.get_default_device())

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype)
    key = torch.randn_like(query)

    offset_map = torch.tensor(
        list(
            accumulate([0] + [
                max_position * scaling_factor * 2
                for scaling_factor in scaling_factors[:-1]
            ])))
    query_types = torch.randint(0,
                                len(scaling_factors), (batch_size, seq_len),
                                device=device)
    query_offsets = offset_map[query_types]

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope.forward_native(positions, query, key,
                                             query_offsets)
    out_query, out_key = rope.forward(positions, query, key,
                                      query_offsets.flatten())
    # Compare the results.
    torch.testing.assert_close(out_query,
                               ref_query,
                               atol=get_default_atol(out_query),
                               rtol=get_default_rtol(out_query))
    torch.testing.assert_close(out_key,
                               ref_key,
                               atol=get_default_atol(out_key),
                               rtol=get_default_rtol(out_key))


@torch.inference_mode()
def test_rope_module_cache():
    MAX_POSITIONS = [123, 1234]
    BASES = [10000, 1000000]
    ROPE_SCALINGS = (None, {
        "rope_type": "linear",
        "factor": (1, )
    }, {
        "rope_type": "dynamic",
        "factor": 1
    })
    settings = (HEAD_SIZES, ROTARY_DIMS, MAX_POSITIONS, BASES, IS_NEOX_STYLE,
                ROPE_SCALINGS, DTYPES)
    rope_setting_id_map: Dict[str, int] = {}
    for setting in product(*settings):
        head_size, rotary_dim, max_position, base, \
            is_neox_stype, rope_scaling, dtype = setting
        if rotary_dim is None:
            rotary_dim = head_size
        rope = get_rope(head_size, rotary_dim, max_position, base,
                        is_neox_stype, rope_scaling, dtype)
        # different settings cannot share the same rope module
        assert id(rope) not in rope_setting_id_map.values()
        assert all(x.dtype == dtype for x in rope.buffers())
        assert all(x.dtype == dtype for x in rope.parameters())
        rope_setting_id_map[str(setting)] = id(rope)

    for setting in product(*settings):
        head_size, rotary_dim, max_position, base, \
            is_neox_stype, rope_scaling, dtype = setting
        if rotary_dim is None:
            rotary_dim = head_size
        rope = get_rope(head_size, rotary_dim, max_position, base,
                        is_neox_stype, rope_scaling, dtype)
        # check if cache take effect
        assert id(rope) == rope_setting_id_map[str(setting)]
