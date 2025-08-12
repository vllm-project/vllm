from itertools import accumulate
from typing import List, Optional

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope

from .allclose_default import get_default_atol, get_default_rtol

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.half, torch.bfloat16, torch.float]
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [7, 17]  # Arbitrary values for testing
BATCH_SIZES = [1, 5]  # Arbitrary values for testing
SEQ_LENS = [11, 8192]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


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
def test_rotary_embedding(
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
    if rotary_dim is None:
        rotary_dim = head_size
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
    rope = rope.to(dtype=dtype)

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype)
    key = torch.randn_like(query)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope._forward(positions, query, key)
    out_query, out_key = rope.forward(positions, query, key)
    # Compare the results.
    assert torch.allclose(out_query,
                          ref_query,
                          atol=get_default_atol(out_query),
                          rtol=get_default_rtol(out_query))
    assert torch.allclose(out_key,
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
def test_batched_rotary_embedding(
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
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style, {
        "type": "linear",
        "factor": (1, )
    })
    rope = rope.to(dtype=dtype)

    positions = torch.randint(0, max_position, (batch_size, seq_len))
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype)
    key = torch.randn_like(query)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope._forward(positions, query, key)
    out_query, out_key = rope.forward(positions,
                                      query,
                                      key,
                                      offsets=torch.zeros(batch_size * seq_len,
                                                          dtype=int,
                                                          device=device))
    # Compare the results.
    assert torch.allclose(out_query,
                          ref_query,
                          atol=get_default_atol(out_query),
                          rtol=get_default_rtol(out_query))
    assert torch.allclose(out_key,
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
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    if rotary_dim is None:
        rotary_dim = head_size
    scaling_factors: List[int] = [1, 2, 4]
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style, {
        "type": "linear",
        "factor": tuple(scaling_factors)
    })
    rope = rope.to(dtype=dtype)

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
    ref_query, ref_key = rope._forward(positions, query, key, query_offsets)
    out_query, out_key = rope.forward(positions, query, key,
                                      query_offsets.flatten())
    # Compare the results.
    assert torch.allclose(out_query,
                          ref_query,
                          atol=get_default_atol(out_query),
                          rtol=get_default_rtol(out_query))
    assert torch.allclose(out_key,
                          ref_key,
                          atol=get_default_atol(out_key),
                          rtol=get_default_rtol(out_key))
