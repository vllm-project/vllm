from typing import Optional

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import get_rope

IS_NEOX_STYLE = [True, False]
DTYPES = [torch.half, torch.bfloat16, torch.float]
HEAD_SIZES = [64, 80, 96, 112, 128, 256]
ROTARY_DIMS = [None, 32]  # None means rotary dim == head size
NUM_HEADS = [7, 17]  # Arbitrary values for testing
BATCH_SIZES = [1, 5]  # Arbitrary values for testing
SEQ_LENS = [11, 8192]  # Arbitrary values for testing
SEEDS = [0]
QUERY_SCALE = [0.02, 0.08]
KEY_SCALE = [0.02, 0.08]


@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
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
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if rotary_dim is None:
        rotary_dim = head_size
    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
    rope = rope.to(dtype).cuda()

    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device="cuda")
    query = torch.randn(batch_size,
                        seq_len,
                        num_heads * head_size,
                        dtype=dtype,
                        device="cuda")
    key = torch.randn_like(query)

    # NOTE(woosuk): The reference implementation should be executed first
    # because the custom kernel is in-place.
    ref_query, ref_key = rope._forward(positions, query, key)
    out_query, out_key = rope.forward(positions, query, key)
    # Compare the results.
    assert torch.allclose(out_query, ref_query, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_key, ref_key, atol=1e-5, rtol=1e-5)

@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("query_scale", QUERY_SCALE)
@pytest.mark.parametrize("key_scale", KEY_SCALE)
@torch.inference_mode()
def test_dequant_rotary_embedding(
    is_neox_style: bool,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_size: int,
    rotary_dim: Optional[int],
    dtype: torch.dtype,
    seed: int,
    query_scale: float,
    key_scale: float,
    max_position: int = 8192,
    base: int = 10000,
) -> None:
    if rotary_dim is None:
        rotary_dim = head_size
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    rope = get_rope(head_size, rotary_dim, max_position, base, is_neox_style)
    rope = rope.to(dtype).cuda()

    positions = torch.randint(0,
                              max_position, (batch_size, seq_len),
                              device="cuda")
    query = torch.randint(
        -1000,
        1000,
        (batch_size, seq_len, num_heads * head_size),
        dtype=torch.int32,
        device="cuda",
    )
    key = torch.randint(
        -1000,
        1000,
        (batch_size, seq_len, num_heads * head_size),
        dtype=torch.int32,
        device="cuda",
    )
    query_ = (query * query_scale).to(dtype)
    key_ = (key * key_scale).to(dtype)

    ref_query, ref_key = rope._forward(positions, query_, key_)

    # ref_query = ref_query.view(num_tokens, num_heads * head_size)
    # ref_key = ref_key.view(num_tokens, num_heads * head_size)
    out2_query = torch.empty_like(query_)
    out2_key = torch.empty_like(key_)

    ops.rotary_embedding(
        positions,
        query,
        key,
        head_size,
        rope.cos_sin_cache,
        is_neox_style,
        out2_query,
        out2_key,
        True,  # use quant
        query_scale,
        key_scale,
    )
    assert torch.allclose(ref_key, out2_key,
                          atol=1e-4), f"diff: {torch.max(ref_key - out2_key)}"
    assert torch.allclose(
        ref_query, out2_query,
        atol=1e-4), f"diff: {torch.max(ref_query - out2_query)}"
