# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the production UltraQuant D=256 FlyDSL decode."""

import pytest
import torch

from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.flydsl_ultraquant_decode import (
    flydsl_ultraquant_decode_attention,
    is_ultraquant_flydsl_available,
    ultraquant_flydsl_decode_eligible,
)
from vllm.v1.attention.ops.ultraquant.format import slot_size
from vllm.v1.attention.ops.ultraquant.reference import (
    hadamard_matrix,
    reference_ultraquant_attention,
)
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


def _build_paged_cache(key, value, block_size):
    batch_size, seq_len, num_kv_heads, head_size = key.shape
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    kv_cache = torch.zeros(
        batch_size * blocks_per_seq,
        block_size,
        num_kv_heads,
        slot_size(head_size),
        dtype=torch.uint8,
        device=key.device,
    )
    block_table = torch.arange(
        batch_size * blocks_per_seq,
        dtype=torch.int32,
        device=key.device,
    ).view(batch_size, blocks_per_seq)
    slot_mapping = torch.arange(
        batch_size * blocks_per_seq * block_size,
        dtype=torch.int64,
        device=key.device,
    ).view(batch_size, -1)[:, :seq_len]
    ultraquant_store(
        key.reshape(-1, num_kv_heads, head_size),
        value.reshape(-1, num_kv_heads, head_size),
        kv_cache,
        slot_mapping.reshape(-1),
    )
    return kv_cache, block_table


@pytest.mark.parametrize(
    ("batch_size", "query_group_size", "block_size", "contiguous_query"),
    [
        (1, 8, 32, True),
        (8, 8, 64, True),
        (9, 8, 256, True),
        (1, 16, 32, True),
        (1, 6, 16, True),
        (1, 6, 32, True),
        (2, 6, 64, True),
        (1, 6, 128, True),
        (1, 6, 256, True),
        (1, 6, 32, False),
    ],
)
def test_flydsl_decode_matches_reference(
    batch_size,
    query_group_size,
    block_size,
    contiguous_query,
):
    if not ultraquant_flydsl_decode_eligible(
        head_size=256,
        num_kv_groups=query_group_size,
        has_sinks=False,
        sliding_window=None,
        flydsl_loaded=is_ultraquant_flydsl_available(),
    ):
        pytest.skip("UltraQuant D=256 FlyDSL kernel is unavailable")

    set_random_seed(0x57A1D + batch_size + block_size + query_group_size)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_kv_heads = 1
    head_size = 256
    seq_len = 2048
    scale = head_size**-0.5

    key = torch.randn(
        batch_size,
        seq_len,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    value = torch.randn_like(key)
    if contiguous_query:
        qkv = torch.randn(
            batch_size,
            query_group_size + 2 * num_kv_heads,
            head_size,
            dtype=dtype,
            device=device,
        )
        query = qkv[:, :query_group_size]
    else:
        query_storage = torch.randn(
            batch_size,
            query_group_size,
            head_size * 2,
            dtype=dtype,
            device=device,
        )
        query = query_storage[..., ::2]
    kv_cache, block_table = _build_paged_cache(key, value, block_size)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    hadamard = hadamard_matrix(head_size, device)

    actual = flydsl_ultraquant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        PiT=hadamard,
        max_seq_len=seq_len,
    )
    expected = reference_ultraquant_attention(
        query=query,
        key=key,
        value=value,
        scale=scale,
    )

    actual_f32 = actual.float()
    expected_f32 = expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    error = (actual_f32 - expected_f32).abs()
    assert cosine > 0.995
    assert error.mean() < 0.05
    assert error.max() < 0.8


def _cosine(actual, expected) -> float:
    return torch.nn.functional.cosine_similarity(
        actual.float().reshape(-1),
        expected.float().reshape(-1),
        dim=0,
    ).item()


def test_flydsl_verify_ragged_and_padded_matches_reference():
    """One KV read scores a ragged verify, including CUDA-graph padding rows."""
    if not ultraquant_flydsl_decode_eligible(
        head_size=256,
        num_kv_groups=6,
        has_sinks=False,
        sliding_window=None,
        flydsl_loaded=is_ultraquant_flydsl_available(),
    ):
        pytest.skip("UltraQuant D=256 FlyDSL kernel is unavailable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    Hk, Hq, D = 1, 6, 256
    block_size = 64
    seq_len = 128
    query_lens = [2, 4]
    scale = D**-0.5
    hadamard = hadamard_matrix(D, device)

    for repeat in range(20):
        set_random_seed(0x71A0 + repeat)
        key = torch.randn(len(query_lens), seq_len, Hk, D, dtype=dtype, device=device)
        value = torch.randn_like(key)
        kv_cache, block_table = _build_paged_cache(key, value, block_size)
        seq_lens = torch.full(
            (len(query_lens),), seq_len, dtype=torch.int32, device=device
        )
        real_rows = sum(query_lens)
        pad_rows = 2
        num_rows = real_rows + pad_rows
        query = torch.randn(num_rows, Hq, D, dtype=dtype, device=device)
        query_start_loc = torch.tensor(
            [0, query_lens[0], real_rows], dtype=torch.int32, device=device
        )
        actual = flydsl_ultraquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            PiT=hadamard,
            max_seq_len=seq_len,
            query_start_loc=query_start_loc,
            num_verify_tokens=max(query_lens),
        )
        row = 0
        for req, q_len in enumerate(query_lens):
            for j in range(q_len):
                ctx = seq_len - q_len + 1 + j
                expected = reference_ultraquant_attention(
                    query=query[row : row + 1],
                    key=key[req : req + 1, :ctx],
                    value=value[req : req + 1, :ctx],
                    scale=scale,
                )
                cos = _cosine(actual[row], expected)
                assert cos > 0.995, (repeat, req, j, cos)
                row += 1
        assert row == real_rows


@pytest.mark.parametrize("query_group_size", [6, 8, 16])
@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize(("num_reqs", "seq_len"), [(1, 1024), (4, 8192), (12, 4096)])
def test_flydsl_verify_matches_single_token_launches(
    query_group_size, width, num_reqs, seq_len
):
    """One verify launch scores the same values as one launch per token.

    The shipping split launches the single-token kernel once per verify token.
    Both read the same cache, so the verify kernel must agree with it row by row.
    """
    if not ultraquant_flydsl_decode_eligible(
        head_size=256,
        num_kv_groups=query_group_size,
        has_sinks=False,
        sliding_window=None,
        flydsl_loaded=is_ultraquant_flydsl_available(),
    ):
        pytest.skip("UltraQuant D=256 FlyDSL kernel is unavailable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    Hk, D = 1, 256
    Hq = Hk * query_group_size
    block_size = 64
    scale = D**-0.5
    hadamard = hadamard_matrix(D, device)

    for repeat in range(3):
        set_random_seed(0xC0DE + width * 31 + num_reqs + repeat)
        key = torch.randn(num_reqs, seq_len, Hk, D, dtype=dtype, device=device)
        value = torch.randn_like(key)
        kv_cache, block_table = _build_paged_cache(key, value, block_size)
        seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int32, device=device)
        # A peaked softmax exposes score errors that a flat one hides.
        query = 4 * torch.randn(num_reqs * width, Hq, D, dtype=dtype, device=device)
        query_start_loc = torch.arange(
            0, (num_reqs + 1) * width, width, dtype=torch.int32, device=device
        )
        # The launcher returns a pooled buffer, so copy before the next launch.
        verify = flydsl_ultraquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            scale=scale,
            PiT=hadamard,
            max_seq_len=seq_len,
            query_start_loc=query_start_loc,
            num_verify_tokens=width,
        ).clone()
        for j in range(width):
            rows = torch.arange(num_reqs, device=device) * width + j
            single = flydsl_ultraquant_decode_attention(
                query=query[rows],
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=(seq_lens - width + 1 + j).to(torch.int32),
                scale=scale,
                PiT=hadamard,
                max_seq_len=seq_len,
            )
            assert _cosine(verify[rows], single) > 0.9999, (repeat, j)
            assert (verify[rows].float() - single.float()).abs().max() < 0.05, (
                repeat,
                j,
            )


def test_flydsl_verify_cache_past_4gib():
    """A block id past 4 GiB must not wrap the 32-bit cache offset."""
    if not ultraquant_flydsl_decode_eligible(
        head_size=256,
        num_kv_groups=6,
        has_sinks=False,
        sliding_window=None,
        flydsl_loaded=is_ultraquant_flydsl_available(),
    ):
        pytest.skip("UltraQuant D=256 FlyDSL kernel is unavailable")

    device = torch.device("cuda")
    free_bytes, _total = torch.cuda.mem_get_info(device)
    slot = slot_size(256)
    block_size = 64
    stride = block_size * slot
    blocks_needed = (1 << 32) // stride + 2
    nbytes = blocks_needed * stride
    if free_bytes < nbytes + (1 << 30):
        pytest.skip("need a KV cache larger than 4 GiB")

    set_random_seed(0x46B)
    dtype = torch.bfloat16
    Hk, Hq, D = 1, 6, 256
    seq_len = 64
    scale = D**-0.5
    key = torch.randn(1, seq_len, Hk, D, dtype=dtype, device=device)
    value = torch.randn_like(key)
    kv_cache = torch.zeros(
        blocks_needed, block_size, Hk, slot, dtype=torch.uint8, device=device
    )
    phys = blocks_needed - 1
    block_table = torch.tensor([[phys]], dtype=torch.int32, device=device)
    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device) + (
        phys * block_size
    )
    ultraquant_store(
        key.reshape(-1, Hk, D),
        value.reshape(-1, Hk, D),
        kv_cache,
        slot_mapping,
    )
    query = torch.randn(1, Hq, D, dtype=dtype, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    actual = flydsl_ultraquant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        PiT=hadamard_matrix(D, device),
        max_seq_len=seq_len,
    )
    expected = reference_ultraquant_attention(
        query=query, key=key, value=value, scale=scale
    )
    assert _cosine(actual, expected) > 0.995
