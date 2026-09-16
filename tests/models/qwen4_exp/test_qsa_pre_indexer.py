# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the fused QSA pre-indexer."""

import pytest
import torch

from vllm.models.qwen4_exp.common.qsa_cache import (
    canonical_qsa_rope_positions,
    circular_qsa_slot_mapping,
    compressed_qsa_slot_mapping,
)
from vllm.models.qwen4_exp.nvidia.indexer_qsa import apply_qsa_rope
from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_compress_groups_with_ratio,
    qsa_store_cache_rows,
)
from vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer import (
    QSAMainPrepare,
    qsa_pre_indexer,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)

HQ, D = 4, 128
CR = 4
MROPE_SECTION = (11, 11, 10)
EPS = 1e-6
BLOCK_SIZE = 16
COMP_PAGE = BLOCK_SIZE // CR
ROPE_POS_OFFSET = D
RTOL = 1.6e-2
ATOL = 1e-2
MIXED_BATCH = ([260, 259, 138], [1, 1, 37], [8, 8, 8])


def _make_block_table(block_counts):
    num_blocks = sum(block_counts)
    table = torch.full((len(block_counts), max(block_counts)), -1, dtype=torch.int32)
    physical_blocks = torch.randperm(num_blocks)
    offset = 0
    for request, count in enumerate(block_counts):
        table[request, :count] = physical_blocks[offset : offset + count]
        offset += count
    return table, num_blocks


def assert_fp8_within_one_ulp(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # e4m3 is sign-magnitude, so within a sign the uint8 code order matches the
    # value order and one ulp is one code step. The two paths' intermediates
    # differ in the pooling accumulation order, which at denormal magnitudes
    # (absolute grid step 2^-9) shows up as up to 2 code steps.
    code_diff = (
        actual.view(torch.uint8).int() - expected.view(torch.uint8).int()
    ).abs()
    abs_diff = (actual.float() - expected.float()).abs()
    assert bool(((code_diff <= 1) | (abs_diff <= 2**-8)).all())


@requires_qsa_kernels
@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("indexer_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "mrope,is_2d_positions,cache_rope_positions,state_size,seq_lens,query_lens,history_lens",
    [
        pytest.param(True, True, True, 4, *MIXED_BATCH, id="mrope"),
        pytest.param(False, False, False, 4, *MIXED_BATCH, id="text"),
        pytest.param(True, False, True, 8, *MIXED_BATCH, id="mrope-model-1d"),
        pytest.param(True, True, False, 4, *MIXED_BATCH, id="mrope-no-position-cache"),
        pytest.param(
            True,
            False,
            False,
            8,
            *MIXED_BATCH,
            id="mrope-model-1d-no-position-cache",
        ),
        pytest.param(
            False, False, True, 4, *MIXED_BATCH, id="text-with-position-cache"
        ),
        pytest.param(True, True, True, 4, [37], [37], [0], id="fresh"),
        pytest.param(True, True, True, 4, [4097], [4097], [0], id="tiled"),
    ],
)
def test_qsa_fused_pre_indexer_matches_unfused(
    indexer_dtype,
    mrope,
    is_2d_positions,
    cache_rope_positions,
    state_size,
    seq_lens,
    query_lens,
    history_lens,
) -> None:
    from flashinfer.norm import gemma_rmsnorm

    from vllm.model_executor.layers.rotary_embedding import get_rope

    device = "cuda"
    rope_params = {
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    }
    if mrope:
        rope_params["mrope_interleaved"] = True
        rope_params["mrope_section"] = list(MROPE_SECTION)
    with torch.device(device):
        rope = get_rope(
            head_size=256,
            max_position=32768,
            rope_parameters=rope_params,
            dtype=torch.bfloat16,
        )

    token_to_req = torch.cat(
        [
            torch.full((length,), request, dtype=torch.int32)
            for request, length in enumerate(query_lens)
        ]
    ).to(device)
    logical_positions = torch.cat(
        [
            torch.arange(seq_len - query_len, seq_len, dtype=torch.int64)
            for seq_len, query_len in zip(seq_lens, query_lens)
        ]
    ).to(device)
    num_tokens = logical_positions.numel()
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    ).to(device)
    positions = (
        torch.stack(
            [
                logical_positions,
                logical_positions // 7 + 3,
                logical_positions // 13 + 11,
            ]
        )
        if is_2d_positions
        else logical_positions
    )
    position_rows = (
        canonical_qsa_rope_positions(positions)
        if cache_rope_positions
        else logical_positions.view(-1, 1, 1).expand(-1, 1, 3)
    )

    compressed_block_counts = [
        (seq_len // CR + COMP_PAGE - 1) // COMP_PAGE for seq_len in seq_lens
    ]
    raw_block_table, num_raw_blocks = _make_block_table([1] * len(seq_lens))
    compressed_block_table, num_compressed_blocks = _make_block_table(
        compressed_block_counts
    )
    raw_block_table = raw_block_table.to(device)
    raw_slots = circular_qsa_slot_mapping(
        raw_block_table,
        token_to_req,
        logical_positions,
        state_size,
        query_start_loc,
    )
    compressed_slots = compressed_qsa_slot_mapping(
        compressed_block_table.to(device),
        token_to_req,
        logical_positions,
        COMP_PAGE,
        CR,
    )
    group_counts = torch.tensor(
        [
            seq_len // CR - (seq_len - query_len) // CR
            for seq_len, query_len in zip(seq_lens, query_lens)
        ],
        dtype=torch.int32,
        device=device,
    )
    k_work_counts = torch.maximum(group_counts, torch.ones_like(group_counts))
    k_start_loc = torch.cat([k_work_counts.new_zeros(1), k_work_counts.cumsum(0)])
    work_requests = torch.repeat_interleave(
        torch.arange(len(query_lens), dtype=torch.int32, device=device),
        k_work_counts,
    )
    local_work = torch.arange(
        int(k_start_loc[-1]), dtype=torch.int32, device=device
    ) - torch.repeat_interleave(k_start_loc[:-1], k_work_counts)
    max_k_work = (num_tokens + (CR - 1) * len(query_lens)) // CR
    k_work_metadata = torch.full((max_k_work, 2), -1, dtype=torch.int32, device=device)
    k_work_metadata[: work_requests.numel()] = torch.stack(
        (work_requests, local_work), dim=1
    )

    raw_width = D + 12 if cache_rope_positions else D
    # Match vLLM's padded-page cache layout: rows are contiguous, while physical
    # blocks have a larger stride than their logical contents.
    raw_page_elements = state_size * raw_width
    fused_raw_storage = torch.zeros(
        num_raw_blocks,
        raw_page_elements + 16,
        dtype=torch.bfloat16,
        device=device,
    )
    fused_raw = torch.as_strided(
        fused_raw_storage,
        (num_raw_blocks, state_size, 1, raw_width),
        (raw_page_elements + 16, raw_width, raw_width, 1),
    )
    for request, history_len in enumerate(history_lens):
        history_end = seq_lens[request] - query_lens[request]
        for position in range(history_end - history_len, history_end):
            block = int(raw_block_table[request, 0])
            row = fused_raw[block, position % state_size, 0]
            row[:D] = torch.randn(D, dtype=torch.bfloat16, device=device)
            if cache_rope_positions:
                row[ROPE_POS_OFFSET:].view(torch.int64).copy_(
                    torch.tensor(
                        [position, position // 7 + 3, position // 13 + 11],
                        dtype=torch.int64,
                        device=device,
                    )
                )
    unfused_raw = fused_raw.clone()
    compressed_page_elements = COMP_PAGE * D
    fused_compressed_storage = torch.zeros(
        num_compressed_blocks,
        compressed_page_elements + 16,
        dtype=indexer_dtype,
        device=device,
    )
    fused_compressed = torch.as_strided(
        fused_compressed_storage,
        (num_compressed_blocks, COMP_PAGE, 1, D),
        (compressed_page_elements + 16, D, D, 1),
    )
    unfused_compressed = fused_compressed.clone()

    projected_qk = torch.randn(
        num_tokens, (HQ + 1) * D, dtype=torch.bfloat16, device=device
    )
    q_weight = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.2
    k_weight = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.2

    fused_query = torch.empty(num_tokens, HQ, D, dtype=indexer_dtype, device=device)
    qsa_pre_indexer(
        projected_qk[:, : HQ * D],
        projected_qk[:, HQ * D :],
        positions,
        rope.cos_sin_cache,
        q_weight,
        k_weight,
        EPS,
        fused_query,
        fused_raw,
        raw_slots,
        raw_block_table,
        query_start_loc,
        logical_positions,
        fused_compressed,
        compressed_slots,
        k_work_metadata,
        compress_ratio=CR,
        mrope_section=MROPE_SECTION if mrope else None,
        rope_pos_offset=ROPE_POS_OFFSET if cache_rope_positions else None,
    )

    unfused_query = projected_qk[:, : HQ * D].reshape(num_tokens, HQ, D)
    unfused_query = gemma_rmsnorm(
        unfused_query.reshape(-1, D), q_weight, EPS
    ).reshape_as(unfused_query)
    unfused_query = apply_qsa_rope(rope, positions, unfused_query)

    raw_keys = unfused_raw[..., :D]
    rope_positions = (
        unfused_raw[..., ROPE_POS_OFFSET:].view(torch.int64)
        if cache_rope_positions
        else None
    )
    pooled, first_positions = qsa_compress_groups_with_ratio(
        projected_qk[:, HQ * D :].reshape(-1, 1, D),
        position_rows,
        raw_keys,
        raw_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        CR,
        rope_positions,
    )
    compressed_rows = gemma_rmsnorm(pooled.reshape(-1, D), k_weight, EPS).reshape(
        -1, 1, D
    )
    group_positions = (
        first_positions.transpose(0, 1) if mrope else first_positions[:, 0]
    )
    compressed_rows = apply_qsa_rope(rope, group_positions, compressed_rows)
    qsa_store_cache_rows(unfused_compressed, compressed_slots, compressed_rows)
    qsa_store_cache_rows(raw_keys, raw_slots, projected_qk[:, HQ * D :])
    if rope_positions is not None:
        qsa_store_cache_rows(rope_positions, raw_slots, position_rows)

    if indexer_dtype == torch.float8_e4m3fn:
        # Both paths round the same ~bf16 intermediates to e4m3. Bitwise
        # equality (the dsv4 indexer test's bar) does not hold here: the 1D
        # reference rope (forward_cuda) differs from _norm_rope by up to 1
        # bf16 ulp, and even the MRoPE pair flips a code occasionally.
        unfused_query = unfused_query.to(indexer_dtype)
        assert_fp8_within_one_ulp(fused_query, unfused_query)
    else:
        torch.testing.assert_close(fused_query, unfused_query, rtol=RTOL, atol=ATOL)
    assert torch.equal(fused_raw.view(torch.int16), unfused_raw.view(torch.int16))
    if indexer_dtype == torch.float8_e4m3fn:
        assert_fp8_within_one_ulp(fused_compressed, unfused_compressed)
    else:
        torch.testing.assert_close(
            fused_compressed, unfused_compressed, rtol=RTOL, atol=ATOL
        )


@requires_qsa_kernels
@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("query_len", [1, 4])
@pytest.mark.parametrize(("main_heads", "main_kv_heads"), [(3, 1), (24, 2)])
def test_qsa_pre_indexer_fused_main_prepare_matches_unfused(
    mrope, query_len, main_heads, main_kv_heads
):
    """Fusing the main Q/K prepare and K/V cache write into the pre-indexer
    launch must leave the indexer outputs bitwise unchanged and reproduce the
    standalone QK-norm/RoPE/gate kernel, also under graph replay."""
    from vllm.model_executor.layers.fused_qk_norm_rope import (
        fused_qk_rmsnorm_rope_gate,
    )
    from vllm.model_executor.layers.rotary_embedding import get_rope

    torch.manual_seed(0)
    device = "cuda"
    main_dim, rotary_dim, page = 256, 64, 16
    state_size = 8
    # Unaligned decode starts: the first compression group reads the ring.
    starts = [8193, 4098, 37]
    seq_lens = [start + query_len for start in starts]
    query_lens = [query_len] * len(starts)
    rope_params = {
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default",
    }
    if mrope:
        rope_params["mrope_interleaved"] = True
        rope_params["mrope_section"] = list(MROPE_SECTION)
    with torch.device(device):
        rope = get_rope(
            head_size=main_dim,
            max_position=32768,
            rope_parameters=rope_params,
            dtype=torch.bfloat16,
        )
    mrope_section = MROPE_SECTION if mrope else None

    token_to_req = torch.cat(
        [
            torch.full((length,), request, dtype=torch.int32)
            for request, length in enumerate(query_lens)
        ]
    ).to(device)
    logical_positions = torch.cat(
        [
            torch.arange(seq_len - query_len, seq_len, dtype=torch.int64)
            for seq_len in seq_lens
        ]
    ).to(device)
    num_tokens = logical_positions.numel()
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    ).to(device)
    positions = (
        torch.stack(
            [
                logical_positions,
                logical_positions // 7 + 3,
                logical_positions // 13 + 11,
            ]
        )
        if mrope
        else logical_positions
    )
    raw_block_table, num_raw_blocks = _make_block_table([1] * len(seq_lens))
    compressed_block_table, num_compressed_blocks = _make_block_table(
        [(seq_len // CR + COMP_PAGE - 1) // COMP_PAGE for seq_len in seq_lens]
    )
    raw_block_table = raw_block_table.to(device)
    raw_slots = circular_qsa_slot_mapping(
        raw_block_table, token_to_req, logical_positions, state_size, query_start_loc
    )
    compressed_slots = compressed_qsa_slot_mapping(
        compressed_block_table.to(device),
        token_to_req,
        logical_positions,
        COMP_PAGE,
        CR,
    )
    k_work_metadata = torch.tensor(
        [
            (request, work)
            for request, seq_len in enumerate(seq_lens)
            for work in range(max(1, seq_len // CR - (seq_len - query_len) // CR))
        ],
        dtype=torch.int32,
        device=device,
    )
    raw_width = D + 12 if mrope else D
    raw_initial = torch.randn(
        num_raw_blocks, state_size, 1, raw_width, dtype=torch.bfloat16, device=device
    )
    if mrope:
        # The ring rows the first group may read carry valid MRoPE coordinates.
        for request, start in enumerate(starts):
            block = int(raw_block_table[request, 0])
            for position in range(start - state_size, start):
                raw_initial[block, position % state_size, 0, D:].view(
                    torch.int64
                ).copy_(
                    torch.tensor(
                        [position, position // 7 + 3, position // 13 + 11],
                        dtype=torch.int64,
                        device=device,
                    )
                )
    compressed_initial = torch.randn(
        num_compressed_blocks, COMP_PAGE, 1, D, dtype=torch.bfloat16, device=device
    )
    projected_qk = torch.randn(
        num_tokens, (HQ + 1) * D, dtype=torch.bfloat16, device=device
    )
    index_q_weight = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.2
    index_k_weight = torch.randn(D, dtype=torch.bfloat16, device=device) * 0.2

    qkv = torch.randn(
        num_tokens,
        (2 * main_heads + 2 * main_kv_heads) * main_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    q_gate, k, v = qkv.split(
        [main_heads * 2 * main_dim, main_kv_heads * main_dim, main_kv_heads * main_dim],
        dim=-1,
    )
    q_weight = torch.randn(main_dim, dtype=torch.bfloat16, device=device) * 0.1
    k_weight = torch.randn(main_dim, dtype=torch.bfloat16, device=device) * 0.1
    num_main_blocks = 4
    main_slots = torch.randperm(num_main_blocks * page, device=device)[:num_tokens]
    main_slots[-1] = -1  # A padded row must not touch the cache.
    kv_initial = torch.randn(
        num_main_blocks,
        main_kv_heads,
        page,
        2 * main_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    raw, compressed, kv_cache = (
        t.clone() for t in (raw_initial, compressed_initial, kv_initial)
    )
    index_query = torch.empty(num_tokens, HQ, D, dtype=torch.bfloat16, device=device)
    q_out = torch.empty(
        num_tokens, main_heads, main_dim, dtype=torch.bfloat16, device=device
    )
    gate_out = torch.empty_like(q_out)

    def run_pre_indexer(query_out, raw_cache, compressed_cache, main):
        qsa_pre_indexer(
            projected_qk[:, : HQ * D],
            projected_qk[:, HQ * D :],
            positions,
            rope.cos_sin_cache,
            index_q_weight,
            index_k_weight,
            EPS,
            query_out,
            raw_cache,
            raw_slots,
            raw_block_table,
            query_start_loc,
            logical_positions,
            compressed_cache,
            compressed_slots,
            k_work_metadata,
            compress_ratio=CR,
            mrope_section=mrope_section,
            rope_pos_offset=ROPE_POS_OFFSET if mrope else None,
            main=main,
        )

    def run_fused():
        run_pre_indexer(
            index_query,
            raw,
            compressed,
            QSAMainPrepare(
                q_gate=q_gate,
                k=k,
                v=v,
                q_norm_weight=q_weight,
                k_norm_weight=k_weight,
                eps=EPS,
                rotary_dim=rotary_dim,
                q_out=q_out,
                gate_out=gate_out,
                kv_cache=kv_cache.transpose(1, 2),
                slot_mapping=main_slots,
            ),
        )

    def check():
        ref_query = torch.empty_like(index_query)
        ref_raw, ref_compressed = raw_initial.clone(), compressed_initial.clone()
        run_pre_indexer(ref_query, ref_raw, ref_compressed, None)
        assert torch.equal(index_query, ref_query)
        assert torch.equal(raw.view(torch.int16), ref_raw.view(torch.int16))
        assert torch.equal(compressed, ref_compressed)

        q_ref, k_ref, gate_ref = fused_qk_rmsnorm_rope_gate(
            q_gate,
            k,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            positions,
            EPS,
            main_heads,
            main_kv_heads,
            main_dim,
            rotary_dim,
            mrope_section=mrope_section,
            norm_beta=1.0,
        )
        torch.testing.assert_close(q_out, q_ref.view_as(q_out), rtol=RTOL, atol=ATOL)
        assert torch.equal(gate_out, gate_ref.view_as(gate_out))
        kv_ref = kv_initial.clone()
        for token, slot in enumerate(main_slots.tolist()):
            if slot >= 0:
                row = kv_ref[slot // page, :, slot % page]
                row[:, :main_dim] = k_ref[token].view(main_kv_heads, main_dim)
                row[:, main_dim:] = v[token].view(main_kv_heads, main_dim)
        torch.testing.assert_close(kv_cache, kv_ref, rtol=RTOL, atol=ATOL)
        # V rows are verbatim copies and unaddressed rows must stay untouched.
        assert torch.equal(kv_cache[..., main_dim:], kv_ref[..., main_dim:])
        written = torch.zeros(num_main_blocks, page, dtype=torch.bool, device=device)
        written.view(-1)[main_slots[main_slots >= 0]] = True
        untouched = ~written[:, None, :, None].expand_as(kv_cache)
        assert torch.equal(kv_cache[untouched], kv_initial[untouched])

    run_fused()
    check()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_fused()
    for _ in range(2):
        qkv.normal_()
        projected_qk.normal_()
        raw.copy_(raw_initial)
        compressed.copy_(compressed_initial)
        kv_cache.copy_(kv_initial)
        graph.replay()
        check()
