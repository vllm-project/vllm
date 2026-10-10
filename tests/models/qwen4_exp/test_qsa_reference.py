# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.common import qsa_cache
from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
from vllm.models.qwen4_exp.nvidia import indexer_qsa
from vllm.models.qwen4_exp.nvidia import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
from vllm.models.qwen4_exp.nvidia.qsa import qsa_kv_cache_dtype
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.utils import clear_layer_kv_caches

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)


def test_qsa_mtp_index_share_updates_cache_but_skips_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = torch.tensor([[3, 1, -1], [5, 2, 0]], dtype=torch.int32)
    raw_metadata = SimpleNamespace(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        block_table=torch.empty(0),
        query_start_loc=torch.arange(3),
        logical_positions=torch.arange(2),
    )
    compressed_metadata = SimpleNamespace(
        num_actual_tokens=2,
        slot_mapping=torch.arange(2),
        k_work_metadata=torch.empty(0),
    )
    updates = []
    selections = []
    indexer = SimpleNamespace(
        skip_topk=True,
        _metadata=lambda: (raw_metadata, compressed_metadata),
        index_n_heads=1,
        index_kv_heads=1,
        index_head_dim=1,
        indexer_dtype=torch.bfloat16,
        raw_key_cache=SimpleNamespace(
            kv_cache=torch.empty(0),
            rope_position_cache=None,
            rope_position_offset=0,
        ),
        compressed_key_cache=SimpleNamespace(kv_cache=torch.empty(0)),
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.empty(0)),
        q_layernorm=SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6),
        k_layernorm=SimpleNamespace(weight=torch.ones(1)),
        compress_ratio=2,
    )
    attn = SimpleNamespace(
        use_fused_qsa_prepare=True,
        kv_cache=torch.empty(0, 1, 1, 2),
        kv_cache_dtype="auto",
        q_norm=SimpleNamespace(weight=torch.ones(1), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.ones(1)),
        _k_scale_float=1.0,
        _v_scale_float=1.0,
    )

    monkeypatch.setattr(
        indexer_qsa,
        "qsa_prepare",
        lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        qsa_indexer_ops,
        "qsa_select_paged_decode",
        lambda *args, **kwargs: selections.append((args, kwargs)),
    )
    monkeypatch.setattr(
        qsa_indexer_ops,
        "qsa_select_paged_prefill",
        lambda *args, **kwargs: selections.append((args, kwargs)),
    )

    actual, _ = indexer_qsa.QSAIndexer.forward(
        indexer,
        torch.zeros(2, 2),
        torch.tensor([7, 8]),
        rows,
        attn=attn,
        qkv=torch.zeros(2, 4),
        slot_mapping=torch.arange(2),
    )

    assert actual is rows
    assert len(updates) == 1
    assert not selections


def _qsa_mqa_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    visible_lengths: torch.Tensor,
) -> torch.Tensor:
    pages = page_table.index_select(0, token_to_req.long()).long()
    keys = k_cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(positions >= visible_lengths.unsqueeze(1), -torch.inf)


def _qsa_relative_topk_reference(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    output = torch.full(
        (logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device
    )
    for row in range(logits.shape[0]):
        start = int(row_starts[row].item())
        length = int((row_ends[row] - row_starts[row]).item())
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def _expand_qsa_indices_reference(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    rows = block_indices.shape[0]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    offsets = torch.arange(compress_ratio, device=block_indices.device)
    blocks = block_indices.long()
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=block_indices.device)
    visible_tokens = query_positions + 1
    tail_start = visible_tokens // compress_ratio * compress_ratio
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_count = (visible_tokens - tail_start).unsqueeze(1)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(output_width, device=result.device).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + output_width)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


def _qsa_select_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
) -> torch.Tensor:
    row_sequence_lengths = sequence_lengths.index_select(0, token_to_req.long())
    visible_blocks = torch.minimum(
        (query_positions + 1) // compress_ratio,
        row_sequence_lengths // compress_ratio,
    ).to(torch.int32)
    logits = _qsa_mqa_paged_reference(
        q,
        k_cache,
        page_table,
        token_to_req,
        visible_blocks,
    )
    starts = torch.zeros_like(visible_blocks)
    return _qsa_relative_topk_reference(
        logits,
        starts,
        visible_blocks,
        token_topk // compress_ratio,
    )


def _qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Dense reference for QSA sparse paged attention.

    Mirrors the kernel's dequant: fp8-e4m3 K/V caches are dequantized with the
    per-tensor k_scale/v_scale host floats; bf16 caches use unit scales.
    """
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            continue
        request = token_to_req[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = (k_cache[pages, offsets].float() * k_scale).repeat_interleave(
            repeats, dim=1
        )
        values = (v_cache[pages, offsets].float() * v_scale).repeat_interleave(
            repeats, dim=1
        )
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys)
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities, values).to(q.dtype)
    return output


@requires_qsa_kernels
def test_qsa_fp8_loaded_scales_reach_writer_and_attention(
    tmp_path, dist_init, workspace_init
) -> None:
    """Checkpoint K/V scales must govern both stored bytes and attention."""
    from transformers import Qwen4ExpTextConfig

    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    config = Qwen4ExpTextConfig(
        architectures=["Qwen4ExpForCausalLM"],
        num_hidden_layers=1,
        layer_types=["full_attention"],
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=24,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        ple_layer_ids=[],
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
        indexer_budget=2048,
    )
    config.save_pretrained(tmp_path)
    vllm_config = EngineArgs(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        kv_cache_dtype="fp8_e4m3",
        block_size=16,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    ).create_engine_config()
    device = torch.device("cuda")
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        device,
    ):
        owner = Qwen4ExpQSAAttention(
            vllm_config=vllm_config,
            config=config,
            layer_id=0,
            prefix="model.layers.0.self_attn",
        )
        model = torch.nn.Module()
        model.add_module("attn", owner)
        AutoWeightsLoader(model).load_weights(
            [
                ("attn._k_scale", torch.tensor(0.5)),
                ("attn._v_scale", torch.tensor(2.0)),
            ]
        )
        process_weights_after_loading(model, vllm_config.model_config, device)

    torch.manual_seed(7)
    key = torch.randn(2, 2, 64, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    query = torch.randn(2, 24, 64, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(1, 2, 16, 128, dtype=torch.uint8, device=device)
    slots = torch.arange(2, dtype=torch.int64, device=device)
    owner.impl.do_kv_cache_update(owner, key, value, cache, slots)
    key_cache, value_cache = cache.transpose(1, 2).split(64, dim=-1)
    expected_key = torch.zeros(1, 16, 2, 64, device=device)
    expected_value = torch.zeros_like(expected_key)
    expected_key[0, :2] = key.float() / 0.5
    expected_value[0, :2] = value.float() / 2.0
    expected_key = expected_key.to(torch.float8_e4m3fn)
    expected_value = expected_value.to(torch.float8_e4m3fn)
    torch.testing.assert_close(key_cache, expected_key.view(torch.uint8))
    torch.testing.assert_close(value_cache, expected_value.view(torch.uint8))

    selected = torch.tensor([[0, -1], [0, 1]], dtype=torch.int32, device=device)
    packed = torch.cat(
        (selected, torch.tensor([[1], [2]], dtype=torch.int32, device=device)), dim=1
    )
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
    requests = torch.zeros(2, dtype=torch.int32, device=device)
    metadata = FlashAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=device),
        max_seq_len=2,
        seq_lens=torch.tensor([2], dtype=torch.int32, device=device),
        block_table=block_table,
        slot_mapping=slots,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    expected = (
        _qsa_sparse_paged_attention_reference(
            query,
            expected_key,
            expected_value,
            selected,
            block_table,
            requests,
            64**-0.5,
            k_scale=0.5,
            v_scale=2.0,
        )
        * 0.5
    )
    actual = owner.impl.forward_qsa(
        owner,
        query,
        key,
        value,
        cache,
        metadata,
        torch.empty_like(query),
        token_to_req=requests,
        use_prefill_config=True,
        output_gate=torch.zeros_like(query),
        topk_indices=packed,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_qsa_kernels
def test_qsa_side_metadata_marks_cudagraph_padding_inert() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 1
    builder.reorder_batch_threshold = 4
    builder.is_circular_buffer = False
    builder.storage_block_size = 64
    builder.token_to_req_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.visible_blocks_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.k_work_metadata_buffer = torch.empty(0, 2, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [0] * 4, device=device)
    common = SimpleNamespace(
        num_actual_tokens=16,
        num_reqs=4,
        max_query_len=4,
        max_seq_len=68,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([68, 68, 68, 0], dtype=torch.int32, device=device),
        slot_mapping=torch.tensor(list(range(12)) + [-1] * 4, device=device),
        block_table_tensor=torch.empty((4, 0), dtype=torch.int32, device=device),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)

    assert metadata.logical_positions.tolist() == [
        64,
        65,
        66,
        67,
        64,
        65,
        66,
        67,
        64,
        65,
        66,
        67,
        -1,
        -1,
        -1,
        -1,
    ]
    assert metadata.slot_mapping.tolist() == list(range(12)) + [-1] * 4
    assert metadata.visible_blocks.tolist() == [65, 66, 67, 68] * 3 + [0] * 4


@requires_qsa_kernels
def test_qsa_circular_buffer_metadata_keeps_only_each_requests_suffix() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 4
    builder.reorder_batch_threshold = 1
    builder.is_circular_buffer = True
    builder.kv_cache_spec = SimpleNamespace(block_size=4)
    builder.storage_block_size = 4
    builder.token_to_req_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.visible_blocks_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.k_work_metadata_buffer = torch.empty(0, 2, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 7, 13, 13], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0] * 7 + [1] * 6 + [0] * 3, device=device)
    block_table = torch.tensor([[1], [3], [2]], dtype=torch.int32, device=device)
    common = SimpleNamespace(
        num_actual_tokens=16,
        num_reqs=3,
        max_query_len=7,
        max_seq_len=11,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([9, 11, 0], dtype=torch.int32, device=device),
        slot_mapping=torch.full((16,), -1, dtype=torch.int64, device=device),
        block_table_tensor=block_table,
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)
    expected = [
        -1,
        -1,
        -1,
        5,
        6,
        7,
        4,
        -1,
        -1,
        15,
        12,
        13,
        14,
        -1,
        -1,
        -1,
    ]

    assert metadata.slot_mapping.tolist() == expected

    # A dummy batch puts every request on the null block, which owns no ring.
    block_table.zero_()
    assert builder.build(0, common).slot_mapping.tolist() == [-1] * 16


@pytest.mark.parametrize("chunk_start", list(range(8)))
def test_qsa_circular_buffer_survives_one_speculative_step(chunk_start: int) -> None:
    """A speculative step must not overwrite the open group's committed keys.

    The step stores every row it computes, drafts included, before acceptance
    is known, while the next step still reads the earlier members of the group
    being compressed from the ring. A ring sized at the compression ratio makes
    those rows alias, so a rejected draft silently replaces a committed key.
    """
    compress_ratio = 4
    num_spec = 3
    capacity = compress_ratio * -(-(compress_ratio + num_spec) // compress_ratio)
    query_len = num_spec + 1

    slots = qsa_cache.circular_qsa_slot_mapping(
        torch.tensor([[1]], dtype=torch.int32),
        torch.zeros(query_len, dtype=torch.int32),
        torch.arange(chunk_start, chunk_start + query_len),
        capacity,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
    )

    committed = torch.arange(chunk_start - chunk_start % compress_ratio, chunk_start)
    assert set((slots % capacity).tolist()).isdisjoint((committed % capacity).tolist())


def _qsa_key_cache(
    block_size: int, compress_ratio: int, **kwargs
) -> qsa_cache.QSAKeyStateCache:
    return qsa_cache.QSAKeyStateCache(
        head_size=64,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=block_size),
        prefix=f"raw.{block_size}.{compress_ratio}",
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        compress_ratio=compress_ratio,
        **kwargs,
    )


def test_qsa_state_caches_adapt_the_unified_logical_layout() -> None:
    raw_cache = _qsa_key_cache(block_size=32, compress_ratio=4)
    compressed_cache = qsa_cache.QSACompressedKeyCache(
        head_size=64,
        dtype=torch.bfloat16,
        cache_config=SimpleNamespace(block_size=32),
        prefix="compressed.bind",
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
        compress_ratio=4,
    )
    raw_view = torch.empty(2, 1, 8, 64, dtype=torch.bfloat16)
    compressed_view = torch.empty(2, 1, 8, 64, dtype=torch.bfloat16)

    raw_cache.bind_kv_cache(raw_view)
    compressed_cache.bind_kv_cache(compressed_view)

    assert raw_cache.kv_cache.shape == (2, 8, 1, 64)
    assert compressed_cache.kv_cache.shape == (2, 8, 1, 64)
    assert raw_cache.kv_cache.data_ptr() == raw_view.data_ptr()
    assert compressed_cache.kv_cache.data_ptr() == compressed_view.data_ptr()


def test_clearing_qsa_key_cache_releases_its_storage() -> None:
    """Profiling teardown clears `kv_cache`; no derived view may keep it alive."""
    raw_cache = _qsa_key_cache(
        block_size=32, compress_ratio=4, cache_rope_positions=True
    )
    kv = torch.zeros(2, 1, 8, raw_cache.head_size, dtype=torch.bfloat16)
    references_before_bind = torch._C._storage_Use_Count(kv.untyped_storage()._cdata)
    raw_cache.bind_kv_cache(kv)
    assert raw_cache.key_cache.shape[-1] == 64
    assert raw_cache.rope_position_cache.dtype == torch.int64

    clear_layer_kv_caches([raw_cache])

    references = torch._C._storage_Use_Count(kv.untyped_storage()._cdata)
    assert references == references_before_bind


@pytest.mark.parametrize(
    ("compress_ratio", "num_spec", "expected"),
    [(4, 0, 4), (4, 1, 8), (4, 3, 8), (4, 4, 8), (4, 5, 12), (2, 3, 6)],
)
def test_qsa_ring_capacity_covers_one_speculative_step(
    compress_ratio: int, num_spec: int, expected: int
) -> None:
    """Capacity spans the open group plus one speculative step, in whole groups."""
    spec = _qsa_key_cache(
        block_size=48, compress_ratio=compress_ratio
    ).get_kv_cache_spec(SimpleNamespace(num_speculative_tokens=num_spec))
    assert spec.block_size == expected


def test_qsa_kv_cache_dtype_honors_skip_layers() -> None:
    """``--kv-cache-dtype-skip-layers`` keeps the listed QSA layers unquantized.

    The MTP layer's own attention is the one that matters on Flash-Next: FP8
    there cuts draft acceptance at depth while the target layers stay FP8.
    """
    cache_config = SimpleNamespace(cache_dtype="fp8", kv_cache_dtype_skip_layers=["48"])
    assert qsa_kv_cache_dtype(cache_config, "mtp.layers.48.self_attn") == "auto"
    assert qsa_kv_cache_dtype(cache_config, "model.layers.47.self_attn") == "fp8"
    cache_config.kv_cache_dtype_skip_layers = []
    assert qsa_kv_cache_dtype(cache_config, "mtp.layers.48.self_attn") == "fp8"


@requires_qsa_kernels
def test_qsa_compressed_metadata_keeps_dummy_slots_inert() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 4
    builder.reorder_batch_threshold = 1
    builder.is_circular_buffer = False
    builder.storage_block_size = 16
    builder.token_to_req_buffer = torch.empty(8, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(8, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(8, dtype=torch.int64, device=device)
    builder.visible_blocks_buffer = torch.empty(8, dtype=torch.int32, device=device)
    # Simulate max_num_seqs exceeding the three live requests below.
    builder.request_capacity = 8
    builder.k_work_metadata_buffer = torch.empty(4, 2, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 3, 3, 8], dtype=torch.int32, device=device)
    token_to_req = torch.tensor(
        [0, 0, 0, 2, 2, 2, 2, 2], dtype=torch.int32, device=device
    )
    common = SimpleNamespace(
        num_actual_tokens=8,
        num_reqs=3,
        max_query_len=5,
        max_seq_len=12,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([7, 0, 12], dtype=torch.int32, device=device),
        slot_mapping=torch.full((8,), -1, dtype=torch.int64, device=device),
        block_table_tensor=torch.zeros((3, 1), dtype=torch.int32, device=device),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)

    assert metadata.slot_mapping.tolist() == [-1] * 8
    assert metadata.visible_blocks.tolist() == [1, 1, 1, 2, 2, 2, 2, 3]
    assert metadata.k_work_metadata.tolist() == [[0, 0], [2, 0], [2, 1], [-1, -1]]


@requires_qsa_kernels
@pytest.mark.usefixtures("default_vllm_config")
def test_qsa_unfused_cache_update_ignores_padded_qk() -> None:
    """Padded projected Q/K rows must not affect either side cache."""
    from vllm.model_executor.layers.rotary_embedding import get_rope

    device = torch.device("cuda")
    # Five tokens complete one compressed group and retain four keys in the ring.
    raw_metadata = SimpleNamespace(
        num_actual_tokens=5,
        slot_mapping=torch.tensor([-1, 1, 2, 3, 0], device=device),
        block_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
        token_to_req=torch.zeros(5, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32, device=device),
        logical_positions=torch.arange(5, device=device),
    )
    compressed_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([-1, -1, -1, 0, -1], device=device),
    )
    with torch.device(device):
        rope = get_rope(
            head_size=128,
            max_position=32,
            rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
            dtype=torch.bfloat16,
        )
    raw_cache = torch.zeros((1, 4, 1, 64), dtype=torch.bfloat16, device=device)
    compressed_cache = torch.zeros((1, 2, 1, 64), dtype=torch.bfloat16, device=device)
    norm = SimpleNamespace(
        weight=torch.zeros(64, dtype=torch.bfloat16, device=device),
        variance_epsilon=1e-6,
    )
    indexer = SimpleNamespace(
        _metadata=lambda: (raw_metadata, compressed_metadata),
        skip_topk=True,
        index_kv_heads=1,
        index_n_heads=1,
        index_head_dim=64,
        indexer_dtype=torch.bfloat16,
        q_layernorm=norm,
        k_layernorm=norm,
        rotary_emb=rope,
        compress_ratio=4,
        raw_key_cache=SimpleNamespace(
            kv_cache=raw_cache, key_cache=raw_cache, rope_position_cache=None
        ),
        compressed_key_cache=SimpleNamespace(kv_cache=compressed_cache),
    )
    keys = torch.arange(1, 6, dtype=torch.bfloat16, device=device)[:, None].expand(
        5, 64
    )
    padded_keys = torch.full((8, 64), torch.nan, dtype=torch.bfloat16, device=device)
    padded_keys[:5].copy_(keys)
    indexer_qsa.QSAIndexer.forward(
        indexer,
        torch.cat((torch.ones_like(padded_keys), padded_keys), dim=-1),
        torch.zeros(8, dtype=torch.long, device=device),
        torch.full((5, 5), -1, dtype=torch.int32, device=device),
        attn=SimpleNamespace(use_fused_qsa_prepare=False),
    )
    torch.testing.assert_close(raw_cache[0, :, 0], keys[[4, 1, 2, 3]])
    expected_compressed = torch.zeros_like(compressed_cache)
    expected_compressed[0, 0] = 1
    torch.testing.assert_close(compressed_cache, expected_compressed)


@requires_qsa_kernels
@pytest.mark.parametrize("compress_ratio", [1, 4])
@pytest.mark.parametrize("num_reqs", [2, 3, 4, 7, 8, 9])
def test_qsa_triton_metadata_matches_pytorch(
    compress_ratio: int, num_reqs: int
) -> None:
    device = torch.device("cuda")
    num_tokens = 8
    query_start_loc = torch.tensor(
        [0, 3, *([3] * (num_reqs - 2)), 8], dtype=torch.int32, device=device
    )
    token_to_req = torch.tensor(
        [0, 0, 0, *([num_reqs - 1] * 5)],
        dtype=torch.int32,
        device=device,
    )
    block_table_rows = torch.tensor(
        [
            [4, -1, 8, -1, 12, -1],
            [1, -1, 2, -1, 3, -1],
            [7, -1, 9, -1, 11, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    block_table_storage = block_table_rows[
        torch.arange(num_reqs, device=device) % block_table_rows.shape[0]
    ]
    seq_lens = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    seq_lens[0] = 10
    seq_lens[-1] = 20
    common = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        slot_mapping=torch.tensor(
            [0, 1, -1, 3, 4, -1, -1, -1], dtype=torch.int64, device=device
        ),
        block_table_tensor=block_table_storage[:, ::2],
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    def make_buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
        )

    max_num_work = (
        (num_tokens + (compress_ratio - 1) * num_reqs) // compress_ratio
        if compress_ratio != 1
        else 0
    )
    actual_k_work = (
        torch.empty(max_num_work, 2, dtype=torch.int32, device=device)
        if max_num_work
        else None
    )
    actual_buffers = make_buffers()
    actual = qsa_cache.build_qsa_metadata_triton(
        common,
        *actual_buffers,
        storage_block_size=2,
        compress_ratio=compress_ratio,
        k_work_metadata_buffer=actual_k_work,
        request_capacity=num_reqs,
    )

    expected_k_work = (
        torch.empty_like(actual_k_work) if actual_k_work is not None else None
    )
    expected_buffers = make_buffers()
    expected = qsa_cache._build_qsa_metadata_torch(
        common,
        *expected_buffers,
        storage_block_size=2,
        compress_ratio=compress_ratio,
        k_work_metadata_buffer=expected_k_work,
        request_capacity=num_reqs,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    if actual_k_work is not None:
        torch.testing.assert_close(actual_k_work, expected_k_work)


@requires_qsa_kernels
def test_qsa_fused_metadata_matches_pytorch_for_large_padded_prefill() -> None:
    device = torch.device("cuda")
    num_mapped_tokens = 4096
    num_tokens = 4224
    query_start_loc = torch.tensor(
        [0, num_mapped_tokens], dtype=torch.int32, device=device
    )
    common = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor(
            [num_mapped_tokens + 32], dtype=torch.int32, device=device
        ),
        block_table_tensor=torch.arange(256, dtype=torch.int32, device=device)[None],
        slot_mapping=torch.tensor(
            [0] * num_mapped_tokens + [-1] * (num_tokens - num_mapped_tokens),
            dtype=torch.int64,
            device=device,
        ),
        token_to_req_indices=lambda buffer: buffer.zero_(),
    )

    def make_buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
        )

    max_num_work = (num_tokens + 3) // 4
    actual_k_work = torch.empty(max_num_work, 2, dtype=torch.int32, device=device)
    expected_k_work = torch.empty_like(actual_k_work)
    actual = qsa_cache.build_qsa_metadata_triton(
        common,
        *make_buffers(),
        storage_block_size=8,
        compress_ratio=4,
        k_work_metadata_buffer=actual_k_work,
    )
    expected = qsa_cache._build_qsa_metadata_torch(
        common,
        *make_buffers(),
        storage_block_size=8,
        compress_ratio=4,
        k_work_metadata_buffer=expected_k_work,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    torch.testing.assert_close(actual_k_work, expected_k_work)


@requires_qsa_kernels
@pytest.mark.parametrize(
    ("decode_query_len", "num_requests"),
    [
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (4, 33),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_qsa_decode_selection_correctness(
    decode_query_len: int, num_requests: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(1)
    heads, head_dim = 4, 128
    rows = num_requests * decode_query_len
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16).to(
        dtype
    )
    page_size, pages_per_request, max_sequence_length = (
        (16, 40, 2560) if num_requests > 32 else (4, 20, 320)
    )
    num_pages = num_requests * pages_per_request
    cache = torch.randn(
        num_pages,
        page_size,
        1,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(dtype)
    page_table = torch.randperm(num_pages, device="cuda", dtype=torch.int32).reshape(
        num_requests, pages_per_request
    )
    token_to_req = torch.repeat_interleave(
        torch.arange(num_requests, device="cuda", dtype=torch.int32),
        decode_query_len,
    )
    sequence_lengths = max_sequence_length - 4 * (
        torch.arange(num_requests, device="cuda", dtype=torch.int32) % 8
    )
    query_positions = torch.cat(
        [
            torch.arange(
                length - decode_query_len,
                length,
                device="cuda",
                dtype=torch.int32,
            )
            for length in sequence_lengths.tolist()
        ]
    )
    visible_blocks = torch.minimum(
        (query_positions + 1) // 4,
        sequence_lengths.index_select(0, token_to_req.long()) // 4,
    )

    token_topk, compress_ratio = 2048, 4
    actual = torch.empty(
        (rows, token_topk // compress_ratio), device="cuda", dtype=torch.int32
    )
    qsa_indexer_ops.qsa_select_paged_decode(
        q,
        cache,
        page_table,
        visible_blocks,
        token_topk,
        compress_ratio,
        decode_query_len,
        actual,
    )
    expected = _qsa_select_paged_reference(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )

    if dtype == torch.float8_e4m3fn:
        # fp8 logits tie at the top-k boundary more often than bf16, so index
        # identity is not stable; compare the selected value multisets.
        # SM90 wgmma accumulates fp8 in reduced precision (~3e-4 abs
        # observed); SM100 tcgen05 is exact fp32.
        rtol = atol = 1e-3 if current_platform.is_device_capability(90) else None
        logits = _qsa_mqa_paged_reference(
            q, cache, page_table, token_to_req, visible_blocks
        )
        for row in range(rows):
            selected = actual[row][actual[row] >= 0]
            wanted = expected[row][expected[row] >= 0]
            assert selected.numel() == wanted.numel()
            torch.testing.assert_close(
                logits[row, selected.long()].sort().values,
                logits[row, wanted.long()].sort().values,
                rtol=rtol,
                atol=atol,
            )
        return

    torch.testing.assert_close(actual.sort().values, expected.sort().values)


@requires_qsa_kernels
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("seq_len_slack", [0, 1792])
@pytest.mark.parametrize("force_chunk", [False, True])
def test_qsa_prefill_selection_correctness(
    monkeypatch: pytest.MonkeyPatch,
    seq_len_slack: int,
    force_chunk: bool,
    dtype: torch.dtype,
) -> None:
    # page_size=24 (does not divide the 64-aligned clipped width) and an
    # oversized page table, so the clipped logits width comes from
    # max_seq_len, not page geometry. seq_len_slack > 0 simulates the
    # spec-decode case where the bound is an over-estimate. force_chunk
    # drives the logits budget to one row per chunk.
    if force_chunk:
        monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "0")
    torch.manual_seed(2)
    query_lens = [3, 33]
    rows, heads, head_dim = sum(query_lens), 4, 128
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16).to(
        dtype
    )
    cache = torch.randn(128, 24, 1, head_dim, device="cuda", dtype=torch.bfloat16).to(
        dtype
    )
    page_table = torch.randperm(128, device="cuda", dtype=torch.int32).reshape(2, 64)
    token_to_req = torch.repeat_interleave(
        torch.arange(2, device="cuda", dtype=torch.int32),
        torch.tensor(query_lens, device="cuda"),
    )
    query_start_loc = torch.tensor([0, 3, 36], device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor([5120, 4224], device="cuda", dtype=torch.int32)
    query_positions = torch.cat(
        [
            torch.arange(length - query_len, length, device="cuda", dtype=torch.int32)
            for query_len, length in zip(
                query_lens, sequence_lengths.tolist(), strict=True
            )
        ]
    )
    token_topk, compress_ratio = 2048, 4
    visible_blocks = torch.minimum(
        (query_positions + 1) // compress_ratio,
        sequence_lengths.index_select(0, token_to_req.long()) // compress_ratio,
    )

    actual = torch.empty(
        (rows, token_topk // compress_ratio), device="cuda", dtype=torch.int32
    )
    qsa_indexer_ops.qsa_select_paged_prefill(
        q,
        cache,
        page_table,
        query_start_loc,
        visible_blocks,
        token_topk,
        compress_ratio,
        max(query_lens),
        actual,
        max_seq_len=sequence_lengths.max().item() + seq_len_slack,
    )
    expected = _qsa_select_paged_reference(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )

    if dtype == torch.float8_e4m3fn:
        # fp8 logits tie at the top-k boundary more often than bf16, so index
        # identity is not stable; compare the selected value multisets.
        # SM90 wgmma accumulates fp8 in reduced precision (~3e-4 abs
        # observed); SM100 tcgen05 is exact fp32.
        rtol = atol = 1e-3 if current_platform.is_device_capability(90) else None
        logits = _qsa_mqa_paged_reference(
            q, cache, page_table, token_to_req, visible_blocks
        )
        for row in range(rows):
            selected = actual[row][actual[row] >= 0]
            wanted = expected[row][expected[row] >= 0]
            assert selected.numel() == wanted.numel()
            torch.testing.assert_close(
                logits[row, selected.long()].sort().values,
                logits[row, wanted.long()].sort().values,
                rtol=rtol,
                atol=atol,
            )
        return

    torch.testing.assert_close(actual.sort().values, expected.sort().values)


@requires_qsa_kernels
def test_qsa_block_expansion_correctness() -> None:
    blocks = torch.tensor([[0, -1], [1, 0]], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([5, 10], device="cuda")
    sequence_lengths = torch.tensor([6, 11], device="cuda")
    token_to_req = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    visible_blocks = torch.minimum(
        (query_positions + 1) // 4,
        sequence_lengths.index_select(0, token_to_req.long()) // 4,
    ).to(torch.int32)

    # Packed layout: one trailing column per row holds the valid-entry count
    # (never a token index). Row 0: 1 visible block + 2 tail; row 1: 2 blocks
    # + 3 tail.
    actual = torch.empty((2, 12), device="cuda", dtype=torch.int32)
    qsa_indexer_ops.expand_qsa_block_indices(
        blocks,
        query_positions,
        visible_blocks,
        compress_ratio=4,
        token_topk=8,
        out=actual,
    )
    expected = _expand_qsa_indices_reference(
        blocks,
        query_positions,
        sequence_lengths,
        compress_ratio=4,
        token_topk=8,
    )

    torch.testing.assert_close(actual[:, :11], expected)
    assert actual[:, 11].tolist() == [6, 11]


@requires_qsa_kernels
@pytest.mark.parametrize(
    (
        "num_rows",
        "num_query_heads",
        "num_kv_heads",
        "page_size",
        "use_prefill_config",
        "num_requests",
        "fp8",
    ),
    [
        # Production page sizes from hybrid-cache block alignment: 784/800
        # at TP4 and 1568/1600 at TP1/TP2 (no-MTP / MTP num_spec=3). Head
        # splits are per-rank TP1/TP2/TP4; the largest batch runs both
        # use_prefill_config variants.
        pytest.param(1, 24, 2, 1600, True, 2, False, id="tp1_r1"),
        pytest.param(16, 12, 1, 1600, True, 3, False, id="tp2_r16"),
        pytest.param(32, 6, 1, 800, True, 5, False, id="tp4_r32"),
        pytest.param(128, 24, 2, 1568, True, 7, False, id="tp1_r128"),
        pytest.param(257, 6, 1, 800, True, 13, False, id="tp4_r257"),
        pytest.param(513, 6, 1, 784, True, 17, False, id="tp4_r513"),
        pytest.param(700, 6, 1, 800, True, 23, False, id="tp4_r700"),
        pytest.param(1024, 24, 2, 1600, True, 33, False, id="tp1_r1024"),
        pytest.param(2048, 24, 2, 1600, True, 63, False, id="tp1_r2048_prefill"),
        pytest.param(2048, 24, 2, 1600, False, 63, False, id="tp1_r2048_uniform"),
        # fp8_e4m3 K/V caches on the TP1 head split.
        pytest.param(1, 24, 2, 1600, True, 2, True, id="tp1_r1_fp8"),
        pytest.param(128, 24, 2, 1568, True, 7, True, id="tp1_r128_fp8"),
        pytest.param(2048, 24, 2, 1600, True, 63, True, id="tp1_r2048_prefill_fp8"),
        pytest.param(2048, 24, 2, 1600, False, 63, True, id="tp1_r2048_uniform_fp8"),
    ],
)
def test_qsa_sparse_paged_attention_correctness(
    num_rows: int,
    num_query_heads: int,
    num_kv_heads: int,
    page_size: int,
    use_prefill_config: bool,
    num_requests: int,
    fp8: bool,
) -> None:
    """QSA sparse paged attention matches the dense reference.

    fp8 only changes the K/V cache dtype (e4m3 with a per-tensor scale pair) and
    the scales; the reference dequantizes the same cache with those scales, so
    both paths compare the production kernel against the reference on identical
    inputs. fp8=True additionally covers the host-side scale folding.
    """
    torch.manual_seed(2)
    # One QSA attention problem: bf16 Q and paged K/V, a packed selection with
    # the trailing count column, block table and row-to-request map.
    head_dim = 256
    num_selected_pages = 64
    # Keep the newest page outside the synthetic top-k as causal headroom.
    num_pages_per_request = num_selected_pages + 1
    num_cache_blocks = num_requests * num_pages_per_request
    indexer_budget = 2048
    indexer_compress_ratio = 4
    selection_width = indexer_budget + indexer_compress_ratio - 1
    q = torch.randn(
        num_rows, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    output_gate = torch.randn_like(q)
    kv_cache = torch.randn(
        num_cache_blocks,
        page_size,
        num_kv_heads,
        2 * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    block_table = (
        torch.randperm(num_cache_blocks, device="cuda")
        .reshape(num_requests, num_pages_per_request)
        .to(torch.int32)
    )
    rows_per_request = math.ceil(num_rows / num_requests)
    row_indices = torch.arange(num_rows, device="cuda", dtype=torch.int32)
    token_to_req = row_indices // rows_per_request
    # Uniform row split; the last request takes the remainder (possibly 0).
    request_row_counts = torch.full(
        (num_requests,), rows_per_request, device="cuda", dtype=torch.int32
    )
    request_row_counts[-1] = num_rows - rows_per_request * (num_requests - 1)

    # Mix context lengths: every third request is short-context, attending
    # to only its first few pages; the rest fill their cache.
    context_lengths = torch.full(
        (num_requests,),
        num_pages_per_request * page_size - 1,
        device="cuda",
        dtype=torch.int32,
    )
    short_requests = torch.arange(num_requests, device="cuda") % 3 == 1
    context_lengths[short_requests] = request_row_counts[short_requests] + 8
    block_topk = indexer_budget // indexer_compress_ratio
    compressed_blocks_per_page = page_size // indexer_compress_ratio
    selection = torch.arange(block_topk, device="cuda")
    selected_pages = selection % num_selected_pages
    selected_offsets = selection // num_selected_pages
    row_shifts = 2 * row_indices.unsqueeze(1)
    # Eight blocks per page; adjacent rows overlap by six of those eight.
    selected_offsets = (selected_offsets + row_shifts) % compressed_blocks_per_page
    block_indices = (selected_pages * compressed_blocks_per_page + selected_offsets).to(
        torch.int32
    )
    rows_within_request = row_indices % rows_per_request
    query_positions = (
        context_lengths[token_to_req.long()]
        - request_row_counts[token_to_req.long()]
        + rows_within_request
    ).to(torch.int64)
    sequence_lengths = context_lengths
    visible_blocks = torch.minimum(
        (query_positions + 1) // indexer_compress_ratio,
        sequence_lengths.index_select(0, token_to_req.long()) // indexer_compress_ratio,
    ).to(torch.int32)
    # +1: the packed trailing column holds each row's valid-entry count
    # (never a token index); the reference reads only the selection region.
    logical_indices = torch.empty(
        (num_rows, selection_width + 1), device="cuda", dtype=torch.int32
    )
    qsa_indexer_ops.expand_qsa_block_indices(
        block_indices,
        query_positions,
        visible_blocks,
        indexer_compress_ratio,
        indexer_budget,
        logical_indices,
    )

    scale = head_dim**-0.5

    if fp8:
        # A fixed non-unit pair (k != v) exercises the host-side scale folding
        # and catches a k/v swap; scales are host floats, as the layer exposes
        # them. Stored values are the scaled ones, as reshape_and_cache does.
        k_scale, v_scale = 0.5, 2.0
        k_cache = (k_cache / k_scale).to(torch.float8_e4m3fn)
        v_cache = (v_cache / v_scale).to(torch.float8_e4m3fn)
    else:
        k_scale, v_scale = 1.0, 1.0

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=use_prefill_config,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
    )
    expected = _qsa_sparse_paged_attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices[:, :selection_width],
        block_table,
        token_to_req,
        scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    expected = expected * torch.sigmoid(output_gate)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_qsa_kernels
@pytest.mark.parametrize("extra_rows", [0, 1], ids=["flat-pages", "strided-pages"])
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_qsa_attention_consumes_hisparse_physical_indices(
    extra_rows: int, fp8: bool
) -> None:
    """Resolved HiSparse rows bypass the logical block table in either view."""
    from vllm.v1.hisparse.runtime import PagedCacheView

    torch.manual_seed(7)
    num_blocks, block_size, num_kv_heads, head_dim = 4, 16, 2, 64
    row_width = num_kv_heads * 2 * head_dim
    block_stride = (2 * block_size + extra_rows) * row_width
    storage_dtype = torch.uint8 if fp8 else torch.bfloat16
    k_scale, v_scale = (0.5, 2.0) if fp8 else (1.0, 1.0)
    backing = torch.full(
        (num_blocks * block_stride,), 7, device="cuda", dtype=storage_dtype
    )
    view = PagedCacheView.bind(
        backing.view(torch.uint8),
        dtype=backing.dtype,
        row_width=row_width,
        byte_offset=block_size * row_width * backing.element_size(),
        block_stride=block_stride * backing.element_size(),
        num_blocks=num_blocks,
        block_size=block_size,
    )
    original = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        2 * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    original_key, original_value = original.split(head_dim, dim=-1)
    if fp8:
        original_key = (original_key.float() / k_scale).to(torch.float8_e4m3fn)
        original_value = (original_value.float() / v_scale).to(torch.float8_e4m3fn)
    stored = torch.cat(
        (original_key.view(storage_dtype), original_value.view(storage_dtype)), dim=-1
    )
    view.cache.copy_(stored.flatten(2))
    attention_cache = view.attention_cache.unflatten(-1, (num_kv_heads, 2 * head_dim))
    key_cache, value_cache = attention_cache.split(head_dim, dim=-1)
    if fp8:
        key_cache = key_cache.view(torch.float8_e4m3fn)
        value_cache = value_cache.view(torch.float8_e4m3fn)
    selected = torch.tensor(
        [[55, 2, 43, -1, -1], [16, 63, 33, 6, 30], [-1, -1, -1, -1, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    counts = torch.tensor([[3], [5], [0]], device="cuda", dtype=torch.int32)
    physical_rows = torch.where(
        selected >= 0,
        selected // block_size * view.attention_block_stride + selected % block_size,
        -1,
    )
    packed = torch.cat((physical_rows, counts), dim=1)
    saved_packed = packed.clone()
    q = torch.randn(3, 24, head_dim, device="cuda", dtype=torch.bfloat16)
    output_gate = torch.randn_like(q)
    token_to_req = torch.zeros(3, device="cuda", dtype=torch.int32)
    identity = torch.arange(num_blocks, device="cuda", dtype=torch.int32)[None]
    expected = _qsa_sparse_paged_attention_reference(
        q,
        original_key,
        original_value,
        selected,
        identity,
        token_to_req,
        head_dim**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
    ) * torch.sigmoid(output_gate)
    block_table = torch.arange(
        attention_cache.shape[0], device="cuda", dtype=torch.int32
    ).roll(1)[None]
    logical_expected = _qsa_sparse_paged_attention_reference(
        q,
        key_cache,
        value_cache,
        physical_rows,
        block_table,
        token_to_req,
        head_dim**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
    ) * torch.sigmoid(output_gate)
    assert not torch.allclose(expected, logical_expected, rtol=2e-2, atol=2e-2)
    logical_output = qsa_ops.qsa_sparse_paged_attention(
        q,
        key_cache,
        value_cache,
        packed,
        block_table,
        token_to_req,
        use_prefill_config=False,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
    )
    torch.testing.assert_close(logical_output, logical_expected, rtol=2e-2, atol=2e-2)

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        key_cache,
        value_cache,
        packed,
        block_table,
        token_to_req,
        use_prefill_config=False,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
        physical_indices=True,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(packed, saved_packed, rtol=0, atol=0)


@requires_qsa_kernels
@pytest.mark.parametrize("fp8", [False, True], ids=["main-bf16", "main-fp8"])
@pytest.mark.parametrize(
    "indexer_fp8", [False, True], ids=["indexer-bf16", "indexer-fp8"]
)
def test_qsa_hisparse_prefill_ignores_unused_request_table_rows(
    tmp_path, dist_init, workspace_init, monkeypatch, fp8: bool, indexer_fp8: bool
) -> None:
    """Unequal active lengths must not index permanent, inactive table rows."""
    _run_qsa_hisparse_prefill_case(tmp_path, monkeypatch, fp8, indexer_fp8)


@requires_qsa_kernels
def test_qsa_hisparse_capture_does_not_submit_dummy_host_mirrors(
    tmp_path, dist_init, workspace_init, monkeypatch
) -> None:
    """Warmup and capture cannot consume the real worker's mirror phase."""
    _run_qsa_hisparse_prefill_case(
        tmp_path, monkeypatch, False, False, capture_warmup=True
    )


@requires_qsa_kernels
def test_qsa_hisparse_jit_warmup_respects_disabled_connector(
    tmp_path, dist_init, workspace_init, monkeypatch
) -> None:
    """Ordinary prefill and K+1 warmup cannot consume a host-mirror phase."""
    _run_qsa_hisparse_prefill_case(
        tmp_path, monkeypatch, False, False, connector_warmup=True
    )


def _run_qsa_hisparse_prefill_case(
    tmp_path,
    monkeypatch,
    fp8: bool,
    indexer_fp8: bool,
    *,
    capture_warmup=False,
    connector_warmup=False,
) -> None:
    from dataclasses import replace

    from transformers import Qwen4ExpTextConfig

    from vllm.config import AttentionConfig, HiSparseConfig, set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.hisparse.runtime import (
        allocate_pinned_host_pool,
        release_pinned_state,
    )

    torch.manual_seed(7)
    num_blocks, block_size, heads, dim = 8, 16, 2, 64
    config = Qwen4ExpTextConfig(
        architectures=["Qwen4ExpForCausalLM"],
        num_hidden_layers=1,
        layer_types=["full_attention"],
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=24,
        num_key_value_heads=heads,
        head_dim=dim,
        partial_rotary_factor=1.0,
        num_experts=4,
        num_experts_per_tok=2,
        ple_layer_ids=[],
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
        indexer_budget=2048,
    )
    config.save_pretrained(tmp_path)
    vllm_config = EngineArgs(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        kv_cache_dtype="fp8_e4m3" if fp8 else "bfloat16",
        block_size=block_size,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        attention_config=AttentionConfig(
            hisparse_config=HiSparseConfig(),
            indexer_kv_dtype="fp8" if indexer_fp8 else "bf16",
        ),
    ).create_engine_config()
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        owner = Qwen4ExpQSAAttention(
            vllm_config=vllm_config,
            config=config,
            layer_id=0,
            prefix="model.layers.0.self_attn",
        )
        k_scale, v_scale = (0.5, 2.0) if fp8 else (1.0, 1.0)
        AutoWeightsLoader(owner).load_weights(
            [
                ("_k_scale", torch.tensor(k_scale)),
                ("_v_scale", torch.tensor(v_scale)),
            ]
        )
        owner.process_weights_after_loading(torch.bfloat16)
    # Even when fusion is supported, main KV writes must use the HiSparse
    # resident target exercised below instead of the ordinary cache.
    assert owner.use_fused_qk_norm_rope_gate
    assert owner.indexer.use_fused_pre_indexer
    assert not owner.use_fused_qsa_prepare
    row_width = heads * 2 * dim
    original = torch.randn(
        num_blocks,
        block_size,
        heads,
        2 * dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    original[6].fill_(17)
    original[7].fill_(-19)
    if fp8:
        original_key, original_value = original.split(dim, dim=-1)
        original = torch.cat(
            (
                (original_key.float() / k_scale)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8),
                (original_value.float() / v_scale)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8),
            ),
            dim=-1,
        )
    backing = original.flatten().clone()
    source_table = torch.tensor(
        [[1, 2, 0], [3, 4, 5], [6, 6, 6], [7, 7, 7]],
        device="cuda",
        dtype=torch.int32,
    )
    resident_table = source_table.clone()
    resident_table[1, 0] = 0
    # Residency uses persistent state rows, independent of this batch's order.
    state_indices = torch.tensor([1, 0, 2, 3], device="cuda", dtype=torch.int32)
    resident_table = resident_table.index_select(0, state_indices)
    slots = torch.tensor([31, 32, 79, 80], device="cuda", dtype=torch.int64)
    requests = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int32)
    positions = torch.tensor([15, 16, 31, 32], device="cuda", dtype=torch.int64)
    selection = torch.arange(33, device="cuda", dtype=torch.int32).expand(4, -1)
    selection = selection.masked_fill(selection > positions[:, None], -1)
    q = torch.zeros(4, 24, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(4, heads, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    hidden = torch.eye(4, 256, device="cuda", dtype=torch.bfloat16)
    # Real projections produce Q=K=gate=0 and the specified V rows. Output
    # projection selects two Q heads from each KV head, exercising both shards.
    output_columns = torch.cat((torch.arange(128), torch.arange(768, 896))).cuda()
    with torch.no_grad():
        for parameter in owner.parameters():
            parameter.zero_()
        owner.qkv_proj.weight[
            2 * owner.q_size + owner.kv_size : 2 * owner.q_size + 2 * owner.kv_size,
            :4,
        ].copy_(value.flatten(1).transpose(0, 1))
        owner.o_proj.weight[torch.arange(256, device="cuda"), output_columns] = 1
    common = CommonAttentionMetadata(
        num_actual_tokens=4,
        num_reqs=2,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        max_seq_len=33,
        seq_lens=torch.tensor([17, 33], device="cuda", dtype=torch.int32),
        block_table_tensor=source_table[:2],
        slot_mapping=slots,
    )
    builder = owner.get_attn_backend().get_builder_cls()(
        owner.get_kv_cache_spec(vllm_config),
        [owner.layer_name],
        vllm_config,
        torch.device("cuda"),
    )
    metadata = (
        builder.build_for_cudagraph_capture(common)
        if capture_warmup
        else builder.build(0, common)
    )
    all_metadata = {owner.layer_name: metadata}
    for side_cache in (
        owner.indexer.raw_key_cache,
        owner.indexer.compressed_key_cache,
    ):
        spec = side_cache.get_kv_cache_spec(vllm_config)
        side_cache.bind_kv_cache(
            torch.zeros(
                num_blocks,
                1,
                spec.num_states,
                128,
                device="cuda",
                dtype=spec.dtype,
            )
        )
        table = (
            torch.tensor([[1], [2]], device="cuda", dtype=torch.int32)
            if side_cache is owner.indexer.raw_key_cache
            else source_table[:2]
        )
        all_metadata[side_cache.prefix] = QSAMetadataBuilder(
            spec, [side_cache.prefix], vllm_config, torch.device("cuda")
        ).build(0, replace(common, block_table_tensor=table))
    cache = owner.hisparse_cache
    assert cache is not None
    runtime = cache.runtime
    cache.bind_cache(
        backing.view(torch.uint8),
        byte_offset=0,
        block_stride=block_size * row_width * backing.element_size(),
        num_blocks=num_blocks,
        block_size=block_size,
        block_table=resident_table,
        slot_mapping=slots,
    )
    cache.source_block_table = source_table
    runtime.request_state_indices = state_indices
    cache.all_context_pages_resident = False
    owner.bind_kv_cache(original.transpose(1, 2))
    if capture_warmup or connector_warmup:
        from functools import partial
        from unittest.mock import MagicMock

        from vllm.distributed.kv_transfer import kv_transfer_state
        from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
            HiSparseConnectorWorker,
        )
        from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

        # Exercise the production layer-once guard. Mock only DMA submission;
        # actual cache writing, sparse attention and native graph replay run.
        worker = object.__new__(HiSparseConnectorWorker)
        worker.cache_handles = [cache]
        worker._per_layer_mirrored = set()
        worker._submitted_mirror_layers = set()
        worker._slot_mapping_staging = None
        worker._layer_ready_events = (MagicMock(),)
        worker._enqueue_row_dma = MagicMock()
        connector = object.__new__(ActiveKVConnector)
        connector.kv_connector = MagicMock()
        monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", None)
        connector.set_disabled(False)
        if capture_warmup:
            cache.submit_layer_mirror = partial(worker._enqueue_layer_mirror, 0)
    selected_before_attention = []
    writer_inputs = []
    update_cache = owner.impl.do_kv_cache_update

    def capture_writer(layer, key, value, kv_cache, slot_mapping):
        writer_inputs.append((key.clone(), value.clone(), slot_mapping.clone()))
        update_cache(layer, key, value, kv_cache, slot_mapping)

    monkeypatch.setattr(owner.impl, "do_kv_cache_update", capture_writer)

    def save_selection(module, inputs, output):
        selected, main_outputs = output
        assert main_outputs is None
        selected_before_attention.append(selected.clone())

    selection_hook = owner.indexer.register_forward_hook(save_selection)
    host, registered = allocate_pinned_host_pool(original.nbytes)
    try:
        host_rows = host.view(original.dtype).view(-1, row_width)
        host_rows.copy_(original.flatten(2).reshape(-1, row_width).cpu())
        runtime.bind_source_cache(host_rows, registered_host_pool=registered)
        assert cache.view is not None
        cache.view.cache[3].fill_(29)  # Only host retains this selected history page.
        reference = original.clone()
        reference_key, reference_value = key, value
        if fp8:
            reference_key = (
                (key.float() / k_scale).to(torch.float8_e4m3fn).view(torch.uint8)
            )
            reference_value = (
                (value.float() / v_scale).to(torch.float8_e4m3fn).view(torch.uint8)
            )
        new_rows = torch.cat((reference_key, reference_value), dim=-1)
        reference[slots // block_size, slots % block_size] = new_rows
        reference_k, reference_v = reference.split(dim, dim=-1)
        if fp8:
            reference_k = reference_k.view(torch.float8_e4m3fn)
            reference_v = reference_v.view(torch.float8_e4m3fn)
        expected = (
            _qsa_sparse_paged_attention_reference(
                q,
                reference_k,
                reference_v,
                selection,
                source_table[:2],
                requests,
                dim**-0.5,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            * 0.5
        )
        expected = expected.flatten(1)[:, output_columns]
        # Two active requests need five pages; max-length staging reserves six.
        # The unused capacity must not index row 2 of the two-row seq_lens.
        with (
            torch.inference_mode(),
            set_current_vllm_config(vllm_config),
            set_forward_context(all_metadata, vllm_config, num_tokens=4),
        ):
            actual = owner(positions, hidden)
            if capture_warmup:
                # The real FULL manager calls NONE for both its warmup and
                # native capture body, without a connector start/finish pair.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = owner(positions, hidden)
                graph.replay()
                worker._enqueue_row_dma.assert_not_called()
                assert not worker._per_layer_mirrored
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        assert len(writer_inputs) == (2 if capture_warmup else 1)
        written_key, written_value, written_slots = writer_inputs[-1]
        torch.testing.assert_close(written_slots, slots, rtol=0, atol=0)
        torch.testing.assert_close(written_key, key, rtol=0, atol=0)
        torch.testing.assert_close(written_value, value, rtol=0, atol=0)
        # RoPE can produce negative zero from the fixture's zero K. Compare
        # storage bytes against the real writer inputs, preserving those signs.
        if fp8:
            written_key = (
                (written_key.float() / k_scale)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            written_value = (
                (written_value.float() / v_scale)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
        written_rows = torch.cat((written_key, written_value), dim=-1)
        torch.testing.assert_close(
            cache.view.cache[slots // block_size, slots % block_size].view(torch.uint8),
            written_rows.flatten(1).view(torch.uint8),
            rtol=0,
            atol=0,
        )
        assert len(selected_before_attention) == (2 if capture_warmup else 1)
        packed = selected_before_attention[-1]
        torch.testing.assert_close(
            owner.topk_indices_buffer[:4], packed, rtol=0, atol=0
        )
        torch.testing.assert_close(packed[:, -1], positions + 1, check_dtype=False)
        for row, position in zip(packed[:, :-1], positions):
            torch.testing.assert_close(
                row[row >= 0].sort().values,
                torch.arange(position + 1, device="cuda", dtype=torch.int32),
            )
        assert torch.all(cache.view.cache[3] == 29)
        mirror_metadata = all_metadata
        mirror_positions, mirror_hidden = positions, hidden
        if connector_warmup:
            cache.submit_layer_mirror = partial(worker._enqueue_layer_mirror, 0)
            cache.all_context_pages_resident = True
            connector.set_disabled(True)
            # Match JIT warmup's ordinary prefill5 followed by MTP verify4.
            # Neither forward has capture metadata or a connector mirror phase.
            for start, length in ((0, 5), (5, 4)):
                warm_positions = torch.arange(start, start + length, device="cuda")
                warm_slots = block_size + warm_positions
                warm_common = CommonAttentionMetadata(
                    num_actual_tokens=length,
                    num_reqs=1,
                    max_query_len=length,
                    query_start_loc=torch.tensor(
                        [0, length], device="cuda", dtype=torch.int32
                    ),
                    query_start_loc_cpu=torch.tensor([0, length], dtype=torch.int32),
                    max_seq_len=start + length,
                    seq_lens=torch.tensor(
                        [start + length], device="cuda", dtype=torch.int32
                    ),
                    block_table_tensor=source_table[:1],
                    slot_mapping=warm_slots,
                )
                warm_metadata = {owner.layer_name: builder.build(0, warm_common)}
                assert not warm_metadata[owner.layer_name].is_cudagraph_capture
                for side_cache in (
                    owner.indexer.raw_key_cache,
                    owner.indexer.compressed_key_cache,
                ):
                    spec = side_cache.get_kv_cache_spec(vllm_config)
                    table = (
                        source_table[:1, :1]
                        if side_cache is owner.indexer.raw_key_cache
                        else source_table[:1]
                    )
                    warm_metadata[side_cache.prefix] = QSAMetadataBuilder(
                        spec, [side_cache.prefix], vllm_config, torch.device("cuda")
                    ).build(0, replace(warm_common, block_table_tensor=table))
                cache.slot_mapping = warm_slots
                warm_hidden = hidden[warm_positions % hidden.shape[0]]
                with (
                    torch.inference_mode(),
                    set_current_vllm_config(vllm_config),
                    set_forward_context(warm_metadata, vllm_config, num_tokens=length),
                ):
                    owner(warm_positions, warm_hidden)
                connector.finish_forward()
            worker._enqueue_row_dma.assert_not_called()
            assert not worker._per_layer_mirrored
            connector.kv_connector.finish_forward.assert_not_called()
            connector.set_disabled(False)
            mirror_metadata = warm_metadata
            mirror_positions, mirror_hidden = warm_positions, warm_hidden
        if capture_warmup or connector_warmup:
            # A real multi-token prefill still pipelines one host submission,
            # and a second live write in that phase must remain an error.
            if capture_warmup:
                mirror_metadata[owner.layer_name] = builder.build(0, common)
            with (
                torch.inference_mode(),
                set_current_vllm_config(vllm_config),
                set_forward_context(mirror_metadata, vllm_config, num_tokens=4),
            ):
                owner(mirror_positions, mirror_hidden)
                worker._enqueue_row_dma.assert_called_once()
                with pytest.raises(RuntimeError, match="layer 0 mirrored twice"):
                    owner(mirror_positions, mirror_hidden)
    finally:
        if capture_warmup or connector_warmup:
            connector.set_disabled(False)
        selection_hook.remove()
        release_pinned_state([], [registered])


@requires_qsa_kernels
@pytest.mark.parametrize("fp8", [False, True], ids=["main-bf16", "main-fp8"])
@pytest.mark.parametrize(
    "indexer_fp8", [False, True], ids=["indexer-bf16", "indexer-fp8"]
)
def test_qsa_hisparse_full_graph_verify_reads_reclaimed_history(
    tmp_path, dist_init, workspace_init, monkeypatch, record_property, fp8, indexer_fp8
) -> None:
    """A captured K+1 graph follows a real worker spill and peer page reuse."""
    import json
    from dataclasses import replace

    from transformers import Qwen4ExpTextConfig

    from vllm.config import AttentionConfig, HiSparseConfig, set_current_vllm_config
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
        HiSparseConnectorWorker,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.hisparse.runtime import (
        allocate_pinned_host_pool,
        initialize_hisparse_runtime_buffers,
        release_pinned_state,
    )
    from vllm.v1.hisparse.types import (
        SparseKVOffloadCommand,
        SparseKVPageTransfer,
        SparseKVResidencyUpdate,
        SparseKVRowMirror,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

    torch.manual_seed(17)
    device = torch.device("cuda")
    blocks, block_size, heads, dim, tokens = 8, 16, 2, 64, 8
    config = Qwen4ExpTextConfig(
        architectures=["Qwen4ExpForCausalLM"],
        num_hidden_layers=1,
        mtp_num_hidden_layers=1,
        mtp={"hybrid": True},
        index_share_for_mtp_iteration=True,
        layer_types=["full_attention"],
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=24,
        num_key_value_heads=heads,
        head_dim=dim,
        num_experts=4,
        num_experts_per_tok=2,
        ple_layer_ids=[],
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
        indexer_budget=2048,
    )
    config.save_pretrained(tmp_path)
    vllm_config = EngineArgs(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        kv_cache_dtype="fp8_e4m3" if fp8 else "bfloat16",
        block_size=block_size,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        enforce_eager=False,
        async_scheduling=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [tokens],
        },
        attention_config=AttentionConfig(
            hisparse_config=HiSparseConfig(),
            indexer_kv_dtype="fp8" if indexer_fp8 else "bf16",
        ),
    ).create_engine_config()
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device(device),
    ):
        owner = Qwen4ExpQSAAttention(
            vllm_config=vllm_config,
            config=config,
            layer_id=0,
            prefix="model.layers.0.self_attn",
        )
        k_scale, v_scale = (0.5, 2.0) if fp8 else (1.0, 1.0)
        AutoWeightsLoader(owner).load_weights(
            [
                ("_k_scale", torch.tensor(k_scale)),
                ("_v_scale", torch.tensor(v_scale)),
            ]
        )
        owner.process_weights_after_loading(torch.bfloat16)
    cache = owner.hisparse_cache
    assert cache is not None
    row_width = heads * 2 * dim
    original = torch.randn(
        blocks, block_size, heads, 2 * dim, device=device, dtype=torch.bfloat16
    )

    def cache_rows(rows):
        if not fp8:
            return rows
        key, value = rows.split(dim, dim=-1)
        return torch.cat(
            (
                (key.float() / k_scale).to(torch.float8_e4m3fn).view(torch.uint8),
                (value.float() / v_scale).to(torch.float8_e4m3fn).view(torch.uint8),
            ),
            dim=-1,
        )

    original = cache_rows(original)
    backing = original.flatten().clone()
    source_table = torch.tensor(
        [[1, 2, 0], [3, 4, 5]], device=device, dtype=torch.int32
    )
    residency = source_table.flip(0).clone().unsqueeze(1)
    resident_table = residency[:, 0]
    saved_source_table = source_table.clone()
    # K+1 verification rows end in each request's live resident tail.
    initial_positions = torch.tensor([28, 29, 30, 31, 44, 45, 46, 47], device=device)
    positions = initial_positions.clone()
    slots = torch.tensor([44, 45, 46, 47, 92, 93, 94, 95], device=device)
    host_slots = slots.clone()
    saved_slots = slots.clone()
    query_starts = torch.tensor([0, 4, 8], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([32, 48], device=device, dtype=torch.int32)
    raw_table = torch.tensor([[1], [2]], device=device, dtype=torch.int32)
    hidden = torch.eye(tokens, 256, device=device, dtype=torch.bfloat16)
    original_hidden = hidden.clone()
    values = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    output_columns = torch.cat((torch.arange(128), torch.arange(768, 896))).to(device)
    with torch.no_grad():
        for parameter in owner.parameters():
            parameter.zero_()
        owner.qkv_proj.weight[
            2 * owner.q_size + owner.kv_size : 2 * owner.q_size + 2 * owner.kv_size,
            :tokens,
        ].copy_(values.flatten(1).transpose(0, 1))
        owner.o_proj.weight[torch.arange(256, device=device), output_columns] = 1
    cache.bind_cache(
        backing.view(torch.uint8),
        byte_offset=0,
        block_stride=block_size * row_width * backing.element_size(),
        num_blocks=blocks,
        block_size=block_size,
        block_table=resident_table,
        slot_mapping=slots,
    )
    cache.source_block_table = source_table
    cache.residency = residency
    cache.mirror_slot_mapping = host_slots
    cache.runtime.resident_source_index = 0
    # This multi-query path uses prefill staging, but the public worker owns a
    # normally sized hot allocation as well (including its shutdown lifetime).
    hot_blocks_per_request = (
        cache.runtime.region_stride + block_size - 1
    ) // block_size
    hot_blocks = 2 * hot_blocks_per_request
    hot_backing = torch.zeros(
        hot_blocks * block_size * row_width * original.element_size(),
        dtype=torch.uint8,
        device=device,
    )
    cache.runtime.bind_hot_cache(
        hot_backing,
        byte_offset=0,
        block_stride=block_size * row_width * original.element_size(),
        num_blocks=hot_blocks,
        block_size=block_size,
        block_table=torch.arange(hot_blocks, device=device, dtype=torch.int32).view(
            2, -1
        ),
    )
    owner.bind_kv_cache(original.transpose(1, 2))
    builder = owner.get_attn_backend().get_builder_cls()(
        owner.get_kv_cache_spec(vllm_config), [owner.layer_name], vllm_config, device
    )
    side_builders = []
    for side in (owner.indexer.raw_key_cache, owner.indexer.compressed_key_cache):
        spec = side.get_kv_cache_spec(vllm_config)
        side.bind_kv_cache(
            torch.zeros(
                blocks, 1, spec.num_states, 128, device=device, dtype=spec.dtype
            )
        )
        side_builders.append(
            QSAMetadataBuilder(spec, [side.prefix], vllm_config, device)
        )
    assert owner.indexer.raw_key_cache.get_kv_cache_spec(vllm_config).block_size == 8
    assert owner.indexer.raw_key_cache.kv_cache.dtype == torch.bfloat16
    assert owner.indexer.compressed_key_cache.kv_cache.dtype == (
        torch.float8_e4m3fn if indexer_fp8 else torch.bfloat16
    )
    all_metadata = {}

    def build_metadata(*, capture=False, active=2):
        common = CommonAttentionMetadata(
            num_actual_tokens=tokens,
            num_reqs=2,
            max_query_len=4,
            query_start_loc=query_starts,
            query_start_loc_cpu=torch.tensor([0, 4, 4 * active], dtype=torch.int32),
            max_seq_len=128,
            seq_lens=seq_lens,
            block_table_tensor=source_table,
            slot_mapping=host_slots,
        )
        metadata = (
            builder.build_for_cudagraph_capture(common)
            if capture
            else builder.build(0, common)
        )
        assert metadata.num_decode_tokens == 0
        assert metadata.num_prefill_tokens == tokens
        all_metadata[owner.layer_name] = metadata
        for side, side_builder, table in zip(
            (owner.indexer.raw_key_cache, owner.indexer.compressed_key_cache),
            side_builders,
            (raw_table, source_table),
        ):
            all_metadata[side.prefix] = side_builder.build(
                0, replace(common, block_table_tensor=table)
            )
        return metadata

    written = []
    original_writer = owner.impl.do_kv_cache_update

    def observe_writer(layer, key, value, kv_cache, slot_mapping):
        written.append(torch.cat((key, value), dim=-1).clone())
        return original_writer(layer, key, value, kv_cache, slot_mapping)

    monkeypatch.setattr(owner.impl, "do_kv_cache_update", observe_writer)
    staging_observations = []
    native_gather = torch.ops._C_cache_ops.hisparse_gather_plan

    def observe_gather(host_cache, staged, rows, destinations, misses, *args):
        result = native_gather(host_cache, staged, rows, destinations, misses, *args)
        staging_observations.append((staged.nbytes, misses.clone(), staged.clone()))
        return result

    monkeypatch.setattr(torch.ops._C_cache_ops, "hisparse_gather_plan", observe_gather)
    pool = BlockPool(blocks, enable_caching=False, hash_block_size=block_size)
    allocated = pool.get_new_blocks(blocks - 1)
    reclaimed = next(block for block in allocated if block.block_id == 1)
    host, registered = allocate_pinned_host_pool(original.nbytes)
    worker = None
    graph = None
    try:
        host_pages = host.view(original.dtype).view_as(original)
        host_pages.copy_(original.cpu())
        cache.runtime.bind_source_cache(
            host_pages.flatten(2), registered_host_pool=registered
        )
        initialize_hisparse_runtime_buffers([cache], max_num_reqs=2)
        worker = HiSparseConnectorWorker(
            vllm_config,
            KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=[]),
        )
        worker.initialize(
            [cache], [owner.layer_name], hot_backing, 2, blocks, device, [registered]
        )
        worker.set_request_state_indices(
            torch.tensor([1, 0], dtype=torch.int32, device=device)
        )
        cache.all_context_pages_resident = True
        build_metadata(capture=True)
        with (
            torch.inference_mode(),
            set_current_vllm_config(vllm_config),
            set_forward_context(all_metadata, vllm_config, num_tokens=tokens),
        ):
            owner(positions, hidden)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = owner(positions, hidden)
            graph.replay()
        assert len(written) == 2
        # The graph also rewrites its observation buffer on every replay.
        initial_written = written[-1].clone()
        torch.testing.assert_close(
            initial_written[..., :dim], torch.zeros_like(values), rtol=0, atol=0
        )
        torch.testing.assert_close(initial_written[..., dim:], values, rtol=0, atol=0)
        reference = original.clone()
        reference[saved_slots // block_size, saved_slots % block_size] = cache_rows(
            initial_written
        )

        def attention_reference(verify_positions, request_rows):
            logical = torch.arange(48, device=device, dtype=torch.int32).expand(
                verify_positions.numel(), -1
            )
            logical = logical.masked_fill(logical > verify_positions[:, None], -1)
            reference_k, reference_v = reference.split(dim, dim=-1)
            if fp8:
                reference_k = reference_k.view(torch.float8_e4m3fn)
                reference_v = reference_v.view(torch.float8_e4m3fn)
            expected = (
                _qsa_sparse_paged_attention_reference(
                    torch.zeros(
                        verify_positions.numel(),
                        24,
                        dim,
                        device=device,
                        dtype=torch.bfloat16,
                    ),
                    reference_k,
                    reference_v,
                    logical,
                    source_table,
                    request_rows,
                    dim**-0.5,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )
                * 0.5
            )
            return expected.flatten(1)[:, output_columns]

        initial_expected = attention_reference(
            initial_positions,
            torch.tensor([0] * 4 + [1] * 4, device=device, dtype=torch.int32),
        )
        torch.testing.assert_close(actual, initial_expected, rtol=2e-2, atol=2e-2)
        pointers = (
            source_table.data_ptr(),
            resident_table.data_ptr(),
            slots.data_ptr(),
        )

        # The real worker copies the sealed page before BlockPool can reuse it.
        transfer = SparseKVPageTransfer(91, 1, (1,), after_forward=False)
        worker.start_step(
            HiSparseConnectorMetadata(
                SparseKVOffloadCommand([transfer]), (), (), {}, True, {}
            ),
            torch.tensor([1, 0], dtype=torch.int32, device=device),
            num_tokens=0,
        )
        worker.finish_forward()
        worker.host_write_event.synchronize()
        assert worker.take_transfer_updates() == ([91], [91])
        torch.testing.assert_close(
            host_pages[1].view(torch.uint8),
            original[1].cpu().view(torch.uint8),
            rtol=0,
            atol=0,
        )
        pool.free_blocks([reclaimed])
        assert reclaimed.ref_cnt == 0
        peer_block = pool.get_new_blocks(1)[0]
        assert peer_block is reclaimed and peer_block.ref_cnt == 1

        # A peer's public QSA forward writes the reused physical span. Its host
        # source is different, so the original request's sealed host page survives.
        source_table.copy_(torch.tensor([[6, 0, 0], [0, 0, 0]], device=device))
        raw_table.copy_(torch.tensor([[3], [0]], device=device))
        query_starts.copy_(torch.tensor([0, 4, 4], device=device))
        seq_lens.copy_(torch.tensor([4, 0], device=device))
        positions.copy_(torch.tensor([0, 1, 2, 3, 0, 0, 0, 0], device=device))
        slots.copy_(torch.tensor([16, 17, 18, 19, -1, -1, -1, -1], device=device))
        host_slots.copy_(torch.tensor([96, 97, 98, 99, -1, -1, -1, -1], device=device))
        hidden.copy_(original_hidden * -3)
        build_metadata(active=1)
        worker.start_step(
            HiSparseConnectorMetadata(
                None,
                (),
                (),
                {"peer": (SparseKVRowMirror((16,), 96, 4),)},
                True,
                {"peer": SparseKVResidencyUpdate([0, 1, 2], ([1, 0, 0],))},
            ),
            torch.tensor([0, -1], dtype=torch.int32, device=device),
            request_ids=["peer"],
            num_tokens=tokens,
        )
        worker.prepare_forward(all_metadata)
        with (
            torch.inference_mode(),
            set_current_vllm_config(vllm_config),
            set_forward_context(all_metadata, vllm_config, num_tokens=tokens),
        ):
            owner(positions, hidden)
        worker.finish_forward()
        worker.host_write_event.synchronize()
        peer_rows = cache_rows(written[-1][:4]).flatten(1)
        torch.testing.assert_close(
            cache.view.cache[1, :4].view(torch.uint8),
            peer_rows.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        assert not torch.equal(
            peer_rows.view(torch.uint8), original[1, :4].flatten(1).view(torch.uint8)
        )
        torch.testing.assert_close(
            host_pages[1].view(torch.uint8),
            original[1].cpu().view(torch.uint8),
            rtol=0,
            atol=0,
        )
        host_before_replay = host_pages.clone()
        resident_before_replay = cache.view.cache.clone()

        # Update persistent inputs as the runner does: request A remains, B is
        # padding, its first page is host-only, and its live tail changes again.
        source_table.copy_(saved_source_table)
        source_table[1].zero_()
        raw_table.copy_(torch.tensor([[1], [0]], device=device))
        seq_lens.copy_(torch.tensor([32, 0], device=device))
        positions.copy_(initial_positions)
        slots.copy_(saved_slots)
        slots[4:].fill_(-1)
        host_slots.copy_(slots)
        hidden.copy_(original_hidden * 2)
        metadata = build_metadata(active=1)
        assert pointers == (
            source_table.data_ptr(),
            resident_table.data_ptr(),
            slots.data_ptr(),
        )
        worker.start_step(
            HiSparseConnectorMetadata(
                None,
                (),
                (),
                {"target": (SparseKVRowMirror((44,), 44, 4),)},
                False,
                {"target": SparseKVResidencyUpdate([0, 1, 2], ([0, 2, 0],))},
            ),
            torch.tensor([1, -1], dtype=torch.int32, device=device),
            request_ids=["target"],
            num_tokens=tokens,
        )
        worker.prepare_forward(all_metadata)
        assert not cache.all_context_pages_resident
        assert not metadata.is_cudagraph_capture
        graph.replay()
        worker.finish_forward()
        worker.host_write_event.synchronize()
        assert len(written) == 3  # Native replay does not execute Python hooks.
        expected_rows = initial_written[:4].clone()
        expected_rows[..., dim:] *= 2
        expected_rows = cache_rows(expected_rows)
        reference[2, 12:16] = expected_rows
        expected_host = host_before_replay.clone()
        expected_host[2, 12:16] = expected_rows.cpu()
        expected_resident = resident_before_replay.clone()
        expected_resident[2, 12:16] = expected_rows.flatten(1)
        torch.testing.assert_close(
            host_pages.view(torch.uint8),
            expected_host.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cache.view.cache.view(torch.uint8),
            expected_resident.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        selected = owner.topk_indices_buffer[:4]
        for row, position in zip(selected, initial_positions[:4]):
            assert row[-1] == position + 1
            torch.testing.assert_close(
                row[:-1][row[:-1] >= 0].sort().values,
                torch.arange(position + 1, device=device, dtype=torch.int32),
            )
        expected = attention_reference(
            initial_positions[:4], torch.zeros(4, device=device, dtype=torch.int32)
        )
        assert len(staging_observations) == 2
        staging_bytes, replay_misses, staged_contents = staging_observations[-1]
        row_bytes = row_width * original.element_size()
        selection_width = owner.topk_indices_buffer.shape[1] - 1
        padded_width = (selection_width + block_size - 1) // block_size * block_size
        assert staging_bytes == tokens * padded_width * row_bytes
        staged_rows = staged_contents.view(-1, row_width)
        verified_staged_rows = 0
        for token_row, selection in enumerate(selected):
            count = int(selection[-1])
            logical = selection[:count].long()
            physical = owner.physical_topk_indices_buffer[token_row, :count].long()
            assert torch.all(physical >= 0)
            pages = saved_source_table[0, logical // block_size].long()
            expected_staged = reference[pages, logical % block_size].flatten(1)
            torch.testing.assert_close(
                staged_rows[physical].view(torch.uint8),
                expected_staged.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            verified_staged_rows += count
        assert verified_staged_rows == 29 + 30 + 31 + 32
        host_rows_copied = replay_misses.sum().item()
        assert host_rows_copied == 4 * block_size
        live_selected_rows = owner.topk_indices_buffer[:tokens, -1].sum().item()
        record_property(
            "mixed_replay_proof",
            json.dumps(
                {
                    "main_dtype": "fp8_e4m3" if fp8 else "bfloat16",
                    "indexer_dtype": "fp8_e4m3" if indexer_fp8 else "bfloat16",
                    "k_scale": k_scale,
                    "v_scale": v_scale,
                    "complete_kv_row_bytes": row_bytes,
                    "verified_staged_kv_rows": verified_staged_rows,
                    "query_len": 4,
                    "capture_requests": 2,
                    "replay_requests": 1,
                    "reused_block": peer_block.block_id,
                    "transfer_completed": 91,
                    "actual_peer_written_rows": 4,
                    "live_tail_rows": 4,
                    "native_replays": 2,
                    "full_host_and_resident_bytes_equal": True,
                    "selected_staging_bytes": staging_bytes,
                    "effective_host_rows_copied": host_rows_copied,
                    "effective_host_bytes_copied": (
                        host_rows_copied * row_width * original.element_size()
                    ),
                    "staging_d2d_rows": replay_misses.numel(),
                    "unused_capacity_d2d_rows": (
                        replay_misses.numel() - live_selected_rows
                    ),
                    "max_abs_error": (actual[:4].float() - expected.float())
                    .abs()
                    .max()
                    .item(),
                    "natural_scheduler_pressure": False,
                    "natural_mtp_sampler": False,
                }
            ),
        )
        torch.testing.assert_close(actual[:4], expected, rtol=2e-2, atol=2e-2)
    finally:
        graph = None
        if worker is not None:
            worker.shutdown()
        else:
            release_pinned_state([], [registered])
        pool.free_blocks([block for block in allocated if block.ref_cnt])


@requires_qsa_kernels
def test_qsa_hisparse_full_graph_replay_uses_current_request_rows(
    tmp_path, dist_init, workspace_init, record_property, monkeypatch
) -> None:
    """One native graph follows padding, batch reorder and recycled host slots."""
    _run_qsa_hisparse_worker_case(
        tmp_path, record_property, monkeypatch, rewrite_cached_row=False
    )


@requires_qsa_kernels
def test_qsa_hisparse_worker_finish_invalidates_rewritten_hot_rows(
    tmp_path, dist_init, workspace_init, record_property, monkeypatch
) -> None:
    """Owner writes and normal worker finish replace a previously cached row."""
    _run_qsa_hisparse_worker_case(
        tmp_path, record_property, monkeypatch, rewrite_cached_row=True
    )


@requires_qsa_kernels
def test_qsa_hisparse_mtp_full_replay_ignores_reused_padding(
    tmp_path, dist_init, workspace_init, record_property, monkeypatch
) -> None:
    """MTP's preserved selection rows cannot mutate live state as FULL padding."""
    _run_qsa_hisparse_worker_case(
        tmp_path,
        record_property,
        monkeypatch,
        rewrite_cached_row=False,
        mtp_reuse_padding=True,
    )


def _run_qsa_hisparse_worker_case(
    tmp_path,
    record_property,
    monkeypatch,
    *,
    rewrite_cached_row,
    mtp_reuse_padding=False,
):
    import json
    from dataclasses import replace

    from transformers import Qwen4ExpTextConfig

    from vllm.config import AttentionConfig, HiSparseConfig, set_current_vllm_config
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
        HiSparseConnectorWorker,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMultiTokenPredictor
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    from vllm.v1.hisparse.runtime import (
        allocate_pinned_host_pool,
        initialize_hisparse_runtime_buffers,
        release_pinned_state,
    )
    from vllm.v1.hisparse.types import SparseKVResidencyUpdate, SparseKVRowMirror
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.states import RequestState

    torch.manual_seed(7)
    device = torch.device("cuda")
    layers, block_size, heads, dim = 2, 64, 2, 64
    config = Qwen4ExpTextConfig(
        architectures=["Qwen4ExpForCausalLM"],
        num_hidden_layers=layers,
        layer_types=["full_attention"] * layers,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=24,
        num_key_value_heads=heads,
        head_dim=dim,
        num_experts=4,
        num_experts_per_tok=2,
        ple_layer_ids=[],
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
        indexer_budget=2048,
    )
    config.save_pretrained(tmp_path)
    vllm_config = EngineArgs(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        block_size=block_size,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        enforce_eager=False,
        async_scheduling=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [2],
        },
        attention_config=AttentionConfig(
            indexer_kv_dtype="bf16",
            flash_attn_version=4,
            hisparse_config=HiSparseConfig(device_buffer_size=2112),
        ),
    ).create_engine_config()
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device(device),
    ):
        owners = [
            Qwen4ExpQSAAttention(
                vllm_config=vllm_config,
                config=config,
                layer_id=layer,
                prefix=f"model.layers.{layer}.self_attn",
            )
            for layer in range(layers)
        ]
    handles = [owner.hisparse_cache for owner in owners]
    assert all(handle is not None for handle in handles)
    builders = [
        owner.get_attn_backend().get_builder_cls()(
            owner.get_kv_cache_spec(vllm_config),
            [owner.layer_name],
            vllm_config,
            device,
        )
        for owner in owners
    ]
    states = RequestState(
        max_num_reqs=2,
        max_model_len=128,
        max_num_batched_tokens=128,
        num_speculative_steps=0,
        vocab_size=128,
        device=device,
    )
    for name in ("old", "peer"):
        states.add_request(name, 1, [1], 0, 4)
    states.apply_staged_writes()
    old_slot, peer_slot = (states.req_id_to_index[name] for name in ("old", "peer"))
    row_width = heads * 2 * dim
    block_bytes = block_size * row_width * torch.bfloat16.itemsize
    hot_blocks = handles[0].runtime.region_stride // block_size
    num_blocks = 3 + 2 * hot_blocks
    # Both layers share the BLNHC slab, but have independent complete K/V rows.
    backing = torch.zeros(
        num_blocks * layers * block_bytes, dtype=torch.uint8, device=device
    )
    hot_table = torch.arange(3, num_blocks, dtype=torch.int32, device=device).view(
        2, -1
    )
    original_hot_table = hot_table.clone()
    source_table = torch.empty((2, 2), dtype=torch.int32, device=device)
    residency = torch.zeros((2, 1, 2), dtype=torch.int32, device=device)
    resident_table = residency[:, 0]
    slots = torch.full((2,), -1, dtype=torch.int64, device=device)
    host_slots = slots.clone()
    query_starts = torch.empty(3, dtype=torch.int32, device=device)
    seq_lens = torch.empty(2, dtype=torch.int32, device=device)
    positions = torch.zeros(2, dtype=torch.int64, device=device)
    hidden = torch.empty(2, 256, dtype=torch.bfloat16, device=device)
    query = torch.zeros(2, 24, dim, dtype=torch.bfloat16, device=device)
    output = torch.empty(layers, 2, 256, dtype=torch.bfloat16, device=device)
    writer_rows = torch.empty(
        layers, 2, heads, 2 * dim, dtype=torch.bfloat16, device=device
    )
    selected_before_attention = [
        torch.empty_like(owner.topk_indices_buffer[:2]) for owner in owners
    ]
    output_columns = torch.cat((torch.arange(128), torch.arange(768, 896))).to(device)
    side_builders = []
    raw_table = torch.empty((2, 1), dtype=torch.int32, device=device)
    selection_hooks = []
    for layer, owner in enumerate(owners):
        # Actual projections produce Q=K=gate=0 and layer-specific V from hidden.
        # The output projection includes two Q heads from each KV shard.
        with torch.no_grad():
            for parameter in owner.parameters():
                parameter.zero_()
            columns = torch.arange(owner.kv_size, device=device)
            owner.qkv_proj.weight[
                2 * owner.q_size + owner.kv_size + columns, columns
            ] = layer + 1
            owner.o_proj.weight[torch.arange(256, device=device), output_columns] = 1
        owner.process_weights_after_loading(torch.bfloat16)
        for side_cache in (
            owner.indexer.raw_key_cache,
            owner.indexer.compressed_key_cache,
        ):
            spec = side_cache.get_kv_cache_spec(vllm_config)
            side_cache.bind_kv_cache(
                torch.zeros(5, 1, spec.num_states, 128, dtype=spec.dtype, device=device)
            )
            table = (
                raw_table if side_cache is owner.indexer.raw_key_cache else source_table
            )
            side_builders.append(
                (
                    side_cache.prefix,
                    QSAMetadataBuilder(spec, [side_cache.prefix], vllm_config, device),
                    table,
                )
            )

        def save_selection(module, inputs, output, *, layer=layer):
            selected, main_outputs = output
            assert main_outputs is None
            selected_before_attention[layer].copy_(selected)

        selection_hooks.append(owner.indexer.register_forward_hook(save_selection))
        update_cache = owner.impl.do_kv_cache_update

        def capture_writer(
            attn_layer,
            key,
            value,
            kv_cache,
            slot_mapping,
            *,
            layer=layer,
            update_cache=update_cache,
        ):
            writer_rows[layer].copy_(torch.cat((key, value), dim=-1))
            update_cache(attn_layer, key, value, kv_cache, slot_mapping)

        monkeypatch.setattr(owner.impl, "do_kv_cache_update", capture_writer)
    reference = torch.randn(layers, 5, block_size, heads, 2 * dim, dtype=torch.bfloat16)
    registered_pools = []
    host_pages = []
    worker = None
    graph = None
    try:
        for layer, handle in enumerate(handles):
            binding = dict(
                byte_offset=layer * block_bytes,
                block_stride=layers * block_bytes,
                num_blocks=num_blocks,
                block_size=block_size,
            )
            handle.bind_cache(
                backing, **binding, block_table=resident_table, slot_mapping=slots
            )
            owners[layer].bind_kv_cache(
                handle.view.cache.unflatten(-1, (heads, 2 * dim)).transpose(1, 2)
            )
            handle.runtime.bind_hot_cache(backing, **binding, block_table=hot_table)
            host, registered = allocate_pinned_host_pool(reference[layer].nbytes)
            registered_pools.append(registered)
            pages = host.view(torch.bfloat16).view(5, block_size, heads, 2 * dim)
            pages.copy_(reference[layer])
            host_pages.append(pages)
            handle.runtime.bind_source_cache(
                pages.flatten(2), registered_host_pool=registered
            )
            handle.runtime.resident_source_index = 0
            handle.residency = residency
            handle.source_block_table = source_table
            handle.mirror_slot_mapping = host_slots
            handle.view.cache[1].copy_(reference[layer, 2].flatten(1).to(device))
            handle.view.cache[2].copy_(reference[layer, 4].flatten(1).to(device))
        initialize_hisparse_runtime_buffers(handles, max_num_reqs=2)
        worker = HiSparseConnectorWorker(
            vllm_config,
            KVCacheConfig(
                num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[]
            ),
        )
        worker.initialize(
            handles,
            [owner.layer_name for owner in owners],
            backing,
            2,
            5,
            device,
            registered_pools,
        )
        mapping = worker.request_state_indices
        metadata: dict[str, FlashAttentionMetadata | qsa_cache.QSAForwardMetadata] = {}

        def build_metadata(common):
            metadata.clear()
            for owner, builder in zip(owners, builders):
                metadata[owner.layer_name] = builder.build(0, common)
            for prefix, builder, table in side_builders:
                metadata[prefix] = builder.build(
                    0, replace(common, block_table_tensor=table)
                )

        def prepare(step, *, write_history=False, start_forward=True):
            names = [("old", "peer"), ("peer",), ("peer", "replacement")][step]
            source_ids = [(1, 3), (3,), (3, 1)][step]
            resident_ids = [(1, 2), (2,), (2, 1)][step]
            active = len(names)
            offset = 3 if write_history else 5 + step
            write_page = 0 if write_history else 1
            source_table.zero_()
            raw_table.zero_()
            slots.fill_(-1)
            host_slots.fill_(-1)
            seq_lens.zero_()
            positions.zero_()
            query_starts.copy_(torch.tensor([0, 1, active], device=device))
            hot_table.copy_(
                original_hot_table if step == 0 else original_hot_table.flip(0)
            )
            hidden.normal_()
            mirrors = {}
            residency_updates = {}
            for row, (name, host_block, resident_block) in enumerate(
                zip(names, source_ids, resident_ids)
            ):
                source_table[row].copy_(
                    torch.tensor([host_block, host_block + 1], device=device)
                )
                resident_blocks = [0, 0]
                resident_blocks[write_page] = resident_block
                residency_updates[name] = SparseKVResidencyUpdate(
                    [0, 1], (resident_blocks,)
                )
                raw_table[row, 0] = host_block
                slots[row] = resident_block * block_size + offset
                destination = (host_block + write_page) * block_size + offset
                host_slots[row] = destination
                seq_lens[row] = write_page * block_size + offset + 1
                positions[row] = write_page * block_size + offset
                mirrors[name] = (
                    SparseKVRowMirror(
                        (resident_block * block_size + offset,),
                        destination,
                        1,
                    ),
                )
                # This component fixture remaps the writable resident page.
                # Its prior contents precede the real owner write below.
                for layer, handle in enumerate(handles):
                    handle.view.cache[resident_block].copy_(
                        reference[layer, host_block + write_page].flatten(1).to(device)
                    )
            common = CommonAttentionMetadata(
                num_actual_tokens=2,
                num_reqs=2,
                max_query_len=1,
                query_start_loc=query_starts,
                query_start_loc_cpu=torch.tensor([0, 1, active], dtype=torch.int32),
                max_seq_len=write_page * block_size + offset + 1,
                seq_lens=seq_lens,
                block_table_tensor=source_table,
                slot_mapping=host_slots,
            )
            build_metadata(common)
            if start_forward:
                worker.start_step(
                    HiSparseConnectorMetadata(
                        None,
                        (),
                        (1, 2) if step == 2 else (),
                        mirrors,
                        False,
                        residency_updates,
                    ),
                    torch.tensor(
                        [states.req_id_to_index[name] for name in names],
                        dtype=torch.int32,
                        device=device,
                    ),
                    request_ids=list(names),
                    num_tokens=active,
                )
                worker.prepare_forward(metadata)
                for name, resident_block in zip(names, resident_ids):
                    state_row = states.req_id_to_index[name]
                    assert (
                        resident_table[state_row, write_page].item() == resident_block
                    )
            return names, source_ids, resident_ids, offset

        def forward():
            with set_forward_context(metadata, vllm_config, num_tokens=2):
                for layer, owner in enumerate(owners):
                    output[layer].copy_(owner(positions, hidden))

        def pointers():
            tensors = [
                backing,
                hot_table,
                source_table,
                resident_table,
                slots,
                host_slots,
                mapping,
                query_starts,
                seq_lens,
                positions,
                hidden,
                writer_rows,
                output,
            ]
            for owner in owners:
                group = owner.hisparse_cache.runtime.index_group
                tensors.extend(
                    (
                        owner.topk_indices_buffer,
                        owner.physical_topk_indices_buffer,
                        group.shared_topk.physical_topk_indices,
                        group.shared_topk.swap_counts,
                        group.device_global_indices,
                        metadata[owner.layer_name].req_id_per_token,
                    )
                )
                for side_cache in (
                    owner.indexer.raw_key_cache,
                    owner.indexer.compressed_key_cache,
                ):
                    side = metadata[side_cache.prefix]
                    tensors.extend(
                        (
                            side_cache.kv_cache,
                            side.slot_mapping,
                            side.logical_positions,
                            side.token_to_req,
                            side.k_work_metadata,
                        )
                    )
            return [tensor.data_ptr() for tensor in tensors]

        def check(names, source_ids, resident_ids, offset, *, write_history=False):
            torch.accelerator.synchronize()
            write_page = 0 if write_history else 1
            for layer, (owner, handle) in enumerate(zip(owners, handles)):
                torch.testing.assert_close(
                    writer_rows[layer, :, :, :dim],
                    torch.zeros_like(writer_rows[layer, :, :, :dim]),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    writer_rows[layer, :, :, dim:],
                    (hidden[:, : heads * dim] * (layer + 1)).view(2, heads, dim),
                    rtol=0,
                    atol=0,
                )
                for row, host_block in enumerate(source_ids):
                    # RoPE's signed zeros are part of the exact stored K bytes.
                    reference[layer, host_block + write_page, offset].copy_(
                        writer_rows[layer, row].cpu()
                    )
                packed = owner.topk_indices_buffer[:2]
                torch.testing.assert_close(
                    packed, selected_before_attention[layer], rtol=0, atol=0
                )
                for row in range(2):
                    count = (
                        write_page * block_size + offset + 1 if row < len(names) else 0
                    )
                    assert packed[row, -1].item() == count
                    torch.testing.assert_close(
                        packed[row, :-1][packed[row, :-1] >= 0].sort().values,
                        torch.arange(count, dtype=torch.int32, device=device),
                    )
                pages = reference[layer].to(device)
                expected = (
                    _qsa_sparse_paged_attention_reference(
                        query,
                        pages[..., :dim],
                        pages[..., dim:],
                        packed[:, :-1],
                        source_table,
                        torch.arange(2, dtype=torch.int32, device=device),
                        dim**-0.5,
                    )
                    * 0.5
                )
                expected = expected.flatten(1)[:, output_columns]
                torch.testing.assert_close(
                    output[layer], expected, rtol=2e-2, atol=2e-2
                )
                physical = owner.physical_topk_indices_buffer[:2]
                torch.testing.assert_close(
                    physical[:, -1], packed[:, -1], rtol=0, atol=0
                )
                for row, (host_block, resident_block) in enumerate(
                    zip(source_ids, resident_ids)
                ):
                    logical = packed[row, :-1]
                    valid = logical >= 0
                    actual = handle.runtime.hot.attention_cache.reshape(-1, row_width)[
                        physical[row, :-1][valid].long()
                    ]
                    logical = logical[valid].long()
                    selected = (
                        reference[
                            layer,
                            host_block + logical.cpu() // block_size,
                            logical.cpu() % block_size,
                        ]
                        .flatten(1)
                        .to(device)
                    )
                    torch.testing.assert_close(
                        actual.view(torch.uint8),
                        selected.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    written = writer_rows[layer, row].flatten()
                    torch.testing.assert_close(
                        handle.view.cache[resident_block, offset].view(torch.uint8),
                        written.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                # Whole host pool equality also proves padding did not mirror a row.
                torch.testing.assert_close(
                    host_pages[layer].view(torch.uint8),
                    reference[layer].view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                assert torch.count_nonzero(handle.view.cache[0]) == 0
                if len(names) == 1:
                    assert (physical[1, :-1] == -1).all()
                    assert torch.count_nonzero(output[layer, 1]) == 0

        def row_locations(owner, logical_position):
            logical = owner.topk_indices_buffer[:2, :-1]
            selected = logical == logical_position
            assert torch.all(selected.sum(dim=1) == 1)
            columns = selected.int().argmax(dim=1, keepdim=True)
            return (
                owner.physical_topk_indices_buffer[:2, :-1]
                .gather(1, columns)
                .flatten()
                .long()
            )

        with torch.inference_mode(), set_current_vllm_config(vllm_config):
            eager_case = prepare(0)
            forward()
            worker.finish_forward()
            check(*eager_case)
            if rewrite_cached_row:
                global_rows = torch.tensor(
                    [block * block_size + 3 for block in eager_case[1]],
                    dtype=torch.int32,
                    device=device,
                )
                stale = []
                for owner, handle in zip(owners, handles):
                    indices = handle.runtime.index_group.device_global_indices
                    assert (indices[mapping] == global_rows[:, None]).any(dim=1).all()
                    physical = row_locations(owner, 3)
                    stale.append(
                        handle.runtime.hot.attention_cache.flatten(0, 1)[
                            physical
                        ].clone()
                    )

                rewritten = prepare(0, write_history=True)
                forward()
                written = writer_rows.flatten(2).clone()
                for owner, handle, packed in zip(
                    owners, handles, selected_before_attention
                ):
                    torch.testing.assert_close(owner.topk_indices_buffer[:2], packed)
                    indices = handle.runtime.index_group.device_global_indices
                    assert (indices[mapping] == global_rows[:, None]).any(dim=1).all()
                # This is the production owner -> DMA mirror -> invalidation path.
                # The test never invokes either invalidation API directly.
                worker.finish_forward()
                torch.accelerator.synchronize()
                check(*rewritten, write_history=True)
                assert worker.host_write_event.query()
                assert all(
                    event.query() for event, _ in worker._pending_dma_descriptors
                )
                for layer, handle in enumerate(handles):
                    indices = handle.runtime.index_group.device_global_indices
                    assert not (indices[mapping] == global_rows[:, None]).any(), (
                        "worker finish retained a stale hot row"
                    )
                    host = handle.runtime.host_cache[global_rows.cpu().long()]
                    torch.testing.assert_close(
                        host.view(torch.uint8),
                        written[layer].cpu().view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    assert not torch.equal(stale[layer], written[layer])

                # The next genuine owner forward has no resident copy of row 3.
                # Its resolver must read the new host bytes into its own hot cache.
                revisit = prepare(0)
                before = [
                    handle.runtime.index_group.swap_stats.tolist() for handle in handles
                ]
                forward()
                worker.finish_forward()
                check(*revisit)
                misses = []
                for layer, (owner, handle) in enumerate(zip(owners, handles)):
                    physical = row_locations(owner, 3)
                    actual = handle.runtime.hot.attention_cache.flatten(0, 1)[physical]
                    torch.testing.assert_close(
                        actual.view(torch.uint8),
                        written[layer].view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    assert not torch.equal(actual, stale[layer])
                    after = handle.runtime.index_group.swap_stats.tolist()
                    delta = [end - start for start, end in zip(before[layer], after)]
                    assert delta[1] >= 2
                    misses.append(delta[1])
                proof = {
                    "controlled_owner_worker_rewrite": True,
                    "public_owner_and_real_indexer": True,
                    "natural_mtp_sampler": False,
                    "host_write_complete": True,
                    "owners": layers,
                    "rewritten_rows_per_owner": 2,
                    "subsequent_host_misses": misses,
                }
                record_property("qsa_worker_rewrite", json.dumps(proof))
                print("QSA_WORKER_REWRITE " + json.dumps(proof))
                return
            if mtp_reuse_padding:
                # Step 0 uses real multi-token prefill and leaves valid rows
                # beyond the one request compacted for subsequent draft steps.
                prepare(1, start_forward=False)
                positions.copy_(torch.tensor([69, 70], device=device))
                slots.copy_(torch.tensor([133, 134], device=device))
                host_slots.copy_(torch.tensor([261, 262], device=device))
                query_starts.copy_(torch.tensor([0, 2, 2], device=device))
                seq_lens.copy_(torch.tensor([71, 0], device=device))
                build_metadata(
                    CommonAttentionMetadata(
                        num_actual_tokens=2,
                        num_reqs=2,
                        max_query_len=2,
                        query_start_loc=query_starts,
                        query_start_loc_cpu=torch.tensor([0, 2, 2], dtype=torch.int32),
                        max_seq_len=71,
                        seq_lens=seq_lens,
                        block_table_tensor=source_table,
                        slot_mapping=host_slots,
                    )
                )
                worker.start_step(
                    HiSparseConnectorMetadata(
                        None,
                        (),
                        (),
                        {"peer": (SparseKVRowMirror((133,), 261, 2),)},
                        False,
                        {"peer": SparseKVResidencyUpdate([0, 1], ([0, 2],))},
                    ),
                    torch.tensor([peer_slot], dtype=torch.int32, device=device),
                    request_ids=["peer"],
                    num_tokens=2,
                )
                worker.prepare_forward(metadata)
                forward()
                worker.finish_forward()
                torch.accelerator.synchronize()
                for layer in range(layers):
                    reference[layer, 4, 5:7].copy_(writer_rows[layer].cpu())
                mtp = SimpleNamespace(_iter_qsa_attentions=lambda: iter(owners))
                Qwen4ExpMultiTokenPredictor.compact_topk_indices(
                    mtp, torch.tensor([1], dtype=torch.int64, device=device)
                )
                Qwen4ExpMultiTokenPredictor.set_skip_topk(mtp, True)
                frozen = [owner.topk_indices_buffer[:2].clone() for owner in owners]
                assert all((packed[:, -1] == 71).all() for packed in frozen)
            # Capture inert inputs at the same addresses; replay must use new data.
            slots.fill_(-1)
            host_slots.fill_(-1)
            mapping.fill_(-1)
            seq_lens.zero_()
            positions.zero_()
            query_starts.copy_(torch.tensor([0, 1, 2], device=device))
            build_metadata(
                CommonAttentionMetadata(
                    num_actual_tokens=2,
                    num_reqs=2,
                    max_query_len=1,
                    query_start_loc=query_starts,
                    query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
                    max_seq_len=0,
                    seq_lens=seq_lens,
                    block_table_tensor=source_table,
                    slot_mapping=host_slots,
                )
            )
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                forward()
            torch.cuda.current_stream().wait_stream(stream)
            worker.reset_hot_state()
            addresses = pointers()
            if mtp_reuse_padding:
                evidence = []
                for replay in range(2):
                    prepare(1)
                    graph.replay()
                    worker.finish_forward()
                    torch.accelerator.synchronize()
                    for layer in range(layers):
                        reference[layer, 4, 6].copy_(writer_rows[layer, 0].cpu())
                    observed = []
                    for owner, handle in zip(owners, handles):
                        packed = owner.topk_indices_buffer[:2]
                        physical = owner.physical_topk_indices_buffer[:2]
                        group = handle.runtime.index_group
                        observed.append(
                            {
                                "layer": owner.layer_name,
                                "request_state_indices": mapping.tolist(),
                                "token_to_req": metadata[
                                    owner.layer_name
                                ].req_id_per_token.tolist(),
                                "logical_counts": packed[:, -1].tolist(),
                                "swap_counts": group.shared_topk.swap_counts[
                                    :2
                                ].tolist(),
                                "padded_physical_valid": int(
                                    (physical[1, :-1] >= 0).sum().item()
                                ),
                                "active_lru_unique": group.lru_slots[peer_slot]
                                .unique()
                                .numel(),
                                "active_lru_size": group.lru_slots.shape[1],
                            }
                        )
                    evidence.append({"replay": replay, "owners": observed})
                    record_property("qsa_mtp_padding", json.dumps(evidence))
                    print("QSA_MTP_PADDING " + json.dumps(evidence))
                    for layer, (owner, handle) in enumerate(zip(owners, handles)):
                        packed = owner.topk_indices_buffer[:2]
                        physical = owner.physical_topk_indices_buffer[:2]
                        torch.testing.assert_close(
                            packed, frozen[layer], rtol=0, atol=0
                        )
                        assert (physical[1, :-1] == -1).all(), (
                            "FULL draft padding resolved into an active request"
                        )
                        assert observed[layer]["swap_counts"][1] == 0
                        assert (
                            observed[layer]["active_lru_unique"]
                            == observed[layer]["active_lru_size"]
                        )
                        count = int(packed[0, -1].item())
                        logical = packed[0, :count].long()
                        actual = handle.runtime.hot.attention_cache.flatten(0, 1)[
                            physical[0, :count].long()
                        ]
                        expected = (
                            reference[
                                layer,
                                3 + logical.cpu() // block_size,
                                logical.cpu() % block_size,
                            ]
                            .flatten(1)
                            .to(device)
                        )
                        torch.testing.assert_close(
                            actual.view(torch.uint8),
                            expected.view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                        pages = reference[layer].to(device)
                        expected_output = (
                            _qsa_sparse_paged_attention_reference(
                                query[:1],
                                pages[..., :dim],
                                pages[..., dim:],
                                packed[:1, :-1],
                                source_table,
                                torch.zeros(1, dtype=torch.int32, device=device),
                                dim**-0.5,
                            )
                            * 0.5
                        )
                        expected_output = expected_output.flatten(1)[:, output_columns]
                        torch.testing.assert_close(
                            output[layer, :1], expected_output, rtol=2e-2, atol=2e-2
                        )
                    assert addresses == pointers()
                return
            evidence = []
            for step in range(3):
                if step == 2:
                    assert states.remove_request("old") == old_slot
                    states.add_request("replacement", 1, [2], 0, 4)
                    states.apply_staged_writes()
                    assert states.req_id_to_index["replacement"] == old_slot
                    for layer in range(layers):
                        reference[layer, 1:3].add_(8)
                        host_pages[layer][1:3].copy_(reference[layer, 1:3])
                case = prepare(step)
                if step == 0:
                    # Prime exactly one hot history row per request through the
                    # real resolver. Replay must combine hot, host and resident.
                    for owner, handle in zip(owners, handles):
                        priming = torch.full(
                            (2, owner.indexer.output_width),
                            -1,
                            dtype=torch.int32,
                            device=device,
                        )
                        priming[:, 0] = 3
                        handle.runtime.begin_forward()
                        handle.swap_in(
                            torch.arange(2, dtype=torch.int32, device=device),
                            source_table,
                            priming,
                            block_size=block_size,
                            num_valid_rows=metadata[owner.layer_name].query_start_loc[
                                -1:
                            ],
                        )
                        handle.runtime.begin_forward()
                before = [
                    handle.runtime.index_group.swap_stats.tolist() for handle in handles
                ]
                graph.replay()
                worker.finish_forward()
                check(*case)
                deltas = []
                for layer, (owner, handle) in enumerate(zip(owners, handles)):
                    torch.testing.assert_close(
                        owner.topk_indices_buffer[:2],
                        selected_before_attention[layer],
                        rtol=0,
                        atol=0,
                    )
                    after = handle.runtime.index_group.swap_stats.tolist()
                    delta = [end - start for start, end in zip(before[layer], after)]
                    assert delta == [[2, 126], [64, 0], [64, 64]][step], delta
                    deltas.append(delta)
                assert addresses == pointers()
                expected_mapping = [
                    (old_slot, peer_slot),
                    (peer_slot, -1),
                    (peer_slot, old_slot),
                ][step]
                assert mapping.tolist() == list(expected_mapping)
                evidence.append(
                    {
                        "step": step,
                        "requests": case[0],
                        "mapping": mapping.tolist(),
                        "hot_hits_host_misses": deltas,
                    }
                )
            assert len(evidence) == 3
            proof = {
                "native_replays": len(evidence),
                "public_owner_and_real_indexer": True,
                "addresses": addresses,
                "steps": evidence,
            }
            record_property("qsa_full_graph_replays", json.dumps(proof))
            print("QSA_FULL_GRAPH_REPLAY " + json.dumps(proof))
    finally:
        for hook in selection_hooks:
            hook.remove()
        if graph is not None:
            graph.reset()
        if worker is not None and worker._initialized:
            worker.shutdown()
        else:
            release_pinned_state([], registered_pools)


@requires_qsa_kernels
@pytest.mark.parametrize("decode_query_len", [1, 2, 3, 4])
def test_qsa_split_selection_correctness(workspace_init, decode_query_len: int) -> None:
    query_lens = [decode_query_len, decode_query_len, 33]
    rows, heads, head_dim = sum(query_lens), 4, 128
    token_topk, compress_ratio = 2048, 4
    torch.manual_seed(13)
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(120, 16, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    page_table = torch.arange(120, device="cuda", dtype=torch.int32).view(3, 40)
    token_to_req = torch.repeat_interleave(
        torch.arange(3, device="cuda", dtype=torch.int32),
        torch.tensor(query_lens, device="cuda"),
    )
    query_start_loc = torch.tensor(
        [0, decode_query_len, 2 * decode_query_len, rows],
        device="cuda",
        dtype=torch.int32,
    )
    sequence_lengths = torch.full((3,), 2560, device="cuda", dtype=torch.int32)
    query_positions = torch.cat(
        [
            torch.arange(2560 - query_len, 2560, device="cuda")
            for query_len in query_lens
        ]
    )

    block_indices = torch.empty(
        rows,
        token_topk // compress_ratio,
        device="cuda",
        dtype=torch.int32,
    )
    visible_blocks = torch.minimum(
        (query_positions + 1) // compress_ratio,
        sequence_lengths.index_select(0, token_to_req.long()) // compress_ratio,
    ).to(torch.int32)
    num_decode_tokens = 2 * decode_query_len
    decode_slice = slice(0, num_decode_tokens)
    qsa_indexer_ops.qsa_select_paged_decode(
        q[decode_slice],
        cache,
        page_table[:2],
        visible_blocks[decode_slice],
        token_topk,
        compress_ratio,
        decode_query_len,
        block_indices[decode_slice],
    )
    prefill_slice = slice(num_decode_tokens, rows)
    qsa_indexer_ops.qsa_select_paged_prefill(
        q[prefill_slice],
        cache,
        page_table[2:],
        query_start_loc[2:],
        visible_blocks[prefill_slice],
        token_topk,
        compress_ratio,
        query_lens[-1],
        block_indices[prefill_slice],
        max_seq_len=sequence_lengths.max().item(),
    )
    # +1: the packed trailing count column (never a token index; excluded
    # from the comparison).
    actual = torch.empty(
        (rows, token_topk + compress_ratio), device="cuda", dtype=torch.int32
    )
    qsa_indexer_ops.expand_qsa_block_indices(
        block_indices,
        query_positions,
        visible_blocks,
        compress_ratio,
        token_topk,
        actual,
    )
    expected_blocks = _qsa_select_paged_reference(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )
    expected = _expand_qsa_indices_reference(
        expected_blocks,
        query_positions,
        sequence_lengths.index_select(0, token_to_req.long()),
        compress_ratio,
        token_topk,
    )

    torch.testing.assert_close(
        actual[:, : token_topk + compress_ratio - 1].sort().values,
        expected.sort().values,
    )


@requires_qsa_kernels
def test_qsa_selection_handles_no_complete_compressed_blocks(workspace_init) -> None:
    q = torch.zeros(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 1, 8, device="cuda", dtype=torch.bfloat16)
    page_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    visible_blocks = torch.zeros(2, device="cuda", dtype=torch.int32)

    block_indices = torch.empty((2, 512), device="cuda", dtype=torch.int32)
    qsa_indexer_ops.qsa_select_paged_prefill(
        q,
        cache,
        page_table,
        torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        visible_blocks,
        token_topk=2048,
        compress_ratio=4,
        max_query_len=2,
        block_indices=block_indices,
        max_seq_len=64,  # clamps to the page-table capacity
    )
    selected = torch.empty((2, 2052), device="cuda", dtype=torch.int32)
    qsa_indexer_ops.expand_qsa_block_indices(
        block_indices,
        query_positions,
        visible_blocks,
        compress_ratio=4,
        token_topk=2048,
        out=selected,
    )

    assert selected[0, :2].tolist() == [0, 1]
    assert selected[1, :3].tolist() == [0, 1, 2]
    assert torch.all(selected[0, 2:2051] == -1)
    assert torch.all(selected[1, 3:2051] == -1)
    # The packed trailing column holds each row's valid-entry count.
    assert selected[:, 2051].tolist() == [2, 3]


@requires_qsa_kernels
def test_qsa_streaming_compression_and_compressor_state_store_match_reference() -> None:
    head_dim = 8
    current_pairs = [
        *((0, position) for position in range(2, 9)),
        *((1, position) for position in range(5, 11)),
    ]

    def key_row(request: int, position: int) -> torch.Tensor:
        return (
            torch.arange(head_dim, dtype=torch.float32) + request * 1000 + position * 10
        )

    def position_row(request: int, position: int) -> torch.Tensor:
        return torch.tensor(
            [
                request * 1000 + position,
                request * 1000 + position + 100,
                request * 1000 + position + 200,
            ],
            dtype=torch.int64,
        )

    raw_keys = (
        torch.stack([key_row(request, position) for request, position in current_pairs])
        .unsqueeze(1)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    raw_positions = (
        torch.stack(
            [position_row(request, position) for request, position in current_pairs]
        )
        .unsqueeze(1)
        .to(device="cuda")
    )
    token_to_req = torch.tensor(
        [request for request, _ in current_pairs],
        dtype=torch.int32,
        device="cuda",
    )
    logical_positions = torch.tensor(
        [position for _, position in current_pairs],
        dtype=torch.int64,
        device="cuda",
    )
    query_start_loc = torch.tensor([0, 7, 13], dtype=torch.int32, device="cuda")
    compressor_state_block_table = torch.tensor(
        [[1], [0]], dtype=torch.int32, device="cuda"
    )
    compressor_state_cache = torch.zeros(
        2, 4, 1, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    rope_cache = torch.zeros(2, 4, 1, 3, dtype=torch.int64, device="cuda")
    for request, position, block in ((0, 0, 1), (0, 1, 1), (1, 4, 0)):
        compressor_state_cache[block, position % 4, 0] = key_row(request, position).to(
            device="cuda", dtype=torch.bfloat16
        )
        rope_cache[block, position % 4, 0] = position_row(request, position).to("cuda")

    compressed_slots = torch.full(
        (len(current_pairs),), -1, dtype=torch.int64, device="cuda"
    )
    valid_rows = torch.tensor([1, 5, 9], dtype=torch.int64, device="cuda")
    compressed_slots[valid_rows] = torch.arange(3, device="cuda")
    pooled, first_positions = qsa_ops.qsa_compress_groups_with_ratio(
        raw_keys,
        raw_positions,
        compressor_state_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        compress_ratio=4,
        rope_cache=rope_cache,
    )
    pooled_without_rope, scalar_first_positions = (
        qsa_ops.qsa_compress_groups_with_ratio(
            raw_keys,
            raw_positions,
            compressor_state_cache,
            compressor_state_block_table,
            token_to_req,
            query_start_loc,
            logical_positions,
            compressed_slots,
            compress_ratio=4,
        )
    )

    groups = [
        [(0, position) for position in range(0, 4)],
        [(0, position) for position in range(4, 8)],
        [(1, position) for position in range(4, 8)],
    ]
    expected_pooled = (
        torch.stack(
            [
                torch.stack([key_row(*pair) for pair in group]).mean(dim=0)
                for group in groups
            ]
        )
        .unsqueeze(1)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    expected_positions = torch.stack(
        [position_row(0, 0), position_row(0, 4), position_row(1, 4)]
    ).to("cuda")
    expected_scalar_positions = torch.tensor(
        [[0, 0, 0], [4, 4, 4], [4, 4, 4]],
        dtype=torch.int64,
        device="cuda",
    )

    torch.testing.assert_close(pooled[valid_rows], expected_pooled)
    torch.testing.assert_close(pooled_without_rope[valid_rows], expected_pooled)
    torch.testing.assert_close(first_positions[valid_rows], expected_positions)
    torch.testing.assert_close(
        scalar_first_positions[valid_rows], expected_scalar_positions
    )

    compressor_state_slots = torch.tensor(
        [-1, -1, -1, 5, 6, 7, 4, -1, -1, 3, 0, 1, 2],
        dtype=torch.int64,
        device="cuda",
    )
    qsa_ops.qsa_store_cache_rows(
        compressor_state_cache, compressor_state_slots, raw_keys
    )
    qsa_ops.qsa_store_cache_rows(rope_cache, compressor_state_slots, raw_positions)
    for request, positions, block in ((0, range(5, 9), 1), (1, range(7, 11), 0)):
        for position in positions:
            torch.testing.assert_close(
                compressor_state_cache[block, position % 4, 0],
                key_row(request, position).to(device="cuda", dtype=torch.bfloat16),
            )
            torch.testing.assert_close(
                rope_cache[block, position % 4, 0],
                position_row(request, position).to("cuda"),
            )
