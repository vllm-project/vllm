# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.models.qwen4_exp.common import qsa_indexer as qsa_common
from vllm.models.qwen4_exp.common.qsa_indexer import (
    QSAIndexerMetadataBuilder,
    qsa_kv_cache_dtype,
)
from vllm.models.qwen4_exp.nvidia import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.nvidia import qsa_indexer
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_fp8 as qsa_fp8_ops
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import clear_layer_kv_caches

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)
requires_qsa_metadata_kernels = pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not HAS_TRITON,
    reason="QSA metadata kernels require CUDA or ROCm and Triton",
)


_HISPARSE_KV_TRANSFER_CONFIG = {
    "kv_connector": "HiSparseConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"host_pool_gib": 1 / 64},
}


@requires_qsa_kernels
def test_qsa_hisparse_starts_with_automatic_cache_layout(
    tmp_path, vllm_runner, monkeypatch
):
    from transformers import Qwen4ExpConfig

    from vllm import SamplingParams

    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    Qwen4ExpConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config={
            "vocab_size": 128,
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "indexed_attention"],
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "output_gate_type": "sigmoid",
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "hc_lowrank": 28,
            "indexer_n_heads": 8,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_compress_ratio": 4,
            "indexer_budget": 2048,
            "mtp_num_hidden_layers": 0,
        },
    ).save_pretrained(tmp_path)
    with vllm_runner(
        str(tmp_path),
        language_model_only=True,
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        kv_cache_dtype="fp8_e4m3",
        attention_config={"hisparse_config": {}, "indexer_kv_dtype": "bf16"},
        kv_transfer_config=_HISPARSE_KV_TRANSFER_CONFIG,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=2,
        kv_cache_memory_bytes=128 * 1024**2,
    ) as runner:
        (result,) = runner.llm.generate(
            {"prompt_token_ids": [1, 2, 3, 4]},
            SamplingParams(
                temperature=0, max_tokens=2, ignore_eos=True, detokenize=False
            ),
        )
        assert len(result.outputs[0].token_ids) == 2


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
        qsa_indexer,
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

    actual, _ = qsa_indexer.QSAIndexer.forward(
        indexer,
        torch.zeros(2, 2),
        torch.tensor([7, 8]),
        rows,
        attn=attn,
        qkv=torch.zeros(2, 4),
        slot_mapping=torch.arange(2),
        main_kv_cache=attn.kv_cache,
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


@requires_qsa_metadata_kernels
def test_qsa_indexer_metadata_marks_cudagraph_padding_inert() -> None:
    device = torch.device("cuda")
    builder = QSAIndexerMetadataBuilder.__new__(QSAIndexerMetadataBuilder)
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


@requires_qsa_metadata_kernels
def test_qsa_circular_buffer_metadata_keeps_only_each_requests_suffix() -> None:
    device = torch.device("cuda")
    builder = QSAIndexerMetadataBuilder.__new__(QSAIndexerMetadataBuilder)
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

    slots = qsa_common.circular_qsa_slot_mapping(
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
) -> qsa_common.QSARawKeyCache:
    return qsa_common.QSARawKeyCache(
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
    compressed_cache = qsa_common.QSACompressedKeyCache(
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
    assert raw_cache.rope_position_cache is not None
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
    layer_config = _qsa_key_cache(
        block_size=48, compress_ratio=compress_ratio
    ).get_kv_cache_config(SimpleNamespace(num_speculative_tokens=num_spec))
    assert layer_config.spec.block_size == expected


def test_qsa_kv_cache_dtype_honors_skip_layers() -> None:
    """`--kv-cache-dtype-skip-layers` keeps the listed QSA layers unquantized.

    The MTP layer's own attention is the one that matters on Flash-Next: FP8
    there cuts draft acceptance at depth while the target layers stay FP8.
    """
    cache_config = SimpleNamespace(cache_dtype="fp8", kv_cache_dtype_skip_layers=["48"])
    assert qsa_kv_cache_dtype(cache_config, "mtp.layers.48.self_attn") == "auto"
    assert qsa_kv_cache_dtype(cache_config, "model.layers.47.self_attn") == "fp8"
    cache_config.kv_cache_dtype_skip_layers = []
    assert qsa_kv_cache_dtype(cache_config, "mtp.layers.48.self_attn") == "fp8"


@requires_qsa_metadata_kernels
def test_qsa_compressed_metadata_keeps_dummy_slots_inert() -> None:
    device = torch.device("cuda")
    builder = QSAIndexerMetadataBuilder.__new__(QSAIndexerMetadataBuilder)
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
    """Padded projected Q/K rows must not affect either indexer cache."""
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
    norm = GemmaRMSNorm(64).to(device=device, dtype=torch.bfloat16)
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
    qsa_indexer.QSAIndexer.forward(
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


@requires_qsa_metadata_kernels
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
    actual = qsa_common.build_qsa_metadata_triton(
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
    expected = qsa_common._build_qsa_metadata_torch(
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


@requires_qsa_metadata_kernels
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
    actual = qsa_common.build_qsa_metadata_triton(
        common,
        *make_buffers(),
        storage_block_size=8,
        compress_ratio=4,
        k_work_metadata_buffer=actual_k_work,
    )
    expected = qsa_common._build_qsa_metadata_torch(
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
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the fp8 kernel runs on sm_100/sm_103",
)
# Per-rank heads at TP1/2/4/8.
@pytest.mark.parametrize(
    "num_query_heads,num_kv_heads", [(24, 2), (12, 1), (6, 1), (3, 1)]
)
@pytest.mark.parametrize("num_requests", [20, 100])
def test_qsa_fp8_kernel_correctness(
    num_query_heads: int, num_kv_heads: int, num_requests: int
) -> None:
    """The Blackwell fp8 kernel matches the dense reference.

    An MTP-verify-shaped batch (num_requests x 4 rows) over the engine's fp8
    cache layout (each token's per-head K|V side by side, page 1920). Covers
    short contexts (masked tails) and a CUDA-graph padding row, which must
    stay zero.
    """
    cpu_generator = torch.Generator().manual_seed(0)
    generator = torch.Generator(device="cuda").manual_seed(0)
    head_dim, page_size = 256, 1920
    rows_per_request, selection_width = 4, 2051
    num_rows = num_requests * rows_per_request
    context_lengths = torch.randint(
        600, 3 * page_size, (num_requests,), generator=cpu_generator
    ).tolist()
    pages = [math.ceil(length / page_size) for length in context_lengths]
    block_table = torch.full(
        (num_requests, max(pages)), -1, device="cuda", dtype=torch.int32
    )
    physical_pages = torch.randperm(
        sum(pages), device="cuda", dtype=torch.int32, generator=generator
    )
    for request, (first, count) in enumerate(
        zip(itertools.accumulate([0] + pages[:-1]), pages)
    ):
        block_table[request, :count] = physical_pages[first : first + count]
    token_to_req = torch.arange(
        num_requests, device="cuda", dtype=torch.int32
    ).repeat_interleave(rows_per_request)
    token_to_req[-1] = -1  # CUDA-graph padding
    logical_indices = torch.full(
        (num_rows, selection_width + 1), -1, device="cuda", dtype=torch.int32
    )
    for row in range(num_rows):
        length = context_lengths[row // rows_per_request]
        count = min(length, selection_width)
        logical_indices[row, :count] = torch.randperm(
            length, device="cuda", generator=generator
        )[:count]
        logical_indices[row, selection_width] = count
    kv_cache = (
        torch.randn(
            sum(pages),
            page_size,
            num_kv_heads,
            2 * head_dim,
            device="cuda",
            generator=generator,
        )
        .to(torch.bfloat16)
        .to(torch.float8_e4m3fn)
    )
    k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    q = torch.randn(
        num_rows, num_query_heads, head_dim, device="cuda", generator=generator
    ).to(torch.bfloat16)
    output_gate = torch.randn(
        num_rows, num_query_heads, head_dim, device="cuda", generator=generator
    ).to(torch.bfloat16)
    k_scale, v_scale = 0.5, 2.0
    # The comparison is vacuous if the public entry does not dispatch to the
    # fp8 kernel at this shape.
    assert qsa_fp8_ops.use_fp8_decode_kernel(num_rows * num_kv_heads, k_cache.dtype)

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        use_prefill_config=False,
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
        head_dim**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    expected = expected * torch.sigmoid(output_gate)

    # The reference computes garbage for the padding row (request -1); the
    # kernel must leave it zero.
    torch.testing.assert_close(actual[:-1], expected[:-1], rtol=2e-2, atol=2e-2)
    assert not actual[-1].any()


@requires_qsa_kernels
@pytest.mark.parametrize("extra_rows", [0, 1], ids=["flat-pages", "strided-pages"])
@pytest.mark.parametrize(
    "cache_dtype,repeats",
    [
        (torch.bfloat16, 1),
        (torch.float8_e4m3fn, 1),
        (torch.uint8, 1),
        (torch.float8_e4m3fn, 11),  # 33 rows x 2 KV heads crosses the FP8 threshold.
    ],
    ids=["bf16", "fp8", "fp8-bytes", "large-fp8"],
)
def test_qsa_attention_consumes_hisparse_physical_indices(
    extra_rows: int, cache_dtype: torch.dtype, repeats: int
) -> None:
    """Resolved HiSparse rows bypass the logical block table in either view."""
    from vllm.v1.hisparse.runtime import PagedCacheView

    if cache_dtype == torch.float8_e4m3fn and not (
        current_platform.has_device_capability(89)
    ):
        pytest.skip("Native FP8 pointers require sm_89 or newer")

    torch.manual_seed(7)
    num_blocks, block_size, num_kv_heads, head_dim = 4, 16, 2, 64
    row_width = num_kv_heads * 2 * head_dim
    block_stride = (2 * block_size + extra_rows) * row_width
    fp8 = cache_dtype != torch.bfloat16
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
    if cache_dtype == torch.float8_e4m3fn:
        key_cache = key_cache.view(torch.float8_e4m3fn)
        value_cache = value_cache.view(torch.float8_e4m3fn)
    selected = torch.tensor(
        [[55, 2, 43, -1, -1], [16, 63, 33, 6, 30], [-1, -1, -1, -1, -1]],
        device="cuda",
        dtype=torch.int32,
    ).repeat(repeats, 1)
    counts = (selected >= 0).sum(dim=1, keepdim=True, dtype=torch.int32)
    physical_rows = torch.where(
        selected >= 0,
        selected // block_size * view.attention_block_stride + selected % block_size,
        -1,
    )
    # Counts stay in the logical packed buffer; resolved rows have no count.
    logical_packed = torch.cat((selected, counts), dim=1)
    saved_rows = physical_rows.clone()
    num_rows = selected.shape[0]
    q = torch.randn(num_rows, 24, head_dim, device="cuda", dtype=torch.bfloat16)
    output_gate = torch.randn_like(q)
    token_to_req = torch.zeros(num_rows, device="cuda", dtype=torch.int32)
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
    block_table = identity.roll(1, dims=1)
    logical_expected = _qsa_sparse_paged_attention_reference(
        q,
        original_key,
        original_value,
        selected,
        block_table,
        token_to_req,
        head_dim**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
    ) * torch.sigmoid(output_gate)
    assert not torch.allclose(expected, logical_expected, rtol=2e-2, atol=2e-2)

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        key_cache,
        value_cache,
        physical_rows,
        block_table,
        token_to_req,
        use_prefill_config=False,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=output_gate,
        valid_counts=logical_packed[:, -1],
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(physical_rows, saved_rows, rtol=0, atol=0)


@requires_qsa_kernels
@pytest.mark.parametrize("fp8", [False, True], ids=["main-bf16", "main-fp8"])
@pytest.mark.parametrize("all_context_pages_resident", [False, True])
def test_qsa_hisparse_prefill_ignores_unused_request_table_rows(
    tmp_path,
    dist_init,
    workspace_init,
    monkeypatch,
    fp8: bool,
    all_context_pages_resident: bool,
) -> None:
    """Unequal active lengths and reordered state slots keep the right KV rows."""
    _run_qsa_hisparse_prefill_case(
        tmp_path,
        monkeypatch,
        fp8,
        False,
        all_context_pages_resident=all_context_pages_resident,
    )


@requires_qsa_kernels
def test_qsa_hisparse_warmup_outside_connector_step_skips_mirror(
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
    connector_warmup=False,
    all_context_pages_resident=False,
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
        kv_transfer_config=_HISPARSE_KV_TRANSFER_CONFIG,
    ).create_engine_config()
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            _qwen4_exp_model.get_qsa_topk_buffer_width(config),
            dtype=torch.int32,
        )
        owner = Qwen4ExpQSAAttention(
            vllm_config=vllm_config,
            config=config,
            layer_id=0,
            topk_indices_buffer=topk_indices_buffer,
            prefix="model.layers.0.self_attn",
        )
        k_scale, v_scale = (0.5, 2.0) if fp8 else (1.0, 1.0)
        AutoWeightsLoader(owner).load_weights(
            [
                ("_k_scale", torch.tensor(k_scale)),
                ("_v_scale", torch.tensor(v_scale)),
            ]
        )
        owner.process_weights_after_loading()
    # Even when fusion is supported, main KV writes must use the HiSparse
    # resident target exercised below instead of the ordinary cache.
    assert owner.use_fused_qk_norm_rope_gate
    assert owner.indexer.use_fused_pre_indexer
    # The hooks below observe the unfused selection and K/V writer.
    owner.use_fused_qsa_prepare = False
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
    if not all_context_pages_resident:
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
        seq_lens_cpu_upper_bound=torch.tensor([17, 33], dtype=torch.int32),
        block_table_tensor=source_table[:2],
        slot_mapping=slots,
    )
    builder = owner.get_attn_backend().get_builder_cls()(
        owner.get_kv_cache_config(vllm_config).spec,
        [owner.layer_name],
        vllm_config,
        torch.device("cuda"),
    )
    metadata = builder.build(0, common)
    all_metadata = {owner.layer_name: metadata}
    for side_cache in (
        owner.indexer.raw_key_cache,
        owner.indexer.compressed_key_cache,
    ):
        spec = side_cache.get_kv_cache_config(vllm_config).spec
        assert isinstance(spec, AttentionSpec)
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
        all_metadata[side_cache.prefix] = QSAIndexerMetadataBuilder(
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
    assert cache.view is not None
    cache.source_block_table = source_table
    runtime.request_state_indices = state_indices
    cache.all_context_pages_resident = all_context_pages_resident
    owner.bind_kv_cache(original.transpose(1, 2))
    if connector_warmup:
        from functools import partial
        from unittest.mock import MagicMock

        from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
            HiSparseConnectorWorker,
        )

        # Exercise the production layer-once guard. Mock only DMA submission;
        # actual cache writing, sparse attention and native graph replay run.
        worker = object.__new__(HiSparseConnectorWorker)
        worker.cache_handles = [cache]
        worker._per_layer_mirrored = set()
        worker._submitted_mirror_layers = set()
        worker._slot_mapping_staging = None
        worker._layer_ready_events = (MagicMock(),)
        worker._in_step = True
        enqueue_row_dma = MagicMock()
        monkeypatch.setattr(worker, "_enqueue_row_dma", enqueue_row_dma)
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
        if not all_context_pages_resident:
            # Only host retains this selected history page.
            cache.view.cache[3].fill_(29)
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
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        assert len(writer_inputs) == 1
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
        assert len(selected_before_attention) == 1
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
        if not all_context_pages_resident:
            assert torch.all(cache.view.cache[3] == 29)
        mirror_metadata = all_metadata
        mirror_positions, mirror_hidden = positions, hidden
        if connector_warmup:
            cache.submit_layer_mirror = partial(worker._enqueue_layer_mirror, 0)
            cache.all_context_pages_resident = True
            # JIT warmup runs prefill5 then MTP verify4 outside a connector step.
            worker._in_step = False
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
                    seq_lens_cpu_upper_bound=torch.tensor(
                        [start + length], dtype=torch.int32
                    ),
                    block_table_tensor=source_table[:1],
                    slot_mapping=warm_slots,
                )
                warm_metadata = {owner.layer_name: builder.build(0, warm_common)}
                for side_cache in (
                    owner.indexer.raw_key_cache,
                    owner.indexer.compressed_key_cache,
                ):
                    spec = side_cache.get_kv_cache_config(vllm_config).spec
                    assert isinstance(spec, AttentionSpec)
                    table = (
                        source_table[:1, :1]
                        if side_cache is owner.indexer.raw_key_cache
                        else source_table[:1]
                    )
                    warm_metadata[side_cache.prefix] = QSAIndexerMetadataBuilder(
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
            enqueue_row_dma.assert_not_called()
            assert not worker._per_layer_mirrored
            worker._in_step = True
            mirror_metadata = warm_metadata
            mirror_positions, mirror_hidden = warm_positions, warm_hidden
        if connector_warmup:
            # A real multi-token prefill still pipelines one host submission,
            # and a second live write in that phase must remain an error.
            with (
                torch.inference_mode(),
                set_current_vllm_config(vllm_config),
                set_forward_context(mirror_metadata, vllm_config, num_tokens=4),
            ):
                owner(mirror_positions, mirror_hidden)
                enqueue_row_dma.assert_called_once()
                with pytest.raises(RuntimeError, match="layer 0 mirrored twice"):
                    owner(mirror_positions, mirror_hidden)
    finally:
        selection_hook.remove()
        release_pinned_state([registered])


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


@requires_qsa_kernels
def test_qsa_hisparse_reused_request_slot_refills_all_independent_kv_layers():
    """A new owner of a host block cannot hit the old owner's full K/V rows."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
        HiSparseConnectorMetadata,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
        HiSparseConnectorWorker,
    )
    from vllm.v1.hisparse.runtime import (
        HiSparseCacheHandle,
        HiSparseRuntime,
        ResolvedHiSparseConfig,
    )
    from vllm.v1.hisparse.types import SparseKVResidencyUpdate
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")
    block_size, heads, head_dim = 64, 2, 64
    row_width = heads * 2 * head_dim
    block_bytes = block_size * row_width * torch.bfloat16.itemsize
    states = RequestState(
        max_num_reqs=2,
        max_model_len=64,
        max_num_batched_tokens=2,
        num_speculative_steps=0,
        vocab_size=128,
        device=device,
    )
    for name in ("old", "peer"):
        states.add_request(name, 1, [1], 0, 4)
    states.apply_staged_writes()
    old_slot = states.req_id_to_index["old"]
    peer_slot = states.req_id_to_index["peer"]
    mapping = torch.full((2,), -1, dtype=torch.int32, device=device)
    residency = torch.zeros((1, 2, 1), dtype=torch.int32, device=device)
    # The HMA slab is shared, while hot groups own different allocation blocks.
    backing = torch.zeros(6 * block_bytes, dtype=torch.uint8, device=device)
    handles, host_pools = [], []
    for layer in range(2):
        runtime = HiSparseRuntime(
            ResolvedHiSparseConfig(top_k=2, device_buffer_size=4, max_union_rows=2),
            max_num_reqs=2,
            row_width=row_width,
            kv_dtype=torch.bfloat16,
            device=device,
        )
        runtime.bind_hot_cache(
            backing,
            byte_offset=0,
            block_stride=block_bytes,
            num_blocks=6,
            block_size=block_size,
            block_table=torch.arange(
                1 + 2 * layer, 3 + 2 * layer, dtype=torch.int32, device=device
            )[:, None],
        )
        runtime.request_state_indices = mapping
        # Each QSA layer leads its own index group and carries every K/V head.
        assert runtime.is_group_leader
        host = (
            torch.arange(3 * block_size * row_width, dtype=torch.float32, device="cpu")
            .remainder(29)
            .add(layer * 32)
            .to(torch.bfloat16)
            .reshape(3, block_size, row_width)
            .pin_memory()
        )
        runtime.bind_source_cache(host)
        handle = HiSparseCacheHandle(runtime)
        handle.bind_cache(
            backing,
            byte_offset=0,
            block_stride=block_bytes,
            num_blocks=6,
            block_size=block_size,
            block_table=residency[0],
            slot_mapping=torch.zeros(2, dtype=torch.int64, device=device),
        )
        handle.residency = residency
        handles.append(handle)
        host_pools.append(host)
    assert handles[0].runtime.index_group is not handles[1].runtime.index_group
    worker = HiSparseConnectorWorker(
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=2),
            num_lookahead_tokens=0,
        ),
        KVCacheConfig(num_blocks=6, kv_cache_tensors=[], kv_cache_groups=[]),
    )
    worker.initialize(
        handles,
        ["qsa.layer.0", "qsa.layer.1"],
        2,
        3,
        device,
        host_pools,
    )

    def start(names, new_host_blocks):
        worker.start_step(
            HiSparseConnectorMetadata(
                None,
                (),
                new_host_blocks,
                {},
                False,
                {name: SparseKVResidencyUpdate([0], ([0],)) for name in names},
            ),
            torch.tensor(
                [states.req_id_to_index[name] for name in names],
                dtype=torch.int32,
                device=device,
            ),
            request_ids=names,
        )

    def read(handle, source_blocks, selected):
        handle.runtime.begin_forward()
        indices, counts = handle.swap_in(
            req_id_per_token=torch.arange(2, dtype=torch.int32, device=device),
            block_table=torch.tensor(source_blocks, dtype=torch.int32, device=device)[
                :, None
            ],
            logical_topk_indices=torch.tensor(
                selected, dtype=torch.int32, device=device
            ),
            block_size=block_size,
            num_valid_rows=torch.tensor([2], dtype=torch.int32, device=device),
            return_valid_counts=True,
        )
        torch.accelerator.synchronize()
        assert counts.tolist() == [2, 2]
        return handle.runtime.hot.attention_cache.reshape(-1, row_width)[
            indices.long()
        ].cpu()

    start(["old", "peer"], (1, 2))
    old_rows, peer_mappings, before_misses = [], [], []
    for handle, host in zip(handles, host_pools):
        actual = read(handle, [1, 2], [[3, 7], [5, 9]])
        expected = torch.stack((host[1, [3, 7]], host[2, [5, 9]]))
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8))
        old_rows.append(actual[0].clone())
        group = handle.runtime.index_group
        assert {block_size + 3, block_size + 7} <= set(
            group.device_global_indices[old_slot].tolist()
        )
        peer_mappings.append(group.device_global_indices[peer_slot].clone())
        before_misses.append(int(group.swap_stats[1].item()))

    assert states.remove_request("old") == old_slot
    states.add_request("replacement", 1, [2], 0, 4)
    states.apply_staged_writes()
    assert states.req_id_to_index["replacement"] == old_slot
    for host in host_pools:
        host[1].add_(128)
    # Hot tables follow batch order, while residency and identities use state slots.
    for handle in handles:
        hot_table = handle.runtime.hot_block_table
        hot_table.copy_(hot_table.flip(0))
    # Reorder batch rows as well as replacing the request, so batch row != state slot.
    start(["peer", "replacement"], (1,))
    assert mapping.tolist() == [peer_slot, old_slot]
    for layer, (handle, host) in enumerate(zip(handles, host_pools)):
        group = handle.runtime.index_group
        assert (group.device_global_indices[old_slot] == -1).all()
        torch.testing.assert_close(
            group.device_global_indices[peer_slot], peer_mappings[layer]
        )
        actual = read(handle, [2, 1], [[5, 9], [3, 7]])
        expected = torch.stack((host[2, [5, 9]], host[1, [3, 7]]))
        assert not torch.equal(actual[1], old_rows[layer])
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8))
        assert int(group.swap_stats[1].item()) - before_misses[layer] == 2
    torch.accelerator.synchronize()
