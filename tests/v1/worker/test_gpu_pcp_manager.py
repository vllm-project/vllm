# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from tests.v1.spec_decode.utils import fake_speculative_config
from vllm.config import CUDAGraphMode, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.linear_attn import LinearAttentionBackend
from vllm.v1.attention.backends.utils import PAD_SLOT_ID, get_dcp_local_seq_lens
from vllm.v1.attention.ops import pcp as attention_pcp
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm.v1.worker.gpu import cp_utils as gpu_cp_utils
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, set_dummy_context
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig
from vllm.v1.worker.utils import AttentionGroup


@pytest.mark.parametrize("pcp_size", [1, 2, 4])
def test_mamba_pcp_metadata_capacity_preserves_scheduler_request_limit(pcp_size):
    config = VllmConfig(
        parallel_config=ParallelConfig(
            prefill_context_parallel_size=pcp_size,
            distributed_executor_backend="mp",
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_model_len=64,
            is_encoder_decoder=False,
        ),
    )
    local_group = AttentionGroup(
        cast(type[AttentionBackend], LinearAttentionBackend),
        ["attn"],
        AttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32),
        0,
    )
    local_group.create_metadata_builders(config, torch.device("cpu"))
    local_config = local_group.get_metadata_builder().vllm_config
    assert local_config.scheduler_config.max_num_seqs == (8 if pcp_size > 1 else 4)
    assert local_config.parallel_config is config.parallel_config

    mamba_group = AttentionGroup(
        cast(type[AttentionBackend], LinearAttentionBackend),
        ["linear"],
        MambaSpec(block_size=16, shapes=((2,),), dtypes=(torch.float32,)),
        1,
    )
    mamba_group.create_metadata_builders(config, torch.device("cpu"))
    mamba_config = mamba_group.get_metadata_builder().vllm_config
    assert mamba_config.scheduler_config.max_num_seqs == 4
    assert config.scheduler_config.max_num_seqs == 4


@pytest.mark.parametrize("shard_decode", [False, True])
@pytest.mark.parametrize("prefill_lengths", [[17], [1, 3, 19]])
def test_mamba_pcp_restores_causal_tokens_without_duplicate_state_rows(
    shard_decode, prefill_lengths
):
    """DualChunkSwap rows share a request, but recurrent updates must not."""
    device = torch.device("cuda:0")
    lengths = np.array([3, *prefill_lengths], dtype=np.int32)
    num_reqs, num_tokens = len(lengths), int(lengths.sum())
    buffers = InputBuffers(8, 64, device)
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)
    starts = np.concatenate(([0], lengths.cumsum())).astype(np.int32)
    batch.query_start_loc.copy_(torch.from_numpy(starts).to(device))
    batch.seq_lens.copy_(torch.from_numpy(lengths).to(device))
    batch.input_ids.copy_(torch.arange(num_tokens, device=device))
    batch.positions.copy_(
        torch.cat([torch.arange(int(length), device=device) for length in lengths])
    )
    prefilling = np.arange(num_reqs) > 0
    batch = replace(
        batch,
        num_scheduled_tokens=lengths,
        query_start_loc_np=starts,
        is_prefilling_np=prefilling,
        has_prefill=True,
        decode_graph_eligible=False,
        prefill_len_np=np.where(prefilling, lengths, 0),
        seq_lens_cpu_upper_bound=torch.from_numpy(lengths.copy()),
        num_draft_tokens_per_req=np.array([2, *([-1] * len(prefill_lengths))]),
    )
    local_batches = []
    contexts = []
    for rank in range(2):
        tables = BlockTables(
            [16], 8, 64, [8], device, slot_mapping_enabled=[False], dcp_sharded=[False]
        )
        manager = PCPManager(
            2,
            rank,
            device,
            shard_decode,
            max_num_reqs=8,
            max_num_tokens=64,
            block_tables=tables,
            has_mamba=True,
        )
        local_batches.append(
            manager.partition_batch(
                batch,
                BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE, num_tokens=64, num_reqs=num_reqs
                ),
            )
        )
        assert manager.mamba_context is not None
        contexts.append(manager.mamba_context)
    gathered = torch.cat([local.input_ids for local in local_batches])
    for local, context in zip(local_batches, contexts):
        assert context is not None
        mamba_batch = context.mamba_batch
        mapping = context.token_mapping
        restored = gathered[mapping.restore_indices][mapping.restore_mask]
        torch.testing.assert_close(restored, batch.input_ids, rtol=0, atol=0)
        assert mamba_batch.num_reqs == num_reqs
        np.testing.assert_array_equal(mamba_batch.num_scheduled_tokens, lengths)
        np.testing.assert_array_equal(
            mamba_batch.num_draft_tokens_per_req, batch.num_draft_tokens_per_req
        )
        torch.testing.assert_close(
            mamba_batch.query_start_loc[: num_reqs + 1], batch.query_start_loc
        )
        torch.testing.assert_close(
            restored[mapping.local_indices][mapping.local_mask],
            local.input_ids[: local.num_tokens],
        )


def test_mamba_pcp_preprocess_state_dispatches_paired_batch_and_tables():
    device = torch.device("cuda:0")
    batch = InputBatch.make_dummy(2, 8, InputBuffers(8, 32, device))
    kv_cache_config = SimpleNamespace()
    num_computed = torch.zeros(8, dtype=torch.int32, device=device)
    descriptor = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=32, num_reqs=2
    )

    recorded: list[tuple[InputBatch, tuple[torch.Tensor, ...]]] = []
    model_state = SimpleNamespace(
        preprocess_state=lambda b, bt, _cfg, _nc: recorded.append((b, bt))
    )

    mamba_tables = BlockTables(
        [16], 8, 32, [8], device, slot_mapping_enabled=[False], dcp_sharded=[False]
    )
    mamba_mgr = PCPManager(
        2,
        0,
        device,
        True,
        max_num_reqs=8,
        max_num_tokens=32,
        block_tables=mamba_tables,
        has_mamba=True,
    )
    local_mamba = mamba_mgr.partition_batch(batch, descriptor)
    local_mamba_bt, _ = mamba_mgr.prepare_attn(local_mamba)
    mamba_mgr.preprocess_state(
        model_state, local_mamba, local_mamba_bt, kv_cache_config, num_computed
    )
    assert mamba_mgr.mamba_context is not None
    assert recorded[-1][0] is mamba_mgr.mamba_context.mamba_batch
    assert recorded[-1][1] is mamba_mgr.mamba_context.block_tables

    attn_tables = BlockTables(
        [16], 8, 32, [8], device, slot_mapping_enabled=[False], dcp_sharded=[False]
    )
    attn_mgr = PCPManager(
        2,
        0,
        device,
        True,
        max_num_reqs=8,
        max_num_tokens=32,
        block_tables=attn_tables,
        has_mamba=False,
    )
    local_attn = attn_mgr.partition_batch(batch, descriptor)
    local_attn_bt, _ = attn_mgr.prepare_attn(local_attn)
    attn_mgr.preprocess_state(
        model_state, local_attn, local_attn_bt, kv_cache_config, num_computed
    )
    assert attn_mgr.mamba_context is None
    assert recorded[-1][0] is local_attn
    assert recorded[-1][1] is local_attn_bt


@pytest.mark.parametrize("query_len", [2, 4])
def test_mamba_pcp_empty_rank_keeps_full_graph_query_width(query_len):
    device = torch.device("cuda:0")
    batch = InputBatch.make_dummy(1, query_len, InputBuffers(4, 16, device))
    batch = replace(
        batch,
        is_prefilling_np=np.zeros(1, dtype=np.bool_),
        has_prefill=False,
        decode_graph_eligible=True,
    )
    manager = PCPManager(2, 1, device, True, max_num_reqs=4, max_num_tokens=16)
    local = manager.partition_batch(
        batch,
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=query_len,
            num_reqs=1,
            uniform_token_count=query_len,
        ),
    )
    # PCP retains one inactive request row so every rank can run the graph.
    assert local.num_reqs == 1
    assert local.num_tokens == 0
    np.testing.assert_array_equal(local.num_scheduled_tokens, [0])
    assert local.num_reqs_after_padding == 1
    assert local.num_tokens_after_padding == query_len
    assert local.max_query_len == query_len
    torch.testing.assert_close(
        local.query_start_loc, torch.zeros(2, device=device, dtype=torch.int32)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU kernels")
@pytest.mark.parametrize("active_rows", [0, 3])
def test_flashinfer_mla_sparse_sm90_slices_padded_mqa_rows_to_planned_width(
    active_rows,
):
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
        FlashInferMLASparseSM90Impl,
    )

    device = torch.device("cuda:0")
    num_tokens, num_heads, kv_lora_rank = 4, 2, 16
    impl = object.__new__(FlashInferMLASparseSM90Impl)
    impl.num_heads = num_heads
    impl.qk_rope_head_dim = 0
    impl.kv_lora_rank = kv_lora_rank
    impl.head_size = kv_lora_rank
    impl.use_fp8_kv_cache = False
    impl.topk_indices_buffer = torch.zeros((8, 128), dtype=torch.int32, device=device)

    packed: list[int] = []

    def fake_run(q_nope, q_pe, _ckv, _kpe, **_kwargs):
        assert q_nope.shape[:2] == (active_rows, num_heads)
        assert q_pe.shape == (active_rows, num_heads, 0)
        return torch.full(
            (active_rows, num_heads, kv_lora_rank),
            3.0,
            dtype=q_nope.dtype,
            device=device,
        )

    state = SimpleNamespace(
        pack_indices=lambda slots: packed.append(slots.shape[0]),
        wrapper=SimpleNamespace(run=fake_run),
    )
    metadata = SimpleNamespace(
        num_mqa_rows=active_rows,
        block_size=4,
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32, device=device),
        block_table=torch.zeros((2, 2), dtype=torch.int32, device=device),
        state=state,
    )
    q_nope = torch.ones(
        (num_tokens, num_heads, kv_lora_rank), dtype=torch.bfloat16, device=device
    )
    q_rope = torch.empty((num_tokens, 0), dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros((2, 4, kv_lora_rank), dtype=torch.bfloat16, device=device)
    out, lse = impl.forward_mqa((q_nope, q_rope), kv_cache, metadata, SimpleNamespace())
    assert lse is None
    assert out.shape == (num_tokens, num_heads, kv_lora_rank)
    if active_rows == 0:
        assert packed == []
        assert torch.count_nonzero(out).item() == 0
    else:
        assert packed == [active_rows]
        assert torch.all(out[:active_rows] == 3.0)
        assert torch.count_nonzero(out[active_rows:]).item() == 0


@pytest.mark.parametrize("shard_decode", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_mamba_pcp_capture_stages_global_decode_at_max_request_capacity(
    shard_decode, rank
):
    device = torch.device("cuda:0")
    tables = BlockTables(
        [16], 4, 16, [8], device, slot_mapping_enabled=[False], dcp_sharded=[False]
    )
    manager = PCPManager(
        2,
        rank,
        device,
        shard_decode,
        max_num_reqs=4,
        max_num_tokens=16,
        block_tables=tables,
        has_mamba=True,
    )
    dummy = InputBatch.make_dummy(4, 4, InputBuffers(4, 16, device), max_query_len=1)
    manager.prepare_inputs_for_capture_or_profile(dummy)
    context = manager.mamba_context
    assert context is not None
    assert context.mamba_batch.num_reqs == 4
    assert context.mamba_batch.num_reqs_after_padding == 4
    assert context.mamba_batch.max_query_len == 1
    assert context.mamba_batch.num_tokens_after_padding == 4
    assert context.token_mapping.restore_indices.shape[0] == 8
    start = rank * 4 if shard_decode else 0
    torch.testing.assert_close(
        context.token_mapping.local_mask,
        torch.arange(start, start + 4, device=device) < context.mamba_batch.num_tokens,
    )
    torch.testing.assert_close(
        context.mamba_batch.seq_lens, torch.ones(4, device=device, dtype=torch.int32)
    )
    pointers = [
        getattr(context.token_mapping, name).data_ptr()
        for name in ("restore_indices", "restore_mask", "local_indices", "local_mask")
    ]
    manager.prepare_inputs_for_capture_or_profile(dummy)
    replay = manager.mamba_context
    assert replay is not None
    assert pointers == [
        getattr(replay.token_mapping, name).data_ptr()
        for name in ("restore_indices", "restore_mask", "local_indices", "local_mask")
    ]

    full_prefill_dummy = InputBatch.make_dummy(4, 16, InputBuffers(4, 16, device))
    manager.prepare_inputs_for_capture_or_profile(full_prefill_dummy)
    prefill_ctx = manager.mamba_context
    assert prefill_ctx is not None
    assert prefill_ctx.mamba_batch.num_tokens == 16
    assert prefill_ctx.mamba_batch.num_tokens_after_padding == 16
    assert prefill_ctx.token_mapping.restore_indices.shape[0] == 32

    sub_decode_dummy = InputBatch.make_dummy(
        2, 2, InputBuffers(4, 16, device), max_query_len=1
    )
    manager.prepare_inputs_for_capture_or_profile(sub_decode_dummy)
    sub_ctx = manager.mamba_context
    assert sub_ctx is not None
    expected_sub = 4 if shard_decode else 2
    assert sub_ctx.mamba_batch.num_reqs == expected_sub
    assert sub_ctx.mamba_batch.num_tokens_after_padding == expected_sub
    assert sub_ctx.token_mapping.restore_indices.shape[0] == 4

    dummy_full_desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=4,
        num_reqs=4,
        uniform_token_count=1,
    )
    warmup_dummy = InputBatch.make_dummy(4, 4, InputBuffers(4, 16, device))
    assert warmup_dummy.max_query_len is None
    staged = manager.prepare_inputs_for_capture_or_profile(
        warmup_dummy, dummy_full_desc
    )
    assert staged.max_query_len == 1
    assert manager.mamba_context is not None
    assert manager.mamba_context.mamba_batch.num_tokens_after_padding == 4


def _copy_to_cpu(value, device=None, dtype=None, out=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def _make_config(cudagraph_mode: CUDAGraphMode):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            dcp_comm_backend="ag_rs",
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        lora_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


def _make_capture_manager(block_table: torch.Tensor):
    block_tables = SimpleNamespace(
        input_block_tables=(block_table,),
        num_kv_cache_groups=1,
        block_sizes=(2,),
        cp_interleave=1,
    )
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        max_num_reqs=4,
        max_num_tokens=8,
        block_tables=block_tables,
    )
    return manager, block_tables


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_validate_config_accepts_decode_only_full_graphs(cudagraph_mode):
    PCPManager.validate_config(_make_config(cudagraph_mode), supports_mm_inputs=False)


def test_validate_config_rejects_full_graph_for_prefills():
    with pytest.raises(NotImplementedError, match="decode-only routines"):
        PCPManager.validate_config(
            _make_config(CUDAGraphMode.FULL), supports_mm_inputs=False
        )


@pytest.mark.parametrize("draft_uses_mla", [False, True])
@pytest.mark.parametrize("dcp_size", [1, 2, 4])
def test_validate_config_dcp_for_dspark(draft_uses_mla, dcp_size):
    config = _make_config(CUDAGraphMode.NONE)
    config.parallel_config.prefill_context_parallel_size = 4
    config.parallel_config.decode_context_parallel_size = dcp_size
    config.model_config.hf_text_config.index_topk = 1
    config.speculative_config = fake_speculative_config(
        DSparkSpeculatorConfig,
        draft_decode_context_parallel_size=dcp_size if draft_uses_mla else 1,
    )
    if draft_uses_mla and dcp_size == 2:
        with pytest.raises(NotImplementedError, match="requires DCP=1 or DCP=PCP"):
            PCPManager.validate_config(config, supports_mm_inputs=False)
        return
    PCPManager.validate_config(config, supports_mm_inputs=False)


def test_sharded_decode_piecewise_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_computed_tokens=np.full(3, 16, dtype=np.int32),
        is_prefilling=np.zeros(3, dtype=np.bool_),
        query_start_loc_np=np.arange(4, dtype=np.int32),
        padded_num_tokens=4,
    )

    assert per_rank_num_tokens == [2, 1]
    request_indices = [
        [segment.global_batch_req_idx for segment in rank] for rank in segments_by_rank
    ]
    assert request_indices == [[0, 2], [1]]
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 4, 1]))
    assert torch.equal(
        manager._padded_gather_idx,
        torch.tensor([0, 2, 0, 0, 1, 0, 0, 0]),
    )
    assert torch.equal(
        manager._gathered_kv_write_mask,
        torch.tensor([True, True, False, False, True, False, False, False]),
    )


def test_input_buffers_are_exposed_for_cudagraph_capture():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        max_num_reqs=4,
        max_num_tokens=8,
    )

    assert manager.input_buffers is manager._input_buffers
    assert manager.input_buffers.input_ids.shape == (8,)
    assert manager.input_buffers.positions.shape == (8,)
    assert manager.input_buffers.is_padding.shape == (8,)


def test_sync_drafts_writes_logits_to_request_slots(monkeypatch):
    """Rank 0's draft logits land in each batch request's persistent slot."""
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=1,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        max_num_reqs=4,
        max_num_tokens=8,
    )
    root_tokens = torch.tensor([[11, 12, 13]])
    root_logits = torch.arange(24, dtype=torch.float32).view(1, 3, 8)
    broadcasts = iter([root_tokens, root_logits])
    group = SimpleNamespace(
        broadcast=lambda tensor, src: tensor.copy_(next(broadcasts))
    )
    monkeypatch.setattr(pcp_manager_module, "get_pcp_group", lambda: group)
    draft_tokens = torch.full((1, 3), -1)
    draft_logits = DraftLogitsCache(
        4, 3, 8, dtype=torch.float32, fill=-1.0, device="cpu"
    )

    manager.sync_drafts(draft_tokens, torch.tensor([2]), draft_logits)

    torch.testing.assert_close(draft_tokens, root_tokens)
    torch.testing.assert_close(draft_logits.values[2], root_logits[0])
    assert (draft_logits.values[[0, 1, 3]] == -1).all()


@pytest.mark.parametrize(
    ("pcp_world_size", "num_scheduled_tokens", "is_prefilling", "expected"),
    [
        (2, [8], [True], 4),
        (2, [7], [True], 4),
        (2, [3], [False], 3),
        (2, [3, 8], [False, True], 7),
        (4, [2, 9], [False, True], 4),
    ],
)
def test_num_tokens_for_dispatch_uses_largest_pcp_rank(
    pcp_world_size, num_scheduled_tokens, is_prefilling, expected
):
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
    )

    actual = manager.get_num_tokens_for_dispatch(
        np.asarray(num_scheduled_tokens, dtype=np.int32),
        np.asarray(is_prefilling, dtype=np.bool_),
    )

    assert actual == expected


def test_graph_padding_cannot_be_smaller_than_largest_pcp_rank(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)

    with pytest.raises(ValueError, match="smaller than the largest rank-local batch"):
        manager._build_batch_layout(
            num_scheduled_tokens=np.ones(3, dtype=np.int32),
            num_computed_tokens=np.full(3, 16, dtype=np.int32),
            is_prefilling=np.zeros(3, dtype=np.bool_),
            query_start_loc_np=np.arange(4, dtype=np.int32),
            padded_num_tokens=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU kernels")
@pytest.mark.parametrize(
    ("cg_mode", "num_reqs", "expected_tokens", "expected_reqs"),
    [
        (CUDAGraphMode.NONE, 4, 2, 2),
        (CUDAGraphMode.PIECEWISE, None, 8, 2),
        (CUDAGraphMode.FULL, 4, 8, 4),
    ],
)
def test_partition_padding_is_derived_from_batch_descriptor(
    cg_mode, num_reqs, expected_tokens, expected_reqs
):
    """partition_batch derives graph padding from the batch descriptor: no
    padding without graphs, token padding only for PIECEWISE, and request
    padding on top for FULL."""
    device = torch.device("cuda:0")
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=device,
        shard_decode_requests=False,
        max_num_reqs=4,
        max_num_tokens=8,
    )
    global_batch = _make_global_decode_batch(
        [16, 24], InputBuffers(4, 8, device), device
    )

    local_batch = manager.partition_batch(
        global_batch,
        BatchExecutionDescriptor(
            cg_mode=cg_mode,
            num_tokens=8,
            num_reqs=num_reqs,
        ),
    )

    assert local_batch.num_tokens_after_padding == expected_tokens
    assert local_batch.num_reqs_after_padding == expected_reqs


def test_dummy_draft_does_not_reuse_previous_graph_batch():
    manager, _ = _make_capture_manager(torch.ones((4, 2), dtype=torch.int32))
    dummy_batch = InputBatch.make_dummy(1, 4, manager.input_buffers)
    manager.draft_prefill_batch = replace(dummy_batch)

    manager.prepare_draft_prefill(dummy_batch, dummy_batch.input_ids)

    draft_inputs, _ = manager.get_draft_input_buffers(
        manager.input_buffers, dummy_batch.num_tokens_after_padding
    )
    assert draft_inputs is manager.input_buffers


def test_capture_uses_pcp_persistent_inputs():
    manager, _ = _make_capture_manager(torch.ones((4, 2), dtype=torch.int32))

    dummy_batch = InputBatch.make_dummy(
        num_reqs=4,
        num_tokens=4,
        input_buffers=InputBuffers(4, 8, torch.device("cpu")),
        max_query_len=1,
    )
    dummy_batch.input_ids.copy_(torch.arange(4))
    dummy_batch.positions.fill_(7)
    input_batch = manager.prepare_inputs_for_capture_or_profile(dummy_batch)

    assert input_batch is not dummy_batch
    assert input_batch.req_ids == dummy_batch.req_ids
    for name in ("input_ids", "positions", "is_padding", "query_start_loc", "seq_lens"):
        actual = getattr(input_batch, name)
        torch.testing.assert_close(actual, getattr(dummy_batch, name))
        assert actual.data_ptr() == getattr(manager.input_buffers, name).data_ptr()


def test_block_table_rebind_updates_interleave():
    manager, block_tables = _make_capture_manager(torch.ones((4, 2), dtype=torch.int32))
    final_block_tables = SimpleNamespace(**vars(block_tables))
    final_block_tables.cp_interleave = 64

    manager.set_block_tables(final_block_tables)

    assert manager._block_tables is final_block_tables
    assert manager.cp_interleave == 64


def test_dummy_context_updates_pcp_local_block_tables():
    global_block_table = torch.full((4, 4), -1, dtype=torch.int32)
    manager, block_tables = _make_capture_manager(global_block_table)
    input_batch = InputBatch.make_dummy(
        num_reqs=2,
        num_tokens=2,
        input_buffers=manager.input_buffers,
        max_query_len=1,
    )
    input_batch = manager.prepare_inputs_for_capture_or_profile(input_batch)
    local_block_tables = manager.get_dummy_block_tables(input_batch.num_reqs)

    set_dummy_context(
        input_batch,
        block_tables,
        context_len=3,
        num_kv_blocks=16,
        max_model_len=16,
        input_block_tables=local_block_tables,
    )

    torch.testing.assert_close(
        local_block_tables[0][:2, :2],
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
    )
    assert torch.all(global_block_table == -1)


def _rank_request_ids(
    manager: PCPManager,
    rank: int,
    req_ids: list[str],
    *,
    is_prefilling: np.ndarray | None = None,
) -> list[str]:
    num_reqs = len(req_ids)
    if is_prefilling is None:
        is_prefilling = np.zeros(num_reqs, dtype=np.bool_)
    segments = manager._get_rank_segments(
        rank=rank,
        num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
        is_prefilling=is_prefilling,
        query_start_loc_np=np.arange(num_reqs + 1, dtype=np.int32),
    )
    return [req_ids[segment.global_batch_req_idx] for segment in segments]


def test_pcp_only_decode_requests_are_round_robin_balanced_each_step():
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    req_ids = [f"request-{idx}" for idx in range(18)]

    owners: dict[str, int] = {}
    for rank in range(manager.pcp_world_size):
        for req_id in _rank_request_ids(manager, rank, req_ids):
            assert req_id not in owners
            owners[req_id] = rank

    assert owners == {
        req_id: index % manager.pcp_world_size for index, req_id in enumerate(req_ids)
    }
    counts = [list(owners.values()).count(rank) for rank in range(4)]
    assert max(counts) - min(counts) == 1

    reordered_req_ids = req_ids[::2] + req_ids[1::2]
    reordered_owners = {
        req_id: rank
        for rank in range(manager.pcp_world_size)
        for req_id in _rank_request_ids(manager, rank, reordered_req_ids)
    }
    assert reordered_owners == {
        req_id: index % manager.pcp_world_size
        for index, req_id in enumerate(reordered_req_ids)
    }
    assert reordered_owners != owners


def test_decode_requests_remain_replicated_when_dcp_is_enabled():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        dcp_world_size=2,
    )
    req_ids = ["request-a", "request-b", "request-c"]

    assert _rank_request_ids(manager, 0, req_ids) == req_ids
    assert _rank_request_ids(manager, 1, req_ids) == req_ids


def test_decode_sharding_allows_ranks_with_no_owned_request():
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    req_ids = ["request-a"]

    assert _rank_request_ids(manager, 0, req_ids) == req_ids
    assert _rank_request_ids(manager, 1, req_ids) == []
    assert _rank_request_ids(manager, 2, req_ids) == []
    assert _rank_request_ids(manager, 3, req_ids) == []


def test_prefill_partitioning_is_preserved_with_sharded_decode():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    segments_by_rank = [
        manager._get_rank_segments(
            rank=rank,
            num_scheduled_tokens=np.array([8, 1, 0, 1], dtype=np.int32),
            is_prefilling=np.array([True, False, False, False]),
            query_start_loc_np=np.array([0, 8, 9, 9, 10], dtype=np.int32),
        )
        for rank in range(manager.pcp_world_size)
    ]

    prefill_tokens = sorted(
        token_idx
        for segments in segments_by_rank
        for segment in segments
        if segment.global_batch_req_idx == 0
        for token_idx in range(
            segment.global_batch_slice.start, segment.global_batch_slice.stop
        )
    )
    decode_owners = {
        segment.global_batch_req_idx: rank
        for rank, segments in enumerate(segments_by_rank)
        for segment in segments
        if segment.global_batch_req_idx in (1, 3)
    }

    assert prefill_tokens == list(range(8))
    assert decode_owners == {1: 0, 3: 1}


def test_sharded_decode_layout_selects_owner_kv_for_replication(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([1, 1, 1], dtype=np.int32),
        num_computed_tokens=np.array([16, 16, 16], dtype=np.int32),
        is_prefilling=np.array([False, False, False]),
        query_start_loc_np=np.array([0, 1, 2, 3], dtype=np.int32),
    )

    gathered_slot_mapping = manager._convert_to_gathered_slot_mappings(
        torch.tensor([[123, 456, 789]], dtype=torch.int64)
    )
    assert torch.equal(
        gathered_slot_mapping,
        torch.tensor([[123, 789, 456, PAD_SLOT_ID]], dtype=torch.int64),
    )
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 2, 1]))

    class FakePCPGroup:
        world_size = 2

        def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
            assert dim == 0
            assert tensor.shape == (2, 1)
            return torch.tensor([[11.0], [33.0], [22.0], [0.0]])

    monkeypatch.setattr(attention_pcp, "get_pcp_group", FakePCPGroup)
    (gathered_kv,), cache_slot_mapping = attention_pcp._gather_prefill_cache_inputs(
        (torch.tensor([[11.0], [33.0]]),),
        gathered_slot_mapping[0],
        num_decode_tokens=0,
        shard_decode_requests=True,
    )

    assert torch.equal(gathered_kv, torch.tensor([[11.0], [33.0], [22.0], [0.0]]))
    assert torch.equal(cache_slot_mapping, torch.tensor([123, 789, 456, PAD_SLOT_ID]))


def test_replicated_draft_cache_inputs_keep_unexpanded_slot_mapping(monkeypatch):
    def unexpected_gather():
        pytest.fail("replicated draft decode must not gather PCP cache inputs")

    monkeypatch.setattr(attention_pcp, "get_pcp_group", unexpected_gather)
    kv = torch.tensor([[11.0], [22.0], [0.0]])
    slot_mapping = torch.tensor([123, 456])

    (cache_kv,), cache_slot_mapping = attention_pcp._gather_prefill_cache_inputs(
        (kv,), slot_mapping, num_decode_tokens=2, shard_decode_requests=True
    )

    assert torch.equal(cache_kv, kv[:2])
    assert torch.equal(cache_slot_mapping, slot_mapping)


def _rank_rows(
    pcp_rank: int,
    pcp_world_size: int,
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    is_prefilling: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One PCP+DCP rank's rows as (global request, extent)."""
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=pcp_rank,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        dcp_world_size=2,
    )
    query_start_loc_np = np.concatenate([[0], np.cumsum(num_scheduled_tokens)]).astype(
        np.int32
    )
    segments = manager._get_rank_segments(
        pcp_rank, num_scheduled_tokens, is_prefilling, query_start_loc_np
    )
    rows = np.array([segment.global_batch_req_idx for segment in segments])
    extents = (num_computed_tokens + num_scheduled_tokens)[rows]
    return rows, extents


@pytest.mark.parametrize(
    ("num_scheduled_tokens", "num_computed_tokens"),
    [
        ([32, 32], [0, 0]),
        ([32, 32, 32, 32], [0, 0, 0, 0]),
        ([32, 32], [100, 0]),  # one continued, one fresh
        ([64, 32, 48], [0, 512, 0]),  # ragged lengths and mixed contexts
        # A replicated prefill BEHIND a split one. Sorting replicated rows first
        # put request 1 ahead of request 0.
        ([64, 3], [0, 0]),
        ([64, 3, 64], [0, 0, 0]),
    ],
)
@pytest.mark.parametrize("pcp_world_size", [2, 4])
def test_published_row_order_is_identical_on_every_pcp_rank(
    num_scheduled_tokens, num_computed_tokens, pcp_world_size
):
    """Every rank must map its rows to the same global requests, in the same order."""
    num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
    num_computed_tokens = np.array(num_computed_tokens, dtype=np.int32)
    is_prefilling = np.ones(len(num_scheduled_tokens), dtype=np.bool_)

    orders = [
        _rank_rows(
            rank,
            pcp_world_size,
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
        )[0]
        for rank in range(pcp_world_size)
    ]
    for rank, order in enumerate(orders[1:], start=1):
        assert np.array_equal(orders[0], order), (
            f"rank {rank} rows map to {order.tolist()}, rank 0 to {orders[0].tolist()}"
        )
    # Grouped by request and ascending: what the indexer plan indexes by.
    assert np.all(np.diff(orders[0]) >= 0)


def test_published_row_order_puts_every_decode_before_every_prefill():
    """split_decodes_and_prefills takes the FIRST prefilling row as the boundary."""
    # req 0 is a continued prefill, req 1 is a decode.
    req_idx, _ = _rank_rows(
        pcp_rank=0,
        pcp_world_size=2,
        num_scheduled_tokens=np.array([32, 1], dtype=np.int32),
        num_computed_tokens=np.array([100, 20], dtype=np.int32),
        is_prefilling=np.array([True, False], dtype=np.bool_),
    )
    # The decode (request 1) must be row 0.
    assert req_idx[0] == 1


def test_split_prefill_rows_repeat_the_request_and_its_full_extent():
    """Equal adjacent request indices are what let a backend share a KV region."""
    req_idx, extents = _rank_rows(
        pcp_rank=1,
        pcp_world_size=2,
        num_scheduled_tokens=np.array([64, 3], dtype=np.int32),
        num_computed_tokens=np.array([100, 0], dtype=np.int32),
        is_prefilling=np.ones(2, dtype=np.bool_),
    )
    assert req_idx.tolist() == [0, 0, 1]
    assert extents.tolist() == [164, 164, 3]


@pytest.mark.parametrize("query_len", [1, 2, 3, 5, 6, 9])
def test_dcp_replicates_prefills_too_short_to_split(query_len):
    """A prefill that cannot fill 2*pcp chunks is replicated, not split."""
    pcp_world_size = 2
    num_scheduled_tokens = np.array([query_len], dtype=np.int32)
    is_prefilling = np.ones(1, dtype=np.bool_)

    rows_per_rank = []
    for rank in range(pcp_world_size):
        manager = PCPManager(
            pcp_world_size=pcp_world_size,
            pcp_rank=rank,
            device=torch.device("cpu"),
            shard_decode_requests=False,
            dcp_world_size=2,
        )
        assert manager.replicated_requests(num_scheduled_tokens, is_prefilling)[0]
        rows = list(
            manager._iter_rank_chunks(rank, num_scheduled_tokens, is_prefilling)
        )
        assert rows == [(0, 0, query_len)]
        rows_per_rank.append(rows)
    assert rows_per_rank[0] == rows_per_rank[1]


@pytest.mark.parametrize("pcp_world_size", [2, 4, 8])
@pytest.mark.parametrize(
    "query_len", [16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 4097]
)
def test_pcp_first_chunk_row_is_never_short(pcp_world_size, query_len):
    num_scheduled_tokens = np.array([query_len], dtype=np.int32)
    is_prefilling = np.ones(1, dtype=np.bool_)

    charged_per_rank = []
    for rank in range(pcp_world_size):
        manager = PCPManager(
            pcp_world_size=pcp_world_size,
            pcp_rank=rank,
            device=torch.device("cpu"),
            shard_decode_requests=False,
            dcp_world_size=pcp_world_size,
        )
        chunk_lens = [
            chunk_len
            for _, _, chunk_len in manager._iter_rank_chunks(
                rank, num_scheduled_tokens, is_prefilling
            )
        ]
        assert chunk_lens, f"rank {rank} got no rows for {query_len=}"
        assert chunk_lens[0] == max(chunk_lens), (
            f"rank {rank} emitted a short first chunk for {query_len=}: {chunk_lens}"
        )
        charged_per_rank.append(len(chunk_lens) * chunk_lens[0])

    assert len(set(charged_per_rank)) == 1, (
        f"ranks would chunk differently for {query_len=}: {charged_per_rank}"
    )


def _make_global_decode_batch(
    num_computed_tokens: list[int], buffers: InputBuffers, device: torch.device
) -> InputBatch:
    """A replicate-decode global batch as `prepare_inputs` would build it."""
    num_reqs = len(num_computed_tokens)
    num_tokens = num_reqs
    seq_lens_np = np.asarray(num_computed_tokens, dtype=np.int32) + 1

    base = InputBatch.make_dummy(num_reqs, num_tokens, buffers)
    buffers.seq_lens[:num_reqs] = torch.from_numpy(seq_lens_np).to(device)
    buffers.positions[:num_reqs] = torch.tensor(num_computed_tokens, device=device)
    query_start_loc_np = np.arange(num_reqs + 1, dtype=np.int32)
    buffers.query_start_loc[: num_reqs + 1] = torch.from_numpy(query_start_loc_np).to(
        device
    )

    return replace(
        base,
        req_ids=[f"req_{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=0,
        num_draft_tokens_per_req=np.zeros(num_reqs, dtype=np.int32),
        query_start_loc=buffers.query_start_loc[: num_reqs + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=buffers.seq_lens[:num_reqs],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.asarray(num_computed_tokens, dtype=np.int32),
        prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        input_ids=buffers.input_ids[:num_tokens],
        positions=buffers.positions[:num_tokens],
        is_padding=buffers.is_padding[:num_tokens],
        prompt_lens=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU kernels")
def test_partition_defers_dcp_metadata_to_post_partition_batch():
    """DCP-local lengths must derive from the partitioned batch, not the
    global one: the partition replaces seq_lens, so a pre-partition value is
    stale. partition_batch therefore returns None, and the runtime populates
    the field afterwards from the PCP-owned buffers.
    """
    device = torch.device("cuda:0")
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=device,
        shard_decode_requests=False,
        max_num_reqs=4,
        max_num_tokens=8,
        dcp_world_size=2,
        dcp_rank=0,
    )

    global_buffers = InputBuffers(4, 8, device)
    global_batch = _make_global_decode_batch([16, 24], global_buffers, device)
    # A leftover from an earlier DCP batch must not survive the partition.
    global_batch.dcp_local_seq_lens = global_buffers.dcp_local_seq_lens[:2]
    global_batch.dcp_local_seq_lens.fill_(-1)

    local_batch = manager.partition_batch(
        global_batch,
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=4,
            num_reqs=4,
        ),
    )

    assert local_batch.dcp_local_seq_lens is None
    assert local_batch.seq_lens.tolist() == [17, 25, 0, 0]
    assert local_batch.pcp is not None
    assert local_batch.pcp.dcp_local_seq_lens_cpu_upper_bound is not None
    expected_dcp_upper_bound = get_dcp_local_seq_lens(
        torch.tensor([17, 25], dtype=torch.int32), 2, 0, 1
    )
    assert torch.equal(
        local_batch.pcp.dcp_local_seq_lens_cpu_upper_bound,
        torch.cat([expected_dcp_upper_bound, torch.zeros(2, dtype=torch.int32)]),
    )

    # What execute_model does next: derive DCP metadata from the final batch
    # on the PCP-owned buffers.
    local_batch.dcp_local_seq_lens = gpu_cp_utils.prepare_dcp_local_seq_lens(
        manager.input_buffers.dcp_local_seq_lens,
        local_batch.seq_lens,
        local_batch.num_reqs,
        dcp_size=2,
        dcp_rank=0,
        cp_interleave=1,
        num_reqs_padded=local_batch.num_reqs_after_padding,
    )
    expected = get_dcp_local_seq_lens(
        torch.tensor([17, 25, 0, 0], dtype=torch.int32), 2, 0, 1
    )
    assert local_batch.dcp_local_seq_lens is not None
    assert torch.equal(local_batch.dcp_local_seq_lens.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU kernels")
def test_moe_padding_matches_pcp_all_gather(monkeypatch):
    """The PCP batch carries the all-gather of every rank's local padding mask,
    the rows the MoE routes, even when ranks own different numbers of tokens."""
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    device = torch.device("cuda:0")
    local_masks, batch_masks = [], []
    for pcp_rank in range(2):
        manager = PCPManager(
            pcp_world_size=2,
            pcp_rank=pcp_rank,
            device=device,
            shard_decode_requests=True,
            max_num_reqs=4,
            max_num_tokens=8,
            moe_gathers_tokens=True,
        )
        global_batch = _make_global_decode_batch(
            [16, 24, 32], InputBuffers(4, 8, device), device
        )
        local_batch = manager.partition_batch(
            global_batch,
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=4, num_reqs=None
            ),
        )
        num_tokens = local_batch.num_tokens_after_padding
        local_masks.append(manager.input_buffers.is_padding[:num_tokens].cpu())
        batch_masks.append(local_batch.is_padding.clone())

    # Sharded decode puts two requests on rank 0 and one on rank 1.
    assert [mask.logical_not().sum().item() for mask in local_masks] == [2, 1]
    for batch_mask in batch_masks:
        assert torch.equal(batch_mask.cpu(), torch.cat(local_masks))
        num_tokens = batch_mask.numel()
        hidden_states = torch.randn(num_tokens, 4, device=device)
        router_logits = torch.randn(num_tokens, 8, device=device)
        with set_forward_context(None, VllmConfig(), is_padding=batch_mask):
            _, topk_ids, _ = fused_topk(hidden_states, router_logits, 2, False)
        assert (topk_ids[~batch_mask] >= 0).all()
        assert (topk_ids[batch_mask] == -1).all()


@pytest.mark.parametrize(
    ("pcp_world_size", "dcp_world_size", "expected"),
    [(1, 1, False), (2, 1, True), (2, 2, False)],
)
def test_parallel_config_manages_decode_sharding(
    pcp_world_size: int, dcp_world_size: int, expected: bool
):
    parallel_config = ParallelConfig(
        prefill_context_parallel_size=pcp_world_size,
        decode_context_parallel_size=dcp_world_size,
        distributed_executor_backend="ray",
    )

    assert parallel_config.pcp_shard_decode_requests is expected
