# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the resident PCP + DCP indexer K replica decisions and slot
kernel."""

import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention import kv_ce_triton_launch as launch
from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)
from vllm.v1.attention.ops import cp_common, pcp_dcp_k_replica
from vllm.v1.attention.ops.pcp_dcp_k_replica import (
    K_CAPACITY_SLACK_ROWS,
    KReplica,
    LayerReplica,
    k_replica_chunk,
    k_replica_for_layer,
    pop_indexer_k_replica,
    replica_bytes,
    replica_mode,
)

# GLM-5.2 at 1M context on PCP4 x DCP4, chunk 16384: each rank holds two eighths
# of the chunk (4096 rows) as two local requests.
LOCAL_ROWS = 4096
NEW_ROWS = 16_384


def _chunk(**overrides) -> DeepseekV32IndexerPrefillChunkMetadata:
    fields = dict(
        block_table=torch.zeros(2, 1, dtype=torch.int32),
        cu_seqlen_ks=torch.zeros(LOCAL_ROWS, dtype=torch.int32),
        cu_seqlen_ke=torch.zeros(LOCAL_ROWS, dtype=torch.int32),
        cu_seq_lens=torch.zeros(3, dtype=torch.int32),
        token_to_seq=torch.zeros(1, dtype=torch.int32),
        total_seq_lens=1_024_000,
        max_seqlen_k=1_024_000,
        token_start=0,
        token_end=LOCAL_ROWS,
        num_reqs=2,
        local_cu_seq_lens=torch.zeros(3, dtype=torch.int32),
        local_total_seq_lens=256_000,
        max_local_total_seq_lens=256_000,
        pcp_deinterleave_idx=torch.zeros(1, dtype=torch.int64),
    )
    return DeepseekV32IndexerPrefillChunkMetadata(**{**fields, **overrides})


def _prefill(**overrides) -> DeepseekV32IndexerPrefillMetadata:
    fields = dict(chunks=[_chunk()], max_prefill_seq_len=1_024_000)
    return DeepseekV32IndexerPrefillMetadata(**{**fields, **overrides})


def _eligibility(prefill, **overrides):
    args = dict(
        num_decode_tokens=0,
        local_rows=LOCAL_ROWS,
        pcp_world_size=4,
        dcp_world_size=4,
        capacity_rows=1_052_672,
        global_new_rows=NEW_ROWS,
        global_end=1_024_000,
    )
    return k_replica_chunk(prefill, **{**args, **overrides})


def test_off_by_default():
    assert not ParallelConfig().pcp_dcp_indexer_k_replica
    assert pop_indexer_k_replica() is None
    assert k_replica_for_layer(False, 1, object(), True, 4, 4, 1 << 20) is None


def test_needs_pcp_and_dcp():
    assert k_replica_for_layer(True, 1, object(), False, 1, 4, 1 << 20) is None
    assert k_replica_for_layer(True, 1, object(), True, 4, 1, 1 << 20) is None


@pytest.mark.parametrize(("pcp_size", "dcp_size"), [(1, 1), (1, 2), (2, 1)])
def test_option_requires_pcp_and_dcp(pcp_size, dcp_size):
    with pytest.raises(ValueError, match="pcp_dcp_indexer_k_replica requires"):
        ParallelConfig(
            tensor_parallel_size=2,
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            pcp_dcp_indexer_k_replica=True,
            distributed_executor_backend="mp",
        )


def test_single_request_chunk_is_eligible():
    prefill = _prefill()
    chunk, reason = _eligibility(prefill)
    assert chunk is prefill.chunks[0] and reason == "eligible"


@pytest.mark.parametrize(
    ("prefill", "overrides", "reason"),
    [
        (None, {}, "no_prefill"),
        (_prefill(chunks=[]), {}, "no_prefill"),
        (_prefill(), {"num_decode_tokens": 1}, "decode_tokens"),
        (_prefill(chunks=[_chunk(), _chunk()]), {}, "many_chunks"),
        (_prefill(chunks=[_chunk(pcp_deinterleave_idx=None)]), {}, "no_gather"),
        # A query sub-chunk does not read its shard.
        (_prefill(chunks=[_chunk(skip_kv_gather=True)]), {}, "no_gather"),
        (_prefill(), {"global_end": None}, "many_requests"),
        # The dedicated buffers cannot hold the context: serial path.
        (_prefill(), {"capacity_rows": 1_000_000}, "over_capacity"),
        (_prefill(), {"global_new_rows": None}, "uneven_partition"),
        (_prefill(), {"global_new_rows": NEW_ROWS + 4}, "uneven_partition"),
        # A request too short to split is replicated on every rank.
        (_prefill(), {"global_new_rows": 2 * LOCAL_ROWS}, "replicated_rows"),
        (_prefill(chunks=[_chunk(token_end=LOCAL_ROWS // 2)]), {}, "partial_query"),
        (_prefill(), {"global_end": 8192}, "short_context"),
        (_prefill(chunks=[_chunk(total_seq_lens=1_000_000)]), {}, "short_context"),
        # Compressed K rows are not one per token.
        (_prefill(max_prefill_seq_len=4_096_000), {}, "compressed_k"),
    ],
)
def test_ineligible_chunks_use_the_serial_path(prefill, overrides, reason):
    assert _eligibility(prefill, **overrides) == (None, reason)


TOKEN_SLOTS_SCRIPT = """
import torch
from vllm.v1.attention.ops import pcp_dcp_k_replica as kr

g = torch.Generator().manual_seed(0)
for rows, extra, start in ((1500, 36, 1_020_000), (1, 0, 0), (1024, 8, 2**33)):
    restore = torch.randperm(rows + extra, generator=g)[:rows]
    slots = torch.full((rows + extra,), -1, dtype=torch.int64)
    token_slots(restore, slots, start)
    positions = torch.arange(start, start + rows, dtype=torch.int64)
    expected = torch.full_like(slots, -1).index_copy_(0, restore, positions)
    assert torch.equal(slots, expected), rows
print("ok")
"""


def test_token_slots_kernel_matches_the_index_copy():
    """Run under the Triton interpreter: no GPU needed."""
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", inspect.getsource(launch) + TOKEN_SLOTS_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


def test_replica_memory_per_rank():
    rows = 1_048_576 + K_CAPACITY_SLACK_ROWS
    assert replica_bytes(rows, 128, 1) == 138_952_704
    assert replica_bytes(rows, 128, 21) == 2_918_006_784


@pytest.mark.parametrize(
    "req_id, start, held_req_id, held_rows, mode",
    [
        ("a", 0, None, -1, "append"),
        ("a", 0, "b", 4096, "append"),
        ("a", 4096, "a", 4096, "append"),
        ("a", 4096, None, -1, "build"),
        ("a", 4096, "b", 4096, "build"),
        ("a", 4096, "a", 8192, "build"),
        ("a", 8192, "a", 4096, "build"),
    ],
)
def test_replica_appends_only_the_next_rows_of_the_request_it_holds(
    req_id, start, held_req_id, held_rows, mode
):
    assert replica_mode(req_id, start, held_req_id, held_rows) == mode


def test_replica_values_and_scales_are_the_rows_of_a_one_block_cache():
    """The cache-write kernel puts row `i`'s values at `[i * D, (i + 1) * D)` of a
    block and its scale at `capacity * D + 4 * i`: the replica's views must read
    exactly those bytes."""
    capacity, head_dim = 16, 8
    buffer = torch.arange(capacity * (head_dim + 4), dtype=torch.int64).to(torch.uint8)
    replica = LayerReplica(buffer, capacity, head_dim)
    assert replica.cache.shape == (1, capacity, head_dim + 4)
    assert replica.values.shape == (capacity, head_dim)
    assert replica.scales.shape == (capacity, 4)
    assert replica.values.is_contiguous() and replica.scales.is_contiguous()
    values = buffer[: capacity * head_dim].view(capacity, head_dim)
    assert torch.equal(replica.values.view(torch.uint8), values)
    assert torch.equal(replica.scales, buffer[capacity * head_dim :].view(capacity, 4))
    replica.req_id, replica.rows = "a", 8
    replica.invalidate()
    assert replica.req_id is None and replica.rows == -1


@pytest.mark.parametrize("max_model_len", [1_048_576, 1_024_000, 163_840, 65_537])
def test_replica_k_starts_where_the_top_k_kernels_can_read_it(max_model_len):
    """The top-k kernels take the K values and the scales as contiguous rows at
    16-byte multiples: both regions start at one whatever the capacity."""
    capacity, head_dim = max_model_len + K_CAPACITY_SLACK_ROWS, 128
    buffer = torch.empty(0, dtype=torch.uint8)
    replica = LayerReplica.__new__(LayerReplica)
    replica.buffer, replica.capacity, replica.head_dim = buffer, capacity, head_dim
    values_bytes = capacity * head_dim
    assert values_bytes % 16 == 0
    assert replica_bytes(capacity, head_dim, 1) == values_bytes + 4 * capacity


def _scheduled_manager(monkeypatch, scheduled: bool = True, **fields):
    """The process's PCP manager, as the replica sees it."""
    from vllm.v1.worker.gpu import pcp_manager

    manager = SimpleNamespace(scheduled_batch=scheduled, **fields)
    monkeypatch.setattr(pcp_manager, "_ACTIVE_PCP_MANAGER", manager)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return manager


def test_a_step_is_scheduled_only_outside_captures_and_dummy_batches(monkeypatch):
    """The replica acts on batches `partition_batch` laid out. A warmup or
    profile batch bypasses it, and a capture runs its eager regions for real:
    none of them may reach the replicas."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.v1.worker.gpu import pcp_manager

    monkeypatch.setattr(pcp_manager, "_ACTIVE_PCP_MANAGER", None)
    assert not cp_common.scheduled_step()
    manager = _scheduled_manager(monkeypatch)
    assert cp_common.scheduled_step() and not cp_common.capturing_run()
    manager.scheduled_batch = False
    assert not cp_common.scheduled_step()
    manager.scheduled_batch = True
    # The eager region of a breakable capture: the stream is not capturing.
    monkeypatch.setattr(BreakableCUDAGraphCapture._tls, "active", object(), False)
    assert cp_common.capturing_run() and not cp_common.scheduled_step()
    monkeypatch.setattr(BreakableCUDAGraphCapture._tls, "active", None, False)
    assert cp_common.scheduled_step()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert cp_common.capturing_run() and not cp_common.scheduled_step()


def _replica_step(monkeypatch, scheduled: bool):
    """A `KReplica` with one layer at the step that prefills rows
    `[1_007_616, 1_024_000)` of request `r`, under a piecewise CUDA graph."""
    restore_idx = torch.zeros(NEW_ROWS, dtype=torch.int64)
    _scheduled_manager(
        monkeypatch,
        scheduled,
        hidden_restore_idx=restore_idx if scheduled else None,
        single_request_end=1_024_000 if scheduled else None,
        global_batch=SimpleNamespace(req_ids=["r"]),
    )
    prefill = _prefill()
    context = SimpleNamespace(
        attn_metadata={"k": SimpleNamespace(prefill=prefill, num_decode_tokens=0)},
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
    )
    monkeypatch.setattr(pcp_dcp_k_replica, "get_forward_context", lambda: context)
    replica = KReplica(4, 4, 1_052_672)
    replica._indexers[3] = SimpleNamespace(k_cache=SimpleNamespace(prefix="k"))
    layer = replica.layers[3] = LayerReplica(torch.zeros(48, dtype=torch.uint8), 4, 8)
    layer.req_id, layer.rows = "r", 1_007_616
    return replica, layer, prefill.chunks[0]


def test_the_replica_serves_a_scheduled_step_under_a_piecewise_graph(monkeypatch):
    """Its entry points run in the eager regions of a replayed graph: the mode
    of the forward does not matter, only that the batch was scheduled."""
    replica, layer, chunk = _replica_step(monkeypatch, scheduled=True)
    decision = replica.decide(3, LOCAL_ROWS)
    assert decision is not None and decision.chunk is chunk
    assert (decision.mode, decision.start, decision.end) == (
        "append",
        1_007_616,
        1_024_000,
    )
    # Decided once per step.
    assert replica.decide(3, LOCAL_ROWS) is decision


def test_a_replica_without_the_earlier_rows_is_rebuilt_from_the_cache(monkeypatch):
    """A chunk that does not continue the replica's rows (a prefix cache hit, a
    resumed request, a step the replica missed) rebuilds the replica from the
    sharded cache once; the next chunk appends to it again."""
    replica, layer, chunk = _replica_step(monkeypatch, scheduled=True)
    layer.rows = 1_003_520
    decision = replica.decide(3, LOCAL_ROWS)
    assert decision is not None and decision.mode == "build"
    assert layer.req_id is None and layer.rows == -1
    built: list[int] = []
    monkeypatch.setattr(
        replica, "_build", lambda layer_id, layer_replica, c: built.append(layer_id)
    )
    gathered_k = torch.zeros(NEW_ROWS, 8)
    assert replica.take(3, chunk, gathered_k, 128, "ue8m0") is not None
    assert built == [3]
    assert (layer.req_id, layer.rows) == ("r", 1_024_000)


def test_a_dummy_batch_invalidates_the_replica(monkeypatch):
    """A capture or warmup forward after a real step would otherwise append the
    dummy batch's rows to what the replica holds of the last request."""
    replica, layer, _ = _replica_step(monkeypatch, scheduled=False)
    assert replica.decide(3, LOCAL_ROWS) is None
    assert layer.req_id is None and layer.rows == -1


@pytest.mark.parametrize("pcp", [2, 4, 8])
@pytest.mark.parametrize("fused_mqa_topk_eligible", [False, True])
def test_every_rank_splits_a_prefill_into_the_same_indexer_chunks(
    pcp, fused_mqa_topk_eligible
):
    """No rank asks another whether a layer's replica serves a step: the
    decision rests on the indexer chunks, among other things, and PCP gives the
    ranks different rows. Every rank gets the same chunks for random split
    prefills laid out by the real PCP partition."""
    import numpy as np

    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    rng = np.random.default_rng(7 * pcp + int(fused_mqa_topk_eligible))
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.dcp_world_size = pcp
    builder.fused_mqa_topk_eligible = fused_mqa_topk_eligible
    builder.max_prefill_buffer_size = 40 * 2**20
    several = 0
    for _ in range(200):
        scheduled = np.array([int(rng.integers(2 * pcp, 40_000))], dtype=np.int32)
        computed = np.array([int(rng.integers(0, 1_040_000))], dtype=np.int32)
        prefilling = np.ones(1, dtype=np.bool_)
        budget = int(rng.choice([64, 512, 2048])) * 2**20
        starts = np.concatenate([[0], np.cumsum(scheduled)]).astype(np.int32)
        plans = []
        for rank in range(pcp):
            manager = PCPManager(
                pcp_world_size=pcp,
                pcp_rank=rank,
                device=torch.device("cpu"),
                shard_decode_requests=False,
                dcp_world_size=pcp,
            )
            if manager.replicated_requests(scheduled, prefilling)[0]:
                break
            segments = manager._get_rank_segments(rank, scheduled, prefilling, starts)
            rows = np.array([segment.global_batch_req_idx for segment in segments])
            lens = torch.tensor([segment.num_tokens for segment in segments])
            ends = torch.from_numpy((computed + scheduled)[rows])
            shard_rows = get_dcp_local_seq_lens(ends, pcp, 0, 1).numpy()
            plans.append(
                builder._split_pcp_dcp_prefill_chunks(
                    rows, shard_rows, lens, budget, request_offset=0
                )
            )
        if not plans:
            continue
        assert all(plan == plans[0] for plan in plans), (scheduled, computed, budget)
        several += len(plans[0]) > 1
    # Both outcomes occur: one chunk (the replica can serve) and several.
    assert several > 0
