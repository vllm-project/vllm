# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SuffixProposerGPU (requires CUDA + suffix_gpu)."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("suffix_gpu")

import vllm.v1.spec_decode.suffix_proposer_gpu as suffix_proposer_module
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode
from vllm.v1.spec_decode.suffix_proposer_gpu import (
    SuffixProposerGPU,
    _SuffixCudagraphDispatcher,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEVICE = torch.device("cuda:0")
K = 8
MAX_NUM_SEQS = 8
MAX_MODEL_LEN = 256


def _make_config(use_cuda_graph: bool, tp_size: int = 1) -> SimpleNamespace:
    spec = SimpleNamespace(
        num_speculative_tokens=K,
        suffix_decoding_max_tree_depth=24,
        suffix_decoding_max_cached_requests=1000,
        suffix_decoding_max_spec_factor=2.0,
        suffix_decoding_min_token_prob=0.1,
        suffix_gpu_global_capacity=1 << 16,
        suffix_gpu_delta_capacity=1 << 12,
        suffix_gpu_max_occurrences=32,
        suffix_gpu_num_backoff=4,
        suffix_gpu_use_cuda_graph=use_cuda_graph,
        suffix_gpu_ingest_chunk=16,
        enforce_eager=False,
    )
    return SimpleNamespace(
        speculative_config=spec,
        model_config=SimpleNamespace(
            max_model_len=MAX_MODEL_LEN, enforce_eager=False
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_SEQS),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            data_parallel_size=1,
            use_sequence_parallel_moe=False,
            is_moe_model=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            fast_moe_cold_start=False,
            static_forward_context={},
        ),
    )


class _ReadyEvent:
    def query(self) -> bool:
        return True


def _seed_pending_rebuild(proposer: SuffixProposerGPU) -> None:
    index = proposer.drafter.global_index
    assert index is not None
    docs = deque()
    index._rebuild_event = _ReadyEvent()
    index._pending = (0, docs)
    index.pending_epoch = 1
    index._pending_signature = index._snapshot_signature(0, docs)


def _capture_graphs(
    proposer: SuffixProposerGPU, token_ids: torch.Tensor
) -> None:
    capture_stream = torch.cuda.Stream(device=DEVICE)
    capture_stream.wait_stream(torch.cuda.current_stream(DEVICE))
    set_cudagraph_capturing_enabled(True)
    try:
        with torch.cuda.stream(capture_stream):
            proposer.dummy_run(
                MAX_NUM_SEQS,
                token_ids,
                use_cudagraphs=True,
                is_graph_capturing=True,
            )
    finally:
        set_cudagraph_capturing_enabled(False)
    torch.cuda.current_stream(DEVICE).wait_stream(capture_stream)


def _propose_repetition(
    proposer: SuffixProposerGPU, token_ids: torch.Tensor | None = None
):
    """History ...[5,6,7,8]*3 then sampled [5]: expect draft [6,7,8,...]."""
    if token_ids is None:
        token_ids = torch.zeros(
            MAX_NUM_SEQS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
        )
    hist = [5, 6, 7, 8] * 3
    token_ids[0, : len(hist)] = torch.tensor(hist, dtype=torch.int32)
    num_tokens = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=DEVICE)
    num_tokens[0] = len(hist)
    sampled = torch.full((MAX_NUM_SEQS, K + 1), -1, dtype=torch.int32, device=DEVICE)
    sampled[0, 0] = 5
    counts = (sampled != -1).sum(dim=1).to(torch.int32)
    draft, nv = proposer.propose(K, num_tokens, token_ids, sampled, counts)
    torch.accelerator.synchronize()
    return draft, nv, token_ids


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_propose_drafts_repetition(use_cuda_graph):
    proposer = SuffixProposerGPU(_make_config(use_cuda_graph), DEVICE)
    token_ids = torch.zeros(
        MAX_NUM_SEQS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    if use_cuda_graph:
        _capture_graphs(proposer, token_ids)
    draft, nv, token_ids = _propose_repetition(proposer, token_ids)
    n = int(nv[0])
    assert n > 0
    assert draft[0, :n].tolist() == ([6, 7, 8, 5] * 3)[:n]
    # Sampled id was scattered into the resident buffer.
    assert int(token_ids[0, 12]) == 5
    # Rows with no sampled tokens must not draft.
    assert int(nv[1]) == 0
    if use_cuda_graph:
        assert proposer._graphs_captured()
        assert proposer._graph_dispatcher.buckets == (1, 2, 4, 8)


def test_graph_dispatcher_covers_max_num_seqs():
    dispatcher = _SuffixCudagraphDispatcher(6)
    assert dispatcher.buckets == (1, 2, 4, 6)
    assert dispatcher.dispatch(3).num_tokens == 4
    assert dispatcher.dispatch(5).num_tokens == 6
    assert dispatcher.dispatch(6).num_tokens == 6
    assert dispatcher.dispatch(7) is None


def test_graph_and_eager_agree():
    cfg_e = _make_config(False)
    cfg_g = _make_config(True)
    d_e, nv_e, _ = _propose_repetition(SuffixProposerGPU(cfg_e, DEVICE))
    graph_proposer = SuffixProposerGPU(cfg_g, DEVICE)
    persistent = torch.zeros(
        MAX_NUM_SEQS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    _capture_graphs(graph_proposer, persistent)
    d_g, nv_g, _ = _propose_repetition(graph_proposer, persistent)
    assert torch.equal(nv_e, nv_g)
    assert torch.equal(d_e, d_g)


def test_graph_buckets_use_wrapper_pool():
    proposer = SuffixProposerGPU(_make_config(True), DEVICE)
    persistent = torch.zeros(
        MAX_NUM_SEQS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    _capture_graphs(proposer, persistent)

    assert proposer._graph_runner is not None
    entries = proposer._graph_runner.concrete_cudagraph_entries
    graphs = [entries[desc].cudagraph for desc in entries]
    assert all(graph is not None for graph in graphs)
    assert {graph.pool() for graph in graphs if graph is not None} == {
        proposer._graph_runner.graph_pool
    }


def test_shared_pool_graphs_replay_alternating_buckets():
    graph_proposer = SuffixProposerGPU(_make_config(True), DEVICE)
    eager_proposer = SuffixProposerGPU(_make_config(False), DEVICE)
    graph_tokens = torch.zeros(
        MAX_NUM_SEQS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    eager_tokens = torch.zeros_like(graph_tokens)
    _capture_graphs(graph_proposer, graph_tokens)
    history = torch.tensor([5, 6, 7, 8] * 3, dtype=torch.int32, device=DEVICE)

    def run(proposer, token_buffer, batch_size):
        token_buffer.zero_()
        token_buffer[:batch_size, : history.numel()] = history
        num_tokens = torch.full(
            (batch_size,), history.numel(), dtype=torch.int32, device=DEVICE
        )
        sampled = torch.full((batch_size, K + 1), -1, dtype=torch.int32, device=DEVICE)
        sampled[:, 0] = 5
        counts = torch.ones(batch_size, dtype=torch.int32, device=DEVICE)
        draft, nv = proposer.propose(
            K, num_tokens, token_buffer[:batch_size], sampled, counts
        )
        torch.accelerator.synchronize()
        return draft.clone(), nv.clone()

    for batch_size in (1, 8, 2, 4, 1):
        graph_draft, graph_nv = run(graph_proposer, graph_tokens, batch_size)
        eager_draft, eager_nv = run(eager_proposer, eager_tokens, batch_size)
        assert torch.equal(graph_nv, eager_nv)
        assert torch.equal(graph_draft, eager_draft)


def test_dummy_run_captures_and_recaptures_graphs():
    proposer = SuffixProposerGPU(_make_config(True), DEVICE)
    persistent = torch.zeros(
        MAX_NUM_SEQS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    _capture_graphs(proposer, persistent)
    assert proposer._graphs_captured()
    assert int(persistent.abs().sum()) == 0

    assert proposer._graph_runner is not None
    graphs = {
        desc: entry.cudagraph
        for desc, entry in proposer._graph_runner.concrete_cudagraph_entries.items()
    }
    d_g, nv_g, _ = _propose_repetition(proposer, persistent)
    assert {
        desc: entry.cudagraph
        for desc, entry in proposer._graph_runner.concrete_cudagraph_entries.items()
    } == graphs
    assert int(persistent[0, 12]) == 5  # graph scattered the sampled id

    d_e, nv_e, _ = _propose_repetition(SuffixProposerGPU(_make_config(False), DEVICE))
    assert torch.equal(nv_e, nv_g)
    assert torch.equal(d_e, d_g)

    # A buffer with different storage must fall back to eager, safely.
    d_f, nv_f, _ = _propose_repetition(proposer)
    assert torch.equal(nv_e, nv_f)
    assert torch.equal(d_e, d_f)

    proposer._graph_runner.clear_graphs()
    assert not proposer._graphs_captured()
    _capture_graphs(proposer, persistent)
    assert proposer._graphs_captured()


def test_dummy_run_warms_up_without_graph():
    """With the graph disabled, warmup (Triton JIT) still runs at startup."""
    proposer = SuffixProposerGPU(_make_config(False), DEVICE)
    persistent = torch.zeros(
        MAX_NUM_SEQS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    proposer.dummy_run(MAX_NUM_SEQS, persistent, is_graph_capturing=True)
    assert proposer._warmed_up
    assert proposer._graph_runner is None
    d, nv, _ = _propose_repetition(proposer, persistent)
    n = int(nv[0])
    assert n > 0
    assert d[0, :n].tolist() == ([6, 7, 8, 5] * 3)[:n]


def test_ingest_and_cross_request_draft():
    proposer = SuffixProposerGPU(_make_config(False), DEVICE)
    token_ids = torch.zeros(
        MAX_NUM_SEQS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    phrase = list(range(100, 108))
    resp = phrase * 4
    token_ids[0, : len(resp)] = torch.tensor(resp, dtype=torch.int32)

    input_batch = SimpleNamespace(
        req_id_to_index={"req-a": 0},
        num_tokens_no_spec=[len(resp)],
        num_prompt_tokens=[0],
    )
    proposer.ingest_active_requests(input_batch, token_ids)
    proposer.on_requests_finished(["req-a"], input_batch, token_ids)
    assert "req-a" not in proposer.drafter._ingested

    # A different request whose tail matches the shared phrase.
    num_tokens = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32, device=DEVICE)
    cur = [7, 9] + phrase[:5]
    token_ids[1, : len(cur)] = torch.tensor(cur, dtype=torch.int32)
    num_tokens[1] = len(cur)
    sampled = torch.full((MAX_NUM_SEQS, K + 1), -1, dtype=torch.int32, device=DEVICE)
    sampled[1, 0] = phrase[5]
    counts = (sampled != -1).sum(dim=1).to(torch.int32)
    draft, nv = proposer.propose(K, num_tokens, token_ids, sampled, counts)
    torch.accelerator.synchronize()
    n = int(nv[1])
    assert n > 0
    expect = (phrase[6:] + phrase * 2)[:n]
    assert draft[1, :n].tolist() == expect


def test_tp_poll_waits_for_all_ranks(monkeypatch):
    proposer = SuffixProposerGPU(_make_config(False, tp_size=2), DEVICE)
    _seed_pending_rebuild(proposer)
    cpu_group = object()
    monkeypatch.setattr(
        suffix_proposer_module,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=cpu_group),
    )

    all_ready = False

    def fake_all_gather(output, input_, group):
        assert group is cpu_group
        rank0 = input_.clone()
        rank1 = input_.clone()
        rank0[2] = int(all_ready)
        rank1[2] = 1
        output.copy_(torch.cat((rank0, rank1)))

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather)
    proposer._poll_rebuild()
    assert proposer.drafter.global_index.active_epoch == 0

    all_ready = True
    proposer._poll_rebuild()
    assert proposer.drafter.global_index.active_epoch == 1


def test_tp_poll_rejects_snapshot_mismatch(monkeypatch):
    proposer = SuffixProposerGPU(_make_config(False, tp_size=2), DEVICE)
    _seed_pending_rebuild(proposer)
    monkeypatch.setattr(
        suffix_proposer_module,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group=object()),
    )

    def fake_all_gather(output, input_, group):
        rank0 = input_.clone()
        rank1 = input_.clone()
        rank1[5] += 1
        output.copy_(torch.cat((rank0, rank1)))

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather)
    with pytest.raises(RuntimeError, match="TP rebuild state mismatch"):
        proposer._poll_rebuild()


def test_tp_ingestion_uses_request_id_order(monkeypatch):
    proposer = SuffixProposerGPU(_make_config(False, tp_size=2), DEVICE)
    token_ids = torch.zeros(
        MAX_NUM_SEQS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
    )
    input_batch = SimpleNamespace(
        req_id_to_index={"req-b": 1, "req-a": 0},
        num_tokens_no_spec=[16, 16],
        num_prompt_tokens=[0, 0],
    )
    calls = []
    monkeypatch.setattr(
        proposer,
        "_ingest_async",
        lambda keys, rows, lengths, final=False: calls.append((keys, final)),
    )

    proposer.ingest_active_requests(input_batch, token_ids)
    proposer.on_requests_finished(["req-b", "req-a"], input_batch, token_ids)

    assert calls == [(["req-a", "req-b"], False), (["req-a", "req-b"], True)]
