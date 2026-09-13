# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for Uno's MRV2 inputs and adapter ownership.

The proposer must form the same seed/noise suffix after rejection and request
reordering without advancing request state. Unit tests directly exercise its
preparation and adapter scope; model/sampler/graph numerics require GPU tests.
"""

import inspect
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.launch_key_debug import _LaunchKeyReceipts, _object_fields
from vllm.v1.worker.gpu.spec_decode.uno import (
    UNO_LORA_ID,
    UnoSpeculator,
    prepare_uno_inputs_reference,
)
from vllm.v1.worker.gpu.spec_decode.uno_lora import draft_lora_mapping


@pytest.mark.parametrize("k", [1, 4, 8])
def test_noise_mapping_excludes_seed_and_padding(k):
    mapping = draft_lora_mapping(3 * k, 2, k, UNO_LORA_ID)
    assert mapping[:k] == (0,) + (UNO_LORA_ID,) * (k - 1)
    assert mapping[k : 2 * k] == mapping[:k]
    assert mapping[2 * k :] == (0,) * k


@pytest.mark.parametrize("args", [(-1, 4, 4), (1, 0, 4), (2, 4, 7)])
def test_query_capacity_rejected(args):
    with pytest.raises(ValueError):
        draft_lora_mapping(args[2], args[0], args[1], UNO_LORA_ID)


def test_rejection_and_prefill_use_correct_seed_and_persistent_slot():
    buffers = InputBuffers(4, 16, torch.device("cpu"))
    slots = torch.full((16,), 99, dtype=torch.int64)
    sample_slots = torch.full((16,), 99, dtype=torch.int32)
    batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([3, 1]),
        query_start_loc=torch.tensor([0, 3, 6]),
        positions=torch.tensor([2, 3, 4, 9, 10, 11]),
    )
    last_sampled = torch.tensor([[100], [101], [102], [103]])
    prefill_tokens = torch.tensor([[200, 201, 202, 203]])
    seeds = torch.tensor([10, 11, 12, 13])
    block_table = torch.tensor([[2, 3, 4, 5], [8, 9, 10, 11]])
    prepare_uno_inputs_reference(
        buffers,
        slots,
        sample_slots,
        batch,
        torch.tensor([1, 0]),
        torch.tensor([0, 2]),
        last_sampled,
        prefill_tokens,
        seeds,
        block_table,
        4,
        4,
        16,
        42,
        1000,
        1,
    )
    assert buffers.input_ids[[0, 4]].tolist() == [103, 201]
    assert buffers.positions[:8].tolist() == [5, 6, 7, 8, 10, 11, 12, 13]
    assert slots.tolist() == [13, 14, 15, 16, 42, 43, 44, 45] + [-1] * 8
    assert buffers.seq_lens.tolist() == [9, 14, 0, 0]
    assert buffers.query_start_loc.tolist() == [0, 4, 8, 8, 8]
    assert sample_slots.tolist() == [3] * 4 + [1] * 4 + [-1] * 8
    noise = buffers.input_ids[[1, 2, 3, 5, 6, 7]]
    assert ((noise >= 1) & (noise < 1000)).all()
    assert last_sampled.tolist() == [[100], [101], [102], [103]]
    assert batch.positions.tolist() == [2, 3, 4, 9, 10, 11]


@pytest.mark.parametrize(
    "block_ids,max_len,expected",
    [
        ([2, 3, 4], 10, [15, 16, 17, -1]),
        ([2, 3, 0], 12, [15, -1, -1, -1]),
        ([2, 3], 12, [15, -1, -1, -1]),
    ],
)
def test_draft_does_not_write_null_or_unallocated_or_out_of_context_slots(
    block_ids, max_len, expected
):
    buffers = InputBuffers(2, 8, torch.device("cpu"))
    slots = torch.empty(8, dtype=torch.int64)
    sample_slots = torch.empty(8, dtype=torch.int32)
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([1]),
        query_start_loc=torch.tensor([0, 1]),
        positions=torch.tensor([6]),
    )
    prepare_uno_inputs_reference(
        buffers,
        slots,
        sample_slots,
        batch,
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([[100], [101]]),
        torch.tensor([[200, 201]]),
        torch.tensor([10, 11]),
        torch.tensor([block_ids]),
        4,
        4,
        max_len,
        42,
        1000,
        1,
    )
    assert slots.tolist() == expected + [-1] * 4
    assert buffers.positions[:4].max() < max_len
    assert buffers.seq_lens[0] <= max_len


@pytest.mark.parametrize("fail_at", [None, "routing", "forward"])
def test_adapter_scope_restores_base_on_success_and_failure(fail_at):
    proposer = object.__new__(UnoSpeculator)
    proposer.k = 4
    history = []

    def hook(mapping):
        history.append(mapping)
        if mapping is not None and fail_at == "routing":
            raise RuntimeError("routing failed")

    proposer.set_lora_hook(hook)
    if fail_at:
        with pytest.raises(RuntimeError), proposer._draft_lora(1, 8):
            raise RuntimeError("forward failed")
    else:
        with proposer._draft_lora(1, 8):
            pass
    assert history == [(1, 8), None]


def test_native_sampling_handoff_uses_persistent_slots_and_columns(monkeypatch):
    proposer = object.__new__(UnoSpeculator)
    proposer.k = 2
    proposer.input_buffers = InputBuffers(2, 4, torch.device("cpu"))
    proposer.input_buffers.positions[:] = torch.tensor([4, 5, 8, 9])
    proposer.sample_idx_mapping = torch.tensor([3, 3, 1, 1])
    proposer.sample_col = torch.tensor([0, 1, 0, 1])
    proposer.temperature = torch.ones(4)
    proposer.seeds = torch.arange(4)
    proposer.draft_logits = torch.empty(4, 2, 7)
    proposer.draft_tokens = torch.zeros(2, 2, dtype=torch.int64)
    proposer.model = Mock(return_value=torch.randn(4, 8))
    proposer.sample_draft = Mock(return_value=torch.tensor([11, 12, 13, 14]))
    proposer.vllm_config = Mock()
    from contextlib import nullcontext

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.uno.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )
    proposer._generate_draft(2, 4, None, {})
    args = proposer.sample_draft.call_args.args
    assert args[1].tolist() == [4, 5, 8, 9]
    assert args[2].tolist() == [3, 3, 1, 1]
    assert args[5].tolist() == [0, 1, 0, 1]
    assert args[6] is proposer.draft_logits
    assert proposer.draft_tokens.tolist() == [[11, 12], [13, 14]]


@pytest.mark.parametrize("full_graph", [False, True])
def test_graph_replay_refreshes_native_backend_without_rebuilding_metadata(
    monkeypatch, full_graph
):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    module = "vllm.v1.worker.gpu.spec_decode.uno"
    proposer = object.__new__(UnoSpeculator)
    proposer.k = 2
    proposer._step = 0
    proposer.num_graph_replays = 0
    proposer.num_eager_proposals = 0
    proposer.max_model_len = 32
    proposer.speculative_config = SimpleNamespace(
        uno_mask_token_id=1000, uno_noise_seed=42
    )
    proposer.input_buffers = InputBuffers(4, 8, torch.device("cpu"))
    proposer.sample_idx_mapping = torch.empty(8, dtype=torch.int32)
    proposer.draft_tokens = torch.empty((4, 2), dtype=torch.int64)
    proposer.block_tables = SimpleNamespace(
        slot_mappings=torch.empty((1, 8), dtype=torch.int64),
        input_block_tables=[torch.ones((4, 8), dtype=torch.int32)],
        kernel_block_sizes=[4],
    )
    proposer.kv_cache_config = Mock()
    desc = BatchExecutionDescriptor(
        CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE, 8, 4
    )
    events = []
    captured_attn = {"layer": object()}
    proposer._graph_attn_metadata = {desc: captured_attn}
    group = Mock()
    group.update_draft_decode_metadata.side_effect = lambda metadata: events.append(
        ("refresh", metadata)
    )
    proposer.attn_groups = [[group]]
    proposer.cudagraph_manager = Mock()
    proposer.cudagraph_manager.dispatch.return_value = desc
    proposer.cudagraph_manager.run_fullgraph.side_effect = lambda _: events.append(
        ("replay", None)
    )
    proposer._copy_request_inputs = Mock()
    proposer._build_draft_attn_metadata = Mock(return_value={"eager": object()})
    proposer._generate_draft = Mock()
    proposer.set_lora_hook(lambda mapping: events.append(("lora", mapping)))
    fused_prepare = Mock()
    slot_builder = Mock(return_value={})
    monkeypatch.setattr(f"{module}.prepare_uno_inputs_fused", fused_prepare)
    monkeypatch.setattr(f"{module}.build_slot_mappings_by_layer", slot_builder)
    batch = SimpleNamespace(num_reqs=3, idx_mapping=torch.tensor([2, 0, 1]))
    # A full replay must not read or construct eager CPU length metadata.
    if not full_graph:
        batch.seq_lens_cpu_upper_bound = torch.tensor([10, 20, 30])
    tensor = torch.empty(4)
    proposer.propose(
        batch, {}, {}, tensor, None, tensor, tensor, tensor, tensor, tensor, tensor
    )
    fused_prepare.assert_called_once()
    assert events[0] == ("lora", (3, 8))
    assert events[-1] == ("lora", None)
    if full_graph:
        assert events[1:3] == [("refresh", captured_attn), ("replay", None)]
        proposer._build_draft_attn_metadata.assert_not_called()
        slot_builder.assert_not_called()
        proposer._generate_draft.assert_not_called()
        assert proposer.num_graph_replays == 1
    else:
        group.update_draft_decode_metadata.assert_not_called()
        proposer.cudagraph_manager.run_fullgraph.assert_not_called()
        proposer._build_draft_attn_metadata.assert_called_once()
        slot_builder.assert_called_once()
        proposer._generate_draft.assert_called_once()
        assert proposer.draft_max_seq_len == 32
        assert proposer.num_eager_proposals == 1


def test_eager_draft_attn_metadata_keeps_k_row_physical_capacity(monkeypatch):
    """The eager draft must size attention metadata to the K query rows.

    Upstream sizes autoregressive draft metadata one query per request
    (`num_tokens_padded = num_reqs`). Uno prepares K queries per request, so the
    eager path must pass its physical K-row capacity and never inherit that
    one-query value.
    """
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    module = "vllm.v1.worker.gpu.spec_decode.uno"
    k = 4
    n = 3
    count = n * k
    proposer = object.__new__(UnoSpeculator)
    proposer.k = k
    proposer._step = 0
    proposer.num_graph_replays = 0
    proposer.num_eager_proposals = 0
    proposer.max_model_len = 64
    proposer.speculative_config = SimpleNamespace(
        uno_mask_token_id=1000, uno_noise_seed=42
    )
    proposer.input_buffers = InputBuffers(4, 32, torch.device("cpu"))
    proposer.sample_idx_mapping = torch.empty(32, dtype=torch.int32)
    proposer.draft_tokens = torch.empty((4, k), dtype=torch.int64)
    proposer.block_tables = SimpleNamespace(
        slot_mappings=torch.empty((1, 32), dtype=torch.int64),
        input_block_tables=[torch.ones((4, 8), dtype=torch.int32)],
        kernel_block_sizes=[4],
    )
    proposer.kv_cache_config = Mock()
    desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, count, n)
    proposer.cudagraph_manager = Mock()
    proposer.cudagraph_manager.dispatch.return_value = desc
    proposer._copy_request_inputs = Mock()
    proposer._generate_draft = Mock()
    captured: dict = {}

    def fake_build(
        num_reqs,
        num_reqs_padded,
        num_tokens_padded,
        seq_lens_cpu_upper_bound,
        step,
        **kwargs,
    ):
        captured.update(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            step=step,
            **kwargs,
        )
        return {"eager": object()}

    proposer._build_draft_attn_metadata = fake_build
    proposer.set_lora_hook(lambda mapping: None)
    monkeypatch.setattr(f"{module}.prepare_uno_inputs_fused", Mock())
    monkeypatch.setattr(f"{module}.build_slot_mappings_by_layer", Mock(return_value={}))
    batch = SimpleNamespace(
        num_reqs=n,
        idx_mapping=torch.tensor([2, 0, 1]),
        seq_lens_cpu_upper_bound=torch.tensor([10, 20, 30]),
    )
    tensor = torch.empty(4)
    proposer.propose(
        batch, {}, {}, tensor, None, tensor, tensor, tensor, tensor, tensor, tensor
    )
    assert captured["num_tokens_padded"] == count
    assert captured["num_reqs_padded"] == n
    assert captured["step"] == k
    assert captured["num_query_per_req"] == k


def _cpu_uno_proposer(
    k: int,
    max_num_seqs: int = 4,
    capture_sizes: list[int] | None = None,
) -> UnoSpeculator:
    """A real UnoSpeculator wired to a CPU CudaGraphManager (no CUDA/build).

    ``max_num_seqs`` and ``capture_sizes`` are parameters because draft graph
    coverage is a property of all three together with K, not of K alone: one
    serving shape covers every request count and another leaves the top of the
    range drafting eagerly.
    """
    from vllm.config import (
        CompilationConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.v1.attention.backend import AttentionCGSupport

    sizes = sorted(capture_sizes or [8, 16, 32, 64])
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=sizes,
    )
    compilation_config.max_cudagraph_capture_size = sizes[-1]
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs
    )
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.speculative_config = None
    vllm_config.num_speculative_tokens = 0

    proposer = object.__new__(UnoSpeculator)
    proposer.k = k
    proposer.vllm_config = vllm_config
    proposer.device = torch.device("cpu")
    proposer.attn_cg_support = SimpleNamespace(
        min_cg_support=AttentionCGSupport.UNIFORM_BATCH
    )
    proposer._graph_attn_metadata = {}
    proposer._step = 0
    proposer.num_graph_replays = 0
    proposer.num_eager_proposals = 0
    proposer.num_warmup_proposals = 0
    proposer.max_num_reqs = max_num_seqs
    proposer.max_model_len = 64
    proposer.speculative_config = SimpleNamespace(
        uno_mask_token_id=1000, uno_noise_seed=42
    )
    rows = max(32, max_num_seqs * k)
    proposer.input_buffers = InputBuffers(max_num_seqs, rows, torch.device("cpu"))
    proposer.sample_idx_mapping = torch.empty(rows, dtype=torch.int32)
    proposer.draft_tokens = torch.empty((max_num_seqs, k), dtype=torch.int64)
    proposer.block_tables = SimpleNamespace(
        slot_mappings=torch.empty((1, rows), dtype=torch.int64),
        input_block_tables=[torch.ones((max_num_seqs, 8), dtype=torch.int32)],
        kernel_block_sizes=[4],
    )
    proposer.kv_cache_config = Mock()
    proposer._copy_request_inputs = Mock()
    proposer._build_draft_attn_metadata = Mock(return_value={"eager": object()})
    proposer._generate_draft = Mock()
    proposer.set_lora_hook(lambda mapping: None)
    proposer.attn_groups = [[Mock()]]
    return proposer


def _cpu_graph_manager_patches(monkeypatch):
    """Let a real CudaGraphManager be built without CUDA or a graph pool."""
    from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    monkeypatch.setattr(gpu_cudagraph_utils, "get_offloader", lambda: Mock())


def _captured_draft_rows(proposer) -> list[int]:
    """The draft row counts this configuration would capture a graph for."""
    manager = proposer.cudagraph_manager
    assert manager is not None
    from vllm.config.compilation import CUDAGraphMode

    descs = manager._capture_descs.get(CUDAGraphMode.FULL, [])
    return sorted({desc.num_tokens for desc in descs})


@pytest.mark.parametrize(
    ("k", "max_num_seqs", "capture_sizes", "expected_rows", "expected_uncovered"),
    [
        # The H100 serving shape the TTFT lane measured: every request count
        # from 1 to 16 has a graph, and the top draft batch (128 rows) is
        # captured rather than dropped.
        (8, 16, [1, 2, 4, 8, 16, 32, 64, 128, 144], [8, 16, 32, 64, 128], []),
        # The same capture list at K=3 does NOT cover the top: the largest
        # captured count is 33 draft rows, so 12 or more concurrent requests
        # draft eagerly. This is the silent gap the startup line now names.
        (
            3,
            16,
            [1, 2, 4, 8, 16, 32, 64, 128, 144],
            [3, 6, 9, 18, 33],
            list(range(12, 17)),
        ),
        # The greedy matrix's own shape: small, and fully covered.
        (8, 4, [8, 16, 32, 64], [8, 16, 32], []),
        # K=1 with a capture list whose smallest entry exceeds max_num_seqs:
        # nothing is captured at all and every proposal drafts eagerly.
        (1, 4, [8, 16, 32, 64], [], [1, 2, 3, 4]),
    ],
    ids=["h100_k8_covered", "h100_k3_top_uncovered", "matrix_k8_covered", "k1_none"],
)
def test_draft_graph_coverage_is_derived_from_k_seqs_and_capture_sizes(
    k, max_num_seqs, capture_sizes, expected_rows, expected_uncovered, monkeypatch
):
    """Which request counts get a draft graph follows from all three inputs.

    The H100 TTFT lane proposed that a 16-request batch at K=8 falls back to
    eager drafting because 128 draft rows exceed the largest capture size. It
    does not: with the default capture list, 128 is captured and the batch
    replays a graph. The gap is real for other shapes, though, and it is
    silent -- K=3 on the same server leaves 12 or more concurrent requests
    drafting eagerly. Pinning all four shapes keeps the arithmetic honest in
    both directions rather than asserting the conclusion for one of them.
    """
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.spec_decode.uno import uncovered_draft_request_counts

    _cpu_graph_manager_patches(monkeypatch)
    proposer = _cpu_uno_proposer(
        k, max_num_seqs=max_num_seqs, capture_sizes=capture_sizes
    )
    proposer.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)

    rows = _captured_draft_rows(proposer)
    assert rows == expected_rows, (
        f"K={k}, max_num_seqs={max_num_seqs}, capture_sizes={capture_sizes} "
        f"should capture draft row counts {expected_rows}, got {rows}"
    )
    assert uncovered_draft_request_counts(rows, k, max_num_seqs) == expected_uncovered


def test_draft_dispatch_pads_up_to_the_next_captured_size(monkeypatch):
    """Every request count in range dispatches to a graph, not only exact fits.

    A draft batch is n*K rows, and most n produce a row count no capture size
    equals: at K=8 the captured counts are 8, 16, 32, 64 and 128, while three
    requests need 24 rows and five need 40. Dispatch pads up to the next
    captured descriptor, so those still replay a graph. If dispatch ever
    required an exact match, this test fails for every n outside the captured
    set, which is the regression the H100 lane suspected had already happened.
    """
    from vllm.config.compilation import CUDAGraphMode

    _cpu_graph_manager_patches(monkeypatch)
    k = 8
    max_num_seqs = 16
    proposer = _cpu_uno_proposer(
        k,
        max_num_seqs=max_num_seqs,
        capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 144],
    )
    proposer.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    manager = proposer.cudagraph_manager
    assert manager is not None
    manager._graphs_captured = True

    padded: dict[int, int] = {}
    for n in range(1, max_num_seqs + 1):
        desc = manager.dispatch(n, n * k, k, 2)
        assert desc.cg_mode == CUDAGraphMode.FULL, (
            f"{n} requests ({n * k} draft rows) fell back to eager drafting"
        )
        assert desc.num_tokens >= n * k
        assert desc.num_reqs is not None and desc.num_reqs >= n
        padded[n] = desc.num_tokens

    # The top of the range is served by its own graph, not by padding into a
    # larger one, and the row counts that are not captured pad upward.
    assert padded[16] == 128
    assert padded[3] == 32 and padded[5] == 64
    assert sorted(set(padded.values())) == [8, 16, 32, 64, 128]


def test_draft_dispatch_falls_back_above_the_largest_captured_size(monkeypatch):
    """The fallback exists; it is the top of the range, and it is silent.

    At K=3 with the same capture list the largest captured draft batch is 33
    rows, so 12 concurrent requests (36 rows) have nothing to pad into and
    drop to eager. Nothing in the dispatch path says so, which is why the
    speculator now reports coverage once at startup.
    """
    from vllm.config.compilation import CUDAGraphMode

    _cpu_graph_manager_patches(monkeypatch)
    k = 3
    proposer = _cpu_uno_proposer(
        k, max_num_seqs=16, capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 144]
    )
    proposer.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    manager = proposer.cudagraph_manager
    assert manager is not None
    manager._graphs_captured = True

    assert manager.dispatch(11, 11 * k, k, 2).cg_mode == CUDAGraphMode.FULL
    assert manager.dispatch(12, 12 * k, k, 2).cg_mode == CUDAGraphMode.NONE


def test_uncovered_request_counts_are_the_top_of_the_range():
    """Padding means the gap can only be at the top, never a hole inside."""
    from vllm.v1.worker.gpu.spec_decode.uno import uncovered_draft_request_counts

    assert uncovered_draft_request_counts([8, 16, 32], 8, 4) == []
    assert uncovered_draft_request_counts([8, 16, 32], 8, 6) == [5, 6]
    assert uncovered_draft_request_counts([], 8, 3) == [1, 2, 3]
    # A contiguous tail, so a reader can act on the first uncovered count.
    gap = uncovered_draft_request_counts([33], 3, 16)
    assert gap == list(range(gap[0], 17))
    # Degenerate shapes answer rather than raising.
    assert uncovered_draft_request_counts([8], 0, 4) == []
    assert uncovered_draft_request_counts([8], 8, 0) == []


def test_startup_coverage_is_logged_for_both_outcomes(monkeypatch, caplog):
    """A deployment learns about eager drafting at startup, not from latency.

    The one-time eager line can fire only once and says nothing about the rest
    of the range, so on its own it cannot tell an operator that this server
    will draft eagerly at its own concurrency. Both outcomes are logged: the
    covered case states the range it covers, the uncovered case warns and
    names the capture size to add.
    """
    import logging

    from vllm.config.compilation import CUDAGraphMode

    _cpu_graph_manager_patches(monkeypatch)
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 144]

    covered = _cpu_uno_proposer(8, max_num_seqs=16, capture_sizes=sizes)
    covered.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    manager = covered.cudagraph_manager
    assert manager is not None
    for desc in manager._capture_descs[CUDAGraphMode.FULL]:
        manager.graphs[desc] = Mock()
    with caplog.at_level(logging.INFO, logger="vllm.v1.worker.gpu.spec_decode.uno"):
        covered._log_draft_graph_coverage()
    assert "cover every request count" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    caplog.clear()
    gapped = _cpu_uno_proposer(3, max_num_seqs=16, capture_sizes=sizes)
    gapped.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    manager = gapped.cudagraph_manager
    assert manager is not None
    for desc in manager._capture_descs[CUDAGraphMode.FULL]:
        manager.graphs[desc] = Mock()
    with caplog.at_level(logging.INFO, logger="vllm.v1.worker.gpu.spec_decode.uno"):
        gapped._log_draft_graph_coverage()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, caplog.text
    message = warnings[0].getMessage()
    assert "will draft eagerly" in message
    # It names the capture size to add, which is what the reader acts on.
    assert "48" in message


def test_warmup_proposals_do_not_consume_the_serving_counters(monkeypatch):
    """Startup work must not spend the one-time eager announcement.

    On the H100 the profiling pass made five eager proposals before the API
    server accepted a request, so the counter was already past one and the
    "drafting eagerly" line had been emitted and could never be emitted again.
    Any real fallback during serving was therefore unobservable, which is why
    that run's receipts cannot answer whether one happened. Profiling and
    capture proposals now count separately.
    """
    from vllm.config.compilation import CUDAGraphMode

    _cpu_graph_manager_patches(monkeypatch)
    proposer = _cpu_uno_proposer(8, max_num_seqs=4, capture_sizes=[8, 16, 32])
    proposer.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    manager = proposer.cudagraph_manager
    assert manager is not None
    manager._graphs_captured = True

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.uno.prepare_uno_inputs_fused",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.uno.build_slot_mappings_by_layer",
        lambda *args, **kwargs: {},
    )
    # The dummy_run call dispatches into a graph, so give it one to replay.
    desc = manager.dispatch(2, 2 * 8, 8, 2)
    proposer._graph_attn_metadata[desc] = {"layer": object()}
    manager.graphs[desc] = Mock()
    manager.run_fullgraph = Mock()
    batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.arange(2),
        seq_lens_cpu_upper_bound=torch.full((4,), 8, dtype=torch.int32),
    )
    common = dict(
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.zeros(1),
        aux_hidden_states=None,
        num_sampled=torch.zeros(2, dtype=torch.int32),
        num_rejected=torch.zeros(2, dtype=torch.int32),
        last_sampled=torch.zeros(2, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(2, dtype=torch.int64),
        temperature=torch.zeros(2),
        seeds=torch.zeros(2, dtype=torch.int64),
    )

    proposer.propose(batch, is_profile=True, **common)
    assert (proposer.num_eager_proposals, proposer.num_graph_replays) == (0, 0)
    assert proposer.num_warmup_proposals == 1

    proposer.propose(batch, dummy_run=True, **common)
    assert (proposer.num_eager_proposals, proposer.num_graph_replays) == (0, 0)
    assert proposer.num_warmup_proposals == 2


@pytest.mark.parametrize(
    ("k", "max_num_seqs"),
    [(1, 4), (8, 4), (1, 16), (8, 16)],
    ids=["k1_seqs4", "k8_seqs4", "k1_seqs16", "k8_seqs16"],
)
def test_draft_warmup_covers_every_reachable_request_count(
    k, max_num_seqs, monkeypatch
):
    """Warmup enumerates the whole request range, not a sample of it.

    The fused draft-input kernel takes the request count as a Triton constexpr,
    so three requests and four requests are separate specialisations and each
    is compiled the first time it is seen. Graph capture never runs that path
    at all, so on an H100 the first served request spent 120 ms inside its
    draft proposal against under 1.2 ms afterwards, and a cell that first
    reached a new request count paid again mid-run. Covering only the counts
    one workload happens to hit would leave the next workload paying at its own
    first request, so the set is the full range and the shape count does not
    depend on K.
    """
    from vllm.config.compilation import CUDAGraphMode

    _cpu_graph_manager_patches(monkeypatch)
    proposer = _cpu_uno_proposer(k, max_num_seqs=max_num_seqs)
    proposer.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)

    counts = proposer.draft_warmup_token_counts()
    assert counts == list(range(1, max_num_seqs + 1)), (
        f"K={k}, max_num_seqs={max_num_seqs} must warm every request count"
    )
    # _dummy_run derives num_reqs as min(num_tokens, max_num_seqs), so each
    # entry produces exactly the request count it names.
    assert [min(n, max_num_seqs) for n in counts] == list(range(1, max_num_seqs + 1))


def test_draft_warmup_request_counts_are_independent_of_k():
    """The shape set follows the request count alone.

    The kernel specialises on the request count and on the draft row count,
    and the row count is the request count times K, so enumerating requests
    covers both. A set that varied with K would silently shrink when a
    deployment lowered K and leave the larger batches cold.
    """
    from vllm.v1.worker.gpu.spec_decode.uno import draft_warmup_request_counts

    assert draft_warmup_request_counts(4) == [1, 2, 3, 4]
    assert draft_warmup_request_counts(1) == [1]
    assert draft_warmup_request_counts(0) == []
    assert draft_warmup_request_counts(-1) == []


@pytest.mark.parametrize("use_flashinfer", [False, True])
def test_uno_served_launches_cover_sampler_flags_and_integer_buckets(use_flashinfer):
    """The production plan models the sampler backend serving can reach."""
    from vllm.v1.worker.gpu.spec_decode.uno import enumerate_uno_served_launches

    plan = enumerate_uno_served_launches(
        max_num_reqs=16,
        k=8,
        max_num_tokens=2048,
        max_model_len=4096,
        num_sm=82,
        use_flashinfer=use_flashinfer,
    )

    assert plan.prepare_request_counts == tuple(range(1, 17))
    assert {warmup.mode.name for warmup in plan.sampler_warmups} == {
        "top_p_only",
        "top_k_top_p",
        "top_k_only",
        "neither",
    }
    if not use_flashinfer:
        assert {warmup.source for warmup in plan.sampler_warmups} >= {
            "pure_prefill",
            "verification",
            "chunked_or_mixed",
        }
    branches = {
        warmup.mode.name: warmup.sampler_branch
        for warmup in plan.sampler_warmups
        if warmup.sampler_branch != "native_verification"
    }
    expected_branches = {
        "top_p_only": "flashinfer" if use_flashinfer else "triton",
        "top_k_top_p": "flashinfer" if use_flashinfer else "triton",
        "top_k_only": "flashinfer" if use_flashinfer else "triton",
        "neither": "triton",
    }
    assert branches == expected_branches
    assert len(plan.sampler_keys) == 46
    for has_k in (False, True):
        assert ("_topp_sb_stats_kernel", has_k, 32) in plan.sampler_keys
        assert ("_topp_sb_stats_kernel", has_k, 4) in plan.sampler_keys
    assert ("_topk_topp_kernel", "one", True, False, False) in plan.sampler_keys
    assert (
        "_topk_topp_kernel",
        "multiple_of_16",
        True,
        True,
        True,
    ) in plan.sampler_keys
    assert plan.served_sampler_keys == plan.sampler_keys


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_num_reqs": 2, "k": 0, "max_num_tokens": 4, "max_model_len": 4},
        {"max_num_reqs": 4, "k": 2, "max_num_tokens": 3, "max_model_len": 4},
        {"max_num_reqs": 4, "k": 2, "max_num_tokens": 8, "max_model_len": 0},
    ],
)
def test_uno_served_launch_enumerator_rejects_unbounded_axes(kwargs):
    from vllm.v1.worker.gpu.spec_decode.uno import enumerate_uno_served_launches

    with pytest.raises(ValueError):
        enumerate_uno_served_launches(**kwargs, num_sm=82, use_flashinfer=False)


@pytest.mark.parametrize(
    "max_num_reqs,k,max_num_tokens,max_model_len",
    [
        (4, 8, 2048, 4),
        (16, 8, 2048, 4096),
        (4, 3, 7, 32),
    ],
)
def test_uno_served_sampler_shapes_bound_every_axis(
    max_num_reqs, k, max_num_tokens, max_model_len
):
    """A new shape axis must acquire an explicit bound in this contract."""
    from vllm.v1.worker.gpu.spec_decode.uno import _served_sampler_shapes

    bounds = dict(
        max_num_reqs=max_num_reqs,
        k=k,
        max_num_tokens=max_num_tokens,
        max_model_len=max_model_len,
    )
    assert set(inspect.signature(_served_sampler_shapes).parameters) == set(bounds)
    for num_reqs, num_rows, _source in _served_sampler_shapes(**bounds):
        assert 1 <= num_reqs <= max_num_reqs
        assert num_reqs <= num_rows <= max_num_tokens
        largest_request = (num_rows + num_reqs - 1) // num_reqs
        assert largest_request <= k + 1
        assert largest_request <= max_model_len


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("speculator_kind", ["none", "other", "uno"])
def test_warmup_kernels_runs_two_full_verifications(
    monkeypatch, num_reqs, speculator_kind
):
    """CPU_OBSERVED/CUDA_UNVERIFIED: add only the missing Uno C=2 extent.

    Dense startup retains its three decode steps and original KV reservation;
    other drafters retain five. Uno's all-request step already covers C=2 when
    max_num_seqs is two, so its two-request extent must execute exactly once.
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
    from vllm.v1.worker.gpu import warmup

    k = 0 if speculator_kind == "none" else 8
    speculator = None if speculator_kind == "none" else object()
    if speculator_kind == "uno":
        speculator = object.__new__(UnoSpeculator)
    runner = SimpleNamespace(
        num_speculative_steps=k,
        speculator=speculator,
        decode_query_len=k + 1,
        max_model_len=4096,
        adaptive_verification=None,
        is_pooling_model=False,
        is_encoder_decoder=False,
        is_last_pp_rank=True,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        model_state=SimpleNamespace(max_encoder_len=0),
        scheduler_config=SimpleNamespace(
            max_num_seqs=num_reqs, max_num_batched_tokens=2048
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=1, num_kv_heads=1, head_size=1, dtype=torch.float32
                    ),
                )
            ],
            # At two dense requests, one extra reserved step would reduce the
            # admitted warmup batch to one request even without running it.
            num_blocks=11 if k == 0 and num_reqs == 2 else 1024,
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=k, is_mm_encoder_only=False),
        kv_block_zeroer=None,
        kv_connector=SimpleNamespace(set_disabled=lambda _: None),
    )
    steps = []
    monkeypatch.setattr(warmup.torch.accelerator, "synchronize", lambda: None)
    warmup.warmup_kernels(runner, steps.append, lambda _: None)
    assert len(steps[0].scheduled_new_reqs) == num_reqs
    decode_steps = steps[1:-1]
    expected = [(num_reqs, num_reqs if k else 0)]
    if num_reqs >= 2:
        if speculator_kind == "uno" and num_reqs > 2:
            expected.append((2, 2))
        expected.append((2, 1 if k else 0))
        if k:
            expected.append((2, 0))
        expected.append((1, 1 if k else 0))
    if k:
        expected.append((1, 0))
    assert [
        (len(step.num_scheduled_tokens), len(step.scheduled_spec_decode_tokens))
        for step in decode_steps
    ] == expected
    if speculator_kind == "uno" and num_reqs >= 2:
        assert sum(
            len(step.scheduled_spec_decode_tokens) == 2
            and step.total_num_scheduled_tokens == 18
            for step in decode_steps
        ) == 1, "startup must execute the two-request full verification exactly once"


@pytest.mark.parametrize("top_k", [None, 50])
@pytest.mark.parametrize("use_flashinfer", [False, True])
def test_uno_warmup_executes_native_verification_s4(monkeypatch, top_k, use_flashinfer):
    """CPU_OBSERVED/CUDA_UNVERIFIED: drive real rejection/filter launch callers.

    CUDA-labelled CPU storage reaches the production CUDA launch selector;
    only device kernels are replaced with recorders, so this proves dispatch
    and all five round launches, not device arithmetic or compilation.
    """
    from vllm.v1.sample.ops import topk_topp_sampler, topk_topp_triton
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.states import SamplingStates
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rejection_module
    from vllm.v1.worker.gpu.spec_decode.uno import enumerate_uno_served_launches

    plan = enumerate_uno_served_launches(
        max_num_reqs=4,
        k=8,
        max_num_tokens=2048,
        max_model_len=4096,
        num_sm=82,
        use_flashinfer=use_flashinfer,
    )
    warmup = next(
        (
            w
            for w in plan.sampler_warmups
            if w.sampler_branch == "native_verification"
            and w.num_reqs == 2
            and w.num_rows == 18
            and w.mode.top_k == top_k
            and w.mode.top_p is not None
        ),
        None,
    )
    assert warmup is not None, "verification must warm S=4 independently of FlashInfer"

    class CudaLabelledTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda", 0)

    class ArrayState:
        def __init__(self, values, dtype):
            self.np = np.asarray(values, dtype=dtype)

        @property
        def gpu(self):
            return torch.from_numpy(self.np)

    states = object.__new__(SamplingStates)
    states.vocab_size = 64
    states.temperature = ArrayState([1.0, 1.0], np.float32)
    states.top_k = ArrayState([64, 64], np.int32)
    states.top_p = ArrayState([1.0, 1.0], np.float32)
    states.min_p = ArrayState([0.0, 0.0], np.float32)
    states.seeds = ArrayState([19, 23], np.int64)
    states.seeds_set = np.asarray([True, False])
    states.num_logprobs = np.asarray([-1, -1])
    states.apply_staged_writes = lambda: None
    states.apply_temperature = lambda *_: None
    sampler = object.__new__(Sampler)
    sampler.use_flashinfer = use_flashinfer
    sampler.compute_nans = False
    sampler.logprobs_mode = "raw_logprobs"
    sampler.use_fp64_gumbel = False
    sampler.needs_logits_processing = np.zeros(2, dtype=bool)
    sampler.sampling_states = states
    sampler.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.zeros(2))
    )
    for attribute, method in (
        ("logit_bias_state", "apply_logit_bias"),
        ("penalties_state", "apply_penalties"),
        ("bad_words_state", "apply_bad_words"),
        ("thinking_budget_state", "apply"),
    ):
        setattr(sampler, attribute, SimpleNamespace(**{method: lambda *_: None}))
    rejection = object.__new__(rejection_module.RejectionSampler)
    rejection.sampler = sampler
    rejection.num_speculative_steps = 8
    rejection.enable_adaptive_verification = False
    rejection.synthetic_conditional_rates = None
    rejection.use_block_verification = False
    runner = object.__new__(GPUModelRunner)
    runner.sampler = sampler
    runner.rejection_sampler = rejection
    runner.device = torch.device("cpu")
    runner.input_buffers = InputBuffers(2, 18, runner.device)
    runner.model = SimpleNamespace(
        compute_logits=lambda hidden: torch.ones((hidden.shape[0], 64)).as_subclass(
            CudaLabelledTensor
        )
    )

    launches = []

    class LaunchRecorder:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.name, args, kwargs))

            return launch

    for name in (
        "_topk_topp_kernel",
        "_topp_sb_stats_kernel",
        "_topp_sb_step_kernel",
        "_topp_sb_mask_kernel",
    ):
        monkeypatch.setattr(topk_topp_triton, name, LaunchRecorder(name))
    monkeypatch.setattr(topk_topp_sampler, "HAS_TRITON", True)
    monkeypatch.setattr(topk_topp_triton, "num_compute_units", lambda _: 82)
    for name in ("_TRITON_SPLIT_CACHE", "_TRITON_BUFFER_CACHE", "_TRITON_TABLE_CACHE"):
        monkeypatch.setattr(topk_topp_triton, name, {})
    monkeypatch.setattr(
        rejection_module,
        "rejection_sample",
        lambda *_a, **_k: (
            torch.zeros((2, 9), dtype=torch.int64),
            torch.ones(2, dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        rejection_module,
        "get_num_sampled_and_rejected",
        lambda num_sampled, *_: (num_sampled, torch.full((2,), 8)),
    )

    assert runner._warm_up_uno_sampler(torch.empty((2, 3)), warmup=warmup) == (
        "native_verification"
    )
    split_launches = [(name, args, kw) for name, args, kw in launches if "_sb_" in name]
    assert {name for name, _, _ in split_launches} == {
        "_topp_sb_stats_kernel",
        "_topp_sb_step_kernel",
        "_topp_sb_mask_kernel",
    }
    assert all(
        kw["S"] == 4 and kw["HAS_K"] == (top_k is not None)
        for _, _, kw in split_launches
    )
    assert [args[6] for name, args, _ in split_launches if "_step_" in name] == list(
        range(5)
    )
    assert sampler.use_flashinfer is use_flashinfer
    assert states.seeds_set.tolist() == [True, False]
    assert states.top_k.np.tolist() == [64, 64]


def test_draft_warmup_runs_every_shape_through_the_real_dummy_run(monkeypatch):
    """The runner drives every production-plan call through its real sampler.

    The prepare and sampler inputs deliberately have different token lengths;
    the runner must use the plan's request count and row count instead of
    reconstituting the old ``num_reqs * K`` table.
    """
    from vllm.v1.worker.gpu import model_runner as model_runner_module
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.uno import (
        UnoSamplerWarmup,
        UnoSamplingMode,
        UnoServedLaunches,
    )

    runner = object.__new__(GPUModelRunner)
    ran: list[int] = []
    reported: list[tuple[object, ...]] = []
    proposer = object.__new__(UnoSpeculator)
    proposer.k = 8
    proposer._step = 9
    proposer.report_draft_warmup = lambda *args: reported.append(args)
    runner.speculator = proposer
    plan = UnoServedLaunches(
        prepare_request_counts=(1, 2, 3, 4),
        sampler_warmups=(
            UnoSamplerWarmup(
                1,
                1,
                UnoSamplingMode("top_p_only", None, 0.9),
                "pure_prefill",
                "triton",
                (("_topp_sb_stats_kernel", False, 32),),
            ),
            UnoSamplerWarmup(
                2,
                18,
                UnoSamplingMode("top_k_top_p", 50, 0.9),
                "verification",
                "native_verification",
                (("_topp_sb_stats_kernel", True, 4),),
            ),
            UnoSamplerWarmup(
                3,
                17,
                UnoSamplingMode("top_k_only", 50, None),
                "chunked_or_mixed",
                "triton",
                (("_topk_topp_kernel", "generic", True, False, False),),
            ),
            UnoSamplerWarmup(
                4,
                4,
                UnoSamplingMode("neither", None, None),
                "pure_prefill",
                "triton",
                (),
            ),
        ),
    )
    runner.max_num_reqs = 4
    runner.max_num_tokens = 64
    runner.max_model_len = 128
    runner.device = torch.device("cuda", 0)
    runner.sampler = SimpleNamespace(use_flashinfer=False)
    monkeypatch.setattr(
        model_runner_module, "enumerate_uno_served_launches", lambda **_: plan
    )
    monkeypatch.setattr(model_runner_module, "num_compute_units", lambda _: 82)
    sampler_runs: list[tuple[int, int, int]] = []

    def dummy_run(num_tokens):
        # Model/draft code is free to consume a global generator during
        # warmup. The production wrapper must put it back before serving.
        torch.rand(3)
        proposer._step += 1
        ran.append(num_tokens)
        return None, torch.empty(num_tokens, 3)

    runner._dummy_run = dummy_run

    def warm_up_uno_sampler(hidden, *, warmup):
        sampler_runs.append((hidden.shape[0], warmup.num_reqs, warmup.num_rows))
        return warmup.sampler_branch

    runner._warm_up_uno_sampler = warm_up_uno_sampler

    prior_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(917)
        served_rng_state = torch.random.get_rng_state()
        runner._warm_up_draft_kernels()

        assert ran == [1, 2, 3, 4]
        assert sampler_runs == [(1, 1, 1), (2, 2, 18), (3, 3, 17), (4, 4, 4)]
        assert reported == [
            (
                4,
                4,
                3,
                {
                    "top_p_only": "triton",
                    "verification/top_k_top_p": "native_verification",
                    "top_k_only": "triton",
                    "neither": "triton",
                },
                {
                    "top_p_only": ("_topp_sb_stats_kernel",),
                    "verification/top_k_top_p": ("_topp_sb_stats_kernel",),
                    "top_k_only": ("_topk_topp_kernel",),
                    "neither": (),
                },
            )
        ]
        assert torch.equal(torch.random.get_rng_state(), served_rng_state)
        assert proposer._step == 0
    finally:
        torch.random.set_rng_state(prior_state)


def test_profile_run_preserves_ordinary_dummy_sampler_dispatch(monkeypatch):
    """The other production dummy-sampler caller must remain non-speculative."""
    from vllm.v1.worker.gpu import model_runner as model_runner_module
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    runner.supports_mm_inputs = False
    runner.max_num_tokens = 3
    runner.is_last_pp_rank = True
    runner.pooling_runner = None
    runner.device = torch.device("cpu")
    runner.input_buffers = InputBuffers(3, 3, runner.device)
    runner._dummy_run = lambda *_a, **_kw: (None, torch.ones((3, 4)))
    runner.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
    runner.sampler = Mock()
    runner.rejection_sampler = Mock()
    runner.reset_encoder_cache = lambda: None
    monkeypatch.setattr(
        model_runner_module.torch.accelerator, "synchronize", lambda: None
    )

    runner.profile_run()

    runner.sampler.assert_called_once()
    _, batch = runner.sampler.call_args.args
    assert batch.num_reqs == batch.num_tokens == 3
    assert batch.num_draft_tokens == 0
    runner.rejection_sampler.assert_not_called()


def test_draft_warmup_skips_a_speculator_that_does_not_ask_for_it():
    """Other drafters are untouched by Uno's fixed launch family."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    ran: list[int] = []
    runner._dummy_run = lambda num_tokens: ran.append(num_tokens)

    runner.speculator = SimpleNamespace()
    runner._warm_up_draft_kernels()
    assert ran == []

    runner.speculator = SimpleNamespace(report_draft_warmup=Mock())
    runner._warm_up_draft_kernels()
    assert ran == []


def test_draft_warmup_reports_execution_not_a_nonlast_rank_plan(monkeypatch):
    """A rank with no sampler reports zero executed sampler rows and keys."""
    from vllm.v1.worker.gpu import model_runner as model_runner_module
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.uno import (
        UnoSamplerWarmup,
        UnoSamplingMode,
        UnoServedLaunches,
    )

    proposer = object.__new__(UnoSpeculator)
    proposer.k = 8
    proposer._step = 5
    reports: list[tuple[object, ...]] = []
    proposer.report_draft_warmup = lambda *args: reports.append(args)
    plan = UnoServedLaunches(
        prepare_request_counts=(1, 2),
        sampler_warmups=(
            UnoSamplerWarmup(
                1,
                1,
                UnoSamplingMode("top_p_only", None, 0.9),
                "pure_prefill",
                "triton",
                (("_topp_sb_stats_kernel", False, 32),),
            ),
        ),
    )
    runner = object.__new__(GPUModelRunner)
    runner.speculator = proposer
    runner.sampler = None
    runner.max_num_reqs = 2
    runner.max_num_tokens = 8
    runner.max_model_len = 32
    runner.device = torch.device("cpu")
    runner._dummy_run = lambda _tokens: (None, None)
    runner._warm_up_uno_sampler = Mock(side_effect=AssertionError("non-last rank"))
    monkeypatch.setattr(
        model_runner_module, "enumerate_uno_served_launches", lambda **_: plan
    )
    monkeypatch.setattr(model_runner_module, "num_compute_units", lambda _: 82)

    runner._warm_up_draft_kernels()

    assert reports == [(2, 0, 0, {}, {})]
    assert proposer._step == 0
    runner._warm_up_uno_sampler.assert_not_called()


@pytest.mark.parametrize("use_flashinfer", [False, True])
def test_draft_warmup_enumerator_uses_the_production_sampler_capability(
    monkeypatch, use_flashinfer
):
    """The only production enumerator caller passes the installed backend."""
    from vllm.v1.worker.gpu import model_runner as model_runner_module
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.uno import UnoServedLaunches

    runner = object.__new__(GPUModelRunner)
    runner.speculator = object.__new__(UnoSpeculator)
    runner.speculator.k = 8
    runner.sampler = SimpleNamespace(use_flashinfer=use_flashinfer)
    runner.max_num_reqs = 4
    runner.max_num_tokens = 64
    runner.max_model_len = 128
    runner.device = torch.device("cpu")
    call_args: list[dict[str, object]] = []

    def enumerate_launches(**kwargs):
        call_args.append(kwargs)
        return UnoServedLaunches((), ())

    monkeypatch.setattr(
        model_runner_module, "enumerate_uno_served_launches", enumerate_launches
    )
    monkeypatch.setattr(model_runner_module, "num_compute_units", lambda _: 82)

    runner._warm_up_draft_kernels()

    assert call_args[0]["use_flashinfer"] is use_flashinfer


@pytest.mark.parametrize(
    ("use_flashinfer", "expected_branch"),
    [(True, "flashinfer"), (False, "triton")],
)
def test_uno_sampler_warmup_keeps_the_served_sampler_backend(
    use_flashinfer, expected_branch
):
    """Warmup neither injects a seed nor changes the sampler's backend flag."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.uno import UnoSamplerWarmup, UnoSamplingMode

    class ArrayState:
        def __init__(self, values):
            self.np = np.asarray(values)

    states = SimpleNamespace(
        vocab_size=100,
        temperature=ArrayState([1.0, 1.0]),
        top_k=ArrayState([100, 100]),
        top_p=ArrayState([1.0, 1.0]),
        min_p=ArrayState([0.0, 0.0]),
        seeds=ArrayState([19, 23]),
        seeds_set=np.asarray([True, False]),
        apply_staged_writes=Mock(),
    )
    flashinfer_calls: list[bool] = []
    triton_top_p_launches: list[bool] = []
    sampler = SimpleNamespace(
        use_flashinfer=use_flashinfer,
        needs_logits_processing=np.zeros(2, dtype=bool),
        sampling_states=states,
    )

    def sample_random(*args, **kwargs):
        use_flashinfer_call = (
            kwargs["use_flashinfer"] if "use_flashinfer" in kwargs else args[6]
        )
        if use_flashinfer_call:
            flashinfer_calls.append(True)
        else:
            triton_top_p_launches.append(True)

    sampler._sample_random = sample_random
    runner = object.__new__(GPUModelRunner)
    runner.sampler = sampler
    runner.device = torch.device("cpu")

    def dummy_sampler_run(_hidden, *, num_reqs):
        has_filters = (
            states.top_k.np[:num_reqs] < states.vocab_size
        ).any() or (states.top_p.np[:num_reqs] < 1.0).any()
        sampler._sample_random(
            None,
            None,
            None,
            None,
            None,
            None,
            bool(
                sampler.use_flashinfer
                and has_filters
                and not states.seeds_set[:num_reqs].any()
            ),
        )

    runner._dummy_sampler_run = dummy_sampler_run
    warmup = UnoSamplerWarmup(
        1,
        1,
        UnoSamplingMode("top_p_only", None, 0.9),
        "pure_prefill",
        expected_branch,
        (),
    )
    before_seeds = states.seeds.np.copy()
    before_seeds_set = states.seeds_set.copy()

    branch = runner._warm_up_uno_sampler(torch.empty(1, 3), warmup=warmup)

    assert branch == expected_branch
    assert sampler.use_flashinfer is use_flashinfer
    assert states.seeds.np.tolist() == before_seeds.tolist()
    assert states.seeds_set.tolist() == before_seeds_set.tolist()
    if use_flashinfer:
        assert flashinfer_calls == [True]
        assert triton_top_p_launches == []
    else:
        assert flashinfer_calls == []
        assert triton_top_p_launches == [True]


@pytest.mark.parametrize("use_flashinfer", [True, False])
def test_sampler_selects_flashinfer_without_an_explicit_seed(
    monkeypatch, use_flashinfer
):
    """The actual production sampler chooses FlashInfer before Triton filters."""
    from vllm.v1.worker.gpu.sample import sampler as sampler_module

    subject = object.__new__(sampler_module.Sampler)
    subject.use_flashinfer = use_flashinfer
    subject.logprobs_mode = "raw_logprobs"
    subject.use_fp64_gumbel = False
    subject.sampling_states = SimpleNamespace(
        get_top_k_top_p=lambda *_args: (None, torch.tensor([0.9])),
        any_greedy=lambda *_args: False,
        any_explicit_seed=lambda *_args: False,
        temperature=SimpleNamespace(gpu=torch.ones(1)),
        seeds=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int64)),
    )
    subject.apply_sampling_params = lambda logits, *_args, **_kwargs: logits
    calls: list[str] = []
    monkeypatch.setattr(
        sampler_module,
        "flashinfer_sample",
        lambda logits, _top_k, _top_p: calls.append("flashinfer")
        or torch.zeros(logits.shape[0], dtype=torch.int64),
    )
    monkeypatch.setattr(
        sampler_module,
        "apply_top_k_top_p",
        lambda logits, _top_k, _top_p: calls.append("triton_filter") or logits,
    )
    monkeypatch.setattr(
        sampler_module,
        "gumbel_sample",
        lambda logits, *_args, **_kwargs: calls.append("triton_sample")
        or torch.zeros(logits.shape[0], dtype=torch.int64),
    )

    subject.sample(
        torch.ones((1, 4)),
        torch.tensor([0]),
        torch.tensor([0]),
        np.asarray([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
    )

    assert calls == (
        ["flashinfer"] if use_flashinfer else ["triton_filter", "triton_sample"]
    )


def test_uno_warmup_key_set_covers_served_prepare_and_sampler_shapes():
    """CPU_OBSERVED/CUDA_UNVERIFIED launch-set contract.

    Warmup and serving use deliberately different input batches. The
    production enumerator supplies both the raw served domain and the selected
    warmup calls, so this can fail when a warmup is changed back to the old
    ``num_reqs * K`` sample list.
    """
    from vllm.v1.worker.gpu.spec_decode.uno import enumerate_uno_served_launches
    from vllm.v1.worker.gpu.spec_decode.uno_prepare import (
        UNO_PREPARE_RUNTIME_SCALARS,
        _target_input_lengths,
        prepare_uno_launch_key,
    )

    max_reqs, k, vocab, num_sms = 16, 8, 151936, 82
    buffers = InputBuffers(max_reqs, 2048, torch.device("cpu"))
    slot_mapping = torch.empty(2048, dtype=torch.int64)
    sample_idx = torch.empty(max_reqs * k, dtype=torch.int32)
    block_table = torch.empty((max_reqs, 256), dtype=torch.int32)
    assert UNO_PREPARE_RUNTIME_SCALARS == (
        "step",
        "target_query_len",
        "target_position_len",
    )

    def prepare_key(input_batch, num_sampled, num_rejected, num_reqs):
        return prepare_uno_launch_key(
            buffers,
            slot_mapping,
            sample_idx,
            input_batch,
            num_sampled,
            num_rejected,
            block_table,
            num_reqs=num_reqs,
            k=k,
            state_capacity=max_reqs,
            block_size=16,
            max_model_len=4096,
            noise_seed=0,
            noise_high=vocab - 267,
            has_rejected=True,
            block=256,
        )

    warmup_prepare_keys = set()
    served_prepare_keys = set()
    for num_reqs in range(1, max_reqs + 1):
        warmup_batch = SimpleNamespace(
            query_start_loc=torch.empty(num_reqs + 1, dtype=torch.int32),
            positions=torch.empty(num_reqs, dtype=torch.int64),
        )
        served_batch = SimpleNamespace(
            query_start_loc=torch.empty(num_reqs + 1, dtype=torch.int32),
            positions=torch.empty(1900 + num_reqs, dtype=torch.int64),
        )
        assert _target_input_lengths(warmup_batch) != _target_input_lengths(
            served_batch
        )
        warmup_num_sampled = torch.empty(num_reqs, dtype=torch.int32)
        warmup_num_rejected = torch.empty(num_reqs, dtype=torch.int32)
        served_num_sampled = torch.empty(num_reqs + 1, dtype=torch.int32)[1:]
        served_num_rejected = torch.empty(num_reqs + 1, dtype=torch.int32)[1:]
        assert served_num_sampled.storage_offset() == 1
        warmup_prepare_keys.add(
            prepare_key(
                warmup_batch,
                warmup_num_sampled,
                warmup_num_rejected,
                num_reqs,
            )
        )
        served_prepare_keys.add(
            prepare_key(
                served_batch,
                served_num_sampled,
                served_num_rejected,
                num_reqs,
            )
        )
    assert served_prepare_keys <= warmup_prepare_keys

    plan = enumerate_uno_served_launches(
        max_num_reqs=max_reqs,
        k=k,
        max_num_tokens=2048,
        max_model_len=4096,
        num_sm=num_sms,
        use_flashinfer=False,
    )
    # ``served_sampler_keys`` is produced before the representative-call
    # selector. Mutating that selector back to the legacy ``num_reqs * K``
    # list therefore leaves the served set intact and makes this contract red.
    assert plan.served_sampler_keys == plan.sampler_keys
    assert {dict(key)["NUM_REQS"] for key in served_prepare_keys} == set(
        plan.prepare_request_counts
    )


def _self_check_sampler(use_flashinfer=False):
    """Return a sampler stub whose real backend decision can be observed."""
    calls: list[bool] = []

    def sample_random(*args, **kwargs):
        use_flashinfer_call = (
            kwargs["use_flashinfer"] if "use_flashinfer" in kwargs else args[6]
        )
        calls.append(bool(use_flashinfer_call))

    return SimpleNamespace(
        use_flashinfer=use_flashinfer,
        _sample_random=sample_random,
        calls=calls,
    )


def _run_self_check_sampler_branch(runner, sampling_params):
    """Send a mocked scheduler pass through the sampler branch observer."""
    use_flashinfer = runner.sampler.use_flashinfer and (
        sampling_params.top_k != -1 or sampling_params.top_p != 1.0
    )
    runner.sampler._sample_random(
        None, None, None, None, None, None, use_flashinfer
    )


def test_uno_startup_jit_self_check_reports_compile_without_forcing_warn_abort(
    monkeypatch,
):
    """Warn mode records a real key without changing the monitor's contract."""
    from vllm.utils import jit_monitor
    from vllm.v1.worker.gpu import warmup

    runner = SimpleNamespace(
        speculator=object.__new__(UnoSpeculator),
        adaptive_verification="saved-adaptive-state",
        # This deliberately oversized fixture proves the full-chunk bound. It
        # is not the production K=8 decode shape, which has query length 9.
        max_num_tokens=2048,
        decode_query_len=2047,
        device=torch.device("cpu"),
        sampler=_self_check_sampler(),
    )
    runner.speculator._step = 0
    worker_execute_model = Mock()
    worker_sample_tokens = Mock()

    @contextmanager
    def launched_kernels():
        yield {
            "_topk_topp_kernel": 1,
            "_topp_sb_stats_kernel": 1,
            "_topp_sb_step_kernel": 5,
            "_topp_sb_mask_kernel": 1,
        }

    def emit_compile(*_args, **_kwargs):
        assert _args[:2] == (runner, worker_execute_model)
        assert _args[3] == 2048
        assert _kwargs["req_id_prefix"].startswith("_uno_jit_self_check_")
        runner.speculator._step += 1
        _run_self_check_sampler_branch(runner, _kwargs["sampling_params"])
        _args[2](None)
        jit_monitor._handle_jit_event(
            backend="Triton",
            event="kernel JIT compilation",
            fn_name="_prepare_uno_inputs_kernel",
            detail="constexprs={K=8}; key=uno-served-key",
        )
        return True

    monkeypatch.setattr(warmup, "run_mixed_prefill_decode_warmup", emit_compile)
    monkeypatch.setattr(warmup, "capture_topk_topp_launches", launched_kernels)
    monkeypatch.setattr(warmup.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(jit_monitor, "_active", True)
    monkeypatch.setattr(jit_monitor, "_mode", "warn")
    warning = Mock()
    monkeypatch.setattr(warmup.logger, "warning", warning)
    monkeypatch.setattr(jit_monitor.logger, "warning_once", Mock())

    result = warmup.run_uno_served_jit_self_check(
        runner, worker_execute_model, worker_sample_tokens
    )

    assert runner.adaptive_verification == "saved-adaptive-state"
    assert runner.speculator._step == 0
    assert not result.passed
    assert result.monitor_armed
    assert result.sampler_calls == {name: 1 for name in (
        "top_p_only", "top_k_top_p", "top_k_only", "neither"
    )}
    assert len(result.compilations) == 4
    assert any(
        "kernel=%s" in call.args[0]
        and call.args[3] == "_prepare_uno_inputs_kernel"
        and call.args[4] == "constexprs={K=8}; key=uno-served-key"
        for call in warning.call_args_list
    )


def test_uno_startup_jit_self_check_uses_the_production_k8_batch_shape(
    monkeypatch,
):
    """K=8 serves ``decode_query_len=9``, so its mixed check uses 11 tokens."""
    from vllm.utils import jit_monitor
    from vllm.v1.worker.gpu import warmup

    runner = SimpleNamespace(
        speculator=object.__new__(UnoSpeculator),
        adaptive_verification=None,
        max_num_tokens=2048,
        decode_query_len=9,
        device=torch.device("cpu"),
        sampler=_self_check_sampler(),
    )
    observed_tokens: list[int] = []

    @contextmanager
    def launched_kernels():
        yield {
            "_topk_topp_kernel": 1,
            "_topp_sb_stats_kernel": 1,
            "_topp_sb_step_kernel": 1,
            "_topp_sb_mask_kernel": 1,
        }

    def sampled(*args, **kwargs):
        observed_tokens.append(args[3])
        _run_self_check_sampler_branch(runner, kwargs["sampling_params"])
        args[2](None)
        return True

    monkeypatch.setattr(jit_monitor, "_active", True)
    monkeypatch.setattr(warmup.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(warmup, "capture_topk_topp_launches", launched_kernels)
    monkeypatch.setattr(warmup, "run_mixed_prefill_decode_warmup", sampled)

    result = warmup.run_uno_served_jit_self_check(runner, Mock(), Mock())

    assert result.passed
    assert observed_tokens == [11, 11, 11, 11]


def test_uno_self_check_token_count_keeps_fixture_and_production_bounds_distinct():
    """K=8 production is 11 rows; the 2048-row fixture remains intentional."""
    from vllm.v1.worker.gpu.warmup import uno_self_check_token_count

    assert uno_self_check_token_count(2048, 9) == 11
    assert uno_self_check_token_count(2048, 2047) == 2048


def test_uno_startup_jit_self_check_cannot_pass_unarmed_or_unreached(monkeypatch):
    """A skipped cycle or missing kernel counter is not a successful check."""
    from vllm.utils import jit_monitor
    from vllm.v1.worker.gpu import warmup

    runner = SimpleNamespace(
        speculator=object.__new__(UnoSpeculator),
        adaptive_verification=None,
        max_num_tokens=32,
        decode_query_len=8,
        device=torch.device("cpu"),
        sampler=_self_check_sampler(),
    )
    monkeypatch.setattr(jit_monitor, "_active", False)
    with pytest.raises(RuntimeError, match="requires an armed JIT monitor"):
        warmup.run_uno_served_jit_self_check(runner, Mock(), Mock())

    monkeypatch.setattr(jit_monitor, "_active", True)
    monkeypatch.setattr(warmup.torch.accelerator, "synchronize", lambda: None)

    @contextmanager
    def no_launches():
        yield {}

    def skipped(*_args, **_kwargs):
        _run_self_check_sampler_branch(runner, _kwargs["sampling_params"])
        return False

    monkeypatch.setattr(warmup, "capture_topk_topp_launches", no_launches)
    monkeypatch.setattr(warmup, "run_mixed_prefill_decode_warmup", skipped)
    result = warmup.run_uno_served_jit_self_check(runner, Mock(), Mock())
    assert not result.passed
    assert not result.ran
    assert result.missing_launches["top_p_only"] == (
        "_topp_sb_stats_kernel",
        "_topp_sb_step_kernel",
        "_topp_sb_mask_kernel",
    )


def test_uno_startup_jit_self_check_restores_rng_state(monkeypatch):
    """The seed-free post-warmup check cannot perturb later sampling RNG."""
    from vllm.utils import jit_monitor
    from vllm.v1.worker.gpu import warmup

    runner = SimpleNamespace(
        speculator=object.__new__(UnoSpeculator),
        adaptive_verification="saved",
        max_num_tokens=32,
        decode_query_len=8,
        device=torch.device("cpu"),
        sampler=_self_check_sampler(),
    )
    monkeypatch.setattr(jit_monitor, "_active", True)
    monkeypatch.setattr(warmup.torch.accelerator, "synchronize", lambda: None)

    @contextmanager
    def launched_kernels():
        yield {
            "_topk_topp_kernel": 1,
            "_topp_sb_stats_kernel": 1,
            "_topp_sb_step_kernel": 5,
            "_topp_sb_mask_kernel": 1,
        }

    def sampled(*args, **_kwargs):
        torch.rand(4)
        _run_self_check_sampler_branch(runner, _kwargs["sampling_params"])
        args[2](None)
        return True

    monkeypatch.setattr(warmup, "capture_topk_topp_launches", launched_kernels)
    monkeypatch.setattr(warmup, "run_mixed_prefill_decode_warmup", sampled)
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    result = warmup.run_uno_served_jit_self_check(runner, Mock(), Mock())
    assert result.passed
    assert torch.equal(torch.random.get_rng_state(), before)


def test_uno_startup_jit_self_check_proves_flashinfer_branches(monkeypatch):
    """FlashInfer modes need branch evidence, not nonexistent Triton kernels."""
    from vllm.utils import jit_monitor
    from vllm.v1.worker.gpu import warmup

    runner = SimpleNamespace(
        speculator=object.__new__(UnoSpeculator),
        adaptive_verification=None,
        max_num_tokens=32,
        decode_query_len=8,
        device=torch.device("cpu"),
        sampler=_self_check_sampler(use_flashinfer=True),
    )
    monkeypatch.setattr(jit_monitor, "_active", True)
    monkeypatch.setattr(warmup.torch.accelerator, "synchronize", lambda: None)

    @contextmanager
    def no_triton_launches():
        yield {}

    def sampled(*args, **kwargs):
        _run_self_check_sampler_branch(runner, kwargs["sampling_params"])
        args[2](None)
        return True

    monkeypatch.setattr(warmup, "capture_topk_topp_launches", no_triton_launches)
    monkeypatch.setattr(warmup, "run_mixed_prefill_decode_warmup", sampled)

    result = warmup.run_uno_served_jit_self_check(runner, Mock(), Mock())

    assert result.passed
    assert result.sampler_branches == {
        "top_p_only": ("flashinfer",),
        "top_k_top_p": ("flashinfer",),
        "top_k_only": ("flashinfer",),
        "neither": ("triton",),
    }
    assert result.sampler_launches == {name: {} for name in result.sampler_branches}


def test_uno_self_check_launch_counter_only_records_an_armed_scope(monkeypatch):
    """The startup-only proxy counts real launch syntax, not a test signal."""
    from vllm.v1.sample.ops import topk_topp_triton
    from vllm.v1.worker.gpu.warmup import capture_topk_topp_launches

    calls: list[tuple[str, object]] = []

    class FakeTritonKernel:
        def __init__(self, name: str) -> None:
            self.name = name

        def __getitem__(self, grid: object):
            def launch(*_args, **_kwargs):
                calls.append((self.name, grid))

            return launch

    monkeypatch.setattr(
        topk_topp_triton, "_topk_topp_kernel", FakeTritonKernel("topk")
    )
    monkeypatch.setattr(
        topk_topp_triton, "_topp_sb_step_kernel", FakeTritonKernel("step")
    )
    with capture_topk_topp_launches() as launches:
        topk_topp_triton._topk_topp_kernel[(1,)]()
        topk_topp_triton._topk_topp_kernel[(2,)]()
        topk_topp_triton._topp_sb_step_kernel[(5,)]()
    assert launches == {"_topk_topp_kernel": 2, "_topp_sb_step_kernel": 1}
    assert calls == [("topk", (1,)), ("topk", (2,)), ("step", (5,))]


def test_uno_launch_key_debug_is_gated_and_has_no_generic_serving_wrapper():
    """Generic sampling stays untouched; only opt-in startup proxies record."""
    from vllm.v1.sample.ops import topk_topp_triton
    from vllm.v1.worker.gpu import launch_key_debug
    from vllm.v1.worker.gpu.sample import gumbel
    from vllm.v1.worker.gpu.spec_decode import uno_prepare

    assert "launch_key_debug" not in inspect.getsource(topk_topp_triton)
    assert "launch_key_debug" not in inspect.getsource(gumbel)
    prepare_source = inspect.getsource(uno_prepare)
    assert "_LAUNCH_KEY_DEBUG_ENABLED = _launch_key_debug_enabled()" in prepare_source
    assert "if record_triton_launch is not None:" in prepare_source
    assert "record_topk_topp_launches" in inspect.getsource(launch_key_debug)


def test_uno_step_timing_debug_flag_is_launch_only_and_tracks_three_steps():
    """The TTFT tracer never reads request data as configuration."""
    from vllm.v1.core.sched.output import parse_uno_step_timing_debug
    from vllm.v1.worker.gpu.uno_step_timing import UnoStepTimingTracer

    assert not parse_uno_step_timing_debug(None)
    assert not parse_uno_step_timing_debug("0")
    assert parse_uno_step_timing_debug("1")
    with pytest.raises(ValueError, match="VLLM_UNO_STEP_TIMING_DEBUG"):
        parse_uno_step_timing_debug("enabled")

    tracer = UnoStepTimingTracer(capture_cuda_events=False)
    output = SimpleNamespace(
        debug_uno_step_id=7,
        debug_schedule_wall_ms=0.5,
        num_scheduled_tokens={"internal-request": 1},
        skip_speculator_proposal=False,
        zero_next_draft_req_ids=set(),
        scheduled_spec_decode_tokens={},
    )
    assert tracer.begin(output, running_count=1) is not None
    assert tracer.begin(output, running_count=1) is not None
    assert tracer.begin(output, running_count=1) is not None
    assert tracer.begin(output, running_count=1) is None


def test_uno_launch_key_debug_records_sampler_launches_only_in_startup_scope(
    monkeypatch,
):
    """The opt-in warmup proxy emits a receipt from each real launch object."""
    from vllm.v1.sample.ops import topk_topp_triton
    from vllm.v1.worker.gpu import launch_key_debug

    calls: list[tuple[str, tuple[int, ...]]] = []
    records: list[tuple[str, object]] = []

    class FakeTritonKernel:
        def __getitem__(self, grid):
            def launch(*_args, **_kwargs):
                calls.append(("launch", grid))

            return launch

    monkeypatch.setattr(topk_topp_triton, "_topk_topp_kernel", FakeTritonKernel())
    monkeypatch.setattr(launch_key_debug, "_ENABLED", True)
    monkeypatch.setattr(
        launch_key_debug,
        "record_triton_launch",
        lambda name, _kernel, grid, *_args, **_kwargs: records.append((name, grid)),
    )

    with launch_key_debug.record_topk_topp_launches():
        topk_topp_triton._topk_topp_kernel[(3,)]()

    assert records == [("_topk_topp_kernel", (3,))]
    assert calls == [("launch", (3,))]


def _uno_sample_tokens_runner(monkeypatch, num_reqs=1):
    """Build CPU collaborators around the production sample_tokens entry."""
    from vllm.v1.worker.gpu import model_runner as model_runner_module
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    events: list[str] = []

    class FakeAsyncOutput:
        def __init__(self, **kwargs) -> None:
            self.model_runner_output = kwargs["model_runner_output"]
            events.append("output")

    monkeypatch.setattr(model_runner_module, "AsyncOutput", FakeAsyncOutput)
    monkeypatch.setattr(
        model_runner_module.pcp,
        "maybe_restore_pcp_for_sampling",
        lambda _manager, hidden_states, input_batch: (hidden_states, input_batch),
    )
    monkeypatch.setattr(
        model_runner_module,
        "use_workspace_lane",
        lambda _lane: nullcontext(),
    )

    runner = object.__new__(GPUModelRunner)
    input_batch = SimpleNamespace(
        req_ids=[f"request-{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_draft_tokens=0,
        logits_indices=torch.arange(num_reqs),
        idx_mapping=torch.arange(num_reqs),
        query_start_loc=torch.arange(num_reqs + 1),
    )
    target_hidden = torch.tensor([[1.0, 2.0]]).repeat(num_reqs, 1)
    runner.execute_model_state = SimpleNamespace(
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings_by_layer={},
        hidden_states=target_hidden,
        aux_hidden_states=None,
        dp_sync=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        routed_experts=None,
        cudagraph_stats=None,
        skip_speculator_proposal=False,
        zero_next_draft_req_ids=frozenset(),
        uno_step_trace=None,
    )
    runner.is_last_pp_rank = True
    runner.pcp_manager = None
    runner.pp_handler = None
    runner.check_ep_fault = False
    runner.main_stream = object()
    runner.output_copy_stream = object()
    runner.eplb = SimpleNamespace(step=lambda **_kwargs: None)
    runner._draft_workspace_lane = 0
    runner.model = SimpleNamespace(compute_logits=object())
    runner.model_state = SimpleNamespace()
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *_args: {}
    )
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.ones(num_reqs)),
            seeds=SimpleNamespace(gpu=torch.zeros(num_reqs, dtype=torch.int64)),
        )
    )
    runner.req_states = SimpleNamespace(
        draft_tokens=torch.zeros((num_reqs, 2), dtype=torch.int64),
        last_sampled_tokens=torch.zeros((num_reqs, 1), dtype=torch.int64),
        next_prefill_tokens=torch.zeros((num_reqs, 1), dtype=torch.int64),
        all_token_ids=SimpleNamespace(
            gpu=torch.zeros((num_reqs, 2), dtype=torch.int64)
        ),
        num_computed_tokens=SimpleNamespace(
            gpu=torch.zeros(num_reqs, dtype=torch.int64)
        ),
        prompt_len=SimpleNamespace(np=torch.ones(num_reqs, dtype=torch.int64).numpy()),
    )
    sampler_output = SimpleNamespace(
        sampled_token_ids=torch.full((num_reqs, 1), 7),
        num_sampled=torch.ones(num_reqs, dtype=torch.int32),
        num_rejected=torch.zeros(num_reqs, dtype=torch.int32),
    )
    runner.sample = lambda *_args: (
        sampler_output,
        sampler_output.num_sampled,
        sampler_output.num_rejected,
    )
    runner.postprocess_sampled = lambda *_args: events.append("postprocess")
    runner.num_speculative_steps = 2
    runner.draft_tokens_handler = SimpleNamespace(
        set_draft_tokens=lambda _batch, _tokens: events.append("publish"),
        get_draft_tokens=lambda: "drafts",
    )
    runner.kv_connector = SimpleNamespace(
        post_forward=lambda _finished: events.append("kv")
    )
    runner.adaptive_verification = None

    proposer = object.__new__(UnoSpeculator)
    proposer.supports_mm_inputs = False
    proposer.draft_token_confidence_probs = None

    proposer._step = 7

    def propose(*args, **kwargs):
        assert args[3] is target_hidden
        assert kwargs == {"dp_sync": None, "mm_inputs": None}
        events.append(f"propose-step-{proposer._step}")
        proposer._step += 1
        return torch.tensor([[8, 9]], dtype=torch.int64).repeat(num_reqs, 1)

    proposer.propose = propose
    runner.speculator = proposer
    return runner, proposer, events, FakeAsyncOutput


@pytest.mark.parametrize("skip_proposal", [False, True])
def test_uno_prefill_preserves_base_output_before_proposal_order(
    monkeypatch, skip_proposal
):
    """D2H is issued before postprocessing, with or without a tail proposal."""
    runner, proposer, events, output_type = _uno_sample_tokens_runner(monkeypatch)
    runner.execute_model_state.skip_speculator_proposal = skip_proposal
    if skip_proposal:
        runner.execute_model_state.zero_next_draft_req_ids = frozenset({"request-0"})

    output = runner.sample_tokens(None)

    assert isinstance(output, output_type)
    proposal_events = [] if skip_proposal else ["propose-step-7"]
    assert events == ["output", "postprocess", *proposal_events, "publish", "kv"]
    # Simulate the step mutation formerly caused by a DP dummy run. It is
    # necessarily after the proposal now, rather than between defer and flush.
    proposer._step += 1
    assert proposer._step == (8 if skip_proposal else 9)
    assert not hasattr(runner, "_pending_uno_proposal")
    expected_drafts = [[0, 0]] if skip_proposal else [[8, 9]]
    assert runner.req_states.draft_tokens.tolist() == expected_drafts


@pytest.mark.skip_global_cleanup
def test_launch_key_receipt_compares_first_served_key_to_all_warmup_keys():
    """The side-by-side record retains the actual warmup key set, not a sample."""
    receipts = _LaunchKeyReceipts()
    warmup_one = {
        "kernel": "_prepare_uno_inputs_kernel",
        "triton_key": "warmup-one",
    }
    warmup_two = {
        "kernel": "_prepare_uno_inputs_kernel",
        "triton_key": "warmup-two",
    }

    assert receipts.add("warmup", warmup_one) == {"record": warmup_one}
    assert receipts.add("warmup", warmup_two) == {"record": warmup_two}
    assert receipts.add("warmup", warmup_one) is None

    served = {
        "kernel": "_prepare_uno_inputs_kernel",
        "triton_key": "warmup-two",
    }
    emitted = receipts.add("served", served)

    assert emitted is not None
    comparison = emitted["compare"]
    assert comparison["served"] is served
    assert comparison["served_key_was_warmed"]
    assert [item["triton_key"] for item in comparison["warmup"]] == [
        "warmup-one",
        "warmup-two",
    ]
    # Only the first actual serving record is the diagnostic comparison.
    assert (
        receipts.add(
            "served",
            {"kernel": "_prepare_uno_inputs_kernel", "triton_key": "another"},
        )
        is None
    )


@pytest.mark.skip_global_cleanup
def test_launch_key_receipt_serializes_slots_backed_options():
    """Triton 3.8 options use slots, not ``__dict__``."""

    class Options:
        __slots__ = ("num_stages", "num_warps")

        def __init__(self) -> None:
            self.num_stages = 3
            self.num_warps = 4

    assert _object_fields(Options()) == {"num_stages": 3, "num_warps": 4}


@pytest.mark.skip_global_cleanup
def test_launch_key_receipt_says_when_served_key_was_not_warmed():
    receipts = _LaunchKeyReceipts()
    receipts.add(
        "warmup",
        {"kernel": "_prepare_uno_inputs_kernel", "triton_key": "warmup-only"},
    )

    emitted = receipts.add(
        "served",
        {"kernel": "_prepare_uno_inputs_kernel", "triton_key": "served-only"},
    )

    assert emitted is not None
    assert not emitted["compare"]["served_key_was_warmed"]


def test_attention_config_leaves_the_version_to_the_platform():
    """Unset means the platform chooses, which is the whole point of round 21.

    Both e2e engines pinned FlashAttention 2, so the module could never
    exercise FA3: the platform prefers FA3 on SM90 only while the version is
    unset, and an explicit 2 overrides that preference. Uno's Hopper support
    was therefore never covered by Uno's own tests. The backend stays pinned,
    because a test that silently ran on another backend would be measuring
    something else.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    assert budget.resolve_attention_config(None) == {"backend": "FLASH_ATTN"}
    assert "flash_attn_version" not in budget.resolve_attention_config(None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2", 2), ("3", 3), ("  3  ", 3), ("\t2\n", 2)],
)
def test_attention_config_honours_an_explicit_pin(raw, expected):
    """A reproducibility study can pin the version back, whitespace and all."""
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    config = budget.resolve_attention_config(raw)
    assert config == {"backend": "FLASH_ATTN", "flash_attn_version": expected}


@pytest.mark.parametrize("raw", ["", "   ", "4", "fa3", "2.0", "23", "-2", "0"])
def test_attention_config_refuses_what_it_cannot_honour(raw):
    """A value this module cannot honour fails loudly, not silently.

    Accepting 4 would be the dangerous case: it is a different kernel family
    with its own downgrade rules, so a run asking for it and quietly getting
    the platform default would report a version it never exercised.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    with pytest.raises(ValueError, match="must be '2' or '3'"):
        budget.resolve_attention_config(raw)


def test_attention_receipt_names_the_built_version_not_the_request():
    """The receipt reports what the engine built, and says what was asked.

    With the version unset the request is silent and the platform decides, so
    a receipt that echoed the request would say nothing at all on exactly the
    runs this change exists to observe.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    hopper = {"backends": ["FlashAttentionImpl"], "flash_attn_versions": [3]}
    line = budget.format_attention_receipt(hopper, None)
    assert "FlashAttention version 3" in line
    assert "platform default" in line
    assert "FlashAttentionImpl" in line

    pinned = budget.format_attention_receipt(
        {"backends": ["FlashAttentionImpl"], "flash_attn_versions": [2]}, 2
    )
    assert "FlashAttention version 2" in pinned and "pinned 2" in pinned


def test_attention_receipt_reports_silence_and_disagreement():
    """Neither an empty reading nor a mixed build is reported as a version.

    A build whose layers resolved different versions is a finding, and one
    that reported none means the probe read the wrong object. Both would be
    hidden by taking the first entry.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    silent = budget.format_attention_receipt({}, None)
    assert "unreported" in silent

    mixed = budget.format_attention_receipt(
        {"backends": ["FlashAttentionImpl"], "flash_attn_versions": [2, 3]}, None
    )
    assert "layers disagree" in mixed and "[2, 3]" in mixed


def test_survivor_receipt_names_the_attention_kernel():
    """The survivor gate is forked, so this can only live in the receipt.

    A printed line never reaches the log or the JUnit attachment from a forked
    child, which is the failure the round-10 receipt work closed. The kernel
    the gate validated belongs with the rest of the evidence.
    """
    from tests.v1.e2e.spec_decode.test_uno import _MixedPhase, _render_receipt

    phase = _MixedPhase()
    assert phase.attention == "unmeasured"
    assert "attention=unmeasured" in _render_receipt(phase, "geometry", "metrics")

    phase.attention = "FlashAttention version 3 (requested: platform default)"
    rendered = _render_receipt(phase, "geometry", "metrics")
    assert "attention=FlashAttention version 3" in rendered


@pytest.mark.parametrize(
    ("k", "expected_active_loras"),
    [(1, 0), (8, 2)],
)
def test_draft_graph_engagement_follows_the_k_rule(
    k, expected_active_loras, monkeypatch
):
    """Draft graph engagement is derived, not assumed from K.

    The capture case follows ``2 if k > 1 else 0`` (adapter noise rows exist
    only for k > 1), but whether a captured decode graph can serve the K-row
    batch is the separate ``CudaGraphManager._init_candidates`` rule: a
    candidate is skipped when ``round_up(num_tokens, decode_query_len)``
    exceeds ``max_num_reqs * decode_query_len``, so a graph exists only when
    some ``cudagraph_capture_sizes`` entry is at most ``max_num_seqs * k``. At
    K=1 here that is ``min([8, 16, 32, 64]) > 4``, so drafting is eager. The
    expectation is computed from that arithmetic so a capture-size or
    max-num-seqs change moves the assertion instead of breaking it, while a
    future change to the ``2 if k > 1 else 0`` rule still fails on CPU.
    """
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils

    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils.current_platform,
        "get_global_graph_pool",
        lambda: object(),
    )
    monkeypatch.setattr(gpu_cudagraph_utils, "get_offloader", lambda: Mock())

    module = "vllm.v1.worker.gpu.spec_decode.uno"
    proposer = _cpu_uno_proposer(k)
    proposer.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    manager = proposer.cudagraph_manager
    assert manager is not None
    assert manager.lora_capture_cases == [expected_active_loras], (
        f"K={k} must capture draft graphs for num_active_loras="
        f"{expected_active_loras}, got {manager.lora_capture_cases}"
    )

    capture_sizes = proposer.vllm_config.compilation_config.cudagraph_capture_sizes
    max_num_seqs = proposer.vllm_config.scheduler_config.max_num_seqs
    expect_graph = min(capture_sizes) <= max_num_seqs * k

    if expect_graph:
        manager._graphs_captured = True
        desc = manager.dispatch(4, 4 * k, k, expected_active_loras)
        assert desc.cg_mode == CUDAGraphMode.FULL, desc
        proposer._graph_attn_metadata[desc] = {"layer": object()}
        manager.graphs[desc] = Mock()
    else:
        assert not manager.needs_capture(), manager._capture_descs

    dispatched_loras: list[int] = []
    real_dispatch = manager.dispatch

    def recording_dispatch(*args, **kwargs):
        dispatched_loras.append(args[3])
        return real_dispatch(*args, **kwargs)

    manager.dispatch = recording_dispatch
    monkeypatch.setattr(f"{module}.prepare_uno_inputs_fused", Mock())
    monkeypatch.setattr(f"{module}.build_slot_mappings_by_layer", Mock(return_value={}))

    batch = SimpleNamespace(
        num_reqs=4,
        idx_mapping=torch.arange(4),
        seq_lens_cpu_upper_bound=torch.tensor([10, 20, 30, 40]),
    )
    tensor = torch.empty(4)
    proposer.propose(
        batch, {}, {}, tensor, None, tensor, tensor, tensor, tensor, tensor, tensor
    )

    assert dispatched_loras == [expected_active_loras], (
        f"K={k} must dispatch num_active_loras={expected_active_loras}, "
        f"got {dispatched_loras}"
    )
    if expect_graph:
        assert proposer.num_graph_replays == 1
        assert proposer.num_eager_proposals == 0
    else:
        assert proposer.num_eager_proposals == 1
        assert proposer.num_graph_replays == 0


@pytest.mark.parametrize("supports_update", [False, True])
def test_uno_rejects_builder_without_native_decode_update_at_initialization(
    monkeypatch, supports_update
):
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    proposer = object.__new__(UnoSpeculator)
    group = SimpleNamespace(supports_draft_decode_metadata_update=supports_update)

    def install_groups(self, *args):
        self.attn_groups = [[group]]

    monkeypatch.setattr(DraftModelSpeculator, "set_attn", install_groups)
    cache = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=FullAttentionSpec(
                    block_size=4,
                    num_kv_heads=1,
                    head_size=16,
                    dtype=torch.bfloat16,
                )
            )
        ]
    )
    if supports_update:
        proposer.set_attn(None, cache, None, None, [])
        assert proposer.attn_groups == [[group]]
    else:
        with pytest.raises(ValueError, match="native draft decode updates"):
            proposer.set_attn(None, cache, None, None, [])


def test_survivor_kv_budget_clears_the_engine_admission_floor():
    """The e2e survivor budget must clear vLLM's single-request floor.

    A `kv_cache_memory_bytes` below the floor makes `get_kv_cache_configs`
    raise before the engine starts, which is exactly how the survivor e2e test
    failed on the GB10. The floor here is the engine's own admission rule fed a
    full-attention spec built from the pinned Qwen3-8B geometry, so a change to
    the model pin, block size or context fails on CPU instead of only on a GPU.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget
    from vllm.v1.core.kv_cache_utils import check_enough_kv_cache_memory
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    block = uno_kv_budget.kv_bytes_per_block()

    def max_len_config(max_model_len: int):
        # The admission path under test reads only these fields; a real
        # ModelConfig here would resolve a default model over the hub.
        return SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=max_model_len),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=None),
        )

    def qwen3_specs() -> dict:
        num_layers, num_kv_heads, head_dim = uno_kv_budget.qwen3_geometry()
        spec = FullAttentionSpec(
            block_size=uno_kv_budget.BLOCK_SIZE,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=torch.bfloat16,
        )
        return {f"model.layers.{index}.self_attn": spec for index in range(num_layers)}

    # Both e2e budgets sit at or above the floor and the engine admits them.
    for max_model_len, budget in (
        (uno_kv_budget.SURVIVOR_MAX_MODEL_LEN, uno_kv_budget.survivor_kv_budget()),
        (uno_kv_budget.MATRIX_MAX_MODEL_LEN, uno_kv_budget.MATRIX_KV_BUDGET_BYTES),
    ):
        floor = uno_kv_budget.engine_minimum_kv_bytes(max_model_len)
        assert budget >= floor, (
            f"KV budget {budget} B ({budget // block} blocks) is below the engine "
            f"floor {floor} B ({floor // block} blocks) at "
            f"max_model_len={max_model_len}"
        )
        check_enough_kv_cache_memory(
            max_len_config(max_model_len), qwen3_specs(), budget
        )

    # fix1 shipped 64 blocks at max_model_len=1024, one below that floor; the
    # engine refuses it. The inverted run mutates the budget helper to this
    # value and records exit 1.
    fix1_budget = 64 * block
    survivor_floor = uno_kv_budget.engine_minimum_kv_bytes(
        uno_kv_budget.SURVIVOR_MAX_MODEL_LEN
    )
    assert fix1_budget < survivor_floor, (
        "the fix1 64-block budget unexpectedly clears the engine floor; the "
        "inverted run no longer proves the guard fires"
    )
    with pytest.raises(ValueError, match="To serve at least one request"):
        check_enough_kv_cache_memory(
            max_len_config(uno_kv_budget.SURVIVOR_MAX_MODEL_LEN),
            qwen3_specs(),
            fix1_budget,
        )


def test_survivor_prompts_require_content_difference_at_shared_positions():
    """Length-only prompt changes must not masquerade as distinct peers."""
    from tests.v1.e2e.spec_decode.uno_kv_budget import (
        prompt_token_ids_are_pairwise_content_distinct,
    )

    assert prompt_token_ids_are_pairwise_content_distinct([[1, 2], [1, 3]])
    assert not prompt_token_ids_are_pairwise_content_distinct([[1, 2], [1, 2]])
    assert not prompt_token_ids_are_pairwise_content_distinct([[1, 2], [1, 2, 3]])


def test_survivor_scheduler_requires_v1_inprocess_mode():
    """Scheduler receipts fail clearly when V1 multiprocessing hides the core."""
    from tests.v1.e2e.spec_decode.test_uno import _scheduler

    scheduler = object()
    inprocess_engine = SimpleNamespace(
        engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=scheduler))
    )
    assert _scheduler(inprocess_engine) is scheduler

    multiprocess_client = type("SyncMPClient", (), {})()
    multiprocess_engine = SimpleNamespace(engine_core=multiprocess_client)
    with pytest.raises(AssertionError, match="VLLM_ENABLE_V1_MULTIPROCESSING=0"):
        _scheduler(multiprocess_engine)


def test_survivor_preemption_arithmetic_fits_then_overflows_the_budget():
    """The survivor window admits four prompts but their growth exceeds it.

    The e2e test derives these token counts from the tokenizer at runtime; here
    they are pinned to the same prompt shapes at the pinned revision, so the
    preemption geometry is provable on CPU. The finish-peer cap is read from
    ``uno_kv_budget.SURVIVOR_FINISH_MAX_TOKENS`` so the CPU pin and the e2e
    cannot drift. Prefix caching is disabled in this phase, so every request's
    complete prompt-plus-generation footprint is counted, including the K
    lookahead slots the allocator reserves.

    The compared request is one of the two long peers, so the crossing point
    must sit below the cap: above it a peer could finish naturally before
    preemption and the resume path would never run. Both crossings are pinned:
    the pair growing together, and the worst case where one peer stalls at its
    admission footprint while the other runs to its cap. An inverted run with a
    cap below either crossing fails these assertions, and
    ``tests/v1/spec_decode/test_uno_preemption.py`` drives the real scheduler
    over the same geometry to prove the arithmetic matches the allocator.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    shared_prefix_tokens = 161
    # seed, finish-0, finish-1, abort. The e2e asserts its runtime tokenisation
    # against this same constant, so a drift fails on the GPU with a message
    # that names the literal instead of moving the geometry underneath it.
    prompt_tokens = list(budget.SURVIVOR_PROMPT_TOKENS)
    max_tokens = [
        96,
        budget.SURVIVOR_FINISH_MAX_TOKENS,
        budget.SURVIVOR_FINISH_MAX_TOKENS,
        64,
    ]
    budget_blocks = budget.survivor_kv_budget() // budget.kv_bytes_per_block()
    # The pool hands out one block fewer than the pinned override (its null
    # block), and `get_kv_cache_usage` divides by the same number, so every
    # inequality below is stated against the allocatable count.
    pool_blocks = budget.allocatable_blocks(budget_blocks)
    assert pool_blocks == budget_blocks - 1

    admission = budget.mixed_admission_blocks(
        prompt_tokens, shared_prefix_tokens, prefix_cache_enabled=False
    )
    growth = budget.mixed_growth_blocks(
        prompt_tokens,
        max_tokens,
        shared_prefix_tokens,
        prefix_cache_enabled=False,
    )
    crossing = budget.mixed_crossing_tokens(
        prompt_tokens[1],
        shared_prefix_tokens,
        pool_blocks,
        prefix_cache_enabled=False,
    )
    worst_case_crossing = budget.worst_case_crossing_tokens(
        prompt_tokens[1], [prompt_tokens[2]], pool_blocks
    )

    assert admission < pool_blocks, (
        f"all four prompts must be admitted together: {admission} blocks of "
        f"unique prompt footprint (incl. one decode block each) vs pool "
        f"{pool_blocks}"
    )
    assert growth > pool_blocks, (
        f"the mixed phase must exhaust the pool by growth: {growth} blocks if "
        f"every request reached its cap vs pool {pool_blocks}"
    )
    assert crossing < budget.SURVIVOR_FINISH_MAX_TOKENS, (
        f"the two long peers together cross the pool only at {crossing} "
        f"generated tokens, at or past the {budget.SURVIVOR_FINISH_MAX_TOKENS} "
        "cap, so a peer could finish before the scheduler preempts it"
    )
    # The crossing must not require the peers to grow at the same rate: Uno's
    # acceptance is prompt-dependent, and a peer that outruns its twin by
    # enough blocks makes a grow-together budget uncrossable. This is the
    # assertion that fails for the former 81-block/512-token configuration,
    # which skipped on two cards with every prompt distinct.
    assert worst_case_crossing < budget.SURVIVOR_FINISH_MAX_TOKENS, (
        f"one long peer plus the other's admission footprint crosses the "
        f"{pool_blocks}-block pool only at {worst_case_crossing} generated "
        f"tokens, at or past the {budget.SURVIVOR_FINISH_MAX_TOKENS} cap, so a "
        "peer that outruns its twin can finish inside the pool and the resume "
        "path is never exercised"
    )
    assert all(
        tokens + cap < budget.SURVIVOR_MAX_MODEL_LEN
        for tokens, cap in zip(prompt_tokens, max_tokens)
    ), (prompt_tokens, max_tokens)

    # The gate must be falsifiable at THIS geometry, and the configuration it
    # replaced is the negative control: an 81-block pool (80 allocatable) with a
    # 512-token cap cannot force a crossing once the peers drift apart, because
    # a leader at its cap plus a stalled twin is 67 of 80 blocks. The worst-case
    # helper must refuse it, and the two GPU receipts that skipped on it are the
    # field evidence.
    round8_pool = budget.allocatable_blocks(81)
    round8_worst_case = budget.worst_case_crossing_tokens(
        prompt_tokens[1], [prompt_tokens[2]], round8_pool
    )
    assert round8_worst_case > 512, (
        "the replaced 81-block/512-token configuration would now pass the "
        f"worst-case pre-gate ({round8_worst_case} tokens vs a 512 cap), so "
        "this gate no longer rejects the geometry that skipped on two cards"
    )


def test_survivor_resident_blocks_track_the_speculative_width():
    """The resident-block arithmetic must move when K moves.

    Uno's ``num_lookahead_tokens`` is K, so a running request holds its
    committed tokens plus the sampled token plus K. Re-deriving that as a bare
    ``+1`` made the worst-case pre-gate optimistic and left it silently stale
    for any other K, which the same test matrix uses (K=1 in the greedy rows).
    One hand computation, and one pair that must differ.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    # cdiv(257 + 0 + 1 + 8, 16) = cdiv(266, 16) = 17 blocks held by a peer that
    # has been admitted and has generated nothing.
    assert budget.min_resident_blocks(257) == 17
    # cdiv(257 + 0 + 1 + 16, 16) = cdiv(274, 16) = 18: K is really read.
    assert budget.min_resident_blocks(257, 16) == 18
    # cdiv(257 + 640 + 1 + 8, 16) = cdiv(906, 16) = 57 blocks at the cap, which
    # is what a solo peer occupied on both GPU receipts (83.8% of 68).
    assert budget.resident_blocks(257, budget.SURVIVOR_FINISH_MAX_TOKENS) == 57

    budget_blocks = budget.survivor_kv_budget() // budget.kv_bytes_per_block()
    pool_blocks = budget.allocatable_blocks(budget_blocks)
    # Worst case by hand: the laggard holds 17, so the leader must reach
    # 68 - 17 + 1 = 52 blocks, i.e. more than 51 * 16 = 816 slots, i.e.
    # 257 + t + 9 > 816 -> t >= 551.
    assert budget.worst_case_crossing_tokens(257, [257], pool_blocks) == 551
    at_k1 = budget.worst_case_crossing_tokens(257, [257], pool_blocks, 1)
    assert at_k1 != 551, (
        "the worst-case crossing did not move with K, so the pre-gate is not "
        f"reading the speculative width: K=1 gives {at_k1}"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("0.35", 0.35), (" 1 ", 1.0)],
)
def test_survivor_memory_override_accepts_a_fraction_or_nothing(raw, expected):
    """Unset means the engine default; a fraction in (0, 1] is honoured."""
    from tests.v1.e2e.spec_decode.test_uno import _gpu_memory_utilization_from_env

    assert _gpu_memory_utilization_from_env(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "high", "0", "1.5", "-0.2"])
def test_survivor_memory_override_refuses_what_the_engine_would(raw):
    """A blank or out-of-range value must name the variable, not float('').

    The lane's own runbook exports this variable, and a shell that expands it to
    nothing used to reach ``float('')`` and abort the run with a ValueError
    mentioning neither the variable nor how to fix it.
    """
    from tests.v1.e2e.spec_decode.test_uno import _gpu_memory_utilization_from_env

    with pytest.raises(ValueError, match="VLLM_UNO_SURVIVOR_GPU_MEMORY_UTILIZATION"):
        _gpu_memory_utilization_from_env(raw)


def test_survivor_receipt_is_written_where_the_evidence_lives(tmp_path, monkeypatch):
    """The receipt must not depend on a forked child's stdout.

    Two GPU runs lost every printed receipt line because the survivor case is
    ``pytest.mark.forked``. The receipt now goes to the path the environment
    names, else beside the JUnit file the run was invoked with, which is what a
    lease lane commits.
    """
    from tests.v1.e2e.spec_decode.test_uno import _write_receipt

    named = tmp_path / "named" / "receipt.txt"
    monkeypatch.setenv("VLLM_UNO_SURVIVOR_RECEIPT", str(named))
    request = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace()))
    assert _write_receipt("hello", request) == str(named)
    assert named.read_text(encoding="utf-8") == "hello"

    monkeypatch.delenv("VLLM_UNO_SURVIVOR_RECEIPT")
    xml = tmp_path / "survivor.xml"
    request = SimpleNamespace(
        config=SimpleNamespace(option=SimpleNamespace(xmlpath=str(xml)))
    )
    written = _write_receipt("beside the junit", request)
    assert written == str(tmp_path / "survivor-survivor-receipt.txt")
    assert Path(written).read_text(encoding="utf-8") == "beside the junit"

    request = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace()))
    assert _write_receipt("nowhere", request) is None


def test_token_agreement_counts_prompts_and_locates_the_first_divergence():
    """The matrix receipt's coordinates must match the lane receipts' (p/t)."""
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    matched, divergences = budget.token_agreement([[1, 2, 3]], [[1, 2, 3]])
    assert (matched, divergences) == (1, [])

    # Prompt 2 differing at token 31 is the exact shape the Ampere control
    # reported for plain-vs-plain graph mode.
    reference = [[0] * 64 for _ in range(4)]
    candidate = [list(row) for row in reference]
    candidate[2][31] = 7
    matched, divergences = budget.token_agreement(reference, candidate)
    assert (matched, divergences) == (3, [(2, 31)])
    assert budget.format_divergences(divergences) == ["p2/t31"]

    # A truncated output diverges at the length of the shorter sequence.
    matched, divergences = budget.token_agreement([[1, 2, 3]], [[1, 2]])
    assert (matched, divergences) == (0, [(0, 2)])

    # A first-token difference is t0, not a falsy value that reads as "no
    # divergence" anywhere downstream.
    matched, divergences = budget.token_agreement([[1, 2]], [[9, 2]])
    assert (matched, divergences) == (0, [(0, 0)])
    assert budget.format_divergences(divergences) == ["p0/t0"]

    with pytest.raises(AssertionError):
        budget.token_agreement([[1]], [[1], [2]])


@pytest.mark.parametrize(
    ("control", "candidate", "expected"),
    [
        # Deterministic regime: an empty control admits only an empty candidate.
        ([], [], True),
        ([], [(0, 5)], False),
        # Non-deterministic regime (sm_86 graph mode): the control diverges at
        # prompt 2, so Uno may diverge there (at any token) and nowhere else.
        ([(2, 31)], [(2, 31)], True),
        ([(2, 31)], [(2, 55)], True),
        ([(2, 31)], [], True),
        ([(2, 31)], [(0, 12)], False),
        # The count comparison this replaced passed exactly this row: three
        # matches each, but Uno broke a prompt the control reproduced.
        ([(2, 31)], [(0, 12), (2, 31)], False),
        # Terra's coordinates for the same hazard, with the divergence at the
        # very first token, which also pins that token_agreement reports t0.
        ([(2, 31)], [(0, 0)], False),
    ],
)
def test_exact_token_verdict_contains_divergences_per_prompt(
    control, candidate, expected
):
    """Uno may only diverge where the plain engine already diverges.

    Comparing counts let a candidate that broke a prompt the control reproduced
    pass whenever the control happened to diverge somewhere else; containment is
    per prompt, so that row now fails while a candidate that merely reproduces
    the control's own unreliability still passes.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    ok, reason = budget.exact_token_verdict(control, candidate, 4)
    assert ok is expected, reason
    assert "4 prompts" in reason
    if not ok:
        assert "reproduced exactly" in reason


def test_text_floor_is_containment_not_a_subtracted_count():
    """A count of coordinates is not a count of prompts, and can go negative.

    The control is a union over arms, so one prompt can contribute several
    coordinates: two arms diverging at prompt 2 tokens 31 and 55 give two
    coordinates for one prompt. Subtracting that from the prompt count
    over-credits the control, and with enough arms drops below zero, which the
    old threshold helper rejects outright -- failing an exact candidate on a
    card whose control merely wobbled twice inside one prompt.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    control = [(2, 31), (2, 55)]
    assert len(control) == 2 and len({prompt for prompt, _ in control}) == 1
    # The arithmetic this replaced: 4 - 2 = 2, against 3 true clean prompts.
    assert len(control) != len({prompt for prompt, _ in control})

    control_prompts = [prompt for prompt, _ in control]
    # An exact candidate passes, whatever the control's coordinate count.
    ok, reason = budget.text_verdict(control_prompts, [], 4)
    assert ok, reason
    # A candidate that only reproduces the control's own wobble passes.
    ok, _ = budget.text_verdict(control_prompts, [2], 4)
    assert ok
    # One that breaks a prompt the control reproduced does not.
    ok, reason = budget.text_verdict(control_prompts, [0], 4)
    assert not ok and "[0]" in reason

    # Five arms wobbling inside one prompt used to give 4 - 5 = -1.
    many = [(2, token) for token in (3, 11, 31, 55, 60)]
    ok, _ = budget.text_verdict([p for p, _ in many], [], 4)
    assert ok


def test_text_agreement_reports_prompt_indices():
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    matched, divergent = budget.text_agreement(["a", "b"], ["a", "b"])
    assert (matched, divergent) == (2, [])
    matched, divergent = budget.text_agreement(["a", "b", "c"], ["a", "x", "c"])
    assert (matched, divergent) == (2, [1])
    with pytest.raises(AssertionError):
        budget.text_agreement(["a"], ["a", "b"])


def _arm(divergences, token_ids, text_prompts=()):
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    return budget.MatrixArm(divergences, token_ids, list(text_prompts))


def _four_prompts(diverge_at=None, replacement=None):
    """Four short outputs; optionally one prompt diverges at one position."""
    ids = [[10, 11, 12, 13] for _ in range(4)]
    if diverge_at is not None:
        prompt, position = diverge_at
        ids[prompt] = list(ids[prompt])
        ids[prompt][position] = replacement
    return ids


def test_matrix_verdicts_hold_every_candidate_to_one_floor():
    """Both matrix arms are judged against the same completed control.

    The adapter-disabled arm is collected inside the Uno engine's context,
    before the separate-engine control has joined the floor, and used to be
    judged there as well -- a weaker floor than the Uno comparison got, under a
    comment promising the same one. Routing every candidate for a batch through
    one call is that promise in code: if either candidate were handed an empty
    floor it would fail here, since both diverge only where the control does.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    control = [(2, 31)]
    verdicts = budget.matrix_verdicts(
        control,
        {
            "uno batch 1": _arm([(2, 31)], _four_prompts()),
            "adapter-disabled": _arm([(2, 55)], _four_prompts()),
        },
        4,
    )
    assert set(verdicts) == {"uno batch 1", "adapter-disabled"}
    for name, verdict in verdicts.items():
        assert verdict.ok, f"{name}: {verdict.reason}"
        assert "p2/t31" in verdict.reason, f"{name} was not judged against the control"
        assert verdict.ties == []

    # And the floor still bites: a candidate outside it fails in the same call.
    verdicts = budget.matrix_verdicts(
        control,
        {
            "uno": _arm([(2, 31)], _four_prompts()),
            "adapter-disabled": _arm([(0, 0)], _four_prompts()),
        },
        4,
    )
    assert verdicts["uno"].ok and not verdicts["adapter-disabled"].ok

    with pytest.raises(TypeError):
        budget.matrix_verdicts(3, {"uno": _arm([(2, 31)], _four_prompts())}, 4)


def test_matrix_verdicts_refuse_a_bare_divergence_list():
    """An arm that cannot carry token ids cannot be judged by the tie rule.

    The verdict helper is the single place the token view, the text view and
    the near-tie exception are decided. A call site that passed coordinates
    alone would silently opt that arm out of the tie rule and out of the text
    verdict, which is the drift the one-call design exists to prevent, so the
    shape that cannot be judged raises instead of being judged partly.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    with pytest.raises(TypeError):
        budget.matrix_verdicts([(2, 31)], {"uno": [(2, 31)]}, 4)


def test_matrix_verdicts_excuse_the_reference_own_near_tie():
    """A runner-up within the margin is the card's rounding, not a divergence.

    The candidate emits the reference's second-ranked token at a gap far below
    the threshold. Nothing about that outcome distinguishes the two arms'
    arithmetic, so the containment failure it would otherwise cause is
    excused -- in the token view and in the text view together, since the
    tokens really do differ and the prompt's text differs with them.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    reference = _four_prompts()
    candidate = _four_prompts(diverge_at=(2, 3), replacement=99)
    ranked: list[list[list[tuple[int, float]]]] = [
        [[] for _ in range(4)] for _ in range(4)
    ]
    ranked[2][3] = [(13, -0.2000), (99, -0.2004)]

    verdicts = budget.matrix_verdicts(
        [],
        {"uno": _arm([(2, 3)], candidate, [2])},
        4,
        reference,
        ranked,
    )
    verdict = verdicts["uno"]
    assert verdict.ok and verdict.text_ok, verdict.reason
    assert [(tie.prompt, tie.token) for tie in verdict.ties] == [(2, 3)]
    assert verdict.ties[0].reference_id == 13
    assert verdict.ties[0].candidate_id == 99
    assert verdict.ties[0].margin == pytest.approx(0.0004, abs=1e-9)
    assert "ties excused" in verdict.reason
    assert "ties excused" in verdict.text_reason
    # The detokenisation check still reads the unfiltered divergences, so an
    # excused prompt is not then reported as text differing without tokens.
    assert verdict.detokenisation_only == []


def test_matrix_verdicts_keep_a_runner_up_at_a_wide_margin():
    """The right token rank is not enough; the gap has to be unresolvable.

    A candidate that picks the runner-up where the reference preferred it by a
    wide margin has computed something different, not rounded something
    differently. Excusing it would let the tie rule absorb real defects, which
    is the only way this rule can do harm.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    reference = _four_prompts()
    candidate = _four_prompts(diverge_at=(2, 3), replacement=99)
    ranked: list[list[list[tuple[int, float]]]] = [
        [[] for _ in range(4)] for _ in range(4)
    ]
    ranked[2][3] = [(13, -0.2), (99, -3.7)]

    verdicts = budget.matrix_verdicts(
        [], {"uno": _arm([(2, 3)], candidate, [2])}, 4, reference, ranked
    )
    verdict = verdicts["uno"]
    assert not verdict.ok and not verdict.text_ok
    assert verdict.ties == []
    assert "p2/t3" in verdict.reason


def test_matrix_verdicts_keep_a_token_outside_the_reference_top_two():
    """A small gap between the top two says nothing about a third token.

    The reference's own top two are a hair apart, but the candidate emitted
    neither of them. That is a different continuation, and the margin between
    tokens it did not choose cannot excuse it.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    reference = _four_prompts()
    candidate = _four_prompts(diverge_at=(2, 3), replacement=7)
    ranked: list[list[list[tuple[int, float]]]] = [
        [[] for _ in range(4)] for _ in range(4)
    ]
    ranked[2][3] = [(13, -0.2000), (99, -0.2001)]

    verdicts = budget.matrix_verdicts(
        [], {"uno": _arm([(2, 3)], candidate, [2])}, 4, reference, ranked
    )
    verdict = verdicts["uno"]
    assert not verdict.ok
    assert verdict.ties == []


def test_matrix_verdicts_do_not_excuse_on_mismatched_diagnostics():
    """Ranked data that disagrees with the emitted token licenses nothing.

    If the top-ranked alternative is not the token the reference actually
    produced, the ranked list does not describe this run. Excusing a divergence
    on diagnostics that do not match the run is exactly the failure mode the
    tie rule must not introduce.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    reference = _four_prompts()
    candidate = _four_prompts(diverge_at=(2, 3), replacement=99)
    ranked: list[list[list[tuple[int, float]]]] = [
        [[] for _ in range(4)] for _ in range(4)
    ]
    ranked[2][3] = [(55, -0.2000), (99, -0.2001)]

    verdicts = budget.matrix_verdicts(
        [], {"uno": _arm([(2, 3)], candidate, [2])}, 4, reference, ranked
    )
    assert not verdicts["uno"].ok
    assert verdicts["uno"].ties == []


def test_matrix_verdicts_replay_the_ampere_p2_t31_coordinate():
    """The r5 failure, replayed: a whitespace tie with an empty control floor.

    On an RTX 3090 (sm_86) in CUDA-graph mode the K=8 case failed because Uno
    diverged at prompt 2 token 31 while three in-process plain arms happened to
    agree. Round 3's fresh-process plain control showed the plain engine itself
    flipping that coordinate between token 18611 and token 2303 in one process
    of three, and round 4's token dump identifies both as whitespace. The token
    ids here are round 4's; the margin is synthetic, because no run has
    recorded one yet. With the margin small, the coordinate is excused and the
    case passes; the control stays empty, so nothing else is weakened.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    prefix = [
        18315,
        295,
        748,
        77778,
        10962,
        220,
        16,
        21,
        18805,
        817,
        1899,
        13,
        2303,
        7941,
        49677,
        220,
        18,
        18805,
        369,
        17496,
        13,
        2303,
        7941,
        5711,
        220,
        19,
        18805,
        369,
        54304,
        1330,
        13,
    ]
    reference = [[1, 2], [3, 4], prefix + [18611], [5, 6]]
    candidate = [[1, 2], [3, 4], prefix + [2303], [5, 6]]
    assert len(reference[2]) == 32 and reference[2][31] == 18611
    assert candidate[2][31] == 2303
    ranked: list[list[list[tuple[int, float]]]] = [
        [[] for _ in range(32)] for _ in range(4)
    ]
    ranked[2][31] = [(18611, -1.250000), (2303, -1.250122)]

    verdicts = budget.matrix_verdicts(
        [], {"uno": _arm([(2, 31)], candidate, [2])}, 4, reference, ranked
    )
    verdict = verdicts["uno"]
    assert verdict.ok and verdict.text_ok, verdict.reason
    assert budget.format_ties(verdict.ties) == [
        "p2/t31 ref=18611 candidate=2303 margin=0.000122"
    ]

    # The same coordinate at a wide margin is still a finding on that card.
    ranked[2][31] = [(18611, -1.25), (2303, -2.75)]
    wide = budget.matrix_verdicts(
        [], {"uno": _arm([(2, 31)], candidate, [2])}, 4, reference, ranked
    )
    assert not wide["uno"].ok


def test_tie_margin_default_is_the_bfloat16_resolution():
    """The default threshold is a stated quantity, not a tuned one.

    bfloat16 carries 8 significand bits, so 2**-8 is its relative resolution
    and the scale below which a reduction-order change cannot be distinguished.
    It is a stand-in rather than a derivation, which is why every divergence
    prints its measured margin and the environment can override the default
    once a card has reported real numbers.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    assert budget.TIE_MARGIN_NATS == 2.0**-8
    assert pytest.approx(0.00390625) == budget.TIE_MARGIN_NATS


def test_classify_ties_without_logprobs_changes_nothing():
    """A run that requested no alternatives behaves as it did before the rule.

    The tie rule is an exception granted on evidence. With no ranked data there
    is no evidence, so every divergence stays a divergence rather than being
    excused by default.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    reference = _four_prompts()
    candidate = _four_prompts(diverge_at=(2, 3), replacement=99)
    real, ties = budget.classify_ties([(2, 3)], reference, candidate, None)
    assert real == [(2, 3)] and ties == []

    # Short ranked data classifies nothing either, rather than raising.
    real, ties = budget.classify_ties([(2, 3)], reference, candidate, [[], [], [], []])
    assert real == [(2, 3)] and ties == []

    with pytest.raises(TypeError):
        budget.classify_ties(3, reference, candidate, None)


def test_exact_token_verdict_refuses_match_counts():
    """Counts must not reach the verdict from any call site.

    Both matrix call sites -- the Uno comparison and the adapter-disabled arm --
    pass divergence lists, and this is the structural guard that keeps it that
    way: the count shape the verdict used to take now raises rather than
    silently answering the wrong question, so a call site that regressed would
    fail loudly on the first card that ran it.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    with pytest.raises(TypeError):
        budget.exact_token_verdict(3, 3, 4)


def test_survivor_receipt_renders_before_the_engine_exists(tmp_path, monkeypatch):
    """The receipt must render from an empty phase, which is the failure path.

    A run that dies while the engine is being built, or inside the driver, is
    the run whose state nobody can otherwise see. The e2e writes the receipt in
    a `finally`, so rendering must not depend on any field the driver fills.
    """
    from tests.v1.e2e.spec_decode.test_uno import (
        _MixedPhase,
        _render_receipt,
        _write_receipt,
    )

    receipt = _render_receipt(_MixedPhase(), "geometry line", "engine: not built")
    assert "geometry line" in receipt
    assert "engine: not built" in receipt
    assert "steps=0, peer_visible_steps=0" in receipt
    assert "receipts=[]" in receipt

    target = tmp_path / "receipt.txt"
    monkeypatch.setenv("VLLM_UNO_SURVIVOR_RECEIPT", str(target))
    request = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace()))
    assert _write_receipt(receipt, request) == str(target)
    assert target.read_text(encoding="utf-8") == receipt


def test_survivor_usage_percentages_are_read_against_the_pinned_pool():
    """A percentage from another geometry must not be quoted as this one's.

    The 81-block pool read one free block as 98.750% (79 of 80). The pinned
    69-block pool reads the same state as 98.529% (67 of 68), and a full pool
    as 100.000%. Receipts quote the pool they were measured on.
    """
    from tests.v1.e2e.spec_decode import uno_kv_budget as budget

    budget_blocks = budget.survivor_kv_budget() // budget.kv_bytes_per_block()
    assert budget.usage_with_free_blocks(budget_blocks, 0) == 1.0
    assert round(budget.usage_with_free_blocks(budget_blocks, 1), 5) == round(
        1 - 1 / 68, 5
    )
    assert round(budget.usage_with_free_blocks(81, 1), 5) == round(1 - 1 / 80, 5)
