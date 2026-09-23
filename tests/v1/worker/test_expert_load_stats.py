# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.expert_load import ExpertLoadStatsConfig
from vllm.distributed.expert_load import (
    ExpertLoadLayer,
    _record_expert_load,
    owned_token_span,
)
from vllm.v1.worker.expert_load_stats import ExpertLoadReporter, ExpertLoadStats


@pytest.mark.parametrize(
    "kwargs",
    [
        {"log_interval": 0},
        {"trace_interval": 0},
        {"flush_interval": 0},
        {"flush_interval": 4097},
        {"trace_max_iterations": 0},
        {"scope": "global"},
        {"enabled": True, "trace": True},
        {"layers": []},
        {"layers": [-1]},
        {"layers": [1, 1]},
    ],
)
def test_reject_invalid_stats_config(kwargs):
    with pytest.raises(ValueError):
        ExpertLoadStatsConfig(**kwargs)


def test_stats_hash_separates_instrumented_graphs():
    disabled = ExpertLoadStatsConfig()
    enabled = ExpertLoadStatsConfig(enabled=True)
    subset = ExpertLoadStatsConfig(enabled=True, layers=[2])
    assert len({cfg.compute_hash() for cfg in [disabled, enabled, subset]}) == 3
    assert (
        enabled.compute_hash()
        == ExpertLoadStatsConfig(
            enabled=True, log_interval=7, output_dir="unused"
        ).compute_hash()
    )


@pytest.mark.parametrize(
    "args,expected",
    [
        ((8, 0, 0, 1, False, None, None), (0, 8, 0)),
        ((8, 0, 1, 1, False, None, None), (0, 0, 0)),
        ((4, 1, 1, 2, False, [3, 8], None), (0, 4, 4)),
        ((12, 1, 0, 1, True, [4, 8], [4, 8]), (4, 12, 0)),
        ((14, 1, 0, 2, True, [5, 7], [3, 3, 4, 4]), (6, 13, 0)),
        ((14, 1, 1, 2, True, [5, 7], [3, 3, 4, 4]), (0, 0, 0)),
    ],
)
def test_owned_rows_do_not_duplicate_tp_or_other_dp_tokens(args, expected):
    assert owned_token_span(*args) == expected


def test_unknown_dispatch_shape_fails_closed():
    with pytest.raises(ValueError, match="dispatch metadata"):
        owned_token_span(13, 1, 0, 2, True, [5, 7], [3, 3, 4, 4])


def test_disabled_stats_do_not_construct_tracker(monkeypatch):
    config = SimpleNamespace(expert_load_stats_config=ExpertLoadStatsConfig())
    monkeypatch.setattr(ExpertLoadStats, "__init__", Mock(side_effect=AssertionError))
    assert ExpertLoadStats.create(config, None, None) is None


def test_reporter_keeps_iteration_layers_separate_and_resets_summary(tmp_path):
    config = ExpertLoadStatsConfig(
        enabled=True, trace=True, output_dir=str(tmp_path), log_interval=2
    )
    reporter = ExpertLoadReporter(config, [3, 7], 3, {"dp_rank": 2, "tp_rank": 0})
    first = np.array([[1, 2, 0], [0, 1, 2]], dtype=np.int64)
    second = np.array([[0, 2, 1], [3, 0, 0]], dtype=np.int64)
    records = reporter.consume(first, first[None], [1], 1, 0, False)
    records += reporter.consume(second, second[None], [2], 2, 1, True)
    summaries = [r for r in records if r["event"] == "vllm.expert_load"]
    assert [r["counts"] for r in summaries] == [[1, 4, 1], [3, 1, 2]]
    assert summaries[0]["max_mean_ratio"] == 2.0
    assert summaries[0]["dropped_trace_iterations"] == 1
    assert [(r["iteration"], r["layer"]) for r in records[:4]] == [
        (1, 3),
        (1, 7),
        (2, 3),
        (2, 7),
    ]
    reporter.write(records)
    assert reporter.path is not None
    assert [
        json.loads(line) for line in reporter.path.read_text().splitlines()
    ] == records
    empty = np.zeros_like(first)
    next_records = reporter.consume(empty, empty[None][:0], [], 3, 0, True)
    assert next_records[0]["step_begin"] == 3
    assert next_records[0]["counts"] == [0, 0, 0]
    assert next_records[0]["max_mean_ratio"] == 0
    assert next_records[0]["dropped_trace_iterations"] == 0


def test_slow_writer_slot_cannot_be_reused():
    stats = object.__new__(ExpertLoadStats)
    stats.current = None
    pending: list[Future[None]] = [Future(), Future()]
    stats.slots = [SimpleNamespace(future=f, iterations=[9]) for f in pending]
    stats._acquire()
    assert stats.current is None
    pending[1].set_result(None)
    stats._acquire()
    assert stats.current is stats.slots[1]
    assert stats.current.iterations == []
    assert stats.slots[0].iterations == [9]


def test_cumulative_summary_retains_counts_without_changing_trace_rows():
    reporter = ExpertLoadReporter(
        ExpertLoadStatsConfig(enabled=True, reset_after_log=False, detail="summary"),
        [2],
        3,
        {"dp_rank": 0},
    )
    counts = np.array([[1, 2, 0]], dtype=np.int64)
    for step in (1, 2):
        records = reporter.consume(counts, counts[None], [step], step, 0, True)
        assert records[0]["counts"] == [1, 2, 0]
        assert "counts" not in records[1]
        assert records[1]["assignments"] == 3 * step
        assert records[1]["step_begin"] == 1
        assert records[1]["step_end"] == step


def test_writer_errors_are_not_silently_discarded():
    stats = object.__new__(ExpertLoadStats)
    stats.current = None
    failed: Future[None] = Future()
    failed.set_exception(OSError("disk full"))
    stats.slots = [SimpleNamespace(future=failed)]
    with pytest.raises(OSError, match="disk full"):
        stats._acquire()


@pytest.mark.parametrize(
    "active,dummy,num_tokens,record",
    [
        (False, False, 3, False),
        (True, True, 3, False),
        (True, False, 0, False),
        (True, False, 3, True),
    ],
)
def test_execute_wrapper_excludes_warmup_dummy_and_empty_calls(
    active, dummy, num_tokens, record
):
    stats = object.__new__(ExpertLoadStats)
    stats.active = active
    stats.begin, stats.end = Mock(), Mock()
    execute = Mock(return_value="output")
    wrapped = stats.wrap_execute(execute)
    assert (
        wrapped(SimpleNamespace(total_num_scheduled_tokens=num_tokens), dummy_run=dummy)
        == "output"
    )
    assert stats.begin.call_count == stats.end.call_count == int(record)


def test_failed_forward_disarms_recording():
    stats = object.__new__(ExpertLoadStats)
    stats.active = True
    stats.begin, stats.end = Mock(), Mock()
    stats.num_valid_tokens = torch.tensor(3)
    stats.counts = torch.ones((2, 3))
    wrapped = stats.wrap_execute(Mock(side_effect=RuntimeError("failed")))
    with pytest.raises(RuntimeError, match="failed"):
        wrapped(SimpleNamespace(total_num_scheduled_tokens=3))
    assert stats.num_valid_tokens.item() == 0
    assert torch.count_nonzero(stats.counts) == 0
    stats.end.assert_not_called()


@pytest.mark.parametrize(
    "trace,interval,limit,expected",
    [
        (False, 1, 8, []),
        (True, 1, 3, [1, 2, 3]),
        (True, 2, 5, [1, 3, 5]),
    ],
)
def test_trace_sampling_and_limit_do_not_skip_summary_steps(
    monkeypatch, trace, interval, limit, expected
):
    import vllm.v1.worker.expert_load_stats as stats_module

    stats = object.__new__(ExpertLoadStats)
    stats.config = ExpertLoadStatsConfig(
        enabled=True,
        trace=trace,
        output_dir="unused",
        trace_interval=interval,
        trace_max_iterations=limit,
        flush_interval=64,
        log_interval=100,
    )
    stats.iteration, stats.summary_begin, stats.dropped = 0, 1, 0
    stats.counts = torch.zeros((1, 3), dtype=torch.int64)
    stats.summary = torch.zeros_like(stats.counts)
    stats.num_valid_tokens = torch.tensor(1)
    stats.current = SimpleNamespace(device=torch.empty((65, 1, 3)), iterations=[])
    stats._flush = Mock()
    finish = Mock()
    monkeypatch.setattr(stats_module, "finish_expert_load_iteration", {(1,): finish})
    for _ in range(8):
        stats.end()
    assert stats.current.iterations == expected
    assert finish.call_count == 8
    assert stats.dropped == 0


def test_full_export_queue_drops_trace_only(monkeypatch):
    import vllm.v1.worker.expert_load_stats as stats_module

    stats = object.__new__(ExpertLoadStats)
    stats.config = ExpertLoadStatsConfig(enabled=True, trace=True, output_dir="unused")
    stats.iteration, stats.summary_begin, stats.dropped = 0, 1, 0
    stats.counts = torch.zeros((1, 3), dtype=torch.int64)
    stats.summary = torch.zeros_like(stats.counts)
    stats.num_valid_tokens = torch.tensor(1)
    stats.current = None
    finish = Mock()
    monkeypatch.setattr(stats_module, "finish_expert_load_iteration", {(1,): finish})
    stats.end()
    assert stats.dropped == 1
    finish.assert_called_once()
    assert finish.call_args.args[1] is stats.summary
    assert finish.call_args.args[2] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_valid", [0, 1, 4, 5])
@pytest.mark.parametrize("span", [(0, 5, 0), (1, 4, 0), (0, 3, 3), (0, 0, 0)])
def test_gpu_histogram_masks_padding_invalid_and_unowned_tokens(num_valid, span):
    ids = torch.tensor([[0, 1], [1, -1], [2, 2**32], [1, 0], [2, 2]], device="cuda")
    counts = torch.zeros(3, dtype=torch.int64, device="cuda")
    valid = torch.tensor(num_valid, dtype=torch.int32, device="cuda")
    start, end, offset = span
    _record_expert_load[(1,)](
        ids, counts, valid, 5, 2, 2, 1, start, end, offset, 3, 4, 256
    )
    expected = torch.zeros(3, dtype=torch.int64)
    for token, row in enumerate(ids.cpu().tolist()):
        if start <= token < end and token - start + offset < num_valid:
            for expert in row:
                if 0 <= expert < 3:
                    expected[expert] += 1
    torch.testing.assert_close(counts.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_graph_replay_reads_live_valid_count_and_exports(tmp_path):
    config = ExpertLoadStatsConfig(
        enabled=True,
        trace=True,
        output_dir=str(tmp_path),
        log_interval=2,
        flush_interval=2,
    )
    stats = ExpertLoadStats(
        config, [4], 3, {"dp_rank": 0, "tp_rank": 0}, torch.device("cuda")
    )
    ids = torch.tensor([[0, 1], [1, 2], [2, 2], [0, 0]], device="cuda")

    def run():
        _record_expert_load[(1,)](
            ids, stats.counts[0], stats.num_valid_tokens, 4, 2, 2, 1, 0, 4, 0, 3, 4, 256
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Capture and warmup had num_valid=0 and must not contribute.
    for valid in (1, 3):
        stats.begin(valid)
        graph.replay()
        stats.end()
    stats.close()
    records = [
        json.loads(line) for line in stats.reporter.path.read_text().splitlines()
    ]
    traces = [r for r in records if "iteration" in r]
    assert [r["counts"] for r in traces] == [[1, 1, 0], [1, 2, 3]]
    summary = [r for r in records if r["event"] == "vllm.expert_load"]
    assert len(summary) == 1
    assert summary[0]["counts"] == [2, 3, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("record_eplb", [False, True])
def test_gpu_eplb_fusion_preserves_mapping_and_sampling(record_eplb, monkeypatch):
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        eplb_map_to_physical_and_record,
    )

    ids = torch.tensor([[0, 1], [2, 1], [-1, 9]], device="cuda", dtype=torch.int32)
    mapping = torch.tensor([[2, 3], [0, 0], [1, 1]], device="cuda", dtype=torch.int32)
    replicas = torch.tensor([2, 1, 1], device="cuda", dtype=torch.int32)
    valid = torch.tensor(2, device="cuda", dtype=torch.int32)
    flag = torch.tensor(record_eplb, device="cuda")
    physical = torch.zeros(4, device="cuda", dtype=torch.int32)
    reference = torch.zeros_like(physical)
    logical = torch.zeros(3, device="cuda", dtype=torch.int64)
    layer = ExpertLoadLayer(logical, valid, 0, 0, 1, False)
    monkeypatch.setattr(layer, "token_span", lambda _: (0, 3, 0))
    expected = eplb_map_to_physical_and_record(
        ids, reference, mapping, replicas, flag, valid
    )
    actual = eplb_map_to_physical_and_record(
        ids, physical, mapping, replicas, flag, valid, layer
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(physical, reference)
    torch.testing.assert_close(logical.cpu(), torch.tensor([1, 2, 1]))
    torch.testing.assert_close(
        mapping.cpu(), torch.tensor([[2, 3], [0, 0], [1, 1]], dtype=torch.int32)
    )
