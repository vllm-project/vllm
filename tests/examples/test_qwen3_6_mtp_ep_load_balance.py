# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_module(module_name: str, file_name: str):
    module_dir = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "features"
        / "speculative_decoding"
    )
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    module_path = module_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helper = _load_module(
    "mtp_ep_load_balance_utils",
    "mtp_ep_load_balance_utils.py",
)
runtime = _load_module(
    "mtp_ep_experiment_runtime",
    "mtp_ep_experiment_runtime.py",
)


def _scheduler_output(num_scheduled_tokens, scheduled_spec_decode_tokens):
    return SimpleNamespace(
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
    )


def _model_runner_output(req_ids, routing_data):
    return SimpleNamespace(
        req_ids=req_ids,
        routed_experts=SimpleNamespace(routing_data=routing_data),
    )


def test_step_selection_skips_baseline_prefill_and_keeps_decode_step():
    prefill_output = _scheduler_output({"0": 4, "1": 5}, {})
    decode_output = _scheduler_output({"0": 1, "1": 1}, {})
    prefill_routing_data = np.zeros((9, 40, 2), dtype=np.int64)
    prefill_model_output = _model_runner_output(["0", "1"], prefill_routing_data)
    decode_routing_data = np.zeros((2, 40, 2), dtype=np.int64)
    decode_model_output = _model_runner_output(["0", "1"], decode_routing_data)

    assert not helper.should_capture_baseline_decode_step(prefill_output)
    assert helper.select_step_routing_data(
        prefill_output,
        prefill_model_output,
        use_spec_decode=False,
    ) is None

    captured = helper.select_step_routing_data(
        decode_output,
        decode_model_output,
        use_spec_decode=False,
    )
    assert captured is not None
    assert captured.step_kind == "baseline_decode"
    assert captured.total_scheduled_tokens == 2
    np.testing.assert_array_equal(captured.routing_data, decode_routing_data)


def test_step_selection_only_keeps_mtp_verification_steps():
    no_spec_routing_data = np.arange(2 * 40 * 2, dtype=np.int64).reshape(2, 40, 2)
    no_spec_model_output = _model_runner_output(
        ["req_b", "req_a"],
        no_spec_routing_data,
    )

    no_spec_output = _scheduler_output({"req_b": 1, "req_a": 1}, {})
    assert not helper.should_capture_mtp_verification_step(no_spec_output)
    assert helper.select_step_routing_data(
        no_spec_output,
        no_spec_model_output,
        use_spec_decode=True,
    ) is None

    routing_data = np.arange(5 * 40 * 2, dtype=np.int64).reshape(5, 40, 2)
    model_output = _model_runner_output(["req_b", "req_a"], routing_data)
    mtp_output = _scheduler_output(
        {"req_b": 2, "req_a": 3},
        {"req_a": [7, 8], "req_b": [9]},
    )
    captured = helper.select_step_routing_data(
        mtp_output,
        model_output,
        use_spec_decode=True,
    )
    assert captured is not None
    assert captured.step_kind == "mtp_verification"
    assert captured.request_ids == ("req_b", "req_a")
    assert captured.total_scheduled_tokens == 5
    np.testing.assert_array_equal(captured.routing_data, routing_data)


def test_counting_and_descending_reorder_are_correct():
    routing_data = np.zeros((2, 40, 2), dtype=np.int64)

    routing_data[0, 0, :] = [5, 1]
    routing_data[1, 0, :] = [5, 2]
    routing_data[0, 9, :] = [7, 7]
    routing_data[1, 9, :] = [3, 7]

    histograms = helper.count_layer_expert_histograms(
        routing_data,
        layers=(0, 9),
        num_experts=8,
    )

    expected_layer0 = np.array([0, 1, 1, 0, 0, 2, 0, 0])
    expected_layer9 = np.array([0, 0, 0, 1, 0, 0, 0, 3])
    np.testing.assert_array_equal(histograms[0], expected_layer0)
    np.testing.assert_array_equal(histograms[1], expected_layer9)

    sorted_counts, sorted_ids = helper.sort_experts_desc(histograms.astype(np.float64))
    np.testing.assert_array_equal(sorted_counts[0, :3], np.array([2.0, 1.0, 1.0]))
    np.testing.assert_array_equal(sorted_ids[0, :3], np.array([5, 1, 2]))
    np.testing.assert_array_equal(sorted_counts[1, :2], np.array([3.0, 1.0]))
    np.testing.assert_array_equal(sorted_ids[1, :2], np.array([7, 3]))


def test_metric_logic_matches_balancedness_gini_and_relative_change():
    avg_histograms = np.array([[10.0, 3.0, 2.0, 1.0]])
    baseline_histograms = np.array([[4.0, 4.0, 4.0, 4.0]])

    rows = helper.build_condition_metrics(
        batch_size=32,
        draft_length=2,
        num_steps=5,
        layers=(0,),
        avg_histograms=avg_histograms,
        baseline_histograms=baseline_histograms,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["batch_size"] == 32
    assert row["draft_length"] == 2
    assert row["num_steps"] == 5
    assert row["avg_total_routed_assignments_per_step"] == 16.0
    assert row["balancedness"] == 0.4
    assert row["baseline_balancedness"] == 1.0
    assert row["balancedness_delta"] == -0.6
    assert row["balancedness_relative_change"] == -0.6
    assert row["gini"] > row["baseline_gini"]
    assert row["imbalance_change"] == "worsened"


def test_aggregate_worker_step_timings_uses_max_per_component():
    timing = helper.aggregate_worker_step_timings(
        [
            {
                "total_ms": 10.0,
                "attention_ms": 1.5,
                "routing_ms": 0.5,
                "prepare_ms": 2.0,
                "finalize_ms": 1.5,
                "ffn_ms": 4.0,
            },
            {
                "total_ms": 11.0,
                "attention_ms": 1.0,
                "routing_ms": 0.75,
                "prepare_ms": 1.0,
                "finalize_ms": 2.5,
                "ffn_ms": 3.0,
            },
        ]
    )
    assert timing.total_ms == 11.0
    assert timing.attention_ms == 1.5
    assert timing.routing_ms == 0.75
    assert timing.prepare_ms == 2.0
    assert timing.finalize_ms == 2.5
    assert timing.ffn_ms == 4.0
    assert timing.all2all_ms == 4.5
    assert timing.unattributed_ms == 0.25


def test_step_time_summary_and_normalization_are_correct():
    summary = helper.summarize_step_time_components(
        step_total_ms=np.array([10.0, 14.0]),
        step_attention_ms=np.array([1.0, 2.0]),
        step_routing_ms=np.array([0.5, 1.5]),
        step_prepare_ms=np.array([2.0, 4.0]),
        step_finalize_ms=np.array([1.0, 3.0]),
        step_ffn_ms=np.array([5.5, 3.5]),
    )
    assert summary["avg_step_total_ms"] == 12.0
    assert summary["avg_attention_ms"] == 1.5
    assert summary["avg_routing_ms"] == 1.0
    assert summary["avg_prepare_ms"] == 3.0
    assert summary["avg_finalize_ms"] == 2.0
    assert summary["avg_all2all_ms"] == 5.0
    assert summary["avg_ffn_ms"] == 4.5

    normalized = helper.normalize_time_components(summary, baseline_total_ms=8.0)
    assert normalized["normalized_attention_ms"] == 0.1875
    assert normalized["normalized_routing_ms"] == 0.125
    assert normalized["normalized_all2all_ms"] == 0.625
    assert normalized["normalized_ffn_ms"] == 0.5625
    assert normalized["ffn_share"] == 0.375


def test_close_ffn_component_folds_small_residual_into_ffn():
    closed = runtime._close_ffn_component(
        helper.StepTiming(
            total_ms=10.0,
            attention_ms=2.0,
            routing_ms=1.0,
            prepare_ms=1.0,
            finalize_ms=1.0,
            ffn_ms=4.0,
        )
    )
    assert closed == 5.0


def test_speedup_and_dataset_slicing_helpers_are_correct():
    rows = helper.build_speedup_rows(
        {
            (32, 0): 20.0,
            (32, 2): 10.0,
            (32, 4): 8.0,
            (64, 0): 30.0,
            (64, 2): 15.0,
            (64, 4): 12.0,
        },
        batch_sizes=(32, 64),
        draft_lengths=(0, 2, 4),
    )
    assert next(
        row["speedup"]
        for row in rows
        if row["batch_size"] == 32 and row["draft_length"] == 2
    ) == 2.0
    np.testing.assert_array_equal(
        helper.select_dataset_indices(4, 8),
        np.array([0, 1, 2, 3]),
    )


def test_prompt_cache_roundtrip(tmp_path):
    prompt_items = [
        {"prompt_token_ids": [1, 2, 3]},
        {"prompt_token_ids": [4, 5]},
    ]
    cache_path = tmp_path / "prompt_cache.json"
    runtime.save_prompt_items_cache(prompt_items, cache_path)

    args = SimpleNamespace(prompt_cache_path=cache_path, batch_size=2)
    loaded = runtime.load_prompt_items(args)
    assert loaded == prompt_items


def test_stop_condition_collection_worker_clears_pending_timings():
    runtime._WORKER_STATE.pending_step_timings.clear()
    runtime._WORKER_STATE.pending_step_timings.append({"total_ms": 1.0})
    runtime._WORKER_STATE.pending_step_timings.append({"total_ms": 2.0})
    runtime._WORKER_STATE.enabled = True

    result = runtime.stop_condition_collection_worker(None)

    assert result == {"pending_timings": 2}
    assert runtime._WORKER_STATE.enabled is False
    assert len(runtime._WORKER_STATE.pending_step_timings) == 0


def test_collect_one_command_includes_prompt_cache_and_warmup(tmp_path):
    args = SimpleNamespace(
        model="m",
        dataset="d",
        dataset_config="cfg",
        dataset_split="split",
        max_tokens=128,
        max_model_len=4096,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        num_experts=256,
        layers=(0, 9),
        enforce_eager=True,
        warmup_rounds=1,
    )
    command = runtime._build_collect_one_command(
        args,
        tmp_path / "out",
        tmp_path / "entry.py",
        batch_size=32,
        draft_length=4,
        prompt_cache=tmp_path / "prompt_cache.json",
    )
    assert "--prompt-cache-path" in command
    assert str(tmp_path / "prompt_cache.json") in command
    assert "--warmup-rounds" in command
    assert command[command.index("--warmup-rounds") + 1] == "1"


def test_baseline_order_reordering_is_stable():
    avg_histograms = np.array(
        [
            [4.0, 2.0, 1.0, 0.0],
            [1.0, 3.0, 0.0, 2.0],
        ]
    )
    _, baseline_order = helper.sort_experts_desc(avg_histograms)
    target_histograms = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [7.0, 5.0, 1.0, 9.0],
        ]
    )
    reordered = helper.reorder_histograms_by_expert_order(
        target_histograms,
        baseline_order,
    )
    np.testing.assert_array_equal(reordered[0], np.array([10.0, 20.0, 30.0, 40.0]))
    np.testing.assert_array_equal(reordered[1], np.array([5.0, 9.0, 7.0, 1.0]))
