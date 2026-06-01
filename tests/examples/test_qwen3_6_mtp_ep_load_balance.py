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
    prefill_decision = helper.classify_step_capture(
        prefill_output,
        prefill_model_output,
        worker_step_metadata={"req_ids": ["0", "1"], "has_prefill": True},
        use_spec_decode=False,
    )
    assert prefill_decision.captured_step is None
    assert prefill_decision.drop_reason == "prefill"

    decision = helper.classify_step_capture(
        decode_output,
        decode_model_output,
        worker_step_metadata={"req_ids": ["0", "1"], "has_prefill": False},
        use_spec_decode=False,
    )
    captured = decision.captured_step
    assert captured is not None
    assert captured.step_kind == "decode_only"
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
    decision = helper.classify_step_capture(
        mtp_output,
        model_output,
        worker_step_metadata={
            "req_ids": ["req_b", "req_a"],
            "has_prefill": False,
        },
        use_spec_decode=True,
    )
    captured = decision.captured_step
    assert captured is not None
    assert captured.step_kind == "verification_only"
    assert captured.request_ids == ("req_b", "req_a")
    assert captured.total_scheduled_tokens == 5
    np.testing.assert_array_equal(captured.routing_data, routing_data)


def test_mixed_mtp_step_is_dropped_without_request_reslicing():
    routing_data = np.arange(3 * 40 * 2, dtype=np.int64).reshape(3, 40, 2)
    model_output = _model_runner_output(["req_a", "req_b"], routing_data)
    mtp_output = _scheduler_output({"req_a": 2, "req_b": 1}, {"req_a": [7]})
    decision = helper.classify_step_capture(
        mtp_output,
        model_output,
        worker_step_metadata={"req_ids": ["req_a", "req_b"], "has_prefill": False},
        use_spec_decode=True,
    )
    assert decision.captured_step is None
    assert decision.drop_reason == "mixed"


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


def _rank_candidate_data(seq_ids, kinds, total_ms, ffn_ms, hist_values=None):
    size = len(seq_ids)
    histograms = np.zeros((size, 1, 4), dtype=np.int64)
    if hist_values is not None:
        histograms[:, 0, :] = np.asarray(hist_values, dtype=np.int64)
    return {
        "candidate_first_ep_collective_seq_ids": np.asarray(seq_ids, dtype=np.int64),
        "candidate_step_kinds": np.asarray(kinds, dtype=np.str_),
        "candidate_step_total_ms": np.asarray(total_ms, dtype=np.float64),
        "candidate_step_ffn_ms": np.asarray(ffn_ms, dtype=np.float64),
        "candidate_step_total_tokens": np.full((size,), 2, dtype=np.int64),
        "candidate_step_histograms": histograms,
    }


def test_global_step_time_aggregation_keeps_strict_ep_seq_intersection():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data(
                [7],
                ["verification_only"],
                [10.0],
                [4.0],
                [[1, 2, 0, 0]],
            ),
            _rank_candidate_data(
                [7],
                ["verification_only"],
                [11.0],
                [5.0],
                [[0, 1, 3, 0]],
            ),
        ],
        data_parallel_size=2,
        expected_step_kind="verification_only",
        layers=(0,),
        num_experts=4,
    )
    np.testing.assert_array_equal(result.global_step_indices, np.array([7]))
    np.testing.assert_array_equal(
        result.global_step_kinds,
        np.array(["verification_only"]),
    )
    np.testing.assert_allclose(result.global_step_total_ms, np.array([11.0]))
    np.testing.assert_allclose(result.global_step_ffn_ms, np.array([5.0]))
    np.testing.assert_allclose(result.global_step_other_ms, np.array([6.0]))
    np.testing.assert_array_equal(
        result.global_step_histograms[0, 0],
        np.array([1, 3, 3, 0]),
    )
    assert result.num_global_candidate_steps == 1
    assert result.num_global_captured_steps == 1


def test_global_step_time_aggregation_drops_prefill_global_step():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data([7], ["decode_only"], [10.0], [4.0]),
            _rank_candidate_data([7], ["prefill"], [11.0], [5.0]),
        ],
        data_parallel_size=2,
        expected_step_kind="decode_only",
        layers=(0,),
        num_experts=4,
    )
    assert result.global_step_indices.size == 0
    assert result.num_global_prefill_dropped_steps == 1


def test_global_step_time_aggregation_drops_mixed_global_step():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data([7], ["verification_only"], [10.0], [4.0]),
            _rank_candidate_data([7], ["mixed"], [11.0], [5.0]),
        ],
        data_parallel_size=2,
        expected_step_kind="verification_only",
        layers=(0,),
        num_experts=4,
    )
    assert result.global_step_indices.size == 0
    assert result.num_global_mixed_dropped_steps == 1


def test_global_step_time_aggregation_drops_missing_join_key():
    result = helper.aggregate_global_step_time_components(
        [
            _rank_candidate_data([7], ["decode_only"], [10.0], [4.0]),
            _rank_candidate_data([-1], ["decode_only"], [11.0], [5.0]),
        ],
        data_parallel_size=2,
        expected_step_kind="decode_only",
        layers=(0,),
        num_experts=4,
    )
    assert result.global_step_indices.size == 0
    assert result.num_global_non_target_dropped_steps == 1


def test_global_step_time_aggregation_rejects_negative_other_time():
    try:
        helper.aggregate_global_step_time_components(
            [
                _rank_candidate_data([7], ["decode_only"], [5.0], [6.0]),
                _rank_candidate_data([7], ["decode_only"], [4.0], [3.0]),
            ],
            data_parallel_size=2,
            expected_step_kind="decode_only",
            layers=(0,),
            num_experts=4,
        )
    except ValueError as exc:
        assert "negative Other time" in str(exc)
    else:
        raise AssertionError("Expected negative Other time to fail.")


def test_global_step_time_summary_and_normalization_are_correct():
    summary = helper.summarize_global_step_time_components(
        step_total_ms=np.array([10.0, 14.0]),
        step_ffn_ms=np.array([4.0, 6.0]),
        step_other_ms=np.array([6.0, 8.0]),
    )
    assert summary["avg_step_total_ms"] == 12.0
    assert summary["avg_ffn_ms"] == 5.0
    assert summary["avg_other_ms"] == 7.0

    normalized = helper.normalize_global_time_components(
        summary, baseline_total_ms=8.0
    )
    assert normalized["normalized_ffn_ms"] == 0.625
    assert normalized["normalized_other_ms"] == 0.875
    assert normalized["ffn_share"] == 5.0 / 12.0
    assert normalized["other_share"] == 7.0 / 12.0


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
        {
            (32, 0): 2000.0,
            (32, 2): 1000.0,
            (32, 4): 800.0,
            (64, 0): 3000.0,
            (64, 2): 1500.0,
            (64, 4): 1200.0,
        },
        {
            (32, 0): 100,
            (32, 2): 100,
            (32, 4): 100,
            (64, 0): 150,
            (64, 2): 150,
            (64, 4): 150,
        },
        batch_sizes=(32, 64),
        draft_lengths=(0, 2, 4),
    )
    assert next(
        row["tpot_speedup"]
        for row in rows
        if row["batch_size"] == 32 and row["draft_length"] == 2
    ) == 2.0
    assert next(
        row["tpot_speedup"]
        for row in rows
        if row["batch_size"] == 32 and row["draft_length"] == 0
    ) == 1.0
    assert next(
        row["decode_throughput_speedup"]
        for row in rows
        if row["batch_size"] == 32 and row["draft_length"] == 0
    ) == 1.0
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
    runtime._WORKER_STATE.pending_step_records.clear()
    runtime._WORKER_STATE.pending_step_records.append({"timing": {"total_ms": 1.0}})
    runtime._WORKER_STATE.pending_step_records.append({"timing": {"total_ms": 2.0}})
    runtime._WORKER_STATE.enabled = True

    result = runtime.stop_condition_collection_worker(None)

    assert result == {"pending_timings": 2}
    assert runtime._WORKER_STATE.enabled is False
    assert len(runtime._WORKER_STATE.pending_step_records) == 0


def test_collect_one_command_includes_prompt_cache_and_warmup(tmp_path):
    args = SimpleNamespace(
        model="m",
        dataset="d",
        dataset_config="cfg",
        dataset_split="split",
        num_samples=1024,
        data_parallel_size=4,
        max_tokens=128,
        max_model_len=4096,
        tensor_parallel_size=1,
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
    assert command[command.index("--data-parallel-size") + 1] == "4"
    assert command[command.index("--num-samples") + 1] == "1024"


def test_dp_sharding_covers_all_samples_without_overlap():
    shards = [
        helper.shard_global_batch_indices(
            num_samples=10,
            global_batch_size=4,
            round_idx=round_idx,
            dp_size=3,
            dp_rank=dp_rank,
        )
        for round_idx in range(helper.num_condition_rounds(10, 4))
        for dp_rank in range(3)
    ]
    flat = np.concatenate(shards, axis=0)
    np.testing.assert_array_equal(np.sort(flat), np.arange(10))


def test_tpot_formula_matches_decode_only_definition():
    output_lengths = np.array([5, 1, 3], dtype=np.int64)
    assert helper.compute_num_output_tokens_excluding_first(output_lengths) == 6
    assert helper.compute_tpot_ms(30.0, output_lengths) == 5.0
    assert helper.compute_tpot_ms_from_finished_stats(30.0, 6) == 5.0
    assert helper.compute_decode_throughput_tok_s(12, 300.0) == 40.0


def test_expert_to_ep_rank_mapping_and_rank_load_are_correct():
    expert_to_ep_rank = helper.build_expert_to_ep_rank(
        num_experts=5,
        ep_size=2,
    )
    np.testing.assert_array_equal(expert_to_ep_rank, np.array([0, 0, 0, 1, 1]))
    avg_histograms = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    rank_load = helper.build_rank_load_from_histograms(
        avg_histograms,
        expert_to_ep_rank,
        ep_size=2,
    )
    np.testing.assert_allclose(rank_load, np.array([[6.0, 9.0]]))


def test_validate_parallel_config_requires_tp1():
    args = SimpleNamespace(tensor_parallel_size=2, data_parallel_size=1)
    try:
        runtime.validate_parallel_config(args)
    except ValueError as exc:
        assert "tensor_parallel_size=1" in str(exc)
    else:
        raise AssertionError("Expected tensor_parallel_size validation to fail.")


def test_extract_worker_step_metadata_supports_legacy_gpu_model_runner():
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=["req0", "req1"],
                num_reqs=2,
                num_computed_tokens_cpu=np.array([0, 8], dtype=np.int32),
                num_prompt_tokens=np.array([16, 8], dtype=np.int32),
            ),
            execute_model_state=None,
        )
    )
    metadata = runtime._extract_worker_step_metadata(worker)
    assert metadata["req_ids"] == ["req0", "req1"]
    assert metadata["has_prefill"] is True


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
