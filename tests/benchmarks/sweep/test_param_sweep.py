# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from vllm.benchmarks.sweep import serve as sweep_serve
from vllm.benchmarks.sweep import serve_workload as sweep_workload
from vllm.benchmarks.sweep import startup as sweep_startup
from vllm.benchmarks.sweep.param_sweep import ParameterSweep, ParameterSweepItem


class TestParameterSweepItem:
    """Test ParameterSweepItem functionality."""

    @pytest.mark.parametrize(
        "input_dict,expected",
        [
            (
                {"compilation_config.use_inductor_graph_partition": False},
                "--compilation-config.use_inductor_graph_partition=false",
            ),
            (
                {"compilation_config.use_inductor_graph_partition": True},
                "--compilation-config.use_inductor_graph_partition=true",
            ),
        ],
    )
    def test_nested_boolean_params(self, input_dict, expected):
        """Test that nested boolean params use =true/false syntax."""
        item = ParameterSweepItem.from_record(input_dict)
        cmd = item.apply_to_cmd(["vllm", "serve", "model"])
        assert expected in cmd

    @pytest.mark.parametrize(
        "input_dict,expected",
        [
            ({"enable_prefix_caching": False}, "--no-enable-prefix-caching"),
            ({"enable_prefix_caching": True}, "--enable-prefix-caching"),
            ({"disable_log_stats": False}, "--no-disable-log-stats"),
            ({"disable_log_stats": True}, "--disable-log-stats"),
        ],
    )
    def test_non_nested_boolean_params(self, input_dict, expected):
        """Test that non-nested boolean params use --no- prefix."""
        item = ParameterSweepItem.from_record(input_dict)
        cmd = item.apply_to_cmd(["vllm", "serve", "model"])
        assert expected in cmd

    @pytest.mark.parametrize(
        "compilation_config",
        [
            {"cudagraph_mode": "full", "mode": 2, "use_inductor_graph_partition": True},
            {
                "cudagraph_mode": "piecewise",
                "mode": 3,
                "use_inductor_graph_partition": False,
            },
        ],
    )
    def test_nested_dict_value(self, compilation_config):
        """Test that nested dict values are serialized as JSON."""
        item = ParameterSweepItem.from_record(
            {"compilation_config": compilation_config}
        )
        cmd = item.apply_to_cmd(["vllm", "serve", "model"])
        assert "--compilation-config" in cmd
        # The dict should be JSON serialized
        idx = cmd.index("--compilation-config")
        assert json.loads(cmd[idx + 1]) == compilation_config

    @pytest.mark.parametrize(
        "input_dict,expected_key,expected_value",
        [
            ({"model": "test-model"}, "--model", "test-model"),
            ({"max_tokens": 100}, "--max-tokens", "100"),
            ({"temperature": 0.7}, "--temperature", "0.7"),
        ],
    )
    def test_string_and_numeric_values(self, input_dict, expected_key, expected_value):
        """Test that string and numeric values are handled correctly."""
        item = ParameterSweepItem.from_record(input_dict)
        cmd = item.apply_to_cmd(["vllm", "serve"])
        assert expected_key in cmd
        assert expected_value in cmd

    @pytest.mark.parametrize(
        "input_dict,expected_key,key_idx_offset",
        [
            ({"max_tokens": 200}, "--max-tokens", 1),
            ({"enable_prefix_caching": False}, "--no-enable-prefix-caching", 0),
        ],
    )
    def test_replace_existing_parameter(self, input_dict, expected_key, key_idx_offset):
        """Test that existing parameters in cmd are replaced."""
        item = ParameterSweepItem.from_record(input_dict)

        if key_idx_offset == 1:
            # Key-value pair
            cmd = item.apply_to_cmd(["vllm", "serve", "--max-tokens", "100", "model"])
            assert expected_key in cmd
            idx = cmd.index(expected_key)
            assert cmd[idx + 1] == "200"
            assert "100" not in cmd
        else:
            # Boolean flag
            cmd = item.apply_to_cmd(
                ["vllm", "serve", "--enable-prefix-caching", "model"]
            )
            assert expected_key in cmd
            assert "--enable-prefix-caching" not in cmd


class TestParameterSweep:
    """Test ParameterSweep functionality."""

    def test_from_records_list(self):
        """Test creating ParameterSweep from a list of records."""
        records: list[dict[str, object]] = [
            {"max_tokens": 100, "temperature": 0.7},
            {"max_tokens": 200, "temperature": 0.9},
        ]
        sweep = ParameterSweep.from_records(records)
        assert len(sweep) == 2
        assert sweep[0]["max_tokens"] == 100
        assert sweep[1]["max_tokens"] == 200

    def test_read_from_dict(self):
        """Test creating ParameterSweep from a dict format."""
        data: dict[str, dict[str, object]] = {
            "experiment1": {"max_tokens": 100, "temperature": 0.7},
            "experiment2": {"max_tokens": 200, "temperature": 0.9},
        }
        sweep = ParameterSweep.read_from_dict(data)
        assert len(sweep) == 2

        # Check that items have the _benchmark_name field
        names = {item["_benchmark_name"] for item in sweep}
        assert names == {"experiment1", "experiment2"}

        # Check that parameters are preserved
        for item in sweep:
            if item["_benchmark_name"] == "experiment1":
                assert item["max_tokens"] == 100
                assert item["temperature"] == 0.7
            elif item["_benchmark_name"] == "experiment2":
                assert item["max_tokens"] == 200
                assert item["temperature"] == 0.9

    def test_read_json_list_format(self):
        """Test reading JSON file with list format."""
        records: list[dict[str, object]] = [
            {"max_tokens": 100, "temperature": 0.7},
            {"max_tokens": 200, "temperature": 0.9},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(records, f)
            temp_path = Path(f.name)

        try:
            sweep = ParameterSweep.read_json(temp_path)
            assert len(sweep) == 2
            assert sweep[0]["max_tokens"] == 100
            assert sweep[1]["max_tokens"] == 200
        finally:
            temp_path.unlink()

    def test_read_json_dict_format(self):
        """Test reading JSON file with dict format."""
        data: dict[str, dict[str, object]] = {
            "experiment1": {"max_tokens": 100, "temperature": 0.7},
            "experiment2": {"max_tokens": 200, "temperature": 0.9},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            sweep = ParameterSweep.read_json(temp_path)
            assert len(sweep) == 2

            # Check that items have the _benchmark_name field
            names = {item["_benchmark_name"] for item in sweep}
            assert names == {"experiment1", "experiment2"}
        finally:
            temp_path.unlink()

    def test_unique_benchmark_names_validation(self):
        """Test that duplicate _benchmark_name values raise an error."""
        # Test with duplicate names in list format
        records: list[dict[str, object]] = [
            {"_benchmark_name": "exp1", "max_tokens": 100},
            {"_benchmark_name": "exp1", "max_tokens": 200},
        ]

        with pytest.raises(ValueError, match="Duplicate _benchmark_name values"):
            ParameterSweep.from_records(records)

    def test_unique_benchmark_names_multiple_duplicates(self):
        """Test validation with multiple duplicate names."""
        records: list[dict[str, object]] = [
            {"_benchmark_name": "exp1", "max_tokens": 100},
            {"_benchmark_name": "exp1", "max_tokens": 200},
            {"_benchmark_name": "exp2", "max_tokens": 300},
            {"_benchmark_name": "exp2", "max_tokens": 400},
        ]

        with pytest.raises(ValueError, match="Duplicate _benchmark_name values"):
            ParameterSweep.from_records(records)

    def test_no_benchmark_names_allowed(self):
        """Test that records without _benchmark_name are allowed."""
        records: list[dict[str, object]] = [
            {"max_tokens": 100, "temperature": 0.7},
            {"max_tokens": 200, "temperature": 0.9},
        ]
        sweep = ParameterSweep.from_records(records)
        assert len(sweep) == 2

    def test_mixed_benchmark_names_allowed(self):
        """Test that mixing records with and without _benchmark_name is allowed."""
        records: list[dict[str, object]] = [
            {"_benchmark_name": "exp1", "max_tokens": 100},
            {"max_tokens": 200, "temperature": 0.9},
        ]
        sweep = ParameterSweep.from_records(records)
        assert len(sweep) == 2


class TestParameterSweepItemKeyNormalization:
    """Test key normalization in ParameterSweepItem."""

    def test_underscore_to_hyphen_conversion(self):
        """Test that underscores are converted to hyphens in CLI."""
        item = ParameterSweepItem.from_record({"max_tokens": 100})
        cmd = item.apply_to_cmd(["vllm", "serve"])
        assert "--max-tokens" in cmd

    def test_nested_key_preserves_suffix(self):
        """Test that nested keys preserve the suffix format."""
        # The suffix after the dot should preserve underscores
        item = ParameterSweepItem.from_record(
            {"compilation_config.some_nested_param": "value"}
        )
        cmd = item.apply_to_cmd(["vllm", "serve"])
        # The prefix (compilation_config) gets converted to hyphens,
        # but the suffix (some_nested_param) is preserved
        assert any("compilation-config.some_nested_param" in arg for arg in cmd)


def test_run_comb_excludes_warmup_from_measured_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[tuple[int, int, str]] = []

    def fake_run_benchmark(
        server,
        bench_cmd,
        *,
        serve_overrides,
        bench_overrides,
        run_number,
        output_path,
        dry_run,
    ):
        calls.append(
            (run_number, int(bench_overrides["num_prompts"]), output_path.name)
        )
        return {"run_number": run_number}

    monkeypatch.setattr(sweep_serve, "run_benchmark", fake_run_benchmark)
    base_path = tmp_path / "combination"
    base_path.mkdir()

    measured = sweep_serve.run_comb(
        None,
        [],
        serve_comb=ParameterSweepItem(),
        bench_comb=ParameterSweepItem({"num_prompts": 320}),
        link_vars=[],
        base_path=base_path,
        num_runs=2,
        warmup_num_prompts=32,
        dry_run=False,
    )

    assert calls == [
        (-1, 32, "warmup.json"),
        (0, 320, "run=0.json"),
        (1, 320, "run=1.json"),
    ]
    assert measured == [{"run_number": 0}, {"run_number": 1}]


def test_run_comb_continue_on_error_keeps_later_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[int] = []

    def fake_run_benchmark(
        server,
        bench_cmd,
        *,
        serve_overrides,
        bench_overrides,
        run_number,
        output_path,
        dry_run,
    ):
        calls.append(run_number)
        if run_number == 0:
            raise RuntimeError("synthetic run failure")
        return {"run_number": run_number}

    monkeypatch.setattr(sweep_serve, "run_benchmark", fake_run_benchmark)
    base_path = tmp_path / "combination"
    base_path.mkdir()

    measured = sweep_serve.run_comb(
        None,
        [],
        serve_comb=ParameterSweepItem(),
        bench_comb=ParameterSweepItem({"num_prompts": 100}),
        link_vars=[],
        base_path=base_path,
        num_runs=2,
        warmup_num_prompts=0,
        dry_run=False,
        continue_on_error=True,
    )

    assert calls == [0, 1]
    assert measured == [{"run_number": 1}]
    assert (base_path / "run=0.failure.json").exists()
    assert (base_path / "summary.json").exists()


def test_run_comb_warmup_default_is_backward_compatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Older callers can omit warmup_num_prompts."""
    calls: list[int] = []

    def fake_run_benchmark(
        server,
        bench_cmd,
        *,
        serve_overrides,
        bench_overrides,
        run_number,
        output_path,
        dry_run,
    ):
        calls.append(run_number)
        return {"run_number": run_number}

    monkeypatch.setattr(sweep_serve, "run_benchmark", fake_run_benchmark)
    base_path = tmp_path / "combination"
    base_path.mkdir()

    measured = sweep_serve.run_comb(
        None,
        [],
        serve_comb=ParameterSweepItem(),
        bench_comb=ParameterSweepItem({"num_prompts": 10}),
        link_vars=[],
        base_path=base_path,
        num_runs=1,
        dry_run=False,
    )

    assert calls == [0]
    assert measured == [{"run_number": 0}]


def _run_sweep(
    mode, tmp_path, serve_params, bench_params, *, dry_run=True, link_vars=None
):
    kwargs: dict[str, Any] = {
        "serve_params": serve_params,
        "experiment_dir": tmp_path,
        "num_runs": 1,
        "show_stdout": False,
        "dry_run": dry_run,
    }
    if mode == "startup":
        return sweep_startup.run_combs([], startup_params=bench_params, **kwargs)

    kwargs.update(
        bench_params=bench_params,
        link_vars=link_vars or [],
        server_ready_timeout=1,
        warmup_num_prompts=0,
        continue_on_error=True,
    )
    if mode == "serve_workload":
        return sweep_workload.explore_combs_workloads(
            [], [], [], workload_var="request_rate", workload_iters=2, **kwargs
        )
    return sweep_serve.run_combs([], [], [], **kwargs)


@pytest.mark.parametrize("mode", ["serve", "serve_workload", "startup"])
@pytest.mark.parametrize("axis", ["serve", "benchmark"])
def test_sweep_rejects_sanitized_path_collisions(mode, axis, monkeypatch, tmp_path):
    """Reject aliases before starting a child, including continue-on-error sweeps."""

    def unexpected_process(*args, **kwargs):
        pytest.fail("A child process was started before result-path validation")

    monkeypatch.setattr(subprocess, "Popen", unexpected_process)
    key = (
        "max_model_len"
        if axis == "serve"
        else ("num_iters_cold" if mode == "startup" else "num_prompts")
    )
    colliding = ParameterSweep.read_from_dict(
        {"experiment/a": {key: 10}, "experiment_a": {key: 20}}
    )
    default = ParameterSweep.from_records([{}])

    with pytest.raises(ValueError, match="same result directory"):
        _run_sweep(
            mode,
            tmp_path,
            colliding if axis == "serve" else default,
            colliding if axis == "benchmark" else default,
            dry_run=False,
        )

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["serve", "serve_workload", "startup"])
def test_sweep_rejects_collisions_between_combined_names(mode, monkeypatch, tmp_path):
    """Distinct names on each axis can still alias after the axes are joined."""

    def unexpected_process(*args, **kwargs):
        pytest.fail("A child process was started before result-path validation")

    monkeypatch.setattr(subprocess, "Popen", unexpected_process)
    marker = "STARTUP" if mode == "startup" else "BENCH"
    serve_params = ParameterSweep.read_from_dict({f"a-{marker}--b": {}, "a": {}})
    bench_params = ParameterSweep.read_from_dict({"c": {}, f"b-{marker}--c": {}})

    with pytest.raises(ValueError, match="same result directory"):
        _run_sweep(mode, tmp_path, serve_params, bench_params, dry_run=False)

    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["serve", "serve_workload", "startup"])
def test_sweep_accepts_distinct_result_paths(mode, tmp_path):
    params = ParameterSweep.read_from_dict({"scenario-a": {}, "scenario-b": {}})

    assert _run_sweep(mode, tmp_path, ParameterSweep.from_records([{}]), params) is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["serve", "serve_workload"])
def test_sweep_ignores_collisions_in_unselected_linked_combinations(mode, tmp_path):
    serve_params = ParameterSweep.from_records([{"max_model_len": 16}])
    bench_params = ParameterSweep.read_from_dict(
        {
            "experiment/a": {"random_input_len": 16},
            "experiment_a": {"random_input_len": 32},
        }
    )

    assert (
        _run_sweep(
            mode,
            tmp_path,
            serve_params,
            bench_params,
            link_vars=[("max_model_len", "random_input_len")],
        )
        is None
    )


def test_sweep_preserves_existing_paths_for_cached_results(tmp_path):
    params = ParameterSweep.read_from_dict(
        {"scenario-a": {"num_prompts": 10}, "scenario-b": {"num_prompts": 20}}
    )
    for name, completed in [("scenario-a", 10), ("scenario-b", 20)]:
        directory = tmp_path / f"BENCH--{name}"
        directory.mkdir()
        (directory / "run=0.json").write_text(json.dumps({"completed": completed}))
        (directory / "summary.json").write_text("[]")

    frame = _run_sweep(
        "serve", tmp_path, ParameterSweep.from_records([{}]), params, dry_run=False
    )

    assert frame["completed"].tolist() == [10, 20]
    assert frame["num_prompts"].tolist() == [10, 20]
    assert sorted(
        directory.name for directory in tmp_path.iterdir() if directory.is_dir()
    ) == [
        "BENCH--scenario-a",
        "BENCH--scenario-b",
    ]
