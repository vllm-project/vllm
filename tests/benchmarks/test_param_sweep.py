# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import tempfile
from pathlib import Path

import pytest

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
            (
                {"compilation_config.use_inductor": False},
                "--compilation-config.use_inductor=false",
            ),
            (
                {"compilation_config.use_inductor": True},
                "--compilation-config.use_inductor=true",
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
        records = [
            {"max_tokens": 100, "temperature": 0.7},
            {"max_tokens": 200, "temperature": 0.9},
        ]
        sweep = ParameterSweep.from_records(records)
        assert len(sweep) == 2
        assert sweep[0]["max_tokens"] == 100
        assert sweep[1]["max_tokens"] == 200

    def test_read_from_dict(self):
        """Test creating ParameterSweep from a dict format."""
        data = {
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
        records = [
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
        data = {
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
        records = [
            {"_benchmark_name": "exp1", "max_tokens": 100},
            {"_benchmark_name": "exp1", "max_tokens": 200},
        ]

        with pytest.raises(ValueError, match="Duplicate _benchmark_name values"):
            ParameterSweep.from_records(records)

    def test_unique_benchmark_names_multiple_duplicates(self):
        """Test validation with multiple duplicate names."""
        records = [
            {"_benchmark_name": "exp1", "max_tokens": 100},
            {"_benchmark_name": "exp1", "max_tokens": 200},
            {"_benchmark_name": "exp2", "max_tokens": 300},
            {"_benchmark_name": "exp2", "max_tokens": 400},
        ]

        with pytest.raises(ValueError, match="Duplicate _benchmark_name values"):
            ParameterSweep.from_records(records)

    def test_no_benchmark_names_allowed(self):
        """Test that records without _benchmark_name are allowed."""
        records = [
            {"max_tokens": 100, "temperature": 0.7},
            {"max_tokens": 200, "temperature": 0.9},
        ]
        sweep = ParameterSweep.from_records(records)
        assert len(sweep) == 2

    def test_mixed_benchmark_names_allowed(self):
        """Test that mixing records with and without _benchmark_name is allowed."""
        records = [
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
