# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import tempfile
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

from vllm.benchmarks.sweep.param_sweep import ParameterSweep, ParameterSweepItem


class TestParameterSweepItem:
    """Test ParameterSweepItem functionality."""

    def test_nested_boolean_false(self):
        """Test that nested boolean false params use =false syntax."""
        item = ParameterSweepItem.from_record({
            'compilation_config.use_inductor_graph_partition': False
        })
        cmd = item.apply_to_cmd(['vllm', 'serve', 'model'])
        assert '--compilation-config.use_inductor_graph_partition=false' in cmd

    def test_nested_boolean_true(self):
        """Test that nested boolean true params use =true syntax."""
        item = ParameterSweepItem.from_record({
            'compilation_config.use_inductor_graph_partition': True
        })
        cmd = item.apply_to_cmd(['vllm', 'serve', 'model'])
        assert '--compilation-config.use_inductor_graph_partition=true' in cmd

    def test_non_nested_boolean_false(self):
        """Test that non-nested boolean false params use --no- prefix."""
        item = ParameterSweepItem.from_record({
            'enable_prefix_caching': False
        })
        cmd = item.apply_to_cmd(['vllm', 'serve', 'model'])
        assert '--no-enable-prefix-caching' in cmd

    def test_non_nested_boolean_true(self):
        """Test that non-nested boolean true params work correctly."""
        item = ParameterSweepItem.from_record({
            'enable_prefix_caching': True
        })
        cmd = item.apply_to_cmd(['vllm', 'serve', 'model'])
        assert '--enable-prefix-caching' in cmd

    def test_nested_dict_value(self):
        """Test that nested dict values are serialized as JSON."""
        item = ParameterSweepItem.from_record({
            'env': {'CUDA_VISIBLE_DEVICES': '0,1'}
        })
        cmd = item.apply_to_cmd(['vllm', 'serve', 'model'])
        assert '--env' in cmd
        # The dict should be JSON serialized
        idx = cmd.index('--env')
        assert json.loads(cmd[idx + 1]) == {'CUDA_VISIBLE_DEVICES': '0,1'}

    def test_string_value(self):
        """Test that string values are preserved."""
        item = ParameterSweepItem.from_record({'model': 'test-model'})
        cmd = item.apply_to_cmd(['vllm', 'serve'])
        assert '--model' in cmd
        assert 'test-model' in cmd

    def test_numeric_value(self):
        """Test that numeric values are converted to strings."""
        item = ParameterSweepItem.from_record({'max_tokens': 100})
        cmd = item.apply_to_cmd(['vllm', 'serve'])
        assert '--max-tokens' in cmd
        assert '100' in cmd


class TestParameterSweep:
    """Test ParameterSweep functionality."""

    def test_from_records_list(self):
        """Test creating ParameterSweep from a list of records."""
        records = [
            {'max_tokens': 100, 'temperature': 0.7},
            {'max_tokens': 200, 'temperature': 0.9}
        ]
        sweep = ParameterSweep.from_records(records)
        assert len(sweep) == 2
        assert sweep[0]['max_tokens'] == 100
        assert sweep[1]['max_tokens'] == 200

    def test_read_from_dict(self):
        """Test creating ParameterSweep from a dict format."""
        data = {
            'experiment1': {'max_tokens': 100, 'temperature': 0.7},
            'experiment2': {'max_tokens': 200, 'temperature': 0.9}
        }
        sweep = ParameterSweep.read_from_dict(data)
        assert len(sweep) == 2

        # Check that items have the name field
        names = {item['name'] for item in sweep}
        assert names == {'experiment1', 'experiment2'}

        # Check that parameters are preserved
        for item in sweep:
            if item['name'] == 'experiment1':
                assert item['max_tokens'] == 100
                assert item['temperature'] == 0.7
            elif item['name'] == 'experiment2':
                assert item['max_tokens'] == 200
                assert item['temperature'] == 0.9

    def test_read_json_list_format(self):
        """Test reading JSON file with list format."""
        records = [
            {'max_tokens': 100, 'temperature': 0.7},
            {'max_tokens': 200, 'temperature': 0.9}
        ]

        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False) as f:
            json.dump(records, f)
            temp_path = Path(f.name)

        try:
            sweep = ParameterSweep.read_json(temp_path)
            assert len(sweep) == 2
            assert sweep[0]['max_tokens'] == 100
            assert sweep[1]['max_tokens'] == 200
        finally:
            temp_path.unlink()

    def test_read_json_dict_format(self):
        """Test reading JSON file with dict format."""
        data = {
            'experiment1': {'max_tokens': 100, 'temperature': 0.7},
            'experiment2': {'max_tokens': 200, 'temperature': 0.9}
        }

        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            sweep = ParameterSweep.read_json(temp_path)
            assert len(sweep) == 2

            # Check that items have the name field
            names = {item['name'] for item in sweep}
            assert names == {'experiment1', 'experiment2'}
        finally:
            temp_path.unlink()


class TestParameterSweepItemKeyNormalization:
    """Test key normalization in ParameterSweepItem."""

    def test_underscore_to_hyphen_conversion(self):
        """Test that underscores are converted to hyphens in CLI."""
        item = ParameterSweepItem.from_record({'max_tokens': 100})
        cmd = item.apply_to_cmd(['vllm', 'serve'])
        assert '--max-tokens' in cmd

    def test_nested_key_preserves_suffix(self):
        """Test that nested keys preserve the suffix format."""
        # The suffix after the dot should preserve underscores
        item = ParameterSweepItem.from_record({
            'compilation_config.some_nested_param': 'value'
        })
        cmd = item.apply_to_cmd(['vllm', 'serve'])
        # The prefix (compilation_config) gets converted to hyphens,
        # but the suffix (some_nested_param) is preserved
        assert any('compilation-config.some_nested_param' in arg
                   for arg in cmd)

    def test_replace_existing_parameter(self):
        """Test that existing parameters in cmd are replaced."""
        item = ParameterSweepItem.from_record({'max_tokens': 200})
        cmd = item.apply_to_cmd(
            ['vllm', 'serve', '--max-tokens', '100', 'model'])
        # The 100 should be replaced with 200
        assert '--max-tokens' in cmd
        idx = cmd.index('--max-tokens')
        assert cmd[idx + 1] == '200'
        assert '100' not in cmd

    def test_replace_existing_boolean_parameter(self):
        """Test that existing boolean parameters are updated correctly."""
        item = ParameterSweepItem.from_record({'enable_prefix_caching': False})
        cmd = item.apply_to_cmd(
            ['vllm', 'serve', '--enable-prefix-caching', 'model'])
        # Should become --no-enable-prefix-caching
        assert '--no-enable-prefix-caching' in cmd
        assert '--enable-prefix-caching' not in cmd
