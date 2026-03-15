# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Ray Compiled Graph buffer size configuration."""

import os
import unittest
from unittest.mock import patch


class TestRayCGraphBufferSize(unittest.TestCase):
    """Test that VLLM_RAY_CGRAPH_BUFFER_SIZE_BYTES is properly handled."""

    def test_default_buffer_size(self):
        """Test that default buffer size is 0 (use Ray default)."""
        with patch.dict(os.environ, {}, clear=True):
            from vllm.envs import environment_variables
            buffer_size = environment_variables["VLLM_RAY_CGRAPH_BUFFER_SIZE_BYTES"]()
            self.assertEqual(buffer_size, 0)

    def test_custom_buffer_size(self):
        """Test that custom buffer size is properly parsed."""
        with patch.dict(os.environ, {"VLLM_RAY_CGRAPH_BUFFER_SIZE_BYTES": "536870912"}):
            from vllm.envs import environment_variables
            buffer_size = environment_variables["VLLM_RAY_CGRAPH_BUFFER_SIZE_BYTES"]()
            self.assertEqual(buffer_size, 536870912)

class TestRayCGraphBufferSizeEnvMapping(unittest.TestCase):
    """Test that VLLM_RAY_CGRAPH_BUFFER_SIZE_BYTES maps to Ray's env var."""

    def test_env_var_mapping(self):
        """Test that setting VLLM_RAY_CGRAPH_BUFFER_SIZE_BYTES sets RAY_CGRAPH_buffer_size_bytes."""
        with patch.dict(os.environ, {"VLLM_RAY_CGRAPH_BUFFER_SIZE_BYTES": "536870912"}):
            from vllm.envs import environment_variables
            buffer_size = environment_variables["VLLM_RAY_CGRAPH_BUFFER_SIZE_BYTES"]()
            os.environ.setdefault("RAY_CGRAPH_buffer_size_bytes", str(buffer_size))
            self.assertEqual(os.environ["RAY_CGRAPH_buffer_size_bytes"], "536870912")

if __name__ == "__main__":
    unittest.main()
