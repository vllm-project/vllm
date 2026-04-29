# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JIT-optimized make_ndarray_with_pad."""

# ============================================================================
# IMPORTANT: Clean JIT cache BEFORE any imports to ensure fresh compilation
# ============================================================================
import os
import shutil

# Clean up stale lock files and compiled artifacts from previous runs
_jit_cache_base = os.path.expanduser("~/.cache/torch_extensions")
_custom_pad_cache = os.path.join(_jit_cache_base, "py312_cu128", "custom_pad_op")
if os.path.exists(_custom_pad_cache):
    # Remove lock file if stale (prevents hanging on load)
    lock_file = os.path.join(_custom_pad_cache, "lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)
        print(f"[Test Setup] Removed stale lock file: {lock_file}")

    # Force recompilation if env var is set (useful for testing JIT changes)
    if os.environ.get("VLLM_FORCE_JIT_RECOMPILE", "0") == "1":
        shutil.rmtree(_custom_pad_cache)
        print(f"[Test Setup] Force recompile: removed JIT cache {_custom_pad_cache}")

# ============================================================================
# Reset JIT state BEFORE importing make_ndarray_with_pad
# This ensures tests use the C++ extension, not the fallback
# ============================================================================
import vllm.utils.torch_utils as _torch_utils_module
_torch_utils_module._CUSTOM_PAD_OP_MODULE = None
_torch_utils_module._JIT_LOAD_FAILED = False

import numpy as np
import pytest
import torch

from vllm.utils.torch_utils import make_ndarray_with_pad


class TestMakeNdarrayWithPad:
    """Tests for the zero-allocation padding function."""

    def test_basic_int_padding(self):
        """Test basic integer list padding."""
        x = [[1, 2], [3, 4, 5], [6]]
        result = make_ndarray_with_pad(x, 0, np.int64)
        expected = np.array([[1, 2, 0], [3, 4, 5], [6, 0, 0]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_empty_input(self):
        """Test empty input list."""
        result = make_ndarray_with_pad([], 0, np.int64)
        assert result.shape == (0, 0)
        assert result.dtype == np.int64

    def test_all_empty_rows(self):
        """Test list with all empty inner lists."""
        result = make_ndarray_with_pad([[], [], []], 0, np.int64)
        assert result.shape == (3, 0)
        assert result.dtype == np.int64

    def test_float_dtype(self):
        """Test float32 dtype padding."""
        x = [[1.0, 2.5], [3.0]]
        result = make_ndarray_with_pad(x, 0.0, np.float32)
        expected = np.array([[1.0, 2.5], [3.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_float16_dtype(self):
        """Test float16 (half) dtype padding."""
        x = [[1, 2, 3], [4]]
        result = make_ndarray_with_pad(x, -1, np.float16)
        expected = np.array([[1, 2, 3], [4, -1, -1]], dtype=np.float16)
        np.testing.assert_array_equal(result, expected)

    def test_bool_dtype(self):
        """Test bool dtype padding - newly supported."""
        x = [[True, False], [True]]
        result = make_ndarray_with_pad(x, False, np.bool_)
        expected = np.array([[True, False], [True, False]], dtype=np.bool_)
        np.testing.assert_array_equal(result, expected)

    def test_int32_dtype(self):
        """Test int32 dtype padding."""
        x = [[100, 200], [300]]
        result = make_ndarray_with_pad(x, -1, np.int32)
        expected = np.array([[100, 200], [300, -1]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_int8_dtype(self):
        """Test int8 dtype padding."""
        x = [[1, 2], [3]]
        result = make_ndarray_with_pad(x, 0, np.int8)
        expected = np.array([[1, 2], [3, 0]], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_uint8_dtype(self):
        """Test uint8 dtype padding."""
        x = [[1, 2], [3]]
        result = make_ndarray_with_pad(x, 0, np.uint8)
        expected = np.array([[1, 2], [3, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_explicit_max_len(self):
        """Test with explicit max_len parameter."""
        x = [[1, 2], [3]]
        result = make_ndarray_with_pad(x, 0, np.int64, max_len=5)
        expected = np.array([[1, 2, 0, 0, 0], [3, 0, 0, 0, 0]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_max_len_smaller_than_actual(self):
        """Test max_len smaller than actual longest row - C++ raises RuntimeError."""
        x = [[1, 2, 3], [4]]
        exception_raised = False
        try:
            make_ndarray_with_pad(x, 0, np.int64, max_len=2)
        except RuntimeError:
            exception_raised = True
        assert exception_raised, "Should have raised RuntimeError"

    def test_different_pad_values(self):
        """Test various pad values."""
        x = [[1], [2, 3]]
        result = make_ndarray_with_pad(x, -999, np.int32)
        expected = np.array([[1, -999], [2, 3]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_single_element_rows(self):
        """Test all single-element inner lists."""
        x = [[1], [2], [3], [4]]
        result = make_ndarray_with_pad(x, 0, np.int64)
        expected = np.array([[1], [2], [3], [4]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_large_batch(self):
        """Test with a larger batch size."""
        x = [[i] * (i % 10 + 1) for i in range(100)]
        result = make_ndarray_with_pad(x, 0, np.int64)
        assert result.shape == (100, 10)
        # Verify first and last rows
        assert result[0, 0] == 0  # row 0: [0] padded to 10
        assert result[9, :10].tolist() == [9] * 10  # row 9: [9]*10, no padding

    def test_unsupported_dtype_fail_fast(self):
        """Test unsupported dtype raises ValueError - fail-fast behavior."""
        x = [[1]]
        exception_raised = False
        try:
            make_ndarray_with_pad(x, 0, np.str_)
        except ValueError:
            exception_raised = True
        assert exception_raised, "Should have raised ValueError"

    def test_mixed_int_float_in_int_tensor(self):
        """Test mixed int/float values in int tensor - works with truncation."""
        x = [[1, 2.5], [3]]
        result = make_ndarray_with_pad(x, 0, np.int64)
        expected = np.array([[1, 2], [3, 0]], dtype=np.int64)  # 2.5 truncated to 2
        np.testing.assert_array_equal(result, expected)

    def test_large_int_values(self):
        """Test large integer values within int64 range."""
        x = [[1000000000, -1000000000], [0]]
        result = make_ndarray_with_pad(x, 0, np.int64)
        expected = np.array([[1000000000, -1000000000], [0, 0]], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    import sys

    # Force re-import to use fresh state
    make_ndarray_with_pad = _torch_utils_module.make_ndarray_with_pad

    test_instance = TestMakeNdarrayWithPad()
    test_methods = [
        ("basic_int_padding", test_instance.test_basic_int_padding),
        ("empty_input", test_instance.test_empty_input),
        ("all_empty_rows", test_instance.test_all_empty_rows),
        ("float_dtype", test_instance.test_float_dtype),
        ("float16_dtype", test_instance.test_float16_dtype),
        ("bool_dtype", test_instance.test_bool_dtype),
        ("int32_dtype", test_instance.test_int32_dtype),
        ("int8_dtype", test_instance.test_int8_dtype),
        ("uint8_dtype", test_instance.test_uint8_dtype),
        ("explicit_max_len", test_instance.test_explicit_max_len),
        ("different_pad_values", test_instance.test_different_pad_values),
        ("single_element_rows", test_instance.test_single_element_rows),
        ("large_batch", test_instance.test_large_batch),
        ("mixed_int_float_in_int_tensor", test_instance.test_mixed_int_float_in_int_tensor),
        ("large_int_values", test_instance.test_large_int_values),
        ("max_len_smaller_than_actual", test_instance.test_max_len_smaller_than_actual),
        ("unsupported_dtype_fail_fast", test_instance.test_unsupported_dtype_fail_fast),
    ]

    passed = 0
    failed = 0

    for name, method in test_methods:
        try:
            method()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)