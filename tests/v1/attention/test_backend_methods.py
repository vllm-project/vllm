# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backend import AttentionBackend, MultipleOf


def _make_test_backend(supported_sizes: list):
    class _TestBackend(AttentionBackend):
        @staticmethod
        def get_supported_kernel_block_sizes():
            return supported_sizes

        @staticmethod
        def get_name():
            return "TEST"

        @staticmethod
        def get_impl_cls():
            return None

        @staticmethod
        def get_builder_cls():
            return None

        @staticmethod
        def get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str="auto",
        ):
            return num_blocks, block_size, num_kv_heads, head_size

    return _TestBackend


def test_empty_supported_sizes_returns_None():
    backend = _make_test_backend([])
    assert backend.get_preferred_block_size(None) is None


def test_non_empty_supported_sizes_returns_None():
    backend = _make_test_backend([16, 32, 64])
    assert backend.get_preferred_block_size(None) is None


def test_empty_supported_sizes_returns_default():
    backend = _make_test_backend([])
    assert backend.get_preferred_block_size(16) == 16
    assert backend.get_preferred_block_size(256) == 256


def test_default_is_directly_supported():
    backend = _make_test_backend([16, 32, 64])
    assert backend.get_preferred_block_size(32) == 32


def test_default_is_multiple_of_supported():
    backend = _make_test_backend([MultipleOf(16)])
    assert backend.get_preferred_block_size(128) == 128


def test_default_is_multiple_of_1_always_supported():
    backend = _make_test_backend([MultipleOf(1)])
    assert backend.get_preferred_block_size(17) == 17


def test_default_not_supported_falls_back_to_smallest_int():
    backend = _make_test_backend([64, 128])
    assert backend.get_preferred_block_size(32) == 64


def test_default_not_supported_falls_back_to_smallest_multiple_of_base():
    backend = _make_test_backend([MultipleOf(32)])
    assert backend.get_preferred_block_size(40) == 32


def test_mixed_sizes_with_supported_default():
    backend = _make_test_backend([64, MultipleOf(16)])
    assert backend.get_preferred_block_size(48) == 48


def test_mixed_sizes_with_unsupported_default():
    backend = _make_test_backend([64, MultipleOf(32)])
    assert backend.get_preferred_block_size(48) == 32


def test_larger_default_that_is_multiple_of_one():
    backend = _make_test_backend([128, 64])
    assert backend.get_preferred_block_size(256) == 256
