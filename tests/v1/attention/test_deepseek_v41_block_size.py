"""
Pytest-compatible tests for DeepSeek-V4.1 block size fix.
These tests can be run with: pytest tests/v1/attention/test_deepseek_v41_block_size.py
"""

from unittest import mock

import pytest

from vllm.v1.attention.backend import MultipleOf


class TestDeepseekV41BlockSizeFix:
    """Test suite for DeepSeek-V4.1 block size fix on Blackwell (SM120/SM121)."""

    def test_deepseek_v4_sparse_mla_backend_sm90(self):
        """Test DeepseekV4SparseMLABackend returns [64] on SM90 (Hopper)."""
        from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4SparseMLABackend

        with mock.patch(
            "vllm.models.deepseek_v4_1.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = (
                lambda fam: fam == 90
            )
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [64]

    def test_deepseek_v4_sparse_mla_backend_sm120(self):
        """Test the sparse MLA backend returns [64] on SM120."""
        from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4SparseMLABackend

        with mock.patch(
            "vllm.models.deepseek_v4_1.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = (
                lambda fam: fam == 120
            )
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [64], f"SM120: Expected [64], got {sizes}"

    def test_deepseek_v4_sparse_mla_backend_sm121(self):
        """Test the sparse MLA backend returns [64] on SM121."""
        from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4SparseMLABackend

        with mock.patch(
            "vllm.models.deepseek_v4_1.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = (
                lambda fam: fam == 121
            )
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [64], f"SM121: Expected [64], got {sizes}"

    def test_deepseek_v4_sparse_mla_backend_other_archs(self):
        """Test DeepseekV4SparseMLABackend returns [128] on other architectures."""
        from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4SparseMLABackend

        with mock.patch(
            "vllm.models.deepseek_v4_1.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: False
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [128]

    def test_deepseek_v41_indexer_backend_sm120(self):
        """Test DeepseekV41IndexerBackend returns [64] on SM120."""
        from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend

        with mock.patch(
            "vllm.v1.attention.backends.mla.indexer.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = (
                lambda fam: fam == 120
            )
            sizes = DeepseekV41IndexerBackend.get_supported_kernel_block_sizes()
            assert sizes == [64]

    def test_deepseek_v41_indexer_backend_sm121(self):
        """Test DeepseekV41IndexerBackend returns [64] on SM121."""
        from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend

        with mock.patch(
            "vllm.v1.attention.backends.mla.indexer.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = (
                lambda fam: fam == 121
            )
            sizes = DeepseekV41IndexerBackend.get_supported_kernel_block_sizes()
            assert sizes == [64]

    def test_swa_backend_supports_block_size_64(self):
        """Test that DeepseekSparseSWABackend supports block_size=64."""
        from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend

        sizes = DeepseekSparseSWABackend.get_supported_kernel_block_sizes()
        assert len(sizes) == 1
        assert isinstance(sizes[0], MultipleOf)
        assert sizes[0].base == 32
        # Verify 64 is compatible
        assert 64 % sizes[0].base == 0

    def test_select_common_block_size_sm120(self):
        """Test select_common_block_size finds 64 on SM120."""
        from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
            DeepseekV4FlashInferMLASparseBackend,
        )
        from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
        from vllm.v1.worker.utils import select_common_block_size

        with (
            mock.patch(
                "vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse.current_platform"
            ) as mock_sparse,
            mock.patch(
                "vllm.v1.attention.backends.mla.indexer.current_platform"
            ) as mock_indexer,
        ):
            mock_sparse.is_device_capability_family.side_effect = lambda fam: fam == 120
            mock_indexer.is_device_capability_family.side_effect = (
                lambda fam: fam == 120
            )

            backends = [DeepseekV4FlashInferMLASparseBackend, DeepseekV41IndexerBackend]
            result = select_common_block_size(128, backends)

            assert result == 64, f"Expected 64, got {result}"

    def test_select_common_block_size_sm121(self):
        """Test select_common_block_size finds 64 on SM121."""
        from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
            DeepseekV4FlashInferMLASparseBackend,
        )
        from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
        from vllm.v1.worker.utils import select_common_block_size

        with (
            mock.patch(
                "vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse.current_platform"
            ) as mock_sparse,
            mock.patch(
                "vllm.v1.attention.backends.mla.indexer.current_platform"
            ) as mock_indexer,
        ):
            mock_sparse.is_device_capability_family.side_effect = lambda fam: fam == 121
            mock_indexer.is_device_capability_family.side_effect = (
                lambda fam: fam == 121
            )

            backends = [DeepseekV4FlashInferMLASparseBackend, DeepseekV41IndexerBackend]
            result = select_common_block_size(128, backends)

            assert result == 64, f"Expected 64, got {result}"

    def test_select_common_block_size_sm90(self):
        """Test select_common_block_size still works on SM90."""
        from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4SparseMLABackend
        from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
        from vllm.v1.worker.utils import select_common_block_size

        with (
            mock.patch(
                "vllm.models.deepseek_v4_1.sparse_mla.current_platform"
            ) as mock_sparse,
            mock.patch(
                "vllm.v1.attention.backends.mla.indexer.current_platform"
            ) as mock_indexer,
        ):
            mock_sparse.is_device_capability_family.side_effect = lambda fam: fam == 90
            mock_indexer.is_device_capability_family.side_effect = lambda fam: fam == 90

            backends = [DeepseekV4SparseMLABackend, DeepseekV41IndexerBackend]
            result = select_common_block_size(128, backends)

            assert result == 64

    def test_no_error_on_sm120_with_all_backends(self):
        """Test that no ValueError is raised on SM120 with all backends."""
        from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import (
            DeepseekV4FlashInferMLASparseBackend,
        )
        from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
        from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
        from vllm.v1.worker.utils import select_common_block_size

        with (
            mock.patch(
                "vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse.current_platform"
            ) as mock_sparse,
            mock.patch(
                "vllm.v1.attention.backends.mla.indexer.current_platform"
            ) as mock_indexer,
        ):
            mock_sparse.is_device_capability_family.side_effect = lambda fam: fam == 120
            mock_indexer.is_device_capability_family.side_effect = (
                lambda fam: fam == 120
            )

            backends = [
                DeepseekV4FlashInferMLASparseBackend,
                DeepseekV41IndexerBackend,
                DeepseekSparseSWABackend,
            ]

            # Should NOT raise ValueError
            try:
                result = select_common_block_size(128, backends)
                assert result == 64
            except ValueError as e:
                pytest.fail(f"Unexpected ValueError: {e}")


class TestBlockSizeBackwardCompatibility:
    """Test backward compatibility with other architectures."""

    def test_sm80_still_returns_128(self):
        """Test SM80 (Ampere) still uses [128]."""
        from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4SparseMLABackend

        with mock.patch(
            "vllm.models.deepseek_v4_1.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: False
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [128]

    def test_cpu_still_works(self):
        """Test CPU backend still works."""
        from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4SparseMLABackend

        with mock.patch(
            "vllm.models.deepseek_v4_1.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: False
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [128]


class TestBlockSizeAlgorithm:
    """Test the core select_common_block_size algorithm."""

    def test_algorithm_handles_multiple_of_constraint(self):
        """Test algorithm correctly handles MultipleOf constraints."""
        from vllm.v1.attention.backend import AttentionBackend, MultipleOf
        from vllm.v1.worker.utils import select_common_block_size

        class FakeBackend1(AttentionBackend):
            @staticmethod
            def get_supported_kernel_block_sizes():
                return [64]

        class FakeBackend2(AttentionBackend):
            @staticmethod
            def get_supported_kernel_block_sizes():
                return [MultipleOf(32)]

        # 128 % 64 = 0, 128 % 32 = 0, so should return 64
        result = select_common_block_size(128, [FakeBackend1, FakeBackend2])
        assert result == 64

    def test_algorithm_no_common_size_error(self):
        """Test algorithm raises error when no common size exists."""
        from vllm.v1.attention.backend import AttentionBackend
        from vllm.v1.worker.utils import select_common_block_size

        class IncompatibleBackend(AttentionBackend):
            @staticmethod
            def get_supported_kernel_block_sizes():
                return [256]  # Only supports 256

        # 128 % 256 != 0, so should raise error
        with pytest.raises(ValueError, match="No common block size"):
            select_common_block_size(128, [IncompatibleBackend])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
