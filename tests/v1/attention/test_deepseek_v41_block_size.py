# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest-compatible tests for DeepSeek-V4.1 block size fix on SM120/SM121."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.v1.attention.backend import MultipleOf


class TestDeepseekV41BlockSizeFix:
    """Test suite for DeepSeek-V4.1 block size fix on Blackwell (SM120/SM121)."""

    def test_deepseek_v4_sparse_mla_backend_sm90(self):
        """Test DeepseekV4SparseMLABackend returns [64] on SM90 (Hopper)."""
        from vllm.models.deepseek_v41.sparse_mla import (
            DeepseekV4SparseMLABackend,
        )

        with mock.patch(
            "vllm.models.deepseek_v41.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: (
                fam == 90
            )
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [64]

    def test_deepseek_v4_sparse_mla_backend_sm120(self):
        """Test the sparse MLA backend returns [64] on SM120."""
        from vllm.models.deepseek_v41.sparse_mla import (
            DeepseekV4SparseMLABackend,
        )

        with mock.patch(
            "vllm.models.deepseek_v41.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: (
                fam == 120
            )
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [64], f"SM120: Expected [64], got {sizes}"

    def test_deepseek_v4_sparse_mla_backend_sm121(self):
        """Test the sparse MLA backend returns [64] on SM121."""
        from vllm.models.deepseek_v41.sparse_mla import (
            DeepseekV4SparseMLABackend,
        )

        with mock.patch(
            "vllm.models.deepseek_v41.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: (
                fam == 121
            )
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [64], f"SM121: Expected [64], got {sizes}"

    def test_deepseek_v4_sparse_mla_backend_other_archs(self):
        """Test DeepseekV4SparseMLABackend returns [128] on other architectures."""
        from vllm.models.deepseek_v41.sparse_mla import (
            DeepseekV4SparseMLABackend,
        )

        with mock.patch(
            "vllm.models.deepseek_v41.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: False
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [128]

    def test_deepseek_v4_flashinfer_sparse_backend_sm120(self):
        """Test DeepseekV4FlashInferMLASparseBackend returns [64, 128] on SM120/121."""
        from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
            DeepseekV4FlashInferMLASparseBackend,
        )

        with mock.patch(
            "vllm.models.deepseek_v41.nvidia.flashinfer_sparse.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: (
                fam == 120
            )
            sizes = (
                DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes()
            )
            assert sizes == [64, 128]

            mock_platform.is_device_capability_family.side_effect = lambda fam: (
                fam == 121
            )
            sizes = (
                DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes()
            )
            assert sizes == [64, 128]

            mock_platform.is_device_capability_family.side_effect = lambda fam: False
            sizes = (
                DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes()
            )
            assert sizes == [128]

    def test_deepseek_v41_indexer_backend_sm120(self):
        """Test DeepseekV41IndexerBackend returns [64, 128] on SM120."""
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV41IndexerBackend,
        )

        with mock.patch(
            "vllm.v1.attention.backends.mla.indexer.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: (
                fam == 120
            )
            sizes = DeepseekV41IndexerBackend.get_supported_kernel_block_sizes()
            assert sizes == [64, 128]

    def test_deepseek_v41_indexer_backend_sm121(self):
        """Test DeepseekV41IndexerBackend returns [64, 128] on SM121."""
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV41IndexerBackend,
        )

        with mock.patch(
            "vllm.v1.attention.backends.mla.indexer.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: (
                fam == 121
            )
            sizes = DeepseekV41IndexerBackend.get_supported_kernel_block_sizes()
            assert sizes == [64, 128]

    def test_indexer_spec_scales_block_size_for_compression(self):
        """Test ratio-2 layers scale spec block size to preserve 64 num_states."""
        from vllm.models.deepseek_v41.attention import DeepseekV4IndexerCache

        cache_ratio2 = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=64, cache_dtype="fp8"),
            compress_ratio=2,
            head_dim=132,
            dtype=torch.uint8,
            sparse_logits=False,
        )
        config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="fp8"))

        spec = DeepseekV4IndexerCache.get_kv_cache_spec(cache_ratio2, config)
        assert spec.block_size == 128
        assert spec.tokens_per_state == 2
        assert spec.num_states == 64

        cache_ratio1 = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=64, cache_dtype="fp8"),
            compress_ratio=1,
            head_dim=132,
            dtype=torch.uint8,
            sparse_logits=False,
        )
        spec1 = DeepseekV4IndexerCache.get_kv_cache_spec(cache_ratio1, config)
        assert spec1.block_size == 64
        assert spec1.tokens_per_state == 1
        assert spec1.num_states == 64

    def test_attention_spec_scales_block_size_for_compression(self):
        """Test compressed-KV attention spec scales block_size per ratio."""
        from vllm.models.deepseek_v41.attention import DeepseekV4Attention

        config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=64, cache_dtype="fp8_ds_mla"),
        )

        layer_ratio2 = SimpleNamespace(
            is_kv_source=True,
            compress_ratio=2,
            head_dim=512,
            kv_cache_dtype="fp8_ds_mla",
            kv_cache_torch_dtype=torch.uint8,
            kv_page_alignment=512,
            compressed_bytes_per_token=528,
        )
        spec2 = DeepseekV4Attention.get_kv_cache_spec(layer_ratio2, config)
        assert spec2 is not None
        assert spec2.block_size == 128
        assert spec2.tokens_per_state == 2
        assert spec2.num_states == 64

        layer_ratio1 = SimpleNamespace(
            is_kv_source=True,
            compress_ratio=1,
            head_dim=512,
            kv_cache_dtype="fp8_ds_mla",
            kv_cache_torch_dtype=torch.uint8,
            kv_page_alignment=512,
            compressed_bytes_per_token=528,
        )
        spec1 = DeepseekV4Attention.get_kv_cache_spec(layer_ratio1, config)
        assert spec1 is not None
        assert spec1.block_size == 64
        assert spec1.tokens_per_state == 1
        assert spec1.num_states == 64

        non_source = SimpleNamespace(is_kv_source=False)
        assert DeepseekV4Attention.get_kv_cache_spec(non_source, config) is None

    def test_spec_sm120_defensive_checks(self):
        """Test SM120/121 raises ValueError if states per page != 64."""
        from vllm.models.deepseek_v41.attention import (
            DeepseekV4Attention,
            DeepseekV4IndexerCache,
        )

        # Invalid global block size (32 gives 32 states for ratio-1)
        config_invalid = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=32, cache_dtype="fp8_ds_mla"),
        )
        layer = SimpleNamespace(
            is_kv_source=True,
            compress_ratio=1,
            head_dim=512,
            kv_cache_dtype="fp8_ds_mla",
            kv_cache_torch_dtype=torch.uint8,
            kv_page_alignment=512,
            compressed_bytes_per_token=528,
        )
        with mock.patch(
            "vllm.models.deepseek_v41.attention.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: (
                fam == 120
            )
            with pytest.raises(
                ValueError, match="SM120/SM121 requires 64 states per page"
            ):
                DeepseekV4Attention.get_kv_cache_spec(layer, config_invalid)

            cache_invalid = SimpleNamespace(
                cache_config=SimpleNamespace(block_size=32, cache_dtype="fp8"),
                compress_ratio=1,
                head_dim=132,
                dtype=torch.uint8,
                sparse_logits=False,
            )
            with pytest.raises(
                ValueError, match="SM120/SM121 requires 64 states per page"
            ):
                DeepseekV4IndexerCache.get_kv_cache_spec(cache_invalid, config_invalid)

    def test_swa_backend_supports_block_size_64(self):
        """Test that DeepseekSparseSWABackend supports block_size=64."""
        from vllm.v1.attention.backends.mla.sparse_swa import (
            DeepseekSparseSWABackend,
        )

        sizes = DeepseekSparseSWABackend.get_supported_kernel_block_sizes()
        assert len(sizes) == 1
        assert isinstance(sizes[0], MultipleOf)
        assert sizes[0].base == 32
        # Verify 64 is compatible
        assert 64 % sizes[0].base == 0

    def test_indexer_ratio2_group_no_split_on_sm120(self):
        """Test ratio-2 indexer (block_size=128) avoids block split on SM120."""
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV41IndexerBackend,
        )
        from vllm.v1.worker.utils import select_common_block_size

        with mock.patch(
            "vllm.v1.attention.backends.mla.indexer.current_platform"
        ) as mock_indexer:
            mock_indexer.is_device_capability_family.side_effect = lambda fam: (
                fam == 120
            )

            # Ratio-2 indexer group has manager block size 128
            result = select_common_block_size(128, [DeepseekV41IndexerBackend])
            assert result == 128, f"Expected 128 directly (no split), got {result}"

    def test_indexer_ratio1_group_no_split_on_sm120(self):
        """Test ratio-1 indexer group with block_size=64 avoids block split on SM120."""
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV41IndexerBackend,
        )
        from vllm.v1.worker.utils import select_common_block_size

        with mock.patch(
            "vllm.v1.attention.backends.mla.indexer.current_platform"
        ) as mock_indexer:
            mock_indexer.is_device_capability_family.side_effect = lambda fam: (
                fam == 120
            )

            # Ratio-1 indexer group has manager block size 64
            result = select_common_block_size(64, [DeepseekV41IndexerBackend])
            assert result == 64, f"Expected 64 directly, got {result}"

    def test_select_common_block_size_sm120(self):
        """Test select_common_block_size on SM120 without splitting."""
        from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
            DeepseekV4FlashInferMLASparseBackend,
        )
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV41IndexerBackend,
        )
        from vllm.v1.worker.utils import select_common_block_size

        with (
            mock.patch(
                "vllm.models.deepseek_v41.nvidia.flashinfer_sparse.current_platform"
            ) as mock_sparse,
            mock.patch(
                "vllm.v1.attention.backends.mla.indexer.current_platform"
            ) as mock_indexer,
        ):
            mock_sparse.is_device_capability_family.side_effect = lambda fam: fam == 120
            mock_indexer.is_device_capability_family.side_effect = lambda fam: (
                fam == 120
            )

            backends = [
                DeepseekV4FlashInferMLASparseBackend,
                DeepseekV41IndexerBackend,
            ]
            # Ratio 1 group takes 64 directly
            assert select_common_block_size(64, backends) == 64
            # Ratio 2 group takes 128 directly
            assert select_common_block_size(128, backends) == 128

    def test_select_common_block_size_sm121(self):
        """Test select_common_block_size on SM121 without splitting."""
        from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
            DeepseekV4FlashInferMLASparseBackend,
        )
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV41IndexerBackend,
        )
        from vllm.v1.worker.utils import select_common_block_size

        with (
            mock.patch(
                "vllm.models.deepseek_v41.nvidia.flashinfer_sparse.current_platform"
            ) as mock_sparse,
            mock.patch(
                "vllm.v1.attention.backends.mla.indexer.current_platform"
            ) as mock_indexer,
        ):
            mock_sparse.is_device_capability_family.side_effect = lambda fam: fam == 121
            mock_indexer.is_device_capability_family.side_effect = lambda fam: (
                fam == 121
            )

            backends = [
                DeepseekV4FlashInferMLASparseBackend,
                DeepseekV41IndexerBackend,
            ]
            # Ratio 1 group takes 64 directly
            assert select_common_block_size(64, backends) == 64
            # Ratio 2 group takes 128 directly
            assert select_common_block_size(128, backends) == 128

    def test_select_common_block_size_sm90(self):
        """Test select_common_block_size still works on SM90."""
        from vllm.models.deepseek_v41.sparse_mla import (
            DeepseekV4SparseMLABackend,
        )
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV41IndexerBackend,
        )
        from vllm.v1.worker.utils import select_common_block_size

        with (
            mock.patch(
                "vllm.models.deepseek_v41.sparse_mla.current_platform"
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
        from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
            DeepseekV4FlashInferMLASparseBackend,
        )
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV41IndexerBackend,
        )
        from vllm.v1.attention.backends.mla.sparse_swa import (
            DeepseekSparseSWABackend,
        )
        from vllm.v1.worker.utils import select_common_block_size

        with (
            mock.patch(
                "vllm.models.deepseek_v41.nvidia.flashinfer_sparse.current_platform"
            ) as mock_sparse,
            mock.patch(
                "vllm.v1.attention.backends.mla.indexer.current_platform"
            ) as mock_indexer,
        ):
            mock_sparse.is_device_capability_family.side_effect = lambda fam: fam == 120
            mock_indexer.is_device_capability_family.side_effect = lambda fam: (
                fam == 120
            )

            backends = [
                DeepseekV4FlashInferMLASparseBackend,
                DeepseekV41IndexerBackend,
                DeepseekSparseSWABackend,
            ]

            try:
                assert select_common_block_size(64, backends) == 64
                assert select_common_block_size(128, backends) == 128
            except ValueError as e:
                pytest.fail(f"Unexpected ValueError: {e}")


class TestBlockSizeBackwardCompatibility:
    """Test backward compatibility with other architectures."""

    def test_sm80_still_returns_128(self):
        """Test SM80 (Ampere) still uses [128]."""
        from vllm.models.deepseek_v41.sparse_mla import (
            DeepseekV4SparseMLABackend,
        )

        with mock.patch(
            "vllm.models.deepseek_v41.sparse_mla.current_platform"
        ) as mock_platform:
            mock_platform.is_device_capability_family.side_effect = lambda fam: False
            sizes = DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            assert sizes == [128]

    def test_cpu_still_works(self):
        """Test CPU backend still works."""
        from vllm.models.deepseek_v41.sparse_mla import (
            DeepseekV4SparseMLABackend,
        )

        with mock.patch(
            "vllm.models.deepseek_v41.sparse_mla.current_platform"
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
