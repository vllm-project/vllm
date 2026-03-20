# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for TpKVTopology with various attention backend configurations.

These tests validate layout detection, block_size_position, and multi-backend
model behavior without loading any models. We use mock backends that replicate
the get_kv_cache_shape signatures of real backends.

Backend shape families:
  - FlashAttn-like:   (2, N, B, H, D)     -- KV-first
  - FlashInfer-like:  (N, 2, B, H, D)     -- blocks-first
  - MLA-like:         (N, B, D)            -- 3-dim, no KV split
  - Mamba-like:       NotImplementedError  -- no KV cache shape
  - TritonAttn-like:  (N, 2, B, H, D)     -- blocks-first (same as FI)
"""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import TpKVTopology
from vllm.v1.attention.backend import AttentionBackend


# ---------------------------------------------------------------------------
# Mock Attention Backends
# ---------------------------------------------------------------------------
class MockFlashAttnBackend(AttentionBackend):
    """Mimics FlashAttentionBackend: shape = (2, N, B, H, D)"""

    @staticmethod
    def get_name() -> str:
        return "MOCK_FLASH_ATTN"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # HND cross-layer: (num_blocks, num_kv_heads, num_layers, 2,
            #                    block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        return (0, 1, 3, 2, 4)


class MockFlashInferBackend(AttentionBackend):
    """Mimics FlashInferBackend: shape = (N, 2, B, H, D)"""

    @staticmethod
    def get_name() -> str:
        return "MOCK_FLASHINFER"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)


class MockTritonAttnBackend(AttentionBackend):
    """Mimics TritonAttentionBackend: shape = (N, 2, B, H, D) -- same as FI"""

    @staticmethod
    def get_name() -> str:
        return "MOCK_TRITON_ATTN"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


class MockMLABackend(AttentionBackend):
    """Mimics MLA backends (FlashMLA, etc.): shape = (N, B, D) -- 3 dims"""

    @staticmethod
    def get_name() -> str:
        return "MOCK_MLA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)


class MockMambaBackend(AttentionBackend):
    """Mimics Mamba backends: get_kv_cache_shape is not implemented."""

    @staticmethod
    def get_name() -> str:
        return "MOCK_MAMBA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        raise NotImplementedError("Mamba backends do not have a KV cache shape")


# A ChunkedLocal backend that inherits FA's get_kv_cache_shape,
# exactly as the real ChunkedLocalAttention backend does.
class MockChunkedLocalFABackend(MockFlashAttnBackend):
    """
    Mimics ChunkedLocalAttention backed by FlashAttn.
    Inherits get_kv_cache_shape from FlashAttn -- same layout.
    """

    @staticmethod
    def get_name() -> str:
        return "MOCK_CHUNKED_LOCAL_FA"


class MockCPUAttnBackend(AttentionBackend):
    """
    Mimics CPU attention backend: shape = (2, N, H, B, D)
    Note different position of block_size vs num_kv_heads compared to FA.
    """

    @staticmethod
    def get_name() -> str:
        return "MOCK_CPU_ATTN"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)


# ---------------------------------------------------------------------------
# Helper to build TpKVTopology with minimal required fields
# ---------------------------------------------------------------------------
def make_topo(
    attn_backend: type[AttentionBackend],
    is_mla: bool = False,
    tp_rank: int = 0,
    tp_size: int = 1,
    total_num_kv_heads: int = 8,
    block_size: int = 16,
    tensor_shape: torch.Size | None = None,
    engine_id: str = "test-engine",
) -> TpKVTopology:
    remote_tp_size = {engine_id: tp_size}
    remote_block_size = {engine_id: block_size}
    return TpKVTopology(
        tp_rank=tp_rank,
        engine_id=engine_id,
        remote_tp_size=remote_tp_size,
        remote_block_size=remote_block_size,
        is_mla=is_mla,
        total_num_kv_heads=total_num_kv_heads,
        attn_backend=attn_backend,
        tensor_shape=tensor_shape,
    )


# ===================================================================
# 1. Layout Detection Tests
# ===================================================================
class TestLayoutDetection:
    """Test is_kv_layout_blocks_first and split_k_and_v for each backend."""

    def test_flash_attn_standard(self):
        """FA: (2, N, B, H, D) -> blocks_first=False, split_k_and_v=True"""
        topo = make_topo(MockFlashAttnBackend)
        assert topo.is_kv_layout_blocks_first is False
        assert topo.split_k_and_v is True
        assert topo.cross_layers_blocks is False

    def test_flashinfer_standard(self):
        """FI: (N, 2, B, H, D) -> blocks_first=True, split_k_and_v=False"""
        topo = make_topo(MockFlashInferBackend)
        assert topo.is_kv_layout_blocks_first is True
        assert topo.split_k_and_v is False
        assert topo.cross_layers_blocks is False

    def test_triton_attn_standard(self):
        """Triton: (N, 2, B, H, D) -> same as FI (blocks_first=True)"""
        topo = make_topo(MockTritonAttnBackend)
        assert topo.is_kv_layout_blocks_first is True
        assert topo.split_k_and_v is False

    def test_flash_attn_mla(self):
        """FA with MLA: blocks_first=False, split_k_and_v=False (MLA overrides)"""
        topo = make_topo(MockFlashAttnBackend, is_mla=True)
        assert topo.is_kv_layout_blocks_first is False
        assert topo.split_k_and_v is False

    def test_flashinfer_mla(self):
        """FI with MLA: blocks_first=True, split_k_and_v=False"""
        topo = make_topo(MockFlashInferBackend, is_mla=True)
        assert topo.is_kv_layout_blocks_first is True
        assert topo.split_k_and_v is False

    def test_mla_backend_3dim(self):
        """
        Pure MLA backend (3-dim shape): blocks_first=False.
        Shape is (N, B, D) -- 3 dims, first dim is num_blocks=1 (mock),
        so the 5-dim blocks_first check fails.
        """
        topo = make_topo(MockMLABackend, is_mla=True, total_num_kv_heads=1)
        assert topo.is_kv_layout_blocks_first is False
        assert topo.split_k_and_v is False

    def test_flash_attn_cross_layers(self):
        """
        FA with cross-layer blocks: tensor_shape has one extra dim.
        Shape from backend = (2, 1, 16, 1, 1) -> 5 dims
        tensor_shape = (80, 2, 1, 16, 1, 1) -> 6 dims = 5 + 1
        => cross_layers_blocks=True, split_k_and_v=False
        """
        kv_shape = MockFlashAttnBackend.get_kv_cache_shape(
            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
        )
        cross_layer_shape = torch.Size((80,) + kv_shape)
        topo = make_topo(MockFlashAttnBackend, tensor_shape=cross_layer_shape)
        assert topo.cross_layers_blocks is True
        assert topo.split_k_and_v is False

    def test_flashinfer_cross_layers(self):
        """FI with cross-layer blocks."""
        kv_shape = MockFlashInferBackend.get_kv_cache_shape(
            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
        )
        cross_layer_shape = torch.Size((80,) + kv_shape)
        topo = make_topo(MockFlashInferBackend, tensor_shape=cross_layer_shape)
        assert topo.cross_layers_blocks is True
        assert topo.split_k_and_v is False

    def test_no_cross_layers_same_ndim(self):
        """
        When tensor_shape has same ndim as kv_cache_shape,
        cross_layers_blocks should be False.
        """
        kv_shape = MockFlashAttnBackend.get_kv_cache_shape(
            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
        )
        topo = make_topo(
            MockFlashAttnBackend, tensor_shape=torch.Size(kv_shape)
        )
        assert topo.cross_layers_blocks is False

    def test_cpu_attn_layout(self):
        """
        CPU attention: (2, N, H, B, D).
        Not blocks_first (first dim is 2 with num_blocks mocked to 1),
        and kv_cache_shape[0] != 1 when we have 5 dims.
        """
        topo = make_topo(MockCPUAttnBackend)
        # Shape with mocked values: (2, 1, 1, 16, 1)
        # First dim = 2 (not 1), and len=5, so blocks_first check:
        # len == 5 and shape[0] == 1? -> 5 dims but shape[0]=2 -> False
        assert topo.is_kv_layout_blocks_first is False
        assert topo.split_k_and_v is True


# ===================================================================
# 2. Block Size Position Tests
# ===================================================================
class TestBlockSizePosition:
    """
    Verify block_size_position is correctly detected.
    block_size_position is a negative index into the shape indicating
    where the block_size dimension lives.
    """

    def test_flash_attn_block_size_position(self):
        """FA shape: (2, N, B=16, H, D) -> B is at index 2, negative = -3"""
        topo = make_topo(MockFlashAttnBackend)
        assert topo.block_size_position == -3

    def test_flashinfer_block_size_position(self):
        """FI shape: (N, 2, B=16, H, D) -> B is at index 2, negative = -3"""
        topo = make_topo(MockFlashInferBackend)
        assert topo.block_size_position == -3

    def test_mla_block_size_position(self):
        """MLA shape: (N, B=16, D) -> B is at index 1, negative = -2"""
        topo = make_topo(MockMLABackend, is_mla=True, total_num_kv_heads=1)
        assert topo.block_size_position == -2

    def test_cpu_attn_block_size_position(self):
        """CPU shape: (2, N, H, B=16, D) -> B is at index 3, negative = -2"""
        topo = make_topo(MockCPUAttnBackend)
        assert topo.block_size_position == -2

    def test_flash_attn_cross_layers_block_size_position(self):
        """
        FA cross-layer: logical shape (L, 2, N, B, H, D), but after
        stride_order permutation for HND cross-layer, the physical position
        of B changes.

        Stride order for FA HND cross-layer: (2, 4, 0, 1, 3, 5)
        Logical shape: (80, 2, 1, 16, 1, 1)
        After permute: shape[2,4,0,1,3,5] = (1, 1, 80, 2, 16, 1)
        B=16 is at physical index 4 -> negative = -2
        """
        kv_shape = MockFlashAttnBackend.get_kv_cache_shape(
            num_blocks=1, block_size=16, num_kv_heads=1, head_size=1
        )
        cross_layer_shape = torch.Size((80,) + kv_shape)
        topo = make_topo(MockFlashAttnBackend, tensor_shape=cross_layer_shape)
        assert topo.cross_layers_blocks is True
        assert topo.block_size_position == -2


# ===================================================================
# 3. Multi-Backend Model Configuration Tests
# ===================================================================
class TestMultiBackendModels:
    """
    Test TpKVTopology behavior for model architectures that use
    different attention backends across layers.
    """

    def test_qwen3_like_uniform_full_attn(self):
        """
        Qwen3-like: All layers use FullAttentionSpec with FlashAttn backend.
        Single backend family, all properties should be standard FA.
        """
        topo = make_topo(MockFlashAttnBackend, is_mla=False)
        assert topo.is_kv_layout_blocks_first is False
        assert topo.split_k_and_v is True
        assert topo.cross_layers_blocks is False
        assert topo.block_size_position == -3

    def test_deepseek_v3_mla(self):
        """
        DeepSeek V3: All layers use MLAAttentionSpec with MLA backend.
        3-dim shape, is_mla=True, no KV split.
        """
        topo = make_topo(MockMLABackend, is_mla=True, total_num_kv_heads=1)
        assert topo.is_kv_layout_blocks_first is False
        assert topo.split_k_and_v is False
        assert topo.cross_layers_blocks is False
        assert topo.block_size_position == -2

    def test_llama4_hybrid_full_and_chunked(self):
        """
        Llama4: Mix of FullAttentionSpec (global NoPE layers) and
        ChunkedLocalAttentionSpec (local RoPE layers).

        Both backends inherit FlashAttn's get_kv_cache_shape, so
        constructing TpKVTopology with either backend gives the same result.
        This test documents that FA and ChunkedLocal-FA are interchangeable
        for topology purposes.
        """
        topo_fa = make_topo(MockFlashAttnBackend, is_mla=False)
        topo_chunked = make_topo(MockChunkedLocalFABackend, is_mla=False)

        # Both should produce identical topology properties
        assert topo_fa.is_kv_layout_blocks_first == topo_chunked.is_kv_layout_blocks_first
        assert topo_fa.split_k_and_v == topo_chunked.split_k_and_v
        assert topo_fa.cross_layers_blocks == topo_chunked.cross_layers_blocks
        assert topo_fa.block_size_position == topo_chunked.block_size_position

        # Confirm they're standard FA properties
        assert topo_fa.is_kv_layout_blocks_first is False
        assert topo_fa.split_k_and_v is True

    def test_gemma3_sliding_window(self):
        """
        Gemma3: All layers use FullAttentionSpec (some with sliding_window set).
        From TpKVTopology's perspective, sliding_window doesn't change the
        backend or cache shape. All layers use the same FA backend.
        """
        # sliding_window is a KVCacheSpec concern, not a backend shape concern
        topo = make_topo(MockFlashAttnBackend, is_mla=False)
        assert topo.is_kv_layout_blocks_first is False
        assert topo.split_k_and_v is True
        assert topo.block_size_position == -3

    def test_jamba_hybrid_mamba_backend_crashes(self):
        """
        Jamba-like hybrid: If get_current_attn_backend() returns a Mamba
        backend (because the first layer is Mamba), TpKVTopology construction
        crashes because Mamba backends don't implement get_kv_cache_shape.

        This documents the current limitation that NIXL cannot work with
        models where the first layer is a Mamba layer.
        """
        with pytest.raises(NotImplementedError):
            make_topo(MockMambaBackend, is_mla=False)

    def test_jamba_hybrid_attention_first_works(self):
        """
        Jamba-like hybrid: If the first layer is an attention layer,
        get_current_attn_backend() returns FA, and TpKVTopology works.
        The Mamba layers are simply not registered with NIXL (they use
        separate state management).
        """
        # Simulates the case where first layer happens to be attention
        topo = make_topo(MockFlashAttnBackend, is_mla=False)
        assert topo.is_kv_layout_blocks_first is False
        assert topo.split_k_and_v is True

    def test_flashinfer_with_chunked_local_inheriting(self):
        """
        If a model uses ChunkedLocal attention backed by FlashInfer,
        verify the topology correctly detects the FI layout.
        """

        class MockChunkedLocalFIBackend(MockFlashInferBackend):
            @staticmethod
            def get_name() -> str:
                return "MOCK_CHUNKED_LOCAL_FI"

        topo = make_topo(MockChunkedLocalFIBackend, is_mla=False)
        assert topo.is_kv_layout_blocks_first is True
        assert topo.split_k_and_v is False

    def test_mixed_fa_and_fi_backends_differ(self):
        """
        Hypothetical model with both FA and FI layers.
        TpKVTopology constructed with FA vs FI gives different properties.
        This documents why a single backend assumption matters.
        """
        topo_fa = make_topo(MockFlashAttnBackend, is_mla=False)
        topo_fi = make_topo(MockFlashInferBackend, is_mla=False)

        # Key property that differs between the two
        assert topo_fa.is_kv_layout_blocks_first is False
        assert topo_fi.is_kv_layout_blocks_first is True

        # split_k_and_v also differs
        assert topo_fa.split_k_and_v is True
        assert topo_fi.split_k_and_v is False

        # block_size_position is the same though
        assert topo_fa.block_size_position == topo_fi.block_size_position == -3


# ===================================================================
# 4. get_current_attn_backend Behavior Tests
# ===================================================================
class TestGetCurrentAttnBackend:
    """
    Test get_current_attn_backend behavior with mocked static_forward_context.
    """

    def test_returns_first_layers_backend(self):
        """
        get_current_attn_backend iterates static_forward_context (dict order)
        and returns the first layer's backend.
        """
        from unittest.mock import MagicMock, patch

        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_current_attn_backend,
        )
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )

        # Create mock layers with different backends
        layer0 = MagicMock(spec=AttentionLayerBase)
        layer0.get_attn_backend.return_value = MockFlashAttnBackend

        layer1 = MagicMock(spec=AttentionLayerBase)
        layer1.get_attn_backend.return_value = MockFlashInferBackend

        mock_context = {"attn_layer_0": layer0, "attn_layer_1": layer1}

        mock_config = MagicMock()
        mock_config.compilation_config.static_forward_context = mock_context

        backend = get_current_attn_backend(mock_config)
        assert backend is MockFlashAttnBackend

    def test_returns_second_when_first_is_different(self):
        """
        Verify that only the FIRST layer's backend is returned,
        even if subsequent layers use a different backend.
        """
        from unittest.mock import MagicMock

        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_current_attn_backend,
        )
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )

        # First layer is FI, second is FA
        layer0 = MagicMock(spec=AttentionLayerBase)
        layer0.get_attn_backend.return_value = MockFlashInferBackend

        layer1 = MagicMock(spec=AttentionLayerBase)
        layer1.get_attn_backend.return_value = MockFlashAttnBackend

        mock_context = {"layer_0": layer0, "layer_1": layer1}

        mock_config = MagicMock()
        mock_config.compilation_config.static_forward_context = mock_context

        backend = get_current_attn_backend(mock_config)
        # Should be the first one
        assert backend is MockFlashInferBackend

    def test_mamba_first_layer_returns_mamba(self):
        """
        If the first layer is Mamba, get_current_attn_backend returns
        the Mamba backend. This would cause TpKVTopology to crash.

        This documents the current problematic behavior that needs fixing:
        get_current_attn_backend should skip non-attention backends.
        """
        from unittest.mock import MagicMock

        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_current_attn_backend,
        )
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )

        # First layer is Mamba, second is FA
        mamba_layer = MagicMock(spec=AttentionLayerBase)
        mamba_layer.get_attn_backend.return_value = MockMambaBackend

        attn_layer = MagicMock(spec=AttentionLayerBase)
        attn_layer.get_attn_backend.return_value = MockFlashAttnBackend

        mock_context = {"mamba_layer_0": mamba_layer, "attn_layer_0": attn_layer}

        mock_config = MagicMock()
        mock_config.compilation_config.static_forward_context = mock_context

        backend = get_current_attn_backend(mock_config)
        # Current behavior: returns Mamba (the first layer's backend)
        assert backend is MockMambaBackend

        # This will crash TpKVTopology:
        with pytest.raises(NotImplementedError):
            make_topo(backend, is_mla=False)

    def test_fallback_when_no_layers(self):
        """
        When static_forward_context is empty, get_current_attn_backend
        falls back to the attention selector.
        """
        from unittest.mock import MagicMock, patch

        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_current_attn_backend,
        )

        mock_config = MagicMock()
        mock_config.compilation_config.static_forward_context = {}
        mock_config.model_config.get_head_size.return_value = 64
        mock_config.model_config.dtype = torch.float16
        mock_config.cache_config.cache_dtype = "auto"
        mock_config.cache_config.block_size = 16
        mock_config.model_config.use_mla = False

        with patch(
            "vllm.distributed.kv_transfer.kv_connector.utils.get_attn_backend"
        ) as mock_selector:
            mock_selector.return_value = MockFlashAttnBackend
            backend = get_current_attn_backend(mock_config)
            assert backend is MockFlashAttnBackend
            mock_selector.assert_called_once()

    def test_all_layers_same_backend_consistency(self):
        """
        When all layers use the same backend, any layer can be used
        to construct TpKVTopology with identical results.
        """
        from unittest.mock import MagicMock

        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_current_attn_backend,
        )
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )

        layers = {}
        for i in range(10):
            layer = MagicMock(spec=AttentionLayerBase)
            layer.get_attn_backend.return_value = MockFlashAttnBackend
            layers[f"layer_{i}"] = layer

        mock_config = MagicMock()
        mock_config.compilation_config.static_forward_context = layers

        backend = get_current_attn_backend(mock_config)
        assert backend is MockFlashAttnBackend

        # All produce the same topology
        topo = make_topo(backend)
        for layer in layers.values():
            other = make_topo(layer.get_attn_backend())
            assert topo.is_kv_layout_blocks_first == other.is_kv_layout_blocks_first
            assert topo.split_k_and_v == other.split_k_and_v
            assert topo.block_size_position == other.block_size_position
