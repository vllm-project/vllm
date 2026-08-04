# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LMCache-side TRT-LLM integration components.

Covers ``EngineType.TRTLLM``, the new ``NB_NL_TWO_NH_BS_HS`` GPU KV
format, format-aware accessors in ``gpu_connector/utils.py``, the
``TRTLLMGPUConnector`` construction path, and the
``RawCudaIPCWrapper`` subclass relationship — without requiring
TensorRT-LLM to be installed.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType


def _has_lmc_ops() -> bool:
    try:
        # First Party
        import lmcache.c_ops  # noqa: F401

        return True
    except ImportError:
        return False


def _has_cuda() -> bool:
    return torch.cuda.is_available()


class TestEngineType:
    def test_trtllm_exists(self) -> None:
        assert hasattr(EngineType, "TRTLLM")
        assert EngineType.TRTLLM.value == "trtllm"

    def test_all_engine_types(self) -> None:
        expected = {"vllm", "sglang", "trtllm", "mock"}
        actual = {e.value for e in EngineType}
        assert expected == actual


class TestAssertContiguous:
    def test_contiguous_tensor_passes(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import assert_contiguous

        t = torch.zeros(2, 3, 4)
        assert_contiguous(t)  # no raise

    def test_non_contiguous_raises(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import assert_contiguous

        t = torch.zeros(2, 3, 4).transpose(0, 1)
        with pytest.raises(ValueError, match="not contiguous"):
            assert_contiguous(t)

    def test_nonzero_storage_offset_raises(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import assert_contiguous

        base = torch.zeros(8)
        view = base[2:6]  # storage_offset = 2
        with pytest.raises(ValueError, match="storage_offset"):
            assert_contiguous(view)


@pytest.mark.skipif(not _has_lmc_ops(), reason="lmcache C ops not built")
class TestGPUKVFormatEnum:
    def test_nb_nl_two_nh_bs_hs_exists(self) -> None:
        # First Party
        import lmcache.c_ops as lmc_ops

        assert hasattr(lmc_ops.EngineKVFormat, "NB_NL_TWO_NH_BS_HS")


@pytest.mark.skipif(not _has_lmc_ops(), reason="lmcache C ops not built")
class TestNormalizeTRTLLM:
    """``normalize_kv_and_discover_format`` for ``EngineType.TRTLLM``."""

    def test_4d_tensor_reshape_to_6d(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            LayoutHints,
            normalize_kv_and_discover_format,
        )
        import lmcache.c_ops as lmc_ops

        nb, nl, kv = 4, 2, 2
        nh, bs, hs = 8, 16, 64
        flat = nh * bs * hs

        t = torch.zeros(nb, nl, kv, flat, dtype=torch.bfloat16)
        layout_hints: LayoutHints = {
            "kv_layout": "HND",
            "num_kv_heads": nh,
            "tokens_per_block": bs,
            "head_dim": hs,
        }

        fmt, normalized = normalize_kv_and_discover_format(
            t, EngineType.TRTLLM, layout_hints=layout_hints
        )

        assert fmt == lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS
        assert isinstance(normalized, torch.Tensor)
        assert tuple(normalized.shape) == (nb, nl, kv, nh, bs, hs)

    def test_collapses_one_element_list(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            LayoutHints,
            normalize_kv_and_discover_format,
        )
        import lmcache.c_ops as lmc_ops

        nb, nl, kv, nh, bs, hs = 2, 2, 2, 4, 8, 32
        t = torch.zeros(nb, nl, kv, nh * bs * hs)
        layout_hints: LayoutHints = {
            "kv_layout": "HND",
            "num_kv_heads": nh,
            "tokens_per_block": bs,
            "head_dim": hs,
        }
        fmt, normalized = normalize_kv_and_discover_format(
            [t], EngineType.TRTLLM, layout_hints=layout_hints
        )
        assert fmt == lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS
        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == (nb, nl, kv, nh, bs, hs)

    def test_missing_layout_hints_raises(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format

        t = torch.zeros(2, 2, 2, 16)
        with pytest.raises(ValueError, match="num_kv_heads"):
            normalize_kv_and_discover_format(t, EngineType.TRTLLM, layout_hints={})

    def test_flat_dim_mismatch_raises(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            LayoutHints,
            normalize_kv_and_discover_format,
        )

        t = torch.zeros(2, 2, 2, 17)  # not divisible by 4*2*2 = 16
        layout_hints: LayoutHints = {
            "kv_layout": "HND",
            "num_kv_heads": 4,
            "tokens_per_block": 2,
            "head_dim": 2,
        }
        with pytest.raises(ValueError, match="flat dim"):
            normalize_kv_and_discover_format(
                t, EngineType.TRTLLM, layout_hints=layout_hints
            )


@pytest.mark.skipif(not _has_lmc_ops(), reason="lmcache C ops not built")
class TestAccessorsTRTLLM:
    """Format accessors for ``NB_NL_TWO_NH_BS_HS``."""

    def _tensor(
        self,
        nb: int = 4,
        nl: int = 3,
        kv: int = 2,
        nh: int = 8,
        bs: int = 16,
        hs: int = 64,
    ) -> torch.Tensor:
        return torch.zeros(nb, nl, kv, nh, bs, hs, dtype=torch.bfloat16)

    def test_get_num_layers(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import get_num_layers
        import lmcache.c_ops as lmc_ops

        t = self._tensor()
        assert get_num_layers(t, lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 3

    def test_get_num_blocks(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import get_num_blocks
        import lmcache.c_ops as lmc_ops

        t = self._tensor()
        assert get_num_blocks(t, lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 4

    def test_get_block_size(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import get_block_size
        import lmcache.c_ops as lmc_ops

        t = self._tensor()
        assert get_block_size(t, lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 16

    def test_get_num_heads(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import get_num_heads
        import lmcache.c_ops as lmc_ops

        t = self._tensor()
        assert get_num_heads(t, lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 8

    def test_get_head_size(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import get_head_size
        import lmcache.c_ops as lmc_ops

        t = self._tensor()
        assert get_head_size(t, lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 64

    def test_get_hidden_dim_size(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import get_hidden_dim_size
        import lmcache.c_ops as lmc_ops

        t = self._tensor()
        assert (
            get_hidden_dim_size(t, lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 8 * 64
        )

    def test_get_page_buffer_size(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import get_page_buffer_size
        import lmcache.c_ops as lmc_ops

        t = self._tensor()
        assert (
            get_page_buffer_size(t, lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 4 * 16
        )

    def test_get_dtype(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import get_dtype
        import lmcache.c_ops as lmc_ops

        t = self._tensor()
        assert get_dtype(t, lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == torch.bfloat16

    def test_get_group_data_ptrs_returns_single_base_pointer(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import get_group_data_ptrs
        import lmcache.c_ops as lmc_ops

        t = self._tensor()
        ptrs = get_group_data_ptrs(
            t, lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS, list(range(3))
        )
        assert len(ptrs) == 1
        assert ptrs[0] == t.data_ptr()

    def test_shape_description_strings(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.utils import (
            get_attention_backend,
            get_concrete_engine_kv_shape,
            get_engine_kv_shape_description,
        )
        import lmcache.c_ops as lmc_ops

        fmt = lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS
        assert get_engine_kv_shape_description(fmt) == "[NB, NL, 2, NH, BS, HS]"
        assert "TRT-LLM" in get_attention_backend(fmt)
        assert get_concrete_engine_kv_shape(self._tensor(), fmt) == (
            "[4, 3, 2, 8, 16, 64]"
        )


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for connector init")
@pytest.mark.skipif(not _has_lmc_ops(), reason="lmcache C ops not built")
class TestTRTLLMGPUConnector:
    def test_construct(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        device = torch.device("cuda:0")
        c = TRTLLMGPUConnector(
            num_kv_heads=2,
            head_dim=64,
            hidden_dim_size=128,
            num_layers=4,
            chunk_size=256,
            dtype=torch.bfloat16,
            device=device,
        )
        assert c.num_kv_heads == 2
        assert c.head_dim == 64
        assert c.kv_cache_tensor is None  # not registered yet

    def test_get_shape(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        device = torch.device("cuda:0")
        c = TRTLLMGPUConnector(
            num_kv_heads=2,
            head_dim=64,
            hidden_dim_size=128,
            num_layers=4,
            chunk_size=256,
            dtype=torch.bfloat16,
            device=device,
        )
        # Memory-obj layout: [2, num_layers, num_tokens, hidden_dim]
        assert tuple(c.get_shape(256)) == (2, 4, 256, 128)

    def test_register_kv_caches_with_4d_tensor(self) -> None:
        # First Party
        from lmcache.v1.gpu_connector.gpu_connectors import TRTLLMGPUConnector

        device = torch.device("cuda:0")
        nh, bs, hs = 2, 16, 64
        nb, nl, kv = 4, 4, 2
        flat = nh * bs * hs
        c = TRTLLMGPUConnector(
            num_kv_heads=nh,
            head_dim=hs,
            hidden_dim_size=nh * hs,
            num_layers=nl,
            chunk_size=256,
            dtype=torch.bfloat16,
            device=device,
        )
        t = torch.zeros(nb, nl, kv, flat, dtype=torch.bfloat16, device=device)
        c.register_kv_caches(t)
        assert c.kv_cache_tensor is not None
        assert tuple(c.kv_cache_tensor.shape) == (nb, nl, kv, nh, bs, hs)
        assert c.tokens_per_block == bs
        assert c.blocks_per_chunk == 256 // bs
        assert c.shape_desc is not None
        assert c.shape_desc.nl == nl
        assert c.shape_desc.nb == nb
        assert c.shape_desc.bs == bs
        assert c.shape_desc.nh == nh
        assert c.shape_desc.hs == hs


class TestRawCudaIPCWrapperType:
    """Subclass relationship — without invoking CUDA IPC syscalls."""

    def test_is_subclass(self) -> None:
        # First Party
        from lmcache.v1.multiprocess.custom_types import (
            DeviceIPCWrapper,
        )
        from lmcache.v1.platform.cuda.ipc_wrapper import RawCudaIPCWrapper

        assert issubclass(RawCudaIPCWrapper, DeviceIPCWrapper)

    def test_kvcache_typing_unchanged(self) -> None:
        """``KVCache = list[DeviceIPCWrapper]`` should accept concrete wrappers
        — load-bearing for msgspec ext-code reuse over ZMQ.
        """
        # First Party
        from lmcache.v1.multiprocess.custom_types import (
            DeviceIPCWrapper,
            KVCache,
        )
        from lmcache.v1.platform.cuda.ipc_wrapper import RawCudaIPCWrapper

        assert KVCache == list[DeviceIPCWrapper]
        # Static check substitute: a concrete wrapper fits the list type.
        assert issubclass(RawCudaIPCWrapper, DeviceIPCWrapper)

    def test_serde_uses_shared_ext_code(self) -> None:
        """Ext code 1 dispatches via ``DeviceIPCWrapper`` for all wrappers."""
        # First Party
        from lmcache.v1.multiprocess import custom_types

        registry = custom_types._CUSTOMERIZED_SERIALIZERS  # noqa: PLC2701
        assert custom_types.DeviceIPCWrapper in registry
        assert registry[custom_types.DeviceIPCWrapper].code == 1
