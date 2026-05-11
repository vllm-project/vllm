# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.cpu_test


def test_bind_routing_capture_to_model_sets_layer_view(monkeypatch):
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec_mod

    class _DummyMoEConfig:
        is_sequence_parallel = False
        dp_size = 1

    class _DummyQuantMethod:
        supports_internal_mk = True

    class DummyFusedMoE:
        _routing_replay_out: torch.Tensor

        def __init__(self, moe_layer_id):
            self.moe_layer_id = moe_layer_id
            self.moe_config = _DummyMoEConfig()
            self.quant_method = _DummyQuantMethod()

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    num_layers, num_tokens, top_k = 4, 8, 2
    buffer = torch.zeros((num_layers, num_tokens, top_k), dtype=torch.int16)

    class DummyDeviceCache:
        def __init__(self, buf):
            self.buffer = buf

    class DummyCapturer:
        def get_device_cache(self):
            return DummyDeviceCache(buffer)

    monkeypatch.setattr(rec_mod, "get_global_experts_capturer", lambda: DummyCapturer())

    m0 = DummyFusedMoE(moe_layer_id=0)
    m2 = DummyFusedMoE(moe_layer_id=2)

    class DummyModel:
        def modules(self):
            return iter([m0, m2])

    rec_mod.bind_routing_capture_to_model(DummyModel())

    assert torch.equal(m0._routing_replay_out, buffer[0])
    assert torch.equal(m2._routing_replay_out, buffer[2])


def test_bind_routing_capture_to_model_noop_when_disabled(monkeypatch):
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec_mod

    class DummyCapturer:
        def get_device_cache(self):
            return None

    monkeypatch.setattr(rec_mod, "get_global_experts_capturer", lambda: DummyCapturer())

    class DummyModel:
        def modules(self):
            return iter([])

    rec_mod.bind_routing_capture_to_model(DummyModel())


# =========================================================================
# Tests for device-cache routing replay architecture
# =========================================================================


class TestRoutedExpertsDeviceCache:
    """Tests for _RoutedExpertsDeviceCache (GPU buffer for routing data)."""

    def test_allocation_shape_and_dtype(self):
        """Device cache allocates (L, N, K) int16 buffer."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDeviceCache,
        )

        cache = _RoutedExpertsDeviceCache(
            num_hidden_layers=40,
            max_num_batched_tokens=8192,
            num_experts_per_tok=8,
            device="cpu",
        )
        assert cache.buffer.shape == (40, 8192, 8)
        assert cache.buffer.dtype == torch.int16

    def test_per_layer_view_is_contiguous(self):
        """buffer[layer_id] gives contiguous (N, K) view for FlashInfer."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsDeviceCache,
        )

        cache = _RoutedExpertsDeviceCache(
            num_hidden_layers=40,
            max_num_batched_tokens=8192,
            num_experts_per_tok=8,
            device="cpu",
        )
        layer_view = cache.buffer[0]
        assert layer_view.is_contiguous()
        assert layer_view.shape == (8192, 8)


class TestRoutedExpertsHostCache:
    """Tests for _RoutedExpertsHostCache (per-request numpy buffer)."""

    def test_sentinel_initialization(self):
        """Host cache initializes with zeros by default."""
        import numpy as np

        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsHostCache,
        )

        cache = _RoutedExpertsHostCache(
            num_hidden_layers=40,
            num_experts_per_tok=8,
            max_model_len=1024,
        )
        buf = cache.get_or_grow_buffer("req1", max_pos=100)
        assert buf.dtype == np.int16
        assert (buf == 0).all(), "Host cache must initialize with zeros"

    def test_grow_preserves_existing_data(self):
        """Growing the buffer preserves previously written data."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsHostCache,
        )

        cache = _RoutedExpertsHostCache(
            num_hidden_layers=40,
            num_experts_per_tok=8,
            max_model_len=1024,
        )
        buf = cache.get_or_grow_buffer("req1", max_pos=50)
        buf[0, 0, 0] = 42
        buf2 = cache.get_or_grow_buffer("req1", max_pos=200)
        assert buf2[0, 0, 0] == 42, "Data lost during buffer grow"

    def test_free_request_removes_buffer(self):
        """Freeing a request removes its buffer."""
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            _RoutedExpertsHostCache,
        )

        cache = _RoutedExpertsHostCache(
            num_hidden_layers=40,
            num_experts_per_tok=8,
            max_model_len=1024,
        )
        cache.get_or_grow_buffer("req1", max_pos=50)
        cache.free_request("req1")
        assert cache.get_buffer("req1") is None
