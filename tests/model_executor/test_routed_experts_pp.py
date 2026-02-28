# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for RoutedExpertsCapturer pipeline-parallelism support.

These tests verify the fix for the bug where non-last PP ranks lost
their captured routing decisions because:
  1. execute_model() returned early before save_captured_experts()
  2. The last rank's bulk save overwrote other ranks' data with zeros
"""

import os
import tempfile
import types

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
)

pytestmark = pytest.mark.cpu_test

NUM_LAYERS = 8
NUM_TOKENS = 4
TOPK = 2


def _make_capturer_with_host_buffer(
    host_buffer: np.ndarray,
    lock_file: str,
) -> RoutedExpertsCapturer:
    """Create a capturer with pre-wired host buffer (bypassing shared memory)."""
    c = RoutedExpertsCapturer()
    c._device_buffer = torch.zeros((NUM_TOKENS, NUM_LAYERS, TOPK), dtype=torch.int32)
    c._host_buffer_view = host_buffer
    c._lock_file = lock_file
    return c


def _fill_device_buffer(capturer: RoutedExpertsCapturer, layers: set[int]):
    """Write distinct non-zero routing data into the given layers of the
    device buffer while leaving all other layers as zeros (mimicking what
    a PP rank's forward pass does)."""
    for layer_id in layers:
        capturer._device_buffer[:, layer_id, :] = layer_id + 1


@pytest.fixture
def lock_file():
    fd, path = tempfile.mkstemp(prefix="vllm_test_lock_")
    os.close(fd)
    yield path
    os.unlink(path)


# ── save_captured_experts with _owned_layers ────────────────────────


def test_owned_layers_writes_only_owned(monkeypatch, lock_file):
    """When _owned_layers is set, only those layers should be written."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe"
        ".routed_experts_capturer.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    host = np.zeros((NUM_TOKENS, NUM_LAYERS, TOPK), dtype=np.int32)
    capturer = _make_capturer_with_host_buffer(host, lock_file)
    capturer._owned_layers = {0, 1, 2, 3}

    _fill_device_buffer(capturer, capturer._owned_layers)
    indices = np.arange(NUM_TOKENS)
    capturer.save_captured_experts(indices)

    for layer_id in range(NUM_LAYERS):
        if layer_id in capturer._owned_layers:
            assert host[0, layer_id, 0] == layer_id + 1, (
                f"layer {layer_id} should have been written"
            )
        else:
            assert host[0, layer_id, 0] == 0, (
                f"layer {layer_id} should NOT have been written"
            )


def test_no_owned_layers_bulk_writes_all(monkeypatch, lock_file):
    """When _owned_layers is None (PP=1), all layers are bulk-written."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe"
        ".routed_experts_capturer.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    host = np.zeros((NUM_TOKENS, NUM_LAYERS, TOPK), dtype=np.int32)
    capturer = _make_capturer_with_host_buffer(host, lock_file)
    assert capturer._owned_layers is None

    _fill_device_buffer(capturer, set(range(NUM_LAYERS)))
    indices = np.arange(NUM_TOKENS)
    capturer.save_captured_experts(indices)

    for layer_id in range(NUM_LAYERS):
        assert host[0, layer_id, 0] == layer_id + 1


# ── PP simulation: two ranks sharing the same host buffer ───────────


def test_pp_two_ranks_preserve_each_others_layers(monkeypatch, lock_file):
    """Simulate PP=2: two capturers share the same host buffer.
    Each writes only its owned layers.  The combined result should have
    correct data for ALL layers.

    Before the fix this would fail: the second rank's bulk save would
    overwrite the first rank's layers with zeros.
    """
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe"
        ".routed_experts_capturer.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    shared_host = np.zeros((NUM_TOKENS, NUM_LAYERS, TOPK), dtype=np.int32)
    indices = np.arange(NUM_TOKENS)

    rank0_layers = {0, 1, 2, 3}
    rank1_layers = {4, 5, 6, 7}

    # PP rank 0: captures first half of layers, saves
    rank0 = _make_capturer_with_host_buffer(shared_host, lock_file)
    rank0._owned_layers = rank0_layers
    _fill_device_buffer(rank0, rank0_layers)
    rank0.save_captured_experts(indices)

    # PP rank 1: captures second half of layers, saves
    rank1 = _make_capturer_with_host_buffer(shared_host, lock_file)
    rank1._owned_layers = rank1_layers
    _fill_device_buffer(rank1, rank1_layers)
    rank1.save_captured_experts(indices)

    # All layers should have correct (non-zero) data
    for layer_id in range(NUM_LAYERS):
        expected = layer_id + 1
        actual = shared_host[0, layer_id, 0]
        assert actual == expected, (
            f"layer {layer_id}: expected {expected}, got {actual}"
        )


def test_pp_bulk_save_overwrites_other_ranks(monkeypatch, lock_file):
    """Demonstrate the pre-fix bug: without _owned_layers the second
    rank's bulk save clobbers the first rank's data with zeros.

    This test PASSES (the clobbering is the expected *broken* behaviour
    it is documenting).
    """
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe"
        ".routed_experts_capturer.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    shared_host = np.zeros((NUM_TOKENS, NUM_LAYERS, TOPK), dtype=np.int32)
    indices = np.arange(NUM_TOKENS)

    rank0_layers = {0, 1, 2, 3}
    rank1_layers = {4, 5, 6, 7}

    # Rank 0 saves (bulk — _owned_layers is None)
    rank0 = _make_capturer_with_host_buffer(shared_host, lock_file)
    _fill_device_buffer(rank0, rank0_layers)
    rank0.save_captured_experts(indices)

    # Rank 1 saves (bulk) — overwrites rank 0's layers with zeros
    rank1 = _make_capturer_with_host_buffer(shared_host, lock_file)
    _fill_device_buffer(rank1, rank1_layers)
    rank1.save_captured_experts(indices)

    # Rank 0's layers are now zeros (the bug)
    for layer_id in rank0_layers:
        assert shared_host[0, layer_id, 0] == 0, (
            f"layer {layer_id} should have been clobbered to 0"
        )
    # Rank 1's layers survive because it wrote last
    for layer_id in rank1_layers:
        assert shared_host[0, layer_id, 0] == layer_id + 1


# ── _bind_routed_experts_capturer sets _owned_layers ────────────────


def test_bind_sets_owned_layers(monkeypatch):
    """_bind_routed_experts_capturer must populate capturer._owned_layers
    with the set of MoE layer IDs present on this PP rank."""
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        BaseRouter,
    )
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyRouter(BaseRouter):
        @property
        def routing_method_type(self) -> RoutingMethodType:
            return RoutingMethodType.FUSED_TOPK

        def _compute_routing(self, hidden_states, router_logits, indices_type):
            return torch.empty(0), torch.empty(0)

    def _make_router():
        from vllm.distributed.eplb.eplb_state import EplbLayerState

        return DummyRouter(
            top_k=2,
            global_num_experts=16,
            eplb_state=EplbLayerState(),
            enable_eplb=False,
            indices_type_getter=None,
        )

    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    modules = {
        "layer_3": DummyFusedMoE(3),
        "layer_7": DummyFusedMoE(7),
        "layer_11": DummyFusedMoE(11),
    }
    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context=modules,
        )
    )

    capturer = RoutedExpertsCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert capturer._owned_layers == {3, 7, 11}


def test_bind_no_moe_layers_leaves_owned_layers_none(monkeypatch):
    """When no FusedMoE layers exist, _owned_layers stays None."""
    from vllm.v1.worker import gpu_model_runner as gmr

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={},
        )
    )

    capturer = RoutedExpertsCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert capturer._owned_layers is None
