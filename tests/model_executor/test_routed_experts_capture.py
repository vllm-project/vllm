# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
    RoutedExpertsCapturer,
    RoutedExpertsTensors,
    RoutedExpertsWriteTask,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

pytestmark = pytest.mark.cpu_test

_CAPTURER_MODULE = (
    "vllm.model_executor.layers.fused_moe.routed_experts_capture.capturer"
)


def test_routed_experts_write_task_publishes_copied_tensors():
    routing_data = torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.int32)
    slot_mapping = torch.tensor([5, 9], dtype=torch.int64)
    writer = Mock()
    output = SimpleNamespace(routed_experts_slots=None)
    write_task = RoutedExpertsWriteTask(
        routed_experts_tensors=RoutedExpertsTensors(routing_data, slot_mapping),
        writer=writer,
    )

    write_task.start_copy()
    write_task.finalize(output)

    stored_routing, stored_slots = writer.store_batch.call_args.args
    assert stored_routing.tolist() == routing_data.tolist()
    assert stored_slots.tolist() == slot_mapping.tolist()
    assert output.routed_experts_slots.tolist() == slot_mapping.tolist()


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    moe_top_k: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    capturer = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    capturer.dp_rank = dp_rank
    capturer.tp_size = tp_size
    capturer.device_buffer = torch.full(
        (max_tokens, num_layers, moe_top_k),
        -1,
        dtype=torch.int32,
    )
    return capturer


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(expert_ids: torch.Tensor) -> None:
        captured.append(expert_ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(expert_ids: torch.Tensor) -> None:
        captured.append(expert_ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_gpu_model_runner_binds_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner

    class _DummyRouter:
        _routing_replay_out: torch.Tensor | None = None

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    # Patch the runtime import inside _bind_routed_experts_capturer.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    capturer = DummyCapturer()
    gpu_model_runner.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert dummy_module.router.capture_fn is not None
    dummy_module.router.capture_fn(torch.tensor([[5, 6]]))

    assert len(capturer.calls) == 1
    layer_id, topk_ids = capturer.calls[0]
    assert layer_id == 7
    assert torch.equal(topk_ids, torch.tensor([[5, 6]]))


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    assert dummy_module.router.capture_fn is None

    capturer = DummyCapturer()
    gpu_model_runner.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[9, 10]]))
    assert len(capturer.calls) == 1


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk_ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    forward_context = SimpleNamespace(dp_metadata=None)
    with patch(
        f"{_CAPTURER_MODULE}.get_forward_context",
        return_value=forward_context,
    ):
        capturer.capture(layer_id=0, topk_ids=topk_ids)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk_ids)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """Slice this rank's rows from routing concatenated across DP ranks."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    forward_context = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(
        f"{_CAPTURER_MODULE}.get_forward_context",
        return_value=forward_context,
    ):
        capturer.capture(layer_id=0, topk_ids=topk_ids)
    expected = topk_ids[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], expected)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """Capture routing that is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    forward_context = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk_ids = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(
        f"{_CAPTURER_MODULE}.get_forward_context",
        return_value=forward_context,
    ):
        capturer.capture(layer_id=0, topk_ids=topk_ids)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk_ids)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    forward_context = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk_ids = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(
            f"{_CAPTURER_MODULE}.get_forward_context",
            return_value=forward_context,
        ),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk_ids)
    assert capturer.device_buffer[0, 0, 0].item() == -1


def test_scheduler_accepts_nixl_without_routing_offload_buffer():
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector="NixlConnector")
    )
    scheduler.connector = Mock()

    assert scheduler._validate_routed_experts_offload(Mock()) == (None, 1)


def test_scheduler_skips_offload_transfers_without_offload_buffer():
    from vllm.v1.core.sched.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.routed_experts_manager = SimpleNamespace(
        routed_experts_by_offload_block=None,
        apply_offload_transfers=Mock(),
    )

    scheduler._apply_routed_experts_offload_transfers(Mock())

    scheduler.routed_experts_manager.apply_offload_transfers.assert_not_called()
