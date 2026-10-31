# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import CUDAGraphMode, DeviceConfig, ModelConfig, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import (
    BatchDescriptor,
    DPMetadata,
    ForwardContext,
    get_forward_context,
    override_forward_context,
    set_forward_context,
)
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.experts.cpu_int4_moe import CPUExpertsInt4
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelMonolithicImpl,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPMonolithic,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    RoutedExpertsSink,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"
_RESTORE_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capture_utils"


def _forward_context(**kwargs):
    return ForwardContext(
        no_compile_layers={}, attn_metadata={}, slot_mapping={}, **kwargs
    )


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.TopK

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


def _make_modular_routed_experts():
    return types.SimpleNamespace(
        quant_method=types.SimpleNamespace(is_monolithic=False),
    )


def _full_attention_kv_group(
    spec_type: type[FullAttentionSpec] = FullAttentionSpec,
) -> KVCacheGroupSpec:
    full_attention = spec_type(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    return KVCacheGroupSpec(
        ["layer"],
        UniformTypeKVCacheSpecs(
            block_size=16,
            kv_cache_specs={"layer": full_attention},
        ),
    )


@pytest.mark.parametrize("eplb_enabled", [False, True])
def test_base_router_capture_pre_eplb_mapping(eplb_enabled):
    eplb_state = None
    if eplb_enabled:
        eplb_state = EplbLayerState()
        eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
        eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
        eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
        eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
        eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_public_binding_binds_target_model_router(monkeypatch):
    class DummyFusedMoE:
        def restore_routed_experts_rows(self, ids):
            return ids

        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self._quant_method = _make_modular_routed_experts().quant_method

    target_module = DummyFusedMoE(layer_id=7)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    calls = []
    capturer = types.SimpleNamespace(
        capture=lambda *args: calls.append(args), row_restorers={}
    )

    bind_routed_experts_capturer(
        types.SimpleNamespace(modules=lambda: [target_module]), capturer
    )

    assert target_module.router.capture_fn is not None
    topk_ids = torch.tensor([[5, 6]])
    target_module.router.capture_fn(topk_ids)
    assert calls == [(7, topk_ids)]


def test_public_binding_supports_direct_capture_source():
    class DirectCaptureSource(torch.nn.Module):
        def restore_routed_experts_rows(self, ids):
            return ids

        def __init__(self):
            super().__init__()
            self.layer_id = 3
            self.capture_fn = None

    source = DirectCaptureSource()
    calls = []
    capturer = types.SimpleNamespace(
        capture=lambda *args: calls.append(args), row_restorers={}
    )

    bind_routed_experts_capturer(source, capturer)

    assert source.capture_fn is not None
    topk_ids = torch.tensor([[1, 2]])
    source.capture_fn(topk_ids)
    assert calls == [(3, topk_ids)]


_SINK_CONFIG = SimpleNamespace(
    use_ep=False,
    dp_size=1,
    ep_size=1,
    max_num_tokens=4,
    experts_per_token=2,
    device="cpu",
)


class _ToyMonolithicExperts(CPUExpertsInt4):
    """A monolithic kernel instance, as a weight reload rebuilds it: routes
    token t to experts (t + offset, t + offset + 1)."""

    moe_config = _SINK_CONFIG
    quant_config = None

    def __init__(self, offset: int = 0, supports_capture: bool = True):
        self.offset = offset
        self.supports_capture = supports_capture

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def supports_routing_replay_capture(self) -> bool:
        return self.supports_capture

    def apply(self, hidden_states, *args, routing_replay_out=None, **kwargs):
        if routing_replay_out is not None:
            t = torch.arange(hidden_states.shape[0])
            routing_replay_out[: len(t)] = torch.stack([t, t + 1], 1) + self.offset
        return hidden_states


class _MonolithicMoE:
    """The parts of a MoE layer that capture binding reads."""

    layer_id = 5

    def restore_routed_experts_rows(self, ids):
        return ids

    def __init__(self, experts: _ToyMonolithicExperts):
        self.router = _make_router()
        kernel = SimpleNamespace(impl=SimpleNamespace(fused_experts=experts))
        self._quant_method = SimpleNamespace(is_monolithic=True, moe_kernel=kernel)
        self.routed_experts = SimpleNamespace(
            quant_method=self._quant_method, routing_sink=None
        )


def _bind_monolithic(
    monkeypatch, experts: _ToyMonolithicExperts, capture
) -> _MonolithicMoE:
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", _MonolithicMoE)
    layer = _MonolithicMoE(experts)
    bind_routed_experts_capturer(
        SimpleNamespace(modules=lambda: [layer]),
        SimpleNamespace(capture=capture, row_restorers={}),
    )
    return layer


def test_public_binding_gives_monolithic_layers_a_sink(monkeypatch):
    """A monolithic kernel routes internally, so binding gives the layer, which
    outlives kernel rebuilds, a sink that captures under its layer id."""
    calls = []
    layer = _bind_monolithic(
        monkeypatch, _ToyMonolithicExperts(), lambda *args: calls.append(args)
    )
    ids = torch.tensor([[1, 2]])
    layer.routed_experts.routing_sink.capture_fn(ids)
    assert calls == [(_MonolithicMoE.layer_id, ids)]


def test_public_binding_rejects_monolithic_without_replay_support(monkeypatch):
    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        _bind_monolithic(
            monkeypatch, _ToyMonolithicExperts(supports_capture=False), Mock()
        )


@pytest.mark.parametrize(
    ("use_ep", "dp_size", "ep_size", "rows"),
    [(False, 1, 1, 4), (False, 2, 4, 8), (True, 2, 4, 16)],
    ids=["single-rank", "dp", "ep"],
)
def test_routed_experts_sink_holds_a_dispatched_batch(use_ep, dp_size, ep_size, rows):
    """One int16 buffer, allocated once, large enough for the batch the kernel
    sees after a DP or EP gather."""
    config = SimpleNamespace(
        **{
            **vars(_SINK_CONFIG),
            "use_ep": use_ep,
            "dp_size": dp_size,
            "ep_size": ep_size,
        }
    )
    sink = RoutedExpertsSink(config, Mock())

    assert sink.buffer.shape == (rows, 2) and sink.buffer.dtype == torch.int16


def test_monolithic_capture_survives_kernel_rebuilds():
    """The bug in #59449: a weight reload rebuilds the kernel. Through the
    production kernel path, the layer's sink captures what the current kernel
    routed, before and after a rebuild, and a rebuilt kernel that cannot
    capture is refused instead of leaving stale rows."""
    captured = []
    sink = RoutedExpertsSink(_SINK_CONFIG, lambda ids: captured.append(ids.tolist()))

    def forward(experts: _ToyMonolithicExperts) -> None:
        kernel = FusedMoEKernelMonolithicImpl(
            MoEPrepareAndFinalizeNoDPEPMonolithic(), experts
        )
        kernel.apply(
            torch.zeros(3, 8),
            w1=None,
            w2=None,
            router_logits=torch.zeros(3, 16),
            activation=None,
            global_num_experts=16,
            expert_map=None,
            apply_router_weight_on_input=False,
            routing_sink=sink,
        )

    for offset in (0, 7):  # the kernel before and after a reload
        forward(_ToyMonolithicExperts(offset))
        assert captured[-1] == [[t + offset, t + offset + 1] for t in range(3)]
    with pytest.raises(ValueError, match="not supported"):
        forward(_ToyMonolithicExperts(supports_capture=False))


def test_routed_experts_capturer_single_dp_no_metadata(restore_capturer):
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _bind_standard_source(restore_capturer, dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = _forward_context()
    with override_forward_context(ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


@pytest.mark.parametrize("tp_size, num_tokens", [(2, 6), (2, 3), (2, 1), (4, 2)])
@pytest.mark.parametrize(
    "cg_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE]
)
def test_single_dp_ep_capture_restores_tp_shards(
    tp_size, num_tokens, cg_mode, restore_capturer
):
    """Capture all current rows after model-level SP, including buffer reuse."""
    capturer = _bind_standard_source(restore_capturer, sp_size=tp_size)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            tensor_parallel_size=tp_size,
            enable_expert_parallel=True,
            use_sequence_parallel_moe=False,
            is_moe_model=True,
        ),
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False, static_forward_context={}
        ),
    )
    tp_group = Mock(world_size=tp_size)
    shard_n = (num_tokens + tp_size - 1) // tp_size
    with (
        set_forward_context(
            None,
            config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cg_mode,
        ),
        patch(f"{_RESTORE_MODULE}.get_tp_group", return_value=tp_group),
    ):
        assert get_forward_context().dp_metadata is None
        for offset in (20, 1):
            expected = torch.arange(num_tokens * 2, dtype=torch.int32).reshape(-1, 2)
            expected += offset
            padded = torch.full((shard_n * tp_size, 2), 99, dtype=torch.int32)
            padded[:num_tokens] = expected
            tp_group.all_gather.return_value = padded
            capturer.capture(0, padded[:shard_n])
            assert torch.equal(
                capturer.snapshot_routing_data(num_tokens)[:, 0].int(), expected
            )
            assert (capturer.device_buffer[num_tokens:, 0] == -1).all()
    assert tp_group.all_gather.call_count == 2 * (num_tokens > 1)


@pytest.mark.parametrize("num_tokens, num_rows", [(6, 6), (3, 4), (0, 0)])
def test_single_dp_tp_capture_does_not_regather_full_batch(
    num_tokens, num_rows, restore_capturer
):
    capturer = _bind_standard_source(restore_capturer, sp_size=2, gathered=True)
    topk = torch.arange(num_rows * 2, dtype=torch.int32).reshape(-1, 2)
    ctx = _forward_context(batch_descriptor=BatchDescriptor(num_tokens=num_tokens))
    with (
        override_forward_context(ctx),
        patch(f"{_RESTORE_MODULE}.get_tp_group") as get_tp_group,
    ):
        capturer.capture(0, topk)
    get_tp_group.assert_not_called()
    assert torch.equal(capturer.device_buffer[:num_tokens, 0], topk[:num_tokens])
    assert (capturer.device_buffer[num_tokens:, 0] == -1).all()


@pytest.mark.parametrize("num_tokens, num_rows", [(6, 2), (0, 1)])
def test_single_dp_tp_capture_rejects_unexpected_batch_without_writing(
    num_tokens, num_rows, restore_capturer
):
    capturer = _bind_standard_source(restore_capturer, sp_size=2)
    topk = torch.ones(num_rows, 2, dtype=torch.int32)
    ctx = _forward_context(batch_descriptor=BatchDescriptor(num_tokens=num_tokens))
    with (
        override_forward_context(ctx),
        pytest.raises(ValueError, match="expected .* rows"),
    ):
        capturer.capture(0, topk)
    assert (capturer.device_buffer == -1).all()


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_capture_uses_pre_sp_size_after_model_shards_padding_mask(
    tp_rank, restore_capturer
):
    """Model-level SP shortens the mask; capture still needs all three rows."""
    from vllm.models.common.ops import sequence_parallel as sp

    capturer = _bind_standard_source(restore_capturer, sp_size=2)
    expected = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    ctx = _forward_context(
        is_padding=torch.zeros(3, dtype=torch.bool),
        batch_descriptor=BatchDescriptor(num_tokens=3),
    )
    with (
        patch.object(sp, "get_tensor_model_parallel_world_size", return_value=2),
        patch.object(sp, "get_tensor_model_parallel_rank", return_value=tp_rank),
    ):
        ctx.is_padding = sp.sp_padding_mask(ctx.is_padding, torch.empty(3, 1))
    assert len(ctx.is_padding) == 2
    tp_group = Mock(world_size=2)
    tp_group.all_gather.return_value = expected
    with (
        override_forward_context(ctx),
        patch(f"{_RESTORE_MODULE}.get_tp_group", return_value=tp_group),
    ):
        capturer.capture(0, expected[2 * tp_rank : 2 * (tp_rank + 1)])
    tp_group.all_gather.assert_called_once()
    torch.testing.assert_close(capturer.device_buffer[:3, 0], expected[:3])
    assert (capturer.device_buffer[3:, 0] == -1).all()


@pytest.mark.parametrize(
    ("num_experts", "dtype_name", "torch_dtype"),
    [(256, "uint8", torch.uint8), (257, "uint16", torch.uint16)],
)
def test_routed_experts_capturer_exposes_output_profile(
    monkeypatch, num_experts, dtype_name, torch_dtype
):
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as module

    monkeypatch.setattr(module, "current_platform", SimpleNamespace(device_type="cpu"))
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: 3,
            get_num_experts=lambda: num_experts,
            get_num_experts_per_tok=lambda: 2,
            hf_text_config=SimpleNamespace(model_type="test"),
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0, tensor_parallel_size=1),
    )

    capturer = RoutedExpertsCapturer(8, config)

    assert capturer.shape_per_token == (3, 2)
    assert capturer.output_dtype_name == dtype_name
    assert capturer.output_dtype == torch_dtype
    assert capturer.device_buffer.shape == (8, 3, 2)


@pytest.mark.parametrize("output_dtype", [torch.uint8, torch.uint16])
def test_routed_experts_capturer_narrows_snapshot(output_dtype, restore_capturer):
    capturer = _bind_standard_source(restore_capturer)
    capturer.output_dtype = output_dtype
    topk = torch.tensor([[1, 2], [254, 255]], dtype=torch.int64)
    ctx = _forward_context()
    with override_forward_context(ctx):
        capturer.capture(layer_id=0, topk_ids=topk)

    output = capturer.snapshot_routing_data(2)
    assert capturer.device_buffer.dtype == torch.int32
    assert capturer.device_buffer[:2, 0, :].tolist() == topk.tolist()
    assert output.dtype == output_dtype
    assert output[:, 0, :].tolist() == topk.tolist()
    assert output.data_ptr() != capturer.device_buffer.data_ptr()


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks(restore_capturer):
    """Naive dispatch routes gathered DP rows; capture selects this DP rank."""
    capturer = _bind_standard_source(restore_capturer, dp_rank=1, gathered=True)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = _forward_context(
        dp_metadata=DPMetadata(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with override_forward_context(ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens(restore_capturer):
    """Modular routing precedes dispatch, so its rows are already DP-local."""
    capturer = _bind_standard_source(restore_capturer, dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = _forward_context(
        dp_metadata=DPMetadata(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with override_forward_context(ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_ep_gathered_shards(restore_capturer):
    capturer = _bind_standard_source(
        restore_capturer, dp_rank=1, sp_size=2, gathered=True
    )
    num_tokens_dp = torch.tensor([3, 2], dtype=torch.int32)
    ctx = _forward_context(
        dp_metadata=DPMetadata(
            num_tokens_across_dp_cpu=num_tokens_dp,
            local_sizes=[2, 2, 1, 1],
        )
    )
    topk = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [-1, -1], [10, 11], [12, 13]],
        dtype=torch.int32,
    )

    with override_forward_context(ctx):
        capturer.capture(layer_id=0, topk_ids=topk)

    assert torch.equal(capturer.device_buffer[:2, 0], topk[4:])


@pytest.mark.parametrize("single_dp", [False, True])
def test_routed_experts_capturer_sp_modular_gathers_tp_shards(
    single_dp, restore_capturer
):
    capturer = _bind_standard_source(
        restore_capturer, dp_rank=0 if single_dp else 1, sp_size=2
    )
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = _forward_context(
        dp_metadata=(
            None if single_dp else DPMetadata(num_tokens_across_dp_cpu=num_tokens_dp)
        ),
        batch_descriptor=BatchDescriptor(num_tokens=3),
    )
    local_shard = torch.tensor([[10, 12], [11, 13]], dtype=torch.int32).t()
    gathered = torch.tensor([[10, 11], [12, 13], [14, 15], [-1, -1]], dtype=torch.int32)
    tp_group = Mock(world_size=2)
    tp_group.all_gather.return_value = gathered

    with (
        override_forward_context(ctx),
        patch(f"{_RESTORE_MODULE}.get_tp_group", return_value=tp_group),
    ):
        capturer.capture(layer_id=0, topk_ids=local_shard)

    tp_group.all_gather.assert_called_once()
    gather_input = tp_group.all_gather.call_args.args[0]
    assert gather_input.is_contiguous()
    torch.testing.assert_close(gather_input, local_shard)
    assert tp_group.all_gather.call_args.kwargs == {"dim": 0}
    assert torch.equal(capturer.device_buffer[:3, 0], gathered[:3])


def test_routed_experts_capturer_rejects_incomplete_gathered_rows(restore_capturer):
    """An incomplete gathered tensor must not silently truncate local rows."""
    capturer = _bind_standard_source(
        restore_capturer, dp_rank=1, sp_size=2, gathered=True
    )
    ctx = _forward_context(
        dp_metadata=DPMetadata(
            num_tokens_across_dp_cpu=torch.tensor([3, 2]),
        )
    )
    with (
        override_forward_context(ctx),
        pytest.raises(ValueError, match="expected 6 rows"),
    ):
        capturer.capture(0, torch.ones(5, 2, dtype=torch.int32))
    assert (capturer.device_buffer == -1).all()


@pytest.mark.parametrize("layer_id", [-1, 4])
def test_routed_experts_capturer_rejects_layer_before_restoring_rows(
    layer_id, restore_capturer
):
    capturer = _bind_standard_source(restore_capturer, sp_size=2)
    ctx = _forward_context(batch_descriptor=BatchDescriptor(num_tokens=3))
    with (
        override_forward_context(ctx),
        patch(f"{_RESTORE_MODULE}.get_tp_group") as get_tp_group,
        pytest.raises(IndexError, match="exceeds capture buffer"),
    ):
        capturer.capture(layer_id, torch.ones(2, 2, dtype=torch.int32))
    get_tp_group.assert_not_called()
    assert (capturer.device_buffer == -1).all()


@pytest.mark.parametrize(
    ("counts", "local_sizes", "start", "stop"),
    [
        ([3, 0], [2, 2, 0, 0], 4, 4),
        ([0, 3], None, 0, 3),
        ([0, 0], [0, 0, 0, 0], 0, 0),
        ([2, 2], [1, 1, 1, 1], 2, 4),
    ],
    ids=["idle-rank", "only-active-rank", "all-idle", "unpadded-gather-tie"],
)
def test_routed_experts_capturer_dp_preserves_empty_and_unpadded_spans(
    counts, local_sizes, start, stop, restore_capturer
):
    capturer = _bind_standard_source(
        restore_capturer, dp_rank=1, sp_size=2 if local_sizes else 1, gathered=True
    )
    physical_rows = sum(local_sizes) if local_sizes else sum(counts)
    topk = torch.arange(physical_rows * 2, dtype=torch.int32).reshape(-1, 2)
    ctx = _forward_context(
        dp_metadata=DPMetadata(
            num_tokens_across_dp_cpu=torch.tensor(counts), local_sizes=local_sizes
        )
    )
    with (
        override_forward_context(ctx),
        patch(f"{_RESTORE_MODULE}.get_tp_group") as get_tp_group,
    ):
        capturer.capture(0, topk)
    get_tp_group.assert_not_called()
    torch.testing.assert_close(
        capturer.device_buffer[: stop - start, 0], topk[start:stop]
    )
    assert (capturer.device_buffer[stop - start :, 0] == -1).all()


def test_routed_experts_capturer_dp_unexpected_batch_raises(restore_capturer):
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _bind_standard_source(restore_capturer, dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = _forward_context(
        dp_metadata=DPMetadata(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Modular routing promises two local rows; one row is incomplete.
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        override_forward_context(ctx),
        pytest.raises(ValueError, match="expected .* rows"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1


def test_get_aux_output_connector_passes_config(monkeypatch):
    import vllm.distributed.aux_output_connector.worker as aux_output_worker

    connector = Mock()
    constructor = Mock(return_value=connector)
    monkeypatch.setattr(aux_output_worker, "AuxOutputWorkerConnector", constructor)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32)
    )
    model = Mock()

    result = aux_output_worker.get_aux_output_connector(model, config)

    constructor.assert_called_once_with(model=model, vllm_config=config)
    assert result is connector


def test_aux_output_worker_connector_binds_capture_on_non_output_rank(monkeypatch):
    import vllm.distributed.aux_output_connector.worker as aux_output_worker

    capturer = Mock()
    constructor = Mock(return_value=capturer)
    bind = Mock()
    monkeypatch.setattr(aux_output_worker, "RoutedExpertsCapturer", constructor)
    monkeypatch.setattr(aux_output_worker, "bind_routed_experts_capturer", bind)
    monkeypatch.setattr(
        aux_output_worker,
        "get_tp_group",
        lambda: SimpleNamespace(is_first_rank=False, world_size=1),
    )

    config = SimpleNamespace(
        aux_output_config=SimpleNamespace(enable_return_routed_experts=True),
        kv_transfer_config=None,
        max_concurrent_batches=2,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
    )
    model = Mock()
    connector = aux_output_worker.AuxOutputWorkerConnector(
        vllm_config=config, model=model
    )
    connector.initialize_output_storage(
        SimpleNamespace(kv_cache_groups=[_full_attention_kv_group()])
    )

    assert connector._store is None and connector._buffer is None
    constructor.assert_called_once_with(
        max_num_batched_tokens=32,
        vllm_config=config,
    )
    bind.assert_called_once_with(model, capturer)
    connector.begin_step(Mock())
    assert connector.prepare_output(Mock()) is None
    capturer.snapshot_routing_data.assert_not_called()


def test_aux_output_worker_connector_default_capacity(monkeypatch):
    import vllm.distributed.aux_output_connector.worker as aux_output_worker

    tp_group = SimpleNamespace(is_first_rank=True, world_size=1)
    store_constructor = Mock()
    background_store_constructor = Mock(side_effect=lambda store, **_: store)
    capturer = SimpleNamespace(shape_per_token=(2,), output_dtype_name="int32")
    monkeypatch.setattr(aux_output_worker, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        aux_output_worker, "RoutedExpertsCapturer", Mock(return_value=capturer)
    )
    monkeypatch.setattr(aux_output_worker, "bind_routed_experts_capturer", Mock())
    monkeypatch.setattr(
        aux_output_worker,
        "resolve_kv_cache_block_sizes",
        lambda *_: (32, 16),
    )
    monkeypatch.setattr(aux_output_worker, "BlockObjectStore", store_constructor)
    monkeypatch.setattr(
        aux_output_worker, "BackgroundBlockObjectStore", background_store_constructor
    )
    buffer_constructor = Mock()
    monkeypatch.setattr(aux_output_worker, "RoutedExpertsBuffer", buffer_constructor)

    config = SimpleNamespace(
        aux_output_config=SimpleNamespace(max_bytes=None),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=32),
        max_concurrent_batches=2,
    )
    connector = aux_output_worker.AuxOutputWorkerConnector(
        vllm_config=config, model=Mock()
    )
    store_constructor.assert_not_called()
    buffer_constructor.assert_not_called()

    connector.initialize_output_storage(
        SimpleNamespace(
            num_blocks=10,
            kv_cache_groups=[_full_attention_kv_group(MLAAttentionSpec)],
        )
    )
    assert store_constructor.call_args.kwargs["max_bytes"] == 2560
    assert store_constructor.call_args.kwargs["object_nbytes"] == 128
    assert background_store_constructor.call_args.kwargs["max_pending_batches"] == 16


def test_v2_model_runner_accepts_routed_experts(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: ())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            use_mla=False,
            logits_processors=None,
            enable_prompt_embeds=False,
        ),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=True),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            distributed_executor_backend=None,
            pipeline_parallel_size=1,
            enable_dbo=False,
            use_ubatching=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        cache_config=SimpleNamespace(
            kv_sharing_fast_prefill=False,
            mamba_cache_mode="none",
        ),
        ec_transfer_config=None,
    )

    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)

    assert "routed experts capture" not in unsupported


@pytest.fixture
def capture_config(tmp_path):
    from transformers import MixtralConfig

    MixtralConfig(
        architectures=["MixtralForCausalLM"],
        num_hidden_layers=4,
        num_local_experts=16,
        num_experts_per_tok=2,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="float32",
            max_model_len=32,
            enforce_eager=True,
            skip_tokenizer_init=True,
        ),
        device_config=DeviceConfig(device="cpu"),
    )
    return config


@pytest.fixture
def restore_capturer(capture_config, monkeypatch):
    from vllm.platforms.cpu import CpuPlatform

    monkeypatch.setattr(f"{_REC_MODULE}.current_platform", CpuPlatform())
    capturer = RoutedExpertsCapturer(8, capture_config)
    capturer.device_buffer.fill_(-1)
    return capturer


class _CaptureSource(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.capture_fn = None

    def restore_routed_experts_rows(self, ids):
        return ids

    def forward(self, ids):
        if self.capture_fn is not None:
            self.capture_fn(ids)
        return ids


class _RestoringCaptureSource(_CaptureSource):
    def restore_routed_experts_rows(self, ids):
        ctx = get_forward_context()
        assert ctx.batch_descriptor is not None
        ctx.additional_kwargs["restored_layers"].append(self.layer_id)
        start = ctx.additional_kwargs["start"]
        if start is None:
            return ids
        start += self.layer_id
        return ids[start : start + ctx.batch_descriptor.num_tokens]


@pytest.mark.parametrize("rows", [0, 3])
def test_capture_restorers_are_layer_local_and_use_current_context(
    rows, restore_capturer
):
    capturer = restore_capturer
    model = torch.nn.ModuleList(
        [_RestoringCaptureSource(0), _RestoringCaptureSource(1), _CaptureSource(2)]
    )
    bind_routed_experts_capturer(model, capturer)
    for step, start in enumerate((1, None, 2)):
        ids = torch.arange(14, dtype=torch.int32).reshape(7, 2) + step
        input_ids = ids if start is not None else ids[:rows]
        original = input_ids.clone()
        ctx = _forward_context(
            batch_descriptor=BatchDescriptor(num_tokens=rows),
            additional_kwargs={"start": start, "restored_layers": []},
        )
        with override_forward_context(ctx):
            for source in model[:2]:
                torch.testing.assert_close(source(input_ids), original)
            model[2](ids[:rows])
        assert ctx.additional_kwargs["restored_layers"] == [0, 1]
        expected = torch.full_like(capturer.device_buffer, -1)
        for layer_id in (0, 1):
            first = start + layer_id if start is not None else 0
            expected[:rows, layer_id] = ids[first : first + rows]
        expected[:rows, 2] = ids[:rows]
        torch.testing.assert_close(capturer.device_buffer, expected)


def test_capture_rejects_duplicate_restorers(restore_capturer):
    model = torch.nn.ModuleList(
        [_RestoringCaptureSource(0), _RestoringCaptureSource(0)]
    )
    with pytest.raises(ValueError, match="Duplicate.*restorer"):
        bind_routed_experts_capturer(model, restore_capturer)


class _StandardCaptureSource(_CaptureSource):
    def __init__(self, *, dp_rank=0, sp_size=1, gathered=False):
        super().__init__(0)
        self.dp_rank = dp_rank
        self.sp_size = sp_size
        self.gathered = gathered

    def restore_routed_experts_rows(self, ids):
        from vllm.model_executor.layers.fused_moe.routed_experts_capture_utils import (
            collect_local_routing_rows,
        )

        return collect_local_routing_rows(
            ids, dp_rank=self.dp_rank, sp_size=self.sp_size, is_gathered=self.gathered
        )


def _bind_standard_source(capturer, **kwargs):
    bind_routed_experts_capturer(_StandardCaptureSource(**kwargs), capturer)
    return capturer


@pytest.mark.parametrize("sp_size, rank", [(1, 0), (2, 0), (2, 1)])
def test_native_mega_moe_restores_input_sp_rows(
    sp_size, rank, restore_capturer, capture_config
):
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as mega
    from vllm.platforms.cpu import CpuPlatform

    with (
        patch.object(mega, "current_platform", CpuPlatform()),
        patch.object(mega, "check_deep_gemm_mega_moe_supported"),
    ):
        experts = mega.DeepGemmMegaMoEExperts(
            capture_config,
            num_experts=8,
            num_local_experts=4,
            experts_start_idx=rank * 4,
            top_k=2,
            hidden_size=128,
            intermediate_size=128,
            activation=mega.MegaMoEActivation(),
            weight_format=mega.MEGA_MOE_MXFP4,
            prefix="model.layers.1.experts",
            input_sp_size=sp_size,
        )
    bind_routed_experts_capturer(experts, restore_capturer)
    ids = torch.arange(6, dtype=torch.int32).reshape(3, 2)
    padded = torch.cat((ids, torch.full((1, 2), 99, dtype=torch.int32)))
    group = Mock(world_size=sp_size)
    group.all_gather.return_value = padded
    ctx = _forward_context(batch_descriptor=BatchDescriptor(num_tokens=3))
    with (
        override_forward_context(ctx),
        patch(f"{_RESTORE_MODULE}.get_tp_group", return_value=group),
    ):
        assert experts.capture_fn is not None
        experts.capture_fn(padded[rank * 2 : (rank + 1) * 2] if sp_size > 1 else ids)
    assert group.all_gather.call_count == int(sp_size > 1)
    expected = torch.full_like(restore_capturer.device_buffer, -1)
    expected[:3, 1] = ids
    torch.testing.assert_close(restore_capturer.device_buffer, expected)


def test_capture_requires_source_to_define_restoration(restore_capturer):
    class Source(torch.nn.Module):
        layer_id = 0
        capture_fn = None

    with pytest.raises(ValueError, match="must implement restore_routed_experts_rows"):
        bind_routed_experts_capturer(Source(), restore_capturer)


@pytest.mark.parametrize(
    "path", ["naive", "modular", "monolithic-local", "monolithic-gathered", "custom"]
)
@pytest.mark.parametrize("sp_size", [1, 2])
def test_moe_runner_restores_rows_for_its_execution_path(
    path, sp_size, capture_config, restore_capturer, monkeypatch
):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
        FusedMoEConfig,
        FusedMoEParallelConfig,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        CPUUnquantizedExperts,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPMonolithic,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.platforms.cpu import CpuPlatform

    for module in (
        "oracle.unquantized",
        "experts.cpu_moe",
        "modular_kernel",
        "runner.moe_runner",
        "unquantized_fused_moe_method",
    ):
        monkeypatch.setattr(
            f"vllm.model_executor.layers.fused_moe.{module}.current_platform",
            CpuPlatform(),
        )

    class BackendMoERunner(MoERunner):
        def restore_routed_experts_rows(self, ids):
            from vllm.distributed.parallel_state import get_tp_group

            ctx = get_forward_context()
            assert ctx.dp_metadata is not None
            rows = int(
                ctx.dp_metadata.num_tokens_across_dp_cpu[self.moe_config.dp_rank]
            )
            # A backend may split padded rows unevenly even when SP is off.
            return get_tp_group().all_gatherv(ids, dim=0, sizes=[2, 1])[:rows]

    parallel = FusedMoEParallelConfig(
        tp_size=2,
        pcp_size=1,
        dp_size=2,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=1,
        ep_rank=0,
        sp_size=sp_size,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=16,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=256,
        num_local_experts=16,
        num_logical_experts=16,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=parallel,
        in_dtype=torch.float32,
    )
    manager = ExpertMapManager(
        max_num_batched_tokens=8,
        top_k=2,
        global_num_experts=16,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy="linear",
        enable_eplb=False,
    )
    with set_current_vllm_config(capture_config):
        experts = RoutedExperts(
            "model.layers.0.experts", torch.float32, config, None, manager
        )
        if path == "modular":
            experts.quant_method.moe_kernel = FusedMoEKernel(
                MoEPrepareAndFinalizeNoDPEPModular(),
                CPUUnquantizedExperts(config, FUSED_MOE_UNQUANTIZED_CONFIG),
            )
        elif path.startswith("monolithic"):
            prepare = (
                MoEPrepareAndFinalizeNaiveDPEPMonolithic(sp_size > 1)
                if path == "monolithic-gathered"
                else MoEPrepareAndFinalizeNoDPEPMonolithic()
            )
            experts.quant_method.moe_kernel = FusedMoEKernel(
                prepare, CPUExpertsInt4(config, FUSED_MOE_UNQUANTIZED_CONFIG)
            )
        runner_cls = BackendMoERunner if path == "custom" else MoERunner
        runner = runner_cls("model.layers.0.experts", config, _make_router(), experts)

    # DP0 owns 3 rows, DP1 owns 2. Gathered SP inputs include DP0's padding.
    ids = torch.tensor([[10, 11], [12, 13]], dtype=torch.int32)
    gathered = path in ("naive", "monolithic-gathered")
    if gathered:
        prefix_rows = 4 if sp_size == 2 else 3
        physical_ids = torch.cat(
            (torch.full((prefix_rows, 2), 99, dtype=torch.int32), ids)
        )
    else:
        physical_ids = ids[:1] if sp_size == 2 and path != "custom" else ids
    group = Mock(world_size=2)
    group.all_gather.return_value = ids
    group.all_gatherv.return_value = torch.cat(
        (ids, torch.full((1, 2), 99, dtype=torch.int32))
    )
    ctx = _forward_context(dp_metadata=DPMetadata(torch.tensor([3, 2])))
    with (
        override_forward_context(ctx),
        patch(f"{_RESTORE_MODULE}.get_tp_group", return_value=group),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=group),
    ):
        actual = runner.restore_routed_experts_rows(physical_ids)
        if not path.startswith("monolithic"):
            bind_routed_experts_capturer(runner, restore_capturer)
            assert isinstance(runner.router, BaseRouter)
            assert runner.router.capture_fn is not None
            runner.router.capture_fn(physical_ids)
            expected = torch.full_like(restore_capturer.device_buffer, -1)
            expected[:2, 0] = ids
            torch.testing.assert_close(restore_capturer.device_buffer, expected)
    torch.testing.assert_close(actual, ids)
    calls = 1 if path.startswith("monolithic") else 2
    assert group.all_gather.call_count == calls * int(
        sp_size > 1 and not gathered and path != "custom"
    )
    assert group.all_gatherv.call_count == calls * int(path == "custom")


@pytest.mark.parametrize("num_tokens", [0, 1])
@pytest.mark.parametrize("dp_rank", [None, 0, 1])
def test_publishing_rank_captures_complete_sp_shard_without_collective(
    num_tokens, dp_rank, restore_capturer
):
    """TP0 already owns every real row of an empty or singleton SP batch."""
    capturer = _bind_standard_source(restore_capturer, dp_rank=dp_rank or 0, sp_size=2)
    counts = [num_tokens, num_tokens]
    ctx = _forward_context(
        dp_metadata=DPMetadata(torch.tensor(counts)) if dp_rank is not None else None,
        batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
    )
    ids = torch.tensor([[5, 6]], dtype=torch.int32)[:num_tokens]
    # There is no initialized TP group: this valid fast path needs no collective.
    with override_forward_context(ctx):
        capturer.capture(0, ids)
    torch.testing.assert_close(capturer.device_buffer[:num_tokens, 0], ids)
    assert (capturer.device_buffer[num_tokens:] == -1).all()
