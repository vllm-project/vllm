# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.experts.cpu_int4_moe import CPUExpertsInt4
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelMonolithicImpl,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPMonolithic,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capture_layout import (
    make_routed_experts_capture_layout,
    prepare_routed_experts_capture_layouts,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCaptureLayout,
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
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self._quant_method = _make_modular_routed_experts().quant_method

    target_module = DummyFusedMoE(layer_id=7)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    calls = []
    capturer = types.SimpleNamespace(capture=lambda *args: calls.append(args))

    bind_routed_experts_capturer(
        types.SimpleNamespace(modules=lambda: [target_module]), capturer
    )

    assert target_module.router.capture_fn is not None
    topk_ids = torch.tensor([[5, 6]])
    target_module.router.capture_fn(topk_ids)
    assert calls == [(7, topk_ids)]


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
        SimpleNamespace(capture=capture),
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


def _capture(capturer, layer_id, ids, layout):
    with override_forward_context(
        _forward_context(routed_experts_capture_layouts={layer_id: layout})
    ):
        capturer.capture(layer_id, ids)


@pytest.mark.parametrize("rows", [0, 3])
@pytest.mark.parametrize("single_rank_group", [False, True])
def test_local_capture_needs_no_communicator(rows, single_rank_group, capturer):
    ids = torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2)
    if single_rank_group:
        group = _capture_group(rank=0, world_size=1)
        assert group.device_communicator is None
        layout = RoutedExpertsCaptureLayout(((0, rows),), rows, group, (rows,))
    else:
        layout = make_routed_experts_capture_layout((rows,), 0)
    _capture(capturer, 0, ids, layout)
    expected = torch.full_like(capturer.device_buffer, -1)
    expected[:rows, 0] = ids
    torch.testing.assert_close(capturer.device_buffer, expected)


@pytest.mark.parametrize(
    "tp_size, num_tokens, dp_rank",
    [(2, 6, 0), (2, 3, 0), (2, 1, 0), (4, 2, 0), (2, 3, 1)],
)
def test_capture_restores_sp_shards_across_forwards(
    tp_size, num_tokens, dp_rank, capturer
):
    """Restore padded SP shards for single/multiple DP ranks, including N < TP."""
    group = _capture_group(rank=0, world_size=tp_size)
    with patch("vllm.distributed.parallel_state.get_tp_group", return_value=group):
        layout = make_routed_experts_capture_layout(
            (num_tokens,) if dp_rank == 0 else (2, num_tokens), dp_rank, sp_size=tp_size
        )
    ctx = _forward_context(
        routed_experts_capture_layouts={0: layout},
    )
    for offset in (20, 1):
        expected = torch.arange(num_tokens * 2, dtype=torch.int32).reshape(-1, 2)
        expected += offset
        padded = torch.full((sum(layout.shard_sizes), 2), 99, dtype=torch.int32)
        padded[:num_tokens] = expected
        with (
            override_forward_context(ctx),
            patch.object(group, "all_gatherv", return_value=padded) as collective,
        ):
            capturer.capture(0, padded[: layout.input_rows])
        assert collective.call_args.kwargs == {
            "dim": 0,
            "sizes": list(layout.shard_sizes),
        }
        torch.testing.assert_close(
            capturer.snapshot_routing_data(num_tokens)[:, 0].int(), expected
        )
        buffer = torch.full_like(capturer.device_buffer, -1)
        buffer[:num_tokens, 0] = expected
        torch.testing.assert_close(capturer.device_buffer, buffer)


@pytest.mark.parametrize("num_tokens, num_rows", [(6, 2), (0, 1)])
def test_capture_rejects_unexpected_batch_without_writing(
    num_tokens, num_rows, capturer
):
    layout = make_routed_experts_capture_layout((num_tokens,), 0)
    with pytest.raises(ValueError, match="input rows"):
        _capture(capturer, 0, torch.ones(num_rows, 2, dtype=torch.int32), layout)
    assert (capturer.device_buffer == -1).all()


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_capture_uses_pre_sp_size_after_model_shards_padding_mask(tp_rank, capturer):
    from vllm.models.common.ops import sequence_parallel as sp

    group = _capture_group(tp_rank)
    with patch("vllm.distributed.parallel_state.get_tp_group", return_value=group):
        layout = make_routed_experts_capture_layout((3,), 0, sp_size=2)
    ctx = _forward_context(
        is_padding=torch.zeros(3, dtype=torch.bool),
        routed_experts_capture_layouts={0: layout},
    )
    with (
        patch.object(sp, "get_tensor_model_parallel_world_size", return_value=2),
        patch.object(sp, "get_tensor_model_parallel_rank", return_value=tp_rank),
    ):
        ctx.is_padding = sp.sp_padding_mask(ctx.is_padding, torch.empty(3, 1))
    assert len(ctx.is_padding) == 2
    ids = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    with (
        override_forward_context(ctx),
        patch.object(group, "all_gatherv", return_value=ids),
    ):
        capturer.capture(0, ids[2 * tp_rank : 2 * (tp_rank + 1)])
    torch.testing.assert_close(capturer.device_buffer[:3, 0], ids[:3])
    assert (capturer.device_buffer[3:, 0] == -1).all()


@pytest.mark.parametrize(
    ("capture_config", "dtype_name", "output_dtype"),
    [(256, "uint8", torch.uint8), (257, "uint16", torch.uint16)],
    indirect=["capture_config"],
)
def test_routed_experts_capturer_narrows_snapshot(capturer, dtype_name, output_dtype):
    assert capturer.shape_per_token == (4, 2)
    assert capturer.output_dtype_name == dtype_name
    assert capturer.output_dtype == output_dtype
    assert capturer.device_buffer.shape == (8, 4, 2)
    max_id = 256 if output_dtype == torch.uint16 else 255
    topk = torch.tensor([[1, 2], [max_id - 1, max_id]], dtype=torch.int64)
    _capture(capturer, 0, topk, make_routed_experts_capture_layout((len(topk),), 0))

    output = capturer.snapshot_routing_data(2)
    assert capturer.device_buffer.dtype == torch.int32
    assert capturer.device_buffer[:2, 0, :].tolist() == topk.tolist()
    assert output.dtype == output_dtype
    assert output[:, 0, :].tolist() == topk.tolist()
    assert output.data_ptr() != capturer.device_buffer.data_ptr()


@pytest.mark.parametrize(
    "counts, sp_size, gathered, physical_rows, expected_rows",
    [
        ((2, 3), 1, True, 5, [2, 3, 4]),
        ((2, 3), 1, False, 3, [0, 1, 2]),
        ((3, 2), 2, True, 6, [4, 5]),
        ((3, 0), 2, True, 4, []),
        ((0, 3), 1, True, 3, [0, 1, 2]),
        ((0, 0), 2, True, 0, []),
        ((2, 2), 2, True, 4, [2, 3]),
    ],
    ids=[
        "dp-gather",
        "dp-local",
        "dp-sp-gather",
        "idle-rank",
        "only-active-rank",
        "all-idle",
        "unpadded-gather",
    ],
)
def test_capture_declared_dp_layout(
    counts, sp_size, gathered, physical_rows, expected_rows, capturer
):
    layout = make_routed_experts_capture_layout(
        counts, dp_rank=1, sp_size=sp_size, is_gathered=gathered
    )
    assert layout.input_rows == physical_rows
    ids = torch.arange(physical_rows * 2, dtype=torch.int32).reshape(-1, 2)
    _capture(capturer, 0, ids, layout)
    expected = torch.full_like(capturer.device_buffer, -1)
    expected[: len(expected_rows), 0] = ids[expected_rows]
    torch.testing.assert_close(capturer.device_buffer, expected)


def test_capture_missing_layout_does_not_guess_from_row_count(capturer):
    with (
        override_forward_context(_forward_context(batch_descriptor=BatchDescriptor(2))),
        pytest.raises(ValueError, match="Missing"),
    ):
        capturer.capture(0, torch.ones(2, 2, dtype=torch.int32))
    assert (capturer.device_buffer == -1).all()


def test_get_aux_output_connector_passes_config(monkeypatch):
    import vllm.distributed.aux_output_connector.worker as aux_output_worker

    connector = Mock()
    constructor = Mock(return_value=connector)
    monkeypatch.setattr(aux_output_worker, "AuxOutputWorkerConnector", constructor)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32)
    )
    model = Mock()
    kv_cache_config = Mock()

    result = aux_output_worker.get_aux_output_connector(model, config, kv_cache_config)

    constructor.assert_called_once_with(
        model=model,
        kv_cache_config=kv_cache_config,
        vllm_config=config,
    )
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
        vllm_config=config,
        model=model,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[_full_attention_kv_group()]),
    )

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
    monkeypatch.setattr(aux_output_worker, "RoutedExpertsBuffer", Mock())

    config = SimpleNamespace(
        aux_output_config=SimpleNamespace(max_bytes=None),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=32),
        max_concurrent_batches=2,
    )
    kwargs = dict(
        vllm_config=config,
        model=Mock(),
        kv_cache_config=SimpleNamespace(
            num_blocks=10,
            kv_cache_groups=[_full_attention_kv_group(MLAAttentionSpec)],
        ),
    )

    aux_output_worker.AuxOutputWorkerConnector(**kwargs)
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
def capture_config(tmp_path, request):
    from transformers import MixtralConfig

    model_path = tmp_path / "mixtral"
    MixtralConfig(
        architectures=["MixtralForCausalLM"],
        num_hidden_layers=4,
        num_local_experts=getattr(request, "param", 16),
        num_experts_per_tok=2,
    ).save_pretrained(model_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(model_path),
            dtype="float32",
            max_model_len=32,
            enforce_eager=True,
            skip_tokenizer_init=True,
        ),
        device_config=DeviceConfig(device="cpu"),
    )

    return config


@pytest.fixture
def capturer(capture_config, monkeypatch):
    from vllm.platforms.cpu import CpuPlatform

    monkeypatch.setattr(f"{_REC_MODULE}.current_platform", CpuPlatform())
    capturer = RoutedExpertsCapturer(8, capture_config)
    capturer.device_buffer.fill_(-1)
    return capturer


@pytest.mark.parametrize(
    ("ranges", "expected_rows"),
    [
        (((7, 7),), list(range(7, 14))),
        (((0, 3),), [0, 1, 2]),
        (((0, 3), (7, 2)), [0, 1, 2, 7, 8]),
        ((), []),
        (((14, 0),), []),
    ],
)
def test_explicit_capture_layout_selects_padded_dp_block(
    ranges, expected_rows, capturer
):
    ids = torch.arange(28, dtype=torch.int32).reshape(14, 2)
    layout = RoutedExpertsCaptureLayout(ranges, len(ids))

    _capture(capturer, 0, ids, layout)

    expected = torch.full_like(capturer.device_buffer, -1)
    expected[: len(expected_rows), 0] = ids[expected_rows]
    torch.testing.assert_close(capturer.device_buffer, expected)


def test_explicit_capture_layout_overwrites_previous_rows(capturer):
    ids = torch.arange(14, dtype=torch.int32).reshape(7, 2)
    _capture(capturer, 0, ids, RoutedExpertsCaptureLayout(((0, 7),), 7))
    previous = capturer.snapshot_routing_data(7)

    _capture(capturer, 0, ids, RoutedExpertsCaptureLayout(((5, 2),), 7))
    current = capturer.snapshot_routing_data(2)

    assert current.shape == (2, 4, 2)
    torch.testing.assert_close(current[:, 0], ids[5:].to(capturer.output_dtype))
    torch.testing.assert_close(previous[:, 0], ids.to(capturer.output_dtype))
    assert (capturer.device_buffer[:, 1:] == -1).all()


def test_explicit_capture_layout_preserves_microbatch_rows(capturer):
    ids = torch.arange(10, dtype=torch.int32).reshape(5, 2)
    with patch(f"{_REC_MODULE}.dbo_current_ubatch_token_offset", side_effect=[0, 3]):
        _capture(capturer, 0, ids, RoutedExpertsCaptureLayout(((0, 3),), 5))
        _capture(capturer, 0, ids, RoutedExpertsCaptureLayout(((3, 2),), 5))

    torch.testing.assert_close(capturer.device_buffer[:5, 0], ids)
    assert (capturer.device_buffer[5:, 0] == -1).all()
    assert (capturer.device_buffer[:, 1:] == -1).all()


def _capture_group(rank=1, world_size=2):
    from vllm.platforms.cpu import CpuPlatform

    with (
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.new_group"),
        patch("vllm.platforms.current_platform", CpuPlatform()),
    ):
        return GroupCoordinator([list(range(world_size))], rank, "gloo", False)


@pytest.mark.parametrize(
    ("shard_sizes", "ranges", "expected_rows"),
    [
        ((3, 2), ((0, 5),), [0, 1, 2, 3, 4]),
        ((7, 7), ((0, 3), (7, 2)), [0, 1, 2, 7, 8]),
        ((7, 7), ((7, 2), (0, 3)), [7, 8, 0, 1, 2]),
        ((2, 0), (), []),
        ((0, 2), ((0, 2),), [0, 1]),
        ((0, 0), (), []),
    ],
)
@pytest.mark.parametrize("rank", [0, 1])
def test_explicit_capture_layout_gathers_and_skips_internal_padding(
    shard_sizes, ranges, expected_rows, rank, capturer
):
    gather_group = _capture_group(rank)
    rows = torch.arange(sum(shard_sizes), dtype=torch.int32)
    ids = torch.stack((rows, (rows + 1) % 16))
    peers = ids.T.split(shard_sizes)
    local = peers[rank]
    if len(local) > 1:
        assert not local.is_contiguous()
    layout = RoutedExpertsCaptureLayout(ranges, len(local), gather_group, shard_sizes)

    def gather(input_, dim, sizes):
        assert dim == 0
        assert sizes == list(shard_sizes)
        assert input_.is_contiguous()
        torch.testing.assert_close(input_, local)
        return torch.cat(peers)

    with patch.object(gather_group, "all_gatherv", side_effect=gather) as collective:
        _capture(capturer, 0, local, layout)

    collective.assert_called_once()
    expected = torch.full_like(capturer.device_buffer, -1)
    expected[: len(expected_rows), 0] = ids.T[expected_rows]
    torch.testing.assert_close(capturer.device_buffer, expected)


@pytest.mark.parametrize("span", [(1, 2), (-1, 1), (0, -1), (3, 0)])
def test_explicit_capture_layout_rejects_invalid_source_rows(span):
    with pytest.raises(ValueError, match="source range"):
        RoutedExpertsCaptureLayout((span,), 2)


@pytest.mark.parametrize(
    ("sizes", "has_group", "message"),
    [
        ((2,), True, "group size"),
        ((2, -1), True, "non-negative"),
        ((2, 3), True, "local rows"),
        ((2, 2), False, "require a gather group"),
    ],
)
def test_capture_rejects_invalid_collective_layout(sizes, has_group, message):
    group = _capture_group() if has_group else None
    with pytest.raises(ValueError, match=message):
        RoutedExpertsCaptureLayout(((0, 2),), 2, group, sizes)


@pytest.mark.parametrize("layer_id", [-1, 4])
def test_capture_layout_rejects_invalid_layer_before_collective(layer_id, capturer):
    group = _capture_group()
    with (
        patch.object(group, "all_gatherv") as gather,
        pytest.raises(IndexError, match="layer"),
    ):
        _capture(
            capturer,
            layer_id,
            torch.ones((2, 2), dtype=torch.int32),
            layout=RoutedExpertsCaptureLayout(((0, 4),), 2, group, (2, 2)),
        )
    gather.assert_not_called()
    assert (capturer.device_buffer == -1).all()


@pytest.mark.parametrize(("offset", "rows"), [(0, 9), (1, 8)])
def test_capture_layout_does_not_write_past_buffer_capacity(offset, rows, capturer):
    with (
        patch(f"{_REC_MODULE}.dbo_current_ubatch_token_offset", return_value=offset),
        pytest.raises(ValueError, match="capacity"),
    ):
        _capture(
            capturer,
            0,
            torch.ones((rows, 2), dtype=torch.int32),
            layout=RoutedExpertsCaptureLayout(((0, rows),), rows),
        )
    assert (capturer.device_buffer == -1).all()


@pytest.mark.parametrize("sp_size, rank", [(1, 0), (2, 0), (2, 1)])
def test_mega_moe_provides_capture_layout(sp_size, rank, capturer, capture_config):
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as mega
    from vllm.platforms.cpu import CpuPlatform

    group = _capture_group(rank)
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
            **({"input_sp_size": sp_size} if sp_size > 1 else {}),
        )
    bind_routed_experts_capturer(experts, capturer)
    ctx = _forward_context()
    ids = torch.arange(6, dtype=torch.int32).reshape(3, 2)
    padded = torch.cat((ids, torch.full((1, 2), 99, dtype=torch.int32)))
    with (
        override_forward_context(ctx),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=group),
        patch.object(group, "all_gatherv", return_value=padded) as gather,
    ):
        ctx.routed_experts_capture_layouts = prepare_routed_experts_capture_layouts(
            experts, 3, torch.tensor([5, 3]), 1
        )
        assert ctx.routed_experts_capture_layouts is not None
        layout = ctx.routed_experts_capture_layouts[1]
        assert layout.input_rows == (2 if sp_size > 1 else 3)
        assert layout.shard_sizes == ((2, 2) if sp_size > 1 else ())
        assert layout.source_ranges == ((0, 3),)
        assert experts.capture_fn is not None
        experts.capture_fn(padded[rank * 2 : (rank + 1) * 2] if sp_size > 1 else ids)
    assert gather.call_count == int(sp_size > 1)
    expected = torch.full_like(capturer.device_buffer, -1)
    expected[:3, 1] = ids
    torch.testing.assert_close(capturer.device_buffer, expected)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("shard_sizes", [(3, 2), (4, 4)])
def test_custom_provider_restores_uneven_or_padded_rows(rank, shard_sizes, capturer):
    """A backend provider uses its active context without the built-in DP/SP helper."""

    class Source(torch.nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id
            self.capture_fn: Callable[[torch.Tensor], None] | None = None

        def forward(self, ids):
            assert self.capture_fn is not None
            self.capture_fn(ids)

    group = _capture_group(rank)

    class MoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([Source(i) for i in range(3)])
            self.layout_calls = 0

        def get_routed_experts_capture_layouts(self, counts, dp_rank):
            self.layout_calls += 1
            first, second = get_forward_context().additional_kwargs["valid_rows"]
            local = RoutedExpertsCaptureLayout(((0, counts[dp_rank]),), counts[dp_rank])
            gathered = RoutedExpertsCaptureLayout(
                ((0, first), (shard_sizes[0], second)),
                shard_sizes[rank],
                group,
                shard_sizes,
            )
            return {0: local, 1: gathered, 2: local}

    model = MoE()
    assert prepare_routed_experts_capture_layouts(model, 5) is None
    bind_routed_experts_capturer(model, capturer)
    assert model.layout_calls == 0
    assert prepare_routed_experts_capture_layouts(torch.nn.Identity(), 5) is None
    for step, valid_rows in enumerate(((3, 2), (1, 1)), start=1):
        rows = sum(valid_rows)
        ids = torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2) + step * 10
        shards = [torch.full((n, 2), -99, dtype=torch.int32) for n in shard_sizes]
        shards[0][: valid_rows[0]] = ids[: valid_rows[0]]
        shards[1][: valid_rows[1]] = ids[valid_rows[0] :]
        ctx = _forward_context()
        ctx.additional_kwargs["valid_rows"] = valid_rows
        capturer.device_buffer.fill_(-1)
        with override_forward_context(ctx):
            layouts = prepare_routed_experts_capture_layouts(
                model, rows, torch.tensor([7, rows]), 1
            )
            assert layouts is not None and layouts[0] is layouts[2]
            ctx.routed_experts_capture_layouts = layouts
            with patch.object(
                group, "all_gatherv", return_value=torch.cat(shards)
            ) as gather:
                model.experts[0](ids)
                model.experts[1](shards[rank])
                model.experts[2](ids)
            assert gather.call_args.kwargs == {"dim": 0, "sizes": list(shard_sizes)}
        assert model.layout_calls == step
        expected = torch.full_like(capturer.device_buffer, -1)
        expected[:rows, :3] = ids[:, None, :]
        torch.testing.assert_close(capturer.device_buffer, expected)


@pytest.mark.parametrize("counts", [(-1, 2), (3, -2)])
def test_layout_rejects_negative_dp_counts(counts):
    with pytest.raises(ValueError, match="non-negative"):
        make_routed_experts_capture_layout(counts, 0, is_gathered=True)


@pytest.mark.parametrize("duplicate", [False, True])
def test_step_rejects_missing_or_ambiguous_layout_providers(duplicate, capturer):
    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_id = 0
            self.capture_fn = None

    class MoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Source()

        def get_routed_experts_capture_layouts(self, num_tokens_across_dp, dp_rank):
            return (
                {0: make_routed_experts_capture_layout(num_tokens_across_dp, dp_rank)}
                if duplicate
                else {}
            )

    model = torch.nn.ModuleList([MoE(), MoE()]) if duplicate else MoE()
    bind_routed_experts_capturer(model, capturer)
    with pytest.raises(ValueError, match="Duplicate" if duplicate else "Missing"):
        prepare_routed_experts_capture_layouts(model, 3)
