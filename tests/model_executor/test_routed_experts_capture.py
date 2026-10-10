# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
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


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
    dtype: torch.dtype = torch.int32,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.output_dtype = torch.uint8
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=dtype,
    )
    return c


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


def test_public_binding_supports_direct_capture_source():
    class DirectCaptureSource(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_id = 3
            self.capture_fn = None

    source = DirectCaptureSource()
    calls = []
    capturer = types.SimpleNamespace(capture=lambda *args: calls.append(args))

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
        SimpleNamespace(modules=lambda: [layer]), SimpleNamespace(capture=capture)
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


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


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
def test_routed_experts_capturer_narrows_snapshot(output_dtype):
    capturer = _capturer_with_buffer(dtype=torch.int32)
    capturer.output_dtype = output_dtype
    topk = torch.tensor([[1, 2], [254, 255]], dtype=torch.int64)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)

    output = capturer.snapshot_routing_data(2)
    assert capturer.device_buffer.dtype == torch.int32
    assert capturer.device_buffer[:2, 0, :].tolist() == topk.tolist()
    assert output.dtype == output_dtype
    assert output[:, 0, :].tolist() == topk.tolist()
    assert output.data_ptr() != capturer.device_buffer.data_ptr()


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """N == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """N == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_ep_gathered_shards():
    capturer = _capturer_with_buffer(dp_rank=1, tp_size=2)
    num_tokens_dp = torch.tensor([3, 2], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(
            num_tokens_across_dp_cpu=num_tokens_dp,
            local_sizes=[2, 2, 1, 1],
        )
    )
    topk = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [-1, -1], [10, 11], [12, 13]],
        dtype=torch.int32,
    )

    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)

    assert torch.equal(capturer.device_buffer[:2, 0], topk[4:])


def test_routed_experts_capturer_sp_modular_gathers_tp_shards():
    capturer = _capturer_with_buffer(dp_rank=1, tp_size=2)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    local_shard = torch.tensor([[10, 11], [12, 13]], dtype=torch.int32)
    gathered = torch.tensor([[10, 11], [12, 13], [14, 15], [-1, -1]], dtype=torch.int32)
    tp_group = Mock()
    tp_group.all_gather.return_value = gathered

    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        patch(f"{_REC_MODULE}.get_tp_group", return_value=tp_group),
    ):
        capturer.capture(layer_id=0, topk_ids=local_shard)

    tp_group.all_gather.assert_called_once_with(local_shard, dim=0)
    assert torch.equal(capturer.device_buffer[:3, 0], gathered[:3])


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
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
        aux_output_config=SimpleNamespace(
            enable_return_routed_experts=True, backend="shm"
        ),
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
    monkeypatch.setattr(aux_output_worker, "ShmBlockObjectStore", store_constructor)
    monkeypatch.setattr(
        aux_output_worker, "BackgroundBlockObjectStore", background_store_constructor
    )
    monkeypatch.setattr(aux_output_worker, "RoutedExpertsBuffer", Mock())

    config = SimpleNamespace(
        aux_output_config=SimpleNamespace(max_bytes=None, backend="shm"),
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
