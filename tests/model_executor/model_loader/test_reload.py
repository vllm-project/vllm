# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import importlib.machinery
import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import WeakKeyDictionary, ref

import pytest
import torch
from torch.nn.parameter import UninitializedParameter

import vllm.model_executor.model_loader.reload.layerwise as reload_layerwise
import vllm.model_executor.model_loader.reload.meta as reload_meta
from vllm.config import ModelConfig
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    initialize_online_processing,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.meta import (
    capture_layer_to_meta,
    get_numel_loaded,
    materialize_layer,
    materialize_meta_tensor,
    restore_layer_on_meta,
    to_meta_tensor,
)
from vllm.model_executor.model_loader.reload.mla import CommonMLAProcessingPolicy
from vllm.model_executor.model_loader.reload.moe import RoutedExpertsReloadPlan
from vllm.model_executor.model_loader.reload.trace import (
    ModelReloadTracer,
    ReloadError,
    ReloadState,
)
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader,
    default_weight_loader,
)
from vllm.platforms import current_platform


def test_model_finalize_binds_broadcast_created_after_state_builder():
    """Derived mHC storage must be bound after PWAL and survive reload."""
    from vllm.model_executor.model_loader.reload.model import (
        create_deepseek_model_reload_state,
    )

    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.hc_attn_fn = torch.nn.Parameter(torch.ones(2, 3))
    model.layer.hc_attn_fn_broadcast = None

    def finalize():
        model.layer.hc_attn_fn_broadcast.copy_(model.layer.hc_attn_fn.sum(0))

    model.finalize_mhc_broadcast_weights = finalize
    state = create_deepseek_model_reload_state(model, "model")
    assert state.dependencies == ("model.layer",)
    model.layer.hc_attn_fn_broadcast = torch.zeros(3)
    broadcast = model.layer.hc_attn_fn_broadcast
    state.policy.bind(state)
    state.policy.finish(state)
    assert model.layer.hc_attn_fn_broadcast is broadcast
    torch.testing.assert_close(broadcast, torch.full((3,), 2.0))
    model.layer.hc_attn_fn_broadcast = broadcast.clone()
    with pytest.raises(ReloadError):
        state.policy.finish(state)


def _trace_weight_loader(param, loaded_weight, shard_id=None):
    if shard_id == "remote":
        return False
    target = param if shard_id is None else param.narrow(0, shard_id * 2, 2)
    target.copy_(loaded_weight)


class _TraceCopyPolicy:
    def __init__(self, finished):
        self.finished = finished

    def bind(self, state):
        pass

    def validate(self, state):
        pass

    def destination(self, state, role, bound):
        return state.source(role, alias_runtime=True)

    def finish(self, state):
        for role in state.roles:
            state.copy_(role, state.work(role).mul_(2))
        self.finished.append(state.key)


def _make_reload_trace(runtime_name=None):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4, 3), requires_grad=False)
    layer.weight.weight_loader = _trace_weight_loader
    finished: list[str] = []
    state = ReloadState("linear", layer, ("weight",), _TraceCopyPolicy(finished))
    trace = ModelReloadTracer()
    trace.register_state(state)
    with trace.observe():
        layer.weight.weight_loader(layer.weight, torch.ones(2, 3), 0)
        layer.weight.weight_loader(layer.weight, torch.ones(2, 3), 1)
        assert (
            layer.weight.weight_loader(layer.weight, torch.ones(2, 3), "remote")
            is False
        )
    if runtime_name is not None:
        setattr(layer, runtime_name, layer.weight)
        del layer.weight
        state.runtime_names = {"weight": runtime_name}
    trace.bind_runtime()
    return layer, trace, state, finished


@pytest.mark.parametrize("preserve", [False, True])
def test_reload_trace_loading_view_reuses_padded_storage(preserve):
    """A canonical proxy may borrow capacity without resizing the live tensor."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(6, 3), requires_grad=False)
    state = ReloadState("linear", layer, ("weight",), _TraceCopyPolicy([]))
    state.metadata["weight"] = to_meta_tensor(torch.zeros(4, 3))
    state.bind_target("weight", lambda: layer.weight)
    state.preserve_checkpoint = preserve
    view = layer.weight.detach().view(-1)[:12].view(4, 3)
    source = state.source("weight", alias_runtime=True, loading_view=view)
    source.fill_(7)
    state.targets["weight"].validate()
    assert source.shape == (4, 3)
    assert (
        source.untyped_storage().data_ptr() == layer.weight.untyped_storage().data_ptr()
    ) is (not preserve)
    torch.testing.assert_close(
        layer.weight[:4], torch.full((4, 3), 0.0 if preserve else 7.0)
    )
    torch.testing.assert_close(layer.weight[4:], torch.zeros(2, 3))


def test_reload_trace_loading_view_rejects_unrelated_storage():
    """A policy cannot masquerade an independent allocation as a runtime view."""
    layer, _, state, _ = _make_reload_trace()
    with pytest.raises(ReloadError, match="invalid runtime loading view"):
        state.source(
            "weight", alias_runtime=True, loading_view=torch.empty_like(layer.weight)
        )


@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_reload_trace_cutlass_prepares_scale_layout(block, preserve):
    """Block scales reuse runtime storage; merged tensor scales need staging."""
    from vllm.model_executor.model_loader.reload.fp8 import CutlassMoEReloadPolicy

    role = "w13_weight_scale_inv" if block else "w13_weight_scale"
    shape = (2, 4, 2) if block else (2, 2)
    runtime_shape = shape if block else (2,)
    layer = torch.nn.Module()
    runtime = torch.nn.Parameter(torch.ones(runtime_shape), requires_grad=False)
    layer.register_parameter(role, runtime)
    policy = CutlassMoEReloadPolicy()
    policy.plan = types.SimpleNamespace(block_shape=(128, 128) if block else None)
    state = ReloadState("experts", layer, (role,), policy)
    state.metadata[role] = to_meta_tensor(torch.empty(shape))
    state.bind_target(role, lambda: getattr(layer, role))
    state.preserve_checkpoint = preserve

    policy.prepare_for_load(state)
    source = state.checkpoint[role]
    assert source.shape == shape
    assert (
        source.untyped_storage().data_ptr() == runtime.untyped_storage().data_ptr()
    ) == (block and not preserve)
    # Preparation never swaps old values or changes the live tensor's metadata.
    torch.testing.assert_close(runtime, torch.ones(runtime_shape))
    state.targets[role].validate()
    source.copy_(torch.full(shape, 3.0))
    torch.testing.assert_close(
        runtime, torch.full(runtime_shape, 3.0 if block and not preserve else 1.0)
    )


@pytest.mark.parametrize("backend", ["marlin", "humming"])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("encoded_scale", [False, True])
def test_reload_trace_packed_policy_loading_sources(
    backend, moe, preserve, encoded_scale
):
    """Packed weights stage; only compatible scale/bias inputs borrow storage."""
    from vllm.model_executor.model_loader.reload.fp8 import (
        HummingFP8LinearReloadPolicy,
        HummingMoEReloadPolicy,
        MarlinFP8LinearReloadPolicy,
        MarlinMoEReloadPolicy,
    )

    policies = {
        ("marlin", False): MarlinFP8LinearReloadPolicy,
        ("marlin", True): MarlinMoEReloadPolicy,
        ("humming", False): HummingFP8LinearReloadPolicy,
        ("humming", True): HummingMoEReloadPolicy,
    }
    kwargs: dict[str, bool | int] = dict(block_quant=True)
    if moe:
        kwargs.update(is_act_and_mul=True, shard_size=4, num_experts=2)
    policy = policies[backend, moe](**kwargs)
    # Isolate destination allocation from backend initialization/kernel imports.
    policy.validate = lambda state: None
    prefixes = ("w13_", "w2_") if moe else ("",)
    roles = tuple(
        prefix + suffix
        for prefix in prefixes
        for suffix in ("weight", "weight_scale_inv")
    ) + (() if moe else ("bias",))
    layer = torch.nn.Module()
    state = ReloadState("packed", layer, roles, policy)
    state.preserve_checkpoint = preserve
    for role in roles:
        weight = role.endswith("weight")
        meta_dtype = torch.float8_e4m3fn if weight else torch.float32
        runtime_dtype = (
            torch.int32
            if weight or ("scale" in role and encoded_scale)
            else torch.float32
        )
        runtime_name = (
            role.replace("weight_scale_inv", "weight_scale")
            if backend == "humming"
            else role
        )
        state.runtime_names[role] = runtime_name
        state.metadata[role] = to_meta_tensor(torch.empty(4, dtype=meta_dtype))
        layer.register_parameter(
            runtime_name,
            torch.nn.Parameter(
                torch.zeros(8, dtype=runtime_dtype), requires_grad=False
            ),
        )
        state.bind_target(role, lambda name=runtime_name: getattr(layer, name))
    # Extra Humming-derived outputs are not checkpoint inputs.
    layer.derived_scale = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    state.bind_target("derived", lambda: layer.derived_scale)
    bound = inspect.signature(default_weight_loader).bind(None, None)
    first = policy.destination(state, roles[0], bound)
    assert set(state.checkpoint) == set(roles)
    assert policy.destination(state, roles[0], bound) is first
    for role, source in state.checkpoint.items():
        runtime = state.targets[role].tensor
        aliases = (
            source.untyped_storage().data_ptr() == runtime.untyped_storage().data_ptr()
        )
        expected = not preserve and (
            role == "bias"
            or (backend == "humming" and "scale" in role and not encoded_scale)
        )
        assert aliases == expected
        assert source.shape == state.metadata[role].shape
        assert source.dtype == state.metadata[role].dtype
        source.copy_(torch.full((4,), 2, dtype=source.dtype))
        state.targets[role].validate()
        torch.testing.assert_close(
            runtime[:4], torch.full((4,), 2 if aliases else 0, dtype=runtime.dtype)
        )
        torch.testing.assert_close(runtime[4:], torch.zeros(4, dtype=runtime.dtype))
    torch.testing.assert_close(layer.derived_scale, torch.ones(2))


def test_mla_reload_lifecycle_retains_projection():
    """Cold MLA PWAL retains kv_b_proj; binding and reload must preserve it."""
    from vllm.model_executor.layers.attention import MLAAttention
    from vllm.model_executor.model_loader.reload.integration import (
        create_model_reload_tracer,
    )

    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_b_proj = torch.nn.Module()
    source = layer.kv_b_proj
    source.quant_method = None
    source.weight = torch.nn.Parameter(
        torch.zeros(6, 4, dtype=torch.bfloat16, device="cuda"),
        requires_grad=False,
    )
    source.weight.weight_loader = default_weight_loader
    layer.num_heads = 2
    layer.kv_lora_rank = 4
    layer.qk_nope_head_dim = 2
    layer.v_head_dim = 1
    layer.is_amx_bmm_enabled = False
    layer.is_aiter_triton_fp8_bmm_enabled = False
    layer.is_aiter_triton_fp4_bmm_enabled = False
    layer.dcp_q_replicate = False
    layer.W_UK_T_dcp_qrep = None
    layer.quant_config = None
    for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
        layer.register_buffer(name, torch.ones((), device="cuda"))
    layer.impl = types.SimpleNamespace(process_weights_after_loading=lambda dtype: None)
    model = torch.nn.Module()
    model.attn = layer
    trace = create_model_reload_tracer(model)
    weight = torch.arange(24, device="cuda", dtype=torch.bfloat16).reshape(6, 4)
    with trace.observe():
        source.weight.weight_loader(source.weight, weight)
    layer.process_weights_after_loading(torch.bfloat16)
    trace.bind_runtime()
    targets = (source.weight, layer.W_UK_T, layer.W_UV)
    pointers = tuple(t.data_ptr() for t in targets)
    for multiplier in (2, 3):
        fresh = weight * multiplier
        with trace.round():
            source.weight.weight_loader(source.weight, fresh)
        expected = fresh.T.reshape(4, 2, 3)
        torch.testing.assert_close(source.weight, fresh)
        torch.testing.assert_close(layer.W_UK_T, expected[:, :, :2].permute(1, 2, 0))
        torch.testing.assert_close(layer.W_UV, expected[:, :, 2:].transpose(0, 1))
        assert tuple(t.data_ptr() for t in targets) == pointers
        assert source.weight is targets[0]
        assert layer.W_UK_T is targets[1]
        assert layer.W_UV is targets[2]


def test_mla_processing_policy_splits_bf16_kv_b_projection():
    layer = types.SimpleNamespace(
        num_heads=2,
        kv_lora_rank=3,
        qk_nope_head_dim=2,
        v_head_dim=1,
    )
    weight = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    values = CommonMLAProcessingPolicy().process_reload(layer, weight, torch.float32)
    w_uk_t, w_uv = values["W_UK_T"], values["W_UV"]

    expected = weight.T.reshape(3, 2, 3)
    assert torch.equal(w_uk_t, expected[:, :, :2].permute(1, 2, 0))
    assert torch.equal(w_uv, expected[:, :, 2:].transpose(0, 1))


def test_mla_processing_policy_consumes_dequantized_canonical_projection():
    """MLA consumes the canonical projection, not backend-packed FP8 storage."""
    layer = types.SimpleNamespace(
        num_heads=1,
        kv_lora_rank=2,
        qk_nope_head_dim=1,
        v_head_dim=1,
    )
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor(2.0)

    values = CommonMLAProcessingPolicy().process_reload(
        layer, weight.to(torch.float32) * scale, torch.float32
    )
    w_uk_t, w_uv = values["W_UK_T"], values["W_UV"]

    expected = (weight.to(torch.float32) * 2).T.reshape(2, 1, 2)
    assert torch.equal(w_uk_t, expected[:, :, :1].permute(1, 2, 0))
    assert torch.equal(w_uv, expected[:, :, 1:].transpose(0, 1))


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize(
    "layout,can_reuse",
    [
        ("transpose", True),
        ("padded", True),
        ("expanded", False),
        ("strided", False),
        ("strided_checkpoint", False),
        ("small", False),
        ("dtype", False),
    ],
)
def test_reload_trace_prepares_canonical_storage(layout, can_reuse, preserve):
    """Reuse dense capacity, never holes, overlapping views or encoded dtypes."""
    runtime = {
        "transpose": lambda: torch.zeros(4, 3).t(),
        "padded": lambda: torch.zeros(6, 4).t(),
        "expanded": lambda: torch.zeros(1, 3).expand(4, 3),
        "strided": lambda: torch.zeros(4, 6)[:, ::2],
        "strided_checkpoint": lambda: torch.zeros(4, 6)[:, ::2],
        "small": lambda: torch.zeros(2, 3),
        "dtype": lambda: torch.zeros(4, 3, dtype=torch.float16),
    }[layout]()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(runtime, requires_grad=False)
    state = ReloadState("linear", layer, ("weight",), _TraceCopyPolicy([]))
    # .to("meta") compacts non-dense strides, so construct this metadata directly.
    checkpoint = torch.empty_strided(
        (4, 3),
        (6, 2) if layout == "strided_checkpoint" else (3, 1),
        device="meta",
    )
    state.metadata["weight"] = checkpoint
    state.bind_target("weight", lambda: layer.weight)
    state.preserve_checkpoint = preserve
    state.prepare_sources(reuse_roles=("weight",))
    source = state.checkpoint["weight"]
    assert source.shape == (4, 3)
    assert source.stride() == checkpoint.stride()
    assert source.dtype == torch.float32
    aliases = (
        source.untyped_storage().data_ptr() == runtime.untyped_storage().data_ptr()
    )
    assert aliases == (can_reuse and not preserve)
    source.fill_(7)
    state.targets["weight"].validate()
    if not aliases:
        torch.testing.assert_close(runtime, torch.zeros_like(runtime))


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("abort", [False, True])
def test_reload_trace_renamed_parameter_loads_by_checkpoint_name(preserve, abort):
    """Backend renaming must not hide checkpoint keys or replace live weights."""
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4, 3), requires_grad=False)
    layer.weight.weight_loader = _trace_weight_loader
    state = ReloadState(
        "linear",
        layer,
        ("weight",),
        _TraceCopyPolicy([]),
        runtime_names={"weight": "packed_weight"},
    )
    trace = ModelReloadTracer()
    trace.register_state(state)
    with trace.observe():
        layer.weight.weight_loader(layer.weight, torch.ones(4, 3))
    layer.packed_weight = layer.weight
    del layer.weight
    trace.bind_runtime()
    runtime = layer.packed_weight
    address = runtime.data_ptr()

    with trace.round():
        assert "weight" in dict(layer.named_parameters())
    assert not hasattr(layer, "weight")
    torch.testing.assert_close(runtime, torch.ones_like(runtime))

    for value in (2, 3):
        try:
            with trace.round(preserve_checkpoint=preserve):
                params = dict(layer.named_parameters())
                param = params["weight"]
                assert params["packed_weight"] is runtime
                assert param is not runtime
                if abort:
                    raise ValueError("interrupt before arrival")
                param.weight_loader(param, torch.full((4, 3), float(value)))
                assert state.complete
                assert bool(state.checkpoint) == preserve
        except ValueError:
            assert abort
        assert not hasattr(layer, "weight")
        assert layer.packed_weight is runtime
        assert runtime.data_ptr() == address
        torch.testing.assert_close(
            runtime, torch.full_like(runtime, 1 if abort else value * 2)
        )
        if abort:
            assert trace.failed
            break


def test_reload_trace_renamed_parameter_rejects_existing_checkpoint_attribute():
    """Installing a checkpoint alias must not overwrite an unrelated tensor."""
    layer, trace, _, _ = _make_reload_trace(runtime_name="packed_weight")
    unexpected = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    layer.weight = unexpected
    with pytest.raises(ReloadError, match="checkpoint alias already exists"):
        trace.begin_round()
    assert layer.weight is unexpected
    assert not trace.active


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("interrupt_setup", [False, True])
def test_reload_trace_consumed_scale_has_no_runtime_parameter(
    monkeypatch, preserve, interrupt_setup
):
    """A discarded checkpoint scale still participates in per-layer readiness."""

    class ScalePolicy(_TraceCopyPolicy):
        def finish(self, state):
            state.copy_("weight", state.work("weight") * state.work("scale"))

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(4, 3), requires_grad=False)
    layer.scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)
    state = ReloadState(
        "linear",
        layer,
        ("weight", "scale"),
        ScalePolicy([]),
        runtime_names={"scale": None},
    )
    trace = ModelReloadTracer()
    trace.register_state(state)
    with trace.observe():
        for param in layer.parameters():
            param.weight_loader(param, torch.ones_like(param))
    del layer.scale
    trace.bind_runtime()
    runtime = layer.weight
    if interrupt_setup:
        install = trace._wrap

        def fail_on_scale(state, role, param):
            if role == "scale":
                raise RuntimeError("interrupted proxy setup")
            install(state, role, param)

        monkeypatch.setattr(trace, "_wrap", fail_on_scale)
        with pytest.raises(RuntimeError, match="interrupted proxy setup"):
            trace.begin_round(preserve_checkpoint=preserve)
        assert trace.failed and not trace.active
        assert "scale" not in dict(layer.named_parameters())
        assert not hasattr(runtime, "weight_loader")
        return
    for value in (2, 3):
        with trace.round(preserve_checkpoint=preserve):
            params = dict(layer.named_parameters())
            params["scale"].weight_loader(params["scale"], torch.tensor(float(value)))
            assert not state.complete
            params["weight"].weight_loader(params["weight"], torch.ones(4, 3))
            assert state.complete
        assert "scale" not in dict(layer.named_parameters())
        assert layer.weight is runtime
        torch.testing.assert_close(runtime, torch.full_like(runtime, value))
        assert bool(state.checkpoint) == preserve


def test_reload_trace_base_loader_captures_checkpoint_load(monkeypatch):
    """The production opt-in must observe before processing and bind afterwards."""
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.model_executor.model_loader import base_loader
    from vllm.model_executor.model_loader.reload.integration import (
        get_model_reload_tracer,
    )

    class Loader(base_loader.BaseModelLoader):
        def download_model(self, model_config):
            pass

        def create_model(self, **kwargs):
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            model.weight.weight_loader = default_weight_loader
            return model

        def load_weights(self, model, model_config):
            model.weight.weight_loader(model.weight, torch.ones(2))

    def process(model, config, device):
        assert not hasattr(model, "_reload_tracer")
        assert model.weight.weight_loader is default_weight_loader
        torch.testing.assert_close(model.weight, torch.ones(2))

    monkeypatch.setattr(base_loader, "process_weights_after_loading", process)
    monkeypatch.setattr(
        base_loader,
        "current_platform",
        types.SimpleNamespace(is_cuda_alike=lambda: False, is_xpu=lambda: False),
    )
    config = types.SimpleNamespace(
        weight_transfer_config=WeightTransferConfig(reload_mode="trace"),
        device_config=types.SimpleNamespace(device="cpu"),
        load_config=types.SimpleNamespace(device=None),
    )
    model_config = types.SimpleNamespace(dtype=torch.float32)
    model = Loader(config.load_config).load_model(config, model_config)
    trace = get_model_reload_tracer(model)
    with trace.round():
        model.weight.weight_loader(model.weight, torch.full((2,), 4.0))
    torch.testing.assert_close(model.weight, torch.full((2,), 4.0))


def test_reload_trace_tied_plain_parameters_have_one_owner():
    """A tied embedding/head must complete once while both references stay valid."""
    from vllm.model_executor.model_loader.reload.integration import (
        create_model_reload_tracer,
    )

    model = torch.nn.Module()
    model.embedding = torch.nn.Module()
    model.head = torch.nn.Module()
    weight = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    weight.weight_loader = default_weight_loader
    model.embedding.weight = model.head.weight = weight
    trace = create_model_reload_tracer(model)
    with trace.observe():
        weight.weight_loader(weight, torch.ones(2))
    trace.bind_runtime()
    with trace.round():
        weight.weight_loader(weight, torch.full((2,), 3.0))
        assert all(state.complete for state in trace.states.values())
    assert model.embedding.weight is model.head.weight is weight
    torch.testing.assert_close(model.head.weight, torch.full((2,), 3.0))


def test_reload_trace_skips_frozen_parameters():
    """Fixed lookup tables stay cold-load-only while normal weights reload."""
    from vllm.model_executor.model_loader.reload.integration import (
        create_model_reload_tracer,
    )

    model = torch.nn.Module()
    model.lookup = torch.nn.Module()
    model.linear = torch.nn.Module()

    lookup = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    lookup.weight_loader = default_weight_loader
    model.lookup.weight = lookup

    weight = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    weight.weight_loader = default_weight_loader
    model.linear.weight = weight

    trace = create_model_reload_tracer(
        model, frozen_parameter_names=["lookup.weight"]
    )
    assert "lookup" not in trace.states
    assert "linear" in trace.states

    with trace.observe():
        lookup.weight_loader(lookup, torch.ones(4))
        weight.weight_loader(weight, torch.full((4,), 2.0))
    trace.bind_runtime()

    with trace.round():
        weight.weight_loader(weight, torch.full((4,), 3.0))

    torch.testing.assert_close(lookup, torch.ones(4))
    torch.testing.assert_close(weight, torch.full((4,), 3.0))


@pytest.mark.parametrize("preserve", [False, True])
def test_reload_trace_repeated_rounds_keep_storage_and_checkpoint(preserve):
    """Reload changes values, not runtime identity; preservation keeps raw input."""
    layer, trace, state, finished = _make_reload_trace()
    original = layer.weight
    pointer = original.data_ptr()
    for value in (3.0, 5.0):
        with trace.round(preserve_checkpoint=preserve):
            for shard in (1, 0):
                source = torch.full((2, 3), value)
                layer.weight.weight_loader(layer.weight, source, shard)
                source.zero_()
        assert layer.weight is original
        assert layer.weight.data_ptr() == pointer
        torch.testing.assert_close(layer.weight, torch.full((4, 3), 2 * value))
        assert trace.finish() is True
        assert state.complete
        if preserve:
            torch.testing.assert_close(
                state.checkpoint["weight"], torch.full((4, 3), value)
            )
            assert state.checkpoint["weight"].data_ptr() != pointer
        else:
            assert not state.checkpoint
    assert finished == ["linear", "linear"]
    assert layer.weight.weight_loader is _trace_weight_loader


@pytest.mark.parametrize("preserve", [False, True])
def test_reload_trace_missing_shard_poisoned_without_finish(preserve):
    layer, trace, state, finished = _make_reload_trace()
    old = layer.weight.detach().clone()
    with (
        pytest.raises(ReloadError, match="Missing reload slots"),
        trace.round(preserve_checkpoint=preserve),
    ):
        layer.weight.weight_loader(layer.weight, torch.full((2, 3), 9.0), 0)
    assert not finished
    assert not state.complete
    assert trace.failed
    assert trace.runtime_modified is (not preserve)
    if preserve:
        torch.testing.assert_close(layer.weight, old)
    with pytest.raises(ReloadError, match="failed"):
        trace.begin_round()


@pytest.mark.parametrize("error", ["duplicate", "unknown"])
def test_reload_trace_bad_arrival_rejected_before_its_write(error):
    layer, trace, _, finished = _make_reload_trace()
    trace.begin_round()
    source, shard = torch.full((2, 3), 7.0), 0
    if error == "duplicate":
        layer.weight.weight_loader(layer.weight, source, shard)
    else:
        shard = 2
    before = layer.weight.detach().clone()
    with pytest.raises(ReloadError):
        layer.weight.weight_loader(layer.weight, source, shard)
    torch.testing.assert_close(layer.weight, before)
    assert not finished
    trace.abort()
    assert layer.weight.weight_loader is _trace_weight_loader


def test_reload_trace_empty_round_and_nonlocal_arrivals_are_noops():
    layer, trace, state, finished = _make_reload_trace()
    with trace.round():
        assert (
            layer.weight.weight_loader(layer.weight, torch.ones(2, 3), "remote")
            is False
        )
    assert trace.finish() is False
    assert not finished
    assert not state.complete
    assert not trace.runtime_modified


def test_reload_trace_runtime_replacement_rejected_before_loading():
    layer, trace, _, _ = _make_reload_trace()
    layer.weight = torch.nn.Parameter(layer.weight.detach().clone())
    with pytest.raises(ReloadError, match="identity"):
        trace.begin_round()


def test_reload_trace_duplicate_after_eager_finish_does_not_rewrite_runtime():
    layer, trace, state, finished = _make_reload_trace()
    with pytest.raises(ReloadError, match="Duplicate"), trace.round():
        for shard in (0, 1):
            layer.weight.weight_loader(layer.weight, torch.ones(2, 3), shard)
        assert state.complete
        assert finished == ["linear"]
        before = layer.weight.detach().clone()
        try:
            layer.weight.weight_loader(layer.weight, torch.full((2, 3), 9.0), 0)
        finally:
            torch.testing.assert_close(layer.weight, before)


def test_reload_trace_finish_revalidates_completed_layer_targets():
    layer, trace, state, _ = _make_reload_trace()
    with pytest.raises(ReloadError, match="identity"), trace.round():
        for shard in (0, 1):
            layer.weight.weight_loader(layer.weight, torch.ones(2, 3), shard)
        assert state.complete
        layer.weight = torch.nn.Parameter(layer.weight.detach().clone())


@pytest.mark.parametrize("preserve", [False, True])
def test_reload_trace_conversion_failure_does_not_complete_state(monkeypatch, preserve):
    layer, trace, state, finished = _make_reload_trace()
    old = layer.weight.detach().clone()

    def fail(state):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(state.policy, "finish", fail)
    with (
        pytest.raises(RuntimeError, match="conversion failed"),
        trace.round(preserve_checkpoint=preserve),
    ):
        for shard in (0, 1):
            layer.weight.weight_loader(layer.weight, torch.full((2, 3), 7.0), shard)
    assert trace.failed
    assert not state.complete
    assert not state.checkpoint
    assert not finished
    assert trace.runtime_modified is (not preserve)
    if preserve:
        torch.testing.assert_close(layer.weight, old)
    assert layer.weight.weight_loader is _trace_weight_loader


@pytest.mark.parametrize("empty_root", [False, True])
def test_reload_trace_dependencies_finish_once_after_sources(empty_root):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4, 3), requires_grad=False)
    layer.weight.weight_loader = _trace_weight_loader
    finished: list[str] = []
    trace = ModelReloadTracer()
    graph = [
        ("parent", ("left", "right"), ()),
        ("right", ("left",), ()),
        ("left", ("root",) if empty_root else (), ("weight",)),
    ]
    if empty_root:
        graph.append(("root", (), ()))
    for name, deps, roles in graph:
        trace.register_state(
            ReloadState(name, layer, roles, _TraceCopyPolicy(finished), deps)
        )
    with trace.observe():
        layer.weight.weight_loader(layer.weight, torch.ones(4, 3))
    trace.bind_runtime()
    expected = (["root"] if empty_root else []) + ["left", "right", "parent"]
    with trace.round():
        layer.weight.weight_loader(layer.weight, torch.ones(4, 3))
        assert finished == expected
    assert finished == expected
    trace.finish()
    assert finished == expected


@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_reload_trace_finishes_ready_layers_before_stream_ends(dependent, preserve):
    """A ready layer releases staging without waiting for unrelated weights."""
    trace = ModelReloadTracer()
    finished: list[str] = []
    layers = {}
    for name in ("first", "second"):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(4, 3), requires_grad=False)
        layer.weight.weight_loader = _trace_weight_loader
        layers[name] = layer
        deps = ("second",) if dependent and name == "first" else ()
        trace.register_state(
            ReloadState(name, layer, ("weight",), _TraceCopyPolicy(finished), deps)
        )
    with trace.observe():
        for layer in layers.values():
            layer.weight.weight_loader(layer.weight, torch.ones(4, 3))
    trace.bind_runtime()
    with trace.round(preserve_checkpoint=preserve):
        first, second = layers.values()
        first.weight.weight_loader(first.weight, torch.ones(4, 3))
        first_state = trace.states["first"]
        assert first_state.complete is (not dependent)
        assert bool(first_state.checkpoint) is (dependent or preserve)
        assert not trace.states["second"].checkpoint
        second.weight.weight_loader(second.weight, torch.ones(4, 3))
        assert finished == (["second", "first"] if dependent else ["first", "second"])
        assert all(state.complete for state in trace.states.values())
        assert all(
            bool(state.checkpoint) is preserve for state in trace.states.values()
        )


def test_reload_trace_observes_shards_without_counting_tensor_operations(monkeypatch):
    def forbidden_counter():
        pytest.fail("ReloadTrace must not use CopyCounter")

    monkeypatch.setattr(reload_meta, "CopyCounter", forbidden_counter)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(4, 3), requires_grad=False)

    def loader(param, loaded_weight):
        param.zero_().add_(loaded_weight)

    layer.weight.weight_loader = loader
    trace = ModelReloadTracer()
    state = ReloadState("linear", layer, ("weight",), _TraceCopyPolicy([]))
    trace.register_state(state)
    with trace.observe():
        layer.weight.weight_loader(layer.weight, torch.ones(4, 3))
    trace.bind_runtime()
    with trace.round():
        layer.weight.weight_loader(layer.weight, torch.full((4, 3), 3.0))
        assert state.complete
    torch.testing.assert_close(layer.weight, torch.full((4, 3), 6.0))


def _make_expert_reload_trace():
    """Use real expert loaders on CPU, with two local slots out of four."""
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.mlp.experts"
    layer.ckpt_gate_proj_name = "gate_proj"
    layer.ckpt_down_proj_name = "down_proj"
    layer.ckpt_up_proj_name = "up_proj"
    layer.lora_base_layer_prefix = ""
    layer.is_fused_checkpoint_transposed = False
    layer.global_num_experts = 4
    layer.local_num_experts = 2
    layer.quant_config = None
    layer.quant_method = types.SimpleNamespace()
    layer.moe_config = types.SimpleNamespace(
        num_experts=4,
        num_logical_experts=3,
        is_act_and_mul=True,
        tp_rank=1,
        moe_parallel_config=types.SimpleNamespace(tp_size=2, enable_eplb=True),
    )
    layer.expert_map_manager = types.SimpleNamespace(
        num_fused_shared_experts=0,
        map_global_to_local=lambda expert: (-1, 0, -1, 1)[expert],
    )
    layer.eplb_state = EplbLayerState(
        logical_to_physical_map=torch.tensor([[0, 3], [1, -1], [2, -1]])
    )
    for name, shape in (("w13_weight", (2, 4, 3)), ("w2_weight", (2, 3, 2))):
        param = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)
        param.weight_loader = layer.weight_loader
        setattr(layer, name, param)
    state = ReloadState(
        "experts",
        layer,
        ("w13_weight", "w2_weight"),
        _TraceCopyPolicy([]),
        expert_plan=RoutedExpertsReloadPlan(),
    )
    trace = ModelReloadTracer()
    trace.register_state(state)
    with trace.observe():
        # MoE metadata capture must neither wrap nor learn from cold arrivals.
        assert layer.w13_weight.weight_loader == layer.weight_loader
    assert not state.slots.expected
    trace.bind_runtime()
    return layer, trace, state


def _expert_checkpoint():
    return [
        (
            f"{expert}.{proj}.weight",
            torch.arange(12, dtype=torch.float32).reshape(shape)
            + expert * 100
            + offset,
        )
        for expert in range(3)
        for proj, shape, offset in (
            ("gate_proj", (4, 3), 0),
            ("up_proj", (4, 3), 20),
            ("down_proj", (3, 4), 40),
        )
    ]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_reload_trace_expert_plan_tracks_eplb_replicas_and_tp(fused, preserve):
    """Each round routes logical weights to current physical replicas and TP slices."""
    layer, trace, state = _make_expert_reload_trace()
    checkpoint = dict(_expert_checkpoint())
    weights = list(checkpoint.items())
    if fused:
        weights = [
            (
                "gate_up_proj",
                torch.stack(
                    [
                        torch.cat(
                            [
                                checkpoint[f"{e}.gate_proj.weight"],
                                checkpoint[f"{e}.up_proj.weight"],
                            ]
                        )
                        for e in range(3)
                    ]
                ),
            ),
            (
                "down_proj",
                torch.stack([checkpoint[f"{e}.down_proj.weight"] for e in range(3)]),
            ),
        ]
    runtime = layer.w13_weight
    for placement, local_logical in (
        ([[0, 3], [1, -1], [2, -1]], [1, 0]),
        ([[2, -1], [0, -1], [1, 3]], [2, 2]),
    ):
        layer.eplb_state.logical_to_physical_map.copy_(torch.tensor(placement))
        with trace.round(preserve_checkpoint=preserve):
            assert len(state.slots.expected) == 6
            list(layer.load_weights(weights))
            assert state.complete
            assert bool(state.checkpoint) is preserve
        assert layer.w13_weight is runtime
        for local, logical in enumerate(local_logical):
            expected_w13 = torch.cat(
                [
                    checkpoint[f"{logical}.gate_proj.weight"][2:],
                    checkpoint[f"{logical}.up_proj.weight"][2:],
                ]
            )
            expected_w2 = checkpoint[f"{logical}.down_proj.weight"][:, 2:]
            torch.testing.assert_close(layer.w13_weight[local], expected_w13 * 2)
            torch.testing.assert_close(layer.w2_weight[local], expected_w2 * 2)


def test_reload_trace_expert_plan_rejects_missing_new_local_expert():
    layer, trace, state = _make_expert_reload_trace()
    layer.eplb_state.logical_to_physical_map.copy_(
        torch.tensor([[2, -1], [0, -1], [1, 3]])
    )
    with pytest.raises(ReloadError, match="Missing reload slots"), trace.round():
        list(layer.load_weights(_expert_checkpoint()[:-1]))
    assert not state.complete


def test_reload_trace_expert_plan_refreshes_rank_ownership():
    layer, trace, state = _make_expert_reload_trace()
    with trace.round():
        list(layer.load_weights(_expert_checkpoint()))
    layer.expert_map_manager.map_global_to_local = lambda expert: (0, -1, 1, -1)[expert]
    with trace.round():
        list(layer.load_weights(_expert_checkpoint()))
        assert state.complete
    checkpoint = dict(_expert_checkpoint())
    for local, logical in enumerate((0, 2)):
        torch.testing.assert_close(
            layer.w2_weight[local], checkpoint[f"{logical}.down_proj.weight"][:, 2:] * 2
        )


def test_reload_trace_expert_plan_requires_initialized_eplb_state():
    layer, trace, _ = _make_expert_reload_trace()
    layer.eplb_state.logical_to_physical_map = None
    with pytest.raises(ReloadError, match="initialized"):
        trace.begin_round()
    assert not trace.active
    assert not trace.runtime_modified


@pytest.mark.parametrize("after_complete", [False, True])
def test_reload_trace_expert_plan_rejects_mid_round_mapping_change(after_complete):
    layer, trace, state = _make_expert_reload_trace()
    with pytest.raises(ReloadError, match="mapping changed"), trace.round():
        if after_complete:
            list(layer.load_weights(_expert_checkpoint()))
            assert state.complete
        layer.eplb_state.logical_to_physical_map.copy_(
            torch.tensor([[2, -1], [0, -1], [1, 3]])
        )
        if not after_complete:
            before = layer.w13_weight.detach().clone()
            try:
                list(layer.load_weights(_expert_checkpoint()))
            finally:
                torch.testing.assert_close(layer.w13_weight, before)
    assert trace.failed


@pytest.mark.parametrize("dependency", ["missing", "self"])
def test_reload_trace_invalid_dependency_graph_rejected(dependency):
    trace = ModelReloadTracer()
    trace.register_state(
        ReloadState("self", torch.nn.Module(), (), _TraceCopyPolicy([]), (dependency,))
    )
    with trace.observe():
        pass
    with pytest.raises(ReloadError, match="dependency|cycle"):
        trace.bind_runtime()


def _fp8_reload_unsupported() -> bool:
    """Whether the FP8 reload/online-quantize tests should be skipped.

    ``supports_fp8()`` returns True on MI250 (gfx90a) because the general
    quantization paths upcast FP8 weights, but gfx90a has no native FP8 and
    cannot run these reload models, so treat it as unsupported here.
    """
    if not current_platform.supports_fp8():
        return True
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx90a

        return on_gfx90a()
    return False


class _AliasedBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.weight = torch.nn.Parameter(weight)
        self.register_buffer(
            "weight_view", self.weight.detach().view(-1), persistent=False
        )


class _ParentAliasedChildBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.conv1d = torch.nn.Linear(3, 2, bias=False)
        self.conv1d.weight.data.copy_(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        self.register_buffer(
            "conv_weights", self.conv1d.weight.detach().view(-1), persistent=False
        )


class _ChildAliasOnlyBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Linear(3, 2, bias=False)
        self.conv1d.weight.data.copy_(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        self.register_buffer(
            "conv_weights", self.conv1d.weight.detach().view(-1), persistent=False
        )


class _AliasedBufferWithUninitializedChildLayer(_AliasedBufferLayer):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Module()
        self.child.register_parameter(
            "lazy_weight", UninitializedParameter(requires_grad=False)
        )


class _NonPersistentBufferLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.register_buffer("scale", torch.tensor(0.25), persistent=False)


class _ReloadableMMEncoderAttention(MMEncoderAttention):
    """Minimal stand-in to test reload lifecycle without encoder initialization."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.weight.weight_loader = default_weight_loader
        self.post_load_called = False

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self.post_load_called = True


class _ReloadableAttentionLayer(
    torch.nn.Module,
    AttentionLayerBase,
):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.weight.weight_loader = default_weight_loader
        self.post_load_called = False

    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):
        return None

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        self.post_load_called = True


def test_move_metatensors():
    tensor = torch.empty((1, 2, 3))
    meta_tensor = to_meta_tensor(tensor)
    materialized_tensor = materialize_meta_tensor(meta_tensor)

    assert meta_tensor.device.type == "meta"
    assert tensor.device == materialized_tensor.device

    assert tensor.dtype == meta_tensor.dtype == materialized_tensor.dtype
    assert tensor.shape == meta_tensor.shape == materialized_tensor.shape
    assert tensor.__class__ == meta_tensor.__class__ == materialized_tensor.__class__
    assert tensor.__dict__ == meta_tensor.__dict__ == materialized_tensor.__dict__


@pytest.mark.parametrize(
    "layer_cls",
    [_ReloadableMMEncoderAttention, _ReloadableAttentionLayer],
)
def test_attention_reload_defers_post_load(default_vllm_config, layer_cls):
    default_vllm_config.model_config = ModelConfig()
    layer = layer_cls()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)

    assert not layer.post_load_called

    finalize_layerwise_reload(model, default_vllm_config.model_config)

    assert layer.post_load_called
    assert torch.equal(layer.weight, loaded_weight)


@pytest.mark.parametrize(
    "layer_cls",
    [_ReloadableMMEncoderAttention, _ReloadableAttentionLayer],
)
def test_attention_first_load_processes_weights(default_vllm_config, layer_cls):
    default_vllm_config.model_config = ModelConfig()
    layer = layer_cls()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)

    initialize_online_processing(layer)
    layer.weight.weight_loader(layer.weight, loaded_weight)

    finalize_layerwise_reload(model, default_vllm_config.model_config)

    assert layer.post_load_called
    assert torch.equal(layer.weight, loaded_weight)


def test_reload_lifecycle():
    layer = torch.nn.Linear(2, 3)
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )

    restore_layer_on_meta(layer, info)
    for name, tensor in get_layer_tensors(layer).items():
        meta_tensor = getattr(layer, name)
        assert tensor.dtype == meta_tensor.dtype
        assert tensor.shape == meta_tensor.shape
        assert tensor.__class__ == meta_tensor.__class__
        assert tensor.__dict__ == meta_tensor.__dict__

    materialize_layer(layer, info)
    for name, tensor in get_layer_tensors(layer).items():
        materialized_tensor = getattr(layer, name)
        assert tensor.dtype == materialized_tensor.dtype
        assert tensor.shape == materialized_tensor.shape
        assert tensor.__class__ == materialized_tensor.__class__
        assert tensor.__dict__ == materialized_tensor.__dict__


def test_restore_layer_replaces_postprocessed_tensor_attribute():
    layer = torch.nn.Linear(2, 3, bias=False)
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )
    del layer.weight
    layer.weight = torch.empty(3, 2)

    restore_layer_on_meta(layer, info)

    assert isinstance(layer.weight, torch.nn.Parameter)
    assert layer.weight.is_meta


def test_materialize_layer_preserves_non_meta_tensors():
    """Ensure that materialize_layer does not overwrite non meta tensors."""
    layer = torch.nn.Linear(2, 3, bias=True)

    # Create a non meta bias tensor and meta weight, which can happen with FP8
    bias_values = torch.ones(3)
    layer.bias.data.copy_(bias_values)
    layer.weight = torch.nn.Parameter(layer.weight.data.to("meta"))

    assert layer.weight.is_meta
    assert not layer.bias.is_meta

    # materialize the layer weights after the bias is initialized
    info = LayerReloadingInfo(
        restore_metadata=({}, {}),
        restore_device=torch.device("cpu"),
    )
    materialize_layer(layer, info)

    # Ensure the weight materialized off meta
    assert not layer.weight.is_meta
    assert layer.weight.device.type == "cpu"

    # Ensure that the bias is (still) not meta and values are unchanged
    assert not layer.bias.is_meta
    assert torch.equal(layer.bias.data, bias_values)


_MARLIN_SIZE_K, _MARLIN_SIZE_N, _MARLIN_GROUP_SIZE = 128, 64, 64


def _stub_marlin_ops(monkeypatch):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(
        ops,
        "gptq_marlin_repack",
        lambda w, size_k, size_n, num_bits, is_a_8bit=False: torch.zeros(
            size_k // 16, size_n * 2, dtype=torch.int32
        ),
    )


def _make_marlin_kernel():
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        MarlinLinearKernel,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    kernel = object.__new__(MarlinLinearKernel)
    kernel.config = MPLinearLayerConfig(
        full_weight_shape=(_MARLIN_SIZE_K, _MARLIN_SIZE_N),
        partition_weight_shape=(_MARLIN_SIZE_K, _MARLIN_SIZE_N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.float16,
        group_size=_MARLIN_GROUP_SIZE,
        zero_points=False,
    )
    kernel.w_q_name = "qweight"
    kernel.w_s_name = "scales"
    kernel.w_zp_name = None
    return kernel


def _load_marlin_checkpoint_format_weights(layer):
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    layer.qweight = PackedvLLMParameter(
        data=torch.zeros(_MARLIN_SIZE_K // 8, _MARLIN_SIZE_N, dtype=torch.int32),
        input_dim=0,
        output_dim=1,
        packed_dim=0,
        packed_factor=8,
        weight_loader=default_weight_loader,
    )
    layer.scales = GroupQuantScaleParameter(
        data=torch.ones(
            _MARLIN_SIZE_K // _MARLIN_GROUP_SIZE, _MARLIN_SIZE_N, dtype=torch.float16
        ),
        input_dim=0,
        output_dim=1,
        weight_loader=default_weight_loader,
    )


def test_marlin_post_load_does_not_own_workspace(monkeypatch, dist_init):
    """Weight reload must not create layer-owned Marlin lock storage."""
    _stub_marlin_ops(monkeypatch)
    kernel = _make_marlin_kernel()

    layer = torch.nn.Module()
    _load_marlin_checkpoint_format_weights(layer)
    kernel.process_weights_after_loading(layer)

    assert not hasattr(kernel, "workspace")

    _load_marlin_checkpoint_format_weights(layer)
    kernel.process_weights_after_loading(layer)

    assert not hasattr(kernel, "workspace")


@pytest.mark.parametrize("variant", ["fp8", "mxfp8", "nvfp4"])
def test_marlin_prepare_layer_does_not_own_workspace(monkeypatch, variant):
    """Weight preparation must not attach runtime workspace to model layers."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import (
        marlin_utils,
        marlin_utils_fp4,
        marlin_utils_fp8,
    )

    size_k, size_n = 128, 64

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    monkeypatch.setattr(
        ops,
        "gptq_marlin_repack",
        lambda b_q_weight, size_k, size_n, num_bits, is_a_8bit=False: torch.zeros(
            size_k // 16, size_n * 2, dtype=torch.int32
        ),
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = size_n
    layer.input_size_per_partition = size_k
    layer.orig_dtype = torch.float16
    layer.params_dtype = torch.float16

    if variant == "fp8":
        prepare = marlin_utils_fp8.prepare_fp8_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_k, size_n, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )
    elif variant == "mxfp8":
        prepare = marlin_utils_fp8.prepare_mxfp8_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_n, size_k, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.full((size_n, size_k // 32), 127, dtype=torch.uint8),
                requires_grad=False,
            )
    else:
        prepare = marlin_utils_fp4.prepare_fp4_layer_for_marlin

        def load_checkpoint_format_weights():
            layer.weight = torch.nn.Parameter(
                torch.zeros(size_n, size_k // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            layer.weight_scale = torch.nn.Parameter(
                torch.ones(size_n, size_k // 16, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.weight_global_scale = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )

    load_checkpoint_format_weights()
    prepare(layer)
    assert not hasattr(layer, "workspace")

    # Reload: fresh checkpoint-format tensors, prepare runs again
    load_checkpoint_format_weights()
    prepare(layer)

    assert not hasattr(layer, "workspace")


def test_marlin_workspace_uses_persistent_workspace_manager(monkeypatch):
    """Calls reuse initialized locks; independent streams get separate storage."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.utils import torch_utils
    from vllm.v1.worker import workspace as workspace_module

    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    device = torch.device("cpu")
    manager = workspace_module.WorkspaceManager(device)
    monkeypatch.setattr(workspace_module, "_manager", manager)
    stream = "main"
    monkeypatch.setattr(torch_utils, "current_stream", lambda: stream)

    workspace = marlin_utils.get_marlin_workspace(device)
    assert workspace.shape == (4 * marlin_utils.MARLIN_MAX_BLOCKS_PER_SM,)
    assert workspace.dtype == torch.int32
    assert torch.count_nonzero(workspace) == 0

    workspace.fill_(1)
    assert marlin_utils.get_marlin_workspace(device) is workspace
    assert torch.all(workspace == 1)

    stream = "aux"
    aux_workspace = marlin_utils.get_marlin_workspace(device)
    assert aux_workspace.data_ptr() != workspace.data_ptr()
    assert torch.count_nonzero(aux_workspace) == 0

    manager.lock()
    assert marlin_utils.get_marlin_workspace(device) is aux_workspace


def test_marlin_workspace_without_manager(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.v1.worker import workspace as workspace_module

    monkeypatch.setattr(workspace_module, "_manager", None)
    monkeypatch.setattr(marlin_utils, "num_compute_units", lambda _: 4)
    first = marlin_utils.get_marlin_workspace(torch.device("cpu"))
    second = marlin_utils.get_marlin_workspace(torch.device("cpu"))
    assert first.data_ptr() != second.data_ptr()
    assert first.shape == (4 * marlin_utils.MARLIN_MAX_BLOCKS_PER_SM,)
    assert torch.count_nonzero(first) == 0


def test_model_cleanup(dist_init, default_vllm_config):
    layer = QKVParallelLinear(2, 3, 4)
    assert layer.weight.weight_loader.__self__ is layer
    info = LayerReloadingInfo(
        restore_metadata=capture_layer_to_meta(layer),
        restore_device=torch.device("cpu"),
    )

    mock_info_dict: WeakKeyDictionary[torch.nn.Module, LayerReloadingInfo] = (
        WeakKeyDictionary()
    )
    mock_info_dict[layer] = info
    layer_ref = ref(layer)

    del layer
    gc.collect()

    assert layer_ref() is None
    assert len(mock_info_dict) == 0


@pytest.mark.parametrize("is_gated", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("padded", [False, True])
def test_padded_moe_reload_releases_each_layer(
    monkeypatch, is_gated, has_bias, tp_rank, padded
):
    """Checkpoint-sized copies finish each layer without global finalization."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    hidden, intermediate, experts = 4, 3, 2
    stored_hidden, stored_intermediate = (8, 8) if padded else (hidden, intermediate)
    config = SimpleNamespace(
        hidden_dim_unpadded=hidden,
        intermediate_size_per_partition_unpadded=intermediate,
        is_act_and_mul=is_gated,
        has_bias=has_bias,
        tp_rank=tp_rank,
        tp_shard_with_padding=False,
        moe_parallel_config=SimpleNamespace(tp_size=2),
    )
    model = torch.nn.ModuleList()
    processed: list[torch.nn.Module] = []
    for _ in range(2):
        method = object.__new__(UnquantizedFusedMoEMethod)
        torch.nn.Module.__init__(method)
        method.moe = config
        # The regression concerns streaming reload, not kernel conversion.
        monkeypatch.setattr(method, "process_weights_after_loading", processed.append)
        layer = object.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.moe_config = config
        layer.quant_config = None
        layer.quant_method = method
        layer.expert_map_manager = SimpleNamespace(map_global_to_local=lambda i: i)
        layer._loaded_expert_biases = set()
        method.create_weights(
            layer,
            experts,
            stored_hidden,
            stored_intermediate,
            torch.float32,
            weight_loader=layer.weight_loader,
        )
        model.append(layer)

    record_metadata_for_reloading(model)
    original_params = [dict(layer.named_parameters()) for layer in model]
    shards = ["w1", "w3", "w2"] if is_gated else ["w1", "w2"]
    for cycle in range(2):
        initialize_layerwise_reload(model)
        for layer_index, layer in enumerate(model):
            info = reload_layerwise.get_layerwise_info(layer)
            inputs = []
            expected = {
                name: torch.full_like(p, float("nan"))
                for name, p in original_params[layer_index].items()
            }
            params = dict(layer.named_parameters())
            calls = [
                (e, s, b)
                for e in range(experts)
                for s in shards
                for b in ([False, True] if has_bias else [False])
            ]
            for call_index, (expert, shard, bias) in enumerate(calls):
                name = ("w2" if shard == "w2" else "w13") + (
                    "_bias" if bias else "_weight"
                )
                shape = (
                    ((hidden,) if bias else (hidden, 2 * intermediate))
                    if shard == "w2"
                    else ((2 * intermediate,) if bias else (2 * intermediate, hidden))
                )
                weight = torch.arange(
                    torch.Size(shape).numel(), dtype=torch.float32
                ).reshape(shape)
                weight = weight + 100 * (1 + call_index + cycle)
                inputs.append(ref(weight))
                # Direct checkpoint loading is the reference for deferred reload.
                layer.weight_loader(expected[name], weight, name, shard, expert)
                param = params[name]
                param.weight_loader(param, weight, name, shard, expert)
                del weight
                if call_index != len(calls) - 1:
                    assert info.can_load(), (
                        "Layer processed before its final checkpoint shard"
                    )

            assert not info.can_load(), (
                "Padding must not defer the layer until finalization"
            )
            assert not info.loaded_weights
            assert len(processed) == cycle * len(model) + layer_index + 1
            assert all(source() is None for source in inputs)
            for name, original in original_params[layer_index].items():
                assert getattr(layer, name) is original
                # Kernel-specific tests cover padding, which is not checkpoint data.
                mask = torch.isfinite(expected[name])
                assert torch.equal(original[mask], expected[name][mask])


def test_get_numel_loaded():
    param = torch.empty(10, device="meta")
    loaded_weight = torch.empty(10)

    def complex_weight_loader(param, loaded_weight):
        param[:3] = loaded_weight[:3]
        param[5:8] = loaded_weight[5:8]
        return "value"

    args = inspect.signature(complex_weight_loader).bind(param, loaded_weight)
    num_loaded, ret = get_numel_loaded(complex_weight_loader, args)
    assert num_loaded == 6
    assert ret == "value"


def test_get_numel_loaded_caps_at_param_size():
    # composed_weight_loader copies into the param twice (the load and the
    # in-place post-load transform), but only param.numel() distinct elements
    # are loaded. get_numel_loaded must not double-count, otherwise a layer's
    # loaded-element total can be reached early and trailing params get dropped.
    param = torch.empty(10)
    loaded_weight = torch.ones(10)
    loader = composed_weight_loader(default_weight_loader, lambda x: x + 1)

    args = inspect.signature(loader).bind(param, loaded_weight)
    num_loaded, _ = get_numel_loaded(loader, args)
    assert num_loaded == 10


def test_layerwise_loading_warning_only_checks_new_layers(monkeypatch):
    layers = [torch.nn.Linear(16, 1, bias=False) for _ in range(2)]

    def partial_weight_loader(param, loaded_weight):
        param.view(-1)[: loaded_weight.numel()].copy_(loaded_weight)

    for layer in layers:
        layer.weight.requires_grad_(False)
        layer.weight.weight_loader = partial_weight_loader
        reload_layerwise.initialize_online_processing(layer)

    monkeypatch.setattr(reload_layerwise, "has_device_tensors", lambda _: True)
    get_info_size = Mock(return_value=0)
    warning_once = Mock()
    monkeypatch.setattr(reload_layerwise, "get_info_size", get_info_size)
    monkeypatch.setattr(reload_layerwise.logger, "warning_once", warning_once)

    reload_layerwise.LOADING_LAYERS.clear()
    try:
        for layer in layers:
            for _ in range(3):
                layer.weight.weight_loader(layer.weight, torch.ones(1))
    finally:
        reload_layerwise.LOADING_LAYERS.clear()

    assert get_info_size.call_count == 2
    warning_once.assert_called_once()


class _ComposedLoaderLayer(torch.nn.Module):
    """Mimics a Mamba2 mixer's equal-numel direct params (A, D, dt_bias).

    ``A`` uses ``composed_weight_loader`` (an extra in-place transform copy),
    matching ``MambaMixer2`` where ``A`` is loaded as ``-exp(A_log)``.
    """

    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(4, dtype=torch.float32))
        self.D = torch.nn.Parameter(torch.ones(4))
        self.dt_bias = torch.nn.Parameter(torch.ones(4))
        self.A.weight_loader = composed_weight_loader(
            default_weight_loader, lambda x: -torch.exp(x.float())
        )
        self.D.weight_loader = default_weight_loader
        self.dt_bias.weight_loader = default_weight_loader


def test_layerwise_reload_composed_loader_does_not_drop_params(monkeypatch):
    # Regression test: a composed_weight_loader param (A) used to double-count
    # its elements, finalizing the layer before the trailing param (D) was
    # loaded and leaving it as uninitialized materialized memory.
    layer = _ComposedLoaderLayer()
    model = torch.nn.Sequential(layer)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(float("nan"))
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    loaded = {
        "A": torch.full((4,), 0.5),
        "dt_bias": torch.full((4,), 3.0),
        "D": torch.full((4,), 7.0),
    }

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    # Mimic real load_weights: resolve params once, then load in checkpoint
    # order with D last (the param that was dropped).
    params = dict(layer.named_parameters())
    for name in ("A", "dt_bias", "D"):
        param = params[name]
        param.weight_loader(param, loaded[name])
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.A, -torch.exp(loaded["A"]))
    assert torch.equal(layer.dt_bias, loaded["dt_bias"])
    assert torch.equal(layer.D, loaded["D"])


class _RecordingQuantMethod(QuantizeMethodBase):
    """Records the layer's bias at the moment processing runs."""

    uses_meta_device = True

    def __init__(self):
        self.bias_at_process = None

    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        pass

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.bias_at_process = layer.bias.detach().clone()


class _LateBiasLayer(torch.nn.Module):
    """Mimics an online-quantized linear: `weight` is created on meta by
    `create_weights()`, which wraps the loaders, and the linear base registers
    `bias` afterwards."""

    def __init__(self, quant_method):
        super().__init__()
        self.quant_method = quant_method
        weight = torch.nn.Parameter(torch.empty(4, 2, device="meta"))
        weight.weight_loader = default_weight_loader
        self.register_parameter("weight", weight)
        initialize_online_processing(self)
        bias = torch.nn.Parameter(torch.zeros(4))
        bias.weight_loader = default_weight_loader
        self.register_parameter("bias", bias)


def test_online_processing_waits_for_late_registered_bias():
    # Regression test: `bias` is skipped by the meta device paths, but it is
    # still loaded by a weight loader. Excluding it from the processing trigger
    # finalized the layer one load early, so the trailing bias was written into
    # an already-processed layer (e.g. over FP8 Marlin's permuted bias).
    quant_method = _RecordingQuantMethod()
    layer = _LateBiasLayer(quant_method)
    loaded_bias = torch.full((4,), 3.0)

    layer.weight.weight_loader(layer.weight, torch.full((4, 2), 2.0))
    assert quant_method.bias_at_process is None

    layer.bias.weight_loader(layer.bias, loaded_bias)
    assert quant_method.bias_at_process is not None
    assert torch.equal(quant_method.bias_at_process, loaded_bias)


def test_layerwise_reload_skips_non_persistent_parameter_alias_buffers(monkeypatch):
    layer = _AliasedBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.weight, loaded_weight)
    assert layer.weight_view.untyped_storage().data_ptr() == (
        layer.weight.untyped_storage().data_ptr()
    )
    assert "weight_view" in layer._non_persistent_buffers_set
    assert "0.weight_view" not in model.state_dict()


def test_capture_layer_to_meta_skips_uninitialized_parameter_storage_ptrs():
    layer = _AliasedBufferWithUninitializedChildLayer()

    _, buffers = capture_layer_to_meta(layer)

    assert "weight_view" not in buffers


def test_layerwise_reload_skips_child_parameter_alias_buffers(monkeypatch):
    layer = _ParentAliasedChildBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_conv = torch.full_like(layer.conv1d.weight, 7.0)
    loaded_scale = torch.full_like(layer.scale, 3.0)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.conv1d.weight.weight_loader(layer.conv1d.weight, loaded_conv)
    layer.scale.weight_loader(layer.scale, loaded_scale)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.conv1d.weight, loaded_conv)
    assert torch.equal(layer.conv_weights, loaded_conv.view(-1))
    assert layer.conv_weights.untyped_storage().data_ptr() == (
        layer.conv1d.weight.untyped_storage().data_ptr()
    )
    assert "conv_weights" in layer._non_persistent_buffers_set
    assert "0.conv_weights" not in model.state_dict()


def test_layerwise_reload_restores_alias_buffer_on_zero_size_layer(monkeypatch):
    layer = _ChildAliasOnlyBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_conv = torch.full_like(layer.conv1d.weight, 7.0)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.conv1d.weight.weight_loader(layer.conv1d.weight, loaded_conv)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.conv_weights, loaded_conv.view(-1))
    assert layer.conv_weights.untyped_storage().data_ptr() == (
        layer.conv1d.weight.untyped_storage().data_ptr()
    )
    assert "conv_weights" in layer._non_persistent_buffers_set
    assert "0.conv_weights" not in model.state_dict()


def test_layerwise_reload_preserves_unloaded_non_persistent_buffers(monkeypatch):
    layer = _NonPersistentBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)
    original_scale = layer.scale.clone()

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.weight, loaded_weight)
    assert torch.equal(layer.scale, original_scale)
    assert "scale" in layer._non_persistent_buffers_set
    assert "0.scale" not in model.state_dict()


def test_layerwise_reload_updates_loaded_non_persistent_buffers(monkeypatch):
    layer = _NonPersistentBufferLayer()
    model = torch.nn.Sequential(layer)
    loaded_weight = torch.full_like(layer.weight, 7.0)
    loaded_scale = torch.full_like(layer.scale, 0.5)

    def materialize_with_sentinel(meta_tensor):
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
        )
        tensor.fill_(-123.0)
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor

    monkeypatch.setattr(
        reload_meta, "materialize_meta_tensor", materialize_with_sentinel
    )

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.weight.weight_loader(layer.weight, loaded_weight)
    layer.scale.weight_loader(layer.scale, loaded_scale)
    finalize_layerwise_reload(model, model_config=None)

    assert torch.equal(layer.weight, loaded_weight)
    assert torch.equal(layer.scale, loaded_scale)
    assert "scale" in layer._non_persistent_buffers_set
    assert "0.scale" not in model.state_dict()


@pytest.fixture
def hpc_rope_norm(monkeypatch, default_vllm_config):
    """Import HpcRopeNorm with the external ``hpc`` package stubbed out."""
    if "hpc" not in sys.modules:
        stub = types.ModuleType("hpc")
        stub.__spec__ = importlib.machinery.ModuleSpec("hpc", loader=None)
        stub.QuantType = types.SimpleNamespace(  # type: ignore[attr-defined]
            QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR=types.SimpleNamespace(value=0)
        )
        monkeypatch.setitem(sys.modules, "hpc", stub)
    from vllm.model_executor.layers.hpc import rope_norm

    monkeypatch.setattr(rope_norm, "_hpc_rope_norm_instances", {})
    return rope_norm


def test_hpc_rope_norm_kernel_sees_refit_norm_weights(monkeypatch, hpc_rope_norm):
    """The fused HPC kernel is handed the live QK-norm weights after a refit.

    Drives the production ``_forward_impl`` with a recording ``hpc`` stub. The
    Q/K norm weights it receives must be the model's own float32 parameters,
    so a layerwise reload that rewrites them in place (same storage) is what
    the kernel sees. Previously the kernel read separate mirrors that no
    reload path refreshed.
    """
    from vllm.model_executor.layers.layernorm import RMSNorm

    head_dim, num_heads, num_kv_heads, block_size = 128, 8, 1, 4
    layer = torch.nn.Module()
    layer.q_norm = RMSNorm(head_dim, 1e-6, dtype=torch.float32)
    layer.k_norm = RMSNorm(head_dim, 1e-6, dtype=torch.float32)
    layer.hpc_rope_norm = hpc_rope_norm.HpcRopeNorm(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cos_sin_cache=torch.ones(16, head_dim),
        use_qk_norm=True,
        fallback_qnorm=layer.q_norm,
        fallback_knorm=layer.k_norm,
        kv_cache_dtype="auto",
        layer_name="hpc_test_layer",
    )
    model = torch.nn.Sequential(layer)
    rnorm = layer.hpc_rope_norm

    calls: list[dict] = []

    def record(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(sys.modules["hpc"], "rope_norm_store_kv", record, raising=False)

    def kernel_norm_weights():
        q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
        qkv = torch.zeros(1, q_size + 2 * kv_size, dtype=torch.bfloat16)
        kv_cache = torch.zeros(
            2, num_kv_heads, block_size, 2 * head_dim, dtype=torch.bfloat16
        )
        attn_layer = types.SimpleNamespace(
            _k_scale=torch.ones(1), _v_scale=torch.ones(1)
        )
        attn_metadata = types.SimpleNamespace(
            num_actual_tokens=1,
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=0,
            num_prefill_tokens=0,
            max_query_len=1,
            decode_query_len=1,
            qo_indptr=None,
            qo_indptr_decode=None,
            slot_mapping=torch.tensor([4]),
            seq_lens=torch.tensor([1]),
            block_table_tensor=torch.tensor([[1]]),
            hpc_kv_written=False,
        )
        output = torch.zeros(1, q_size, dtype=torch.bfloat16)
        rnorm._forward_impl(qkv, kv_cache, attn_metadata, attn_layer, output)
        return calls[-1]["q_norm_weight"], calls[-1]["k_norm_weight"]

    def loaded(value):
        return torch.full((head_dim,), value, dtype=torch.bfloat16)

    default_weight_loader(layer.q_norm.weight, loaded(0.5))
    default_weight_loader(layer.k_norm.weight, loaded(0.25))
    q, k = kernel_norm_weights()
    assert torch.equal(q, loaded(0.5).float())
    assert torch.equal(k, loaded(0.25).float())
    q_ptr, k_ptr = q.data_ptr(), k.data_ptr()

    record_metadata_for_reloading(model)
    initialize_layerwise_reload(model)
    layer.q_norm.weight.weight_loader(layer.q_norm.weight, loaded(2.0))
    layer.k_norm.weight.weight_loader(layer.k_norm.weight, loaded(3.0))
    finalize_layerwise_reload(model, model_config=None)

    q, k = kernel_norm_weights()
    assert q.dtype == k.dtype == torch.float32
    assert torch.equal(q, loaded(2.0).float())
    assert torch.equal(k, loaded(3.0).float())
    assert (q.data_ptr(), k.data_ptr()) == (q_ptr, k_ptr)
    assert not hasattr(rnorm, "qnorm_weight")


@pytest.mark.parametrize(
    "tp_size", [pytest.param(1), pytest.param(2, marks=[pytest.mark.slow_test])]
)
@pytest.mark.parametrize(
    "base_model,mul_model,add_model",
    [
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/Qwen3-0.6B-FP8_BLOCK",
            "inference-optimization/Qwen3-0.6B-debug-multiply-FP8_BLOCK",
            "inference-optimization/Qwen3-0.6B-debug-add-FP8_BLOCK",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/Qwen3-0.6B-W4A16-G128",
            "inference-optimization/Qwen3-0.6B-debug-multiply-W4A16-G128",
            "inference-optimization/Qwen3-0.6B-debug-add-W4A16-G128",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC",
            "inference-optimization/DeepSeek-V3-debug-multiply-FP8_DYNAMIC",
            "inference-optimization/DeepSeek-V3-debug-add-FP8_DYNAMIC",
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty-NVFP4A16",
            "inference-optimization/DeepSeek-V3-debug-multiply-NVFP4A16",
            "inference-optimization/DeepSeek-V3-debug-add-NVFP4A16",
            marks=[pytest.mark.slow_test],
        ),
    ],
)
def test_reload_weights(base_model, mul_model, add_model, tp_size, vllm_runner):
    if current_platform.device_count() < tp_size:
        pytest.skip(reason="Not enough CUDA devices")

    if "FP8" in base_model and _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    with vllm_runner(
        model_name=base_model,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=(tp_size > 1 and "DeepSeek" in base_model),
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp


def test_kv_scale_reload(vllm_runner):
    """Test reloading a checkpoint that contains k_scale/v_scale weights."""
    if _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    model = "nm-testing/Llama-3.2-1B-Instruct-FP8-KV"

    # Load dummy weights, then reload real checkpoint
    with vllm_runner(
        model_name=model,
        load_format="dummy",
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc(
            "update_config",
            kwargs={"overrides": {"load_config": {"load_format": "auto"}}},
        )
        llm.collective_rpc("reload_weights", kwargs={"weights_path": model})
        reloaded_perp = llm.generate_prompt_perplexity(
            ["The capital of France is the city of Paris"],
            mask=["The capital of France is"],
        )[0]

    assert reloaded_perp < 10


@pytest.mark.parametrize(
    "tp_size", [pytest.param(1), pytest.param(2, marks=[pytest.mark.slow_test])]
)
@pytest.mark.parametrize(
    "base_model,mul_model,add_model,quantization",
    [
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            "fp8",
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "fp8",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "Qwen/Qwen3-0.6B",
            "inference-optimization/Qwen3-0.6B-debug-multiply",
            "inference-optimization/Qwen3-0.6B-debug-add",
            "mxfp8",
            marks=[pytest.mark.slow_test],
        ),
        pytest.param(
            "inference-optimization/DeepSeek-V3-debug-empty",
            "inference-optimization/DeepSeek-V3-debug-multiply",
            "inference-optimization/DeepSeek-V3-debug-add",
            "mxfp8",
            marks=[
                pytest.mark.slow_test,
                pytest.mark.xfail(reason="mxfp4 & mla is not supported yet"),
            ],
        ),
    ],
)
def test_online_quantize_reload(
    base_model, mul_model, add_model, quantization, tp_size, vllm_runner
):
    if current_platform.device_count() < tp_size:
        pytest.skip(reason="Not enough GPU devices")

    if quantization == "fp8" and _fp8_reload_unsupported():
        pytest.skip(reason="Requires FP8 support")

    with vllm_runner(
        model_name=base_model,
        quantization=quantization,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=(tp_size > 1 and "DeepSeek" in base_model),
        enable_prefix_caching=False,
        max_model_len=16,
        max_num_seqs=1,
    ) as llm:
        llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert mul_perp < add_perp

        llm.collective_rpc("reload_weights", kwargs={"weights_path": add_model})
        mul_perp = llm.generate_prompt_perplexity(["3 4 = 12"], mask=["3 4 ="])[0]
        add_perp = llm.generate_prompt_perplexity(["3 4 = 7"], mask=["3 4 ="])[0]
        assert add_perp < mul_perp
