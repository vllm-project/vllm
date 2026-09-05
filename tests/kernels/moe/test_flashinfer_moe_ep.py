# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config.kernel import (
    FLASHINFER_MOE_EP_CUTEDSL,
    FLASHINFER_MOE_EP_DEEP_GEMM,
)
from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep as fi_ep
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization import modelopt, mxfp4
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4a4_mxfp4 as ct_mxfp4,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4a4_nvfp4 as ct_nvfp4,
)
from vllm.model_executor.layers.quantization.inc.schemes import inc_mxfp4_moe


@pytest.fixture(autouse=True)
def clear_flashinfer_moe_ep_registry():
    fi_ep._FLASHINFER_MOE_EP_ADAPTERS.clear()
    yield
    fi_ep.destroy_flashinfer_moe_ep()


class _FakeMegaLayer:
    supports_output_view = True

    def __init__(self) -> None:
        self.forward_tensors: list[SimpleNamespace] = []
        self.owned_forward_tensors: list[SimpleNamespace] = []
        self.warmup_tensors: list[SimpleNamespace] = []
        self.warmup_inference_modes: list[bool] = []
        self.destroy_calls = 0

    def forward(self, tensors, *, return_workspace_view: bool):
        assert return_workspace_view
        self.forward_tensors.append(tensors)
        return torch.empty_like(tensors.hidden_states)

    def __call__(self, tensors):
        self.owned_forward_tensors.append(tensors)
        return torch.empty_like(tensors.hidden_states)

    def warmup(self, tensors) -> None:
        self.warmup_tensors.append(tensors)
        self.warmup_inference_modes.append(torch.is_inference_mode_enabled())

    def destroy(self) -> None:
        self.destroy_calls += 1


def _namespace(**kwargs):
    return SimpleNamespace(**kwargs)


def _moe(
    backend: str = FLASHINFER_MOE_EP_CUTEDSL,
    routing_method: RoutingMethodType = RoutingMethodType.Default,
) -> SimpleNamespace:
    return SimpleNamespace(
        moe_backend=backend,
        num_experts=8,
        max_num_tokens=32,
        hidden_dim=8,
        intermediate_size=4,
        experts_per_token=2,
        swiglu_limit=7.0,
        routing_method=routing_method,
    )


@pytest.mark.parametrize(
    ("backend", "config_name"),
    (
        (FLASHINFER_MOE_EP_CUTEDSL, "cutedsl"),
        (FLASHINFER_MOE_EP_DEEP_GEMM, "deep_gemm"),
    ),
)
def test_adapter_builds_public_backend_and_owns_lifecycle(
    monkeypatch,
    backend: str,
    config_name: str,
):
    mega_layer = _FakeMegaLayer()
    constructed: dict[str, SimpleNamespace] = {}
    monkeypatch.setattr(
        fi_ep,
        "_expose_deep_gemm_to_flashinfer",
        lambda: None,
        raising=False,
    )

    def make_mega_layer(bootstrap, fleet, weights, backend_config):
        constructed.update(
            bootstrap=bootstrap,
            fleet=fleet,
            weights=weights,
            backend=backend_config,
        )
        return mega_layer

    api = SimpleNamespace(
        BootstrapConfig=_namespace,
        DeepGemmMegaMoeConfig=lambda **kwargs: _namespace(
            config_name="deep_gemm", **kwargs
        ),
        FleetParams=_namespace,
        MegaConfig=_namespace,
        MoEEpMegaLayer=make_mega_layer,
        MoEEpTensors=_namespace,
        MoEWeightPack=_namespace,
        Nvfp4CutedslMegaMoeConfig=lambda **kwargs: _namespace(
            config_name="cutedsl", **kwargs
        ),
    )
    process_group = object()
    monkeypatch.setattr(fi_ep, "_load_flashinfer_moe_ep_api", lambda: api)
    monkeypatch.setattr(
        fi_ep,
        "get_ep_group",
        lambda: SimpleNamespace(
            world_size=4,
            rank_in_group=2,
            device_group=process_group,
        ),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

    weights = fi_ep.FlashInferMoeEpWeights(
        w13=torch.empty(2, 8, 4, dtype=torch.uint8),
        w2=torch.empty(2, 8, 2, dtype=torch.uint8),
        w13_scale=torch.empty(2, 8, 1),
        w2_scale=torch.empty(2, 8, 1),
    )
    fc1_alpha = torch.tensor([0.5, 0.25])
    fc2_alpha = torch.tensor([0.125, 0.0625])
    fc1_norm_const = torch.tensor([143.0, 143.0])
    epilogue = fi_ep.FlashInferMoeEpEpilogue(
        input_norm_const=135.0,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
    )
    adapter = fi_ep.FlashInferMoeEp(
        _moe(backend, RoutingMethodType.DeepseekV4),
        weights,
        epilogue,
        apply_topk_in_fc1=True,
    )

    kernel_config = constructed["backend"].megakernel
    assert kernel_config.config_name == config_name
    assert constructed["bootstrap"].process_group is process_group
    if backend == FLASHINFER_MOE_EP_CUTEDSL:
        assert kernel_config.apply_topk_in_fc1 is True
        assert kernel_config.in_kernel_fc2_reduce is False
        assert kernel_config.combine_dtype == "bf16"
        assert kernel_config.input_norm_const == 135.0
        assert kernel_config.gate_up_clamp == 7.0
    else:
        assert kernel_config.activation_clamp == 7.0

    hidden_states = torch.empty(0, 8, dtype=torch.bfloat16)
    topk_ids = torch.empty(0, 2, dtype=torch.int32)
    topk_weights = torch.empty(0, 2, dtype=torch.float32)
    output = adapter(hidden_states, topk_ids, topk_weights)
    assert output.shape == hidden_states.shape
    call = mega_layer.forward_tensors[0]
    assert call.fc1_alpha is fc1_alpha
    assert call.fc2_alpha is fc2_alpha
    assert call.fc1_norm_const is fc1_norm_const

    adapter.warmup()
    warmup = mega_layer.warmup_tensors[0]
    assert mega_layer.warmup_inference_modes == [True]
    assert warmup.hidden_states.shape == (1, 8)
    torch.testing.assert_close(
        warmup.topk_ids,
        torch.tensor([[0, 4]], dtype=torch.int32),
    )

    adapter.destroy()
    adapter.destroy()
    assert mega_layer.destroy_calls == 1


def test_shared_adapter_exposes_vllm_deep_gemm_to_flashinfer(monkeypatch):
    deep_gemm = SimpleNamespace()
    monkeypatch.delitem(sys.modules, "deep_gemm", raising=False)
    monkeypatch.setattr(
        fi_ep,
        "import_deep_gemm",
        lambda: deep_gemm,
        raising=False,
    )

    fi_ep._expose_deep_gemm_to_flashinfer()

    assert sys.modules["deep_gemm"] is deep_gemm


def test_shared_adapter_rejects_unavailable_deep_gemm(monkeypatch):
    monkeypatch.delitem(sys.modules, "deep_gemm", raising=False)
    monkeypatch.setattr(fi_ep, "import_deep_gemm", lambda: None)

    with pytest.raises(RuntimeError, match="requires a usable DeepGEMM"):
        fi_ep._expose_deep_gemm_to_flashinfer()


def test_non_dsv4_adapter_returns_owned_output():
    mega_layer = _FakeMegaLayer()
    adapter = object.__new__(fi_ep.FlashInferMoeEp)
    adapter._mega_layer = mega_layer
    adapter._max_num_tokens = 32
    adapter._epilogue = fi_ep.FlashInferMoeEpEpilogue()
    adapter._moe_ep_tensors_cls = _namespace
    adapter._return_workspace_view = False

    hidden_states = torch.empty(1, 8, dtype=torch.bfloat16)
    adapter(
        hidden_states,
        torch.zeros(1, 2, dtype=torch.int32),
        torch.full((1, 2), 0.5),
    )

    assert len(mega_layer.owned_forward_tensors) == 1
    assert not mega_layer.forward_tensors


@pytest.mark.parametrize(
    ("routing_method", "router_weight_on_input", "expected"),
    (
        (RoutingMethodType.Default, False, False),
        (RoutingMethodType.Default, True, True),
        (RoutingMethodType.DeepseekV4, False, True),
    ),
)
def test_shared_adapter_preserves_router_weight_policy(
    monkeypatch,
    routing_method: RoutingMethodType,
    router_weight_on_input: bool,
    expected: bool,
):
    captured: dict[str, Any] = {}

    def make_adapter(moe, weights, epilogue, *, apply_topk_in_fc1):
        captured["apply_topk_in_fc1"] = apply_topk_in_fc1
        return SimpleNamespace()

    monkeypatch.setattr(fi_ep, "FlashInferMoeEp", make_adapter)
    moe = _moe(routing_method=routing_method)
    layer = SimpleNamespace(
        moe_config=moe,
        apply_router_weight_on_input=router_weight_on_input,
        expert_map_manager=SimpleNamespace(placement_strategy="linear"),
    )

    fi_ep.make_flashinfer_moe_ep(
        moe,
        layer,
        fi_ep.FlashInferMoeEpWeights(torch.empty(1), torch.empty(1)),
    )

    assert captured["apply_topk_in_fc1"] is expected


def test_modelopt_nvfp4_data_folds_exact_gate_up_ratio():
    w13 = torch.empty(2, 4, 2, dtype=torch.uint8)
    w2 = torch.empty(2, 2, 1, dtype=torch.uint8)
    w13_scale = torch.ones(2, 4, 1, dtype=torch.float8_e4m3fn)
    w2_scale = torch.ones(2, 2, 1, dtype=torch.float8_e4m3fn)
    w13_scale_2 = torch.tensor([[2.0, 4.0], [4.0, 4.0]])
    w2_scale_2 = torch.tensor([8.0, 16.0])

    weights, epilogue = fi_ep.modelopt_nvfp4_moe_ep_data(
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_scale_2,
        w2_scale_2,
        intermediate_size=2,
    )

    torch.testing.assert_close(
        weights.w13_scale.float()[:, 2:, :],
        torch.tensor([[[2.0], [2.0]], [[1.0], [1.0]]]),
    )
    torch.testing.assert_close(epilogue.fc1_alpha, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(epilogue.fc2_alpha, torch.tensor([8.0, 16.0]))
    assert weights.w13 is w13
    assert weights.w2 is w2


def test_modelopt_nvfp4_data_rejects_lossy_gate_up_ratio():
    with pytest.raises(ValueError, match="not exactly representable"):
        fi_ep.modelopt_nvfp4_moe_ep_data(
            torch.empty(1, 4, 2, dtype=torch.uint8),
            torch.empty(1, 2, 1, dtype=torch.uint8),
            torch.ones(1, 4, 1, dtype=torch.float8_e4m3fn),
            torch.ones(1, 2, 1, dtype=torch.float8_e4m3fn),
            torch.tensor([[1.0, 1.1]]),
            torch.ones(1),
            intermediate_size=2,
        )


def test_mxfp4_weights_pass_through_for_deep_gemm_and_dequantize_for_cutedsl():
    packed = torch.arange(16, dtype=torch.uint8).view(1, 1, 16)
    scale = torch.tensor([[[127]]], dtype=torch.uint8)

    deep_gemm = fi_ep.mxfp4_moe_ep_weights(
        FLASHINFER_MOE_EP_DEEP_GEMM,
        packed,
        packed,
        scale,
        scale,
    )
    assert deep_gemm.w13 is packed
    assert deep_gemm.w13_scale is scale

    cutedsl = fi_ep.mxfp4_moe_ep_weights(
        FLASHINFER_MOE_EP_CUTEDSL,
        packed,
        packed,
        scale,
        scale,
    )
    assert cutedsl.w13.shape == (1, 1, 32)
    assert cutedsl.w13.dtype == torch.bfloat16
    assert cutedsl.w13_scale is None
    expected = torch.empty(32, dtype=torch.bfloat16)
    expected[::2] = torch.tensor(fi_ep._E2M1_LUT, dtype=torch.bfloat16)
    expected[1::2] = 0
    torch.testing.assert_close(cutedsl.w13[0, 0], expected)


def test_mxfp4_dequantization_preserves_e8m0_endpoint_encodings():
    packed = torch.full((2, 16), 0x22, dtype=torch.uint8)
    scale = torch.tensor([[0], [255]], dtype=torch.uint8)

    output = fi_ep._dequant_mxfp4_ue8m0_gran32(packed, scale)

    smallest_scale = (
        torch.tensor([0], dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .float()
        .to(torch.bfloat16)
    )
    assert output[0, 0] == smallest_scale[0]
    assert torch.isnan(output[1]).all()


def test_candidate_input_scale_loader_keeps_w1_w3_shards_separate():
    routed = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed)
    routed.quant_config = None
    routed.quant_method = SimpleNamespace(
        load_input_scales_by_shard=True,
        use_global_sf=False,
    )
    routed.moe_config = SimpleNamespace(tp_rank=0)
    routed._map_global_expert_id_to_local_expert_id = lambda expert_id: 0

    w13_scale = torch.nn.Parameter(torch.full((1, 2), torch.nan))
    assert routed.weight_loader(
        w13_scale,
        torch.tensor([135.0]),
        "w13_input_global_scale",
        "w1",
        7,
        return_success=True,
    )
    assert routed.weight_loader(
        w13_scale,
        torch.tensor([136.0]),
        "w13_input_global_scale",
        "w3",
        7,
        return_success=True,
    )
    torch.testing.assert_close(w13_scale, torch.tensor([[135.0, 136.0]]))


def test_compressed_tensors_finalization_uses_shared_adapter(monkeypatch):
    monkeypatch.setattr(
        ct_nvfp4,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=object()),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    captured: dict[str, Any] = {}
    adapter = SimpleNamespace()

    def make_adapter(moe, layer, weights, epilogue):
        captured.update(
            moe=moe,
            layer=layer,
            weights=weights,
            epilogue=epilogue,
        )
        return adapter

    monkeypatch.setattr(ct_nvfp4, "make_flashinfer_moe_ep", make_adapter)

    method = object.__new__(ct_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod)
    method.moe = _moe(routing_method=RoutingMethodType.Default)
    method.direct_backend = None
    layer = torch.nn.Module()
    layer.moe_config = method.moe
    layer.apply_router_weight_on_input = False
    layer.expert_map_manager = SimpleNamespace(placement_strategy="linear")
    tensors = {
        "w13_weight_packed": torch.empty(2, 1, dtype=torch.uint8),
        "w2_weight_packed": torch.empty(2, 1, dtype=torch.uint8),
        "w13_weight_scale": torch.empty(2, 1),
        "w2_weight_scale": torch.empty(2, 1),
        "w13_weight_global_scale": torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
        "w2_weight_global_scale": torch.tensor([5.0, 10.0]),
        "w13_input_global_scale": torch.tensor([[135.0, 135.0], [135.0, 135.0]]),
        "w2_input_global_scale": torch.tensor([143.0, 143.0]),
    }
    for name, value in tensors.items():
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))

    method._process_flashinfer_moe_ep_weights(layer)

    assert captured["epilogue"].input_norm_const == 135.0
    torch.testing.assert_close(
        captured["epilogue"].fc1_alpha,
        torch.tensor([1.0 / 270.0, 1.0 / 540.0]),
    )
    torch.testing.assert_close(
        captured["epilogue"].fc2_alpha,
        torch.tensor([1.0 / 715.0, 1.0 / 1430.0]),
    )
    assert captured["layer"] is layer
    assert method.direct_backend is adapter
    for name in tensors:
        assert not hasattr(layer, name)


def test_modelopt_finalization_uses_shared_adapter(monkeypatch):
    captured: dict[str, Any] = {}
    adapter = SimpleNamespace()

    def make_adapter(moe, layer, weights, epilogue):
        captured.update(
            moe=moe,
            layer=layer,
            weights=weights,
            epilogue=epilogue,
        )
        return adapter

    monkeypatch.setattr(modelopt, "make_flashinfer_moe_ep", make_adapter)

    method = object.__new__(modelopt.ModelOptNvFp4FusedMoE)
    method.moe = _moe(routing_method=RoutingMethodType.DeepseekV4)
    method.direct_backend = None
    layer = torch.nn.Module()
    layer.moe_config = method.moe
    layer.apply_router_weight_on_input = False
    layer.expert_map_manager = SimpleNamespace(placement_strategy="linear")
    tensors = {
        "w13_weight": torch.empty(2, 8, 4, dtype=torch.uint8),
        "w2_weight": torch.empty(2, 8, 2, dtype=torch.uint8),
        "w13_weight_scale": torch.ones(2, 8, 1, dtype=torch.float8_e4m3fn),
        "w2_weight_scale": torch.ones(2, 8, 1, dtype=torch.float8_e4m3fn),
        "w13_weight_scale_2": torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
        "w2_weight_scale_2": torch.tensor([8.0, 16.0]),
        "w13_input_scale": torch.ones(2, 2),
        "w2_input_scale": torch.ones(2),
    }
    for name, value in tensors.items():
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))

    method._process_flashinfer_moe_ep_weights(layer)

    assert captured["weights"].w13.data_ptr() == tensors["w13_weight"].data_ptr()
    assert captured["layer"] is layer
    torch.testing.assert_close(
        captured["epilogue"].fc1_alpha,
        torch.tensor([2.0, 4.0]),
    )
    assert method.direct_backend is adapter
    for name in tensors:
        assert not hasattr(layer, name)


def test_mxfp4_finalization_uses_shared_adapter(monkeypatch):
    adapter = SimpleNamespace()
    captured: dict[str, Any] = {}

    def make_adapter(moe, layer, *weights):
        captured.update(moe=moe, layer=layer, weights=weights)
        return adapter

    monkeypatch.setattr(mxfp4, "make_mxfp4_flashinfer_moe_ep", make_adapter)

    method = object.__new__(mxfp4.Mxfp4MoEMethod)
    method.moe = _moe(routing_method=RoutingMethodType.DeepseekV4)
    method.direct_backend = None
    layer = torch.nn.Module()
    layer.moe_config = method.moe
    layer.apply_router_weight_on_input = False
    layer.expert_map_manager = SimpleNamespace(placement_strategy="linear")
    tensors = {
        "w13_weight": torch.empty(1),
        "w2_weight": torch.empty(1),
        "w13_weight_scale": torch.empty(1),
        "w2_weight_scale": torch.empty(1),
    }
    for name, value in tensors.items():
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))

    method._process_flashinfer_moe_ep_weights(layer, *tensors.values())

    assert captured["layer"] is layer
    assert all(
        actual is expected
        for actual, expected in zip(captured["weights"], tensors.values(), strict=True)
    )
    assert method.direct_backend is adapter
    for name in tensors:
        assert not hasattr(layer, name)


@pytest.mark.parametrize(
    ("module", "method_cls"),
    (
        (ct_mxfp4, ct_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod),
        (inc_mxfp4_moe, inc_mxfp4_moe.INCMxfp4MoEMethod),
    ),
)
def test_group_mxfp4_finalization_uses_shared_adapter(
    monkeypatch,
    module: Any,
    method_cls: type,
):
    adapter = SimpleNamespace()
    captured: dict[str, Any] = {}

    def make_adapter(moe, layer, *weights):
        captured.update(moe=moe, layer=layer, weights=weights)
        return adapter

    monkeypatch.setattr(module, "make_mxfp4_flashinfer_moe_ep", make_adapter)
    method: Any = object.__new__(method_cls)
    method.moe = _moe()
    method.use_flashinfer_moe_ep = True
    method.direct_backend = None
    layer = torch.nn.Module()
    layer.moe_config = method.moe
    layer.apply_router_weight_on_input = False
    layer.expert_map_manager = SimpleNamespace(placement_strategy="linear")
    tensors = {
        "w13_weight_packed": torch.empty(1),
        "w2_weight_packed": torch.empty(1),
        "w13_weight_scale": torch.empty(1),
        "w2_weight_scale": torch.empty(1),
    }
    for name, value in tensors.items():
        setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))

    method.process_weights_after_loading(layer)

    assert captured["layer"] is layer
    assert all(
        actual.data_ptr() == expected.data_ptr()
        for actual, expected in zip(captured["weights"], tensors.values(), strict=True)
    )
    assert method.direct_backend is adapter
    for name in tensors:
        assert not hasattr(layer, name)


def test_is_monolithic_tolerates_deferred_expert_selection():
    method = object.__new__(mxfp4.Mxfp4MoEMethod)
    method.direct_backend = None
    method.moe_kernel = None
    method.experts_cls = None
    assert not method.is_monolithic


def test_direct_backend_contracts_are_traceable():
    direct_backend = SimpleNamespace(
        can_overlap_shared_experts=False,
        output_is_reduced=True,
        topk_indices_dtype=torch.int32,
        is_monolithic=False,
    )
    method = object.__new__(ct_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod)
    method.direct_backend = direct_backend
    method.moe_kernel = None

    def apply_contracts(value: torch.Tensor) -> torch.Tensor:
        if method.supports_internal_mk:
            value = value + 1
        if method.mk_can_overlap_shared_experts:
            value = value + 2
        if method.output_is_reduced:
            value = value + 4
        return value

    compiled = torch.compile(apply_contracts, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(torch.ones(1)), torch.tensor([6.0]))


def test_runner_accepts_direct_backend_reduced_output_contract():
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(output_is_reduced=True)
    )
    assert runner._fused_output_is_reduced is True
