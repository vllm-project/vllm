# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep_cutedsl as fi_ep
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4a4_nvfp4 as ct_nvfp4,
)


@pytest.fixture(autouse=True)
def clear_flashinfer_moe_ep_registry():
    fi_ep._FLASHINFER_MOE_EP_CUTEDSL.clear()
    yield
    fi_ep.destroy_flashinfer_moe_ep_cutedsl()


class _FakeMegaLayer:
    def __init__(self) -> None:
        self.forward_tensors: list[SimpleNamespace] = []
        self.warmup_tensors: list[SimpleNamespace] = []
        self.destroy_calls = 0

    def __call__(self, tensors):
        self.forward_tensors.append(tensors)
        return torch.empty_like(tensors.hidden_states)

    def warmup(self, tensors) -> None:
        self.warmup_tensors.append(tensors)

    def destroy(self) -> None:
        self.destroy_calls += 1


def _namespace(**kwargs):
    return SimpleNamespace(**kwargs)


def test_adapter_uses_public_config_and_calls_fi_for_zero_tokens(monkeypatch):
    mega_layer = _FakeMegaLayer()
    constructed: dict[str, SimpleNamespace] = {}

    def make_mega_layer(bootstrap, fleet, weights, backend):
        constructed.update(
            bootstrap=bootstrap,
            fleet=fleet,
            weights=weights,
            backend=backend,
        )
        return mega_layer

    api = SimpleNamespace(
        BootstrapConfig=_namespace,
        FleetParams=_namespace,
        MegaConfig=_namespace,
        MoEEpMegaLayer=make_mega_layer,
        MoEEpTensors=_namespace,
        MoEWeightPack=_namespace,
        Nvfp4CutedslMegaMoeConfig=_namespace,
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

    layer = SimpleNamespace(
        w13_weight_packed=torch.empty(2, 8, 4, dtype=torch.uint8),
        w2_weight_packed=torch.empty(2, 8, 2, dtype=torch.uint8),
        w13_weight_scale=torch.empty(2, 8, 1),
        w2_weight_scale=torch.empty(2, 8, 1),
        apply_router_weight_on_input=False,
    )
    moe = SimpleNamespace(
        num_experts=8,
        max_num_tokens=32,
        hidden_dim=8,
        intermediate_size=4,
        experts_per_token=2,
        swiglu_limit=7.0,
    )
    fc1_alpha = torch.tensor([0.5, 0.25])
    fc2_alpha = torch.tensor([0.125, 0.0625])
    fc1_norm_const = torch.tensor([143.0, 143.0])
    adapter = fi_ep.FlashInferMoeEpCutedsl(
        layer,
        moe,
        input_norm_const=135.0,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
    )

    kernel_config = constructed["backend"].megakernel
    assert kernel_config.apply_topk_in_fc1 is False
    assert kernel_config.in_kernel_fc2_reduce is False
    assert kernel_config.combine_dtype == "bf16"
    assert kernel_config.input_norm_const == 135.0
    assert kernel_config.gate_up_clamp == 7.0
    assert kernel_config.fc1_alpha is None
    assert kernel_config.fc2_alpha is None
    assert kernel_config.fc1_norm_const is None
    assert constructed["bootstrap"].process_group is process_group

    hidden_states = torch.empty(0, 8, dtype=torch.bfloat16)
    topk_ids = torch.empty(0, 2, dtype=torch.int32)
    topk_weights = torch.empty(0, 2, dtype=torch.float32)
    output = adapter(hidden_states, topk_ids, topk_weights)
    assert output.shape == hidden_states.shape
    assert len(mega_layer.forward_tensors) == 1
    call = mega_layer.forward_tensors[0]
    assert call.fc1_alpha is fc1_alpha
    assert call.fc2_alpha is fc2_alpha
    assert call.fc1_norm_const is fc1_norm_const

    adapter.warmup()
    warmup = mega_layer.warmup_tensors[0]
    assert warmup.hidden_states.shape == (1, 8)
    assert warmup.topk_ids.dtype == torch.int32
    torch.testing.assert_close(
        warmup.topk_ids, torch.tensor([[0, 4]], dtype=torch.int32)
    )

    adapter.destroy()
    adapter.destroy()
    assert mega_layer.destroy_calls == 1


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

    w2_scale = torch.nn.Parameter(torch.full((1,), torch.nan))
    assert routed.weight_loader(
        w2_scale,
        torch.tensor([143.0]),
        "w2_input_global_scale",
        "w2",
        7,
        return_success=True,
    )
    torch.testing.assert_close(w2_scale, torch.tensor([143.0]))


def test_candidate_weight_finalization_validates_scales_and_releases_source(
    monkeypatch,
):
    monkeypatch.setattr(
        ct_nvfp4,
        "get_ep_group",
        lambda: SimpleNamespace(device_group=object()),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    captured = {}
    adapter = object()

    def make_adapter(layer, moe, **kwargs):
        captured.update(kwargs)
        return adapter

    monkeypatch.setattr(ct_nvfp4, "FlashInferMoeEpCutedsl", make_adapter)

    method = object.__new__(ct_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod)
    method.moe = SimpleNamespace(ep_rank=0, num_local_experts=2)
    layer = torch.nn.Module()
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

    method._process_flashinfer_moe_ep_cutedsl_weights(layer)

    assert captured["input_norm_const"] == 135.0
    torch.testing.assert_close(
        captured["fc1_alpha"], torch.tensor([1.0 / 270.0, 1.0 / 540.0])
    )
    torch.testing.assert_close(
        captured["fc2_alpha"], torch.tensor([1.0 / 715.0, 1.0 / 1430.0])
    )
    torch.testing.assert_close(captured["fc1_norm_const"], torch.full((2,), 143.0))
    assert method._flashinfer_moe_ep_cutedsl is adapter
    for name in tensors:
        assert not hasattr(layer, name)


def test_candidate_rejects_missing_or_mismatched_input_scale_shards():
    with pytest.raises(ValueError, match="finite positive"):
        ct_nvfp4._require_exact_shard_match(
            "input_scale", torch.tensor([[135.0, torch.nan]])
        )
    with pytest.raises(ValueError, match="match exactly"):
        ct_nvfp4._require_exact_shard_match(
            "input_scale", torch.tensor([[135.0, 136.0]])
        )


def test_registry_warmup_and_destroy_are_ordered_and_idempotent():
    events = []

    class FakeAdapter:
        def __init__(self, index):
            self.index = index
            self.destroyed = False

        def warmup(self):
            events.append(("warmup", self.index))

        def destroy(self):
            if not self.destroyed:
                events.append(("destroy", self.index))
                self.destroyed = True

    adapters = [FakeAdapter(0), FakeAdapter(1)]
    for adapter in adapters:
        fi_ep._register_flashinfer_moe_ep_cutedsl(adapter)

    fi_ep.warmup_flashinfer_moe_ep_cutedsl()
    fi_ep.destroy_flashinfer_moe_ep_cutedsl()
    fi_ep.destroy_flashinfer_moe_ep_cutedsl()

    assert events == [
        ("warmup", 0),
        ("warmup", 1),
        ("destroy", 0),
        ("destroy", 1),
    ]
    assert fi_ep._FLASHINFER_MOE_EP_CUTEDSL == []


@pytest.mark.parametrize(
    ("dp_size", "hidden_dim", "intermediate_size", "num_local_experts"),
    (
        (4, 7168, 4096, 32),
        (2, 4096, 2048, 64),
    ),
    ids=("ml3-ep4", "ms4-ep2"),
)
def test_candidate_gate_accepts_framework_managed_topology(
    monkeypatch,
    dp_size: int,
    hidden_dim: int,
    intermediate_size: int,
    num_local_experts: int,
):
    parallel = SimpleNamespace(
        tensor_parallel_size=2,
        enable_dbo=False,
    )
    config = SimpleNamespace(
        parallel_config=parallel,
        weight_transfer_config=None,
    )
    moe = SimpleNamespace(
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        num_experts=128,
        num_local_experts=num_local_experts,
        num_logical_experts=128,
        experts_per_token=4,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        has_bias=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        is_lora_enabled=False,
        skip_final_all_reduce=False,
    )
    monkeypatch.setattr(ct_nvfp4, "get_current_vllm_config", lambda: config)

    ct_nvfp4._validate_flashinfer_moe_ep_cutedsl_config(moe, use_a16=False)

    config.weight_transfer_config = object()
    with pytest.raises(ValueError, match="runtime weight transfer"):
        ct_nvfp4._validate_flashinfer_moe_ep_cutedsl_config(moe, use_a16=False)


def test_runner_accepts_direct_backend_reduced_output_contract():
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(output_is_reduced=True)
    )
    assert runner._fused_output_is_reduced is True


@pytest.mark.parametrize(
    ("use_mega", "expected"),
    ((False, 1.0), (True, 6.0)),
)
def test_backend_contracts_are_fullgraph_traceable(
    use_mega: bool,
    expected: float,
) -> None:
    method = object.__new__(ct_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod)
    method.use_flashinfer_moe_ep_cutedsl = use_mega
    method.moe_kernel = None

    def apply_contracts(value: torch.Tensor) -> torch.Tensor:
        if method.supports_internal_mk:
            value = value + 1
        if method.mk_can_overlap_shared_experts:
            value = value + 2
        if method.output_is_reduced:
            value = value + 4
        return value

    value = torch.ones(1)
    compiled = torch.compile(apply_contracts, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(value), torch.tensor([expected]))
