# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x ROCm MoE experts, deployment checks and post-expert finalize."""

from collections.abc import Callable
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    derive_aiter_quant_method,
    mxfp4_w4a16_extra_kwargs,
    rocm_aiter_fused_experts,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
    QuarkW8A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.models.muse_spark.amd.muse_moe_runner import (
    MuseMoERunner,
    MuseSparkAiterExperts,
    MuseSparkQuarkOCPMXFusedMoE,
    MuseSparkQuarkOCPMXMoEQuantConfig,
    MuseSparkQuarkW8A8Fp8FusedMoE,
    MuseSparkTopKWeightAndReduce,
    MuseSparkTritonExperts,
    MuseSparkUnquantizedFusedMoE,
    MuseSparkUnquantizedMoEQuantConfig,
    validate_bf16_deployment,
    validate_fp8_deployment,
    validate_mxfp4_deployment,
)
from vllm.platforms import current_platform


def _parallel_config(
    *,
    tp_size: int = 1,
    dp_size: int = 1,
    pcp_size: int = 1,
    ep_size: int = 1,
    sp_size: int = 1,
    enable_eplb: bool = False,
) -> FusedMoEParallelConfig:
    return FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=pcp_size,
        dp_size=dp_size,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=sp_size,
        use_ep=ep_size > 1,
        all2all_backend="allgather_reducescatter",
        enable_eplb=enable_eplb,
    )


def _moe_config(
    *,
    parallel_config: FusedMoEParallelConfig | None = None,
    moe_backend: MoEBackend = "triton",
    is_lora_enabled: bool = False,
    skip_final_all_reduce: bool = False,
    num_experts: int = 4,
    top_k: int = 2,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    device: str = "cpu",
    routing_method: RoutingMethodType = RoutingMethodType.Custom,
) -> FusedMoEConfig:
    parallel_config = parallel_config or _parallel_config()
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts // parallel_config.ep_size,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=torch.device(device),
        routing_method=routing_method,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        moe_backend=moe_backend,
        is_lora_enabled=is_lora_enabled,
        skip_final_all_reduce=skip_final_all_reduce,
    )


def _quant_config(
    norm_fn: Callable[[torch.Tensor], torch.Tensor],
    config_cls: type = MuseSparkUnquantizedMoEQuantConfig,
):
    return config_cls(
        **{
            f.name: getattr(FUSED_MOE_UNQUANTIZED_CONFIG, f.name)
            for f in fields(FusedMoEQuantConfig)
            if f.init
        },
        post_expert_norm_fn=norm_fn,
    )


def _rms_norm(x: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + 1e-6)).to(out_dtype or x.dtype)


def _route_launcher(
    experts: MuseSparkTritonExperts | MuseSparkAiterExperts,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    intermediate_size: int,
    num_experts: int,
    num_local_experts: int | None = None,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Callable[[], None]]:
    """Allocate outputs once; return them with a graph-capturable launcher."""
    num_tokens, hidden_size = hidden_states.shape
    shapes = experts.workspace_shapes(
        num_tokens,
        2 * intermediate_size,
        hidden_size,
        topk_ids.shape[1],
        num_experts,
        num_local_experts or num_experts,
        None,
        MoEActivation.SILU,
    )
    workspace13, workspace2, routes = (
        torch.empty(shape, device=hidden_states.device, dtype=hidden_states.dtype)
        for shape in shapes
    )

    def launch() -> None:
        experts.apply(
            routes,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            MoEActivation.SILU,
            num_experts,
            expert_map,
            None,
            None,
            workspace13,
            workspace2,
            None,
            False,
        )

    return routes, launch


def test_muse_aiter_experts_expand_routes_with_unit_weights(monkeypatch) -> None:
    config = _moe_config(parallel_config=_parallel_config(tp_size=8))
    experts = MuseSparkAiterExperts(
        config, _quant_config(_rms_norm, MuseSparkQuarkOCPMXMoEQuantConfig)
    )
    calls: list[dict[str, torch.Tensor | bool]] = []

    def _fake_aiter_fused_experts(**kwargs) -> torch.Tensor:
        hidden_states = kwargs["hidden_states"]
        topk_weights = kwargs["topk_weights"]
        topk_ids = kwargs["topk_ids"]
        calls.append(
            {
                "hidden_states": hidden_states.clone(),
                "topk_weights": topk_weights.clone(),
                "topk_ids": topk_ids.clone(),
                "apply_router_weight_on_input": kwargs["apply_router_weight_on_input"],
            }
        )
        return hidden_states + topk_ids.to(hidden_states.dtype)

    monkeypatch.setattr(
        "vllm.models.muse_spark.amd.muse_moe_runner.rocm_aiter_fused_experts",
        _fake_aiter_fused_experts,
    )

    hidden_states = torch.arange(16, dtype=torch.bfloat16).view(2, 8)
    topk_ids = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    output, launch = _route_launcher(
        experts,
        hidden_states,
        torch.empty(0, dtype=torch.bfloat16),
        torch.empty(0, dtype=torch.bfloat16),
        topk_weights,
        topk_ids,
        intermediate_size=8,
        num_experts=4,
    )
    assert output.shape == (2, 2, 8)
    launch()

    assert len(calls) == 1
    call = calls[0]
    expected_states = hidden_states[:, None, :].expand(2, 2, 8).reshape(4, 8)
    torch.testing.assert_close(call["hidden_states"], expected_states)
    torch.testing.assert_close(
        call["topk_ids"], torch.tensor([[3], [1], [0], [2]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        call["topk_weights"], torch.ones(4, 1, dtype=torch.float32)
    )
    assert call["apply_router_weight_on_input"] is False
    routed_ids = call["topk_ids"]
    assert isinstance(routed_ids, torch.Tensor)
    expected_output = (expected_states + routed_ids.to(expected_states.dtype)).view(
        2, 2, 8
    )
    routes = experts._take_route_output()
    assert routes is not None
    torch.testing.assert_close(routes, expected_output)
    assert experts._take_route_output() is None

    reducer = experts.finalize_weight_and_reduce_impl()
    assert isinstance(reducer, MuseSparkTopKWeightAndReduce)
    assert reducer.pre_norm_all_reduce


def test_muse_mxfp4_method_requires_w4a4_aiter(monkeypatch) -> None:
    state = {
        "scheme": "w_mxfp4_a_mxfp4",
        "backend": Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
    }

    def _fake_quark_init(self, moe, weight_quant_key, activation_quant_key) -> None:
        self.moe = moe
        self.weight_quant_key = weight_quant_key
        self.activation_quant_key = activation_quant_key
        self.ocp_mx_scheme = state["scheme"]
        self.mxfp4_backend = state["backend"]
        self.experts_cls = None

    monkeypatch.setattr(QuarkOCP_MX_MoEMethod, "__init__", _fake_quark_init)
    config = _moe_config(parallel_config=_parallel_config(tp_size=8))
    method = MuseSparkQuarkOCPMXFusedMoE(config, kMxfp4Static, kMxfp4Dynamic)
    assert method.experts_cls is MuseSparkAiterExperts

    state["scheme"] = "w_mxfp4"
    with pytest.raises(NotImplementedError, match="W4A4"):
        MuseSparkQuarkOCPMXFusedMoE(config, kMxfp4Static, kMxfp4Dynamic)

    state["scheme"] = "w_mxfp4_a_mxfp4"
    state["backend"] = Mxfp4MoeBackend.EMULATION
    with pytest.raises(NotImplementedError, match="AITER_MXFP4_MXFP4"):
        MuseSparkQuarkOCPMXFusedMoE(config, kMxfp4Static, kMxfp4Dynamic)


def test_muse_fp8_method_requires_route_preserving_triton(monkeypatch) -> None:
    state = {"backend": Fp8MoeBackend.TRITON}

    def _fake_quark_init(self, moe, weight_quant_key, activation_quant_key) -> None:
        self.moe = moe
        self.weight_quant_key = weight_quant_key
        self.activation_quant_key = activation_quant_key
        self.fp8_backend = state["backend"]
        self.experts_cls = None

    monkeypatch.setattr(QuarkW8A8Fp8MoEMethod, "__init__", _fake_quark_init)
    config = _moe_config(
        parallel_config=_parallel_config(tp_size=8), moe_backend="triton"
    )
    method = MuseSparkQuarkW8A8Fp8FusedMoE(
        config, kFp8StaticChannelSym, kFp8DynamicTokenSym
    )
    assert method.experts_cls is MuseSparkTritonExperts

    state["backend"] = Fp8MoeBackend.AITER
    with pytest.raises(NotImplementedError, match="Triton FP8"):
        MuseSparkQuarkW8A8Fp8FusedMoE(config, kFp8StaticChannelSym, kFp8DynamicTokenSym)


def test_muse_unquantized_method_is_pinned_to_triton(
    monkeypatch, default_vllm_config
) -> None:
    def _unexpected_backend_selection(*args, **kwargs):
        raise AssertionError("the generic backend oracle must not run")

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method."
        "select_unquantized_moe_backend",
        _unexpected_backend_selection,
    )
    method = MuseSparkUnquantizedFusedMoE(_moe_config(moe_backend="auto"))
    assert method.experts_cls is MuseSparkTritonExperts


def test_muse_quant_config_ignores_cached_properties(default_vllm_config) -> None:
    method = MuseSparkUnquantizedFusedMoE(_moe_config(moe_backend="auto"))
    # Other models may have populated this cache on the shared base config.
    assert FUSED_MOE_UNQUANTIZED_CONFIG.ocp_mx_scheme is None
    norm = lambda x: x
    layer = SimpleNamespace(post_expert_norm_fn=norm)
    config = method.get_fused_moe_quant_config(layer)
    assert isinstance(config, MuseSparkUnquantizedMoEQuantConfig)
    assert config.post_expert_norm_fn is norm
    assert config.ocp_mx_scheme is None


def test_post_expert_norm_precedes_router_weight_and_topk_reduce() -> None:
    routed = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0]],
            [[2.0, 3.0, 5.0, 7.0], [7.0, 5.0, 3.0, 2.0]],
        ],
        dtype=torch.bfloat16,
    )
    weights = torch.tensor([[0.8, 0.2], [0.35, 0.65]])
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    reducer = MuseSparkTopKWeightAndReduce(_rms_norm, pre_norm_all_reduce=False)

    actual = reducer.apply(None, routed, weights, ids, False)
    expected = (
        (_rms_norm(routed, out_dtype=torch.float32) * weights[..., None].float())
        .sum(dim=1)
        .to(routed.dtype)
    )
    wrong_order = _rms_norm(
        (routed.float() * weights[..., None]).sum(dim=1).to(routed.dtype)
    ).float()

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual.float(), wrong_order)


def test_tp_partial_routes_are_reduced_before_norm(monkeypatch) -> None:
    routed = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]], dtype=torch.bfloat16)
    peer = torch.tensor([[[7.0, 11.0], [13.0, 17.0]]], dtype=torch.float32)
    weights = torch.tensor([[0.25, 0.75]])
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    calls: list[torch.Tensor] = []
    norm_calls: list[tuple[torch.dtype, torch.dtype | None]] = []

    def _all_reduce(x: torch.Tensor) -> torch.Tensor:
        calls.append(x.clone())
        assert x.dtype == torch.float32
        return x + peer

    monkeypatch.setattr(
        "vllm.models.muse_spark.amd.muse_moe_runner.tensor_model_parallel_all_reduce",
        _all_reduce,
    )

    def _tracked_norm(
        x: torch.Tensor, *, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        norm_calls.append((x.dtype, out_dtype))
        return _rms_norm(x, out_dtype=out_dtype)

    reducer = MuseSparkTopKWeightAndReduce(_tracked_norm, pre_norm_all_reduce=True)
    actual = reducer.apply(None, routed, weights, ids, False)

    reduced = routed.float() + peer
    expected = (
        (_rms_norm(reduced, out_dtype=torch.float32) * weights[..., None].float())
        .sum(dim=1)
        .to(routed.dtype)
    )
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1
    assert norm_calls == [(torch.float32, torch.float32)]


def test_finalize_is_chunk_invariant() -> None:
    torch.manual_seed(0)
    routed = torch.randn(11, 3, 7, dtype=torch.bfloat16)
    weights = torch.softmax(torch.randn(11, 3), dim=-1)
    ids = torch.arange(33, dtype=torch.int32).view(11, 3)

    unchunked = MuseSparkTopKWeightAndReduce(_rms_norm, pre_norm_all_reduce=False)
    chunked = MuseSparkTopKWeightAndReduce(_rms_norm, pre_norm_all_reduce=False)
    chunked._chunk_size = lambda top_k, hidden_dim: 2  # type: ignore[method-assign]

    expected = unchunked.apply(None, routed, weights, ids, False)
    actual = chunked.apply(None, routed, weights, ids, False)
    torch.testing.assert_close(actual, expected)


_FAIL_CLOSED_EVERYWHERE = [
    (_moe_config(parallel_config=_parallel_config(dp_size=2)), "DP1"),
    (_moe_config(parallel_config=_parallel_config(pcp_size=2)), "PCP1"),
    (_moe_config(parallel_config=_parallel_config(sp_size=2)), "sequence"),
    (_moe_config(parallel_config=_parallel_config(enable_eplb=True)), "EPLB"),
    (_moe_config(is_lora_enabled=True), "LoRA"),
    (_moe_config(skip_final_all_reduce=True), "unreduced"),
]


# All three validators share the parallelism checks; test those once.
@pytest.mark.parametrize(
    "validate,config,reason",
    [(validate_mxfp4_deployment, *case) for case in _FAIL_CLOSED_EVERYWHERE]
    + [
        (validate_bf16_deployment, _moe_config(moe_backend="aiter"), "Triton"),
        (validate_fp8_deployment, _moe_config(moe_backend="auto"), "Triton"),
        (validate_fp8_deployment, _moe_config(moe_backend="aiter"), "Triton"),
    ],
)
def test_unsupported_deployments_fail_closed(
    validate: Callable[[FusedMoEConfig], None], config: FusedMoEConfig, reason: str
) -> None:
    with pytest.raises(NotImplementedError, match=reason):
        validate(config)


@pytest.mark.parametrize(
    "tp_size,ep_size,reduced", [(1, 1, False), (8, 1, True), (1, 4, False)]
)
def test_runner_reduces_only_expert_parallel_partials(
    tp_size: int, ep_size: int, reduced: bool
) -> None:
    """TP ranks are complete after the pre-norm all-reduce; EP ranks still need one."""
    config = _moe_config(
        parallel_config=_parallel_config(tp_size=tp_size, ep_size=ep_size)
    )
    validate_bf16_deployment(config)
    validate_fp8_deployment(config)
    validate_mxfp4_deployment(config)
    runner = MuseMoERunner.__new__(MuseMoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = config
    assert runner._fused_output_is_reduced is reduced


ROCM_AVAILABLE = current_platform.is_rocm()
ROCM_GFX950 = False
ROCM_AITER_SUPPORTED = False

if ROCM_AVAILABLE:
    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx950

    ROCM_GFX950 = on_gfx950()
    ROCM_AITER_SUPPORTED = is_aiter_found_and_supported()


@pytest.fixture
def enable_rocm_aiter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    rocm_aiter_ops.refresh_env_variables()
    yield
    rocm_aiter_ops.refresh_env_variables()


def _explicit_expert_outputs(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_tokens, top_k = topk_ids.shape
    intermediate_size = w2.shape[-1]
    outputs = []
    for token_idx in range(num_tokens):
        routed = []
        for route_idx in range(top_k):
            expert_idx = int(topk_ids[token_idx, route_idx])
            gate_up = (hidden_states[token_idx].float() @ w1[expert_idx].float().T).to(
                hidden_states.dtype
            )
            gate, up = gate_up.split(intermediate_size)
            activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                hidden_states.dtype
            )
            routed.append(
                (activated.float() @ w2[expert_idx].float().T).to(hidden_states.dtype)
            )
        outputs.append(torch.stack(routed))
    return torch.stack(outputs)


def _mxfp4_weights(
    w13: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, MuseSparkQuarkOCPMXMoEQuantConfig]:
    """Quantize to MXFP4 and convert to the AITER kernel layout."""
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    num_experts, fused_intermediate, hidden_size = w13.shape
    w13_quant, w13_scale = dynamic_mxfp4_quant(w13.reshape(-1, hidden_size))
    w13_quant = w13_quant.reshape(num_experts, fused_intermediate, -1)
    w13_scale = w13_scale.reshape(num_experts, fused_intermediate, -1)

    _, output_size, intermediate_size = w2.shape
    w2_quant, w2_scale = dynamic_mxfp4_quant(w2.reshape(-1, intermediate_size))
    w2_quant = w2_quant.reshape(num_experts, output_size, -1)
    w2_scale = w2_scale.reshape(num_experts, output_size, -1)

    layer = SimpleNamespace(
        w13_weight=w13_quant,
        w2_weight=w2_quant,
        w13_weight_scale=w13_scale,
        w2_weight_scale=w2_scale,
        w13_input_scale=None,
        w2_input_scale=None,
    )
    w13_quant, w2_quant, w13_scale, w2_scale, _, _ = (
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            layer=layer,
            w13_weight=w13_quant,
            w2_weight=w2_quant,
            w13_weight_scale=w13_scale,
            w2_weight_scale=w2_scale,
        )
    )
    base_quant_config = make_mxfp4_moe_quant_config(
        mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
    )
    assert base_quant_config is not None
    quant_config = MuseSparkQuarkOCPMXMoEQuantConfig(
        **vars(base_quant_config), post_expert_norm_fn=_rms_norm
    )
    return w13_quant, w2_quant, quant_config


@pytest.mark.skipif(not ROCM_GFX950, reason="requires a GFX950 GPU")
@pytest.mark.skipif(not ROCM_AITER_SUPPORTED, reason="requires supported AITER")
@torch.inference_mode()
def test_rocm_aiter_mxfp4_route_expansion_numeric_and_graph_replay(
    enable_rocm_aiter,
) -> None:
    torch.manual_seed(1234)
    num_tokens, num_experts, top_k = 16, 8, 4
    hidden_size, intermediate_size = 256, 256
    dtype = torch.bfloat16
    device = torch.device("cuda")
    config = _moe_config(
        moe_backend="auto",
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
    )

    w13_float = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        / hidden_size**0.5
    )
    w2_float = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=dtype,
            device=device,
        )
        / intermediate_size**0.5
    )
    w13, w2, quant_config = _mxfp4_weights(w13_float, w2_float)
    experts = MuseSparkAiterExperts(config, quant_config)
    reducer = experts.finalize_weight_and_reduce_impl()

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    topk_logits, topk_ids = torch.topk(
        torch.randn(num_tokens, num_experts, device=device), top_k, dim=-1
    )
    topk_weights = torch.softmax(topk_logits, dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    route_buffer, launch = _route_launcher(
        experts,
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )
    output = torch.empty(num_tokens, hidden_size, device=device, dtype=dtype)
    parked: list[torch.Tensor] = []

    def _run() -> torch.Tensor:
        launch()
        parked[:] = [experts._route_output]
        return reducer.apply(output, route_buffer, topk_weights, topk_ids, False)

    direct = rocm_aiter_fused_experts(
        hidden_states=hidden_states,
        w1=w13,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        moe_config=config,
        activation=MoEActivation.SILU,
        quant_config=quant_config,
        quant_method=derive_aiter_quant_method(quant_config),
        output_dtype=dtype,
        **mxfp4_w4a16_extra_kwargs(quant_config, config),
    ).clone()
    actual = _run().clone()
    routes = parked[0]
    assert routes.shape == route_buffer.shape
    assert torch.isfinite(direct).all()
    assert torch.isfinite(routes).all()
    recombined = (routes.float() * topk_weights[..., None]).sum(dim=1).to(dtype)
    expected = (
        (_rms_norm(routes, out_dtype=torch.float32) * topk_weights[..., None])
        .sum(dim=1)
        .to(dtype)
    )
    torch.testing.assert_close(recombined, direct, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(actual, expected)

    for _ in range(3):
        _run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run()

    first = captured.clone()
    hidden_states.mul_(0.75)
    graph.replay()
    torch.accelerator.synchronize()
    assert not torch.equal(first, captured)


@pytest.mark.skipif(not ROCM_AVAILABLE, reason="requires a ROCm GPU")
@torch.inference_mode()
def test_rocm_triton_experts_numeric_and_graph_replay() -> None:
    torch.manual_seed(1234)
    num_tokens, num_experts, top_k = 7, 4, 2
    hidden_size, intermediate_size = 64, 128
    dtype = torch.bfloat16

    norm_scale = torch.linspace(0.55, 1.45, hidden_size, device="cuda", dtype=dtype)

    def _input_scaled_rms_norm(
        x: torch.Tensor, *, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        scaled = x.float() * norm_scale.float()
        variance = scaled.square().mean(dim=-1, keepdim=True)
        return (scaled * torch.rsqrt(variance + 1e-6)).to(out_dtype or x.dtype)

    moe_config = _moe_config(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device="cuda",
    )
    experts = MuseSparkTritonExperts(moe_config, _quant_config(_input_scaled_rms_norm))
    reducer = experts.finalize_weight_and_reduce_impl()

    hidden_states = (torch.randn(num_tokens, hidden_size, device="cuda") * 0.2).to(
        dtype
    )
    w1 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
        )
        * 0.08
    ).to(dtype)
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
        )
        * 0.08
    ).to(dtype)
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [3, 0], [1, 2], [0, 3], [2, 1], [3, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.tensor(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.15, 0.85],
            [0.55, 0.45],
            [0.72, 0.28],
            [0.41, 0.59],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    raw_buffer, launch = _route_launcher(
        experts,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )
    output = torch.empty(num_tokens, hidden_size, device="cuda", dtype=dtype)
    parked: list[torch.Tensor] = []

    def _run() -> torch.Tensor:
        launch()
        parked[:] = [experts._route_output]
        return reducer.apply(output, raw_buffer, topk_weights, topk_ids, False)

    actual = _run().clone()
    raw = parked[0]
    assert raw.shape == raw_buffer.shape
    raw_reference = _explicit_expert_outputs(hidden_states, w1, w2, topk_ids)
    expected = (
        _input_scaled_rms_norm(raw_reference, out_dtype=torch.float32)
        * topk_weights[..., None]
    ).sum(dim=1)
    torch.testing.assert_close(raw, raw_reference, rtol=2.5e-2, atol=3e-2)
    torch.testing.assert_close(actual.float(), expected, rtol=2.5e-2, atol=3e-2)

    for _ in range(3):
        _run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run()

    hidden_states.copy_(torch.flip(hidden_states, dims=[0]))
    expected_replay = (
        _input_scaled_rms_norm(
            _explicit_expert_outputs(hidden_states, w1, w2, topk_ids),
            out_dtype=torch.float32,
        )
        * topk_weights[..., None]
    ).sum(dim=1)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        captured.float(), expected_replay, rtol=2.5e-2, atol=3e-2
    )


@pytest.mark.skipif(not ROCM_AVAILABLE, reason="requires a ROCm GPU")
@pytest.mark.parametrize("routing", ["random", "first_rank_only"])
@pytest.mark.parametrize("backend", ["triton", "aiter_mxfp4"])
@torch.inference_mode()
def test_rocm_ep_ranks_sum_to_the_unsharded_output(
    request: pytest.FixtureRequest, backend: str, routing: str
) -> None:
    """Each EP rank's finalize is a disjoint addend, whatever non-local rows hold."""
    if backend == "aiter_mxfp4":
        if not (ROCM_GFX950 and ROCM_AITER_SUPPORTED):
            pytest.skip("requires a GFX950 GPU with supported AITER")
        request.getfixturevalue("enable_rocm_aiter")
    torch.manual_seed(1234)
    num_tokens, num_experts, top_k, ep_size = 300, 8, 4, 2
    hidden_size = intermediate_size = 256
    dtype = torch.bfloat16
    w13_full = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device="cuda"
    ) / (hidden_size**0.5)
    w2_full = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=dtype, device="cuda"
    ) / (intermediate_size**0.5)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    if routing == "first_rank_only":
        logits[:, num_experts // ep_size :] = -torch.inf
    topk_logits, topk_ids = torch.topk(logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    def _finalize(
        size: int, ep_rank: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        local = num_experts // size
        rows = slice(ep_rank * local, (ep_rank + 1) * local)
        w13, w2 = w13_full[rows].contiguous(), w2_full[rows].contiguous()
        _, expert_map, expert_mask = determine_expert_map(
            size, ep_rank, num_experts, return_expert_mask=True
        )
        config = _moe_config(
            parallel_config=_parallel_config(ep_size=size),
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device="cuda",
            routing_method=RoutingMethodType.Renormalize,
        )
        experts: MuseSparkTritonExperts | MuseSparkAiterExperts
        if backend == "triton":
            experts = MuseSparkTritonExperts(config, _quant_config(_rms_norm))
            kernel_map = expert_map
        else:
            w13, w2, quant_config = _mxfp4_weights(w13, w2)
            experts = MuseSparkAiterExperts(config, quant_config)
            kernel_map = expert_mask
        route_buffer, launch = _route_launcher(
            experts,
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_ids,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=local,
            expert_map=None if kernel_map is None else kernel_map.cuda(),
        )
        launch()
        assert experts._route_output is not None
        routes = experts._route_output.clone()
        out = torch.empty(num_tokens, hidden_size, device="cuda", dtype=dtype)
        reducer = experts.finalize_weight_and_reduce_impl()
        out = reducer.apply(out, route_buffer, topk_weights, topk_ids, False)
        return routes, out, expert_map

    full_routes, full, _ = _finalize(1, 0)
    summed = torch.zeros(num_tokens, hidden_size, device="cuda")
    for ep_rank in range(ep_size):
        routes, partial, expert_map = _finalize(ep_size, ep_rank)
        assert expert_map is not None
        is_local = expert_map.cuda()[topk_ids.long()] >= 0
        torch.testing.assert_close(routes[is_local], full_routes[is_local])
        summed += partial.float()
    torch.testing.assert_close(summed, full.float(), rtol=1e-2, atol=1e-2)


def _post_expert_norm_module(hidden_size: int) -> torch.nn.Module:
    from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNormWithInputScale

    norm = MuseSpark1xRMSNormWithInputScale(
        hidden_size=hidden_size,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None,
        output_norm_gain_center_type="zero",
    )
    with torch.no_grad():
        norm.weight.normal_(0.0, 0.1)
    return norm


def test_fused_finalize_is_only_used_for_supported_norm_modules() -> None:
    assert (
        MuseSparkTopKWeightAndReduce(_rms_norm, pre_norm_all_reduce=False).fused_norm
        is None
    )
    norm = _post_expert_norm_module(16)
    reducer = MuseSparkTopKWeightAndReduce(norm.forward, pre_norm_all_reduce=False)
    assert reducer.fused_norm is norm


@pytest.mark.skipif(not ROCM_AVAILABLE, reason="requires a ROCm GPU")
@pytest.mark.parametrize("routed_dtype", [torch.float32, torch.bfloat16])
def test_fused_finalize_matches_eager_tail(routed_dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    num_tokens, top_k, hidden = 37, 8, 4096
    norm = _post_expert_norm_module(hidden).cuda()
    routed = (torch.randn(num_tokens, top_k, hidden, device="cuda") * 3).to(
        routed_dtype
    )
    weights = torch.softmax(torch.randn(num_tokens, top_k, device="cuda"), dim=-1)
    ids = torch.arange(num_tokens * top_k, dtype=torch.int32, device="cuda").view(
        num_tokens, top_k
    )

    fused = MuseSparkTopKWeightAndReduce(norm.forward, pre_norm_all_reduce=False)
    assert fused.fused_norm is not None
    assert fused.fused_norm is norm
    eager = MuseSparkTopKWeightAndReduce(norm.forward, pre_norm_all_reduce=False)
    eager.fused_norm = None

    out_dtype = torch.bfloat16
    actual = fused.apply(
        torch.empty(num_tokens, hidden, device="cuda", dtype=out_dtype),
        routed,
        weights,
        ids,
        False,
    )
    expected = eager.apply(
        torch.empty(num_tokens, hidden, device="cuda", dtype=out_dtype),
        routed,
        weights,
        ids,
        False,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not ROCM_AVAILABLE, reason="requires a ROCm GPU")
def test_token_sharded_finalize_pads_and_slices_per_rank(monkeypatch) -> None:
    """Emulate TP2 rank 0 whose peer contributes nothing to the reduction."""
    torch.manual_seed(0)
    num_tokens, top_k, hidden = 5, 8, 4096
    norm = _post_expert_norm_module(hidden).cuda()
    routed = torch.randn(num_tokens, top_k, hidden, device="cuda", dtype=torch.bfloat16)
    weights = torch.softmax(torch.randn(num_tokens, top_k, device="cuda"), dim=-1)
    ids = torch.zeros(num_tokens, top_k, dtype=torch.int32, device="cuda")
    scattered: list[torch.Tensor] = []

    def _reduce_scatter(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0 and x.shape[0] % 2 == 0
        scattered.append(x)
        return x[: x.shape[0] // 2]

    def _all_gather(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0
        return torch.cat([x, torch.zeros_like(x)])

    module = "vllm.models.muse_spark.amd.muse_moe_runner"
    monkeypatch.setattr(
        f"{module}.tensor_model_parallel_reduce_scatter", _reduce_scatter
    )
    monkeypatch.setattr(f"{module}.tensor_model_parallel_all_gather", _all_gather)
    monkeypatch.setattr(f"{module}.get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(f"{module}.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        f"{module}.tensor_model_parallel_all_reduce",
        lambda x: pytest.fail("all-reduce must not run on the token-sharded path"),
    )

    reducer = MuseSparkTopKWeightAndReduce(norm.forward, pre_norm_all_reduce=True)
    reducer.rs_min_tokens = 1
    actual = reducer.apply(None, routed, weights, ids, False)

    # 5 tokens pad to 6; rank 0 owns tokens 0..2, the gathered rows 3..4 are the
    # peer's (zeros here).
    assert scattered[0].shape[0] == 6
    reference = MuseSparkTopKWeightAndReduce(norm.forward, pre_norm_all_reduce=False)
    expected = reference.apply(None, routed[:3], weights[:3], ids[:3], False)
    torch.testing.assert_close(actual[:3], expected, atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(actual[3:]) == 0
