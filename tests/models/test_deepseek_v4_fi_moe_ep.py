# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config.kernel import (
    FLASHINFER_MOE_EP_BACKENDS,
    FLASHINFER_MOE_EP_CUTEDSL,
    FLASHINFER_MOE_EP_DEEP_GEMM,
    MEGA_MOE_BACKENDS,
    NATIVE_MEGA_MOE_BACKENDS,
    validate_flashinfer_moe_ep_model,
)
from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep as fi_ep
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import (
    FlashInferMoeEpExperts,
    FlashInferMoeEpPrepareAndFinalize,
    epilogue_from_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    FLASHINFER_MOE_EP_MXFP4_BACKENDS,
    Mxfp4MoeBackend,
    make_mxfp4_moe_quant_config,
    map_mxfp4_backend,
    mxfp4_round_up_hidden_size_and_intermediate_size,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    backend_to_kernel_cls as mxfp4_backend_to_kernel_cls,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    map_nvfp4_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    backend_to_kernel_cls as nvfp4_backend_to_kernel_cls,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)


def test_flashinfer_backends_are_megakernels_outside_the_native_model_path():
    assert FLASHINFER_MOE_EP_BACKENDS <= MEGA_MOE_BACKENDS
    assert FLASHINFER_MOE_EP_BACKENDS.isdisjoint(NATIVE_MEGA_MOE_BACKENDS)


def test_only_deep_gemm_backend_is_dsv4_specific():
    validate_flashinfer_moe_ep_model(
        FLASHINFER_MOE_EP_CUTEDSL,
        ["MixtralForCausalLM"],
    )
    with pytest.raises(ValueError, match="only supported for DeepSeek-V4"):
        validate_flashinfer_moe_ep_model(
            FLASHINFER_MOE_EP_DEEP_GEMM,
            ["MixtralForCausalLM"],
        )
    validate_flashinfer_moe_ep_model(
        FLASHINFER_MOE_EP_DEEP_GEMM,
        ["DeepseekV4ForCausalLM"],
    )


@pytest.mark.parametrize(
    "architectures",
    [["KimiK3ForConditionalGeneration"], ["MixtralForCausalLM"]],
)
def test_native_deep_gemm_mega_moe_not_arch_gated(architectures):
    """Models validate native deep_gemm mega constraints at construction time."""
    validate_flashinfer_moe_ep_model("deep_gemm_mega_moe", architectures)


def test_non_fi_backend_ignores_architectures():
    validate_flashinfer_moe_ep_model("auto", ["MixtralForCausalLM"])


@pytest.mark.parametrize("moe_backend", sorted(MEGA_MOE_BACKENDS))
def test_token_sharding_backends_enable_dsv4_sequence_parallel(moe_backend: str):
    from vllm.models.deepseek_v4.nvidia.model import _use_sequence_parallel

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            tensor_parallel_size=8,
            data_parallel_size=1,
        ),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
    )
    assert _use_sequence_parallel(vllm_config)


def test_dsv4_requests_pre_fc2_router_weight_placement():
    dsv4 = SimpleNamespace(routing_method=RoutingMethodType.DeepseekV4)
    default = SimpleNamespace(routing_method=RoutingMethodType.Default)
    assert fi_ep.apply_topk_in_fc1(dsv4)
    assert not fi_ep.apply_topk_in_fc1(default)


def test_backend_specs_preserve_weight_format_contracts():
    cutedsl = fi_ep.flashinfer_moe_ep_backend_spec(FLASHINFER_MOE_EP_CUTEDSL)
    assert cutedsl.kernel == "cutedsl"
    assert cutedsl.weight_formats == frozenset({"nvfp4", "mxfp4"})

    deep_gemm = fi_ep.flashinfer_moe_ep_backend_spec(FLASHINFER_MOE_EP_DEEP_GEMM)
    assert deep_gemm.kernel == "deep_gemm"
    assert deep_gemm.weight_formats == frozenset({"mxfp4"})


def _megakernel_moe(moe_backend: str, **overrides) -> SimpleNamespace:
    fields = dict(
        moe_backend=moe_backend,
        moe_parallel_config=SimpleNamespace(
            use_ep=True, use_batched_activation_format=False
        ),
        is_act_and_mul=True,
        activation=MoEActivation.SILU,
        has_hash_routing=False,
        routing_method=RoutingMethodType.DeepseekV4,
        router_logits_dtype=torch.float32,
        hidden_dim=256,
        is_lora_enabled=False,
        skip_final_all_reduce=False,
        in_dtype=torch.bfloat16,
        has_bias=False,
        swiglu_alpha=None,
        swiglu_beta=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _is_supported(moe, weight_key, activation_key) -> tuple[bool, str]:
    supported, reason = FlashInferMoeEpExperts.is_supported_config(
        FlashInferMoeEpExperts,
        moe,
        weight_key,
        activation_key,
        mk.FusedMoEActivationFormat.Standard,
    )
    return supported, reason or ""


@pytest.fixture
def megakernel_host(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A vLLM config without DBO/EPLB/weight transfer, on a supported device."""
    config = SimpleNamespace(
        weight_transfer_config=None,
        parallel_config=SimpleNamespace(enable_dbo=False, enable_eplb=False),
    )
    monkeypatch.setattr(fi_ep, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        FlashInferMoeEpExperts, "_supports_current_device", staticmethod(lambda: True)
    )
    return config


def test_megakernel_requires_expert_parallel(megakernel_host):
    """The oracle rejects the megakernel through the experts' support check."""
    moe = _megakernel_moe(FLASHINFER_MOE_EP_CUTEDSL)
    assert _is_supported(moe, kNvfp4Static, kNvfp4Dynamic) == (True, "")

    moe.moe_parallel_config.use_ep = False
    supported, reason = _is_supported(moe, kNvfp4Static, kNvfp4Dynamic)
    assert not supported and "parallel config" in reason


def test_megakernel_reports_its_own_constraints(megakernel_host):
    """Constraints the framework does not know about surface as reasons."""
    _, reason = _is_supported(
        _megakernel_moe(FLASHINFER_MOE_EP_CUTEDSL), kNvfp4Static, None
    )
    assert "A16 activations" in reason

    deep_gemm = _megakernel_moe(
        FLASHINFER_MOE_EP_DEEP_GEMM, routing_method=RoutingMethodType.Renormalize
    )
    _, reason = _is_supported(deep_gemm, kMxfp4Static, None)
    assert "routing method" in reason

    megakernel_host.parallel_config.enable_eplb = True
    _, reason = _is_supported(
        _megakernel_moe(FLASHINFER_MOE_EP_DEEP_GEMM), kMxfp4Static, None
    )
    assert reason.endswith("EPLB")


def test_megakernels_are_oracle_backends():
    """Quant methods reach the megakernel through the NVFP4/MXFP4 oracles."""
    nvfp4 = map_nvfp4_backend(FLASHINFER_MOE_EP_CUTEDSL)
    assert nvfp4 is NvFp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL
    assert nvfp4_backend_to_kernel_cls(nvfp4) == [FlashInferMoeEpExperts]

    assert map_mxfp4_backend(FLASHINFER_MOE_EP_DEEP_GEMM) == [
        Mxfp4MoeBackend.FLASHINFER_MOE_EP_DEEP_GEMM
    ]
    for backend in FLASHINFER_MOE_EP_MXFP4_BACKENDS:
        assert mxfp4_backend_to_kernel_cls(backend) == [FlashInferMoeEpExperts]
        # The megakernel takes the model's own dimensions.
        assert mxfp4_round_up_hidden_size_and_intermediate_size(
            backend, 2880, 2880
        ) == (2880, 2880)

    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL, w1_scale=None, w2_scale=None
    )
    assert quant_config is not None and quant_config.use_nvfp4_w4a4


def test_megakernel_is_a_modular_kernel_with_pass_through_stages():
    """The megakernel plugs into ``FusedMoEKernel`` like any other experts impl.

    Routing stays outside (top-k ids in), dispatch and combine happen inside,
    so prepare passes tokens through untouched and finalize has nothing to do.
    """
    moe = make_dummy_moe_config(
        num_experts=8, experts_per_token=2, hidden_dim=256, intermediate_size=128
    )
    quant_config = FusedMoEQuantConfig.make("nvfp4", weight_dtype="nvfp4")
    kernel = mk.FusedMoEKernel(
        FlashInferMoeEpPrepareAndFinalize(),
        FlashInferMoeEpExperts(moe, quant_config),
    )

    assert not kernel.is_monolithic
    assert kernel.prepare_finalize.output_is_reduced()
    assert kernel.prepare_finalize.topk_indices_dtype() is torch.int32
    assert kernel.fused_experts.expects_unquantized_inputs
    assert isinstance(
        kernel.fused_experts.finalize_weight_and_reduce_impl(), TopKWeightAndReduceNoOP
    )
    assert kernel.fused_experts.workspace_shapes(
        16, 256, 256, 2, 8, 8, None, MoEActivation.SILU
    ) == ((0,), (0,), (16, 256))

    tokens = torch.zeros(4, 256, dtype=torch.bfloat16)
    prepared = kernel.prepare_finalize.prepare(
        tokens,
        torch.ones(4, 2),
        torch.zeros(4, 2, dtype=torch.int32),
        8,
        None,
        False,
        quant_config,
    )
    assert prepared[0] is tokens and all(item is None for item in prepared[1:])


def test_epilogue_takes_the_weight_global_scales_from_the_quant_config():
    """The canonical ``g1/g2_alphas`` become the per-expert fc1/fc2 alphas and
    keep aliasing the layer's tensors, so EPLB permutations reach the kernel."""
    g1_alphas = torch.tensor([1.0, 2.0])
    g2_alphas = torch.tensor([3.0, 4.0])
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        a1_gscale=torch.ones(2),
        a2_gscale=torch.ones(2),
        w1_scale=torch.ones(2, 4, 1),
        w2_scale=torch.ones(2, 2, 1),
    )

    epilogue = epilogue_from_quant_config(quant_config)

    assert epilogue.fc1_alpha is not None and epilogue.fc2_alpha is not None
    assert epilogue.fc1_alpha.data_ptr() == g1_alphas.data_ptr()
    assert epilogue.fc2_alpha.data_ptr() == g2_alphas.data_ptr()
    # Activations are quantized dynamically: no norm constants.
    assert epilogue.input_norm_const == 1.0 and epilogue.fc1_norm_const is None


def test_epilogue_rejects_unloaded_weight_global_scales():
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=torch.tensor([1.0, float("nan")]),
        g2_alphas=torch.ones(2),
        a1_gscale=torch.ones(2),
        a2_gscale=torch.ones(2),
        w1_scale=torch.ones(2, 4, 1),
        w2_scale=torch.ones(2, 2, 1),
    )
    with pytest.raises(ValueError, match="g1_alphas"):
        epilogue_from_quant_config(quant_config)


def test_mxfp4_checkpoints_keep_the_default_epilogue():
    """MXFP4 has no per-tensor global scales; the megakernel needs no alphas."""
    quant_config = FusedMoEQuantConfig.make("nvfp4", weight_dtype="mxfp4")
    assert epilogue_from_quant_config(quant_config) == fi_ep.FlashInferMoeEpEpilogue()
