# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import regex as re
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config.kernel import (
    FLASHINFER_MOE_EP_BACKENDS,
    FLASHINFER_MOE_EP_CUTEDSL,
    FLASHINFER_MOE_EP_DEEP_GEMM,
    MEGA_MOE_BACKENDS,
    NATIVE_MEGA_MOE_BACKENDS,
    PASSTHROUGH_ALL2ALL_BACKEND,
    bind_passthrough_all2all_backend,
    validate_flashinfer_moe_ep_model,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep as fi_ep
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts import flashinfer_moe_ep as fi_experts
from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import (
    FlashInferMoeEpExperts,
    epilogue_from_quant_config,
    sm107_nvfp4_epilogue,
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
    convert_to_nvfp4_moe_kernel_format,
    map_nvfp4_backend,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    backend_to_kernel_cls as nvfp4_backend_to_kernel_cls,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.passthrough import (
    PassThroughPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.argparse_utils import FlexibleArgumentParser


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
        require_decomposed_backend=False,
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
    monkeypatch.setattr(fi_ep, "_current_capability", lambda: (10, 0))
    monkeypatch.setattr(
        FlashInferMoeEpExperts, "_supports_current_device", staticmethod(lambda: True)
    )
    return config


@pytest.mark.parametrize(
    "capability, supported, sm107",
    [
        (DeviceCapability(10, 0), True, False),
        (DeviceCapability(10, 7), True, True),
        (DeviceCapability(12, 0), False, False),
        (None, False, False),
    ],
    ids=["sm100", "sm107", "sm120", "unknown"],
)
def test_device_gate_admits_sm107(monkeypatch, capability, supported, sm107):
    monkeypatch.setattr(
        fi_ep.current_platform, "get_device_capability", lambda: capability
    )
    assert fi_ep.supports_current_device() is supported
    assert FlashInferMoeEpExperts._supports_current_device() is supported
    assert fi_ep.is_sm107_device() is sm107


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


def test_sm107_admits_only_nvfp4_cutedsl(monkeypatch, megakernel_host):
    """SM107 runs the CuTeDSL megakernel for NVFP4 checkpoints only."""
    monkeypatch.setattr(fi_ep, "_current_capability", lambda: (10, 7))
    monkeypatch.setattr(fi_ep, "flashinfer_sm107_nvfp4_gap", lambda: None)

    cutedsl = _megakernel_moe(FLASHINFER_MOE_EP_CUTEDSL)
    assert _is_supported(cutedsl, kNvfp4Static, kNvfp4Dynamic) == (True, "")
    for backend in (FLASHINFER_MOE_EP_CUTEDSL, FLASHINFER_MOE_EP_DEEP_GEMM):
        supported, reason = _is_supported(_megakernel_moe(backend), kMxfp4Static, None)
        assert not supported and "MXFP4 weights on this GPU" in reason


def test_sm107_appends_flashinfer_gap_verbatim(monkeypatch, megakernel_host):
    monkeypatch.setattr(fi_ep, "_current_capability", lambda: (10, 7))
    monkeypatch.setattr(
        fi_ep, "flashinfer_sm107_nvfp4_gap", lambda: "SM107 with FlashInfer X"
    )
    moe = _megakernel_moe(FLASHINFER_MOE_EP_CUTEDSL)
    supported, reason = _is_supported(moe, kNvfp4Static, kNvfp4Dynamic)
    assert not supported and reason.endswith("SM107 with FlashInfer X")

    monkeypatch.setattr(fi_ep, "_current_capability", lambda: (10, 0))
    monkeypatch.setattr(
        fi_ep,
        "flashinfer_sm107_nvfp4_gap",
        lambda: pytest.fail("SM100 must not probe FlashInfer for SM107"),
    )
    assert _is_supported(moe, kNvfp4Static, kNvfp4Dynamic) == (True, "")
    assert _is_supported(moe, kMxfp4Static, None) == (True, "")


def _fake_flashinfer(config_cls, version: str) -> SimpleNamespace:
    moe_ep = SimpleNamespace(Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig=config_cls)
    return SimpleNamespace(__version__=version, moe_ep=moe_ep)


@dataclasses.dataclass
class _Sm107ConfigV073:
    apply_routing_weights_before_fc2: bool = True


@dataclasses.dataclass
class _Sm107ConfigV072:
    apply_topk_in_fc1: bool = True


@pytest.mark.parametrize(
    "flashinfer, expected, unexpected",
    [
        (_fake_flashinfer(_Sm107ConfigV073, "0.7.3rc1"), None, None),
        (
            _fake_flashinfer(_Sm107ConfigV072, "0.7.2.post1"),
            (
                "SM107 with FlashInfer 0.7.2.post1",
                "flashinfer-python>=0.7.3",
                "vllm download-kernels",
            ),
            None,
        ),
        (
            _fake_flashinfer(_Sm107ConfigV072, "0.7.3"),
            ("SM107 with FlashInfer 0.7.3", "flashinfer-python>=0.7.3"),
            None,
        ),
        (None, ("ModuleNotFoundError", "flashinfer.moe_ep"), ">=0.7.3"),
    ],
    ids=["v0.7.3rc1", "v0.7.2", "v0.7.3-old-field", "import-error"],
)
def test_sm107_flashinfer_probe(monkeypatch, flashinfer, expected, unexpected):
    """The probe admits FlashInfer by the 0.7.3 config field, not its version."""
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(
        sys.modules, "flashinfer.moe_ep", getattr(flashinfer, "moe_ep", None)
    )
    fi_ep.flashinfer_sm107_nvfp4_gap.cache_clear()
    try:
        gap = fi_ep.flashinfer_sm107_nvfp4_gap()
    finally:
        fi_ep.flashinfer_sm107_nvfp4_gap.cache_clear()
    if expected is None:
        assert gap is None
    else:
        assert gap is not None
        assert all(part in gap for part in expected), gap
        assert unexpected is None or unexpected not in gap


def test_sm107_refuses_eplb(monkeypatch, megakernel_host):
    megakernel_host.parallel_config.enable_eplb = True
    monkeypatch.setattr(fi_ep, "flashinfer_sm107_nvfp4_gap", lambda: None)
    moe = _megakernel_moe(FLASHINFER_MOE_EP_CUTEDSL)

    monkeypatch.setattr(fi_ep, "_current_capability", lambda: (10, 7))
    _, reason = _is_supported(moe, kNvfp4Static, kNvfp4Dynamic)
    assert reason.endswith("EPLB on SM107")

    monkeypatch.setattr(fi_ep, "_current_capability", lambda: (10, 0))
    assert _is_supported(moe, kNvfp4Static, kNvfp4Dynamic) == (True, "")


def test_sm107_epilogue_uses_checkpoint_activation_scales(monkeypatch):
    """The layer maxima of the activation scales set the norm constants and
    scale the alphas in place, so they keep aliasing the layer's tensors."""
    monkeypatch.setattr(
        fi_experts, "get_ep_group", lambda: SimpleNamespace(world_size=1)
    )
    g1 = torch.tensor([1e-4, 2e-4])
    g2 = torch.tensor([3e-4, 4e-4])
    g1_orig, g2_orig = g1.clone(), g2.clone()

    epilogue = sm107_nvfp4_epilogue(
        fi_ep.FlashInferMoeEpEpilogue(fc1_alpha=g1, fc2_alpha=g2),
        torch.tensor([[1e-3, 1e-3], [4e-3, 4e-3]]),
        torch.tensor([1e-4, 5e-4]),
    )

    assert epilogue.input_norm_const == pytest.approx(1 / 4e-3)
    assert epilogue.fc1_alpha is not None and epilogue.fc2_alpha is not None
    assert epilogue.fc1_alpha.data_ptr() == g1.data_ptr()
    assert epilogue.fc2_alpha.data_ptr() == g2.data_ptr()
    torch.testing.assert_close(epilogue.fc1_alpha, g1_orig * 4e-3, rtol=1e-6, atol=0)
    torch.testing.assert_close(epilogue.fc2_alpha, g2_orig * 5e-4, rtol=1e-6, atol=0)
    fc1_norm_const = epilogue.fc1_norm_const
    assert fc1_norm_const is not None
    assert fc1_norm_const.dtype == torch.float32 and fc1_norm_const.is_contiguous()
    torch.testing.assert_close(fc1_norm_const, torch.full((2,), 1 / 5e-4))


@pytest.mark.parametrize(
    "w13_input_scale, w2_input_scale, match",
    [
        (
            torch.tensor([[1e-3, float("nan")]]),
            torch.tensor([1e-3]),
            re.escape("got inf (w13) and 0.001"),
        ),
        (
            torch.tensor([[1e-3, 1e-3]]),
            torch.tensor([float("inf")]),
            re.escape("and inf (w2)"),
        ),
        (torch.zeros(1, 2), torch.zeros(1), "activation input scales"),
        (None, torch.tensor([1e-3]), "activation input scales"),
    ],
    ids=["nan", "inf", "all-zero", "missing"],
)
def test_sm107_epilogue_rejects_invalid_activation_scales(
    monkeypatch, w13_input_scale, w2_input_scale, match
):
    monkeypatch.setattr(
        fi_experts, "get_ep_group", lambda: SimpleNamespace(world_size=1)
    )
    epilogue = fi_ep.FlashInferMoeEpEpilogue(
        fc1_alpha=torch.ones(1), fc2_alpha=torch.ones(1)
    )
    with pytest.raises(ValueError, match=match) as error:
        sm107_nvfp4_epilogue(epilogue, w13_input_scale, w2_input_scale)
    assert "activation input scales" in str(error.value)


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
        PassThroughPrepareAndFinalize(),
        FlashInferMoeEpExperts(moe, quant_config),
    )

    assert not kernel.is_monolithic
    assert kernel.prepare_finalize.output_is_reduced()
    # Generic stages: the int32 top-k ids the megakernel wants are its concern.
    assert kernel.prepare_finalize.topk_indices_dtype() is None
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


def test_nvfp4_kernel_format_passes_megakernel_weights_through():
    """The megakernel takes the canonical tensors as they are; the oracle must
    not reject or reshape them."""
    tensors = tuple(torch.arange(4.0).reshape(2, 2) + i for i in range(8))
    converted = convert_to_nvfp4_moe_kernel_format(
        NvFp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL,
        SimpleNamespace(),
        *tensors,
        is_act_and_mul=True,
    )
    assert all(out is src for out, src in zip(converted, tensors))


def test_passthrough_all2all_backend_selects_the_pass_through_stages():
    """``all2all_backend="passthrough"`` is how the oracles get the
    pass-through prepare/finalize."""
    moe = make_dummy_moe_config(
        num_experts=8, experts_per_token=2, hidden_dim=256, intermediate_size=128
    )
    moe.moe_parallel_config = replace(
        moe.moe_parallel_config, all2all_backend=PASSTHROUGH_ALL2ALL_BACKEND
    )
    assert not moe.moe_parallel_config.use_all2all_kernels

    prepare_finalize = maybe_make_prepare_finalize(
        moe, FusedMoEQuantConfig.make("nvfp4", weight_dtype="nvfp4")
    )

    assert isinstance(prepare_finalize, PassThroughPrepareAndFinalize)


def test_passthrough_all2all_backend_is_not_a_cli_choice():
    """Users cannot pick the passthrough all2all backend explicitly."""
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    with pytest.raises(SystemExit):
        parser.parse_args(["--all2all-backend", PASSTHROUGH_ALL2ALL_BACKEND])
    parser.parse_args(["--all2all-backend", "deepep_low_latency"])


def test_megakernel_moe_backend_binds_the_all2all_backend():
    """A megakernel deployment gets its all2all backend by default."""
    parallel = SimpleNamespace(all2all_backend="allgather_reducescatter")
    bind_passthrough_all2all_backend(FLASHINFER_MOE_EP_CUTEDSL, parallel)
    assert parallel.all2all_backend == PASSTHROUGH_ALL2ALL_BACKEND
    bind_passthrough_all2all_backend(FLASHINFER_MOE_EP_CUTEDSL, parallel)

    with pytest.raises(ValueError, match="cannot be paired"):
        bind_passthrough_all2all_backend(
            FLASHINFER_MOE_EP_DEEP_GEMM,
            SimpleNamespace(all2all_backend="deepep_low_latency"),
        )
    with pytest.raises(ValueError, match="only valid with"):
        bind_passthrough_all2all_backend(
            "flashinfer_trtllm",
            SimpleNamespace(all2all_backend=PASSTHROUGH_ALL2ALL_BACKEND),
        )

    untouched = SimpleNamespace(all2all_backend="deepep_low_latency")
    bind_passthrough_all2all_backend("flashinfer_trtllm", untouched)
    assert untouched.all2all_backend == "deepep_low_latency"
