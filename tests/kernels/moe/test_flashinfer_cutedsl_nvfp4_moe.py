# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FlashInfer CuTeDSL NVFP4 MoE."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    break_fp4_bytes,
)
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import (
    SiluAndMul,
    SiluAndMulWithClamp,
    SwigluOAIAndMul,
)
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    make_nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    prepare_nvfp4_moe_layer_for_flashinfer_cutedsl,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    quantize_nvfp4_per_token_input,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutedsl_moe_nvfp4
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if not has_flashinfer_cutedsl_moe_nvfp4() or not (
    current_platform.is_device_capability_family(100)
):
    pytest.skip(
        "Requires FlashInfer CuTeDSL NVFP4 MoE on SM100",
        allow_module_level=True,
    )

_SWIGLU_ALPHA = 1.702
_SWIGLU_BETA = 1.0
# The GEMM1 output is O(0.5) at these tensor scales, so the OAI default of 7.0
# would never clamp anything. Pick a limit that actually bites.
_SWIGLU_LIMIT = 0.3

_ACT_CASES = [
    pytest.param(MoEActivation.SILU, None, None, None, id="silu"),
    pytest.param(MoEActivation.SILU, None, None, _SWIGLU_LIMIT, id="silu-clamp"),
    pytest.param(MoEActivation.RELU2_NO_MUL, None, None, None, id="relu2_no_mul"),
    pytest.param(
        MoEActivation.SWIGLUOAI,
        _SWIGLU_ALPHA,
        _SWIGLU_BETA,
        _SWIGLU_LIMIT,
        id="swigluoai",
    ),
    pytest.param(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        _SWIGLU_ALPHA,
        _SWIGLU_BETA,
        _SWIGLU_LIMIT,
        id="swigluoai_uninterleave",
    ),
]


def _reference_activation(
    activation: MoEActivation,
    alpha: float | None,
    beta: float | None,
    limit: float | None,
):
    """VLLM's own op for this activation, so the reference is not re-derived."""
    if activation == MoEActivation.RELU2_NO_MUL:
        return lambda x: torch.square(torch.relu(x))
    if activation == MoEActivation.SWIGLUOAI:
        # SwigluOAIAndMul hardcodes beta=1 and reads gate/up interleaved.
        assert beta == 1.0
        return SwigluOAIAndMul(alpha=alpha, limit=limit)
    if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        return SiluAndMulWithClamp(limit, alpha, beta, compile_native=False)
    if limit is not None:
        return SiluAndMulWithClamp(limit, compile_native=False)
    return SiluAndMul()


def _torch_moe_reference(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    act_fn,
) -> torch.Tensor:
    m = a.shape[0]
    weights, ids = torch.topk(torch.softmax(score, dim=-1, dtype=torch.float32), topk)
    x = a.view(m, 1, -1).repeat(1, topk, 1).reshape(m * topk, -1)
    flat_ids = ids.reshape(-1)
    out = torch.zeros(m * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    for expert in range(w1.shape[0]):
        mask = flat_ids == expert
        if mask.any():
            acc = act_fn(x[mask] @ w1[expert].transpose(0, 1)).to(a.dtype)
            out[mask] = acc @ w2[expert].transpose(0, 1)
    return (
        (out.view(m, topk, -1).float() * weights.view(m, topk, 1))
        .sum(dim=1)
        .to(a.dtype)
    )


def _quantize_nvfp4_linear(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights_q = []
    scales = []
    global_scales = []
    for expert_weight in weight:
        global_scale = (
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / expert_weight.abs().max()
        ).to(torch.float32)
        weight_q, scale = ops.scaled_fp4_quant(
            expert_weight,
            global_scale,
            is_sf_swizzled_layout=False,
        )
        weights_q.append(weight_q)
        scales.append(scale)
        global_scales.append(global_scale)
    return torch.stack(weights_q), torch.stack(scales), torch.stack(global_scales)


def _dequantize_nvfp4_linear(
    tensor_fp4: torch.Tensor,
    tensor_sf: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(m, k // 16, 16)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn).to(torch.float32)
    tensor_sf = tensor_sf[:, : k // 16] / global_scale
    return (tensor_f32 * tensor_sf.unsqueeze(-1)).reshape(m, k).to(dtype)


@pytest.mark.parametrize(
    "m,n,k,e,topk",
    [
        (16, 128, 512, 4, 2),
        (16, 192, 512, 4, 2),
        (16, 192, 2688, 4, 2),
        (16, 256, 512, 1, 1),
        (16, 256, 512, 4, 2),
    ],
)
@pytest.mark.parametrize("activation,alpha,beta,limit", _ACT_CASES)
@pytest.mark.parametrize("per_token_activation", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("localization", ["off", "retain", "free"])
@torch.inference_mode()
def test_flashinfer_cutedsl_fp4_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    activation: MoEActivation,
    alpha: float | None,
    beta: float | None,
    limit: float | None,
    per_token_activation: bool,
    dtype: torch.dtype,
    localization: str,
    monkeypatch,
    workspace_init,
):
    import vllm.distributed.localization as loc
    import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe as mod

    if current_platform.is_device_capability(107) and per_token_activation:
        pytest.skip("Rubin CuTeDSL MoE does not support per-token activation scales")
    if current_platform.is_device_capability(107) and (
        alpha is not None or beta is not None or limit is not None
    ):
        pytest.skip("Rubin CuTeDSL does not support custom SwiGLU parameters")
    resources = None
    if localization != "off":
        if per_token_activation:
            pytest.skip("Localized CuTeDSL MoE does not support per-token scales")
        if n != 256:
            pytest.skip("Localized regression uses aligned per-domain shards")
        if not current_platform.is_device_capability(107):
            pytest.skip("Localized CuTeDSL MoE requires Rubin")
        if not hasattr(torch.cuda, "LocalizedAllocator") or not hasattr(
            torch.cuda, "execute_on_streams"
        ):
            pytest.skip(
                "Requires PyTorch locality allocation and stream execution APIs"
            )
        monkeypatch.setenv("VLLM_USE_LOCALITY_DOMAINS", "1")
        resources = loc.maybe_init_localized_resources(torch.device("cuda"))
        assert resources is not None, "localization initialization failed"
    monkeypatch.setattr(loc, "get_localized_resources", lambda: resources)
    monkeypatch.setattr(mod, "get_localized_resources", lambda: resources)
    monkeypatch.setenv(
        "VLLM_FLASHINFER_CUTEDSL_MOE_LOCALIZE_FREE_FULL_WEIGHTS",
        str(int(localization == "free")),
    )
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        # Keep GEMM1 activation variance stable as K changes, so the fixed
        # tolerance against the BF16 GEMM2 reference measures comparable FP4
        # quantization error at each hidden width.
        hidden_states *= (512 / k) ** 0.5

        w1_rows = 2 * n if activation.is_gated else n
        w1 = torch.randn((e, w1_rows, k), device="cuda", dtype=dtype) / 15
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15
        w1_q, w1_scale, w1_global_scale = _quantize_nvfp4_linear(w1)
        w2_q, w2_scale, w2_global_scale = _quantize_nvfp4_linear(w2)

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, score, topk, renormalize=False
        )

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )
        fake_layer = torch.nn.Module()
        fake_layer.activation = activation
        fake_layer.moe_config = moe_config
        a1_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        a2_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        (
            w1_cutedsl,
            w1_scale_cutedsl,
            w1_alpha,
            a1_scale,
            w2_cutedsl,
            w2_scale_cutedsl,
            w2_alpha,
            a2_scale,
        ) = prepare_nvfp4_moe_layer_for_flashinfer_cutedsl(
            layer=fake_layer,
            w13=w1_q.clone(),
            w13_scale=w1_scale,
            w13_scale_2=(1.0 / w1_global_scale),
            a13_scale=a1_scale,
            w2=w2_q.clone(),
            w2_scale=w2_scale,
            w2_scale_2=(1.0 / w2_global_scale),
            a2_scale=a2_scale,
        )
        # I=192 must remain unpadded for gated experts, but become 256 for
        # non-gated experts. The reference below uses the checkpoint shapes.
        expected_n = n if activation.is_gated else (n + 127) // 128 * 128
        assert moe_config.intermediate_size_per_partition == expected_n
        assert w1_cutedsl.shape[1] == expected_n * (2 if activation.is_gated else 1)
        assert w2_cutedsl.shape[2] * 2 == expected_n
        expected_k = (k + 255) // 256 * 256
        assert moe_config.hidden_dim == expected_k
        assert moe_config.hidden_dim_unpadded == k
        assert w1_cutedsl.shape[2] * 2 == expected_k
        assert w2_cutedsl.shape[1] == expected_k
        quant_config = make_nvfp4_moe_quant_config(
            backend=NvFp4MoeBackend.FLASHINFER_CUTEDSL,
            w13_scale_2=w1_alpha,
            w2_scale_2=w2_alpha,
            a13_scale=a1_scale,
            a2_scale=a2_scale,
            w13_scale=w1_scale_cutedsl,
            w2_scale=w2_scale_cutedsl,
            # Unset params must be omitted rather than forwarded as None into
            # the kernel's float-typed SwiGLU arguments.
            swiglu_alpha=alpha,
            swiglu_beta=beta,
            swiglu_limit=limit,
        )

        if resources is not None and activation.is_gated:
            from cuda.bindings import driver
            from torch.cuda._utils import _check_cuda_bindings as check

            for domain, shard in enumerate(fake_layer.localized_shards):
                for name in ("w13_weight", "w2_weight"):
                    weight = shard[name]
                    assert weight.untyped_storage().nbytes() > 0
                    handle = check(
                        driver.cuMemRetainAllocationHandle(weight.data_ptr())
                    )
                    try:
                        prop = check(
                            driver.cuMemGetAllocationPropertiesFromHandle(handle)
                        )
                        assert prop.location.type == (
                            driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
                        )
                        assert prop.location.localized.localityDomainId == domain
                        assert prop.location.localized.deviceId == weight.device.index
                    finally:
                        check(driver.cuMemRelease(handle))
            if localization == "free":
                assert w1_cutedsl.untyped_storage().nbytes() == 0
                assert w2_cutedsl.untyped_storage().nbytes() == 0
        else:
            assert not getattr(fake_layer, "localized_shards", None)
            assert w1_cutedsl.untyped_storage().nbytes() > 0
            assert w2_cutedsl.untyped_storage().nbytes() > 0

        experts = FlashInferCuteDSLExperts(
            moe_config, quant_config, per_token_activation=per_token_activation
        )
        fake_layer.w13_weight_scale_2 = w1_alpha
        fake_layer.w2_weight_scale_2 = w2_alpha
        fake_layer.w13_input_scale = a1_scale
        fake_layer.w2_input_scale = a2_scale
        experts.process_weights_after_loading(fake_layer)

        cutedsl_experts = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            experts,
        )

        # The model runner pads inputs and trims outputs around this kernel.
        padded_hidden_states = torch.nn.functional.pad(
            hidden_states, (0, expected_k - k)
        )

        def run():
            out = cutedsl_experts.apply(
                hidden_states=padded_hidden_states,
                w1=w1_cutedsl,
                w2=w2_cutedsl,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=e,
                expert_map=None,
                apply_router_weight_on_input=False,
            )
            assert isinstance(out, torch.Tensor)
            return out[:, :k]

        cutedsl_output = run()
        assert cutedsl_output.dtype == dtype
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = run()
        for _ in range(3):
            graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(captured_output, cutedsl_output)

        a_global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        if per_token_activation:
            a_q, a_scale, a_row_scale = quantize_nvfp4_per_token_input(hidden_states)
        else:
            a_q, a_scale = ops.scaled_fp4_quant(
                hidden_states,
                a_global_scale,
                is_sf_swizzled_layout=False,
            )
        a_in_dtype = _dequantize_nvfp4_linear(
            a_q,
            a_scale,
            a_global_scale,
            dtype=dtype,
        )
        if per_token_activation:
            a_in_dtype = (a_in_dtype.float() * a_row_scale.view(-1, 1)).to(dtype)

        w1_d = torch.empty((e, w1_rows, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)
        for idx in range(e):
            w1_d[idx] = _dequantize_nvfp4_linear(
                w1_q[idx],
                w1_scale[idx],
                w1_global_scale[idx],
                dtype=dtype,
            )
            w2_d[idx] = _dequantize_nvfp4_linear(
                w2_q[idx],
                w2_scale[idx],
                w2_global_scale[idx],
                dtype=dtype,
            )

        torch_output = _torch_moe_reference(
            a_in_dtype,
            w1_d,
            w2_d,
            score,
            topk,
            _reference_activation(activation, alpha, beta, limit),
        )
        torch.testing.assert_close(
            torch_output,
            cutedsl_output,
            atol=3e-2,
            rtol=2e-1,
        )
        # Outputs here are O(1e-2) while NVFP4 noise is O(1e-3), so an absolute
        # tolerance loose enough for the quantization error also accepts a zero
        # tensor. Compare direction too, which dropped SwiGLU params or a wrong
        # w13 layout would break.
        cosine = torch.nn.functional.cosine_similarity(
            cutedsl_output.flatten().float(), torch_output.flatten().float(), dim=0
        )
        assert cosine > 0.99, f"cosine similarity {cosine:.4f} below 0.99"


@pytest.mark.parametrize("localized", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("max_capture_size", [0, 256])
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.SITU])
def test_localization_arguments_initialized_after_loading(
    monkeypatch, localized, overlap, max_capture_size, activation
):
    import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe as mod

    scale = torch.ones(1)
    config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=2,
        num_logical_experts=2,
        activation=activation,
        activation_situ_beta=1.0 if activation == MoEActivation.SITU else None,
        activation_situ_linear_beta=0.5 if activation == MoEActivation.SITU else None,
        device="cpu",
        in_dtype=torch.bfloat16,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        routing_method=RoutingMethodType.TopK,
        max_capture_size=max_capture_size,
    )
    experts = FlashInferCuteDSLExperts(
        config,
        nvfp4_moe_quant_config(
            g1_alphas=scale,
            g2_alphas=scale,
            a1_gscale=scale,
            a2_gscale=scale,
            w1_scale=scale,
            w2_scale=scale,
        ),
    )
    shard = {
        name: torch.ones(1)
        for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    }
    layer = SimpleNamespace(
        w13_weight_scale_2=scale.clone(),
        w2_weight_scale_2=scale.clone(),
        w13_input_scale=scale,
        w2_input_scale=scale,
        localized_shards=[shard] if localized else [],
    )
    resources = SimpleNamespace(
        split_streams=[object()],
        split_sm_counts=[8],
        aux_stream=object(),
    )
    lookup = Mock(return_value=resources)
    monkeypatch.setattr(mod, "get_localized_resources", lookup)
    monkeypatch.setenv(
        "VLLM_FLASHINFER_CUTEDSL_MOE_LOCALIZE_MEMSET_OVERLAP", str(int(overlap))
    )
    monkeypatch.setenv("VLLM_FLASHINFER_CUTEDSL_MOE_LOCALIZE_FREE_FULL_WEIGHTS", "1")
    experts.process_weights_after_loading(layer)
    lookup.assert_called_once_with()
    lookup.side_effect = AssertionError(
        "apply must not discover localization resources"
    )
    backend = Mock()
    monkeypatch.setattr(mod, "flashinfer_cute_dsl_fused_moe_nvfp4", backend)
    experts.apply(
        output=scale,
        hidden_states=scale,
        w1=scale,
        w2=scale,
        topk_weights=scale,
        topk_ids=scale.int(),
        activation=activation,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=scale,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    kwargs = backend.call_args.kwargs
    if activation == MoEActivation.SITU:
        assert kwargs["situ_beta"] == 1.0
        assert kwargs["situ_linear_beta"] == 0.5
        assert kwargs["activation_type"] == mod.activation_to_flashinfer_int(
            MoEActivation.SILU
        )
    else:
        assert "situ_beta" not in kwargs
        assert "situ_linear_beta" not in kwargs
    assert kwargs["autotune_use_cuda_graph"] is (max_capture_size > 0)
    if localized:
        assert kwargs["localized_weights"][0]["w1_weight"] is shard["w13_weight"]
        assert kwargs["localized_streams"] is resources.split_streams
        assert kwargs["localized_sm_count"] == 8
        assert kwargs["localized_allow_nonlocalized"] is False
        assert ("localized_memset_stream" in kwargs) == overlap
        if overlap:
            assert kwargs["localized_memset_stream"] is resources.aux_stream
    else:
        assert not any(key.startswith("localized_") for key in kwargs)
