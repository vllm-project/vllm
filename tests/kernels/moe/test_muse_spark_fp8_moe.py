# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark online FP8 routed experts (`fp8_per_channel`) on Triton."""

import pytest
import torch

from tests.kernels.moe.muse_spark_utils import (
    assert_cudagraph_replays,
    make_runner,
    run_moe,
    swiglu_moe_ref,
)
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineMoEMethod,
)
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)
from vllm.models.muse_spark.common.layers import MuseSparkRMSNormWithInputScale
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MuseSparkFp8FusedMoE,
    MuseSparkRoutedExperts,
    MuseSparkUnquantizedFusedMoE,
)
from vllm.platforms import current_platform

requires_fp8_triton = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(89)),
    reason="Triton FP8 w8a8 MoE needs an SM89+ NVIDIA GPU",
)

E, N, K, TOPK, M = 8, 256, 512, 2, 8
PREFIX = "moe.experts"


def _online_config(**kwargs) -> OnlineQuantizationConfig:
    return OnlineQuantizationConfig(QuantizationConfigArgs(**kwargs))


def test_moe_only_config_resolves_experts_to_fp8_per_channel():
    config = _online_config(moe="fp8_per_channel")
    resolved = config.resolve_quant_method_cls(MuseSparkRoutedExperts, PREFIX)
    assert resolved is not None and resolved[-1] is Fp8PtpcOnlineMoEMethod
    ignored = _online_config(moe="fp8_per_channel", ignore=[PREFIX])
    assert ignored.resolve_quant_method_cls(MuseSparkRoutedExperts, PREFIX) is None


def _norm() -> MuseSparkRMSNormWithInputScale:
    return MuseSparkRMSNormWithInputScale(
        K,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    )


def test_ignored_experts_stay_bf16(dist_init):
    quant_config = _online_config(moe="fp8_per_channel", ignore=[PREFIX])
    _, runner, _ = make_runner(
        _norm(),
        torch.zeros(E),
        None,
        quant_config,
        hidden_size=K,
        intermediate_size=N,
        top_k=TOPK,
    )
    assert type(runner.routed_experts.quant_method) is MuseSparkUnquantizedFusedMoE


def _checkpoint(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda", generator=g)

    return {"w13_weight": rand(E, 2 * N, K) / 10, "w2_weight": rand(E, K, N) / 10}


def _build(checkpoint: dict[str, torch.Tensor], quant_config):
    norm = _norm().cuda()
    norm.weight.copy_(torch.linspace(-0.2, 0.2, K, device="cuda"))
    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")
    with torch.device("cuda"):
        router, runner, vllm_config = make_runner(
            norm,
            expert_bias,
            checkpoint,
            quant_config,
            hidden_size=K,
            intermediate_size=N,
            top_k=TOPK,
        )
    return router, norm, runner, vllm_config


def _dequantized(routed) -> tuple[torch.Tensor, torch.Tensor]:
    w13 = routed.w13_weight.float() * routed.w13_weight_scale
    w2 = routed.w2_weight.float() * routed.w2_weight_scale
    return w13.to(torch.bfloat16), w2.to(torch.bfloat16)


def _rel_to_reference(router, norm, runner, vllm_config, x, logits, norm_fn=None):
    """Relative distance of the layer output from an fp32 MoE on the layer's
    own (dequantized) FP8 weights."""
    out = run_moe(runner, vllm_config, x, logits)
    topk_weights, topk_ids = router._compute_routing(x, logits, None)
    w13, w2 = _dequantized(runner.routed_experts)
    if norm_fn is None:

        def norm_fn(y):
            return norm.forward_native(y, out_dtype=torch.float32)

    ref = swiglu_moe_ref(x, w13, w2, topk_weights, topk_ids, norm_fn)
    return ((out.float() - ref).norm() / ref.norm()).item()


@requires_fp8_triton
@torch.inference_mode()
def test_fp8_experts_match_reference_with_post_expert_norm(dist_init, workspace_init):
    router, norm, runner, vllm_config = _build(
        _checkpoint(0), _online_config(moe="fp8_per_channel")
    )
    routed = runner.routed_experts
    assert type(routed.quant_method) is MuseSparkFp8FusedMoE
    kernel = routed.quant_method.moe_kernel
    assert kernel is not None and type(kernel.fused_experts) is TritonExperts
    assert routed.w13_weight.dtype == current_platform.fp8_dtype()

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 10
    logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    # Only the dynamic per-token activation quantization separates the two.
    assert _rel_to_reference(router, norm, runner, vllm_config, x, logits) < 5e-2
    # The norm runs on each expert's output, before the top-k reduction.
    unnormed = _rel_to_reference(
        router, norm, runner, vllm_config, x, logits, norm_fn=lambda y: y
    )
    assert unnormed > 1e-1

    assert_cudagraph_replays(
        runner, vllm_config, x, logits, torch.randn_like(x), torch.randn_like(logits)
    )


@requires_fp8_triton
@torch.inference_mode()
def test_layerwise_reload_requantizes_in_place(dist_init, workspace_init):
    """An RL weight update re-quantizes the new BF16 weights into the tensors
    captured CUDA graphs read."""
    router, norm, runner, vllm_config = _build(
        _checkpoint(0), _online_config(moe="fp8_per_channel")
    )
    routed = runner.routed_experts
    before = _dequantized(routed)
    ptrs = {n: p.data_ptr() for n, p in routed.named_parameters()}

    initialize_layerwise_reload(routed)
    list(routed.load_weights(list(_checkpoint(1).items())))
    finalize_layerwise_reload(routed, None)

    assert {n: p.data_ptr() for n, p in routed.named_parameters()} == ptrs
    kernel = routed.quant_method.moe_kernel
    assert kernel is not None
    assert kernel.fused_experts.post_gemm2_hook is runner.post_expert_finalize
    assert not torch.equal(_dequantized(routed)[0], before[0])
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 10
    logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    assert _rel_to_reference(router, norm, runner, vllm_config, x, logits) < 5e-2


def test_other_online_moe_formats_are_rejected(dist_init):
    with pytest.raises(NotImplementedError, match="fp8_per_channel only"):
        make_runner(
            _norm(),
            torch.zeros(E),
            None,
            _online_config(moe="fp8_per_tensor"),
            hidden_size=K,
            intermediate_size=N,
            top_k=TOPK,
        )


@requires_fp8_triton
@torch.inference_mode()
def test_direct_load_weights_after_init_is_refused(dist_init, workspace_init):
    """A direct load_weights would cast BF16 into the FP8 experts unscaled."""
    _, _, runner, _ = _build(_checkpoint(0), _online_config(moe="fp8_per_channel"))
    with pytest.raises(RuntimeError, match="layerwise reload"):
        list(runner.routed_experts.load_weights(list(_checkpoint(1).items())))
