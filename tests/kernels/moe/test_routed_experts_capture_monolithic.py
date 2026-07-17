# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for routed-expert capture on the monolithic MoE path.

These tests exercise the wiring that lets ``RoutedExpertsCapturer`` see the
expert IDs picked by FlashInfer's fused router-and-experts kernels (the
"monolithic" path). When ``set_capture_fn`` is installed on
a ``FusedMoEExpertsMonolithic`` subclass that supports it, the kernel call
should:

  * allocate an int16 ``(num_tokens, top_k)`` buffer,
  * pass it to FlashInfer as ``routing_replay_out``,
  * invoke the callback after the kernel returns.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    TrtLlmBf16ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
    TrtLlmFp8ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.platforms import current_platform

try:
    from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
except ImportError:
    pytest.skip("flashinfer not available", allow_module_level=True)

if not has_flashinfer_trtllm_fused_moe() or not current_platform.is_cuda():
    pytest.skip(
        "Requires FlashInfer TRT-LLM fused MoE on CUDA",
        allow_module_level=True,
    )

if not current_platform.has_device_capability(100):
    pytest.skip(
        "TRT-LLM fused MoE kernels require SM100+",
        allow_module_level=True,
    )


def _shuffle_bf16_weights_block_major_k(
    w: torch.Tensor, epilogue_tile_m: int = 64, block_k: int = 128
) -> torch.Tensor:
    """Reshape ``w`` (E, M, K) into the ``BlockMajorK`` layout expected by
    ``trtllm_bf16_moe``: ``(E, K/block_k, M, block_k)`` after a per-expert
    row shuffle.
    """
    from flashinfer import shuffle_matrix_a
    from flashinfer.fused_moe import convert_to_block_layout

    num_experts = w.shape[0]
    shuffled = []
    for i in range(num_experts):
        t = shuffle_matrix_a(w[i].view(torch.uint8), epilogue_tile_m)
        shuffled.append(convert_to_block_layout(t, block_k))
    return torch.stack(shuffled).view(torch.bfloat16)


def _make_bf16_monolithic_experts(
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    routing_method: RoutingMethodType,
    device: torch.device,
) -> tuple[TrtLlmBf16ExpertsMonolithic, torch.Tensor, torch.Tensor]:
    """Construct the monolithic BF16 experts plus the BlockMajorK weights
    expected by ``trtllm_bf16_moe``.
    """
    parallel_cfg = FusedMoEParallelConfig.make_no_parallel()
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=parallel_cfg,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=routing_method,
        max_num_tokens=max(8, 1),
    )
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=None,
        per_act_token_quant=False,
        per_out_ch_quant=False,
        block_shape=None,
    )

    experts = TrtLlmBf16ExpertsMonolithic(
        moe_config=moe_config, quant_config=quant_config
    )

    gemm1 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    gemm2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    w13 = _shuffle_bf16_weights_block_major_k(gemm1)
    w2 = _shuffle_bf16_weights_block_major_k(gemm2)
    return experts, w13, w2


def _run_bf16_monolithic(
    experts: TrtLlmBf16ExpertsMonolithic,
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    *,
    n_group: int | None = None,
    topk_group: int | None = None,
    routed_scaling_factor: float | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return experts.apply(
        hidden_states=hidden_states,
        w1=w13,
        w2=w2,
        router_logits=router_logits,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        apply_router_weight_on_input=False,
        num_expert_group=n_group,
        topk_group=topk_group,
        e_score_correction_bias=e_score_correction_bias,
        routed_scaling_factor=routed_scaling_factor,
    )


_DSV3_NUM_EXPERTS = 32
_DSV3_N_GROUP = 4
_DSV3_TOPK_GROUP = 2


def _make_dsv3_routing_bias(num_experts: int, device: torch.device) -> torch.Tensor:
    return torch.randn(num_experts, device=device, dtype=torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [2, 7, 16])
@pytest.mark.parametrize("top_k", [2, 4])
def test_trtllm_bf16_monolithic_routing_replay_records_valid_experts(
    num_tokens: int,
    top_k: int,
) -> None:
    """The capture callback should receive the int16 routed-expert IDs the
    kernel actually used, the values should be valid expert indices, and
    each token should pick ``top_k`` distinct experts."""
    if top_k > _DSV3_N_GROUP * _DSV3_TOPK_GROUP:
        pytest.skip(
            f"DSV3 requires top_k <= n_group * topk_group "
            f"({_DSV3_N_GROUP * _DSV3_TOPK_GROUP})"
        )
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    num_experts = _DSV3_NUM_EXPERTS
    hidden_size = 1024
    intermediate_size = 1024

    experts, w13, w2 = _make_bf16_monolithic_experts(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        routing_method=RoutingMethodType.DeepSeekV3,
        device=device,
    )

    captured: list[torch.Tensor] = []

    def capture_fn(replay_out: torch.Tensor) -> None:
        captured.append(replay_out.clone())

    assert experts.supports_routing_replay_capture()
    experts.set_capture_fn(capture_fn)

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    )
    router_logits = torch.rand(
        num_tokens, num_experts, device=device, dtype=torch.float32
    )
    routing_bias = _make_dsv3_routing_bias(num_experts, device)

    _ = _run_bf16_monolithic(
        experts,
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        router_logits=router_logits,
        num_experts=num_experts,
        n_group=_DSV3_N_GROUP,
        topk_group=_DSV3_TOPK_GROUP,
        routed_scaling_factor=1.0,
        e_score_correction_bias=routing_bias,
    )

    assert len(captured) == 1
    replay = captured[0]
    assert replay.dtype == torch.int16
    assert replay.shape == (num_tokens, top_k)
    assert (replay >= 0).all(), f"got out-of-range values: {replay}"
    assert (replay < num_experts).all(), f"got out-of-range values: {replay}"

    for t in range(num_tokens):
        unique = replay[t].unique()
        assert unique.numel() == top_k, (
            f"token {t}: expected {top_k} distinct experts, "
            f"got {unique.numel()} ({replay[t].tolist()})"
        )


@pytest.mark.parametrize("num_tokens", [2, 7, 16])
@pytest.mark.parametrize(
    "routing_method",
    [
        RoutingMethodType.Renormalize,
        RoutingMethodType.RenormalizeNaive,
    ],
)
def test_trtllm_bf16_monolithic_routing_replay_non_dsv3(
    num_tokens: int,
    routing_method: RoutingMethodType,
) -> None:
    """Routing replay works for non-DeepSeekV3 routing methods too.
    FlashInfer's ``routing_replay_out`` is routing-method-agnostic."""
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    num_experts = 8
    top_k = 2
    hidden_size = 1024
    intermediate_size = 1024

    experts, w13, w2 = _make_bf16_monolithic_experts(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        routing_method=routing_method,
        device=device,
    )

    captured: list[torch.Tensor] = []
    experts.set_capture_fn(lambda r: captured.append(r.clone()))

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    )
    router_logits = torch.rand(
        num_tokens, num_experts, device=device, dtype=torch.float32
    )

    _ = _run_bf16_monolithic(
        experts,
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        router_logits=router_logits,
        num_experts=num_experts,
    )

    assert len(captured) == 1
    replay = captured[0]
    assert replay.dtype == torch.int16
    assert replay.shape == (num_tokens, top_k)
    assert (replay >= 0).all(), f"got out-of-range values: {replay}"
    assert (replay < num_experts).all(), f"got out-of-range values: {replay}"
    for t in range(num_tokens):
        unique = replay[t].unique()
        assert unique.numel() == top_k, (
            f"token {t}: expected {top_k} distinct experts, "
            f"got {unique.numel()} ({replay[t].tolist()})"
        )


def test_trtllm_bf16_monolithic_capture_disabled_skips_buffer_alloc() -> None:
    """With no callback installed the kernel should not see a
    ``routing_replay_out`` tensor — verify the helper short-circuits."""
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    experts, _, _ = _make_bf16_monolithic_experts(
        num_experts=_DSV3_NUM_EXPERTS,
        top_k=2,
        hidden_size=1024,
        intermediate_size=1024,
        routing_method=RoutingMethodType.DeepSeekV3,
        device=device,
    )
    # No callback installed.
    buf = experts._maybe_make_routing_replay_buffer(num_tokens=4, device=device)
    assert buf is None

    # Dispatch is also a no-op.
    experts._maybe_dispatch_routing_replay(buf, num_tokens=4)


def test_trtllm_bf16_monolithic_supports_capture_for_all_routing() -> None:
    """FlashInfer's ``routing_replay_out`` is supported by all routing
    methods, so ``supports_routing_replay_capture`` should be True
    regardless of routing method."""
    device = torch.device("cuda:0")
    for routing_method in (
        RoutingMethodType.DeepSeekV3,
        RoutingMethodType.Renormalize,
        RoutingMethodType.RenormalizeNaive,
    ):
        experts, _, _ = _make_bf16_monolithic_experts(
            num_experts=_DSV3_NUM_EXPERTS,
            top_k=2,
            hidden_size=1024,
            intermediate_size=1024,
            routing_method=routing_method,
            device=device,
        )
        assert experts.supports_routing_replay_capture() is True, (
            f"{routing_method!r} should support routing replay capture"
        )


def test_trtllm_bf16_monolithic_capture_buffer_shape_and_dtype() -> None:
    """When capture is installed, the allocated buffer is int16 and shaped
    ``(num_tokens, experts_per_token)``."""
    device = torch.device("cuda:0")
    experts, _, _ = _make_bf16_monolithic_experts(
        num_experts=_DSV3_NUM_EXPERTS,
        top_k=4,
        hidden_size=1024,
        intermediate_size=1024,
        routing_method=RoutingMethodType.DeepSeekV3,
        device=device,
    )
    experts.set_capture_fn(lambda r: None)
    buf = experts._maybe_make_routing_replay_buffer(num_tokens=11, device=device)
    assert buf is not None
    assert buf.dtype == torch.int16
    assert buf.shape[0] >= 11
    assert buf.shape[1] == 4
    assert buf.device.type == "cuda"


def test_routed_experts_capturer_e2e_via_monolithic_experts() -> None:
    """End-to-end: bind ``RoutedExpertsCapturer.capture`` as the callback
    on the monolithic experts and verify the captured rows land in the
    capturer's device buffer at the correct layer slot.

    Mirrors the wiring done in ``GPUModelRunner._bind_routed_experts_capturer``
    for the monolithic path: a single closure is installed on the monolithic
    ``fused_experts`` (in addition to ``router.set_capture_fn`` on the
    non-monolithic path) and the capturer routes per-layer based on the
    closed-over ``layer_id``.
    """
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
        RoutedExpertsCapturer,
    )

    torch.manual_seed(7)
    device = torch.device("cuda:0")
    num_tokens = 4
    top_k = 2
    num_experts = _DSV3_NUM_EXPERTS
    hidden_size = 1024
    intermediate_size = 1024

    experts, w13, w2 = _make_bf16_monolithic_experts(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        routing_method=RoutingMethodType.DeepSeekV3,
        device=device,
    )

    num_layers = 3
    layer_id = 1
    capturer = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    capturer.dp_rank = 0
    capturer.tp_size = 1
    capturer.device_buffer = torch.full(
        (num_tokens + 4, num_layers, top_k),
        -1,
        dtype=torch.int32,
        device=device,
    )

    def capture_fn(replay_out: torch.Tensor) -> None:
        capturer.capture(layer_id, replay_out)

    experts.set_capture_fn(capture_fn)

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    )
    router_logits = torch.rand(
        num_tokens, num_experts, device=device, dtype=torch.float32
    )
    routing_bias = _make_dsv3_routing_bias(num_experts, device)

    # Patch get_forward_context to return a dp_metadata=None context so the
    # capturer takes the single-DP branch.
    import vllm.model_executor.layers.fused_moe.routed_experts_capturer as rec

    with patch.object(
        rec,
        "get_forward_context",
        return_value=SimpleNamespace(dp_metadata=None),
    ):
        _ = _run_bf16_monolithic(
            experts,
            hidden_states=hidden_states,
            w13=w13,
            w2=w2,
            router_logits=router_logits,
            num_experts=num_experts,
            n_group=_DSV3_N_GROUP,
            topk_group=_DSV3_TOPK_GROUP,
            routed_scaling_factor=1.0,
            e_score_correction_bias=routing_bias,
        )

    captured = capturer.device_buffer[:num_tokens, layer_id, :].cpu()
    # Valid expert IDs at this layer.
    assert (captured >= 0).all()
    assert (captured < num_experts).all()
    for t in range(num_tokens):
        unique = captured[t].unique()
        assert unique.numel() == top_k, (
            f"token {t}: expected {top_k} distinct experts at layer "
            f"{layer_id}, got {unique.numel()}"
        )

    # Other layers / trailing token rows untouched.
    for other_layer in range(num_layers):
        if other_layer == layer_id:
            continue
        assert (capturer.device_buffer[:, other_layer, :].cpu() == -1).all(), (
            f"layer {other_layer} should be untouched, got writes"
        )
    assert (capturer.device_buffer[num_tokens:, layer_id, :].cpu() == -1).all(), (
        "tail rows beyond num_tokens should remain sentinel"
    )


# ----------------------------------------------------------------------------
# FP8 block-scale (DeepSeekFp8) — vLLM's ``TrtLlmFp8ExpertsMonolithic``
# ----------------------------------------------------------------------------


def _make_fp8_block_scale_monolithic_experts(
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[TrtLlmFp8ExpertsMonolithic, torch.Tensor, torch.Tensor]:
    """Set up ``TrtLlmFp8ExpertsMonolithic`` for the DeepSeekFp8 block-scale
    code path with DSV3 routing.

    Weights are shuffled into the BlockMajorK layout the kernel expects
    (same helper the vLLM weight loader uses for DeepSeek-FP8 models).
    """
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _shuffle_deepseek_fp8_moe_weights,
    )

    block_k = 128
    parallel_cfg = FusedMoEParallelConfig.make_no_parallel()
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=parallel_cfg,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.DeepSeekV3,
        max_num_tokens=max(8, 1),
    )

    # Random fp8 weights + ones-block scales (the kernel decoder only cares
    # that the per-block scales are present and finite for routing/replay).
    gemm1 = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, device=device
    ).to(torch.float8_e4m3fn)
    gemm2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
        torch.float8_e4m3fn
    )
    w13_shuffled, w2_shuffled = _shuffle_deepseek_fp8_moe_weights(gemm1, gemm2)

    w1_scale = torch.ones(
        num_experts,
        2 * intermediate_size // block_k,
        hidden_size // block_k,
        device=device,
        dtype=torch.float32,
    )
    w2_scale = torch.ones(
        num_experts,
        hidden_size // block_k,
        intermediate_size // block_k,
        device=device,
        dtype=torch.float32,
    )
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=[block_k, block_k],
        per_act_token_quant=False,
    )

    experts = TrtLlmFp8ExpertsMonolithic(
        moe_config=moe_config, quant_config=quant_config
    )
    return experts, w13_shuffled, w2_shuffled


def _run_fp8_block_scale_monolithic(
    experts: TrtLlmFp8ExpertsMonolithic,
    hidden_states_fp8: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    routing_bias: torch.Tensor,
) -> torch.Tensor:
    return experts.apply(
        hidden_states=hidden_states_fp8,
        w1=w13,
        w2=w2,
        router_logits=router_logits,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        # The block-scale apply path reads ``a1q_scale`` and transposes it
        # to ``(hidden_size/128, num_tokens)`` for the kernel call.
        a1q_scale=hidden_states_scale,
        apply_router_weight_on_input=False,
        num_expert_group=_DSV3_N_GROUP,
        topk_group=_DSV3_TOPK_GROUP,
        e_score_correction_bias=routing_bias,
        routed_scaling_factor=1.0,
    )


@pytest.mark.parametrize("num_tokens", [2, 7, 16])
@pytest.mark.parametrize("top_k", [2, 4])
def test_trtllm_fp8_block_scale_monolithic_routing_replay_records_valid_experts(
    num_tokens: int,
    top_k: int,
) -> None:
    """End-to-end: ``TrtLlmFp8ExpertsMonolithic`` (DeepSeekFp8 block-scale
    path, DSV3 routing) captures valid expert IDs."""
    if top_k > _DSV3_N_GROUP * _DSV3_TOPK_GROUP:
        pytest.skip(
            f"DSV3 requires top_k <= n_group * topk_group "
            f"({_DSV3_N_GROUP * _DSV3_TOPK_GROUP})"
        )
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    num_experts = _DSV3_NUM_EXPERTS
    hidden_size = 1024
    intermediate_size = 1024

    experts, w13, w2 = _make_fp8_block_scale_monolithic_experts(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    assert experts.supports_routing_replay_capture()

    captured: list[torch.Tensor] = []
    experts.set_capture_fn(lambda r: captured.append(r.clone()))

    # Per-token / per-block hidden scales (ones is fine for the routing
    # path; the GEMM output isn't being asserted on).
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    ).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.ones(
        num_tokens, hidden_size // 128, device=device, dtype=torch.float32
    )
    router_logits = torch.rand(
        num_tokens, num_experts, device=device, dtype=torch.float32
    )
    routing_bias = _make_dsv3_routing_bias(num_experts, device)

    _ = _run_fp8_block_scale_monolithic(
        experts,
        hidden_states_fp8=hidden_states,
        hidden_states_scale=hidden_states_scale,
        w13=w13,
        w2=w2,
        router_logits=router_logits,
        num_experts=num_experts,
        routing_bias=routing_bias,
    )

    assert len(captured) == 1
    replay = captured[0]
    assert replay.dtype == torch.int16
    assert replay.shape == (num_tokens, top_k)
    assert (replay >= 0).all(), f"got out-of-range values: {replay}"
    assert (replay < num_experts).all(), f"got out-of-range values: {replay}"
    for t in range(num_tokens):
        unique = replay[t].unique()
        assert unique.numel() == top_k, (
            f"token {t}: expected {top_k} distinct experts, "
            f"got {unique.numel()} ({replay[t].tolist()})"
        )


# ----------------------------------------------------------------------------
# NVFP4 — vLLM's ``TrtLlmNvFp4ExpertsMonolithic``
# ----------------------------------------------------------------------------


def _make_nvfp4_monolithic_experts(
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[
    TrtLlmNvFp4ExpertsMonolithic,
    torch.Tensor,  # w13 (packed nvfp4 uint8)
    torch.Tensor,  # w13 block-scale (fp8)
    torch.Tensor,  # w2  (packed nvfp4 uint8)
    torch.Tensor,  # w2  block-scale (fp8)
    torch.Tensor,  # input global scale (per-tensor float32)
]:
    """Set up ``TrtLlmNvFp4ExpertsMonolithic`` with NVFP4-quantized weights
    and DSV3 routing.

    NVFP4 = per-block-of-16 fp4 with an fp8 scale, plus a per-tensor
    "global" scale. We follow the layout in ``test_ocp_mx_moe.py`` /
    ``flashinfer/tests/moe/test_trtllm_gen_routed_fused_moe.py``:
      * weights: uint8 (packed fp4) ``(E, M, K//2)``
      * weight scales: fp8 ``(E, M, K//16)``
      * hidden states: uint8 (packed fp4) ``(N, K//2)``
      * hidden state scales: fp8 ``(N, K//16)``
    """
    from flashinfer import fp4_quantize

    block_size = 16
    parallel_cfg = FusedMoEParallelConfig.make_no_parallel()
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=parallel_cfg,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.DeepSeekV3,
        max_num_tokens=max(8, 1),
    )

    gemm1 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    )
    # Per-tensor weight scaling factor (used to build ``g1_alphas`` /
    # ``g2_alphas`` below).
    w_global_scale = torch.tensor(1.0, device=device)
    # Per-tensor input scaling factor.
    a_global_scale = torch.tensor(1.0, device=device)

    w13_q, w13_scale = fp4_quantize(
        gemm1,
        w_global_scale,
        block_size,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * intermediate_size, hidden_size // block_size
    )
    w2_q, w2_scale = fp4_quantize(
        gemm2,
        w_global_scale,
        block_size,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, hidden_size, intermediate_size // block_size
    )

    # NVFP4 dq scale chain: g1_alphas = w1_scale_2 * a1_scale_2,
    # g2_alphas = w2_scale_2 * a2_scale_2. The kernel multiplies by these.
    g_alphas = torch.full((num_experts,), 1.0, device=device, dtype=torch.float32)
    a2_gscale = torch.full((num_experts,), 1.0, device=device, dtype=torch.float32)

    quant_config = FusedMoEQuantConfig.make(
        quant_dtype="nvfp4",
        per_act_token_quant=False,
        per_out_ch_quant=False,
        block_shape=None,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        g1_alphas=g_alphas,
        g2_alphas=g_alphas,
        a1_gscale=a_global_scale,
        a2_gscale=a2_gscale,
    )

    experts = TrtLlmNvFp4ExpertsMonolithic(
        moe_config=moe_config, quant_config=quant_config
    )
    return experts, w13_q, w13_scale, w2_q, w2_scale, a_global_scale


def _run_nvfp4_monolithic(
    experts: TrtLlmNvFp4ExpertsMonolithic,
    hidden_states_q: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    router_logits: torch.Tensor,
    num_experts: int,
    routing_bias: torch.Tensor,
) -> torch.Tensor:
    """The monolithic NVFP4 apply expects packed fp4 hidden states + the
    matching fp8 per-block scale stored in the ``a1q_scale`` slot."""
    # Stash the weight tensors on the experts in the locations the apply()
    # implementation reads from (it pulls them from quant_config / scales
    # already; w1/w2 come in as args).
    return experts.apply(
        hidden_states=hidden_states_q,
        w1=experts._w13_packed,
        w2=experts._w2_packed,
        router_logits=router_logits,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=hidden_states_scale,
        apply_router_weight_on_input=False,
        num_expert_group=_DSV3_N_GROUP,
        topk_group=_DSV3_TOPK_GROUP,
        e_score_correction_bias=routing_bias,
        routed_scaling_factor=1.0,
    )


@pytest.mark.parametrize("num_tokens", [2, 7, 16])
@pytest.mark.parametrize("top_k", [2, 4])
def test_trtllm_nvfp4_monolithic_routing_replay_records_valid_experts(
    num_tokens: int,
    top_k: int,
) -> None:
    """End-to-end: ``TrtLlmNvFp4ExpertsMonolithic`` captures valid expert IDs
    on the DSV3 routing path."""
    if top_k > _DSV3_N_GROUP * _DSV3_TOPK_GROUP:
        pytest.skip(
            f"DSV3 requires top_k <= n_group * topk_group "
            f"({_DSV3_N_GROUP * _DSV3_TOPK_GROUP})"
        )
    from flashinfer import fp4_quantize

    torch.manual_seed(0)
    device = torch.device("cuda:0")

    num_experts = _DSV3_NUM_EXPERTS
    hidden_size = 1024
    intermediate_size = 1024
    block_size = 16

    experts, w13_q, _w13_s, w2_q, _w2_s, a_gs = _make_nvfp4_monolithic_experts(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    # The apply() reads w1/w2 from its args, but we keep them on the experts
    # for convenience of the helper.
    experts._w13_packed = w13_q
    experts._w2_packed = w2_q

    assert experts.supports_routing_replay_capture()
    captured: list[torch.Tensor] = []
    experts.set_capture_fn(lambda r: captured.append(r.clone()))

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    )
    hidden_states_q, hidden_states_scale = fp4_quantize(
        hidden_states,
        a_gs,
        block_size,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    # The vLLM apply() does the .view(fp8_e4m3fn).reshape itself, so leave
    # ``hidden_states_scale`` in its native (uint8 packed) form.
    router_logits = torch.rand(
        num_tokens, num_experts, device=device, dtype=torch.float32
    )
    routing_bias = _make_dsv3_routing_bias(num_experts, device)

    _ = _run_nvfp4_monolithic(
        experts,
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        router_logits=router_logits,
        num_experts=num_experts,
        routing_bias=routing_bias,
    )

    assert len(captured) == 1
    replay = captured[0]
    assert replay.dtype == torch.int16
    assert replay.shape == (num_tokens, top_k)
    assert (replay >= 0).all(), f"got out-of-range values: {replay}"
    assert (replay < num_experts).all(), f"got out-of-range values: {replay}"
    for t in range(num_tokens):
        unique = replay[t].unique()
        assert unique.numel() == top_k, (
            f"token {t}: expected {top_k} distinct experts, "
            f"got {unique.numel()} ({replay[t].tolist()})"
        )
