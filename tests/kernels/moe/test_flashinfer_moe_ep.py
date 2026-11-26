# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the FlashInfer MoE-EP adapter on Hopper and Blackwell GPUs.

Each test spawns one process per GPU, bootstraps vLLM's expert-parallel
groups, builds a `FlashInferMoeEp` adapter from bf16 expert weights at the
MoE geometry of a real model, warms it up and runs a forward whose inputs are
chosen so that every quantization step in the megakernel is exact, for the
CuTeDSL kernel (NVFP4 activations and weights, E4M3 scales per 16) as well as
the DeepGEMM kernel (FP8 activations and FP4 weights, power-of-two scales per
32). The expected output is then known in closed form and compared bit for
bit.

On SM107 the CuTeDSL kernel instead takes checkpoint-format NVFP4 experts and
the epilogue calibrated from their activation scales, and checkpoint-format
MXFP4 experts with MXFP8 activations, again exactly.

On Hopper the same exact forward runs the SM90 megakernels on block-FP8
weights (power-of-two scales per 128x128 block, FP8 activations per 1x128)
and on packed MXFP4 weights (unit E8M0 scales, FP8 activations per token).

Tests that need no GPU also check the adapter's kwargs against the installed
FlashInfer's config types, so config drift fails on any CI runner with
FlashInfer installed.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.distributed.eplb_utils import distributed_run, set_env_vars_and_device
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.kernel import FLASHINFER_MOE_EP_CUTEDSL, FLASHINFER_MOE_EP_DEEP_GEMM
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    set_custom_all_reduce,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep as fi_ep
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.experts.flashinfer_moe_ep import (
    FlashInferMoeEpExperts,
    sm107_nvfp4_epilogue,
)
from vllm.model_executor.layers.fused_moe.flashinfer_moe_ep import (
    FlashInferMoeEp,
    FlashInferMoeEpEpilogue,
    FlashInferMoeEpWeights,
    apply_topk_in_fc1,
    mxfp4_moe_ep_weights,
    sm90_fp8_block_moe_ep_weights,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.passthrough import (
    PassThroughPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulatorRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm
from vllm.v1.worker.workspace import init_workspace_manager


@dataclass(frozen=True, slots=True)
class Geometry:
    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int


# Routed-expert geometry of the models the branch was evaluated on. DeepGEMM
# keeps one UE8M0 scale byte per 32 elements and requires the per-token scale
# rows to be TMA aligned, so hidden and intermediate sizes must be multiples
# of 512 (FlashInfer's own validation only checks 128); all four qualify.
GEOMETRIES = (
    Geometry("dsv4-pro", 7168, 3072, 384, 6),
    Geometry("dsv4-flash", 4096, 2048, 256, 6),
    Geometry("mistral-small-4", 4096, 2048, 128, 4),
    Geometry("mistral-large-3", 7168, 4096, 128, 4),
)
BACKENDS = (FLASHINFER_MOE_EP_CUTEDSL, FLASHINFER_MOE_EP_DEEP_GEMM)
# Routed-expert geometries of recent MoE models (Kimi-K3 at its 3584-wide
# latent), run with synthetic block-FP8 or MXFP4 weights on the Hopper
# megakernels.
SM90_GEOMETRIES = (
    Geometry("glm-5.3-flash", 4096, 2048, 288, 8),
    Geometry("deepseek-v4.1-flash", 5120, 2304, 384, 6),
    Geometry("kimi-k3", 3584, 3072, 896, 16),
    Geometry("qwen3.8-flash-next", 2560, 640, 512, 10),
)

NUM_TOKENS_PER_RANK = 4
MAX_TOKENS_PER_RANK = 128

# Every block scale below is either 1 or a power of two, so the FP4 codes
# and the FP8 values survive quantization exactly: NVFP4 scales a 16-block by
# absmax / 6 in E4M3, DeepGEMM scales a 32-block by absmax / 6 (FP4) or
# absmax / 448 (FP8) rounded up to a power of two. A 6.0 in a block pins the
# FP4 scale to 1; 2.0, 6.0 and 144 are exact FP8 values at any such scale.
#
# Token t carries X_SIGNAL at hidden column 16 * t and a 6.0 sentinel next to
# it; every weight column the sentinel meets is zero.
X_SIGNAL = 2.0
X_SENTINEL = 6.0
# FP8 activations take absmax / 448 per 1x128 group: 448 pins that scale to 1.
X_SENTINEL_FP8 = 448.0
# Gate and up rows are identical, so h = silu(12) * 12 = 144 = 6 * 24
# whichever half the kernel treats as the gate, and 144 requantizes exactly.
W13_DIAG = 6.0
H_SIGNAL = 144.0
# Expert e writes E2M1_CODES[e % 7] * H_SIGNAL into output channel e. The 6.0
# sentinel sits next to a zero intermediate channel and pins the block scale.
W2_SENTINEL = 6.0
E2M1_CODES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# The SM107 test feeds checkpoint-format NVFP4 experts with distinct gate (6)
# and up (1) rows, so h = silu(12) * 2 requantizes to 24, whereas swapped
# halves would give silu(2) * 12, which requantizes to 21.
SM107_W13_GATE = 6.0
SM107_W13_UP = 1.0
SM107_H_SIGNAL = 24.0
SM107_ROUTE_CHECK_ENV = "FLASHINFER_SM107_MEGA_CHECK_ROUTES"
SM107_CASES = (
    (Geometry("dsv4.1-flash", 5120, 2304, 384, 6), RoutingMethodType.DeepseekV4),
    (GEOMETRIES[1], RoutingMethodType.Default),
)

# The SM107 MXFP4 test feeds checkpoint-format MXFP4 experts (E2M1 codes, E8M0
# scales per 32) that dequantize to gate 12 and up 0.5. Against x = 2 the
# kernel's sigmoid(24) is exactly 1 in fp32, so h = 24 with no rounding, while
# swapped halves give 24 * sigmoid(1), which requantizes to 18.
SM107_MX_GATE = 12.0
SM107_MX_UP = 0.5
# E8M0 exponent of every scale that only meets zero codes, so a scale read from
# the wrong expert, row or block shows in the output.
SM107_MX_BACKGROUND_EXPONENT = 4
# Token t's intermediate channel. An MXFP8 scale block spans two epilogue warps,
# and 48 and 304 sit in the half whose scale the other warp stores; 160 and 304
# are in later 16-row gate/up groups and tiles.
SM107_MX_CHANNELS = (0, 48, 160, 304)
# DeepSeek-V4.1's SwiGLU limit clamps the gate to 10, and h = 10 * sigmoid(10)
# requantizes to 10; without the clamp h stays 24, swapped halves give 7.5.
SM107_MX_CLAMP = 10.0
SM107_MX_CASES = (
    *((geometry, routing, None) for geometry, routing in SM107_CASES),
    # The MTP draft layers of DeepSeek-V4.1-Flash-NVFP4 are MXFP4.
    (
        Geometry("dsv4.1-flash-mtp", 5120, 2304, 128, 3),
        RoutingMethodType.DeepseekV4,
        SM107_MX_CLAMP,
    ),
)


def _expert_out(expert: int) -> float:
    return E2M1_CODES[expert % len(E2M1_CODES)]


def _topk_weights(top_k: int) -> tuple[float, ...]:
    # The DeepSeek-V4 routing policy applies the router weight before fc2, so
    # weight * 144 must requantize exactly too: powers of two do.
    return tuple(2.0 ** -(k + 1) for k in range(top_k))


def _routing_method(backend: str, sm90_format: str | None = None) -> RoutingMethodType:
    # DeepSeek-V4 models are the MXFP4 checkpoints Hopper serves.
    if backend == FLASHINFER_MOE_EP_DEEP_GEMM or sm90_format == "mxfp4":
        return RoutingMethodType.DeepseekV4
    return RoutingMethodType.Default


def _routing(
    geometry: Geometry, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """top_k distinct experts per token, strided across ranks so dispatch goes
    remote."""
    base = (
        torch.arange(NUM_TOKENS_PER_RANK, device=device) + rank * NUM_TOKENS_PER_RANK
    ) % geometry.num_experts
    stride = geometry.num_experts // geometry.top_k
    offsets = torch.arange(geometry.top_k, device=device) * stride
    topk_ids = ((base[:, None] + offsets[None, :]) % geometry.num_experts).to(
        torch.int32
    )
    topk_weights = torch.tensor(
        [_topk_weights(geometry.top_k)] * NUM_TOKENS_PER_RANK,
        dtype=torch.float32,
        device=device,
    )
    return topk_ids, topk_weights


def _hidden_states(
    geometry: Geometry, device: torch.device, sentinel: float = X_SENTINEL
) -> torch.Tensor:
    x = torch.zeros(
        NUM_TOKENS_PER_RANK, geometry.hidden_size, dtype=torch.bfloat16, device=device
    )
    for t in range(NUM_TOKENS_PER_RANK):
        x[t, 16 * t] = X_SIGNAL
        x[t, 16 * t + 1] = sentinel
    return x


def _local_expert_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> FlashInferMoeEpWeights:
    w13 = torch.zeros(
        num_local_experts,
        2 * geometry.intermediate_size,
        geometry.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.zeros(
        num_local_experts,
        geometry.hidden_size,
        geometry.intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    token_columns = [16 * t for t in range(NUM_TOKENS_PER_RANK)]
    for local in range(num_local_experts):
        expert = rank * num_local_experts + local
        w13[local, 0, token_columns] = W13_DIAG
        w13[local, geometry.intermediate_size, token_columns] = W13_DIAG
        w2[local, expert, 0] = _expert_out(expert)
        w2[local, expert, 2] = W2_SENTINEL
    return FlashInferMoeEpWeights(w13=w13, w2=w2)


def _unit_scale(expert: int) -> float:
    return 1.0


def _block_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block FP8 with power-of-two scales: exact for the grid weights."""
    experts, rows, cols = weight.shape
    blocks = weight.float().view(experts, rows // 128, 128, cols // 128, 128)
    absmax = blocks.abs().amax(dim=(2, 4))
    scale = torch.where(
        absmax > 0,
        torch.exp2(torch.ceil(torch.log2(absmax / 448.0))),
        torch.ones_like(absmax),
    )
    quantized = blocks / scale[:, :, None, :, None]
    return quantized.view(experts, rows, cols).to(torch.float8_e4m3fn), scale


def _local_block_fp8_expert_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> FlashInferMoeEpWeights:
    """The grid weights in block FP8, in the oracle's SM90 kernel layout."""
    weights = _local_expert_weights(geometry, rank, num_local_experts, device)
    w13, w13_scale = sm90_fp8_block_moe_ep_weights(*_block_fp8(weights.w13))
    w2, w2_scale = _block_fp8(weights.w2)
    return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale)


def _mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack E2M1 grid weights with unit E8M0 scales."""
    magnitudes = torch.tensor((0.0, *E2M1_CODES), device=weight.device)
    experts, rows, cols = weight.shape
    packed = torch.empty(
        experts, rows, cols // 2, dtype=torch.uint8, device=weight.device
    )
    # Bound the float32 values and int64 codes to one expert.
    for expert in range(experts):
        values = weight[expert].float()
        codes = torch.searchsorted(magnitudes, values)
        assert torch.equal(magnitudes[codes], values)
        packed[expert] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    # E8M0 127 encodes 2**0.
    scale = torch.full(
        (experts, rows, cols // 32), 127, dtype=torch.uint8, device=weight.device
    )
    return packed, scale


def _local_mxfp4_expert_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> FlashInferMoeEpWeights:
    """The grid weights as an MXFP4 checkpoint stores them."""
    weights = _local_expert_weights(geometry, rank, num_local_experts, device)
    w13, w13_scale = _mxfp4(weights.w13)
    w2, w2_scale = _mxfp4(weights.w2)
    return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale)


def _expected_output(
    geometry: Geometry,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    h_signal: float = H_SIGNAL,
    expert_scale: Callable[[int], float] = _unit_scale,
) -> torch.Tensor:
    expected = torch.zeros(
        NUM_TOKENS_PER_RANK,
        geometry.hidden_size,
        dtype=torch.float32,
        device=topk_ids.device,
    )
    for t in range(NUM_TOKENS_PER_RANK):
        for k in range(geometry.top_k):
            expert = int(topk_ids[t, k])
            weight = float(topk_weights[t, k])
            expected[t, expert] += (
                weight * _expert_out(expert) * h_signal * expert_scale(expert)
            )
    return expected.to(torch.bfloat16)


def _e2m1_code(value: float) -> int:
    return fi_ep._E2M1_LUT.index(value)


def _sm107_expert_scale(expert: int) -> float:
    return 2.0 ** -(expert % 3)


def _local_expert_nvfp4_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> tuple[FlashInferMoeEpWeights, torch.Tensor, torch.Tensor]:
    """Checkpoint-format NVFP4 experts and their weight global scales.

    Every expert dequantizes to the same w13, but its block scale 2**j and
    global scale 2**-j differ per expert, so a misindexed fc1 alpha breaks
    exactness. Expert e's w2 global scale scales output channel e. Even
    columns are the low nibbles.
    """
    hidden, intermediate = geometry.hidden_size, geometry.intermediate_size
    experts = [rank * num_local_experts + local for local in range(num_local_experts)]
    w13 = torch.zeros(
        num_local_experts,
        2 * intermediate,
        hidden // 2,
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.zeros(
        num_local_experts, hidden, intermediate // 2, dtype=torch.uint8, device=device
    )
    token_bytes = [16 * t // 2 for t in range(NUM_TOKENS_PER_RANK)]
    w13[:, 0, token_bytes] = _e2m1_code(SM107_W13_GATE)
    w13[:, intermediate, token_bytes] = _e2m1_code(SM107_W13_UP)
    for local, expert in enumerate(experts):
        w2[local, expert, 0] = _e2m1_code(_expert_out(expert))
        w2[local, expert, 1] = _e2m1_code(W2_SENTINEL)

    w13_block_scale = torch.tensor(
        [2.0 ** (expert % 4) for expert in experts], device=device
    )
    w13_scale = (
        w13_block_scale.view(-1, 1, 1)
        .expand(num_local_experts, 2 * intermediate, hidden // 16)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    w2_scale = (
        torch.ones((), device=device)
        .expand(num_local_experts, hidden, intermediate // 16)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    g1 = 1.0 / w13_block_scale
    g2 = torch.tensor(
        [_sm107_expert_scale(expert) for expert in experts],
        dtype=torch.float32,
        device=device,
    )
    return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale), g1, g2


def _e8m0_byte(exponent: int) -> int:
    return 127 + exponent


def _local_expert_mxfp4_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Checkpoint-format MXFP4 experts, as `Mxfp4MoEMethod` stores them.

    uint8 E2M1 `w13 [E, 2I, H/2]` and `w2 [E, H, I/2]` (even columns are
    the low nibbles) and uint8 E8M0 scale bytes `[E, 2I, H/32]` and
    `[E, H, I/32]`. Token t reaches only gate row c and up row I + c, with
    c = SM107_MX_CHANNELS[t]. With j = e % 3, expert e's gate rows hold code
    6 / 2**j at scale 2**(j + 1) and its up rows code 4 / 2**j at scale
    2**(j - 3): every expert dequantizes to the same w13, but a scale read from
    another expert or the other half, or a value cast of the scale bytes,
    breaks exactness. Expert e writes E2M1_CODES[e % 7] * 2**-j into output
    channel e.
    """
    hidden, intermediate = geometry.hidden_size, geometry.intermediate_size
    background = _e8m0_byte(SM107_MX_BACKGROUND_EXPONENT)
    w13 = torch.zeros(
        num_local_experts,
        2 * intermediate,
        hidden // 2,
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.zeros(
        num_local_experts, hidden, intermediate // 2, dtype=torch.uint8, device=device
    )
    w13_scale = torch.full(
        (num_local_experts, 2 * intermediate, hidden // 32),
        background,
        dtype=torch.uint8,
        device=device,
    )
    w2_scale = torch.full(
        (num_local_experts, hidden, intermediate // 32),
        background,
        dtype=torch.uint8,
        device=device,
    )
    for local in range(num_local_experts):
        expert = rank * num_local_experts + local
        j = expert % 3
        gate_code = _e2m1_code(SM107_MX_GATE / 2 ** (j + 1))
        up_code = _e2m1_code(SM107_MX_UP / 2 ** (j - 3))
        for t, channel in enumerate(SM107_MX_CHANNELS):
            column = 16 * t
            w13[local, channel, column // 2] = gate_code
            w13[local, intermediate + channel, column // 2] = up_code
            w13_scale[local, channel, column // 32] = _e8m0_byte(j + 1)
            w13_scale[local, intermediate + channel, column // 32] = _e8m0_byte(j - 3)
            w2[local, expert, channel // 2] = _e2m1_code(_expert_out(expert))
            w2_scale[local, expert, channel // 32] = _e8m0_byte(-j)
        w2[local, expert, 1] = _e2m1_code(W2_SENTINEL)
    return w13, w2, w13_scale, w2_scale


def _moe_config(
    vllm_config: VllmConfig,
    geometry: Geometry,
    world_size: int,
    device: torch.device,
    backend: str,
    routing_method: RoutingMethodType | None = None,
    swiglu_limit: float | None = None,
    *,
    sm90_format: str | None = None,
    dp_size: int | None = None,
) -> FusedMoEConfig:
    dp_size = world_size if dp_size is None else dp_size
    assert world_size % dp_size == 0
    parallel = FusedMoEParallelConfig.make(
        tp_size_=world_size // dp_size,
        pcp_size_=1,
        dp_size_=dp_size,
        sp_size_=1,
        vllm_parallel_config=vllm_config.parallel_config,
    )
    return FusedMoEConfig(
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
        hidden_dim=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        num_local_experts=geometry.num_experts // world_size,
        num_logical_experts=geometry.num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=(
            _routing_method(backend, sm90_format)
            if routing_method is None
            else routing_method
        ),
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        moe_backend=backend,
        max_num_tokens=MAX_TOKENS_PER_RANK,
        swiglu_limit=swiglu_limit,
    )


def _check_expert_views_alias_kernel_memory(
    adapter: FlashInferMoeEp,
    geometry: Geometry,
    rank: int,
    num_local_experts: int,
    device: torch.device,
    sentinel: float = X_SENTINEL,
    h_signal: float = H_SIGNAL,
    expert_scale: Callable[[int], float] = _unit_scale,
) -> None:
    """EPLB moves experts through `expert_weight_views`; the kernel must see it.

    Swap local experts 0 and 1 in place on every rank, then route through the
    swapped physical slots: the output must be the one the unswapped logical
    experts would have produced.
    """
    for view in adapter.expert_weight_views():
        first = view[0].clone()
        view[0].copy_(view[1])
        view[1].copy_(first)
    torch.distributed.barrier()

    topk_ids, topk_weights = _routing(geometry, rank, device)
    slot = topk_ids % num_local_experts
    physical_ids = torch.where(slot == 0, topk_ids + 1, topk_ids)
    physical_ids = torch.where(slot == 1, topk_ids - 1, physical_ids)
    with torch.inference_mode():
        output = adapter(
            _hidden_states(geometry, device, sentinel), physical_ids, topk_weights
        )
    torch.accelerator.synchronize()
    expected = _expected_output(
        geometry, topk_ids, topk_weights, h_signal, expert_scale
    )
    assert torch.equal(output, expected)


def _check_sm90_modular_kernel(
    vllm_config: VllmConfig,
    geometry: Geometry,
    moe: FusedMoEConfig,
    rank: int,
    device: torch.device,
    sm90_format: str,
) -> None:
    """Hopper weights through the loading hook and `FusedMoEKernel`.

    Rank 1 has no tokens, but the other ranks route to its experts and it owns
    part of the collective, so it must still enter the megakernel. A full batch
    on every rank is then replayed from a CUDA graph. The DP-wide token counts
    come from the forward context, as in serving, so both use the 64-token
    workspace.
    """
    init_workspace_manager(device)
    if sm90_format == "fp8_block":
        weights = _local_block_fp8_expert_weights(
            geometry, rank, moe.num_local_experts, device
        )
        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=weights.w13_scale,
            w2_scale=weights.w2_scale,
            block_shape=[128, 128],
        )
        kernel_dtype = torch.float8_e4m3fn
    else:
        assert sm90_format == "mxfp4"
        weights = _local_mxfp4_expert_weights(
            geometry, rank, moe.num_local_experts, device
        )
        quant_config = make_mxfp4_moe_quant_config(
            Mxfp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL, w1_scale=None, w2_scale=None
        )
        kernel_dtype = torch.uint8
    layer = torch.nn.Module()
    for name, tensor in zip(
        ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
        (weights.w13, weights.w13_scale, weights.w2, weights.w2_scale),
    ):
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.apply_router_weight_on_input = False
    layer.expert_map_manager = SimpleNamespace(placement_strategy="linear")
    experts = FlashInferMoeEpExperts(moe, quant_config)
    experts.process_weights_after_loading(layer)
    experts.keep_constants(layer)
    assert layer.w13_weight.dtype == kernel_dtype
    # The layer's scales are the kernel's block scales: one per block as
    # loaded, not the per-expert factor MXFP4 adds.
    for name, loaded in (("w13", weights.w13_scale), ("w2", weights.w2_scale)):
        scale = getattr(layer, f"{name}_weight_scale")
        assert loaded is not None and scale.numel() == loaded.numel()
    kernel = mk.FusedMoEKernel(PassThroughPrepareAndFinalize(), experts)

    topk_ids, topk_weights = _routing(geometry, rank, device)
    hidden_states = _hidden_states(geometry, device, X_SENTINEL_FP8)
    expected = _expected_output(geometry, topk_ids, topk_weights)
    world_size = vllm_config.parallel_config.data_parallel_size
    full = [NUM_TOKENS_PER_RANK] * world_size
    idle = [0 if r == 1 else NUM_TOKENS_PER_RANK for r in range(world_size)]

    def forward(across_dp: list[int]) -> torch.Tensor:
        tokens = across_dp[rank]
        with (
            torch.inference_mode(),
            set_forward_context(
                None,
                vllm_config,
                num_tokens=tokens,
                num_tokens_across_dp=torch.tensor(across_dp, dtype=torch.int32),
            ),
        ):
            return kernel.apply(
                hidden_states[:tokens],
                layer.w13_weight,
                layer.w2_weight,
                topk_weights[:tokens],
                topk_ids[:tokens],
                MoEActivation.SILU,
                geometry.num_experts,
                None,
                False,
            )

    try:
        for across_dp in (full, idle):
            output = forward(across_dp)
            torch.accelerator.synchronize()
            assert torch.equal(output, expected[: across_dp[rank]])

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward(full)
        with torch.inference_mode():
            output.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(output, expected)
    finally:
        assert experts._adapter is not None
        experts._adapter.destroy()


def _init_warmup_forward(
    env: dict[str, str],
    world_size: int,
    geometry: Geometry,
    backend: str,
    sm90_format: str | None = None,
) -> None:
    set_env_vars_and_device(env)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = backend
    if sm90_format is not None:
        # What KernelConfig binds for megakernel backends.
        vllm_config.parallel_config.all2all_backend = "passthrough"
        vllm_config.parallel_config.data_parallel_rank = int(env["RANK"])

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        moe = _moe_config(
            vllm_config, geometry, world_size, device, backend, sm90_format=sm90_format
        )
        weight_key, activation_key = kMxfp4Static, None
        if sm90_format == "fp8_block":
            weight_key, activation_key = kFp8Static128BlockSym, kFp8Dynamic128Sym
            make_weights = _local_block_fp8_expert_weights
            sentinel = X_SENTINEL_FP8
        elif sm90_format == "mxfp4":
            # The kernel scales FP8 activations per token: 448 pins it to 1.
            make_weights = _local_mxfp4_expert_weights
            sentinel = X_SENTINEL_FP8
        else:
            make_weights = _local_expert_weights
            sentinel = X_SENTINEL
        supported, reason = FlashInferMoeEpExperts.is_supported_config(
            FlashInferMoeEpExperts,
            moe,
            weight_key,
            activation_key,
            mk.FusedMoEActivationFormat.Standard,
        )
        assert supported, reason

        adapter = FlashInferMoeEp(
            moe,
            make_weights(geometry, rank, moe.num_local_experts, device),
            apply_topk_in_fc1=apply_topk_in_fc1(moe),
            weight_format="fp8_block" if sm90_format == "fp8_block" else "mxfp4",
        )
        try:
            adapter.warmup()

            topk_ids, topk_weights = _routing(geometry, rank, device)
            # Serving runs the model under inference mode, where the
            # workspace warmup also allocates the kernel's buffers.
            with torch.inference_mode():
                output = adapter(
                    _hidden_states(geometry, device, sentinel), topk_ids, topk_weights
                )
            torch.accelerator.synchronize()

            expected = _expected_output(geometry, topk_ids, topk_weights)
            assert output.dtype == expected.dtype and output.shape == expected.shape
            assert torch.equal(output, expected), (
                f"rank {rank}: max abs diff "
                f"{(output.float() - expected.float()).abs().max().item()}, "
                f"nonzero output channels per token "
                f"{[torch.nonzero(row).flatten().tolist() for row in output]}"
            )

            # Hopper MXFP4 rejects EPLB.
            if backend == FLASHINFER_MOE_EP_CUTEDSL and sm90_format != "mxfp4":
                _check_expert_views_alias_kernel_memory(
                    adapter, geometry, rank, moe.num_local_experts, device, sentinel
                )
        finally:
            adapter.destroy()

        if sm90_format is not None:
            _check_sm90_modular_kernel(
                vllm_config, geometry, moe, rank, device, sm90_format
            )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100)
    or current_platform.is_device_capability(107),
    reason="The bf16 MXFP4 path and the DeepGEMM megakernel require SM100/SM103",
)
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=[g.name for g in GEOMETRIES])
@pytest.mark.parametrize("backend", BACKENDS, ids=("cutedsl", "deep_gemm"))
@pytest.mark.parametrize("world_size", [2, 4])
def test_init_warmup_forward_is_exact_on_grid_inputs(
    world_size: int, backend: str, geometry: Geometry
):
    pytest.importorskip("flashinfer.moe_ep")
    if backend == FLASHINFER_MOE_EP_DEEP_GEMM and not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    distributed_run(_init_warmup_forward, world_size, geometry, backend)


def _check_sm107_forward_is_exact(
    adapter: FlashInferMoeEp,
    geometry: Geometry,
    rank: int,
    num_local_experts: int,
    device: torch.device,
    h_signal: float,
) -> None:
    """Warm up, compare one forward bit for bit, check the EPLB views, and
    destroy the adapter."""
    try:
        adapter.warmup()

        topk_ids, topk_weights = _routing(geometry, rank, device)
        output = adapter(_hidden_states(geometry, device), topk_ids, topk_weights)
        torch.accelerator.synchronize()

        expected = _expected_output(
            geometry, topk_ids, topk_weights, h_signal, _sm107_expert_scale
        )
        assert output.dtype == expected.dtype and output.shape == expected.shape
        assert torch.equal(output, expected), (
            f"rank {rank}: max abs diff "
            f"{(output.float() - expected.float()).abs().max().item()}"
        )

        _check_expert_views_alias_kernel_memory(
            adapter,
            geometry,
            rank,
            num_local_experts,
            device,
            h_signal=h_signal,
            expert_scale=_sm107_expert_scale,
        )
    finally:
        adapter.destroy()


def _sm107_nvfp4_init_warmup_forward(
    env: dict[str, str],
    world_size: int,
    geometry: Geometry,
    routing_method: RoutingMethodType,
) -> None:
    set_env_vars_and_device(env)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = FLASHINFER_MOE_EP_CUTEDSL

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        moe = _moe_config(
            vllm_config,
            geometry,
            world_size,
            device,
            FLASHINFER_MOE_EP_CUTEDSL,
            routing_method,
        )
        supported, reason = FlashInferMoeEpExperts.is_supported_config(
            FlashInferMoeEpExperts,
            moe,
            kNvfp4Static,
            kNvfp4Dynamic,
            mk.FusedMoEActivationFormat.Standard,
        )
        assert supported, reason

        num_local_experts = moe.num_local_experts
        weights, g1, g2 = _local_expert_nvfp4_weights(
            geometry, rank, num_local_experts, device
        )
        g1_orig, g2_orig = g1.clone(), g2.clone()
        # Ranks disagree on the activation scales; the EP maximum wins.
        epilogue = sm107_nvfp4_epilogue(
            FlashInferMoeEpEpilogue(fc1_alpha=g1, fc2_alpha=g2),
            torch.full((num_local_experts, 2), 2.0 ** -(3 + rank % 2), device=device),
            torch.full((num_local_experts,), 2.0 ** -(4 + rank % 2), device=device),
        )
        assert epilogue.fc1_alpha is not None and epilogue.fc2_alpha is not None
        assert epilogue.fc1_norm_const is not None
        assert epilogue.fc1_alpha.data_ptr() == g1.data_ptr()
        assert epilogue.fc2_alpha.data_ptr() == g2.data_ptr()
        # Without the EP reduce only odd ranks fail; fail all ranks together.
        reduced = torch.tensor(
            float(
                epilogue.input_norm_const == 8.0
                and torch.equal(g1, g1_orig / 8)
                and torch.equal(g2, g2_orig / 16)
                and torch.equal(epilogue.fc1_norm_const, torch.full_like(g2, 16.0))
            ),
            device=device,
        )
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.MIN)
        assert reduced.item() == 1.0, (
            f"EP reduce of the activation scales failed on some rank; rank {rank} "
            f"has input_norm_const={epilogue.input_norm_const}"
        )

        adapter = FlashInferMoeEp(
            moe,
            weights,
            epilogue,
            apply_topk_in_fc1=apply_topk_in_fc1(moe),
            weight_format="nvfp4",
        )
        _check_sm107_forward_is_exact(
            adapter, geometry, rank, num_local_experts, device, SM107_H_SIGNAL
        )


@pytest.mark.skipif(
    not current_platform.is_device_capability(107),
    reason="The SM107 NVFP4 MoE-EP megakernel requires SM107",
)
@pytest.mark.parametrize(
    "geometry, routing_method",
    SM107_CASES,
    ids=[geometry.name for geometry, _ in SM107_CASES],
)
@pytest.mark.parametrize("world_size", [2, 4])
def test_sm107_nvfp4_init_warmup_forward_is_exact_on_grid_inputs(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    geometry: Geometry,
    routing_method: RoutingMethodType,
):
    """Checkpoint NVFP4 experts and the calibrated epilogue on SM107.

    Covers the prequantized weight handoff, gate/up order, per-expert and
    EP-local epilogue indexing, the EP reduce of the activation scales and
    FlashInfer's owned-output forward.
    """
    pytest.importorskip("flashinfer.moe_ep")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    # Spawned workers inherit it before CuTe DSL picks a target.
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_107a")
    distributed_run(
        _sm107_nvfp4_init_warmup_forward, world_size, geometry, routing_method
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability(90),
    reason="The Hopper FlashInfer MoE-EP megakernels require SM90",
)
@pytest.mark.parametrize(
    "geometry", SM90_GEOMETRIES, ids=[g.name for g in SM90_GEOMETRIES]
)
@pytest.mark.parametrize("sm90_format", ["fp8_block", "mxfp4"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_sm90_init_warmup_forward_is_exact_on_grid_inputs(
    world_size: int, sm90_format: str, geometry: Geometry
):
    pytest.importorskip("flashinfer.moe_ep")
    pytest.importorskip("nvshmem.core")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    distributed_run(
        _init_warmup_forward,
        world_size,
        geometry,
        FLASHINFER_MOE_EP_CUTEDSL,
        sm90_format,
    )


def _sm107_mxfp4_init_warmup_forward(
    env: dict[str, str],
    world_size: int,
    geometry: Geometry,
    routing_method: RoutingMethodType,
    swiglu_limit: float | None,
) -> None:
    set_env_vars_and_device(env)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = FLASHINFER_MOE_EP_CUTEDSL

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        moe = _moe_config(
            vllm_config,
            geometry,
            world_size,
            device,
            FLASHINFER_MOE_EP_CUTEDSL,
            routing_method,
            swiglu_limit,
        )
        supported, reason = FlashInferMoeEpExperts.is_supported_config(
            FlashInferMoeEpExperts,
            moe,
            kMxfp4Static,
            None,
            mk.FusedMoEActivationFormat.Standard,
        )
        assert supported, reason

        num_local_experts = moe.num_local_experts
        weights = mxfp4_moe_ep_weights(
            FLASHINFER_MOE_EP_CUTEDSL,
            *_local_expert_mxfp4_weights(geometry, rank, num_local_experts, device),
        )
        adapter = FlashInferMoeEp(
            moe,
            weights,
            apply_topk_in_fc1=apply_topk_in_fc1(moe),
            weight_format="mxfp4",
        )
        # The up rows give 1, so the clamped gate is h.
        h_signal = SM107_H_SIGNAL if swiglu_limit is None else swiglu_limit
        _check_sm107_forward_is_exact(
            adapter, geometry, rank, num_local_experts, device, h_signal
        )


@pytest.mark.skipif(
    not current_platform.is_device_capability(107),
    reason="The SM107 MXFP4 MoE-EP megakernel requires SM107",
)
@pytest.mark.parametrize(
    "geometry, routing_method, swiglu_limit",
    SM107_MX_CASES,
    ids=[geometry.name for geometry, _, _ in SM107_MX_CASES],
)
@pytest.mark.parametrize("world_size", [2, 4])
def test_sm107_mxfp4_init_warmup_forward_is_exact_on_grid_inputs(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    geometry: Geometry,
    routing_method: RoutingMethodType,
    swiglu_limit: float | None,
):
    """Checkpoint MXFP4 experts on SM107's MXFP8-activation megakernel.

    Covers the native weight handoff (E8M0 scale bytes reinterpreted, not
    converted), gate/up order, per-expert and EP-local scale indexing, the
    paired-warp scale exchange of MXFP8, the SwiGLU gate clamp, the EPLB weight
    views and FlashInfer's owned-output forward.
    """
    pytest.importorskip("flashinfer.moe_ep")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    # Spawned workers inherit it before CuTe DSL picks a target.
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_107a")
    distributed_run(
        _sm107_mxfp4_init_warmup_forward,
        world_size,
        geometry,
        routing_method,
        swiglu_limit,
    )


@pytest.mark.parametrize(
    "backend, capability, routing, weight_format",
    [
        (FLASHINFER_MOE_EP_CUTEDSL, (10, 0), RoutingMethodType.Default, "nvfp4"),
        (FLASHINFER_MOE_EP_DEEP_GEMM, (10, 0), RoutingMethodType.DeepseekV4, "mxfp4"),
        (FLASHINFER_MOE_EP_CUTEDSL, (10, 7), RoutingMethodType.DeepseekV4, "nvfp4"),
        (FLASHINFER_MOE_EP_CUTEDSL, (10, 7), RoutingMethodType.Default, "nvfp4"),
        (FLASHINFER_MOE_EP_CUTEDSL, (10, 7), RoutingMethodType.DeepseekV4, "mxfp4"),
        (FLASHINFER_MOE_EP_CUTEDSL, (10, 7), RoutingMethodType.Default, "mxfp4"),
    ],
    ids=[
        "cutedsl",
        "deep_gemm",
        "cutedsl-sm107",
        "cutedsl-sm107-default",
        "cutedsl-sm107-mxfp4",
        "cutedsl-sm107-mxfp4-default",
    ],
)
def test_adapter_kwargs_match_installed_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    capability: tuple[int, int],
    routing: RoutingMethodType,
    weight_format: str,
):
    """The adapter's kwargs construct the installed FlashInfer's config types.

    Needs no SM100 or SM107, so config kwarg drift fails in CI rather than at
    model load. Fakes only the megakernel layer (symmetric memory), the device
    capability, the EP group, the device index and the DeepGEMM import. On
    SM107 the routing flag must reach the config of the expert weight format,
    and MXFP4 scale bytes must reach FlashInfer typed as E8M0.
    """
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    sm107 = capability == (10, 7)
    layer_args = _fake_mega_layer(monkeypatch, moe_ep)
    monkeypatch.setattr(fi_ep, "_current_capability", lambda: capability)
    # Record the original value so teardown also removes what the adapter sets.
    monkeypatch.setenv(SM107_ROUTE_CHECK_ENV, "x")
    monkeypatch.delenv(SM107_ROUTE_CHECK_ENV)
    moe = _adapter_moe(backend, GEOMETRIES[0])
    moe.routing_method = routing
    if weight_format == "mxfp4":
        packed = torch.zeros(1, dtype=torch.uint8)
        scale = torch.zeros(1, dtype=torch.uint8)
        weights = FlashInferMoeEpWeights(packed, packed, scale, scale)
    else:
        weight = torch.zeros(1, dtype=torch.bfloat16)
        weights = FlashInferMoeEpWeights(w13=weight, w2=weight)
    adapter = FlashInferMoeEp(
        moe,
        weights,
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
        weight_format=weight_format,
    )

    ((_, _, weight_pack, mega_config),) = layer_args
    megakernel = mega_config.megakernel
    if sm107:
        expected_config = (
            moe_ep.Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
            if weight_format == "nvfp4"
            else moe_ep.Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
        )
        assert isinstance(megakernel, expected_config)
        assert megakernel.knobs is None
        assert megakernel.apply_routing_weights_before_fc2 is (
            routing is RoutingMethodType.DeepseekV4
        )
        assert adapter._return_workspace_view is False
        assert os.environ.get(SM107_ROUTE_CHECK_ENV) == (
            "0" if routing is RoutingMethodType.DeepseekV4 else None
        )
        if weight_format == "mxfp4":
            # The packed data as is, the scale bytes reinterpreted, not converted.
            assert weight_pack.w13 is packed and weight_pack.w2 is packed
            assert weight_pack.w13_scale.dtype == torch.float8_e8m0fnu
            assert weight_pack.w2_scale.dtype == torch.float8_e8m0fnu
            assert weight_pack.w13_scale.data_ptr() == scale.data_ptr()
    else:
        expected = (
            moe_ep.Nvfp4CutedslMegaMoeConfig
            if backend == FLASHINFER_MOE_EP_CUTEDSL
            else moe_ep.DeepGemmMegaMoeConfig
        )
        assert isinstance(megakernel, expected)
        assert adapter._return_workspace_view is (
            routing is RoutingMethodType.DeepseekV4
        )
        assert SM107_ROUTE_CHECK_ENV not in os.environ


def test_block_fp8_adapter_hands_kernel_ready_weights_to_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
):
    """Block-FP8 weights build the SM90 config and skip FlashInfer's own
    preprocessing, which would requantize them from bf16. The kernel tensors
    bound back to the layer keep their dtypes, so a rebuild from them (e.g.
    preprocessed-weight loading) selects the same kernel."""
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    workspace_capacities: list[int] = []
    layer_args = _fake_mega_layer(monkeypatch, moe_ep, workspace_capacities)
    moe = _adapter_moe(FLASHINFER_MOE_EP_CUTEDSL, TINY_SM90)
    weights = _tiny_block_fp8_weights()
    adapter = FlashInferMoeEp(
        moe,
        weights,
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
        weight_format="fp8_block",
    )
    w13, w13_scale, w2, w2_scale = adapter.kernel_weights()
    FlashInferMoeEp(
        moe,
        FlashInferMoeEpWeights(w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale),
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
        weight_format="fp8_block",
    )

    assert [t.dtype for t in (w13, w13_scale, w2, w2_scale)] == [
        torch.float8_e4m3fn,
        torch.float32,
        torch.float8_e4m3fn,
        torch.float32,
    ]
    (_, _, weight_pack, mega_config), (_, _, _, rebuilt) = layer_args
    assert isinstance(rebuilt.megakernel, moe_ep.Sm90PullFp8MegaMoeConfig)
    assert _kernel_storage(rebuilt) == _kernel_storage(mega_config)
    assert weight_pack is None and not mega_config.preprocess_weights
    # Batches up to 64 tokens get their own workspace below the 128 maximum.
    assert workspace_capacities == [64, 64]
    megakernel = mega_config.megakernel
    assert isinstance(megakernel, moe_ep.Sm90PullFp8MegaMoeConfig)
    assert megakernel.fp8_scale_mode == "blockwise"
    assert megakernel.gate_up_clamp == moe.swiglu_limit
    (fc1, _, *_), (fc2, _, *_) = mega_config.transformed_weights
    hidden, intermediate = TINY_SM90.hidden_size, TINY_SM90.intermediate_size
    # (E, K, N) views over the layer's storage, K contiguous.
    assert fc1.shape == (2, hidden, 2 * intermediate) and fc1.stride(1) == 1
    assert fc2.shape == (2, intermediate, hidden) and fc2.stride(1) == 1
    assert weights.w13_scale is not None and weights.w2_scale is not None
    assert _kernel_storage(mega_config) == [
        t.data_ptr()
        for t in (weights.w13, weights.w13_scale, weights.w2, weights.w2_scale)
    ]


def test_block_fp8_workspace_follows_the_dp_wide_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    """Every EP rank must launch on the same workspace, so the size comes from
    the DP-wide token counts, and without them the default workspace is used."""
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    used: list[object] = []
    _fake_mega_layer(monkeypatch, moe_ep, [], used)
    moe = _adapter_moe(FLASHINFER_MOE_EP_CUTEDSL, TINY_SM90)
    adapter = FlashInferMoeEp(
        moe,
        _tiny_block_fp8_weights(),
        apply_topk_in_fc1=apply_topk_in_fc1(moe),
        weight_format="fp8_block",
    )
    hidden_states = torch.zeros(3, TINY_SM90.hidden_size, dtype=torch.bfloat16)
    topk_ids = torch.zeros(3, TINY_SM90.top_k, dtype=torch.int32)
    topk_weights = torch.ones(3, TINY_SM90.top_k)

    adapter(hidden_states, topk_ids, topk_weights)
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = 2
    for across_dp in ([3, 64], [3, 65]):
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=3,
            num_tokens_across_dp=torch.tensor(across_dp, dtype=torch.int32),
        ):
            adapter(hidden_states, topk_ids, topk_weights)

    assert used == [None, ("workspace", 64), None]


def test_sm90_mxfp4_adapter_hands_packed_weights_to_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
):
    """Pass packed E2M1/E8M0 checkpoint weights into the SM90 MXFP4 config."""
    moe_ep = pytest.importorskip("flashinfer.moe_ep")
    config_cls = moe_ep.Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig
    workspace_capacities: list[int] = []
    layer_args = _fake_mega_layer(monkeypatch, moe_ep, workspace_capacities)
    monkeypatch.setattr(fi_ep, "_current_capability", lambda: (9, 0))
    moe = _adapter_moe(FLASHINFER_MOE_EP_CUTEDSL, TINY_SM90)
    hidden, intermediate = TINY_SM90.hidden_size, TINY_SM90.intermediate_size
    weights = FlashInferMoeEpWeights(
        w13=torch.zeros(2, 2 * intermediate, hidden // 2, dtype=torch.uint8),
        w2=torch.zeros(2, hidden, intermediate // 2, dtype=torch.uint8),
        w13_scale=torch.zeros(2, 2 * intermediate, hidden // 32, dtype=torch.uint8),
        w2_scale=torch.zeros(2, hidden, intermediate // 32, dtype=torch.uint8),
    )
    FlashInferMoeEp(
        moe, weights, apply_topk_in_fc1=apply_topk_in_fc1(moe), weight_format="mxfp4"
    )

    ((_, _, weight_pack, mega_config),) = layer_args
    assert isinstance(weight_pack, moe_ep.PrequantizedMoEWeights)
    assert all(
        getattr(weight_pack, name) is getattr(weights, name)
        for name in ("w13", "w2", "w13_scale", "w2_scale")
    )
    assert mega_config.preprocess_weights
    assert isinstance(mega_config.megakernel, config_cls)
    assert mega_config.megakernel.gate_up_clamp == moe.swiglu_limit
    assert workspace_capacities == [64]


def _quantize_block_fp8(real: torch.Tensor, power_of_two: bool):
    *lead, rows, cols = real.shape
    blocks = real.reshape(*lead, rows // 128, 128, cols // 128, 128)
    scale = blocks.abs().amax(dim=(-3, -1)) / 448.0
    if power_of_two:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return (real / _per_element(scale)).to(torch.float8_e4m3fn), scale


def _per_element(scale: torch.Tensor) -> torch.Tensor:
    return scale.repeat_interleave(128, -2).repeat_interleave(128, -1)


def _gate_up_interleaved(w13: torch.Tensor) -> torch.Tensor:
    """Rows ordered [pair, {gate, up}, 8], the SM90 megakernel's FC1 layout."""
    experts, rows, cols = w13.shape
    gate, up = w13.view(experts, 2, rows // 16, 8, cols).unbind(1)
    return torch.stack((gate, up), dim=2).reshape(experts, rows, cols)


@pytest.mark.parametrize("power_of_two", [True, False])
def test_sm90_fp8_block_weights_keep_checkpoint_values(power_of_two: bool):
    """Interleaving gate and up rows puts two checkpoint blocks under one scale.

    Power-of-two scales survive that up to rounding below the E4M3 normal
    range; other scales round the rescaled half once.
    """
    torch.manual_seed(0)
    real = torch.randn(2, 512, 256) * 0.02
    # Gate blocks 8x the up blocks, so each interleaved block mixes scales.
    real[:, :256] *= 8.0
    w13, scale = _quantize_block_fp8(real, power_of_two)

    kernel_w13, kernel_scale = sm90_fp8_block_moe_ep_weights(w13, scale)

    assert kernel_w13.dtype == w13.dtype and kernel_w13.shape == w13.shape
    assert kernel_scale.shape == scale.shape
    restored = kernel_w13.float() * _per_element(kernel_scale)
    expected = _gate_up_interleaved(w13.float() * _per_element(scale))
    error = (restored - expected).abs()
    if power_of_two:
        # Half the smallest E4M3 subnormal step.
        assert torch.all(error <= 2.0**-10 * _per_element(kernel_scale))
    else:
        assert (error.norm() / expected.norm()).item() < 0.03


@pytest.mark.parametrize("sm90_format", ["fp8_block", "mxfp4"])
def test_sm90_megakernels_reject_input_side_router_weights(
    monkeypatch: pytest.MonkeyPatch, sm90_format: str
):
    """The SM90 kernels weight the activation before fc2, which is not the same
    as weighting the expert input (apply_router_weight_on_input)."""
    monkeypatch.setattr(fi_ep, "_current_capability", lambda: (9, 0))
    moe = make_dummy_moe_config(
        num_experts=8, experts_per_token=2, hidden_dim=256, intermediate_size=128
    )
    if sm90_format == "fp8_block":
        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=torch.ones(8, 2, 2),
            w2_scale=torch.ones(8, 2, 1),
            block_shape=[128, 128],
        )
    else:
        quant_config = make_mxfp4_moe_quant_config(
            Mxfp4MoeBackend.FLASHINFER_MOE_EP_CUTEDSL, w1_scale=None, w2_scale=None
        )
    layer = SimpleNamespace(
        expert_map_manager=SimpleNamespace(placement_strategy="linear"),
        apply_router_weight_on_input=True,
    )
    with pytest.raises(ValueError, match="apply_router_weight_on_input"):
        FlashInferMoeEpExperts(moe, quant_config).process_weights_after_loading(layer)


TINY_SM90 = Geometry("tiny", 256, 128, 4, 2)


def _tiny_block_fp8_weights() -> FlashInferMoeEpWeights:
    hidden, intermediate = TINY_SM90.hidden_size, TINY_SM90.intermediate_size
    return FlashInferMoeEpWeights(
        w13=torch.zeros(2, 2 * intermediate, hidden, dtype=torch.float8_e4m3fn),
        w2=torch.zeros(2, hidden, intermediate, dtype=torch.float8_e4m3fn),
        w13_scale=torch.ones(2, 2 * intermediate // 128, hidden // 128),
        w2_scale=torch.ones(2, hidden // 128, intermediate // 128),
    )


def _kernel_storage(mega_config) -> list[int]:
    (fc1, fc1_scale, *_), (fc2, fc2_scale, *_) = mega_config.transformed_weights
    return [t.data_ptr() for t in (fc1, fc1_scale, fc2, fc2_scale)]


def _fake_mega_layer(
    monkeypatch: pytest.MonkeyPatch,
    moe_ep,
    workspace_capacities: list[int] | None = None,
    used_workspaces: list[object] | None = None,
) -> list[tuple]:
    layer_args: list[tuple] = []

    class FakeMegaLayer:
        def __init__(self, *args) -> None:
            layer_args.append(args)
            self._transformed = args[3].transformed_weights

        def create_workspace(self, max_tokens_per_rank: int) -> object:
            assert workspace_capacities is not None
            workspace_capacities.append(max_tokens_per_rank)
            return ("workspace", max_tokens_per_rank)

        def forward(self, tensors, workspace=None, return_workspace_view=False):
            assert used_workspaces is not None
            used_workspaces.append(workspace)
            return torch.zeros_like(tensors.hidden_states)

        def destroy(self) -> None:
            pass

    monkeypatch.setattr(moe_ep, "MoEEpMegaLayer", FakeMegaLayer)
    monkeypatch.setattr(fi_ep, "_expose_deep_gemm_to_flashinfer", lambda: None)
    monkeypatch.setattr(
        fi_ep,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=0, device_group=None),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    return layer_args


def _adapter_moe(backend: str, geometry: Geometry) -> SimpleNamespace:
    return SimpleNamespace(
        moe_backend=backend,
        num_experts=geometry.num_experts,
        experts_per_token=geometry.top_k,
        hidden_dim=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        max_num_tokens=MAX_TOKENS_PER_RANK,
        swiglu_limit=10.0,
        routing_method=_routing_method(backend),
    )


def _local_nosync_weights(
    geometry: Geometry, rank: int, num_local_experts: int, device: torch.device
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20261121 + rank)
    weights = {}
    for expert in range(rank * num_local_experts, (rank + 1) * num_local_experts):
        for projection in ("w1", "w2", "w3"):
            name = f"layers.0.ffn.experts.{expert}.{projection}"
            shape = (
                (geometry.intermediate_size, geometry.hidden_size // 2)
                if projection != "w2"
                else (geometry.hidden_size, geometry.intermediate_size // 2)
            )
            weights[f"{name}.weight"] = torch.randint(
                0, 256, shape, device=device, generator=generator, dtype=torch.uint8
            )
            scale_shape = (*shape[:-1], shape[-1] // 16)
            weights[f"{name}.scale"] = torch.randint(
                120,
                124,
                scale_shape,
                device=device,
                generator=generator,
                dtype=torch.uint8,
            )
    return weights


class _DeterministicRoutingSimulatorRouter(RoutingSimulatorRouter):
    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights, ids = torch.topk(router_logits, self.top_k, dim=-1)
        if indices_type is not None:
            ids = ids.to(indices_type)
        return torch.softmax(weights, dim=-1), ids


def _nosync_sm120_layer_checks(
    env: dict[str, str],
    world_size: int,
    geometry: Geometry,
    dp_size: int | None = None,
    fallback_tokens: int = 129,
    execution_tokens: int = 64,
    max_tokens_per_rank: int = 128,
    dedup_uniform: bool = False,
    require_custom_ar: bool = False,
) -> None:
    dp_size = world_size if dp_size is None else dp_size
    tp_size = world_size // dp_size
    set_env_vars_and_device(env)
    if dedup_uniform:
        os.environ["VLLM_FLASHINFER_MOE_EP_NOSYNC_TP_DEDUP"] = "1"
        set_custom_all_reduce(require_custom_ar)
    os.environ["VLLM_FLASHINFER_MOE_EP_NOSYNC_MAX_TOKENS_PER_RANK"] = str(
        max_tokens_per_rank
    )
    rank = int(env["RANK"])
    device = torch.device(f"cuda:{rank}")
    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = dp_size
    vllm_config.parallel_config.data_parallel_rank = rank // tp_size
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = "flashinfer_moe_ep_nosync"
    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(tp_size, 1)
        moe = _moe_config(
            vllm_config,
            geometry,
            world_size,
            device,
            "flashinfer_moe_ep_nosync",
            dp_size=dp_size,
        )
        layer = RoutedExperts(
            layer_name="layers.0.ffn.experts",
            params_dtype=torch.bfloat16,
            moe_config=moe,
            quant_config=Mxfp4Config(),
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            expert_map_manager=ExpertMapManager(
                max_num_batched_tokens=128,
                top_k=geometry.top_k,
                global_num_experts=geometry.num_experts,
                num_redundant_experts=0,
                num_expert_group=None,
                moe_parallel_config=moe.moe_parallel_config,
                placement_strategy="linear",
                enable_eplb=False,
            ),
        ).to(device)
        weights = _local_nosync_weights(geometry, rank, moe.num_local_experts, device)
        loaded = list(
            layer.load_weights(
                (
                    name.removeprefix("layers.0.ffn.experts.").removesuffix(".scale")
                    + (".weight_scale" if name.endswith(".scale") else ""),
                    tensor,
                )
                for name, tensor in weights.items()
            )
        )
        assert len(loaded) == 6 * moe.num_local_experts
        raw_scales = layer.w13_weight_scale.clone(), layer.w2_weight_scale.clone()
        if require_custom_ar:
            with pytest.raises(ValueError, match="CustomAllreduce"):
                layer.quant_method.process_weights_after_loading(layer)
            return
        if dedup_uniform and dp_size != 1:
            with pytest.raises(ValueError, match="DP or SP"):
                layer.quant_method.process_weights_after_loading(layer)
            return
        layer.quant_method.process_weights_after_loading(layer)
        init_workspace_manager(device)
        if dp_size == 1:
            os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "normal_routing"
            router = _DeterministicRoutingSimulatorRouter(
                top_k=geometry.top_k,
                global_num_experts=geometry.num_experts,
                kernel_routing_method_type=RoutingMethodType.RenormalizeNaive,
            )
            runner = MoERunner(
                layer_name="layers.0.ffn.experts",
                moe_config=moe,
                router=router,
                routed_experts=layer,
            )
            if dedup_uniform:
                torch.manual_seed(20261122)
            hidden_states = torch.randn(
                execution_tokens,
                geometry.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            if dedup_uniform:
                replicas = [torch.empty_like(hidden_states) for _ in range(world_size)]
                torch.distributed.all_gather(replicas, hidden_states)
                assert all(torch.equal(hidden_states, t) for t in replicas)
            router_logits = torch.zeros(
                execution_tokens,
                geometry.num_experts,
                dtype=torch.float32,
                device=device,
            )

            def forward() -> torch.Tensor:
                torch.manual_seed(20261122)
                return runner.forward(hidden_states, router_logits)

            with (
                torch.inference_mode(),
                set_forward_context(
                    None,
                    vllm_config,
                    num_tokens=execution_tokens,
                    num_tokens_across_dp=None,
                ),
            ):
                torch.manual_seed(20261122)
                topk_weights, topk_ids = router.select_experts(
                    hidden_states, router_logits
                )
                expected = layer.forward_modular(
                    hidden_states, topk_weights, topk_ids
                ).clone()
                if dedup_uniform:
                    from dataclasses import replace

                    from flashinfer.moe_ep import (
                        MoEEpLayer,
                        PrequantizedMoEWeights,
                    )

                    leader_output = expected
                    adapter = layer.quant_method._nosync
                    mega_layer = adapter._mega_layer
                    nosync_config = mega_layer._megakernel_config
                    assert nosync_config.dedup_replicated_inputs
                    reference = MoEEpLayer(
                        bootstrap=mega_layer._bootstrap,
                        fleet_params=mega_layer._fleet_params,
                        weights=PrequantizedMoEWeights(
                            w13=layer.w13_weight.view(torch.float4_e2m1fn_x2),
                            w2=layer.w2_weight.view(torch.float4_e2m1fn_x2),
                            w13_scale=layer.w13_weight_scale,
                            w2_scale=layer.w2_weight_scale,
                        ),
                        backend=replace(
                            mega_layer._mega_config,
                            megakernel=replace(
                                nosync_config, dedup_replicated_inputs=False
                            ),
                            transformed_weights=mega_layer._get_transformed_weights(),
                        ),
                    )
                    reference.warmup()
                    expected = reference.forward(
                        adapter._tensors(hidden_states, topk_ids, topk_weights)
                    ).clone()
                    reference.destroy()
                    leader = (
                        torch.arange(execution_tokens, device=device)
                        .remainder(world_size)
                        .eq(rank)
                    )
                    assert not leader_output[~leader].any()
                    reduced = tensor_model_parallel_all_reduce(leader_output)
                    torch.testing.assert_close(reduced, expected, atol=0, rtol=0)
                actual = forward()
                torch.accelerator.synchronize()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

                expected_graph = actual.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = forward()
                actual.zero_()
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(actual, expected_graph, atol=0, rtol=0)

            large_hidden = torch.randn(
                fallback_tokens,
                geometry.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            large_logits = torch.zeros(
                fallback_tokens,
                geometry.num_experts,
                dtype=torch.float32,
                device=device,
            )
            with (
                torch.inference_mode(),
                set_forward_context(
                    None,
                    vllm_config,
                    num_tokens=fallback_tokens,
                    num_tokens_across_dp=None,
                ),
            ):
                topk_weights, topk_ids = router.select_experts(
                    large_hidden, large_logits
                )
                expected = layer.forward_modular(large_hidden, topk_weights, topk_ids)
                expected = tensor_model_parallel_all_reduce(expected)
                actual = runner.forward(large_hidden, large_logits)
                torch.accelerator.synchronize()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            return
        method = layer.quant_method
        transformed = method._nosync._mega_layer._get_transformed_weights()
        assert transformed.scale_layout == "deepgemm"
        for name, parameter in (
            ("w13", layer.w13_weight),
            ("w2", layer.w2_weight),
            ("w13_scale", layer.w13_weight_scale),
            ("w2_scale", layer.w2_weight_scale),
        ):
            assert getattr(transformed, name).data_ptr() == parameter.data_ptr()
        for actual, raw in (
            (transformed.w13_scale, raw_scales[0]),
            (transformed.w2_scale, raw_scales[1]),
        ):
            expected = (
                raw.reshape(raw.shape[0], raw.shape[1], -1, 4)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(torch.uint8)
            )
            assert torch.equal(actual.flatten(), expected.flatten())

        hidden_states = torch.randn(
            129, geometry.hidden_size, dtype=torch.bfloat16, device=device
        )
        topk_ids = (
            torch.arange(129 * geometry.top_k, dtype=torch.int64, device=device)
            .remainder(geometry.num_experts)
            .view(129, geometry.top_k)
        )
        topk_weights = torch.full_like(topk_ids, 0.5, dtype=torch.float32)
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=129,
            num_tokens_across_dp=torch.full(
                (world_size,), 129, dtype=torch.int32, device="cpu"
            ),
        ):
            assert not method._nosync.should_use(64)
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None
            with dp_metadata.sp_local_sizes(1, pcp_size=1, use_ep=True):
                actual = method.apply(
                    layer, hidden_states, topk_weights, topk_ids, None, None
                )
                expected = method.moe_kernel.apply(
                    hidden_states=hidden_states,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    activation=layer.activation,
                    global_num_experts=geometry.num_experts,
                    apply_router_weight_on_input=False,
                    expert_map=layer.expert_map,
                    shared_experts=None,
                    shared_experts_input=None,
                )
        torch.accelerator.synchronize()
        assert torch.equal(actual, expected)
        with (
            pytest.raises(RuntimeError, match="DP token metadata"),
            torch.cuda.graph(torch.cuda.CUDAGraph()),
        ):
            method._nosync.should_use(64)
        method._nosync.destroy()


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="SM120 nosync MoE requires SM120 GPUs",
)
def test_nosync_sm120_scale_lifetime_and_dp_fallback():
    """Borrow final packed scales and use a global DP bound for fallback."""
    pytest.importorskip("flashinfer.moe_ep")
    if not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")
    world_size = 2
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    geometry = Geometry("sm120-nosync", 512, 256, 16, 2)
    distributed_run(_nosync_sm120_layer_checks, world_size, geometry)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="SM120 nosync MoE requires SM120 GPUs",
)
def test_nosync_sm120_tp_only_output_is_not_reduced_twice():
    """TP-only nosync must not repeat reduction of its complete expert output."""
    pytest.importorskip("flashinfer.moe_ep")
    if not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")
    world_size = 2
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    geometry = Geometry("sm120-nosync", 512, 256, 16, 2)
    distributed_run(
        partial(_nosync_sm120_layer_checks, dp_size=1), world_size, geometry
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="SM120 nosync MoE requires SM120 GPUs",
)
def test_nosync_sm120_tp_fallback_keeps_tp_allreduce():
    """The DeepGEMM fallback must retain TP all-reduce on the same adapter."""
    pytest.importorskip("flashinfer.moe_ep")
    if not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")
    world_size = 2
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    geometry = Geometry("sm120-nosync", 512, 256, 16, 2)
    distributed_run(
        partial(_nosync_sm120_layer_checks, dp_size=1), world_size, geometry
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="SM120 nosync MoE requires SM120 GPUs",
)
def test_nosync_sm120_tp_token_limit_fallback():
    """A smaller token limit must retain the correctly reduced DeepGEMM fallback."""
    pytest.importorskip("flashinfer.moe_ep")
    if not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")
    world_size = 2
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs")
    geometry = Geometry("sm120-nosync", 512, 256, 16, 2)
    distributed_run(
        partial(
            _nosync_sm120_layer_checks,
            dp_size=1,
            fallback_tokens=65,
            max_tokens_per_rank=64,
        ),
        world_size,
        geometry,
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="SM120 nosync MoE requires SM120 GPUs",
)
def test_nosync_sm120_tp_dedup():
    """Leader masking must remove duplicate work and preserve the full runner output."""
    pytest.importorskip("flashinfer.moe_ep")
    if not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")
    if torch.accelerator.device_count() < 2:
        pytest.skip("Need at least two GPUs")
    geometry = Geometry("sm120-nosync", 512, 256, 16, 2)
    distributed_run(
        partial(_nosync_sm120_layer_checks, dp_size=1, dedup_uniform=True),
        2,
        geometry,
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="SM120 nosync MoE requires SM120 GPUs",
)
def test_nosync_sm120_tp_capacity_192():
    """Run fused dedup at 192 tokens and retain DeepGEMM at 193 tokens."""
    pytest.importorskip("flashinfer.moe_ep")
    if not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")
    if torch.accelerator.device_count() < 2:
        pytest.skip("Need at least two GPUs")
    geometry = Geometry("sm120-nosync", 512, 256, 16, 2)
    distributed_run(
        partial(
            _nosync_sm120_layer_checks,
            dp_size=1,
            dedup_uniform=True,
            execution_tokens=192,
            fallback_tokens=193,
            max_tokens_per_rank=192,
        ),
        2,
        geometry,
    )


@pytest.mark.parametrize(
    "dp_size, require_custom_ar",
    [(1, True), (None, False)],
    ids=["custom_ar", "nonuniform_dp"],
)
def test_nosync_sm120_tp_dedup_rejects_unsupported_inputs(
    dp_size: int | None, require_custom_ar: bool
):
    """Reject unsupported routing and reduction backends before kernel launch."""
    pytest.importorskip("flashinfer.moe_ep")
    if not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")
    if torch.accelerator.device_count() < 2:
        pytest.skip("Need at least two GPUs")
    geometry = Geometry("sm120-nosync", 512, 256, 16, 2)
    distributed_run(
        partial(
            _nosync_sm120_layer_checks,
            dp_size=dp_size,
            dedup_uniform=True,
            require_custom_ar=require_custom_ar,
        ),
        2,
        geometry,
    )
