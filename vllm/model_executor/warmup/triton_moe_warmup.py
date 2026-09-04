# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the Triton fused MoE expert GEMMs before serving.

``fused_moe_kernel`` is the expert GEMM whenever DeepGEMM declines the shape.
The most common case is ``N <= 512``, which every shard of a model with a
small ``moe_intermediate_size`` hits at high TP -- GLM-5.2-FP8 at TP=8 gives
``N=256``, so its MoE runs on Triton for every request.

Startup never compiles every variant that real batches select. Two things
vary with the token count ``M``: the tile config, and the 16-divisibility of
``EM`` and ``num_valid_tokens``, which Triton specializes on. Profiling and
CUDA graph capture only reach a few of the combinations, so the first request
landing on an uncompiled one pays the Triton JIT mid-serving.

This walks the ``M`` values that can select a distinct tile config, each
paired with a neighbour of opposite parity so both divisibility variants get
compiled, and runs the expert GEMMs once for every value.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

# M values that can select a distinct tile config: the union of the tuned
# config-file keys and the `get_default_config` breakpoints. Larger batches
# reuse the largest entry's config.
_WARMUP_M_GRID = (
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
)


def _triton_experts(module: torch.nn.Module) -> "TritonExperts | None":
    """Return the TritonExperts instance this MoE layer can dispatch to."""
    from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    if not isinstance(module, MoERunner):
        return None

    moe_kernel = getattr(module._quant_method, "moe_kernel", None)
    experts = getattr(getattr(moe_kernel, "impl", None), "fused_experts", None)
    if isinstance(experts, FallbackExperts):
        experts = experts.fallback_experts
    return experts if isinstance(experts, TritonExperts) else None


def _naive_block_assignment_max_m(
    global_num_experts: int,
    top_k: int,
    expert_map: torch.Tensor | None,
) -> int:
    """Largest M that still takes the naive block assignment path.

    Mirrors the condition in `_prepare_expert_assignment`, which is a
    ``naive_block_assignment`` constexpr to the kernel.
    """
    if expert_map is not None:
        return 0
    return global_num_experts // (top_k * 4)


def _warmup_m_values(max_tokens: int, naive_max_m: int) -> list[int]:
    values: set[int] = set()
    for m in _WARMUP_M_GRID:
        if m > max_tokens:
            break
        values.add(m)
        # A neighbour of opposite parity keeps the same tile config but flips
        # the 16-divisibility of EM and num_valid_tokens, which is a separate
        # Triton compile key. Config-file keys are almost all even, so without
        # this the non-divisible variant never gets compiled.
        values.add(min(m + 1, max_tokens))
    # The tile config just above the naive cutoff is shared by too few grid
    # entries to be guaranteed both parities, so pin them explicitly.
    values.update(m for m in (naive_max_m + 1, naive_max_m + 2) if 0 < m <= max_tokens)
    values.add(max_tokens)
    if max_tokens > 1:
        values.add(max_tokens - 1)
    return sorted(values)


def _warmup_expert_gemms(
    experts: "TritonExperts",
    layer: "RoutedExperts",
    max_tokens: int,
) -> None:
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    w13 = layer.w13_weight
    w2 = layer.w2_weight
    top_k = layer.top_k
    num_experts = w13.size(0)
    hidden_size = w2.size(1)
    device = w13.device
    act_dtype = layer.params_dtype
    quant_config = experts.quant_config

    m_values = _warmup_m_values(
        max_tokens,
        _naive_block_assignment_max_m(
            layer.global_num_experts, top_k, layer.expert_map
        ),
    )
    max_m = m_values[-1]

    # Allocate for the largest M once and slice; `apply` resizes the
    # workspaces it is given, so an oversized buffer is fine.
    hidden_states = torch.zeros((max_m, hidden_size), device=device, dtype=act_dtype)
    # Spread tokens over all experts so every expert block is exercised and
    # the token-to-block assignment matches a realistic batch.
    topk_ids = torch.randint(
        0, num_experts, (max_m, top_k), device=device, dtype=torch.int32
    )
    topk_weights = torch.ones((max_m, top_k), device=device, dtype=torch.float32)
    workspace13_shape, workspace2_shape, output_shape = experts.workspace_shapes(
        max_m,
        w13.size(1) // 2,
        hidden_size,
        top_k,
        layer.global_num_experts,
        layer.local_num_experts,
        None,
        layer.activation,
    )
    workspace_dtype = experts.workspace_dtype(act_dtype)
    workspace13 = torch.zeros(workspace13_shape, device=device, dtype=workspace_dtype)
    workspace2 = torch.zeros(workspace2_shape, device=device, dtype=workspace_dtype)
    output = torch.zeros(output_shape, device=device, dtype=act_dtype)

    for num_tokens in m_values:
        # Quantize exactly like the prepare step would, so the activation
        # dtype and scale layout reaching the kernel match serving.
        a1q, a1q_scale = moe_kernel_quantize_input(
            hidden_states[:num_tokens],
            quant_config.a1_scale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
            quantization_emulation=experts.quantization_emulation,
        )
        experts.apply(
            output=output[:num_tokens],
            hidden_states=a1q,
            w1=w13,
            w2=w2,
            topk_weights=topk_weights[:num_tokens],
            topk_ids=topk_ids[:num_tokens],
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            a1q_scale=a1q_scale,
            a2_scale=quant_config.a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )


def triton_moe_warmup(worker: "Worker") -> None:
    """Compile the Triton fused MoE GEMMs for the tile configs used at runtime."""
    if not current_platform.is_cuda_alike():
        return
    if worker.model_runner.is_pooling_model:
        return

    # Above the largest grid entry the tile config no longer changes, so
    # there is nothing to gain from warming the full token budget.
    max_tokens = min(worker.scheduler_config.max_num_batched_tokens, _WARMUP_M_GRID[-1])
    if max_tokens <= 0:
        return

    # All MoE layers of a model share weight shapes, so one representative
    # per (w13, w2) shape pair covers every layer.
    layers: dict[tuple[torch.Size, torch.Size], tuple] = {}
    for module in worker.get_model().modules():
        experts = _triton_experts(module)
        if experts is None:
            continue
        layer = module.routed_experts
        layers.setdefault(
            (layer.w13_weight.shape, layer.w2_weight.shape), (experts, layer)
        )

    if not layers:
        return

    logger.info_once(
        "Warming up Triton fused MoE kernels for %d weight shape(s).", len(layers)
    )
    with torch.inference_mode():
        for experts, layer in layers.values():
            try:
                _warmup_expert_gemms(experts, layer, max_tokens)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "Skipping Triton fused MoE warmup: out of memory while "
                    "building warmup activations."
                )
                break
        torch.accelerator.synchronize()
