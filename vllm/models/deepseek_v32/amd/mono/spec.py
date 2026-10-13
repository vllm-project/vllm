# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the GLM-5 MonoKernel supports: config refusals and the per-step decision.

Plain Python on purpose, so the rules are testable on CPU.
"""

from __future__ import annotations

from typing import Any

TP_SIZES = (4, 8)
SPARSE_TOPK = 2048
# Decode steps of these token counts take the kernel; one token runs as a
# two-row launch with a pad row.
STEP_TOKENS = (1, 2, 4, 8)
# One workgroup per CU, all resident: fewer CUs (a partitioned GPU) deadlocks.
GRID = 256
# Re-polls of one mailbox wait before it expires (about 4 s).
POLL_LIMIT = 20_000_000
# Where the kernel's arithmetic differs from vLLM's layers.
NUMERICS = (
    "expert inputs quantized to MXFP8 (vLLM: MXFP4)",
    "TP all-reduces in BF16 inside the launch",
)


def width_for(tokens: int) -> int:
    return max(tokens, 2)


def refuse_config(vllm_config: Any, on_gfx950: bool, compute_units: int) -> list[str]:
    """Every reason this deployment cannot run the kernel (empty: supported)."""
    from vllm import envs
    from vllm.config.compilation import CUDAGraphMode

    pc = vllm_config.parallel_config
    mc = vllm_config.model_config
    cc = vllm_config.compilation_config
    offload = vllm_config.offload_config
    hf = mc.hf_config
    decode_graphs = cc.cudagraph_mode.decode_mode() if cc.cudagraph_mode else None
    checks = (
        (on_gfx950, "needs gfx950"),
        (
            compute_units >= GRID,
            f"needs {GRID} compute units, the GPU has {compute_units}",
        ),
        (getattr(hf, "model_type", None) == "glm_moe_dsa", "needs glm_moe_dsa"),
        (pc.tensor_parallel_size in TP_SIZES, "needs tensor parallel size 4 or 8"),
        (pc.pipeline_parallel_size == 1, "pipeline parallel"),
        (pc.data_parallel_size == 1, "data parallel"),
        (not pc.enable_expert_parallel, "expert parallel"),
        (not pc.enable_eplb, "EPLB"),
        (pc.decode_context_parallel_size == 1, "decode context parallel"),
        (pc.prefill_context_parallel_size == 1, "prefill context parallel"),
        (vllm_config.speculative_config is None, "speculative decoding"),
        (vllm_config.lora_config is None, "LoRA"),
        (vllm_config.kv_transfer_config is None, "KV transfer connector"),
        (
            not vllm_config.aux_output_config.enable_return_routed_experts,
            "routed-expert capture",
        ),
        (not mc.enable_prompt_embeds, "prompt embeds"),
        (not mc.enable_sleep_mode, "sleep mode"),
        (
            offload.uva.cpu_offload_gb == 0
            and offload.prefetch.offload_group_size == 0,
            "CPU weight offload",
        ),
        (
            vllm_config.cache_config.cache_dtype in ("fp8", "fp8_e4m3"),
            "needs --kv-cache-dtype fp8",
        ),
        (
            getattr(hf, "indexer_rope_interleave", False),
            "needs indexer_rope_interleave",
        ),
        (vllm_config.use_v2_model_runner, "needs Model Runner V2"),
        (
            decode_graphs != CUDAGraphMode.PIECEWISE,
            "piecewise-only CUDA graphs (decode needs FULL graphs or eager)",
        ),
        (not envs.VLLM_USE_BREAKABLE_CUDAGRAPH, "breakable CUDA graphs"),
    )
    return [reason for ok, reason in checks if not ok]


def step_reason(
    tokens: int, metadata: Any, layer_name: str, piecewise: bool = False
) -> str | None:
    """Why this forward is not a kernel step, or None. Depends only on what every
    TP rank builds alike from the scheduler output. A piecewise CUDA graph is
    replayed for mixed batches, so it never records a kernel launch."""
    if piecewise:
        return "piecewise CUDA graph"
    if tokens not in STEP_TOKENS:
        return f"{tokens} tokens"
    if not isinstance(metadata, dict) or layer_name not in metadata:
        return "no attention metadata"
    if metadata[layer_name].max_query_len != 1:
        return "not a pure decode step"
    return None
