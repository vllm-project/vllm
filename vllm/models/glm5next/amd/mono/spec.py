# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash MonoKernel spec: whole-layer launches for the KDA and the
sparse-attention MoE layers.

Config-time code imports only this module; the model-side callbacks import the
kernel lazily."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.models.common.mono.spec import CacheReq, MonoSpec
from vllm.models.glm5next.amd.ops.mono.ffn.layout import POLL_STAGES

if TYPE_CHECKING:
    from vllm.config import VllmConfig

TP_SIZES = (2, 4, 8)
WIDTHS = (2, 4, 8)
# wider steps at these TP sizes are slower than vLLM's path
TP_MAX_WIDTH = {2: 4}
STEP_TOKENS = (1, 2, 4, 8)
GRID = 256
NUMERICS = (
    "expert inputs quantized to MXFP8 (vLLM: MXFP4)",
    "attention TP reduce summed in BF16 inside the launch",
    "KDA conv taps, L2 norms, gates and recurrence in FP32, as vLLM's kernels",
    "sparse-attention FP8 projections on bf16 activations (vLLM: FP8 per 128)",
    "W_UK / W_UV in bf16 from kv_b_proj (vLLM: AITER FP4 / FP8 BMM)",
    "split-attention partial sums stored in bf16 before the merge",
    (
        "a 2048-token selection a tail overflows drops the last positions of its "
        "lowest-scored pool (vLLM: the last of its top-k order)"
    ),
)
GEOMETRY = {
    "hidden_size": 4096,
    "n_routed_experts": 288,
    "num_experts_per_tok": 8,
    "n_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "swiglu_limit": 10.0,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "mhc": True,
    "linear_num_heads": 64,
    "linear_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "linear_lower_bound": -5.0,
}


def kernel_layers(hf: Any) -> list[int]:
    """The MoE layers, except the last (its mHC post contracts the streams)."""
    types = getattr(hf, "mlp_layer_types", None) or []
    last = hf.num_hidden_layers - 1
    return [i for i, t in enumerate(types[:last]) if t == "sparse"]


def is_kda(layer: Any) -> bool:
    return getattr(layer, "layer_kind", None) == "kda"


def is_dsa(layer: Any) -> bool:
    return getattr(layer, "layer_kind", None) == "mla"


def refuse_config(vllm_config: VllmConfig) -> list[str]:
    from vllm.platforms import current_platform

    hf = vllm_config.model_config.hf_text_config
    reasons = []
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx950

        if not on_gfx950():
            reasons.append("needs gfx950")
    else:
        reasons.append("needs ROCm")
    if getattr(hf, "model_type", None) != "glm5_next_text":
        reasons.append("needs glm5_next")
    else:
        for key, want in GEOMETRY.items():
            if getattr(hf, key, None) != want:
                reasons.append(f"needs {key} {want}, got {getattr(hf, key, None)}")
    if vllm_config.parallel_config.tensor_parallel_size not in TP_SIZES:
        reasons.append("needs tensor parallel size 2, 4 or 8")
    if vllm_config.parallel_config.use_sequence_parallel_moe:
        reasons.append("sequence-parallel MoE")
    if vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        reasons.append(f"{vllm_config.cache_config.cache_dtype} KV cache (needs bf16)")
    ssm = getattr(vllm_config.cache_config, "mamba_ssm_cache_dtype", "auto")
    if ssm not in ("auto", "float32"):
        reasons.append(f"{ssm} KDA recurrent state (needs float32)")
    if vllm_config.speculative_config is not None:
        reasons.append("speculative decoding (a wider kpool tail ring)")
    return reasons


def _model_side(name: str):
    def call(*args, **kwargs):
        from vllm.models.glm5next.amd.mono import kernel

        return getattr(kernel, name)(*args, **kwargs)

    return call


MONO_SPEC = MonoSpec(
    name="GLM-5.3-Flash",
    layers=kernel_layers,
    widths=WIDTHS,
    supports=frozenset(),
    caches={
        "conv": CacheReq("self_attn.kv_cache[0]", (torch.bfloat16,), when=is_kda),
        "rec": CacheReq("self_attn.kv_cache[1]", (torch.float32,), when=is_kda),
        "mla": CacheReq(
            "self_attn.mla_attn.mla_attn.kv_cache", (torch.bfloat16,), when=is_dsa
        ),
        "icache": CacheReq(
            "self_attn.indexer.k_cache.kv_cache", (torch.uint8,), when=is_dsa
        ),
        "tail": CacheReq(
            "self_attn.indexer.tail_cache.kv_cache", (torch.bfloat16,), when=is_dsa
        ),
    },
    weight_source="views",
    grid=GRID,
    stages=POLL_STAGES,
    refuse_config=refuse_config,
    refuse_model=_model_side("refuse_model"),
    pack=_model_side("pack"),
    bind=_model_side("bind"),
    eligible=_model_side("eligible"),
    build=_model_side("build"),
    metadata_names=_model_side("metadata_names"),
    options={
        "tokens": STEP_TOKENS,
        "tp_max_width": TP_MAX_WIDTH,
        "mode": "whole layer (KDA and sparse attention)",
        "numerics": NUMERICS,
    },
)
