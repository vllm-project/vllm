# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash MonoKernel spec: FFN-layer launches for the MoE layers.

Config-time code imports only this module; the model-side callbacks import the
kernel lazily."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.models.common.mono.spec import MonoSpec
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
}


def kernel_layers(hf: Any) -> list[int]:
    """The MoE layers, except the last (its mHC post contracts the streams)."""
    types = getattr(hf, "mlp_layer_types", None) or []
    last = hf.num_hidden_layers - 1
    return [i for i, t in enumerate(types[:last]) if t == "sparse"]


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
    caches={},
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
        "mode": "ffn layer",
        "numerics": NUMERICS,
    },
)
