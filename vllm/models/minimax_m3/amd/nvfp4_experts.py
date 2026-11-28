# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 routed experts of MiniMax-M3.1 checkpoints on the AITER HIP kernels.

The routed experts ship as NVFP4 (`moe_quant_algo: NVFP4`). Without
`--quantization-config.moe`, gfx950 feeds the checkpoint NVFP4 weights to
`aiter.ops.moe_nvfp4` for every T; no modular kernel, so SP prefill is off.
`--quantization-config.moe mxfp4|mxfp8` re-quantizes them online instead.

The kernels return the unscaled routed sum (the runner applies the routed
scaling) and hard-code SwiGLU-OAI with alpha 1.702, beta 1 and limit 7.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.platforms.rocm import on_gfx950

logger = init_logger(__name__)

_SWIGLU_ALPHA = 1.702
_SWIGLU_BETA = 1.0
_SWIGLU_LIMIT = 7.0
# Checkpoint parameters of CompressedTensorsW4A4Nvfp4MoEMethod.create_weights.
_CKPT_PARAMS = tuple(
    f"{w}_{p}"
    for w in ("w13", "w2")
    for p in ("weight_packed", "weight_scale", "weight_global_scale")
    + ("input_global_scale",)
)


def nvfp4_moe_supported() -> bool:
    if not on_gfx950():
        return False
    try:
        import aiter.ops.moe_nvfp4  # noqa: F401
    except ImportError:
        return False
    return True


def _check_layer(layer) -> None:
    cfg = layer.moe_config
    if cfg.moe_parallel_config.use_ep:
        raise NotImplementedError("MiniMax-M3 NVFP4 HIP MoE: EP is unsupported")
    if layer.expert_map_manager.num_fused_shared_experts:
        raise NotImplementedError(
            "MiniMax-M3 NVFP4 HIP MoE: fused shared experts are unsupported"
        )
    if (
        cfg.hidden_dim != cfg.hidden_dim_unpadded
        or cfg.intermediate_size_per_partition
        != cfg.intermediate_size_per_partition_unpadded
    ):
        raise NotImplementedError("MiniMax-M3 NVFP4 HIP MoE: padded MoE sizes")
    if (
        layer.activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE
        or layer.swiglu_alpha != _SWIGLU_ALPHA
        or layer.swiglu_limit != _SWIGLU_LIMIT
        or layer.swiglu_beta not in (None, _SWIGLU_BETA)
        or layer.apply_router_weight_on_input
    ):
        raise NotImplementedError(
            f"MiniMax-M3 NVFP4 HIP MoE: needs SwiGLU-OAI (alpha {_SWIGLU_ALPHA}, beta "
            f"{_SWIGLU_BETA}, limit {_SWIGLU_LIMIT}) and router weights on the output"
        )


class MiniMaxM3NvFp4HipMoEMethod(CompressedTensorsW4A4Nvfp4MoEMethod):
    """Checkpoint NVFP4 experts on `aiter.ops.moe_nvfp4`."""

    nvfp4_weights = None

    @property
    def is_monolithic(self) -> bool:
        return False

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        return False

    def get_fused_moe_quant_config(self, layer):
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def process_weights_after_loading(self, layer) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        from aiter.ops.moe_nvfp4 import NvFp4MoeWeights

        _check_layer(layer)
        # Dequant = code * scale / global_scale.
        self.nvfp4_weights = NvFp4MoeWeights(
            layer.w13_weight_packed.data,
            layer.w13_weight_scale.data,
            layer.w2_weight_packed.data,
            layer.w2_weight_scale.data,
            1.0 / layer.w13_weight_global_scale.data.float(),
            1.0 / layer.w2_weight_global_scale.data.float(),
        )
        # NvFp4MoeWeights holds shuffled copies.
        for name in _CKPT_PARAMS:
            delattr(layer, name)
        self.moe_quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input,
    ) -> torch.Tensor:
        from aiter.ops.moe_nvfp4 import moe_nvfp4

        return moe_nvfp4(self.nvfp4_weights, x, topk_ids, topk_weights)


def keep_nvfp4_experts(experts) -> None:
    """Route `experts` (a `FusedMoEFactory` runner with checkpoint NVFP4
    experts) through the HIP kernels. Call before weight loading."""
    qm = experts.routed_experts.quant_method
    if type(qm) is not CompressedTensorsW4A4Nvfp4MoEMethod:
        raise ValueError(f"MiniMax-M3 NVFP4 HIP MoE: unexpected {type(qm).__name__}")
    qm.__class__ = MiniMaxM3NvFp4HipMoEMethod
    logger.info_once(
        "MiniMax-M3 NVFP4 routed experts: AITER moe_nvfp4 for every T. Prefill "
        "(large T) may be slower than with --quantization-config.moe mxfp4"
    )
