# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models.utils import WeightsMapper

# Parameter aliases belong to checkpoint loading, not quantization module names.
HYV4_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        ".gate.e_score_correction_bias": ".expert_bias",
        ".router.gate.": ".gate.",
    },
    orig_to_new_suffix={
        ".hc_fn": ".hc_fn.weight",
        ".hc_head_fn": ".hc_head_fn.weight",
    },
)

HYV4_MLP_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
        ".mlp.up_proj": (".mlp.gate_up_proj", 1),
        ".shared_experts.gate_proj": (".shared_experts.gate_up_proj", 0),
        ".shared_experts.up_proj": (".shared_experts.gate_up_proj", 1),
    }
)

HYV4_PACKED_MODULES_MAPPING = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    # MLA runs both latent down-projections as one GEMM.
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
}


def remap_block_fp8_scale(
    name: str, weight: torch.Tensor, quant_config: QuantizationConfig | None
) -> tuple[str, torch.Tensor]:
    """Map a block-FP8 checkpoint `.scale` onto the `weight_scale_inv` param."""
    if (
        isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size is not None
        and name.endswith(".scale")
    ):
        name = name.removesuffix(".scale") + ".weight_scale_inv"
        if weight.dtype == torch.uint8:
            # UINT8 stores UE8M0 exponents, not numeric scales.
            weight = weight.view(torch.float8_e8m0fnu)
    return name, weight
