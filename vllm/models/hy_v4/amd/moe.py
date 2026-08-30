# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PretrainedConfig

from vllm.model_executor.layers.fused_moe.utils import (
    resolve_layer_fused_shared_expert,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.hy_v4.nvidia.moe import HYV4MoEFused as BaseMoEFused


class HYV4MoEFused(BaseMoEFused):
    """HY V4 MoE with compatible routed/shared-expert fusion."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        fuse_shared_experts = False
        if config.num_shared_experts > 0:
            fuse_shared_experts = resolve_layer_fused_shared_expert(
                quant_config,
                prefix,
            )
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            enable_eplb=enable_eplb,
            fuse_shared_experts=fuse_shared_experts,
        )
