# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

from .replicated_linear import ReplicatedLinearWithLoRA


class GateLinearWithLoRA(ReplicatedLinearWithLoRA):
    def __init__(self, base_layer: GateLinear) -> None:
        super().__init__(
            base_layer,
        )

    # GateLinearWithLoRA should always be replaced, regardless of the fully
    # sharded LoRAs setting, because it is, by definition, copied per GPU.
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is maybe_get_oot_by_class(GateLinear)
