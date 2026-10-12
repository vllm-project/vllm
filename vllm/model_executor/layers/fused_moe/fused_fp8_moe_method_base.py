# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    maybe_round_up_hidden_size_and_intermediate_size,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


class FusedFp8MoEMethodBase(FusedMoEMethodBase):
    """Base for FP8 MoE quant methods; rounds the layer sizes up to the FP8
    backend's alignment. Subclasses set ``fp8_backend`` and ``weight_quant_key``.
    """

    fp8_backend: Fp8MoeBackend
    weight_quant_key: QuantKey

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size,
            intermediate_size_per_partition,
            act_dtype,
            moe_parallel_config,
        )
        return maybe_round_up_hidden_size_and_intermediate_size(
            self.fp8_backend,
            self.weight_quant_key,
            self.moe.activation,
            hidden_size,
            intermediate_size_per_partition,
        )
