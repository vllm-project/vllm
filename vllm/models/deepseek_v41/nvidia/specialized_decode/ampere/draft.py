# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft model's specialized decode executor on Ampere SM80."""

from typing import TYPE_CHECKING

import torch

from ..runtime.decode import LayerPlan
from .dspark_fused import DSparkFusedMarkovSampler
from .steps import AmpereAttentionStep, AmpereMoEStep

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.nvidia.dspark import DSparkDeepseekV4Model

    from .decode import AmpereSpecializedDecode


class AmpereDraftDecode:
    """Specialized executor for DSpark draft backbone on Ampere SM80."""

    def __init__(
        self,
        target_executor: "AmpereSpecializedDecode",
        backbone: "DSparkDeepseekV4Model",
    ) -> None:
        self.target_executor = target_executor
        self.backbone = backbone
        self.max_rows = target_executor.max_rows

        layers = backbone.layers
        self.plans = [
            LayerPlan(
                layer_id=idx,
                layer=layer,
                attention=AmpereAttentionStep(layer.attn),
                moe=AmpereMoEStep(layer.ffn, self.max_rows),
                engram=None,
            )
            for idx, layer in enumerate(layers)
        ]

        # Sampler will be built when embed weights are available
        self.fused_sampler: DSparkFusedMarkovSampler | None = None

    def build_sampler(
        self, max_num_reqs: int, vocab_size: int, embed_weight: torch.Tensor
    ):
        self.fused_sampler = DSparkFusedMarkovSampler(
            max_num_reqs, vocab_size, embed_weight, use_fp64=False
        )
        return self.fused_sampler
