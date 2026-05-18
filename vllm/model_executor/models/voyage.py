# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

WeightItem = tuple[str, torch.Tensor]


class VoyageQwen3BidirectionalEmbedModel(Qwen3Model):
    """
    Qwen3Model + Voyage embedding head + bidirectional attention.

    Checkpoint conventions (HF):
      - MLP: gate_proj + up_proj (unfused)
      - Attn: q_proj + k_proj + v_proj (unfused)
      - Linear head: linear.weight
      - Weights prefixed with "model." (e.g., model.layers.0...)

    vLLM Qwen3Model expects:
      - mlp.gate_up_proj (fused)
      - self_attn.qkv_proj (fused)
      - No "model." prefix

    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Embedding head (hidden_size -> num_labels, bias=False)
        self.linear = nn.Linear(
            self.config.hidden_size,
            self.config.num_labels,
            bias=False,
        )

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return self.linear(out)

    def load_weights(self, weights: Iterable[WeightItem]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
