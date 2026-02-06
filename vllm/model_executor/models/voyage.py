# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import WeightsMapper

WeightItem = tuple[str, torch.Tensor]

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")


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

    We remap/fuse weights using generator pipeline and load directly
    (bypassing parent's stacked_params_mapping which would cause
    double-transformation like qkv_proj -> qkqkv_proj).
    """

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

    def _fuse_qkv_proj(self, weights: Iterable[WeightItem]) -> Iterable[WeightItem]:
        """Fuse q_proj, k_proj, v_proj into qkv_proj."""
        qkv_buf: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        qkv_suffixes = {
            "self_attn.q_proj.weight": "q",
            "self_attn.k_proj.weight": "k",
            "self_attn.v_proj.weight": "v",
        }

        for name, tensor in weights:
            m = _LAYER_RE.match(name)
            if m and m.group(2) in qkv_suffixes:
                layer_idx = int(m.group(1))
                qkv_buf[layer_idx][qkv_suffixes[m.group(2)]] = tensor
            else:
                yield name, tensor

        # Yield fused QKV weights
        for layer_idx in sorted(qkv_buf.keys()):
            parts = qkv_buf[layer_idx]
            if all(p in parts for p in ("q", "k", "v")):
                fused = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
                yield f"layers.{layer_idx}.self_attn.qkv_proj.weight", fused
            elif parts:
                missing = [p for p in ("q", "k", "v") if p not in parts]
                raise ValueError(f"Layer {layer_idx} missing QKV parts: {missing}")

    def _fuse_gate_up_proj(self, weights: Iterable[WeightItem]) -> Iterable[WeightItem]:
        """Fuse gate_proj and up_proj into gate_up_proj."""
        mlp_buf: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        mlp_suffixes = {
            "mlp.gate_proj.weight": "gate",
            "mlp.up_proj.weight": "up",
        }

        for name, tensor in weights:
            m = _LAYER_RE.match(name)
            if m and m.group(2) in mlp_suffixes:
                layer_idx = int(m.group(1))
                mlp_buf[layer_idx][mlp_suffixes[m.group(2)]] = tensor
            else:
                yield name, tensor

        # Yield fused gate_up weights
        for layer_idx in sorted(mlp_buf.keys()):
            parts = mlp_buf[layer_idx]
            if all(p in parts for p in ("gate", "up")):
                fused = torch.cat([parts["gate"], parts["up"]], dim=0)
                yield f"layers.{layer_idx}.mlp.gate_up_proj.weight", fused
            elif parts:
                missing = [p for p in ("gate", "up") if p not in parts]
                raise ValueError(f"Layer {layer_idx} missing MLP parts: {missing}")

    def load_weights(self, weights: Iterable[WeightItem]) -> set[str]:
        """Remap, fuse, and load weights using generator pipeline."""
        # Chain weight transformations
        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = self._fuse_qkv_proj(weights)
        weights = self._fuse_gate_up_proj(weights)

        # Load weights directly into model parameters
        # (bypass parent's stacked_params_mapping)
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
