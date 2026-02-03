from __future__ import annotations

import re
from collections import defaultdict
from typing import Set, Tuple
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3Model

WeightItem = Tuple[str, torch.Tensor]

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


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

    We remap/fuse weights and load directly (bypassing parent's stacked_params_mapping
    which would cause double-transformation like qkv_proj -> qkqkv_proj).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Embedding head (hidden_size -> num_labels, bias=False)
        self.linear = nn.Linear(
            self.config.hidden_size,
            self.config.num_labels,
            bias=False,
        )

        # Patch get_kv_cache_spec to return None for encoder-only attention
        # This bypasses the "assert self.attn_type == AttentionType.DECODER" in vLLM 0.14.x
        self._patch_kv_cache_spec()

    def _patch_kv_cache_spec(self):
        """Patch get_kv_cache_spec to return None for encoder-only layers (no KV cache needed)."""
        for layer in getattr(self, "layers", []):
            if not hasattr(layer, "self_attn"):
                continue
            attn = getattr(layer.self_attn, "attn", None)
            if attn is not None and hasattr(attn, "get_kv_cache_spec"):
                # Patch to return None (encoder-only doesn't need KV cache)
                attn.get_kv_cache_spec = lambda vllm_config, self=attn: None

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return self.linear(out)

    def load_weights(self, weights: Iterable[WeightItem]) -> Set[str]:
        """Remap, fuse, and load weights directly (bypass parent's stacked_params_mapping)."""
        out_w: dict[str, torch.Tensor] = {}
        qkv_buf: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        mlp_buf: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)

        for name, tensor in weights:
            m = _LAYER_RE.match(name)
            if not m:
                # Non-layer weights: strip "model." prefix if present
                new_name = (
                    name[len("model.") :] if name.startswith("model.") else name
                )
                out_w[new_name] = tensor
                continue

            layer_idx = int(m.group(1))
            suffix = m.group(2)

            # Accumulate Q/K/V for fusion
            if suffix == "self_attn.q_proj.weight":
                qkv_buf[layer_idx]["q"] = tensor
                continue
            if suffix == "self_attn.k_proj.weight":
                qkv_buf[layer_idx]["k"] = tensor
                continue
            if suffix == "self_attn.v_proj.weight":
                qkv_buf[layer_idx]["v"] = tensor
                continue

            # Accumulate gate/up for fusion
            if suffix == "mlp.gate_proj.weight":
                mlp_buf[layer_idx]["gate"] = tensor
                continue
            if suffix == "mlp.up_proj.weight":
                mlp_buf[layer_idx]["up"] = tensor
                continue

            # Other layer weights: output with stripped prefix
            out_w[f"layers.{layer_idx}.{suffix}"] = tensor

        # Fuse Q/K/V -> qkv_proj
        for layer_idx, parts in qkv_buf.items():
            if "q" in parts and "k" in parts and "v" in parts:
                fused = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
                out_w[f"layers.{layer_idx}.self_attn.qkv_proj.weight"] = fused

        # Fuse gate/up -> gate_up_proj
        for layer_idx, parts in mlp_buf.items():
            if "gate" in parts and "up" in parts:
                fused = torch.cat([parts["gate"], parts["up"]], dim=0)
                out_w[f"layers.{layer_idx}.mlp.gate_up_proj.weight"] = fused

        # Load weights directly into model parameters (bypass parent's stacked_params_mapping)
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in out_w.items():
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(
                param, "weight_loader", default_weight_loader
            )
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
