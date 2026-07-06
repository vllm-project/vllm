# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fuse common Linear groups in HF modules into vLLM packed layers.

Detects the canonical SwiGLU MLP shape (gate_proj/up_proj sharing input,
followed by down_proj) and the canonical attention QKV shape (q_proj/k_proj/
v_proj sharing input, followed by o_proj) and replaces them with vLLM's
QKVParallelLinear / MergedColumnParallelLinear, then rebinds the parent
module's forward to call the fused projection once instead of 2-3 times.

Detection is structural: child module names + Linear shapes. We additionally
require that the children that *will* be fused are listed in the model's
tp_plan as colwise — this prevents accidentally fusing layers that were
intentionally not column-parallel-shardable.

Models without the canonical naming or shape just don't match and fall
through to the existing per-Linear replacement path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import regex as re
import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.models.utils import maybe_prefix

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)


# Records what was fused, used to drive weight loading. One per fused group.
class FusedGroup:
    __slots__ = ("kind", "fused_qualname", "source_qualnames")

    def __init__(self, kind: str, fused_qualname: str, source_qualnames: list[str]):
        # kind: "qkv" or "gate_up"
        self.kind = kind
        self.fused_qualname = fused_qualname  # e.g. "model.layers.0.self_attn.qkv_proj"
        self.source_qualnames = source_qualnames  # e.g. ["...q_proj", "...k_proj", "...v_proj"]


def _tp_plan_matches(qualname: str, style: str, tp_plan: dict[str, str]) -> bool:
    """Return True iff `qualname` matches a tp_plan entry with the given style."""
    for pattern, plan_style in tp_plan.items():
        if plan_style != style:
            continue
        if re.match(pattern, qualname):
            return True
    return False


def _get_act_fn_name(parent: nn.Module) -> str | None:
    """Return the canonical activation name for a parent that has an act_fn."""
    act = getattr(parent, "act_fn", None)
    if act is None:
        return None
    # ACT2FN entries are torch.nn modules (e.g. SiLUActivation). Fall back to class name.
    return type(act).__name__.lower()


def _is_silu(parent: nn.Module) -> bool:
    name = _get_act_fn_name(parent)
    if name is None:
        return False
    return "silu" in name or "swish" in name


def apply_fused_linears(
    model: nn.Module,
    tp_plan: dict[str, str],
    quant_config: "QuantizationConfig | None",
    *,
    head_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    prefix: str = "model",
) -> list[FusedGroup]:
    """Walk `model` and replace fusable {q,k,v}/{gate,up} groups with packed layers.

    Returns the list of fused groups (used by weight loading to map checkpoint
    names to fused layer shards).
    """
    # tp_plan keys live without the "model." prefix; we walk starting at `model`
    # and use qualnames *with* that prefix, so prefix tp_plan to match.
    tp_plan = {maybe_prefix("model", k): v for k, v in tp_plan.items()}

    fused: list[FusedGroup] = []

    def _walk(module: nn.Module, qualprefix: str) -> None:
        # Try to fuse children of this module before recursing.
        _maybe_fuse_qkv(module, qualprefix, tp_plan, quant_config,
                        head_dim, num_attention_heads, num_key_value_heads, fused)
        _maybe_fuse_gate_up(module, qualprefix, tp_plan, quant_config, fused)
        for child_name, child in module.named_children():
            child_qual = maybe_prefix(qualprefix, child_name)
            _walk(child, child_qual)

    _walk(model, prefix)
    return fused


# ---------------------------------------------------------------------------
# QKV fusion
# ---------------------------------------------------------------------------


def _maybe_fuse_qkv(
    parent: nn.Module,
    parent_qual: str,
    tp_plan: dict[str, str],
    quant_config: "QuantizationConfig | None",
    head_dim: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    fused: list[FusedGroup],
) -> None:
    children = dict(parent.named_children())
    if not all(n in children for n in ("q_proj", "k_proj", "v_proj")):
        return
    q, k, v = children["q_proj"], children["k_proj"], children["v_proj"]
    # All three must be plain nn.Linear (not yet replaced) and share in_features.
    if not all(isinstance(m, nn.Linear) for m in (q, k, v)):
        return
    if not (q.in_features == k.in_features == v.in_features):
        return
    # tp_plan must mark all three as colwise.
    q_qual = maybe_prefix(parent_qual, "q_proj")
    k_qual = maybe_prefix(parent_qual, "k_proj")
    v_qual = maybe_prefix(parent_qual, "v_proj")
    if not all(_tp_plan_matches(qn, "colwise", tp_plan) for qn in (q_qual, k_qual, v_qual)):
        return
    # Sanity-check shapes against config.
    expected_q = num_attention_heads * head_dim
    expected_kv = num_key_value_heads * head_dim
    if (q.out_features, k.out_features, v.out_features) != (expected_q, expected_kv, expected_kv):
        # Non-standard head shape; bail out rather than risk wrong sharding.
        return
    has_bias = q.bias is not None
    if (k.bias is not None) != has_bias or (v.bias is not None) != has_bias:
        return  # all three must agree on bias

    fused_qual = maybe_prefix(parent_qual, "qkv_proj")
    qkv = QKVParallelLinear(
        hidden_size=q.in_features,
        head_size=head_dim,
        total_num_heads=num_attention_heads,
        total_num_kv_heads=num_key_value_heads,
        bias=has_bias,
        quant_config=quant_config,
        prefix=fused_qual,
        return_bias=False,
    )

    # Stash q/k/v output sizes (per-rank) for forward-time split.
    q_size = qkv.num_heads * qkv.head_size
    kv_size = qkv.num_kv_heads * qkv.head_size

    # Install on parent and remove the originals so HF doesn't iterate them.
    parent.qkv_proj = qkv
    del parent.q_proj
    del parent.k_proj
    del parent.v_proj

    parent._fused_qkv_q_size = q_size
    parent._fused_qkv_kv_size = kv_size
    parent._fused_qkv_head_dim = head_dim

    # Rebind forward.
    parent.forward = _fused_attention_forward.__get__(parent, type(parent))

    fused.append(FusedGroup(
        kind="qkv",
        fused_qualname=fused_qual,
        source_qualnames=[q_qual, k_qual, v_qual],
    ))
    logger.debug("Fused QKV at %s", parent_qual)


def _fused_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    """Drop-in replacement for HF attention.forward that calls qkv_proj once.

    Mirrors the canonical HF decoder attention pattern (granite/llama/qwen2/
    qwen3/mistral): project hidden_states -> [q;k;v], split, reshape into
    head layout, optional rotary, dispatch to attention_interface (which will
    be vllm_flash_attention_forward in this backend), o_proj.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    qkv = self.qkv_proj(hidden_states)
    q_size = self._fused_qkv_q_size
    kv_size = self._fused_qkv_kv_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    head_dim = self._fused_qkv_head_dim
    input_shape = hidden_states.shape[:-1]
    # [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
    query_states = q.view(*input_shape, -1, head_dim).transpose(1, 2)
    key_states = k.view(*input_shape, -1, head_dim).transpose(1, 2)
    value_states = v.view(*input_shape, -1, head_dim).transpose(1, 2)

    if position_embeddings is not None:
        # Lazy import to avoid forcing a transformers dep at module load.
        from transformers.models.granite.modeling_granite import apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # No-op for vLLM (no past_key_values cache - vLLM owns the KV cache).
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, None
    )
    # Pass attention_multiplier (granite) / standard scaling.
    scaling = getattr(self, "scaling", None)
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# gate/up fusion
# ---------------------------------------------------------------------------


def _maybe_fuse_gate_up(
    parent: nn.Module,
    parent_qual: str,
    tp_plan: dict[str, str],
    quant_config: "QuantizationConfig | None",
    fused: list[FusedGroup],
) -> None:
    children = dict(parent.named_children())
    if not all(n in children for n in ("gate_proj", "up_proj", "down_proj")):
        return
    gate, up, down = children["gate_proj"], children["up_proj"], children["down_proj"]
    if not all(isinstance(m, nn.Linear) for m in (gate, up, down)):
        return
    if gate.in_features != up.in_features:
        return
    if gate.out_features != up.out_features:
        return
    if not _is_silu(parent):
        return
    gate_qual = maybe_prefix(parent_qual, "gate_proj")
    up_qual = maybe_prefix(parent_qual, "up_proj")
    if not all(_tp_plan_matches(qn, "colwise", tp_plan) for qn in (gate_qual, up_qual)):
        return
    has_bias = gate.bias is not None
    if (up.bias is not None) != has_bias:
        return

    fused_qual = maybe_prefix(parent_qual, "gate_up_proj")
    gate_up = MergedColumnParallelLinear(
        input_size=gate.in_features,
        output_sizes=[gate.out_features, up.out_features],
        bias=has_bias,
        quant_config=quant_config,
        prefix=fused_qual,
        return_bias=False,
    )

    parent.gate_up_proj = gate_up
    del parent.gate_proj
    del parent.up_proj

    if not hasattr(parent, "_silu_and_mul"):
        parent.add_module("_silu_and_mul", SiluAndMul())
    parent.forward = _fused_mlp_forward.__get__(parent, type(parent))

    fused.append(FusedGroup(
        kind="gate_up",
        fused_qualname=fused_qual,
        source_qualnames=[gate_qual, up_qual],
    ))
    logger.debug("Fused gate/up at %s", parent_qual)


def _fused_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_up = self.gate_up_proj(x)
    out = self._silu_and_mul(gate_up)
    return self.down_proj(out)


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------


def stacked_params_mapping_from_groups(
    groups: list[FusedGroup],
) -> list[tuple[str, str, "str | int"]]:
    """Build a (fused_suffix, source_suffix, shard_id) list usable by the
    standard vLLM weight-loading loop.

    We deduplicate by (fused_short, source_short) so we emit one mapping per
    distinct (fused_proj, source_proj) shape regardless of how many layers
    share it.
    """
    QKV_IDS = {"q_proj": "q", "k_proj": "k", "v_proj": "v"}
    GATE_UP_IDS = {"gate_proj": 0, "up_proj": 1}

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, "str | int"]] = []
    for g in groups:
        for src in g.source_qualnames:
            src_short = "." + src.rsplit(".", 1)[-1]
            fused_short = "." + g.fused_qualname.rsplit(".", 1)[-1]
            key = (fused_short, src_short)
            if key in seen:
                continue
            seen.add(key)
            if g.kind == "qkv":
                shard_id = QKV_IDS[src_short.lstrip(".")]
            else:
                shard_id = GATE_UP_IDS[src_short.lstrip(".")]
            out.append((fused_short, src_short, shard_id))
    return out
