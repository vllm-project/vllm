# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers modeling backend's MLA fuser.

The fuser must discover every MLA submodule structurally (never by assuming the
Transformers attribute names) and, when the query is low-rank, merge the checkpoint's
separate `q_a_proj`/`kv_a_proj_with_mqa` weights into the single fused down-projection.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.models.transformers.fuser import get_fuser
from vllm.model_executor.models.transformers.fusers import MLAFuser
from vllm.model_executor.models.transformers.fx_utils import trace

_FUSED_QKV_A_PROJ = MLAFuser.merged_name


def _match(q_lora_rank: int | None) -> MLAFuser | None:
    """Match a meta `DeepseekV2Attention` directly (bypassing the per-class
    `get_fuser` cache, so both q_lora variants of the same class are seen)."""
    pytest.importorskip("transformers.models.deepseek_v2.modeling_deepseek_v2")
    from transformers.models.deepseek_v2.configuration_deepseek_v2 import (
        DeepseekV2Config,
    )
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        DeepseekV2Attention,
    )

    cfg = DeepseekV2Config(
        hidden_size=256,
        num_attention_heads=16,
        kv_lora_rank=128,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
        q_lora_rank=q_lora_rank,
        num_hidden_layers=1,
    )
    with torch.device("meta"):
        attn = DeepseekV2Attention(cfg, layer_idx=0)
    return MLAFuser.match(trace(attn), attn)


def test_discovers_modules_without_q_lora():
    fuser = _match(q_lora_rank=None)
    assert isinstance(fuser, MLAFuser)
    assert not fuser.has_q_lora
    assert fuser.q_proj_name == "q_proj"
    assert fuser.kv_a_proj_name == "kv_a_proj_with_mqa"
    assert fuser.kv_a_layernorm_name == "kv_a_layernorm"
    assert fuser.kv_b_proj_name == "kv_b_proj"
    assert fuser.o_proj_name == "o_proj"
    assert fuser.q_a_proj_name is None
    # Nothing is stacked without a query LoRA.
    assert fuser.packed_modules_mapping == {}
    assert fuser.orig_to_new_stacked("model.layers.0.self_attn") == {}


def test_discovers_modules_with_q_lora():
    fuser = _match(q_lora_rank=64)
    assert isinstance(fuser, MLAFuser)
    assert fuser.has_q_lora
    assert fuser.q_a_proj_name == "q_a_proj"
    assert fuser.q_a_layernorm_name == "q_a_layernorm"
    assert fuser.q_b_proj_name == "q_b_proj"
    assert fuser.kv_a_proj_name == "kv_a_proj_with_mqa"
    assert fuser.kv_a_layernorm_name == "kv_a_layernorm"
    assert fuser.kv_b_proj_name == "kv_b_proj"
    assert fuser.o_proj_name == "o_proj"
    assert fuser.q_proj_name is None


def test_q_lora_stacks_qkv_a_proj():
    """The MLA layer reads `q_a_proj` and `kv_a_proj_with_mqa` fused into one
    down-projection, so both checkpoint weights must remap into it."""
    fuser = _match(q_lora_rank=64)
    assert isinstance(fuser, MLAFuser)
    prefix = "model.layers.0.self_attn"
    merged = f"{prefix}.{_FUSED_QKV_A_PROJ}"
    assert fuser.packed_modules_mapping == {
        _FUSED_QKV_A_PROJ: ["q_a_proj", "kv_a_proj_with_mqa"]
    }
    assert fuser.orig_to_new_stacked(prefix) == {
        f"{prefix}.q_a_proj": (merged, 0),
        f"{prefix}.kv_a_proj_with_mqa": (merged, 1),
    }


class _Norm(nn.Module):
    """A real RMSNorm computation: `match` verifies chain norms via `RMSNormFuser`."""

    def __init__(self, n: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + 1e-6))


class RenamedMLA(nn.Module):
    """An MLA-shaped attention whose children have non-standard names, proving
    discovery is by structure and not attribute name."""

    def __init__(self):
        super().__init__()
        heads, kv_lora, rope, nope, v, hidden = 4, 32, 8, 8, 16, 64
        self.alpha = nn.Linear(hidden, heads * (nope + rope))  # q_proj
        self.beta = nn.Linear(hidden, kv_lora + rope)  # kv_a_proj_with_mqa
        self.gamma = _Norm(kv_lora)  # kv_a_layernorm
        self.delta = nn.Linear(kv_lora, heads * (nope + v))  # kv_b_proj
        self.omega = nn.Linear(heads * v, hidden)  # o_proj (uncalled)
        self.kv_lora, self.rope = kv_lora, rope

    def forward(self, hidden_states):
        q = self.alpha(hidden_states)
        kv_lora, k_pe = torch.split(
            self.beta(hidden_states), [self.kv_lora, self.rope], dim=-1
        )
        expanded = self.delta(self.gamma(kv_lora))
        # Stand-in for the attention interface; `match` finds `o_proj` (omega) from
        # the source as the Linear producing the returned value.
        attn_output = expanded.sum() + q.sum() + k_pe.sum()
        attn_output = self.omega(attn_output)
        return attn_output


def test_discovers_modules_under_arbitrary_names():
    """Discovery is purely structural: `RenamedMLA` gives its children non-standard
    names, and `match` still locates each projection by dataflow."""
    with torch.device("meta"):
        module = RenamedMLA()
        fuser = MLAFuser.match(trace(module), module)
    assert isinstance(fuser, MLAFuser)
    assert not fuser.has_q_lora
    assert fuser.q_proj_name == "alpha"
    assert fuser.kv_a_proj_name == "beta"
    assert fuser.kv_a_layernorm_name == "gamma"
    assert fuser.kv_b_proj_name == "delta"
    assert fuser.o_proj_name == "omega"


class GLU(nn.Module):
    """A gated MLP: matches the GLU fuser, never MLA."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(16, 16)
        self.up = nn.Linear(16, 16)

    def forward(self, x):
        return self.up(F.silu(self.gate(x)))


def test_non_mla_is_not_matched():
    with torch.device("meta"):
        assert not isinstance(get_fuser(GLU()), MLAFuser)
