# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers modeling backend's linear fusers."""

import inspect
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.models.transformers.fuser import get_fuser
from vllm.model_executor.models.transformers.fusers import (
    GLUFuser,
    PackedQKVFuser,
    QKVFuser,
    packed_qkv,
    qkv,
)


class SiluAndMulStub(nn.Module):
    """Stand-in for vLLM's `SiluAndMul` (no vLLM config required)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


class NoDownGLU(nn.Module):
    """`act(gate(x)) * up(x)` with no output projection -> `down_name` is None."""

    def __init__(self, hidden: int = 16, inter: int = 32, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=bias)
        self.up_proj = nn.Linear(hidden, inter, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.act_fn(self.gate_proj(x)) * self.up_proj(x)


class GLUMLP(NoDownGLU):
    """`down(act(gate(x)) * up(x))` — the canonical HF GLU MLP."""

    def __init__(self, hidden: int = 16, inter: int = 32, bias: bool = False):
        super().__init__(hidden, inter, bias)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ReversedGLUMLP(GLUMLP):
    """`up(x) * act(gate(x))` — operands swapped (multiply is commutative)."""

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x)))


class NotAnMLP(nn.Module):
    """Two linears but no activation*linear multiply -> must not match."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class NotAnActGLUMLP(GLUMLP):
    """GLU-shaped, but the "activation" is not a known activation module."""

    def __init__(self):
        super().__init__()
        self.act_fn = nn.Dropout()


class UntraceableMLP(GLUMLP):
    """Data-dependent control flow *before* the GLU -> no match."""

    def forward(self, x):
        if x.sum() > 0:  # noqa: SIM108 - intentionally untraceable
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return x


class UntraceableTailGLUMLP(GLUMLP):
    """Data-dependent control flow *after* the GLU -> still fusable."""

    def forward(self, x):
        y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if y.sum() > torch.inf:  # intentionally untraceable
            y = y * 0
        return y


class FakeAttention(nn.Module):
    """HF v5-style attention: shape unpacking, dead KV branch, kwargs interface."""

    is_causal = True

    def __init__(
        self,
        hidden: int = 32,
        head_dim: int = 8,
        heads: int = 4,
        kv_heads: int = 4,
        bias: bool = False,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="vllm")
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=bias)

    def forward(
        self, hidden_states, attention_mask=None, past_key_values=None, **kwargs
    ):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, None
        )
        attn_output, attn_weights = attention_interface(
            self, q, k, v, attention_mask, scaling=self.scaling, **kwargs
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class ReversedFakeAttention(FakeAttention):
    """Projections computed in (v, k, q) order — q must still be identified."""

    def forward(
        self, hidden_states, attention_mask=None, past_key_values=None, **kwargs
    ):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, None
        )
        attn_output, _ = attention_interface(
            self, q, k, v, attention_mask, scaling=self.scaling, **kwargs
        )
        return self.o_proj(attn_output.reshape(*input_shape, -1)), None


class ExtraProjAttention(FakeAttention):
    """A second non-qkv linear of a different width -> `o_proj` still found."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sink_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)


class QKNormAttention(FakeAttention):
    """OLMoE-style: a full-dim norm applied to the whole q/k projection output."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q_norm = nn.RMSNorm(self.q_proj.out_features)
        self.k_norm = nn.RMSNorm(self.k_proj.out_features)

    def forward(
        self, hidden_states, attention_mask=None, past_key_values=None, **kwargs
    ):
        q = self.q_norm(self.q_proj(hidden_states))
        k = self.k_norm(self.k_proj(hidden_states))
        v = self.v_proj(hidden_states)
        return self.o_proj(q + k + v), None


class PerHeadQKNormAttention(FakeAttention):
    """Qwen3-style: a per-head norm (`head_dim`) applied after the head reshape."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(
        self, hidden_states, attention_mask=None, past_key_values=None, **kwargs
    ):
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(shape))
        k = self.k_norm(self.k_proj(hidden_states).view(shape))
        v = self.v_proj(hidden_states).view(shape)
        return self.o_proj((q + k + v).flatten(-2)), None


class ResidDropoutAttention(FakeAttention):
    """GPT-style dropout after `o_proj` -> the output projection is still found."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resid_dropout = nn.Dropout(0.0)

    def forward(
        self, hidden_states, attention_mask=None, past_key_values=None, **kwargs
    ):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, None
        )
        attn_output, _ = attention_interface(
            self, q, k, v, attention_mask, scaling=self.scaling, **kwargs
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.resid_dropout(self.o_proj(attn_output)), None


class PackedQKVAttention(nn.Module):
    """GPTBigCode-style: one packed projection split into q/k/v in the forward."""

    is_causal = True

    def __init__(
        self,
        hidden: int = 32,
        head_dim: int = 8,
        heads: int = 4,
        kv_heads: int = 1,
        bias: bool = False,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="vllm")
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.embed_dim = heads * head_dim
        self.kv_dim = kv_heads * head_dim
        self.c_attn = nn.Linear(hidden, self.embed_dim + 2 * self.kv_dim, bias=bias)
        self.c_proj = nn.Linear(self.embed_dim, hidden, bias=bias)
        self.resid_dropout = nn.Dropout(0.0)

    def forward(
        self, hidden_states, attention_mask=None, past_key_values=None, **kwargs
    ):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        input_shape = hidden_states.shape[:-1]
        q, k, v = (
            self.c_attn(hidden_states)
            .unsqueeze(1)
            .split((self.embed_dim, self.kv_dim, self.kv_dim), dim=3)
        )
        q = q.view(*input_shape, -1, self.head_dim).transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, None
        )
        attn_output, attn_weights = attention_interface(
            self, q, k, v, attention_mask, scaling=self.scaling, **kwargs
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.resid_dropout(self.c_proj(attn_output)), attn_weights


class PerHeadSplitAttention(nn.Module):
    """A packed projection reshaped and split *per head* -> not a q/k/v split."""

    def __init__(self, hidden: int = 32, head_dim: int = 8, heads: int = 4):
        super().__init__()
        self.head_dim = head_dim
        self.heads = heads
        self.c_attn = nn.Linear(hidden, 3 * heads * head_dim)
        self.c_proj = nn.Linear(heads * head_dim, hidden)

    def forward(self, hidden_states):
        shape = (*hidden_states.shape[:2], self.heads, 3 * self.head_dim)
        q, k, v = (
            self.c_attn(hidden_states)
            .view(shape)
            .transpose(1, 2)
            .split((self.head_dim, self.head_dim, self.head_dim), dim=3)
        )
        return self.c_proj((q + k + v).transpose(1, 2).flatten(-2))


class FakeSelfAttn(nn.Module):
    """Stand-in for the vLLM `Attention` looked up in `attention_instances`."""

    def __init__(self):
        super().__init__()
        self.impl = SimpleNamespace(scale=None)

    def forward(self, q, k, v):
        # MHA-shaped stub: any deterministic combination of q/k/v will do
        return q + 2 * k + 3 * v


class FakeMQASelfAttn(FakeSelfAttn):
    """Stand-in for grouped/multi-query layouts, where `k`/`v` are narrower."""

    def forward(self, q, k, v):
        groups = q.shape[-1] // k.shape[-1]
        return q + (2 * k + 3 * v).repeat(1, groups)


@pytest.fixture(autouse=True)
def _clear_fuser_cache():
    get_fuser.cache_clear()
    yield
    get_fuser.cache_clear()


def _apply_glu_fuser_with_stubs(module: nn.Module, fuser: GLUFuser):
    """Apply a fuser using plain stand-ins (merged `nn.Linear` + silu AndMul)."""
    gate = module.get_submodule(fuser.gate_name)
    up = module.get_submodule(fuser.up_name)
    merged = nn.Linear(
        gate.in_features,
        gate.out_features + up.out_features,
        bias=gate.bias is not None,
    )
    with torch.no_grad():
        merged.weight.copy_(torch.cat([gate.weight, up.weight], dim=0))
        if gate.bias is not None:
            merged.bias.copy_(torch.cat([gate.bias, up.bias], dim=0))
    setattr(module, fuser.merged_name, merged)
    setattr(module, fuser.act_name, SiluAndMulStub())
    delattr(module, fuser.gate_name)
    delattr(module, fuser.up_name)
    module.forward = MethodType(fuser.fused_forward, module)
    return module


def _apply_qkv_fuser_with_stubs(module: nn.Module, fuser: QKVFuser):
    """Apply a fuser using a plain merged `nn.Linear` (no TP sharding)."""
    q, k, v = (
        module.get_submodule(name)
        for name in (fuser.q_name, fuser.k_name, fuser.v_name)
    )
    merged = nn.Linear(
        q.in_features,
        q.out_features + k.out_features + v.out_features,
        bias=q.bias is not None,
    )
    with torch.no_grad():
        merged.weight.copy_(torch.cat([q.weight, k.weight, v.weight], dim=0))
        if q.bias is not None:
            merged.bias.copy_(torch.cat([q.bias, k.bias, v.bias], dim=0))
    merged.output_sizes = [q.out_features, k.out_features, v.out_features]
    merged.tp_size = 1
    setattr(module, fuser.merged_name, merged)
    for name in (fuser.q_name, fuser.k_name, fuser.v_name):
        delattr(module, name)
    module.forward = MethodType(fuser.fused_forward, module)
    return module


def _apply_packed_qkv_fuser_with_stubs(module: nn.Module, fuser: PackedQKVFuser):
    """Apply a fuser at `tp_size == 1`, where the rewritten split is unchanged."""
    qkv = module.get_submodule(fuser.qkv_name)
    qkv.output_sizes = [fuser.q_size, fuser.kv_size, fuser.kv_size]
    qkv.tp_size = 1
    module.forward = MethodType(fuser.fused_forward, module)
    return module


@pytest.mark.parametrize("mlp_cls", [GLUMLP, ReversedGLUMLP])
@pytest.mark.parametrize("bias", [False, True])
def test_detects_and_rewrites_glu(mlp_cls, bias):
    with torch.device("meta"):
        meta = mlp_cls(bias=bias)
    fuser = get_fuser(meta)
    assert isinstance(fuser, GLUFuser)
    assert (
        fuser.gate_name,
        fuser.up_name,
        fuser.act_name,
        fuser.down_name,
    ) == ("gate_proj", "up_proj", "act_fn", "down_proj")

    # The rewritten forward references the merged projection instead of the
    # sources; the rest of the forward is untouched.
    names = fuser.fused_forward.__code__.co_names
    assert "gate_up_proj" in names and "act_fn" in names and "down_proj" in names
    assert not {"gate_proj", "up_proj"} & set(names)

    # Numerics: the fused forward must match the original on a real instance.
    real = mlp_cls(bias=bias)
    for p in real.parameters():
        nn.init.normal_(p, std=0.05)
    x = torch.randn(4, 16)
    expected = real(x)
    fused = _apply_glu_fuser_with_stubs(real, fuser)

    # Fusion is in place: the module keeps its class and other attributes
    assert fused is real and type(fused) is mlp_cls
    torch.testing.assert_close(fused(x), expected, atol=1e-5, rtol=1e-5)


def test_glu_identifies_down_projection():
    """The row projection consuming `act(gate(x)) * up(x)` is identified.

    It is forced to `RowParallelLinear` in `update_attrs` so its sharded input
    matches the column-parallel merged gate/up; `None` when there is no such
    projection to force (fusion of gate/up still applies).
    """
    with torch.device("meta"):
        assert get_fuser(GLUMLP()).down_name == "down_proj"
        assert get_fuser(ReversedGLUMLP()).down_name == "down_proj"
        assert get_fuser(NoDownGLU()).down_name is None


@pytest.mark.parametrize("attn_cls", [FakeAttention, ReversedFakeAttention])
@pytest.mark.parametrize("kv_heads", [4, 2])
def test_detects_and_rewrites_qkv(attn_cls, kv_heads):
    if attn_cls is ReversedFakeAttention and kv_heads == 4:
        pytest.skip("MHA q/k/v assignment is order-based by design")
    with torch.device("meta"):
        meta = attn_cls(kv_heads=kv_heads)
    fuser = get_fuser(meta)
    assert isinstance(fuser, QKVFuser)
    # q (sharded differently under TP) must be identified exactly; k/v may be
    # swapped for non-canonical compute order, which is numerically consistent
    # because the weight mapping and the split indices follow the same
    # assignment.
    assert fuser.q_name == "q_proj"
    assert {fuser.k_name, fuser.v_name} == {"k_proj", "v_proj"}
    assert fuser.o_name == "o_proj"

    # The projections are merged; everything else stays live Python with its
    # original semantics (branches, kwargs, attribute reads)
    code = fuser.fused_forward.__code__
    names = code.co_names
    assert "qkv_proj" in names and "output_sizes" in names and "o_proj" in names
    assert "tp_size" in names
    assert not {"q_proj", "k_proj", "v_proj"} & set(names)
    if attn_cls is FakeAttention:
        assert "update" in names  # the cache branch survives
    assert code.co_flags & inspect.CO_VARKEYWORDS  # **kwargs survives

    # Numerics: the fused forward must match the original on a real instance,
    # with a different layer_idx than the traced instance (kv_heads == heads so
    # the q/k/v stub combination is shape-compatible).
    real = attn_cls(kv_heads=4, layer_idx=3)
    for p in real.parameters():
        nn.init.normal_(p, std=0.05)
    x = torch.randn(1, 5, 32)
    attention_instances = {3: FakeSelfAttn()}
    expected, _ = real(x, attention_instances=attention_instances)
    fused = _apply_qkv_fuser_with_stubs(real, fuser)

    # Fusion is in place: the module keeps its class and other attributes
    assert fused is real and type(fused) is attn_cls
    assert fused.layer_idx == 3 and fused.is_causal and fused.config is not None
    out, _ = fused(x, attention_instances=attention_instances)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_qkv_identifies_output_projection():
    with torch.device("meta"):
        assert get_fuser(FakeAttention()).o_name == "o_proj"
        assert get_fuser(ReversedFakeAttention()).o_name == "o_proj"
        assert get_fuser(ExtraProjAttention()).o_name == "o_proj"
        # Norm children (q_norm/k_norm) must not disturb o_proj identification.
        assert get_fuser(QKNormAttention()).o_name == "o_proj"
        assert get_fuser(PerHeadQKNormAttention()).o_name == "o_proj"
        # A module between o_proj and the return is transparent.
        assert get_fuser(ResidDropoutAttention()).o_name == "o_proj"


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_detects_and_rewrites_packed_qkv(kv_heads):
    """A single projection split into q/k/v must be re-sharded, not merged.

    Only the split sizes change: `QKVParallelLinear` loads the packed
    checkpoint weight as-is, and shards q by heads while replicating k/v.
    """
    with torch.device("meta"):
        meta = PackedQKVAttention(kv_heads=kv_heads)
    fuser = get_fuser(meta)
    assert isinstance(fuser, PackedQKVFuser)
    assert (fuser.qkv_name, fuser.o_name) == ("c_attn", "c_proj")
    assert (fuser.q_size, fuser.kv_size) == (32, 8 * kv_heads)

    # The hard-coded widths become the per-rank widths of the sharded linear
    names = fuser.fused_forward.__code__.co_names
    assert "output_sizes" in names and "tp_size" in names
    assert "kv_dim" not in names and "embed_dim" not in names

    # Numerics: the rewritten forward must match the original on a real instance
    real = PackedQKVAttention(kv_heads=kv_heads, layer_idx=3)
    for p in real.parameters():
        nn.init.normal_(p, std=0.05)
    x = torch.randn(1, 5, 32)
    attention_instances = {3: FakeMQASelfAttn()}
    expected, _ = real(x, attention_instances=attention_instances)
    fused = _apply_packed_qkv_fuser_with_stubs(real, fuser)

    # Fusion is in place: the module keeps its class and other attributes
    assert fused is real and type(fused) is PackedQKVAttention
    assert fused.layer_idx == 3 and fused.is_causal
    out, _ = fused(x, attention_instances=attention_instances)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_per_head_split_is_not_packed_qkv():
    """The split must consume the whole projection, else its sizes are head
    widths and re-sharding by them would be wrong.
    """
    with torch.device("meta"):
        assert get_fuser(PerHeadSplitAttention()) is None


def test_fuser_is_cached_per_class_and_structure():
    with torch.device("meta"):
        fuser_a = get_fuser(GLUMLP())
        fuser_b = get_fuser(GLUMLP())
    assert fuser_a is fuser_b
    assert any(key[0] is GLUMLP for key in get_fuser.cache)


@pytest.mark.parametrize("cls", [NotAnMLP, UntraceableMLP])
def test_non_matching_modules_return_none(cls):
    with torch.device("meta"):
        module = cls()
    assert get_fuser(module) is None


def test_untraceable_tail_still_fuses():
    with torch.device("meta"):
        meta = UntraceableTailGLUMLP()
    fuser = get_fuser(meta)
    assert isinstance(fuser, GLUFuser)

    # Numerics: the live tail must survive the rewrite
    real = UntraceableTailGLUMLP()
    for p in real.parameters():
        nn.init.normal_(p, std=0.05)
    x = torch.randn(4, 16)
    expected = real(x)
    fused = _apply_glu_fuser_with_stubs(real, fuser)
    torch.testing.assert_close(fused(x), expected, atol=1e-5, rtol=1e-5)


def test_weight_mappings_are_scoped_to_fused_prefixes():
    from vllm.model_executor.models.utils import WeightsMapper

    with torch.device("meta"):
        glu_fuser = get_fuser(GLUMLP())
        qkv_fuser = get_fuser(FakeAttention())

    mapper = WeightsMapper()
    for prefix in ("model.layers.0.mlp", "model.layers.1.mlp"):
        mapper.orig_to_new_stacked.update(glu_fuser.orig_to_new_stacked(prefix))
    mapper.orig_to_new_stacked.update(
        qkv_fuser.orig_to_new_stacked("model.layers.0.self_attn")
    )

    names = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        # Unfused modules at other prefixes must be left untouched.
        "model.layers.2.mlp.experts.0.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
    ]
    # `apply` rewrites the name and stamps the shard id onto each tensor.
    weights = [(name, torch.empty(0)) for name in names]
    mapped = list(mapper.apply(weights))
    mapped_names = [name for name, _ in mapped]
    shard_ids = [getattr(data, "shard_id", None) for _, data in mapped]

    assert mapped_names == [
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.1.mlp.gate_up_proj.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        # Only the exact fused layers are remapped; everything else is untouched.
        "model.layers.2.mlp.experts.0.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
    ]
    assert shard_ids == [0, 1, 0, "q", "k", "v", None, None]

    # The fused layers are exposed to the quantization machinery via their
    # original constituent projection names (what the checkpoint stores).
    assert glu_fuser.packed_modules_mapping == {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    assert qkv_fuser.packed_modules_mapping == {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }


@pytest.mark.parametrize("cls", [NotAnMLP, NotAnActGLUMLP])
def test_unfusable_modules_are_not_fused(cls, default_vllm_config):
    with torch.device("meta"):
        module = cls()
    fuser = get_fuser(module)
    # Either no pattern matches the class, or this instance fails validation
    # (`recursive_replace` gates fusion and its weight mappings on `validate`)
    assert fuser is None or not fuser.validate(module, default_vllm_config)


def test_act_and_mul_derived_from_module(default_vllm_config):
    from transformers.activations import GELUTanh, SiLUActivation

    from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul

    assert isinstance(GLUFuser._get_act_and_mul(nn.SiLU()), SiluAndMul)
    assert isinstance(GLUFuser._get_act_and_mul(SiLUActivation()), SiluAndMul)
    gelu_tanh = GLUFuser._get_act_and_mul(GELUTanh())
    assert isinstance(gelu_tanh, GeluAndMul) and gelu_tanh.approximate == "tanh"
    gelu = GLUFuser._get_act_and_mul(nn.GELU())
    assert isinstance(gelu, GeluAndMul) and gelu.approximate == "none"
    # Not activations at all -> no fusion
    assert GLUFuser._get_act_and_mul_name(nn.Dropout()) is None
    assert GLUFuser._get_act_and_mul_name(nn.LayerNorm(8)) is None
    with pytest.raises(ValueError, match="No AndMul equivalent"):
        GLUFuser._get_act_and_mul(nn.Dropout())


def _wider_model_config(head_dim: int) -> SimpleNamespace:
    """A model whose global head size is twice `head_dim`, as a wider layer
    elsewhere in a heterogeneous checkpoint would make it.
    """
    return SimpleNamespace(
        model_config=SimpleNamespace(get_head_size=lambda: 2 * head_dim),
        quant_config=None,
    )


@pytest.mark.parametrize(
    "cls, fuser_module", [(FakeAttention, qkv), (PackedQKVAttention, packed_qkv)]
)
def test_head_counts_come_from_the_module_not_the_model(cls, fuser_module, monkeypatch):
    """A layer narrower than the model-wide head size must not be miscounted.

    On a heterogeneous checkpoint (Gemma 4) the model-wide head size is the
    largest across layers, so deriving `total_num_heads = out_features //
    head_size` from it undercounts heads on a narrower layer. The widths still
    add up, so nothing raises below TP=4: the layer is just sharded wrong.
    """
    head_dim, heads, kv_heads = 8, 8, 4
    vllm_config = _wider_model_config(head_dim)
    with torch.device("meta"):
        module = cls(hidden=32, head_dim=head_dim, heads=heads, kv_heads=kv_heads)

    # Both replacements shard, so they need a TP group; only the head counts
    # the fuser derives are under test here.
    captured = {}
    monkeypatch.setattr(
        fuser_module,
        "QKVParallelLinear",
        lambda **kwargs: captured.update(kwargs) or nn.Identity(),
    )
    monkeypatch.setattr(
        fuser_module, "replace_linear_class", lambda *a, **kw: nn.Identity()
    )
    fuser = get_fuser(module)
    assert fuser is not None and fuser.validate(module, vllm_config)
    fuser.update_attrs(module, "model.layers.0.self_attn", vllm_config)

    assert captured["head_size"] == head_dim
    assert captured["total_num_heads"] == heads
    assert captured["total_num_kv_heads"] == kv_heads


def test_validate_accepts_a_layer_the_model_wide_head_size_would_reject():
    """`validate` gates fusion, so a wrong head size silently disables it."""
    head_dim, heads, kv_heads = 8, 8, 3
    vllm_config = _wider_model_config(head_dim)
    with torch.device("meta"):
        module = FakeAttention(
            hidden=32, head_dim=head_dim, heads=heads, kv_heads=kv_heads
        )

    # kv width is 24, not a multiple of the model-wide 16, but is of this
    # layer's 8.
    fuser = get_fuser(module)
    assert fuser is not None and fuser.validate(module, vllm_config)
