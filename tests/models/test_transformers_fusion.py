# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers backend's fx-based fusion (GLU and QKV).

These exercise the structural detection and surgical subgraph rewrite in
`vllm.model_executor.models.transformers.fusion` without needing a distributed
environment; the real `MergedColumnParallelLinear` / `QKVParallelLinear` /
`...AndMul` integration is covered end-to-end by `test_fusion`, `test_models`
and `test_quantization` in `tests/models/test_transformers.py`.
"""

import inspect
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Registers the "vllm" attention interface in ALL_ATTENTION_FUNCTIONS
import vllm.model_executor.models.transformers.base  # noqa: F401
from vllm.model_executor.models.transformers import fusion
from vllm.model_executor.models.transformers.fusion import (
    GLUFuser,
    QKVFuser,
    get_fuser,
)


class SiluAndMulStub(nn.Module):
    """Stand-in for vLLM's `SiluAndMul` (no vLLM config required)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


class GLUMLP(nn.Module):
    """`act(gate(x)) * up(x)` — the canonical HF GLU MLP."""

    def __init__(self, hidden: int = 16, inter: int = 32, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=bias)
        self.up_proj = nn.Linear(hidden, inter, bias=bias)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)
        self.act_fn = nn.SiLU()

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
    """Data-dependent control flow *before* the pattern -> no match (the
    partial trace ends before reaching the GLU)."""

    def forward(self, x):
        if x.sum() > 0:  # noqa: SIM108 - intentionally untraceable
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return x


class UntraceableTailGLUMLP(GLUMLP):
    """Data-dependent control flow *after* the pattern -> still fusable (the
    partial trace contains the GLU, and the AST rewrite keeps the tail live)."""

    def forward(self, x):
        y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        if y.sum() > torch.inf:  # intentionally untraceable
            y = y * 0
        return y


class FakeAttention(nn.Module):
    """Mimics the structure of a HF v5 attention module under vLLM.

    The forward exercises the patterns the fusion tracer must support: shape
    unpacking, the dead KV cache branch, and the `attention_interface` call
    splatting `**kwargs` (resolved to vLLM's interface, which is traced
    through).
    """

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


class FakeSelfAttn(nn.Module):
    """Stand-in for the vLLM `Attention` looked up in `attention_instances`."""

    def __init__(self):
        super().__init__()
        self.impl = SimpleNamespace(scale=None)

    def forward(self, q, k, v):
        # MHA-shaped stub: any deterministic combination of q/k/v will do
        return q + 2 * k + 3 * v


@pytest.fixture(autouse=True)
def _clear_fuser_cache():
    fusion.get_fuser.cache_clear()
    yield
    fusion.get_fuser.cache_clear()


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
    module.forward = types.MethodType(fuser.fused_forward, module)
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
    merged.split_sizes = [q.out_features, k.out_features, v.out_features]
    setattr(module, fuser.merged_name, merged)
    for name in (fuser.q_name, fuser.k_name, fuser.v_name):
        delattr(module, name)
    module.forward = types.MethodType(fuser.fused_forward, module)
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
    ) == ("gate_proj", "up_proj", "act_fn")

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

    # The projections are merged; everything else stays live Python with its
    # original semantics (branches, kwargs, attribute reads)
    code = fuser.fused_forward.__code__
    names = code.co_names
    assert "qkv_proj" in names and "split_sizes" in names and "o_proj" in names
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


def test_fuser_is_cached_per_class():
    with torch.device("meta"):
        fuser_a = get_fuser(GLUMLP())
        fuser_b = get_fuser(GLUMLP())
    assert fuser_a is fuser_b
    assert GLUMLP in fusion.get_fuser.cache


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
        mapper.orig_to_new_substr.update(glu_fuser.weight_mappings(prefix))
    mapper.orig_to_new_substr.update(
        qkv_fuser.weight_mappings("model.layers.0.self_attn")
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
    mapped = mapper.apply_list(names)
    assert mapped == [
        "model.layers.0.mlp.gate_up_proj.0.weight",
        "model.layers.0.mlp.gate_up_proj.1.weight",
        "model.layers.1.mlp.gate_up_proj.0.weight",
        "model.layers.0.self_attn.qkv_proj.q.weight",
        "model.layers.0.self_attn.qkv_proj.k.weight",
        "model.layers.0.self_attn.qkv_proj.v.weight",
        "model.layers.2.mlp.experts.0.gate_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
    ]
    # The mappings must also be visible to the quantization machinery,
    # which derives packed modules from substr (not regex) entries.
    assert mapper.get_packed_modules_mapping() == {
        "gate_up_proj": ["gate_up_proj.0", "gate_up_proj.1"],
        "qkv_proj": ["qkv_proj.q", "qkv_proj.k", "qkv_proj.v"],
    }


@pytest.mark.parametrize("cls", [NotAnMLP, NotAnActGLUMLP])
def test_unfusable_modules_are_not_fused(cls):
    with torch.device("meta"):
        module = cls()
    fuser = get_fuser(module)
    # Either no pattern matches the class, or this instance fails validation
    # (`recursive_replace` gates fusion and its weight mappings on `validate`)
    assert fuser is None or not fuser.validate(module)


def test_act_and_mul_derived_from_module(default_vllm_config):
    from transformers.activations import GELUTanh, SiLUActivation

    from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul

    assert isinstance(fusion.GLUFuser._get_act_and_mul(nn.SiLU()), SiluAndMul)
    assert isinstance(fusion.GLUFuser._get_act_and_mul(SiLUActivation()), SiluAndMul)
    gelu_tanh = fusion.GLUFuser._get_act_and_mul(GELUTanh())
    assert isinstance(gelu_tanh, GeluAndMul) and gelu_tanh.approximate == "tanh"
    gelu = fusion.GLUFuser._get_act_and_mul(nn.GELU())
    assert isinstance(gelu, GeluAndMul) and gelu.approximate == "none"
    # Not activations at all -> no fusion
    assert fusion.GLUFuser._get_act_and_mul_name(nn.Dropout()) is None
    assert fusion.GLUFuser._get_act_and_mul_name(nn.LayerNorm(8)) is None
    with pytest.raises(ValueError, match="No AndMul equivalent"):
        fusion.GLUFuser._get_act_and_mul(nn.Dropout())
