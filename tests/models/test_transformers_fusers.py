# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers backend's fx fusers (MoE, GLU, QKV, RMSNorm)."""

import inspect
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Registers the "vllm" attention interface in ALL_ATTENTION_FUNCTIONS
import vllm.model_executor.models.transformers.base  # noqa: F401
from vllm.model_executor.models.transformers.fuser import (
    GLUFuser,
    QKVFuser,
    RMSNormFuser,
    get_fuser,
)
from vllm.model_executor.models.transformers.fusers.moe import MoEFuser


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
    assert GLUMLP in get_fuser.cache


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
    model_config = default_vllm_config.model_config
    assert fuser is None or not fuser.validate(module, model_config)


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


class LlamaRMSNorm(nn.Module):
    """The canonical HF RMSNorm: `weight * x * rsqrt(mean(x**2) + eps)`."""

    def __init__(self, hidden: int = 16, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class GemmaRMSNorm(nn.Module):
    """Zero-centered weight: `(1 + weight) * normalized`."""

    def __init__(self, dim: int = 16, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class WeightlessRMSNorm(nn.Module):
    """No scale parameter (e.g. Gemma3n `with_scale=False`)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x)


class T5LayerNorm(nn.Module):
    """An RMSNorm whose class name is *not* `*RMSNorm` (name-independence)."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class NotAnRMSNorm(nn.Module):
    """Mean-subtracting LayerNorm-like math -> not an RMSNorm."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x):
        x = x - x.mean(-1, keepdim=True)
        return self.weight * x / torch.sqrt(x.var(-1, keepdim=True) + self.eps)


@pytest.mark.parametrize(
    "cls,eps,has_weight,zero_centered",
    [
        (LlamaRMSNorm, 1e-5, True, False),
        (GemmaRMSNorm, 1e-6, True, True),
        (WeightlessRMSNorm, 1e-6, False, False),
        (T5LayerNorm, 1e-6, True, False),
    ],
)
def test_detects_rms_norm_variants(cls, eps, has_weight, zero_centered):
    with torch.device("meta"):
        fuser = get_fuser(cls())
    assert isinstance(fuser, RMSNormFuser)
    # eps and the weight form are read from the graph, not the class name.
    assert fuser.eps == eps
    assert fuser.has_weight == has_weight
    assert fuser.zero_centered == zero_centered


@pytest.mark.parametrize("cls", [NotAnRMSNorm, nn.LayerNorm, nn.SiLU])
def test_non_rms_norms_are_not_matched(cls):
    with torch.device("meta"):
        module = cls(16) if cls is nn.LayerNorm else cls()
    assert not isinstance(get_fuser(module), RMSNormFuser)


@pytest.mark.parametrize(
    "cls,expected,zero_centered",
    [
        (LlamaRMSNorm, "RMSNorm", False),
        (GemmaRMSNorm, "GemmaRMSNorm", True),
        (WeightlessRMSNorm, "RMSNorm", False),
    ],
)
def test_rms_norm_builds_vllm_class(cls, expected, zero_centered, default_vllm_config):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm as VLLMGemmaRMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm

    # `default_vllm_config` supplies the config context the CustomOp needs; the
    # weightless path reads hidden size from the model config, so stub it.
    model_config = SimpleNamespace(get_hidden_size=lambda: 16)
    with torch.device("meta"):
        module = cls()
        fuser = get_fuser(module)
        built = fuser.fuse(module, "norm", model_config, None)
    types_by_name = {"RMSNorm": VLLMRMSNorm, "GemmaRMSNorm": VLLMGemmaRMSNorm}
    assert type(built) is types_by_name[expected]
    assert built.variance_epsilon == fuser.eps


class TopKRouter(nn.Module):
    """HF v5 top-k router: `linear -> softmax -> topk (-> renorm)`."""

    def __init__(self, num_experts=8, hidden=16, top_k=2, sigmoid=False):
        super().__init__()
        self.top_k = top_k
        self.sigmoid = sigmoid
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden))

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight)
        scores = torch.sigmoid(logits) if self.sigmoid else F.softmax(logits, dim=-1)
        value, index = torch.topk(scores, self.top_k, dim=-1)
        value = value / value.sum(dim=-1, keepdim=True)
        return logits, value, index


class CorrectionRouter(nn.Module):
    """Grouped/noaux router whose score-correction bias is a buffer (as in
    DeepSeek-V3).

    It is `weight`-only in its parameters, so it is declined via the correction
    bias read structurally from the graph, not by the parameter check."""

    def __init__(self, num_experts=8, hidden=16):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden))
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight)
        scores = torch.sigmoid(logits) + self.e_score_correction_bias
        _, index = torch.topk(scores, 2, dim=-1)
        return logits, scores, index


class BiasedRouter(TopKRouter):
    """A valid top-k router but not `weight`-only (an extra `bias` parameter).

    `build_gate` rebuilds the router as a bias-free `ReplicatedLinear`, so a
    router carrying any other parameter cannot be reproduced faithfully."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight) + self.bias
        scores = F.softmax(logits, dim=-1)
        value, index = torch.topk(scores, self.top_k, dim=-1)
        return logits, value, index


class MoEExperts(nn.Module):
    """Packed experts (3D weights); only its name (`experts`) matters here."""

    def __init__(self, num_experts=8, hidden=16, inter=32):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(num_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.zeros(num_experts, hidden, inter))

    def forward(self, hidden_states, index, weights):
        return hidden_states


class MoEBlock(nn.Module):
    """A single-tensor-returning MoE block (Qwen3-style)."""

    def __init__(self, router_cls=TopKRouter):
        super().__init__()
        self.experts = MoEExperts()
        self.gate = router_cls()

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, weights, index = self.gate(x)
        return self.experts(x, index, weights).reshape(hidden_states.shape)


class MoEBlockShared(MoEBlock):
    """A block with a shared expert and its sigmoid gate (Qwen2-style)."""

    def __init__(self):
        super().__init__()
        self.shared_expert = GLUMLP()
        self.shared_expert_gate = nn.Linear(16, 1, bias=False)

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, weights, index = self.gate(x)
        out = self.experts(x, index, weights)
        out = out + torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return out.reshape(hidden_states.shape)


class MoEBlockSharedNoGate(MoEBlock):
    """A block with an ungated shared expert -> native, shared passed through."""

    def __init__(self):
        super().__init__()
        self.shared_expert = GLUMLP()

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, weights, index = self.gate(x)
        out = self.experts(x, index, weights) + self.shared_expert(x)
        return out.reshape(hidden_states.shape)


class MoEBlockTuple(MoEBlock):
    """A tuple-returning block (gpt-oss-style) -> must decline."""

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, weights, index = self.gate(x)
        return self.experts(x, index, weights), index


class PlainMLP(nn.Module):
    """A non-GLU FFN: `down(act(up(x)))`, no gating multiply."""

    def __init__(self, hidden: int = 16, inter: int = 32):
        super().__init__()
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class MoEBlockSharedNonGLU(MoEBlock):
    """A block whose shared expert is a non-GLU MLP -> detected by dataflow.

    It is added to the experts' output, so it is recognised as the shared
    expert even though it is not a GLU (no gate/up merge is needed).
    """

    def __init__(self):
        super().__init__()
        self.shared_expert = PlainMLP()

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, weights, index = self.gate(x)
        out = self.experts(x, index, weights) + self.shared_expert(x)
        return out.reshape(hidden_states.shape)


class MoEBlockUnaccounted(MoEBlock):
    """A block with a weight-bearing child outside the experts + shared-expert
    dataflow (here a pre-router transform) -> must decline, since the rewritten
    forward would drop it."""

    def __init__(self):
        super().__init__()
        self.extra = nn.Linear(16, 16, bias=False)

    def forward(self, hidden_states):
        x = self.extra(hidden_states.reshape(-1, hidden_states.shape[-1]))
        _, weights, index = self.gate(x)
        return self.experts(x, index, weights).reshape(hidden_states.shape)


class BufferScale(nn.Module):
    """A stateful child carrying only a buffer (no parameters)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.register_buffer("scale", torch.ones(hidden))

    def forward(self, x):
        return x * self.scale


class MoEBlockUnaccountedBuffer(MoEBlock):
    """Like `MoEBlockUnaccounted`, but the dropped child holds only a buffer, so
    the fail-closed check must inspect buffers as well as parameters."""

    def __init__(self):
        super().__init__()
        self.extra = BufferScale()

    def forward(self, hidden_states):
        x = self.extra(hidden_states.reshape(-1, hidden_states.shape[-1]))
        _, weights, index = self.gate(x)
        return self.experts(x, index, weights).reshape(hidden_states.shape)


@pytest.mark.parametrize("sigmoid", [False, True])
def test_moe_fuser_detects_router(sigmoid):
    with torch.device("meta"):
        block = MoEBlock(lambda: TopKRouter(sigmoid=sigmoid))
    fuser = MoEFuser.match(block)
    assert isinstance(fuser, MoEFuser)
    assert fuser.gate_name == "gate"
    assert fuser.scoring_func == ("sigmoid" if sigmoid else "softmax")
    assert fuser.shared_name is None and fuser.shared_gate_name is None


def test_moe_fuser_detects_shared_experts():
    with torch.device("meta"):
        block = MoEBlockShared()
    fuser = MoEFuser.match(block)
    assert isinstance(fuser, MoEFuser)
    assert fuser.shared_name == "shared_expert"
    assert fuser.shared_gate_name == "shared_expert_gate"
    # The shared expert's gate/up projections are merged, scoped to its qualname.
    assert fuser.orig_to_new_stacked("model.layers.0.mlp") == {
        "model.layers.0.mlp.shared_expert.gate_proj": (
            "model.layers.0.mlp.shared_expert.gate_up_proj",
            0,
        ),
        "model.layers.0.mlp.shared_expert.up_proj": (
            "model.layers.0.mlp.shared_expert.gate_up_proj",
            1,
        ),
    }


def test_moe_fuser_shared_without_gate():
    with torch.device("meta"):
        block = MoEBlockSharedNoGate()
    fuser = MoEFuser.match(block)
    assert isinstance(fuser, MoEFuser)
    assert fuser.shared_name == "shared_expert"
    assert fuser.shared_gate_name is None


def test_moe_fuser_detects_non_glu_shared_expert():
    with torch.device("meta"):
        block = MoEBlockSharedNonGLU()
    fuser = MoEFuser.match(block)
    assert isinstance(fuser, MoEFuser)
    # Recognised by dataflow (added to the experts' output), though not a GLU.
    assert fuser.shared_name == "shared_expert"
    assert fuser.shared_gate_name is None
    # A non-GLU shared expert needs no gate/up merge; it loads by identity.
    assert fuser.shared_glu is None
    assert fuser.orig_to_new_stacked("model.layers.0.mlp") == {}


@pytest.mark.parametrize(
    "block_cls",
    [
        lambda: MoEBlock(CorrectionRouter),  # score-correction buffer (grouped)
        lambda: MoEBlock(BiasedRouter),  # router not weight-only (extra param)
        MoEBlockTuple,  # tuple-returning block (e.g. gpt-oss)
        MoEBlockUnaccounted,  # weight-bearing child outside the fused dataflow
        MoEBlockUnaccountedBuffer,  # buffer-only child outside the fused dataflow
    ],
)
def test_moe_fuser_declines_unsupported(block_cls):
    with torch.device("meta"):
        block = block_cls()
    assert MoEFuser.match(block) is None
