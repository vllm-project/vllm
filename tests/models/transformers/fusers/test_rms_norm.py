# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers modeling backend's RMSNorm fuser."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.models.transformers.fuser import get_fuser
from vllm.model_executor.models.transformers.fusers import RMSNormFuser


class RMSNorm(nn.Module):
    """The canonical HF RMSNorm: `weight * x * rsqrt(mean(x**2) + eps)`."""

    def __init__(self, hidden: int = 16, eps: float = 1e-5, weight: bool = True):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = eps

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, x):
        return self.weight * self._rms(x.to(torch.float32)).to(x.dtype)


class GemmaRMSNorm(RMSNorm):
    """Zero-centered weight: `(1 + weight) * normalized`."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__(hidden, eps)
        self.weight = nn.Parameter(torch.zeros(hidden))

    def forward(self, x):
        return (1.0 + self.weight) * self._rms(x.to(torch.float32)).to(x.dtype)


class WeightlessRMSNorm(RMSNorm):
    """No scale parameter (e.g. Gemma3n `with_scale=False`)."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__(hidden, eps, weight=False)

    def forward(self, x):
        return self._rms(x.to(torch.float32)).to(x.dtype)


class LayerNorm(RMSNorm):
    """An RMSNorm not named `*RMSNorm`, keeping the input dtype (no upcast)."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__(hidden, eps)

    def forward(self, x):
        return self.weight * self._rms(x)


class NotAnRMSNorm(RMSNorm):
    """Mean-subtracting LayerNorm-like math -> not an RMSNorm."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__(hidden, eps)

    def forward(self, x):
        x = x - x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True)
        return self.weight * x / torch.sqrt(variance + self.variance_epsilon)


class GatedRMSNorm(RMSNorm):
    """Second input and tail compute -> not an RMSNorm."""

    def forward(self, x, gate=None):
        normed = self.weight * self._rms(x.to(torch.float32)).to(x.dtype)
        return normed * F.silu(gate)


class GatedFusedRMSNorm(nn.Module):
    """Same as GatedRMSNorm, but built on the fused `rms_norm` op -> not an RMSNorm."""

    def __init__(self, hidden: int = 16, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x, gate=None):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps) * F.silu(gate)


class UntraceableGatedRMSNorm(RMSNorm):
    """Tracer can't see tail compute in forward, but still has a second input (gate)."""

    def forward(self, x, gate=None):
        normed = self.weight * self._rms(x.to(torch.float32)).to(x.dtype)
        if gate.sum() > 0:  # untraceable -> partial graph, no visible tail
            normed = normed * F.silu(gate)
        return normed


@pytest.mark.parametrize(
    "cls,eps,zero_centered",
    [
        (RMSNorm, 1e-5, False),
        (GemmaRMSNorm, 1e-6, True),
        (WeightlessRMSNorm, 1e-6, False),
        (LayerNorm, 1e-6, False),
        (torch.nn.RMSNorm, 1e-5, False),  # fused `F.rms_norm` op
    ],
)
def test_detects_rms_norm_variants(cls, eps, zero_centered):
    with torch.device("meta"):
        fuser = get_fuser(cls(16, eps=eps))
    assert isinstance(fuser, RMSNormFuser)
    assert fuser.zero_centered == zero_centered


@pytest.mark.parametrize("cls", [NotAnRMSNorm, nn.LayerNorm, nn.SiLU])
def test_non_rms_norms_are_not_matched(cls):
    with torch.device("meta"):
        module = cls(16) if cls is nn.LayerNorm else cls()
    assert not isinstance(get_fuser(module), RMSNormFuser)


@pytest.mark.parametrize(
    "cls", [GatedRMSNorm, GatedFusedRMSNorm, UntraceableGatedRMSNorm]
)
def test_gated_rms_norm_is_not_fused(cls):
    with torch.device("meta"):
        assert not isinstance(get_fuser(cls()), RMSNormFuser)


@pytest.mark.parametrize(
    "cls,expected,zero_centered",
    [
        (RMSNorm, "RMSNorm", False),
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
    from vllm.model_executor.models.transformers.fusers.rms_norm import (
        TPAwareNormMixin,
    )

    types_by_name = {"RMSNorm": VLLMRMSNorm, "GemmaRMSNorm": VLLMGemmaRMSNorm}
    assert isinstance(built, types_by_name[expected])
    assert isinstance(built, TPAwareNormMixin)  # fused norms self-correct under TP
    assert built.variance_epsilon == module.variance_epsilon
    assert isinstance(built.weight, nn.Parameter) == (
        getattr(module, "weight", None) is not None
    )


def test_fused_rms_norm_op_default_eps(default_vllm_config):
    """`torch.nn.RMSNorm` (a single `F.rms_norm` call) matches via the fast path;
    its default `eps=None` resolves to `finfo(dtype).eps` in `fuse`."""
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm

    with torch.device("meta"):
        module = torch.nn.RMSNorm(16)  # forward is a single `F.rms_norm` call
        fuser = get_fuser(module)
        assert isinstance(fuser, RMSNormFuser)
        assert not fuser.zero_centered
        model_config = SimpleNamespace(get_hidden_size=lambda: 16, dtype=torch.float32)
        built = fuser.fuse(module, "norm", model_config, None)
    assert isinstance(built, VLLMRMSNorm)
    assert built.variance_epsilon == torch.finfo(torch.float32).eps


def test_eps_is_derived_per_instance(default_vllm_config):
    """Two instances of the same norm class with different eps must fuse to their
    own eps: the type-cached fuser holds only structure, not this value."""
    model_config = SimpleNamespace(get_hidden_size=lambda: 16)
    with torch.device("meta"):
        for eps in (1e-5, 1e-6):
            module = RMSNorm(16, eps=eps)
            built = get_fuser(module).fuse(module, "norm", model_config, None)
            assert built.variance_epsilon == eps


def test_fused_norm_is_gather_capable(default_vllm_config):
    """Every fused norm is emitted gather-capable, so a norm on a head-sharded
    projection (OLMoE-style) self-corrects at runtime with no QKV-specific
    plumbing. A full-width input skips the gather and equals a plain norm."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
    from vllm.model_executor.models.transformers.fusers import rms_norm

    torch.manual_seed(0)
    x = torch.randn(4, 16)
    for gathered_cls, plain_cls in [
        (rms_norm.TPAwareRMSNorm, RMSNorm),
        (rms_norm.TPAwareGemmaRMSNorm, GemmaRMSNorm),
    ]:
        gathered = gathered_cls(hidden_size=16, eps=1e-6)
        assert isinstance(gathered, rms_norm.TPAwareNormMixin)
        plain = plain_cls(hidden_size=16, eps=1e-6)
        with torch.no_grad():
            weight = torch.randn(16)
            gathered.weight.copy_(weight)
            plain.weight.copy_(weight)
        torch.testing.assert_close(gathered(x), plain(x))


def test_gathered_norm_rejects_uneven_sharding(default_vllm_config):
    """A sharded input (narrower than the full-width weight) that does not tile
    the weight evenly across ranks is rejected before any collective."""
    from vllm.model_executor.models.transformers.fusers import rms_norm

    norm = rms_norm.TPAwareRMSNorm(hidden_size=8, eps=1e-6)
    norm.tp_size = 2  # emulate TP=2 without a real process group
    with pytest.raises(ValueError, match="does not tile it evenly"):
        norm(torch.randn(2, 3))  # 3 * 2 != 8
