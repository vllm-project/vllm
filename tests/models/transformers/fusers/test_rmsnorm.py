# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers modeling backend's RMSNorm fuser."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

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


@pytest.mark.parametrize(
    "cls,eps,has_weight,zero_centered",
    [
        (RMSNorm, 1e-5, True, False),
        (GemmaRMSNorm, 1e-6, True, True),
        (WeightlessRMSNorm, 1e-6, False, False),
        (LayerNorm, 1e-6, True, False),
        (torch.nn.RMSNorm, 1e-5, True, False),  # fused `F.rms_norm` op
    ],
)
def test_detects_rms_norm_variants(cls, eps, has_weight, zero_centered):
    with torch.device("meta"):
        fuser = get_fuser(cls(16, eps=eps))
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
    assert built.variance_epsilon == fuser.eps


def test_fused_rms_norm_op_default_eps(default_vllm_config):
    """`torch.nn.RMSNorm` (a single `F.rms_norm` call) matches via the fast path;
    its default `eps=None` resolves to `finfo(dtype).eps` in `fuse`."""
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm

    with torch.device("meta"):
        module = torch.nn.RMSNorm(16)  # forward is a single `F.rms_norm` call
        fuser = get_fuser(module)
        assert isinstance(fuser, RMSNormFuser)
        assert fuser.has_weight and not fuser.zero_centered
        assert fuser.eps is None  # default eps, resolved at fuse time
        model_config = SimpleNamespace(get_hidden_size=lambda: 16, dtype=torch.float32)
        built = fuser.fuse(module, "norm", model_config, None)
    assert isinstance(built, VLLMRMSNorm)
    assert built.variance_epsilon == torch.finfo(torch.float32).eps
