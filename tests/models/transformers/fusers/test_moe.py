# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers modeling backend's MoE fuser."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.models.transformers.fusers import MoEBlockFuser

from .test_linear import GLUMLP


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
    """Grouped router with a score-correction bias buffer (DeepSeek-V3) -> declined."""

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
    """A valid top-k router but not `weight`-only (extra `bias` param) -> declined."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight) + self.bias
        scores = F.softmax(logits, dim=-1)
        value, index = torch.topk(scores, self.top_k, dim=-1)
        return logits, value, index


class DisconnectedRouter(TopKRouter):
    """linear+softmax+top-k present but top-k ignores the logits -> not a router."""

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight)
        _ = F.softmax(logits, dim=-1)  # scored, but not consumed by top-k
        value, index = torch.topk(hidden_states, self.top_k, dim=-1)
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
    """Single-tensor MoE block (Qwen3-style); subclasses override `_shared`."""

    def __init__(self, router_cls=TopKRouter):
        super().__init__()
        self.experts = MoEExperts()
        self.gate = router_cls()

    def _shared(self, x, logits):
        """The term added to the experts' output (none for a plain block)."""
        return 0

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits, weights, index = self.gate(x)
        out = self.experts(x, index, weights) + self._shared(x, logits)
        return out.reshape(hidden_states.shape)


class MoEBlockNoShared(MoEBlock):
    """No shared-expert child but a gate-derived add -> trace skipped, still fuses."""

    def _shared(self, x, logits):
        return logits.sum()


class MoEBlockShared(MoEBlock):
    """A block with a shared expert and its sigmoid gate (Qwen2-style)."""

    def __init__(self):
        super().__init__()
        self.shared_expert = GLUMLP()
        self.shared_expert_gate = nn.Linear(16, 1, bias=False)

    def _shared(self, x, logits):
        return torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


class MoEBlockSharedNoGate(MoEBlock):
    """A block with an ungated shared expert -> native, shared passed through."""

    def __init__(self):
        super().__init__()
        self.shared_expert = GLUMLP()

    def _shared(self, x, logits):
        return self.shared_expert(x)


class MoEBlockTuple(MoEBlock):
    """A tuple-returning block (gpt-oss-style) -> must decline."""

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, weights, index = self.gate(x)
        return self.experts(x, index, weights), index


class MoEBlockTupleVar(MoEBlock):
    """Returns a name bound to a tuple, not a literal tuple -> must still decline."""

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, weights, index = self.gate(x)
        result = self.experts(x, index, weights), index
        return result


class MoEBlockNestedTupleReturn(MoEBlock):
    """Tuple `return` in a nested helper; block returns one tensor -> still fuses."""

    def forward(self, hidden_states):
        def keep(a, b):
            return a, b

        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, weights, index = self.gate(x)
        out, _ = keep(self.experts(x, index, weights), index)
        return out.reshape(hidden_states.shape)


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
    """A non-GLU shared expert -> detected by dataflow (no gate/up merge)."""

    def __init__(self):
        super().__init__()
        self.shared_expert = PlainMLP()

    def _shared(self, x, logits):
        return self.shared_expert(x)


class MoEBlockUnaccounted(MoEBlock):
    """A weight-bearing child outside the fused dataflow (pre-router) -> declined."""

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


class MoEBlockUnaccountedBuffer(MoEBlockUnaccounted):
    """Like `MoEBlockUnaccounted`, but the extra child holds only a buffer."""

    def __init__(self):
        super().__init__()
        self.extra = BufferScale()


@pytest.mark.parametrize("sigmoid", [False, True])
def test_moe_fuser_detects_router(sigmoid):
    with torch.device("meta"):
        block = MoEBlock(lambda: TopKRouter(sigmoid=sigmoid))
    fuser = MoEBlockFuser.match(block, "experts")
    assert isinstance(fuser, MoEBlockFuser)
    assert fuser.gate_name == "gate"
    assert fuser.scoring_func == ("sigmoid" if sigmoid else "softmax")
    assert fuser.shared_name is None and fuser.shared_gate_name is None


def test_moe_fuser_detects_shared_experts():
    with torch.device("meta"):
        block = MoEBlockShared()
    fuser = MoEBlockFuser.match(block, "experts")
    assert isinstance(fuser, MoEBlockFuser)
    assert fuser.shared_name == "shared_expert"
    assert fuser.shared_gate_name == "shared_expert_gate"


def test_moe_fuser_skips_shared_detection_without_extra_children():
    """With only experts and gate, shared-expert detection (and its block trace)
    is skipped, so a gate-derived add is not misread as a shared expert."""
    with torch.device("meta"):
        block = MoEBlockNoShared()
    fuser = MoEBlockFuser.match(block, "experts")
    assert isinstance(fuser, MoEBlockFuser)
    assert fuser.shared_name is None and fuser.shared_gate_name is None


def test_moe_fuser_shared_without_gate():
    with torch.device("meta"):
        block = MoEBlockSharedNoGate()
    fuser = MoEBlockFuser.match(block, "experts")
    assert isinstance(fuser, MoEBlockFuser)
    assert fuser.shared_name == "shared_expert"
    assert fuser.shared_gate_name is None


def test_moe_fuser_detects_non_glu_shared_expert():
    with torch.device("meta"):
        block = MoEBlockSharedNonGLU()
    fuser = MoEBlockFuser.match(block, "experts")
    assert isinstance(fuser, MoEBlockFuser)
    # Recognised by dataflow (added to the experts' output), though not a GLU.
    assert fuser.shared_name == "shared_expert"
    assert fuser.shared_gate_name is None


@pytest.mark.parametrize(
    "block_cls",
    [
        lambda: MoEBlock(CorrectionRouter),  # score-correction buffer (grouped)
        lambda: MoEBlock(BiasedRouter),  # router not weight-only (extra param)
        MoEBlockTuple,  # tuple-returning block (e.g. gpt-oss)
        MoEBlockTupleVar,  # tuple returned via a name binding, not a literal
        MoEBlockUnaccounted,  # weight-bearing child outside the fused dataflow
        MoEBlockUnaccountedBuffer,  # buffer-only child outside the fused dataflow
    ],
)
def test_moe_fuser_declines_unsupported(block_cls):
    with torch.device("meta"):
        block = block_cls()
    assert MoEBlockFuser.match(block, "experts") is None


def test_moe_fuser_ignores_nested_returns():
    """A tuple `return` inside a nested helper must not decline a block whose own
    forward returns a single tensor."""
    with torch.device("meta"):
        block = MoEBlockNestedTupleReturn()
    assert isinstance(MoEBlockFuser.match(block, "experts"), MoEBlockFuser)


def test_moe_fuser_router_requires_connected_dataflow():
    """A gate with linear + softmax + top-k present but not wired as a router
    (top-k selects over the input, not the scored logits) is not detected."""
    with torch.device("meta"):
        block = MoEBlock(DisconnectedRouter)
    assert MoEBlockFuser.match(block, "experts") is None
