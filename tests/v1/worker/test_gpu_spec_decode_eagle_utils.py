# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    share_target_lm_head,
)


class _UnquantizedHead(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


class _QuantizedHead(torch.nn.Module):
    """A head whose quant method packs the weights, leaving no `weight`."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight_packed = torch.nn.Parameter(
            weight.to(torch.int8), requires_grad=False
        )
        self.weight_scale = torch.nn.Parameter(
            torch.ones(weight.shape[0]), requires_grad=False
        )


def _eagle(has_own: bool = True):
    return SimpleNamespace(has_own_lm_head=has_own)


def test_quantized_draft_head_is_not_shared():
    """A quantized draft head cannot be compared, so it must keep its own copy.

    Reading `.weight` off a quantized ParallelLMHead raises AttributeError and
    used to abort engine startup for eagle3 drafts with an int8 lm_head.
    """
    weight = torch.ones(4, 8)
    shared = _should_share(
        _eagle(), "has_own_lm_head", _QuantizedHead(weight), _UnquantizedHead(weight)
    )
    assert shared is False


def test_quantized_target_head_is_not_shared():
    weight = torch.ones(4, 8)
    shared = _should_share(
        _eagle(), "has_own_lm_head", _UnquantizedHead(weight), _QuantizedHead(weight)
    )
    assert shared is False


def test_identical_unquantized_heads_are_shared():
    weight = torch.ones(4, 8)
    shared = _should_share(
        _eagle(), "has_own_lm_head", _UnquantizedHead(weight), _UnquantizedHead(weight)
    )
    assert shared is True


def test_distinct_unquantized_heads_are_not_shared():
    shared = _should_share(
        _eagle(),
        "has_own_lm_head",
        _UnquantizedHead(torch.ones(4, 8)),
        _UnquantizedHead(torch.zeros(4, 8)),
    )
    assert shared is False


def test_draft_without_own_head_is_shared():
    """The draft carries no head of its own, so the target's is adopted."""
    shared = _should_share(
        _eagle(has_own=False),
        "has_own_lm_head",
        None,
        _UnquantizedHead(torch.ones(4, 8)),
    )
    assert shared is True


def _drafter():
    drafter = torch.nn.Module()
    drafter.has_own_lm_head = False
    return drafter


def test_share_target_lm_head_installs_the_copy_the_caller_resolved(monkeypatch):
    """EAGLE/MTP and DSpark both hand the resolved copy spec to this function;
    the drafter ends up with the copy, and the target keeps its own head."""
    target_head, spec, copy = _UnquantizedHead(torch.ones(4, 8)), object(), object()
    seen = []

    def fake_copy(head, got):
        seen.append((head, got))
        return copy

    monkeypatch.setattr(eagle_utils, "quantized_lm_head_copy", fake_copy)
    drafter = _drafter()
    assert share_target_lm_head(drafter, target_head, spec) is copy
    assert seen == [(target_head, spec)]
    assert drafter.lm_head is copy

    shared = _drafter()
    assert share_target_lm_head(shared, target_head, None) is target_head
    assert shared.lm_head is target_head


def test_share_target_lm_head_rejects_a_copy_without_sharing():
    """A quantized copy only exists for a drafter that shares the target head,
    e.g. not one with its own vocabulary (can_share False in DSpark)."""
    with pytest.raises(ValueError, match="needs a drafter that shares"):
        share_target_lm_head(
            _drafter(), _UnquantizedHead(torch.ones(4, 8)), object(), can_share=False
        )
    assert (
        share_target_lm_head(
            _drafter(), _UnquantizedHead(torch.ones(4, 8)), None, can_share=False
        )
        is None
    )
