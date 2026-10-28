# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share


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
