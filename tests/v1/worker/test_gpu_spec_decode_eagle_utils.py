# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share


class _StubQuantMethod(QuantizeMethodBase):
    """Minimal non-unquantized method that keeps a `.weight` parameter."""

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs):
        pass

    def apply(self, layer: torch.nn.Module, *args, **kwargs):
        raise NotImplementedError


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


class _QuantizedWeightHead(torch.nn.Module):
    """A quantized head that retains `.weight` alongside a separate scale."""

    def __init__(self, weight: torch.Tensor, scale: float = 1.0):
        super().__init__()
        self.quant_method = _StubQuantMethod()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(
            torch.full((weight.shape[0],), scale), requires_grad=False
        )


def _parallel_lm_head(
    vocab_size: int,
    hidden_size: int,
    fill_value: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
) -> ParallelLMHead:
    head = ParallelLMHead(
        vocab_size,
        hidden_size,
        params_dtype=dtype,
        padding_size=1,
        disable_tp=True,
    )
    head.weight.data.fill_(fill_value)
    return head


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


def test_identical_parallel_lm_heads_are_shared():
    draft = _parallel_lm_head(64, 16, fill_value=1.0)
    target = _parallel_lm_head(64, 16, fill_value=1.0)
    assert _should_share(_eagle(), "has_own_lm_head", draft, target) is True


def test_mismatched_dtype_heads_are_not_shared():
    draft = _parallel_lm_head(64, 16, fill_value=1.0, dtype=torch.bfloat16)
    target = _parallel_lm_head(64, 16, fill_value=1.0, dtype=torch.float16)
    assert _should_share(_eagle(), "has_own_lm_head", draft, target) is False


def test_bf16_draft_and_fp8_quantized_target_are_not_shared():
    draft = _parallel_lm_head(64, 16, fill_value=1.0, dtype=torch.bfloat16)
    target = _QuantizedWeightHead(torch.ones(64, 16, dtype=torch.float8_e4m3fn))
    assert _should_share(_eagle(), "has_own_lm_head", draft, target) is False


def test_non_standard_float_dtype_heads_are_not_shared():
    fp8_weight = torch.ones(4, 8, dtype=torch.float8_e4m3fn)
    draft = _UnquantizedHead(fp8_weight.clone())
    target = _UnquantizedHead(fp8_weight.clone())
    assert _should_share(_eagle(), "has_own_lm_head", draft, target) is False


def test_quantized_weight_heads_with_scales_are_not_shared():
    fp8_weight = torch.ones(4, 8, dtype=torch.float8_e4m3fn)
    draft = _QuantizedWeightHead(fp8_weight.clone(), scale=1.0)
    target = _QuantizedWeightHead(fp8_weight.clone(), scale=2.0)
    assert _should_share(_eagle(), "has_own_lm_head", draft, target) is False


def test_quantized_and_unquantized_same_dtype_heads_are_not_shared():
    weight = torch.ones(64, 16, dtype=torch.bfloat16)
    draft = _QuantizedWeightHead(weight.clone())
    target = _parallel_lm_head(64, 16, fill_value=1.0, dtype=torch.bfloat16)
    assert _should_share(_eagle(), "has_own_lm_head", draft, target) is False


def test_mismatched_shape_heads_are_not_shared():
    draft = _parallel_lm_head(64, 16, fill_value=1.0)
    target = _parallel_lm_head(128, 16, fill_value=1.0)
    assert _should_share(_eagle(), "has_own_lm_head", draft, target) is False


def test_mismatched_device_heads_are_not_shared():
    draft = _UnquantizedHead(torch.ones(4, 8, dtype=torch.bfloat16))
    target = _UnquantizedHead(torch.empty(4, 8, dtype=torch.bfloat16, device="meta"))
    assert _should_share(_eagle(), "has_own_lm_head", draft, target) is False


def test_draft_without_own_head_still_shares_quantized_target():
    target = _QuantizedWeightHead(torch.ones(4, 8, dtype=torch.float8_e4m3fn))
    assert _should_share(_eagle(has_own=False), "has_own_lm_head", None, target) is True
