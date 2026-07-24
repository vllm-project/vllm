# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling's muP logits path must honor an fp32 ``head_dtype``.

Inkling divides the final logits by a muP width multiplier and, for the served
checkpoint, folds ``1/mup`` into a bespoke ``addmm`` that emits logits in the
input (bf16) dtype. That fast path silently dropped an fp32 ``head_dtype``
(``--hf-overrides '{"head_dtype": "float32"}'``), which RL training-inference
consistency requires -- for both the target model and the MTP draft, whose
logits drive the rejection-sampling acceptance distribution.
"""

import types

import pytest
import torch

from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.models.inkling.nvidia.logits_processor import InklingLogitsProcessor
from vllm.models.inkling.nvidia.mtp import InklingMTP

MUP = 8.0
VOCAB = 64
HIDDEN = 16
NUM_TOKENS = 4


class _FakeLmHead:
    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.quant_method = UnquantizedEmbeddingMethod()


def _inputs():
    torch.manual_seed(0)
    hidden = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16)
    return hidden, weight


def _expected_fp32(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(hidden.float(), weight.float()) * (1.0 / MUP)


def test_target_logits_processor_fp32_head(default_vllm_config):
    lp = InklingLogitsProcessor(VOCAB, logits_mup_width_multiplier=MUP)
    lp.head_dtype = torch.float32
    lp._gather_logits = lambda logits: logits

    hidden, weight = _inputs()
    logits = lp(_FakeLmHead(weight), hidden)

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, _expected_fp32(hidden, weight))


def test_target_logits_processor_default_head_stays_bf16(default_vllm_config):
    # head_dtype unset -> the muP addmm fast path is preserved (bf16 out).
    lp = InklingLogitsProcessor(VOCAB, logits_mup_width_multiplier=MUP)
    assert lp.head_dtype is None
    lp._gather_logits = lambda logits: logits

    hidden, weight = _inputs()
    logits = lp(_FakeLmHead(weight), hidden)

    assert logits.dtype == torch.bfloat16
    torch.testing.assert_close(
        logits.float(), _expected_fp32(hidden, weight), atol=0.5, rtol=0.05
    )


def _fake_mtp(head_dtype: torch.dtype | None) -> InklingMTP:
    mtp = InklingMTP.__new__(InklingMTP)
    torch.nn.Module.__init__(mtp)
    mtp.config = types.SimpleNamespace(logits_mup_width_multiplier=MUP)
    lp = InklingLogitsProcessor(VOCAB)
    lp.head_dtype = head_dtype
    lp._gather_logits = lambda logits: logits
    mtp.logits_processor = lp
    mtp._logits_zero = None
    return mtp


def test_mtp_draft_compute_logits_fp32_head(default_vllm_config):
    hidden, weight = _inputs()
    mtp = _fake_mtp(torch.float32)
    mtp.lm_head = _FakeLmHead(weight)

    logits = mtp.compute_logits(hidden)

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, _expected_fp32(hidden, weight))


def test_mtp_draft_compute_logits_default_head_stays_bf16(default_vllm_config):
    hidden, weight = _inputs()
    mtp = _fake_mtp(None)
    mtp.lm_head = _FakeLmHead(weight)

    logits = mtp.compute_logits(hidden)

    assert logits.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp32 fold is CUDA-only")
def test_target_fp32_fold_matches_unfused_projection(default_vllm_config):
    # On CUDA the fp32 head folds muP into ``addmm(out_dtype=fp32)``; it must be
    # bit-for-bit identical to the dtype-aware projection + elementwise multiply.
    lp = InklingLogitsProcessor(VOCAB, logits_mup_width_multiplier=MUP)
    lp.head_dtype = torch.float32
    lp._gather_logits = lambda logits: logits

    hidden, weight = _inputs()
    hidden, weight = hidden.cuda(), weight.cuda()
    head = _FakeLmHead(weight)

    folded = lp(head, hidden)
    unfused = lp._get_logits(hidden, head, None) * (1.0 / MUP)

    assert folded.dtype == torch.float32
    torch.testing.assert_close(folded, unfused, atol=0.0, rtol=0.0)
