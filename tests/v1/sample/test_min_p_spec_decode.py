# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``MinPLogitsProcessor.apply_with_spec_decode``.

The bug fixed here is that ``RejectionSampler.apply_logits_processors``
previously only iterated ``logitsprocs.non_argmax_invariant`` (which excludes
``MinPLogitsProcessor`` since min_p is argmax-invariant). As a result min_p
was silently dropped for verified target tokens during speculative decoding,
and only applied to the bonus token via the regular ``Sampler`` path.
"""
import pytest
import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor


DEVICE_TYPE = current_platform.device_type


def _reference_min_p_mask(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """Reference: same body as ``MinPLogitsProcessor.apply`` but on a copy."""
    out = logits.clone()
    probs = torch.nn.functional.softmax(out, dim=-1)
    max_p = torch.amax(probs, dim=-1, keepdim=True)
    threshold = max_p * min_p
    out.masked_fill_(probs < threshold, -float("inf"))
    return out


def _build_proc(max_num_seqs: int, device: str) -> MinPLogitsProcessor:
    cfg = VllmConfig()
    cfg.scheduler_config.max_num_seqs = max_num_seqs
    proc = MinPLogitsProcessor(
        vllm_config=cfg,
        device=torch.device(device),
        is_pin_memory=is_pin_memory_available(),
    )
    return proc


def _seed_min_p(proc: MinPLogitsProcessor, values: list[float]) -> None:
    """Populate the proc's min_p state without going through BatchUpdate."""
    n = len(values)
    for i, v in enumerate(values):
        proc.min_p_cpu[i] = v
        if v:
            proc.min_p_count += 1
    proc.min_p = proc.min_p_device[:n]
    if proc.use_double_tensor:
        proc.min_p.copy_(proc.min_p_cpu_tensor[:n], non_blocking=False)
    proc.min_p.unsqueeze_(1)


@pytest.mark.parametrize("device", [DEVICE_TYPE])
def test_apply_with_spec_decode_noop_when_disabled(device):
    """No request has min_p > 0 → method must be a pass-through."""
    proc = _build_proc(max_num_seqs=4, device=device)
    _seed_min_p(proc, [0.0, 0.0])

    logits = torch.randn(5, 32, device=device)
    before = logits.clone()
    out = proc.apply_with_spec_decode(logits, num_draft_tokens=[2, 3])
    assert torch.equal(out, before)


@pytest.mark.parametrize("device", [DEVICE_TYPE])
def test_apply_with_spec_decode_masks_per_request_rows(device):
    """Per-row min_p must be expanded from per-request via repeat_interleave.

    Two requests with different min_p, num_draft_tokens=[2, 3] → 5 rows total.
    Rows 0,1 should be masked using request 0's min_p; rows 2,3,4 using
    request 1's min_p. Output must match a per-row reference apply.
    """
    proc = _build_proc(max_num_seqs=4, device=device)
    _seed_min_p(proc, [0.1, 0.05])  # request 0 stricter than request 1

    torch.manual_seed(0)
    logits = torch.randn(5, 64, device=device)
    expected_min_p = torch.tensor(
        [[0.1], [0.1], [0.05], [0.05], [0.05]], device=device
    )
    expected = _reference_min_p_mask(logits, expected_min_p)

    out = proc.apply_with_spec_decode(logits, num_draft_tokens=[2, 3])
    assert torch.equal(out, expected)


@pytest.mark.parametrize("device", [DEVICE_TYPE])
def test_apply_with_spec_decode_mixed_zero_min_p(device):
    """A request with min_p=0 must leave its rows untouched while a peer with
    min_p>0 still gets masking."""
    proc = _build_proc(max_num_seqs=4, device=device)
    _seed_min_p(proc, [0.0, 0.2])

    torch.manual_seed(1)
    logits = torch.randn(4, 32, device=device)
    expected_min_p = torch.tensor(
        [[0.0], [0.0], [0.2], [0.2]], device=device
    )
    expected = _reference_min_p_mask(logits, expected_min_p)

    out = proc.apply_with_spec_decode(logits, num_draft_tokens=[2, 2])
    assert torch.equal(out, expected)
    # min_p=0 rows must be byte-identical to input (no masking, no NaN).
    assert not torch.isinf(out[:2]).any()
