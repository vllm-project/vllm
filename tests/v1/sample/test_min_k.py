# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric unit tests for the Min-k logits processor.

These pin the semantic-cliff truncation and the temperature-invariance
property from https://arxiv.org/abs/2604.11012, and run on CPU so no
accelerator is needed.
"""
import torch

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import MinKLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate

VOCAB_SIZE = 256


def _reference_k(row: torch.Tensor, tau: float, eps: float = 1e-8) -> int:
    """Algorithm 1 from the paper, computed on a single logit row."""
    sorted_logits, _ = torch.sort(row, descending=True)
    logit_range = (sorted_logits[0] - sorted_logits[-1]).item()
    best_decay = -1.0
    best_i = 1
    for i in range(1, sorted_logits.numel()):
        decay = (sorted_logits[i - 1] - sorted_logits[i]).item() / (
            i * (logit_range + eps)
        )
        if decay > best_decay:
            best_decay = decay
            best_i = i
    k_cliff = best_i
    k_fallback = int(max(0.0, tau) / (logit_range + eps))
    return max(k_cliff, k_fallback)


def _build_processor() -> MinKLogitsProcessor:
    return MinKLogitsProcessor(
        VllmConfig(), device=torch.device("cpu"), is_pin_memory=False
    )


def _update(proc: MinKLogitsProcessor, params: list[SamplingParams]) -> None:
    proc.update_state(
        BatchUpdate(
            batch_size=len(params),
            removed=[],
            added=[(i, p, None, []) for i, p in enumerate(params)],
            moved=[],
        )
    )


def test_min_k_matches_algorithm_reference():
    """Enabled rows truncate at the reference cliff position, and a disabled
    row is untouched, all within one mixed batch."""
    torch.manual_seed(0)
    proc = _build_processor()
    params = [
        SamplingParams(temperature=1.0, min_k=True, min_k_tau=3.0),
        SamplingParams(temperature=1.0, min_k=True, min_k_tau=0.0),
        SamplingParams(temperature=1.0),  # disabled
    ]
    _update(proc, params)

    logits = torch.randn(3, VOCAB_SIZE)
    logits[:, :4] += 6.0  # a clear head, then a cliff into the tail
    out = proc.apply(logits.clone())

    for i, p in enumerate(params):
        if not p.min_k:
            torch.testing.assert_close(out[i], logits[i])
            continue
        k = _reference_k(logits[i], p.min_k_tau)
        kth = torch.sort(logits[i], descending=True).values[k - 1]
        expected_kept = logits[i] >= kth
        got_kept = ~torch.isinf(out[i])
        assert torch.equal(got_kept, expected_kept), f"row {i}, k={k}"
        # The top-ranked token always survives.
        assert out[i].argmax() == logits[i].argmax()


def test_min_k_is_temperature_invariant():
    """The kept set is identical across temperatures for the cliff mechanism."""
    torch.manual_seed(1)
    logits = torch.randn(1, VOCAB_SIZE)
    logits[:, :5] += 8.0

    kept_sets = []
    for temperature in (1.0, 5.0, 10.0):
        proc = _build_processor()
        # tau=0 isolates the cliff mechanism from the fallback.
        _update(proc, [SamplingParams(temperature=1.0, min_k=True, min_k_tau=0.0)])
        out = proc.apply((logits / temperature).clone())
        kept_sets.append(~torch.isinf(out[0]))

    assert torch.equal(kept_sets[0], kept_sets[1])
    assert torch.equal(kept_sets[0], kept_sets[2])


def test_min_k_fallback_prevents_single_token_collapse():
    """On a near-flat distribution with no cliff, the fallback keeps more than
    one candidate."""
    proc = _build_processor()
    _update(proc, [SamplingParams(temperature=1.0, min_k=True, min_k_tau=3.0)])
    # Tiny, near-uniform spread so the logit range is small and the fallback
    # floor(tau / range) exceeds one.
    logits = torch.linspace(0.0, 0.05, VOCAB_SIZE).unsqueeze(0)
    out = proc.apply(logits.clone())
    assert int((~torch.isinf(out[0])).sum()) > 1


def test_min_k_disabled_is_noop():
    proc = _build_processor()
    _update(proc, [SamplingParams(temperature=1.0)])
    logits = torch.randn(1, VOCAB_SIZE)
    torch.testing.assert_close(proc.apply(logits.clone()), logits)


def test_min_k_tau_must_be_non_negative():
    try:
        SamplingParams(temperature=1.0, min_k=True, min_k_tau=-1.0)
    except ValueError:
        return
    raise AssertionError("negative min_k_tau should have been rejected")
