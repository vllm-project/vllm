"""Unit tests for the target-token-scoring compact fast path.

These test the three concerns in isolation: the compact projector's numerical
equivalence to full-vocab candidate columns, the wave admission gate (one
rejection per condition, plus an eligible wave), and the compact sampler's
shape/stability/NaN contract. The runner integration itself is covered by the
existing sampler integration tests; this suite guards the math and the gate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.worker.target_token_scoring.admission import (
    evaluate_wave_admission,
)
from vllm.v1.worker.target_token_scoring.compact_sampler import compact_sample
from vllm.v1.worker.target_token_scoring.projector import (
    CompactLMHeadCache,
    project_target_token_logits,
)
from vllm.v1.worker.target_token_scoring.state import TargetTokenScoringState


def _make_model(vocab, hidden, *, bias=True, device="cpu"):
    weight = torch.randn(vocab, hidden, device=device)
    lm_head = SimpleNamespace(
        weight=weight,
        bias=torch.randn(vocab, device=device) if bias else None,
        tp_size=1,
        quant_method=None,
    )
    return SimpleNamespace(lm_head=lm_head, logits_processor=None)


def _decision(target_ids):
    from vllm.v1.worker.target_token_scoring.admission import AdmissionDecision

    return AdmissionDecision(
        ok=True, target_token_ids=list(target_ids)
    )


# --------------------------------------------------------------------------- #
# Projector: compact[:, j] == full[:, target_ids[j]]
# --------------------------------------------------------------------------- #


def test_projector_matches_full_vocab_columns():
    torch.manual_seed(0)
    model = _make_model(vocab=50, hidden=8)
    hidden_states = torch.randn(4, 8)
    target_ids = [3, 5, 7, 11]
    out = project_target_token_logits(
        model, hidden_states, _decision(target_ids), CompactLMHeadCache()
    )
    assert out is not None
    compact_logits, _state = out

    full = F.linear(hidden_states, model.lm_head.weight, model.lm_head.bias)
    expected = full[:, torch.tensor(target_ids)]
    assert torch.allclose(compact_logits, expected, atol=1e-5)


def test_projector_cache_invalidates_on_version_bump():
    torch.manual_seed(1)
    model = _make_model(vocab=20, hidden=4)
    hs = torch.randn(2, 4)
    ids = [1, 2]
    cache = CompactLMHeadCache()
    project_target_token_logits(
        model, hs, _decision(ids), cache
    )  # warm cache
    key = id(model.lm_head)
    assert key in cache._cache

    # Mutate the weight in place: data_ptr unchanged, _version increments.
    with torch.no_grad():
        model.lm_head.weight.add_(1.0)
    out = project_target_token_logits(model, hs, _decision(ids), cache)
    compact, _ = out
    full = F.linear(hs, model.lm_head.weight, model.lm_head.bias)
    expected = full[:, torch.tensor(ids)]
    assert torch.allclose(compact, expected, atol=1e-5), (
        "stale cached rows must not survive an in-place weight mutation"
    )


# --------------------------------------------------------------------------- #
# Admission: one rejection per gate condition, plus an eligible wave
# --------------------------------------------------------------------------- #


def _sp(**over):
    # Base request explicitly opts into target-set semantics; this is the
    # contract that admits the compact path.
    base = dict(
        logprob_token_ids=[3, 5],
        max_tokens=1,
        temperature=0,
        n=1,
        prompt_logprobs=None,
        logit_bias=None,
        allowed_token_ids=None,
        bad_words=[],
        guided_decoding=None,
        logits_processors=None,
        target_token_scoring_normalization="target_set",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _batch(req_ids):
    return SimpleNamespace(req_ids=list(req_ids), num_reqs=len(req_ids))


def _requests(req_ids, sp):
    return {rid: SimpleNamespace(sampling_params=sp) for rid in req_ids}


def _lm_head_eligible(vocab=100):
    return SimpleNamespace(
        weight=torch.randn(vocab, 4), tp_size=1, quant_method=None
    )


def _model_config(on=True):
    return SimpleNamespace(
        target_token_scoring=on, logits_processors=None
    )


def test_admission_eligible():
    ids = ["a", "b"]
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), _requests(ids, _sp()),
        _lm_head_eligible(), spec_decode_metadata=None,
    )
    assert d.ok and d.target_token_ids == [3, 5]


def test_admission_default_normalization_is_native():
    """A request that does NOT opt into target_set (the default full_vocab)
    must fall back to native. This is the footgun guard: enabling the engine
    flag never silently changes logprob_token_ids semantics for a request
    that only asked for full-vocab logprobs."""
    ids = ["a"]
    sp = _sp(target_token_scoring_normalization="full_vocab")
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), _requests(ids, sp),
        _lm_head_eligible(), spec_decode_metadata=None,
    )
    assert not d.ok
    assert "target_set" in (d.reason or "")


@pytest.mark.parametrize(
    "field,value,frag",
    [
        ("max_tokens", 2, "max_tokens"),
        ("temperature", 0.7, "temperature"),
        ("n", 2, "n="),
        ("prompt_logprobs", 1, "prompt_logprobs"),
        ("logit_bias", {1: 0.1}, "logit_bias"),
        ("allowed_token_ids", [1], "allowed_token_ids"),
        ("bad_words", ["x"], "bad_words"),
        ("guided_decoding", object(), "guided"),
        ("logits_processors", [object()], "logits_processors"),
        ("target_token_scoring_normalization", "full_vocab", "target_set"),
        ("logprob_token_ids", None, "no logprob_token_ids"),
    ],
)
def test_admission_rejects(field, value, frag):
    ids = ["a"]
    sp = _sp(**{field: value})
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), _requests(ids, sp),
        _lm_head_eligible(), spec_decode_metadata=None,
    )
    assert not d.ok
    assert frag in (d.reason or "")


def test_admission_rejects_mismatched_candidate_ids():
    ids = ["a", "b"]
    requests = {
        "a": SimpleNamespace(sampling_params=_sp(logprob_token_ids=[3, 5])),
        "b": SimpleNamespace(sampling_params=_sp(logprob_token_ids=[3, 7])),
    }
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), requests,
        _lm_head_eligible(), spec_decode_metadata=None,
    )
    assert not d.ok and "differ" in d.reason


def test_admission_rejects_spec_decode_or_flag_off_or_sharded_head():
    ids = ["a"]
    # spec decode
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), _requests(ids, _sp()),
        _lm_head_eligible(), spec_decode_metadata=object(),
    )
    assert not d.ok and "speculative" in d.reason
    # flag off
    d = evaluate_wave_admission(
        _model_config(on=False), _batch(ids), _requests(ids, _sp()),
        _lm_head_eligible(), spec_decode_metadata=None,
    )
    assert not d.ok and "flag" in d.reason
    # sharded head
    head = _lm_head_eligible()
    head.tp_size = 2
    d = evaluate_wave_admission(
        _model_config(), _batch(ids), _requests(ids, _sp()),
        head, spec_decode_metadata=None,
    )
    assert not d.ok and "sharded" in d.reason


# --------------------------------------------------------------------------- #
# Compact sampler: shape, stability, NaN
# --------------------------------------------------------------------------- #


def _state(target_ids, device="cpu"):
    return TargetTokenScoringState.from_ids(target_ids, torch.device(device))


def test_compact_sample_shape_and_logprobs_sum_to_one():
    torch.manual_seed(2)
    B, K = 5, 3
    logits = torch.randn(B, K)
    state = _state([10, 20, 30])
    out = compact_sample(logits, state, sampling_metadata=None)
    assert out.sampled_token_ids.shape == (B, 1)
    lt = out.logprobs_tensors
    assert lt is not None
    assert lt.logprob_token_ids.shape == (B, K + 1)
    assert lt.logprobs.shape == (B, K + 1)
    # candidate columns (1..K) are target-set logprobs -> exp sums to 1.
    probs = lt.logprobs[:, 1:].exp()
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
    # sampled is the argmax candidate id; ranks 0 for the argmax.
    assert (lt.selected_token_ranks == 0).all()
    assert out.sampled_token_ids[:, 0].tolist() == [
        [10, 20, 30][int(i)] for i in logits.argmax(-1)
    ]


def test_compact_sample_overflow_stability():
    # Huge logits must not overflow the stable logsumexp.
    logits = torch.tensor([[1e30, -1e30, 0.0]])
    out = compact_sample(logits, _state([1, 2, 3]), sampling_metadata=None)
    lp = out.logprobs_tensors.logprobs[0, 1:]
    assert torch.isfinite(lp).all()
    assert torch.allclose(lp.exp().sum(), torch.tensor(1.0), atol=1e-4)


def test_compact_sample_nan_fail_closed():
    logits = torch.tensor([[0.1, float("nan"), 0.3]])
    out = compact_sample(logits, _state([1, 2, 3]), sampling_metadata=None)
    # NaN row's emitted logprobs must be NaN (not a silently-wrong finite pick).
    assert torch.isnan(out.logprobs_tensors.logprobs[0, 1:]).any()
    assert out.sampled_token_ids.shape == (1, 1)
