# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DRY (Don't Repeat Yourself) logits processor.

Covers the penalty computation (against llama.cpp's own worked example and
a brute-force oracle), the exponent-clamp float32 semantics, breaker
containment resolution, batch state management, and parameter validation.
"""

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.dry_utils import max_exponent as _max_exponent
from vllm.v1.sample.logits_processor import dry_batched
from vllm.v1.sample.logits_processor.dry import (
    DryLogitsProcessor,
    _dry_penalties,
    _DryState,
)
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)

DEVICE = torch.device("cpu")


def _penalties(window, multiplier=1.0, base=2.0, allowed_length=1, breakers=()):
    return _dry_penalties(
        window,
        frozenset(breakers),
        multiplier,
        base,
        allowed_length,
        _max_exponent(base),
    )


def test_worked_example():
    # llama.cpp's own commented example (src/llama-sampler.cpp): tokens
    # "a b c c b c y a b c" -> per-position repeat counts
    # [0,0,3,1,0,2,0,0,0,0]; extenders: c (len 3), b (len 1), y (len 2).
    # With multiplier=1, base=2, allowed_length=1: c -> 4, b -> 1, y -> 2.
    got = _penalties([1, 2, 3, 3, 2, 3, 9, 1, 2, 3])
    assert got == {3: 4.0, 2: 1.0, 9: 2.0}


def test_z_box_equality_boundary():
    # The Z-algorithm's copy-vs-extend equality boundary: when
    # cnt[last - p] == right_part_len the match may extend past the Z-box
    # and must be re-scanned, not copied. Copying here drops
    # token 0's penalty on this input.
    got = _penalties([0, 0, 0, 1, 0, 0], allowed_length=2)
    assert got == {0: 1.0, 1: 1.0}


def test_exponent_clamp_float32_semantics():
    # llama.cpp computes FLOAT_MAX_LOG / log(base) in float32. At base=2.0
    # that is exactly 128.0; float64 gives 127.99999998 -> 127. The clamp
    # must land on 128 or long-repeat penalties come out half of
    # llama.cpp's and never saturate to -inf.
    assert _max_exponent(2.0) == 128
    assert _max_exponent(1.75) == 158
    assert _max_exponent(1.0) == 0  # disabled guard
    assert _max_exponent(1.0000005) == 0  # llama.cpp's 1.000001 guard


def test_long_repeat_saturates_to_neg_inf():
    # 200 identical tokens at base=2.0: clamped exponent 128 ->
    # 2**128 overflows float32 -> logit must land at exactly -inf,
    # matching llama.cpp.
    proc = _make_processor()
    params = SamplingParams(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=0,
        dry_sequence_breakers=[],
    )
    _add_request(proc, index=0, params=params, prompt=[7] * 200, output=[])
    logits = torch.zeros(1, 16, device=DEVICE)
    out = proc.apply(logits)
    assert out[0, 7].item() == -float("inf")


def test_allowed_length_threshold():
    # Match of length 2 is penalized at allowed_length=2, not at 3.
    window = [1, 2, 3, 1, 2]
    assert _penalties(window, allowed_length=3) == {}
    assert _penalties(window, allowed_length=2) == {3: 1.0}


def test_breakers_cap_and_exclude():
    # An unbroken run is penalized; making that token a breaker
    # collapses rep_limit to 0 and produces nothing.
    assert _penalties([7] * 6, allowed_length=2) == {7: 8.0}
    assert _penalties([7] * 6, allowed_length=2, breakers=[7]) == {}


def test_short_windows_are_safe():
    assert _penalties([]) == {}
    assert _penalties([5]) == {}


def _make_processor() -> DryLogitsProcessor:
    return DryLogitsProcessor(
        SimpleNamespace(model_config=None), DEVICE, is_pin_memory=False
    )


def _add_request(proc, index, params, prompt, output):
    update = BatchUpdate(
        batch_size=index + 1,
        removed=[],
        added=[(index, params, prompt, output)],
        moved=[],
    )
    proc.update_state(update)


def test_state_windowing_spans_prompt_and_output():
    # penalty_last_n window must cross the prompt/output boundary.
    proc = _make_processor()
    params = SamplingParams(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=1,
        dry_sequence_breakers=[],
    )
    # worked example split across prompt and live output list
    output = [9, 1, 2, 3]
    _add_request(proc, 0, params, prompt=[1, 2, 3, 3, 2, 3], output=output)
    logits = torch.zeros(1, 16, device=DEVICE)
    out = proc.apply(logits)
    assert out[0, 3].item() == -4.0
    assert out[0, 2].item() == -1.0
    assert out[0, 9].item() == -2.0

    # The output list reference is live: appending tokens changes the
    # next apply() with no update_state call. The suffix is now
    # "... 3 3 3", whose longest earlier match has length 2, so token 3's
    # penalty is -2.0; a stale window would still give -4.0.
    output.extend([3, 3])
    out2 = proc.apply(torch.zeros(1, 16, device=DEVICE))
    assert out2[0, 3].item() == -2.0
    assert out2[0, 2].item() == -2.0
    assert out2[0, 9].item() == -1.0


def test_state_disabled_requests_not_tracked():
    proc = _make_processor()
    for params in (
        SamplingParams(),  # dry_multiplier defaults to 0
        SamplingParams(dry_multiplier=1.0, dry_base=0.5),  # base<1 disables
        SamplingParams(dry_multiplier=1.0, dry_penalty_last_n=0),
    ):
        _add_request(proc, 0, params, prompt=[1, 1, 1, 1], output=[])
        assert not proc.reqs
    logits = torch.zeros(1, 8, device=DEVICE)
    assert proc.apply(logits) is logits  # zero-cost passthrough


def test_state_remove_and_move():
    proc = _make_processor()
    params = SamplingParams(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=1,
        dry_sequence_breakers=[],
    )
    _add_request(proc, 0, params, prompt=[7] * 8, output=[])
    assert 0 in proc.reqs

    # Move 0 -> 2 (unidirectional): penalties must follow the request.
    proc.update_state(
        BatchUpdate(
            batch_size=3,
            removed=[],
            added=[],
            moved=[(0, 2, MoveDirectionality.UNIDIRECTIONAL)],
        )
    )
    assert 2 in proc.reqs and 0 not in proc.reqs
    logits = torch.zeros(3, 16, device=DEVICE)
    out = proc.apply(logits)
    assert out[2, 7].item() < 0
    assert out[0, 7].item() == 0

    # Remove.
    proc.update_state(BatchUpdate(batch_size=3, removed=[2], added=[], moved=[]))
    assert not proc.reqs


class _StubTokenizer:
    """Minimal tokenizer for breaker-resolution tests."""

    _TEXTS = ["a", "\n", "b:\n", "hello", ":", "x:y", "*", "**"]

    vocab_size = len(_TEXTS)

    def batch_decode(self, ids_lists):
        return [self._TEXTS[ids[0]] for ids in ids_lists]


def test_breaker_containment_resolution():
    # Resolution must match every token whose text CONTAINS the breaker
    # string (llama.cpp get_overlapping_token_sequences), not just exact
    # encodings. Shared between both sampler stacks via dry_utils.
    from vllm.v1.sample.dry_utils import resolve_dry_breakers

    tok = _StubTokenizer()
    assert resolve_dry_breakers(tok, ("\n",)) == [1, 2]
    assert resolve_dry_breakers(tok, (":",)) == [2, 4, 5]
    assert resolve_dry_breakers(tok, ("*",)) == [6, 7]
    assert resolve_dry_breakers(tok, ()) == []
    # Repeat resolution is consistent, and callers get their own copy
    # (mutating a result must not poison the cache).
    first = resolve_dry_breakers(tok, ("\n",))
    first.append(999)
    assert resolve_dry_breakers(tok, ("\n",)) == [1, 2]


def test_sampling_params_rejects_json_booleans():
    # bool is an int subclass; JSON true/false must 400, not crash the
    # engine at apply time.
    for name in (
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_penalty_last_n",
    ):
        with pytest.raises((VLLMValidationError, ValueError)):
            SamplingParams(**{name: True})


def test_sampling_params_validation():
    with pytest.raises((VLLMValidationError, ValueError)):
        SamplingParams(dry_multiplier=-1.0)
    with pytest.raises((VLLMValidationError, ValueError)):
        SamplingParams(dry_penalty_last_n=-7)
    with pytest.raises((VLLMValidationError, ValueError)):
        SamplingParams(dry_allowed_length=-1)
    with pytest.raises((VLLMValidationError, ValueError)):
        SamplingParams(dry_sequence_breakers=[1, 2])
    with pytest.raises((VLLMValidationError, ValueError)):
        SamplingParams(dry_allowed_length=1.5)
    with pytest.raises((VLLMValidationError, ValueError)):
        SamplingParams(dry_penalty_last_n=2.5)
    with pytest.raises((VLLMValidationError, ValueError)):
        SamplingParams(dry_sequence_breakers=[f"b{i}" for i in range(65)])
    # Valid corner values pass.
    SamplingParams(dry_multiplier=0.0)
    SamplingParams(dry_multiplier=1.0, dry_penalty_last_n=-1)
    SamplingParams(dry_multiplier=1.0, dry_sequence_breakers=[])


def _oracle(window, breakers, multiplier, base, allowed_length, max_exponent):
    """Brute-force oracle: computes the same penalties by exhaustive
    comparison instead of the Z-algorithm, so it cannot share the fast
    path's bugs. Steps mirror llama_sampler_dry_apply directly."""
    m = len(window)
    rep_limit = m
    for i in range(m):
        if window[m - 1 - i] in breakers:
            rep_limit = i
            break
    if rep_limit < allowed_length:
        return {}
    cnt = [0] * m
    for i in range(m - 1):
        length = 0
        while (
            length < m
            and i - length >= 0
            and window[i - length] == window[m - 1 - length]
        ):
            length += 1
        cnt[i] = min(length, rep_limit)
    max_tok: dict[int, int] = {}
    for i in range(m - 1):
        if cnt[i] >= allowed_length:
            tok = window[i + 1]
            if max_tok.get(tok, -1) < cnt[i]:
                max_tok[tok] = cnt[i]
    penalties = {}
    for tok, repeat_len in max_tok.items():
        if tok in breakers:
            continue
        exponent = repeat_len - allowed_length
        if max_exponent and exponent > max_exponent:
            exponent = max_exponent
        penalties[tok] = multiplier * (base**exponent)
    return penalties


def test_differential_against_oracle():
    rng = random.Random(0)
    for case in range(400):
        if case % 10 == 0:
            # Long, highly repetitive: the only way to reach the clamp.
            alphabet, n = rng.randint(1, 2), rng.randint(200, 320)
        else:
            alphabet, n = rng.randint(2, 6), rng.randint(0, 48)
        window = [rng.randrange(alphabet) for _ in range(n)]
        multiplier = rng.choice([0.5, 0.8, 1.0, 2.0])
        base = rng.choice([1.1, 1.75, 2.0, 3.0])
        allowed = rng.randint(0, 4)
        breakers = frozenset(
            rng.sample(range(alphabet), rng.randint(0, min(2, alphabet)))
        )
        max_exp = _max_exponent(base)
        want = _oracle(window, breakers, multiplier, base, allowed, max_exp)
        got = _dry_penalties(window, breakers, multiplier, base, allowed, max_exp)
        assert set(want) == set(got), f"case {case}: keys {want} vs {got}"
        for tok in want:
            a, b = want[tok], got[tok]
            assert math.isclose(a, b, rel_tol=1e-9), (
                f"case {case} token {tok}: {a} vs {b}"
            )


def _rand_state(rng, alphabet):
    n = rng.randint(3, 120)

    base = rng.choice([1.1, 1.75, 2.0, 3.0])
    return _DryState(
        multiplier=rng.choice([0.5, 0.8, 1.0, 2.0]),
        base=base,
        allowed_length=rng.randint(0, 3),
        penalty_last_n=rng.choice([-1, 5, 50]),
        breakers=frozenset(
            rng.sample(range(alphabet), rng.randint(0, min(2, alphabet)))
        ),
        max_exponent=_max_exponent(base),
        prompt_tok_ids=[rng.randrange(alphabet) for _ in range(n)],
        output_tok_ids=[],
    )


def test_batched_matches_reference_fuzz():
    # Mixed-parameter multi-row batches through the vectorized path must
    # reproduce the sequential reference exactly (inf included).
    vocab = 32
    rng = random.Random(1)
    total_rows = 0
    for _ in range(40):
        entries: list[tuple[int, _DryState, list[int]]] = []
        for row in range(rng.randint(1, 8)):
            state = _rand_state(rng, alphabet=rng.randint(1, 6))
            if state.window() is None or not dry_batched.eligible(state):
                continue
            entries.append((len(entries), state, state.window()))
        if not entries:
            continue
        total_rows += len(entries)
        logits = torch.zeros(len(entries), vocab)
        out = dry_batched.apply_dry_batched(logits, entries)
        for row, state, window in entries:
            want = _dry_penalties(
                window,
                state.breakers,
                state.multiplier,
                state.base,
                state.allowed_length,
                state.max_exponent,
            )
            for tok in range(vocab):
                expected = -want.get(tok, 0.0)
                got = out[row, tok].item()
                if math.isinf(expected) or math.isinf(got):
                    assert math.isinf(expected) == math.isinf(got), (
                        f"row {row} tok {tok}: {expected} vs {got}"
                    )
                else:
                    assert math.isclose(expected, got, rel_tol=1e-6, abs_tol=1e-6), (
                        f"row {row} tok {tok}: {expected} vs {got}"
                    )
    assert total_rows > 50  # the fuzz actually exercised batches


def test_batched_double_pow_saturation():
    # llama.cpp's std::pow(float, int) computes in double: 0.8 * 2**128
    # is finite in float32 (-2.72e38) while 1.0 * 2**128 saturates to
    # -inf. A float32 pow gets the first case wrong.
    for mult, expect_inf in ((0.8, False), (1.0, True)):
        state = _DryState(
            multiplier=mult,
            base=2.0,
            allowed_length=0,
            penalty_last_n=-1,
            breakers=frozenset(),
            max_exponent=_max_exponent(2.0),
            prompt_tok_ids=[7] * 200,
            output_tok_ids=[],
        )
        logits = torch.zeros(1, 16)
        out = dry_batched.apply_dry_batched(logits, [(0, state, state.window())])
        val = out[0, 7].item()
        if expect_inf:
            assert val == -float("inf")
        else:
            assert math.isfinite(val) and math.isclose(
                val, -0.8 * 2.0**128, rel_tol=1e-6
            )


def test_processor_dispatch_batched():
    # Forcing the batched path through the processor must give the same
    # worked-example result as the reference path.
    proc = _make_processor()
    proc.use_batched = True
    params = SamplingParams(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=1,
        dry_sequence_breakers=[],
    )
    _add_request(proc, 0, params, prompt=[1, 2, 3, 3, 2, 3, 9, 1, 2, 3], output=[])
    out = proc.apply(torch.zeros(1, 16, device=DEVICE))
    assert out[0, 3].item() == -4.0
    assert out[0, 2].item() == -1.0
    assert out[0, 9].item() == -2.0


# ---------------------------------------------------------------------------
# V2-runner DryState module
# ---------------------------------------------------------------------------


def _make_v2_state(max_num_reqs=8, vocab=32, max_model_len=256):
    from vllm.v1.worker.gpu.sample.dry import DryState

    all_tokens = torch.zeros(max_num_reqs, max_model_len, dtype=torch.int32)
    req_states = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        vocab_size=vocab,
        device=DEVICE,
        all_token_ids=SimpleNamespace(gpu=all_tokens),
    )
    return DryState(req_states), all_tokens


def test_v2_worked_example_and_pos_semantics():
    # ``pos`` is the position of the last INPUT token; the window is
    # [0, pos + 1). Without the + 1 the window drops the final context
    # token and every penalty lands on the continuation of the previous
    # suffix instead of the token being sampled.
    state, all_tokens = _make_v2_state()
    params = SamplingParams(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=1,
        dry_sequence_breakers=[],
    )
    state.add_request(3, params)
    seq = [1, 2, 3, 3, 2, 3, 9, 1, 2, 3]
    all_tokens[3, : len(seq)] = torch.tensor(seq, dtype=torch.int32)

    logits = torch.zeros(1, 32, device=DEVICE)
    idx_mapping = np.array([3])
    # last input token sits at position len(seq)-1
    pos = torch.tensor([len(seq) - 1], device=DEVICE)
    state.apply_dry(logits, idx_mapping, pos, expanded_logits=False)
    assert logits[0, 3].item() == -4.0
    assert logits[0, 2].item() == -1.0
    assert logits[0, 9].item() == -2.0


def test_v2_disabled_and_gating():
    state, _ = _make_v2_state()
    state.add_request(0, SamplingParams())  # dry off by default
    state.add_request(1, SamplingParams(dry_multiplier=1.0, dry_base=0.5))
    state.add_request(2, SamplingParams(dry_multiplier=1.0, dry_penalty_last_n=0))
    assert not state.use_dry[:3].any()
    # Re-adding a slot with DRY off after one with DRY on must clear it.
    on = SamplingParams(dry_multiplier=0.8, dry_sequence_breakers=[])
    state.add_request(4, on)
    assert state.use_dry[4]
    state.add_request(4, SamplingParams())
    assert not state.use_dry[4]


def test_v2_penalty_last_n_window():
    # A cap smaller than the history must limit the visible window.
    state, all_tokens = _make_v2_state()
    params = SamplingParams(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=1,
        dry_penalty_last_n=4,
        dry_sequence_breakers=[],
    )
    state.add_request(0, params)
    seq = [7] * 20
    all_tokens[0, : len(seq)] = torch.tensor(seq, dtype=torch.int32)
    logits = torch.zeros(1, 32, device=DEVICE)
    state.apply_dry(
        logits, np.array([0]), torch.tensor([19], device=DEVICE), expanded_logits=False
    )
    # window = last 4 tokens of an identical run: longest match ending
    # before the suffix is 3 (rep-limited by the window), penalty 2**(3-1).
    assert logits[0, 7].item() == -4.0


def test_v2_spec_decode_skipped_with_warning():
    state, all_tokens = _make_v2_state()
    state.add_request(0, SamplingParams(dry_multiplier=0.8, dry_sequence_breakers=[]))
    all_tokens[0, :8] = 7
    logits = torch.zeros(3, 32, device=DEVICE)  # expanded: 3 rows, 1 req
    state.apply_dry(
        logits, np.array([0]), torch.tensor([7], device=DEVICE), expanded_logits=True
    )
    assert (logits == 0).all()
    assert state._warned_spec_decode


def test_v2_matches_reference_fuzz():
    # The V2 window-gather + routing path must agree with the sequential
    # reference on randomized histories, including degenerate bases that
    # route through the slow path.
    state, all_tokens = _make_v2_state(max_num_reqs=8, vocab=16, max_model_len=200)
    rng = random.Random(7)
    for trial in range(60):
        n_reqs = rng.randint(1, 5)
        histories = {}
        pos_list = []
        idx_list = []
        for r in range(n_reqs):
            params = SamplingParams(
                dry_multiplier=rng.choice([0.8, 1.0, 2.0]),
                dry_base=rng.choice([1.0000005, 1.1, 1.75, 2.0]),
                dry_allowed_length=rng.randint(0, 3),
                dry_penalty_last_n=rng.choice([-1, 5, 50]),
                dry_sequence_breakers=[],
            )
            state.add_request(r, params)
            n = rng.randint(2, 180)
            hist = [rng.randrange(rng.randint(1, 4) + 1) for _ in range(n)]
            all_tokens[r, :n] = torch.tensor(hist, dtype=torch.int32)
            histories[r] = (params, hist)
            idx_list.append(r)
            pos_list.append(n - 1)

        logits = torch.zeros(n_reqs, 16, device=DEVICE)
        state.apply_dry(
            logits,
            np.array(idx_list),
            torch.tensor(pos_list, device=DEVICE),
            expanded_logits=False,
        )
        for r in range(n_reqs):
            params, hist = histories[r]
            base32 = float(np.float32(params.dry_base))
            if base32 < 1.0 or not params.dry_multiplier:
                want: dict[int, float] = {}
            else:
                last_n = params.dry_penalty_last_n
                w = len(hist) if last_n == -1 else min(len(hist), last_n)
                window = hist[-w:] if w > params.dry_allowed_length else None
                want = (
                    _dry_penalties(
                        window,
                        frozenset(),
                        float(np.float32(params.dry_multiplier)),
                        base32,
                        params.dry_allowed_length,
                        _max_exponent(base32),
                    )
                    if window is not None
                    else {}
                )
            for tok in range(16):
                expected = -want.get(tok, 0.0)
                got = logits[r, tok].item()
                assert math.isclose(expected, got, rel_tol=1e-5, abs_tol=1e-6), (
                    f"trial {trial} req {r} tok {tok}: {expected} vs {got}"
                )


def test_needs_output_token_ids_tracks_dry_requests():
    # Async scheduling only backfills the output token id lists when a
    # processor in the batch reports needing them; see
    # LogitsProcessor.needs_output_token_ids.
    proc = _make_processor()
    assert not proc.needs_output_token_ids()
    params = SamplingParams(dry_multiplier=0.8, dry_sequence_breakers=[])
    _add_request(proc, 0, params, prompt=[1, 2, 3], output=[])
    assert proc.needs_output_token_ids()
    proc.update_state(BatchUpdate(batch_size=1, removed=[0], added=[], moved=[]))
    assert not proc.needs_output_token_ids()


def test_window_boundary_last_n_equals_output_len():
    # penalty_last_n == len(output) with a nonempty prompt: the window
    # must be exactly the last penalty_last_n tokens, not prompt+output
    # (the <= boundary in window()).
    st = _DryState(
        multiplier=1.0,
        base=2.0,
        allowed_length=1,
        penalty_last_n=2,
        breakers=frozenset(),
        max_exponent=_max_exponent(2.0),
        prompt_tok_ids=[7, 9],
        output_tok_ids=[7, 7],
    )
    assert st.window() == [7, 7]


def test_processor_honors_frontend_resolved_breaker_ids():
    # The engine frontend resolves breaker strings to ids and stores them
    # on the request (_dry_breaker_ids); the processor must consume them.
    proc = _make_processor()
    params = SamplingParams(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=1,
        dry_sequence_breakers=["x"],
    )
    params._dry_breaker_ids = [7]
    _add_request(proc, 0, params, prompt=[7] * 6, output=[])
    out = proc.apply(torch.zeros(1, 16, device=DEVICE))
    assert out[0, 7].item() == 0.0


def test_v2_breaker_ids_reach_dry_core():
    # Resolved breaker ids must flow through V2 add_request into the
    # penalty computation (and be cleared when the slot is reused).
    state, all_tokens = _make_v2_state()
    params = SamplingParams(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=1,
        dry_sequence_breakers=["x"],
    )
    params._dry_breaker_ids = [7]
    state.add_request(0, params)
    all_tokens[0, :8] = 7
    logits = torch.zeros(1, 32, device=DEVICE)
    state.apply_dry(
        logits,
        np.array([0]),
        torch.tensor([7], device=DEVICE),
        expanded_logits=False,
    )
    assert logits[0, 7].item() == 0.0

    # Slot reuse with a breaker-less request must not inherit the mask.
    fresh = SamplingParams(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=1,
        dry_sequence_breakers=[],
    )
    state.add_request(0, fresh)
    logits2 = torch.zeros(1, 32, device=DEVICE)
    state.apply_dry(
        logits2,
        np.array([0]),
        torch.tensor([7], device=DEVICE),
        expanded_logits=False,
    )
    assert logits2[0, 7].item() < 0.0


def test_v2_degenerate_base_routes_to_unclamped_slow_path():
    # base <= 1.000001 gives max_exponent == 0, which in the sequential
    # reference means "no clamp"; the vectorized path would clamp the
    # exponent to 0, so such requests must route to the reference scan.
    state, all_tokens = _make_v2_state()
    params = SamplingParams(
        dry_multiplier=1.0,
        dry_base=1.0000005,
        dry_allowed_length=0,
        dry_sequence_breakers=[],
    )
    state.add_request(0, params)
    all_tokens[0, :150] = 7
    logits = torch.zeros(1, 32, device=DEVICE)
    state.apply_dry(
        logits,
        np.array([0]),
        torch.tensor([149], device=DEVICE),
        expanded_logits=False,
    )
    base32 = float(np.float32(1.0000005))
    want = -1.0 * base32**149
    got = logits[0, 7].item()
    assert math.isclose(got, want, rel_tol=1e-6)
    assert got != -1.0


def test_breaker_resolution_covers_added_tokens():
    # Added/special tokens (chat markers) live above vocab_size on many
    # tokenizers; llama.cpp's containment scan covers the full id range,
    # so ours must too (max_token_id, not vocab_size).
    class AddedTokenTokenizer:
        _TEXTS = ["a", "\n", "b", "<|im_start|>"]
        vocab_size = 3  # the added token sits above vocab_size
        max_token_id = 3

        def batch_decode(self, ids_lists):
            return [self._TEXTS[ids[0]] for ids in ids_lists]

    from vllm.v1.sample.dry_utils import resolve_dry_breakers

    tok = AddedTokenTokenizer()
    assert resolve_dry_breakers(tok, ("<|im_start|>",)) == [3]
    assert resolve_dry_breakers(tok, ("\n",)) == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_batched_peak_memory_bounded():
    # base=1.1 gives max_exponent=930, a large J for the batched path
    # (the cap is allowed_length + max_exponent <= _J_BUDGET = 2048).
    # The chunk budget is denominated in bytes and the gather is int32,
    # so peak transient memory stays near _CHUNK_BYTE_BUDGET even at
    # large batch x window; an element-denominated budget with an int64
    # gather allocated ~8x more and OOMed 8 GB GPUs.
    from vllm.v1.sample.logits_processor.dry_batched import _CHUNK_BYTE_BUDGET

    device = torch.device("cuda")
    rng = random.Random(3)
    entries = []
    for r in range(32):
        state = _DryState(
            multiplier=0.8,
            base=1.1,
            allowed_length=2,
            penalty_last_n=-1,
            breakers=frozenset(),
            max_exponent=_max_exponent(1.1),
            prompt_tok_ids=[rng.randrange(1000) for _ in range(2048)],
            output_tok_ids=[],
        )
        entries.append((r, state, state.window()))
    logits = torch.zeros(32, 128256, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    dry_batched.apply_dry_batched(logits, entries)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base_alloc
    # generous headroom over the budget for l_max, W and allocator slack
    assert peak < 2 * _CHUNK_BYTE_BUDGET, f"peak transient {peak / 2**20:.0f} MiB"
