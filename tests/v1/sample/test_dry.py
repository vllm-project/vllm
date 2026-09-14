# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DRY (Don't Repeat Yourself) repetition penalty.

Covers the penalty computation (against llama.cpp's own worked example and
a brute-force oracle), the exponent-clamp float32 semantics, breaker
containment resolution, the sampler state module, and parameter validation.
"""

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.dry_core import _dry_penalties
from vllm.v1.sample.dry_utils import max_exponent as _max_exponent

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
    # matching llama.cpp. The saturation happens in the float32 logit
    # store, so this has to go through the tensor path, not _dry_penalties.
    state, all_tokens = _make_v2_state(max_num_reqs=1, vocab=16)
    state.add_request(
        0,
        SamplingParams(
            dry_multiplier=1.0,
            dry_base=2.0,
            dry_allowed_length=0,
            dry_sequence_breakers=[],
        ),
    )
    all_tokens[0, :200] = 7
    logits = torch.zeros(1, 16, device=DEVICE)
    state.apply_dry(
        logits,
        np.array([0]),
        torch.tensor([199], device=DEVICE),
        expanded_logits=False,
    )
    assert logits[0, 7].item() == -float("inf")


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


class _StubTokenizer:
    """Minimal tokenizer for breaker-resolution tests."""

    _TEXTS = ["a", "\n", "b:\n", "hello", ":", "x:y", "*", "**"]

    vocab_size = len(_TEXTS)

    def batch_decode(self, ids_lists):
        return [self._TEXTS[ids[0]] for ids in ids_lists]


def test_breaker_containment_resolution():
    # Resolution must match every token whose text CONTAINS the breaker
    # string (llama.cpp get_overlapping_token_sequences), not just exact
    # encodings.
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


def test_double_pow_saturation():
    # llama.cpp's std::pow(float, int) computes in double: 0.8 * 2**128
    # is finite in float32 (-2.72e38) while 1.0 * 2**128 saturates to
    # -inf. A float32 pow gets the first case wrong.
    for mult, expect_inf in ((0.8, False), (1.0, True)):
        state, all_tokens = _make_v2_state(max_num_reqs=1, vocab=16)
        state.add_request(
            0,
            SamplingParams(
                dry_multiplier=mult,
                dry_base=2.0,
                dry_allowed_length=0,
                dry_sequence_breakers=[],
            ),
        )
        all_tokens[0, :200] = 7
        logits = torch.zeros(1, 16, device=DEVICE)
        state.apply_dry(
            logits,
            np.array([0]),
            torch.tensor([199], device=DEVICE),
            expanded_logits=False,
        )
        val = logits[0, 7].item()
        if expect_inf:
            assert val == -float("inf")
        else:
            assert math.isfinite(val) and math.isclose(
                val, -0.8 * 2.0**128, rel_tol=1e-6
            )


# ---------------------------------------------------------------------------
# V2-runner DryState module
# ---------------------------------------------------------------------------


def _make_v2_state(max_num_reqs=8, vocab=32, max_model_len=256, device=DEVICE):
    from vllm.v1.worker.gpu.sample.dry import DryState

    all_tokens = torch.zeros(
        max_num_reqs, max_model_len, dtype=torch.int32, device=device
    )
    req_states = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        vocab_size=vocab,
        device=device,
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
            # Exercise the breaker path in the batched core too: the
            # frontend resolves strings to ids, so set them directly.
            breakers = frozenset(rng.sample(range(4), rng.randint(0, 2)))
            params._dry_breaker_ids = sorted(breakers)
            state.add_request(r, params)
            n = rng.randint(2, 180)
            hist = [rng.randrange(rng.randint(1, 4) + 1) for _ in range(n)]
            all_tokens[r, :n] = torch.tensor(hist, dtype=torch.int32)
            histories[r] = (params, hist, breakers)
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
            params, hist, breakers = histories[r]
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
                        breakers,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_peak_memory_bounded():
    # base=1.1 gives max_exponent=930, a large J for the vectorized path
    # (the cap is allowed_length + max_exponent <= _J_BUDGET = 2048).
    # The chunk budget is denominated in bytes and the gather is int32,
    # so peak transient memory stays near _CHUNK_BYTE_BUDGET even at
    # large batch x window; an element-denominated budget with an int64
    # gather allocated ~8x more and OOMed 8 GB GPUs.
    device = torch.device("cuda")
    rng = random.Random(3)
    n_reqs, window = 32, 2048
    state, all_tokens = _make_v2_state(
        max_num_reqs=n_reqs, vocab=128256, max_model_len=window, device=device
    )
    params = SamplingParams(
        dry_multiplier=0.8,
        dry_base=1.1,
        dry_allowed_length=2,
        dry_sequence_breakers=[],
    )
    # A SMALL ALPHABET, deliberately. With randrange(1000) over 2048 positions a 2-token
    # repeat essentially never occurs, so nothing is ever charged, dry_core returns at
    # its ``if not charged`` guard, and this test measured only the match scan - never
    # the penalty application it exists to bound. Eight symbols guarantee charges, and
    # the assertion below fails if that ever stops being true rather than passing
    # vacuously again.
    hist = torch.tensor(
        [[rng.randrange(8) for _ in range(window)] for _ in range(n_reqs)],
        dtype=torch.int32,
    )
    all_tokens.copy_(hist.to(device))
    for r in range(n_reqs):
        state.add_request(r, params)
    logits = torch.zeros(n_reqs, 128256, device=device)
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    base_alloc = torch.accelerator.memory_allocated()
    state.apply_dry(
        logits,
        np.arange(n_reqs),
        torch.full((n_reqs,), window - 1, device=device),
        expanded_logits=False,
    )
    torch.accelerator.synchronize()
    peak = torch.accelerator.max_memory_allocated() - base_alloc
    # The penalty path must actually have run, or the bound below is measuring the scan
    # alone.
    charged = int((logits < 0).sum().item())
    assert charged > 0, (
        "no token was penalized, so this test did not reach the penalty path"
    )
    # A ceiling chosen to DISCRIMINATE, which two earlier versions did not. The
    # original was 2 * _CHUNK_BYTE_BUDGET (512 MiB); the second was 320 MiB,
    # picked against the old dense peak of 165 MiB and therefore still passing
    # the very regression it was tightened to catch. Measured on this
    # configuration: 76.2 MiB with the sparse penalty path (bit-stable across
    # runs), 165.3 MiB with the dense [R, vocab] formulation it replaced. 128
    # MiB sits above the first with 1.7x headroom and below the second, so a
    # return to dense fails here.
    limit = 128 * 1024 * 1024
    assert peak < limit, f"peak {peak / 2**20:.1f} MiB over {limit / 2**20:.0f} MiB"


def _validate_with_runner(params, *, use_v2_model_runner):
    """Drive InputProcessor._validate_params with a stub self.

    Only the attributes the DRY branch and the branches before it read are
    stubbed; params.verify() and the sampling-mask branch are no-ops here.
    """
    from vllm.v1.engine.input_processor import InputProcessor

    model_config = SimpleNamespace(
        return_sampling_mask=False, enable_trace_replay=False
    )
    stub = SimpleNamespace(
        model_config=model_config,
        speculative_config=None,
        structured_outputs_config=None,
        tokenizer=None,
        vllm_config=SimpleNamespace(
            reasoning_config=None,
            use_v2_model_runner=use_v2_model_runner,
        ),
    )
    params.verify = lambda *a, **k: None
    InputProcessor._validate_params(stub, params, ("generate",))


def test_dry_rejected_on_v1_model_runner():
    # DRY is implemented once, in the sampler state module the V2 model
    # runner drives. A request that asks for it on an engine that fell
    # back to the V1 runner must fail loudly rather than be ignored.
    params = SamplingParams(dry_multiplier=0.8)
    with pytest.raises(VLLMValidationError, match="Model Runner V2"):
        _validate_with_runner(params, use_v2_model_runner=False)


def test_dry_accepted_on_v2_model_runner():
    _validate_with_runner(SamplingParams(dry_multiplier=0.8), use_v2_model_runner=True)


def test_no_dry_is_not_rejected_on_v1_model_runner():
    # dry_multiplier=0 (the default) must not trip the V2 requirement.
    _validate_with_runner(SamplingParams(), use_v2_model_runner=False)


def test_update_from_tokenizer_resolves_breakers():
    # The engine frontend is the only place breaker strings become ids.
    # Resolving them directly is covered above; this pins the hook that
    # actually runs, and that the sampler state then consumes the result.
    params = SamplingParams(dry_multiplier=0.8, dry_sequence_breakers=["\n"])
    assert params._dry_breaker_ids is None
    params.update_from_tokenizer(_StubTokenizer())
    assert params._dry_breaker_ids == [1, 2]

    state, _ = _make_v2_state(max_num_reqs=1, vocab=8)
    state.add_request(0, params)
    assert state.breaker_ids[0] == [1, 2]


def test_update_from_tokenizer_skips_when_dry_is_off():
    # dry_multiplier=0 must not pay for the O(vocab) decode.
    params = SamplingParams(dry_sequence_breakers=["\n"])
    params.update_from_tokenizer(_StubTokenizer())
    assert params._dry_breaker_ids is None


# ---------------------------------------------------------------------------
# The REST boundary. Added 2026-09-13 after the Python-side bool guard was
# mistaken, in our own notes, for a guard that also covers HTTP.
# ---------------------------------------------------------------------------


def test_rest_numeric_fields_coerce_json_booleans_like_every_other_param():
    """Where the bool guard reaches, and where it does not.

    ``SamplingParams._verify_args`` rejects a bool in a ``dry_*`` numeric field
    (``test_sampling_params_rejects_json_booleans`` above), because ``bool`` is
    an ``int`` subclass and an isinstance check alone would let JSON ``true``
    through to the sampler.

    Over the REST API that guard is never reached. Pydantic coerces JSON
    ``true`` to ``1.0`` while validating the request model, so what arrives at
    ``SamplingParams`` is a genuine float and ``isinstance(x, bool)`` is False.
    ``{"dry_multiplier": true}`` is therefore a request for multiplier 1.0
    rather than an error.

    That is not specific to DRY: ``temperature`` and every other numeric
    sampling field in vLLM behave identically, which is why this test asserts
    the same coercion for an upstream field beside ours. It is pinned so that
    nobody reads the Python-side guard as a REST-side guard, and so that a
    future pydantic that starts rejecting bools fails here loudly rather than
    silently changing what an existing request means.
    """
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    req = CompletionRequest(
        model="m", prompt="p", dry_multiplier=True, temperature=True
    )
    # Coerced at the model boundary, before any vLLM validation runs.
    assert req.dry_multiplier == 1.0
    assert not isinstance(req.dry_multiplier, bool)
    assert req.temperature == 1.0, "upstream's own numeric field coerces the same way"

    params = req.to_sampling_params(max_tokens=8)
    assert params.dry_multiplier == 1.0
    # And it is a real enable, not a no-op: the sampler gate is multiplier > 0.
    from vllm.v1.worker.gpu.sample.dry import use_dry

    assert use_dry(params)


def test_rest_dry_fields_default_to_off():
    """A request that says nothing about DRY must produce DRY-off params.

    This is the property that matters to anyone who applies the patch to a
    running server and does not want the feature: the REST default has to be
    the disabled default, not merely a small multiplier.
    """
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    params = CompletionRequest(model="m", prompt="p").to_sampling_params(max_tokens=8)
    assert params.dry_multiplier == 0.0
    from vllm.v1.worker.gpu.sample.dry import use_dry

    assert not use_dry(params)


# ---------------------------------------------------------------------------
# Added 2026-09-13 after an adversarial review. Each of these pins a defect
# that review found, so that a later change cannot reintroduce it unnoticed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["dry_penalty_last_n", "dry_allowed_length"])
def test_dry_int_params_reject_values_that_would_kill_the_worker(field):
    """An out-of-range integer must be refused at the front door.

    Both fields are stored into an int64 numpy array in the worker. A value at
    or above 2**63 raises OverflowError there, inside ``execute_model``, which
    the engine core escalates to a fatal error and ``_send_engine_dead()`` - so
    before this bound existed, a single request could end the server. The cap
    is llama-server's INT32_MAX, which is also far past any useful context.
    """
    with pytest.raises(VLLMValidationError, match=field):
        SamplingParams(dry_multiplier=0.8, **{field: 2**63})
    with pytest.raises(VLLMValidationError, match=field):
        SamplingParams(dry_multiplier=0.8, **{field: 2**31})
    # The boundary itself is legal, and so are the ordinary values.
    SamplingParams(dry_multiplier=0.8, **{field: 2**31 - 1})
    SamplingParams(dry_multiplier=0.8, **{field: 2})


def test_dry_rejected_under_speculative_decoding():
    """DRY must be refused with a speculative config, not skipped in the sampler.

    The sampler's skip triggers on draft-expanded logits, which is false on any
    step where no request carries draft tokens - the step that finishes a
    prefill, among others. A request allowed through would therefore get DRY on
    some steps and not others, flickering with the schedule. Refusing is the
    honest behaviour and matches how min_p and logit_bias are handled.
    """
    spec = SimpleNamespace()  # only `is None` is tested by the validator
    with pytest.raises(VLLMValidationError, match="speculative"):
        SamplingParams(dry_multiplier=0.8)._validate_spec_decode(spec)
    # Without a speculative config, and with DRY off, nothing is raised.
    SamplingParams(dry_multiplier=0.8)._validate_spec_decode(None)
    SamplingParams()._validate_spec_decode(spec)


def test_dry_base_below_one_warns_that_it_disabled_dry():
    """`dry_base < 1.0` disables DRY, as in llama.cpp. Say so.

    Accepting a multiplier and then silently applying no penalty is the failure
    mode; the likeliest cause is `dry_base: 0.8` typed where
    `dry_multiplier: 0.8` was meant. Asserted against the module logger rather
    than caplog, because vLLM's logger does not propagate to pytest's handler.
    """
    from unittest.mock import patch

    from vllm.v1.worker.gpu.sample.dry import use_dry

    with patch("vllm.sampling_params.logger") as log:
        params = SamplingParams(dry_multiplier=0.8, dry_base=0.8)
    assert not use_dry(params), "base < 1.0 must disable DRY"
    assert log.warning.called, "the disabling must be reported, not silent"
    assert "disables DRY" in log.warning.call_args[0][0]

    # A normal configuration says nothing.
    with patch("vllm.sampling_params.logger") as log:
        params = SamplingParams(dry_multiplier=0.8, dry_base=1.75)
    assert use_dry(params)
    assert not log.warning.called
