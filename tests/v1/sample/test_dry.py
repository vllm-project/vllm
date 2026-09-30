# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DRY (Don't Repeat Yourself) repetition penalty."""

import math
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.sample.logits_processor.loader as loader
from examples.features.logits_processor import dry
from examples.features.logits_processor.dry import (
    _MAX_CACHED_BREAKER_MASKS,
    DEFAULT_DRY_SEQUENCE_BREAKERS,
    DryState,
    _dry_penalties,
    use_dry,
)
from examples.features.logits_processor.dry import max_exponent as _max_exponent
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams

DEVICE = torch.device("cpu")
DRY_FQCN = "examples.features.logits_processor.dry:DryState"


def _params(**dry_args) -> SamplingParams:
    """SamplingParams carrying DRY arguments the way a client sends them."""
    return SamplingParams(extra_args=dry_args or None)


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
    # llama.cpp's commented example (src/llama-sampler.cpp): tokens
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
    # matching llama.cpp. Only the tensor path saturates, on the cast into
    # dry_core's float32 accumulator.
    state, all_tokens = _make_v2_state(max_num_reqs=1, vocab=16)
    state.add_request(
        0,
        _params(
            dry_multiplier=1.0,
            dry_base=2.0,
            dry_allowed_length=0,
            dry_sequence_breakers=[],
        ),
    )
    all_tokens[0, :200] = 7
    logits = torch.zeros(1, 16, device=DEVICE)
    _apply(state, logits, np.array([0]), np.array([200]))
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
    # Resolution must match every token whose text contains the breaker
    # string (llama.cpp get_overlapping_token_sequences), not only exact
    # encodings.
    from examples.features.logits_processor.dry import resolve_dry_breakers

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


def test_validate_params_rejects_json_booleans():
    for name in (
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_penalty_last_n",
    ):
        with pytest.raises(ValueError, match=name):
            DryState.validate_params(_params(**{name: True}))


def test_validate_params():
    for bad in (
        {"dry_multiplier": -1.0},
        {"dry_multiplier": float("nan")},
        {"dry_penalty_last_n": -7},
        {"dry_penalty_last_n": 2.5},
        {"dry_allowed_length": -1},
        {"dry_allowed_length": 1.5},
        {"dry_sequence_breakers": [1, 2]},
        {"dry_sequence_breakers": [f"b{i}" for i in range(65)]},
    ):
        with pytest.raises(ValueError):
            DryState.validate_params(_params(**bad))
    # Valid corner values pass, as does a request that says nothing at all.
    DryState.validate_params(SamplingParams())
    DryState.validate_params(_params())
    DryState.validate_params(_params(dry_multiplier=0.0))
    DryState.validate_params(_params(dry_multiplier=1.0, dry_penalty_last_n=-1))
    DryState.validate_params(_params(dry_multiplier=1.0, dry_sequence_breakers=[]))
    # An explicit null means "unset", as it does for every SamplingParams
    # field; it must not be read as a non-numeric value and rejected.
    DryState.validate_params(_params(dry_multiplier=None, dry_base=None))


def test_validate_params_rejects_unknown_dry_keys():
    """An unknown dry_* key fails the request instead of being ignored."""
    with pytest.raises(ValueError, match="dry_multipliers"):
        DryState.validate_params(_params(dry_multiplier=0.8, dry_multipliers=0.8))
    # Keys outside the namespace belong to somebody else; leave them alone.
    DryState.validate_params(_params(dry_multiplier=0.8, target_token=3))


def test_validate_params_raises_plain_valueerror():
    """validate_params raises ``ValueError``, as the V2 interface specifies."""
    with pytest.raises(ValueError) as excinfo:
        DryState.validate_params(_params(dry_multiplier=-1.0))
    assert type(excinfo.value) is ValueError


def _oracle(window, breakers, multiplier, base, allowed_length, max_exponent):
    """llama_sampler_dry_apply's steps, with a brute-force match scan in place of
    the Z-algorithm that ``_dry_penalties`` ports."""
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
            _params(
                dry_multiplier=mult,
                dry_base=2.0,
                dry_allowed_length=0,
                dry_sequence_breakers=[],
            ),
        )
        all_tokens[0, :200] = 7
        logits = torch.zeros(1, 16, device=DEVICE)
        _apply(state, logits, np.array([0]), np.array([200]))
        val = logits[0, 7].item()
        if expect_inf:
            assert val == -float("inf")
        else:
            assert math.isfinite(val) and math.isclose(
                val, -0.8 * 2.0**128, rel_tol=1e-6
            )


def test_dry_core_rejects_j_budget_over_the_budget():
    """dry_core raises on a j_budget over _J_BUDGET and accepts _J_BUDGET itself."""

    def run(j_budget):
        def col(value, dtype=torch.int64):
            return torch.full((1,), value, dtype=dtype, device=DEVICE)

        dry.dry_core(
            torch.zeros(1, 8, device=DEVICE),
            row_idx=col(0),
            W=torch.zeros(1, 4, dtype=torch.int64, device=DEVICE),
            n_r=col(4),
            allowed=col(2),
            max_exp=col(1),
            mult=col(1.0, torch.float32),
            base=col(2.0, torch.float32),
            breaker_masks=[None],
            j_budget=j_budget,
        )

    run(dry._J_BUDGET)
    with pytest.raises(ValueError, match="j_budget"):
        run(dry._J_BUDGET + 1)


# ---------------------------------------------------------------------------
# The DryState custom logits processor
# ---------------------------------------------------------------------------


def _make_vllm_config(speculative_config=None, skip_tokenizer_init=True):
    """Stub VllmConfig with only the fields DryState reads."""
    return SimpleNamespace(
        speculative_config=speculative_config,
        model_config=SimpleNamespace(skip_tokenizer_init=skip_tokenizer_init),
    )


def _make_v2_state(
    max_num_reqs=8,
    vocab=32,
    max_model_len=256,
    device=DEVICE,
    tokenizer=None,
):
    all_tokens = torch.zeros(
        max_num_reqs, max_model_len, dtype=torch.int32, device=device
    )
    req_states = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        vocab_size=vocab,
        device=device,
        all_token_ids=SimpleNamespace(gpu=all_tokens),
    )
    state = DryState(_make_vllm_config(), req_states)
    if tokenizer is not None:
        # Re-resolve the default set now that a tokenizer is installed.
        state._tokenizer = tokenizer
        state._default_breaker_ids = state._resolve_breakers(
            DEFAULT_DRY_SEQUENCE_BREAKERS
        )
    return state, all_tokens


def _apply(state, logits, idx_mapping_np, seq_lens_np):
    """Call ``DryState.apply`` with a LogitsContext built as the sampler builds it.

    Args:
      state: the ``DryState`` under test.
      logits: [num_reqs, vocab] tensor, one row per request, modified in place.
      idx_mapping_np: [num_reqs] batch position -> request slot.
      seq_lens_np: [num_reqs] host-side context lengths.

    """
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext

    n_reqs = idx_mapping_np.shape[0]
    assert logits.shape[0] == n_reqs
    idx_mapping = torch.from_numpy(idx_mapping_np).to(logits.device)
    ctx = LogitsContext(
        expanded_idx_mapping=idx_mapping,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping_np,
        expanded_local_pos=torch.zeros(n_reqs, dtype=torch.int64, device=logits.device),
        input_ids=torch.zeros(n_reqs, dtype=torch.int32, device=logits.device),
        pos=torch.from_numpy(seq_lens_np.astype("int64") - 1).to(logits.device),
        seq_lens_upper_bound_np=seq_lens_np,
    )
    return state.apply(logits, ctx)


def test_v2_worked_example_and_window_bounds():
    # The window is [0, seq_len), including the last input token. Without it every
    # penalty lands on the continuation of the previous suffix instead of the one
    # being sampled.
    state, all_tokens = _make_v2_state()
    params = _params(
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
    seq_lens = np.array([len(seq)])
    _apply(state, logits, idx_mapping, seq_lens)
    assert logits[0, 3].item() == -4.0
    assert logits[0, 2].item() == -1.0
    assert logits[0, 9].item() == -2.0


def test_v2_disabled_and_gating():
    state, _ = _make_v2_state()
    assert not state.add_request(0, SamplingParams())  # dry off by default
    assert not state.add_request(1, _params(dry_multiplier=1.0, dry_base=0.5))
    assert not state.add_request(2, _params(dry_multiplier=1.0, dry_penalty_last_n=0))
    assert not state.use_dry[:3].any()
    # Re-adding a slot with DRY off after one with DRY on must clear it.
    on = _params(dry_multiplier=0.8, dry_sequence_breakers=[])
    assert state.add_request(4, on)
    assert state.use_dry[4]
    assert not state.add_request(4, SamplingParams())
    assert not state.use_dry[4]


def test_v2_penalty_last_n_window():
    # A cap smaller than the history must limit the visible window.
    state, all_tokens = _make_v2_state()
    params = _params(
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
    _apply(state, logits, np.array([0]), np.array([20]))
    # window = last 4 tokens of an identical run: longest match ending
    # before the suffix is 3 (rep-limited by the window), penalty 2**(3-1).
    assert logits[0, 7].item() == -4.0


def test_spec_decode_is_refused_at_construction():
    """A speculative config makes DRY refuse to load."""
    req_states = SimpleNamespace(
        max_num_reqs=4,
        vocab_size=32,
        device=DEVICE,
        all_token_ids=SimpleNamespace(gpu=torch.zeros(4, 8, dtype=torch.int32)),
    )
    config = _make_vllm_config(speculative_config=SimpleNamespace())
    with pytest.raises(ValueError, match="speculative decoding"):
        DryState(config, req_states)


def test_v2_draft_expanded_logits_raise():
    """``DryState.apply`` raises on more logits rows than requests.

    Only speculative decoding lays logits out that way, and construction refuses it.
    """
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext

    state, _ = _make_v2_state()
    state.add_request(0, _params(dry_multiplier=0.8, dry_sequence_breakers=[]))
    # One request with one draft token has two logits rows.
    rows = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    ctx = LogitsContext(
        expanded_idx_mapping=rows,
        idx_mapping=rows[:1],
        idx_mapping_np=np.array([0]),
        expanded_local_pos=torch.arange(2, device=DEVICE),
        input_ids=torch.zeros(2, dtype=torch.int32, device=DEVICE),
        pos=torch.tensor([7, 8], device=DEVICE),
        seq_lens_upper_bound_np=np.array([9]),
    )
    logits = torch.zeros(2, 32, device=DEVICE)
    with pytest.raises(RuntimeError, match="draft-expanded"):
        state.apply(logits, ctx)
    assert not logits.any()


def test_constructs_without_speculative_config():
    state, _ = _make_v2_state()
    assert isinstance(state, DryState)
    assert state._tokenizer is None  # skip_tokenizer_init in the stub config


@pytest.mark.parametrize("chunk_budget", [4096, dry._CHUNK_BYTE_BUDGET])
def test_v2_matches_reference_fuzz(chunk_budget, monkeypatch):
    # The V2 window-gather and routing path must agree with the sequential
    # reference on randomized histories, including degenerate bases that
    # route through the slow path. At these sizes the default budget is always one
    # chunk. The 4096 B case splits most calls into several, to cover how per-chunk
    # results combine.
    monkeypatch.setattr(dry, "_CHUNK_BYTE_BUDGET", chunk_budget)
    state, all_tokens = _make_v2_state(max_num_reqs=8, vocab=16, max_model_len=200)
    rng = random.Random(7)
    for trial in range(60):
        n_reqs = rng.randint(1, 5)
        histories = {}
        seq_lens = []
        idx_list = []
        for r in range(n_reqs):
            args: dict[str, Any] = dict(
                dry_multiplier=rng.choice([0.8, 1.0, 2.0]),
                dry_base=rng.choice([1.0000005, 1.1, 1.75, 2.0]),
                dry_allowed_length=rng.randint(0, 3),
                dry_penalty_last_n=rng.choice([-1, 5, 50]),
                dry_sequence_breakers=[],
            )
            state.add_request(r, _params(**args))
            # Breaker ids go straight into the state and must follow add_request,
            # which pops the slot's ids. String resolution is tested above.
            breakers = frozenset(rng.sample(range(4), rng.randint(0, 2)))
            if breakers:
                state.breaker_ids[r] = breakers
            n = rng.randint(2, 180)
            hist = [rng.randrange(rng.randint(1, 4) + 1) for _ in range(n)]
            all_tokens[r, :n] = torch.tensor(hist, dtype=torch.int32)
            histories[r] = (args, hist, breakers)
            idx_list.append(r)
            seq_lens.append(n)

        logits = torch.zeros(n_reqs, 16, device=DEVICE)
        _apply(state, logits, np.array(idx_list), np.array(seq_lens))
        for r in range(n_reqs):
            args, hist, breakers = histories[r]
            base32 = float(np.float32(args["dry_base"]))
            if base32 < 1.0 or not args["dry_multiplier"]:
                want: dict[int, float] = {}
            else:
                last_n = args["dry_penalty_last_n"]
                w = len(hist) if last_n == -1 else min(len(hist), last_n)
                allowed = args["dry_allowed_length"]
                window = hist[-w:] if w > allowed else None
                want = (
                    _dry_penalties(
                        window,
                        breakers,
                        float(np.float32(args["dry_multiplier"])),
                        base32,
                        allowed,
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


def test_v2_batch_sharded_sampling_matches_unsharded(monkeypatch):
    """Each rank's DRY rows under batch-sharded sampling match the unsharded rows.

    With ``--enable-batch-sharded-sampling``, ``BatchSharder`` gives every TP rank
    a local batch of the requests it owns and localizes the two fields DRY reads
    from ``LogitsContext``: ``idx_mapping_np``, and ``seq_lens_cpu_upper_bound``,
    which the sampler passes as ``seq_lens_upper_bound_np``. Its Triton plan kernel
    writes only device fields DRY does not read, so it is replaced here and the
    sharding runs on the host.
    """
    from dataclasses import replace
    from unittest import mock

    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.sample import batch_shard

    vocab = 16
    state, all_tokens = _make_v2_state(
        max_num_reqs=8, vocab=vocab, tokenizer=_StubTokenizer()
    )
    # Slots out of batch order, so a rank's rows are not its slots, and only rank 2 at
    # tp_size 3 owns a prefix of the batch. Slot 3 has DRY off, slot 6 takes the
    # sequential path and slot 0 caps its window. Slot 5 breaks on "\n" (tokens 1
    # and 2), slot 6 on "a" (token 0), and slots 7 and 1 on "b" (token 2).
    slots = np.array([5, 2, 7, 0, 3, 6, 1])
    seq_lens = np.array([40, 23, 57, 31, 12, 48, 36])
    breakers = {5: ["\n"], 6: ["a"], 7: ["b"], 1: ["b"]}
    rng = random.Random(5)
    for slot, n in zip(slots.tolist(), seq_lens.tolist()):
        args: dict[str, Any] = dict(
            dry_multiplier=0.8, dry_sequence_breakers=breakers.get(slot, [])
        )
        if slot == 6:
            args["dry_base"] = 1.0000005
        if slot == 0:
            args["dry_penalty_last_n"] = 16
        state.add_request(slot, SamplingParams() if slot == 3 else _params(**args))
        hist = [rng.randrange(3) for _ in range(n)]
        all_tokens[slot, :n] = torch.tensor(hist, dtype=torch.int32)
    assert state.breaker_ids == {5: {1, 2}, 6: {0}, 7: {2}, 1: {2}}
    full = torch.zeros(len(slots), vocab, device=DEVICE)
    _apply(state, full, slots, seq_lens)
    assert (full < 0).any(dim=1).tolist() == [s != 3 for s in slots.tolist()]

    buffers = InputBuffers(max_num_reqs=8, max_num_tokens=len(slots), device=DEVICE)
    batch = replace(
        InputBatch.make_dummy(len(slots), len(slots), buffers),
        req_ids=[f"req-{i}" for i in range(len(slots))],
        idx_mapping=torch.from_numpy(slots).to(DEVICE),
        idx_mapping_np=slots,
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens),
    )
    monkeypatch.setattr(batch_shard, "_build_shard_plan_kernel", mock.MagicMock())
    for tp_size in (2, 3):
        covered: list[int] = []
        for rank in range(tp_size):
            group = SimpleNamespace(rank_in_group=rank, world_size=tp_size)
            with mock.patch.object(batch_shard, "get_tp_group", return_value=group):
                sharder = batch_shard.BatchSharder(
                    max_num_reqs=8, max_num_logits_per_req=1, device=DEVICE
                )
            local, _, _, _ = sharder.shard_sampler_inputs(batch, None)
            owned = [int(req_id.split("-")[1]) for req_id in local.req_ids]
            covered += owned
            logits = torch.zeros(len(owned), vocab, device=DEVICE)
            seq_lens_np = local.seq_lens_cpu_upper_bound.numpy()
            _apply(state, logits, local.idx_mapping_np, seq_lens_np)
            assert torch.equal(logits, full[owned]), f"tp_size {tp_size} rank {rank}"
        assert sorted(covered) == list(range(len(slots)))


def test_v2_breaker_strings_resolve_and_reach_dry_core():
    # Breaker strings arrive in extra_args and are resolved by the processor's
    # own tokenizer; the ids must reach the penalty computation, and the slot
    # must not inherit them when it is reused.
    state, all_tokens = _make_v2_state(tokenizer=_StubTokenizer())
    breaking = _params(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=1,
        dry_sequence_breakers=["*"],
    )
    state.add_request(0, breaking)
    # "*" is token 6 and is contained in token 7 ("**"), so both break.
    assert state.breaker_ids[0] == frozenset({6, 7})
    all_tokens[0, :8] = 7
    logits = torch.zeros(1, 32, device=DEVICE)
    _apply(state, logits, np.array([0]), np.array([8]))
    assert logits[0, 7].item() == 0.0

    fresh = _params(
        dry_multiplier=1.0,
        dry_base=2.0,
        dry_allowed_length=1,
        dry_sequence_breakers=[],
    )
    state.add_request(0, fresh)
    assert 0 not in state.breaker_ids
    logits2 = torch.zeros(1, 32, device=DEVICE)
    _apply(state, logits2, np.array([0]), np.array([8]))
    assert logits2[0, 7].item() < 0.0


def test_breaker_mask_cache_is_bounded():
    """The device-mask cache must not grow with the number of distinct sets."""
    state, _ = _make_v2_state(vocab=128)
    sets = [frozenset({i}) for i in range(_MAX_CACHED_BREAKER_MASKS + 8)]
    for breakers in sets:
        state.breaker_ids[0] = breakers
        assert state._breaker_mask(0) is not None
        assert len(state._breaker_masks) <= _MAX_CACHED_BREAKER_MASKS
    # The first set was evicted and the newest kept.
    assert sets[-1] in state._breaker_masks
    assert sets[0] not in state._breaker_masks


def test_default_breaker_set_is_resolved_eagerly():
    """A request that names no breakers must not pay to resolve them."""
    state, _ = _make_v2_state(tokenizer=_StubTokenizer())
    # "\n" -> {1, 2}, ":" -> {2, 4, 5}, '"' -> {}, "*" -> {6, 7}.
    assert state._default_breaker_ids == frozenset({1, 2, 4, 5, 6, 7})
    state.add_request(0, _params(dry_multiplier=0.8))
    assert state.breaker_ids[0] is state._default_breaker_ids
    # Naming the default set explicitly must land on the same object.
    state.add_request(
        1,
        _params(
            dry_multiplier=0.8,
            dry_sequence_breakers=list(DEFAULT_DRY_SEQUENCE_BREAKERS),
        ),
    )
    assert state.breaker_ids[1] is state._default_breaker_ids


def test_init_warms_up_dry_core_once(monkeypatch):
    """__init__ calls dry_core once, the way apply calls it."""
    calls = []
    # Taken before the patch below replaces it in the same module.
    original = dry.dry_core

    def spy(logits, **kwargs):
        calls.append((logits, kwargs))
        return original(logits, **kwargs)

    def kernel_args(logits, kwargs):
        def sig(t):
            return None if t is None else (t.dtype, t.dim(), t.device)

        masks = {sig(m) for m in kwargs["breaker_masks"]}
        other = {
            name: sig(arg) if isinstance(arg, torch.Tensor) else type(arg)
            for name, arg in kwargs.items()
            if name != "breaker_masks"
        }
        return sig(logits), masks, other

    monkeypatch.setattr("examples.features.logits_processor.dry.dry_core", spy)
    _make_v2_state(vocab=1)
    assert not calls, "a vocabulary of one leaves no id for the breaker"
    state, all_tokens = _make_v2_state(vocab=64, tokenizer=_StubTokenizer())
    assert len(calls) == 1
    assert (calls[0][0] < 0).any(), "the warmup matched nothing"
    assert not state._breaker_masks
    assert not state.breaker_ids
    assert not state.use_dry.any()

    state.add_request(0, _params(dry_multiplier=0.8))
    all_tokens[0, :8] = 3
    _apply(state, torch.zeros(1, 64, device=DEVICE), np.array([0]), np.array([8]))
    assert len(calls) == 2
    assert kernel_args(*calls[0]) == kernel_args(*calls[1])


def test_no_tokenizer_warns_once_and_still_penalizes():
    # Without a tokenizer, breaker strings are dropped with one warning.
    state, all_tokens = _make_v2_state()
    assert state._tokenizer is None
    assert state._default_breaker_ids == frozenset()
    state.add_request(0, _params(dry_multiplier=0.8))
    assert state._warned_unresolved
    assert 0 not in state.breaker_ids
    all_tokens[0, :8] = 7
    logits = torch.zeros(1, 32, device=DEVICE)
    _apply(state, logits, np.array([0]), np.array([8]))
    assert logits[0, 7].item() < 0.0


def test_v2_degenerate_base_routes_to_unclamped_slow_path():
    # base <= 1.000001 gives max_exponent == 0, which in the sequential
    # reference means "no clamp"; the vectorized path would clamp the
    # exponent to 0, so such requests must route to the reference scan.
    state, all_tokens = _make_v2_state()
    params = _params(
        dry_multiplier=1.0,
        dry_base=1.0000005,
        dry_allowed_length=0,
        dry_sequence_breakers=[],
    )
    state.add_request(0, params)
    all_tokens[0, :150] = 7
    logits = torch.zeros(1, 32, device=DEVICE)
    _apply(state, logits, np.array([0]), np.array([150]))
    base32 = float(np.float32(1.0000005))
    want = -1.0 * base32**149
    got = logits[0, 7].item()
    assert math.isclose(got, want, rel_tol=1e-6)
    assert got != -1.0


def test_breaker_resolution_covers_added_tokens():
    class AddedTokenTokenizer:
        _TEXTS = ["a", "\n", "b", "<|im_start|>"]
        vocab_size = 3  # the added token sits above vocab_size
        max_token_id = 3

        def batch_decode(self, ids_lists):
            return [self._TEXTS[ids[0]] for ids in ids_lists]

    from examples.features.logits_processor.dry import resolve_dry_breakers

    tok = AddedTokenTokenizer()
    assert resolve_dry_breakers(tok, ("<|im_start|>",)) == [3]
    assert resolve_dry_breakers(tok, ("\n",)) == [1]


class _TextsTokenizer:
    """Tokenizer stub over an explicit list of decoded token texts."""

    def __init__(self, texts: list[str], vocab_size: int):
        self._texts = texts
        self.vocab_size = vocab_size
        self.max_token_id = len(texts) - 1

    def batch_decode(self, ids_lists):
        return [self._texts[ids[0]] for ids in ids_lists]


def _containment_scan(texts: list[str], breaker_strs: tuple[str, ...]) -> list[int]:
    """Resolution as a plain scan: the semantics the character index must keep."""
    ids: set[int] = set()
    for s in breaker_strs:
        ids.update(i for i, text in enumerate(texts) if s in text)
    return sorted(ids)


def test_breaker_index_matches_containment_scan():
    """Breaker resolution must match a plain containment scan.

    Every character of a synthetic vocabulary is checked, including multi-byte
    and combining ones, added tokens and an empty token, then multi-character
    and mixed breaker sets.
    """
    from examples.features.logits_processor.dry import (
        _MAX_BREAKER_CHAR_LEN,
        _vocab_index,
        resolve_dry_breakers,
    )

    rng = random.Random(11)
    # U+4E2D and U+1F600 are multi-byte in UTF-8. "e\u0301" is two code points, so
    # U+0301 alone resolves by lookup and the pair by scan.
    alphabet = list('abcde \n:"*.,') + ["\u4e2d", "\u00e9", "e\u0301", "\U0001f600"]
    texts = [""]  # a token that decodes to nothing
    for _ in range(8000):
        texts.append("".join(rng.choices(alphabet, k=rng.randint(1, 12))))
    vocab_size = len(texts)
    # Added tokens, above vocab_size; one holds the vocabulary's only U+2028.
    texts += ["<|im_start|>", "<|im_end|>", "\u2028sep"]
    tok = _TextsTokenizer(texts, vocab_size)

    def check(breakers: tuple[str, ...]) -> list[int]:
        kept = tuple(s[:_MAX_BREAKER_CHAR_LEN] for s in breakers if s)
        want = _containment_scan(texts, kept)
        assert resolve_dry_breakers(tok, breakers) == want, breakers
        return want

    # Every character present is indexed, so a lookup miss means "in no token".
    present = sorted(set("".join(texts)))
    assert set(_vocab_index(tok).char_ids) == set(present)
    for char in present:
        check((char,))
    assert len(check(("a",))) > 2000  # a character in thousands of tokens
    assert check(("\u2028",)) == [len(texts) - 1]  # only above vocab_size
    for absent in ("q", "Z", "\u2603", "\U0001f601", "\u0300"):
        assert check((absent,)) == []

    substrings = []
    for _ in range(300):
        text = rng.choice(texts[1:vocab_size])
        start = rng.randrange(len(text))
        substrings.append(text[start : start + rng.randint(2, 5)])
    for s in substrings + ["<|im_start|>", "e\u0301", "\u2028sep", "zz", "a" * 41]:
        check((s,))
    for _ in range(200):
        check(tuple(rng.sample(present, rng.randint(1, 4)) + rng.sample(substrings, 2)))
    assert check(()) == []
    assert check(("",)) == []
    assert check(("", "\n")) == check(("\n",))


# The accumulator scales with R and the chunk transients do not, hence two batch
# sizes. The breaker path's masks get their own case.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("n_reqs", "limit_mib", "n_breakers"),
    [(32, 128, 0), (256, 256, 0), (256, 288, 3900)],
)
def test_peak_memory_bounded(n_reqs, limit_mib, n_breakers):
    # base=1.1 gives max_exponent=930, a large J for the vectorized path
    # (the cap is allowed_length + max_exponent <= _J_BUDGET = 2048).
    device = torch.device("cuda")
    rng = random.Random(3)
    window = 2048
    state, all_tokens = _make_v2_state(
        max_num_reqs=n_reqs, vocab=128256, max_model_len=window, device=device
    )
    params = _params(
        dry_multiplier=0.8,
        dry_base=1.1,
        dry_allowed_length=2,
        dry_sequence_breakers=[],
    )
    breaker_ids: frozenset[int] = frozenset()
    if n_breakers:
        # About the size of llama.cpp's default set on a Llama-3 tokenizer. Id 0 is
        # in the history, so breakers cut some matches but not all.
        breaker_ids = frozenset({0} | set(rng.sample(range(8, 128256), n_breakers - 1)))
    # Eight symbols, so 2-token repeats are common.
    hist = torch.tensor(
        [[rng.randrange(8) for _ in range(window)] for _ in range(n_reqs)],
        dtype=torch.int32,
    )
    all_tokens.copy_(hist.to(device))
    for r in range(n_reqs):
        state.add_request(r, params)
        if breaker_ids:
            state.breaker_ids[r] = breaker_ids
    logits = torch.zeros(n_reqs, 128256, device=device)
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    base_alloc = torch.accelerator.memory_allocated()
    _apply(state, logits, np.arange(n_reqs), np.full(n_reqs, window))
    torch.accelerator.synchronize()
    peak = torch.accelerator.max_memory_allocated() - base_alloc
    charged = int((logits < 0).sum().item())
    assert charged > 0, "no token was penalized"
    # The limits catch the exponent and pow being held at full [R, vocab] width.
    limit = limit_mib * 1024 * 1024
    assert peak < limit, f"peak {peak / 2**20:.1f} MiB over {limit / 2**20:.0f} MiB"


# ---------------------------------------------------------------------------
# Plugin wiring: loading by FQCN, and validation at request admission.
# ---------------------------------------------------------------------------


def test_loads_through_the_loader_by_fqcn():
    """The example's FQCN must resolve and construct."""
    req_states = SimpleNamespace(
        device=DEVICE,
        max_num_reqs=4,
        vocab_size=32,
        all_token_ids=SimpleNamespace(gpu=torch.zeros(4, 8, dtype=torch.int32)),
        prompt_len=None,
        prefill_len=None,
        total_len=None,
    )
    procs = loader.build_custom_logits_processors(
        _make_vllm_config(), req_states, False, [DRY_FQCN]
    )
    assert [type(p) for p in procs] == [DryState]


def test_loader_import_does_not_initialize_cuda():
    """Importing dry.py must not initialize CUDA.

    The frontend imports it under ``guard_cuda_initialization()`` to validate
    params, and a device touched at import time fails the engine at startup.
    """
    code = (
        "import torch\n"
        "from vllm.utils.torch_utils import guard_cuda_initialization\n"
        "with guard_cuda_initialization():\n"
        "    import examples.features.logits_processor.dry  # noqa: F401\n"
        "assert not torch.cuda.is_initialized()\n"
    )
    # From the repository root, where `examples` is importable.
    subprocess.run(
        [sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[3]
    )


def test_admission_validator_rejects_bad_extra_args():
    """The real frontend validator must turn a bad dry_* value into a 400.

    ``build_custom_logits_processors_params_validator`` is what
    ``InputProcessor`` calls per request; it wraps a processor's ``ValueError``
    as ``VLLMValidationError``, which the entrypoints map to a client error.
    """
    loader._cached_load_v2_logitsprocs.cache_clear()
    validate = loader.build_custom_logits_processors_params_validator([DRY_FQCN])
    validate(SamplingParams())
    validate(_params(dry_multiplier=0.8))
    with pytest.raises(VLLMValidationError, match="dry_multiplier"):
        validate(_params(dry_multiplier=-1.0))
    with pytest.raises(VLLMValidationError, match="dry_multipliers"):
        validate(_params(dry_multiplier=0.8, dry_multipliers=0.8))


@pytest.mark.parametrize("field", ["dry_penalty_last_n", "dry_allowed_length"])
def test_dry_int_params_upper_bound(field):
    """Both integer fields are capped at INT32_MAX; see ``_DRY_INT_MAX``."""
    for bad in (2**63, 2**31):
        with pytest.raises(ValueError, match=field):
            DryState.validate_params(_params(dry_multiplier=0.8, **{field: bad}))
    # The boundary itself is legal, and so are the ordinary values.
    DryState.validate_params(_params(dry_multiplier=0.8, **{field: 2**31 - 1}))
    DryState.validate_params(_params(dry_multiplier=0.8, **{field: 2}))


def test_dry_base_below_one_warns_that_it_disabled_dry():
    """`dry_base < 1.0` disables DRY, as in llama.cpp; validate_params warns.

    Patches the module logger because vLLM's logger does not propagate to caplog.
    """
    from unittest.mock import patch

    with patch("examples.features.logits_processor.dry.logger") as log:
        DryState.validate_params(_params(dry_multiplier=0.8, dry_base=0.8))
    assert not use_dry(0.8, 0.8, -1), "base < 1.0 must disable DRY"
    assert log.warning.called, "expected a warning"
    assert "disables DRY" in log.warning.call_args[0][0]

    # A normal configuration says nothing.
    with patch("examples.features.logits_processor.dry.logger") as log:
        DryState.validate_params(_params(dry_multiplier=0.8, dry_base=1.75))
    assert use_dry(0.8, 1.75, -1)
    assert not log.warning.called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_result_does_not_depend_on_the_chunk_budget():
    """Logits must not depend on how the offsets are chunked."""
    from examples.features.logits_processor import dry as dc

    device = torch.device("cuda")
    vocab, window, n_reqs = 64, 600, 1
    # A short alphabet repeated, so one follower token recurs at offsets far
    # enough apart to land in different chunks.
    W = torch.tensor([([5, 6, 7] * 200)[:window]], dtype=torch.int64, device=device)
    col = lambda v, dt=torch.int64: torch.full((n_reqs,), v, dtype=dt, device=device)  # noqa: E731

    def run(budget):
        saved = dc._CHUNK_BYTE_BUDGET
        dc._CHUNK_BYTE_BUDGET = budget
        try:
            logits = torch.zeros(n_reqs, vocab, device=device)
            dc.dry_core(
                logits,
                torch.arange(n_reqs, device=device),
                W,
                col(window),
                col(2),
                col(60),
                col(0.8, torch.float32),
                col(1.75, torch.float32),
                [None] * n_reqs,
                j_budget=62,
            )
            return logits
        finally:
            dc._CHUNK_BYTE_BUDGET = saved

    many = run(4096)
    one = run(1 << 30)
    assert torch.equal(many, one), (
        f"chunking changed the result: {many[0][many[0] < 0][:3].tolist()} vs "
        f"{one[0][one[0] < 0][:3].tolist()}"
    )
    assert (many < 0).any(), "nothing was penalized"
