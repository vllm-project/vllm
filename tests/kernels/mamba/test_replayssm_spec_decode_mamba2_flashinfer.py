# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer ReplaySSM speculative verify step vs a serial SSU reference.

The oracle is ``selective_state_update`` stepped one token at a time from the
checkpoint: through the ``pnat`` replayed history tokens, then through the
verify window, capturing the output at each window position. That is the
definition of what ``checkpointing_ssu`` computes in one launch.

The Triton ReplaySSM kernel is deliberately not imported here -- it has its own
non-regression suite in ``test_replayssm_spec_decode_mamba2.py`` and a different
ring layout (token-major, power-of-two, raw dt).

Ring contract under test:
  * physical ring is exactly ``B + T`` rows, head-major;
  * the kernel replays ``[ring_start, ring_start + pnat)`` and appends the new
    tokens at ``(ring_start + pnat + j) % R``;
  * it checkpoints iff ``pnat + seq_len > B``, folding the replayed history into
    ``state`` and leaving the caller to advance ``ring_start``;
  * ``dt_cache`` holds *processed* dt (bias + softplus applied), unlike the
    Triton ring which caches raw dt.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import set_random_seed

DEV = "cuda"

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA device"),
    pytest.mark.skipif(
        not has_flashinfer(), reason="Need flashinfer with the ring checkpointing_ssu"
    ),
]

# (nheads, head_dim, dstate, ngroups)
_TINY = (4, 64, 128, 1)
_SMALL = (8, 64, 128, 2)
# Nemotron-H at TP1/TP4-ish head counts.
_REAL = (128, 64, 128, 8)
_REAL_TP4 = (32, 64, 128, 2)


def _make_backend(algorithm: str = "monolith", stochastic_rounding: bool = False):
    """Build the adapter without running its FlashInfer availability probe.

    Bypassing __init__ means every field the call path reads has to be set here;
    going through a single helper keeps the three fixtures from drifting as the
    adapter gains state.
    """
    from flashinfer.mamba import checkpointing_ssu

    from vllm.model_executor.layers.mamba.ops.replayssm_spec_flashinfer import (
        ReplaySSMSpecFlashInferBackend,
        _resolve_rounding_policy,
    )

    backend = ReplaySSMSpecFlashInferBackend.__new__(ReplaySSMSpecFlashInferBackend)
    backend._kernel = checkpointing_ssu
    backend.algorithm = algorithm
    backend.rounding = _resolve_rounding_policy(
        SimpleNamespace(
            enable_stochastic_rounding=stochastic_rounding,
            stochastic_rounding_philox_rounds=0,
        )
    )
    return backend


def _tolerances(act_dtype: torch.dtype) -> tuple[float, float]:
    # The serial oracle and the fused kernel accumulate the same products in a
    # different order; at bf16/fp16 activations the operands dominate the gap.
    if act_dtype == torch.float32:
        return 1e-4, 1e-3
    return 6e-2, 2e-1


def _tied_A(nheads: int, head_dim: int, dstate: int) -> torch.Tensor:
    """TIE_HDIM: A is scalar per head (stride(-1) == stride(-2) == 0)."""
    a = -torch.rand(nheads, device=DEV, dtype=torch.float32) - 1.0
    return a.view(nheads, 1, 1).expand(nheads, head_dim, dstate)


def _processed_dt(dt_raw: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """What the kernel stores in dt_cache: bias applied, then softplus."""
    return torch.nn.functional.softplus(dt_raw.float() + dt_bias.float())


def _make_tokens(n: int, geom, act_dtype: torch.dtype):
    nheads, head_dim, dstate, ngroups = geom
    return (
        torch.randn(n, nheads, head_dim, device=DEV, dtype=act_dtype) * 0.1,
        torch.randn(n, nheads, device=DEV, dtype=act_dtype) * 0.1,
        torch.randn(n, ngroups, dstate, device=DEV, dtype=act_dtype) * 0.1,
        torch.randn(n, ngroups, dstate, device=DEV, dtype=act_dtype) * 0.1,
    )


def _serial_reference(
    *,
    state0: torch.Tensor,  # (H, P, N) checkpoint for this row
    hist,  # (x, dt, B, C) for the pnat replayed tokens
    window,  # (x, dt, B, C) for the seq_len new tokens
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step one token at a time. Returns (window outputs, post-history state).

    The post-history state is what the kernel must write back on a flush; the
    window outputs are what it must return regardless.
    """
    nheads, head_dim, dstate = state0.shape
    state = state0[None].clone()
    tied_a = A
    tied_bias = dt_bias[:, None].expand(-1, head_dim)
    tied_d = D[:, None].expand(-1, head_dim)

    def step(x_t, dt_t, b_t, c_t):
        out = torch.empty(1, nheads, head_dim, device=DEV, dtype=x_t.dtype)
        selective_state_update(
            state,
            x_t[None],
            dt_t[None, :, None].expand(-1, -1, head_dim),
            tied_a,
            b_t[None],
            c_t[None],
            tied_d,
            tied_bias,
            dt_softplus=True,
            out=out,
        )
        return out[0]

    x_h, dt_h, b_h, c_h = hist
    for i in range(x_h.shape[0]):
        step(x_h[i], dt_h[i], b_h[i], c_h[i])
    post_history_state = state[0].clone()

    x_w, dt_w, b_w, c_w = window
    outs = [step(x_w[i], dt_w[i], b_w[i], c_w[i]) for i in range(x_w.shape[0])]
    return torch.stack(outs), post_history_state


def _seed_ring(x_cache, b_cache, dt_cache, slot, ring_start, hist, dt_bias):
    """Write the replayed history into [ring_start, ring_start + pnat) mod R."""
    x_h, dt_h, b_h, _ = hist
    pnat = x_h.shape[0]
    ring_len = x_cache.shape[2]
    rows = (ring_start + torch.arange(pnat, device=DEV)) % ring_len
    # Token-major tests, head-major ring.
    x_cache[slot][:, rows] = x_h.permute(1, 0, 2).to(x_cache.dtype)
    b_cache[slot][:, rows] = b_h.permute(1, 0, 2).to(b_cache.dtype)
    dt_cache[slot][:, rows] = _processed_dt(dt_h, dt_bias).permute(1, 0)


def _run_step(
    *,
    geom,
    buffer_len: int,
    max_spec_len: int,
    seq_len: int,
    pnat: int,
    ring_start: int,
    act_dtype: torch.dtype,
    state_dtype: torch.dtype,
    algorithm: str,
    cache_rows: int = 4,
    slot: int = 2,
):
    """One FlashInfer verify step against the serial oracle for a single row."""

    nheads, head_dim, dstate, ngroups = geom
    ring_len = buffer_len + max_spec_len

    state = torch.randn(
        cache_rows, nheads, head_dim, dstate, device=DEV, dtype=state_dtype
    )
    x_cache = torch.randn(
        cache_rows, nheads, ring_len, head_dim, device=DEV, dtype=act_dtype
    )
    b_cache = torch.randn(
        cache_rows, ngroups, ring_len, dstate, device=DEV, dtype=act_dtype
    )
    # Positive: these are softplus outputs in production, and negative decays
    # would blow up the replay.
    dt_cache = (
        torch.randn(cache_rows, nheads, ring_len, device=DEV, dtype=torch.float32).abs()
        + 0.1
    )

    a = _tied_A(nheads, head_dim, dstate)
    d = torch.randn(nheads, device=DEV, dtype=torch.float32)
    dt_bias = torch.rand(nheads, device=DEV, dtype=torch.float32) * 0.1

    hist = _make_tokens(pnat, geom, act_dtype)
    window = _make_tokens(seq_len, geom, act_dtype)
    _seed_ring(x_cache, b_cache, dt_cache, slot, ring_start, hist, dt_bias)

    expected_out, post_history_state = _serial_reference(
        state0=state[slot].float(),
        hist=hist,
        window=window,
        A=a,
        D=d,
        dt_bias=dt_bias,
    )

    x_w, dt_w, b_w, c_w = window
    out = torch.empty(1, seq_len, nheads, head_dim, device=DEV, dtype=act_dtype)
    scratch = {}
    if algorithm != "monolith":
        k_old = ((buffer_len + 7) // 8) * 8
        scratch = {
            "cb_scaled": torch.empty(1, nheads, 32, 8, device=DEV, dtype=act_dtype),
            "cumAdt_vec": torch.empty(1, nheads, 16, device=DEV, dtype=torch.float32),
            "cb_old": torch.empty(
                1, nheads, 32, k_old // 2, device=DEV, dtype=act_dtype
            ),
        }

    backend = _make_backend(algorithm)

    state_before = state.clone()
    backend(
        state,
        x_cache,
        b_cache,
        dt_cache,
        torch.full((cache_rows,), ring_start, dtype=torch.int32, device=DEV),
        torch.full((cache_rows,), pnat, dtype=torch.int32, device=DEV),
        x_w.unsqueeze(0),
        dt_w[None, :, :, None].expand(-1, -1, -1, head_dim),
        a,
        b_w.unsqueeze(0),
        c_w.unsqueeze(0),
        out,
        D=d[:, None].expand(-1, head_dim),
        dt_bias=dt_bias[:, None].expand(-1, head_dim),
        dt_softplus=True,
        state_batch_indices=torch.tensor([slot], dtype=torch.int32, device=DEV),
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32, device=DEV),
        max_spec_len=max_spec_len,
        replayssm_buffer_len=buffer_len,
        **scratch,
    )
    return dict(
        out=out[0],
        expected_out=expected_out,
        state=state,
        state_before=state_before,
        post_history_state=post_history_state,
        x_cache=x_cache,
        b_cache=b_cache,
        dt_cache=dt_cache,
        window=window,
        dt_bias=dt_bias,
        slot=slot,
        ring_len=ring_len,
        should_flush=(pnat + seq_len) > buffer_len,
    )


# ---------------------------------------------------------------- suites ----
# Bounded suites rather than a Cartesian product: each one isolates a single
# axis against a small representative shape.


@pytest.mark.parametrize("algorithm", ["monolith", "two-kernel"])
@pytest.mark.parametrize("pnat,seq_len", [(0, 4), (3, 4), (12, 4), (13, 4)])
def test_forced_algorithm_matches_serial_reference(algorithm, pnat, seq_len):
    set_random_seed(0)
    r = _run_step(
        geom=_SMALL,
        buffer_len=16,
        max_spec_len=4,
        seq_len=seq_len,
        pnat=pnat,
        ring_start=0,
        act_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        algorithm=algorithm,
    )
    rtol, atol = _tolerances(torch.bfloat16)
    torch.testing.assert_close(
        r["out"].float(), r["expected_out"].float(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    "act_dtype,state_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.float16, torch.float16),
    ],
)
def test_dtype_coverage(act_dtype, state_dtype):
    set_random_seed(1)
    r = _run_step(
        geom=_SMALL,
        buffer_len=16,
        max_spec_len=4,
        seq_len=4,
        pnat=5,
        ring_start=0,
        act_dtype=act_dtype,
        state_dtype=state_dtype,
        algorithm="monolith",
    )
    rtol, atol = _tolerances(act_dtype)
    torch.testing.assert_close(
        r["out"].float(), r["expected_out"].float(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    "buffer_len,max_spec_len,seq_len",
    [
        (16, 16, 16),  # T == B
        (16, 8, 8),
        (16, 4, 1),  # actual_len == 1, well under T
        (16, 4, 4),  # actual_len == T
        (8, 4, 3),
    ],
)
def test_boundary_shapes(buffer_len, max_spec_len, seq_len):
    set_random_seed(2)
    r = _run_step(
        geom=_TINY,
        buffer_len=buffer_len,
        max_spec_len=max_spec_len,
        seq_len=seq_len,
        pnat=min(3, buffer_len - seq_len),
        ring_start=1,
        act_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        algorithm="monolith",
    )
    rtol, atol = _tolerances(torch.bfloat16)
    torch.testing.assert_close(
        r["out"].float(), r["expected_out"].float(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("ring_start", [0, 7, 17, 19])
def test_ring_wraparound_on_a_non_pow2_ring(ring_start):
    """R = 20 here, so a replay window starting near the end must wrap."""
    set_random_seed(3)
    r = _run_step(
        geom=_TINY,
        buffer_len=16,
        max_spec_len=4,
        seq_len=4,
        pnat=6,
        ring_start=ring_start,
        act_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        algorithm="monolith",
    )
    assert r["ring_len"] == 20
    rtol, atol = _tolerances(torch.bfloat16)
    torch.testing.assert_close(
        r["out"].float(), r["expected_out"].float(), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    "pnat,seq_len,expect_flush",
    [
        (12, 4, False),  # 12 + 4 == 16, not > B
        (13, 4, True),
        (13, 1, False),  # same history, shorter row -> no flush
        (16, 1, True),
    ],
)
def test_checkpoint_written_iff_the_actual_row_length_overflows(
    pnat, seq_len, expect_flush
):
    """The kernel's varlen predicate is pnat + seq_len > max_window.

    A host that used max T here would advance ring_start on a step the kernel
    did not checkpoint, permanently desynchronising the ring.
    """
    set_random_seed(4)
    r = _run_step(
        geom=_TINY,
        buffer_len=16,
        max_spec_len=4,
        seq_len=seq_len,
        pnat=pnat,
        ring_start=0,
        act_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        algorithm="monolith",
    )
    assert r["should_flush"] is expect_flush
    slot = r["slot"]
    changed = not torch.equal(r["state"][slot], r["state_before"][slot])
    assert changed is expect_flush

    if expect_flush:
        # The written checkpoint must be the state after the replayed history
        # only -- not including the verify window.
        torch.testing.assert_close(
            r["state"][slot].float(),
            r["post_history_state"].float(),
            rtol=6e-2,
            atol=2e-1,
        )


def test_new_tokens_are_appended_after_the_replayed_history():
    set_random_seed(5)
    pnat, seq_len, ring_start = 6, 4, 17
    r = _run_step(
        geom=_TINY,
        buffer_len=16,
        max_spec_len=4,
        seq_len=seq_len,
        pnat=pnat,
        ring_start=ring_start,
        act_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        algorithm="monolith",
    )
    x_w, dt_w, b_w, _ = r["window"]
    rows = (ring_start + pnat + torch.arange(seq_len, device=DEV)) % r["ring_len"]
    slot = r["slot"]

    torch.testing.assert_close(
        r["x_cache"][slot][:, rows].float(),
        x_w.permute(1, 0, 2).float(),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        r["b_cache"][slot][:, rows].float(),
        b_w.permute(1, 0, 2).float(),
        rtol=1e-2,
        atol=1e-2,
    )
    # dt_cache holds processed dt, not the raw dt the Triton ring stores.
    torch.testing.assert_close(
        r["dt_cache"][slot][:, rows],
        _processed_dt(dt_w, r["dt_bias"]).permute(1, 0),
        rtol=1e-2,
        atol=1e-2,
    )


def test_padded_rows_are_skipped_via_pad_slot_id():
    """CUDA-graph padding rows carry NULL_BLOCK_ID and must not touch a page."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    set_random_seed(6)

    nheads, head_dim, dstate, ngroups = _TINY
    buffer_len, max_spec_len, seq_len = 16, 4, 4
    ring_len = buffer_len + max_spec_len
    rows, cache_rows = 2, 4

    state = torch.randn(
        cache_rows, nheads, head_dim, dstate, device=DEV, dtype=torch.float32
    )
    before = state.clone()
    x_cache = torch.randn(
        cache_rows, nheads, ring_len, head_dim, device=DEV, dtype=torch.bfloat16
    )
    b_cache = torch.randn(
        cache_rows, ngroups, ring_len, dstate, device=DEV, dtype=torch.bfloat16
    )
    dt_cache = (
        torch.randn(cache_rows, nheads, ring_len, device=DEV, dtype=torch.float32).abs()
        + 0.1
    )
    x_before, b_before = x_cache.clone(), b_cache.clone()

    x, dt, b, c = _make_tokens(rows * seq_len, _TINY, torch.bfloat16)
    out = torch.zeros(
        1, rows * seq_len, nheads, head_dim, device=DEV, dtype=torch.bfloat16
    )
    a = _tied_A(nheads, head_dim, dstate)
    dt_bias = torch.rand(nheads, device=DEV, dtype=torch.float32) * 0.1

    backend = _make_backend("monolith")
    backend(
        state,
        x_cache,
        b_cache,
        dt_cache,
        torch.zeros(cache_rows, dtype=torch.int32, device=DEV),
        torch.full((cache_rows,), 3, dtype=torch.int32, device=DEV),
        x.unsqueeze(0),
        dt[None, :, :, None].expand(-1, -1, -1, head_dim),
        a,
        b.unsqueeze(0),
        c.unsqueeze(0),
        out,
        D=torch.randn(nheads, device=DEV)[:, None].expand(-1, head_dim),
        dt_bias=dt_bias[:, None].expand(-1, head_dim),
        dt_softplus=True,
        # Row 0 is real (slot 3), row 1 is graph padding.
        state_batch_indices=torch.tensor(
            [3, NULL_BLOCK_ID], dtype=torch.int32, device=DEV
        ),
        query_start_loc=torch.tensor(
            [0, seq_len, 2 * seq_len], dtype=torch.int32, device=DEV
        ),
        max_spec_len=max_spec_len,
        replayssm_buffer_len=buffer_len,
    )

    # The null page (row 0 of the cache) must be untouched.
    torch.testing.assert_close(state[NULL_BLOCK_ID], before[NULL_BLOCK_ID])
    torch.testing.assert_close(x_cache[NULL_BLOCK_ID], x_before[NULL_BLOCK_ID])
    torch.testing.assert_close(b_cache[NULL_BLOCK_ID], b_before[NULL_BLOCK_ID])


@pytest.mark.parametrize("geom", [_REAL, _REAL_TP4])
@pytest.mark.parametrize("algorithm", ["monolith", "two-kernel", "auto"])
def test_production_geometry_smoke(geom, algorithm):
    set_random_seed(7)
    r = _run_step(
        geom=geom,
        buffer_len=16,
        max_spec_len=4,
        seq_len=4,
        pnat=9,
        ring_start=11,
        act_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        algorithm=algorithm,
    )
    rtol, atol = _tolerances(torch.bfloat16)
    torch.testing.assert_close(
        r["out"].float(), r["expected_out"].float(), rtol=rtol, atol=atol
    )
    assert torch.isfinite(r["out"].float()).all()


def test_multi_step_lifecycle_tracks_the_serial_reference():
    """Non-flush accumulation, a flush, a rollback, wraparound, another flush.

    Drives the host cursor rules alongside the kernel so a desync between the
    two shows up as an output mismatch rather than only as a cursor assertion.
    """

    set_random_seed(8)
    nheads, head_dim, dstate, ngroups = _TINY
    buffer_len, max_spec_len = 8, 4
    ring_len = buffer_len + max_spec_len
    slot, cache_rows = 1, 2
    act_dtype = torch.bfloat16

    state = torch.randn(
        cache_rows, nheads, head_dim, dstate, device=DEV, dtype=torch.float32
    )
    x_cache = torch.zeros(
        cache_rows, nheads, ring_len, head_dim, device=DEV, dtype=act_dtype
    )
    b_cache = torch.zeros(
        cache_rows, ngroups, ring_len, dstate, device=DEV, dtype=act_dtype
    )
    dt_cache = torch.ones(cache_rows, nheads, ring_len, device=DEV, dtype=torch.float32)

    a = _tied_A(nheads, head_dim, dstate)
    d = torch.randn(nheads, device=DEV, dtype=torch.float32)
    dt_bias = torch.rand(nheads, device=DEV, dtype=torch.float32) * 0.1

    backend = _make_backend("monolith")

    # Independent ground truth: the accepted token stream fed serially through
    # a state that is never rolled back.
    truth_state = state[slot][None].float().clone()
    history, origin, is_flush = 0, 0, False
    accepted_pattern = [4, 1, 4, 2, 4, 3, 4, 4, 1, 4]
    saw_flush = saw_wrap = False

    for step, accepted in enumerate(accepted_pattern):
        seq_len = 4
        window = _make_tokens(seq_len, _TINY, act_dtype)
        x_w, dt_w, b_w, c_w = window

        out = torch.empty(1, seq_len, nheads, head_dim, device=DEV, dtype=act_dtype)
        backend(
            state,
            x_cache,
            b_cache,
            dt_cache,
            torch.full((cache_rows,), origin, dtype=torch.int32, device=DEV),
            torch.full((cache_rows,), history, dtype=torch.int32, device=DEV),
            x_w.unsqueeze(0),
            dt_w[None, :, :, None].expand(-1, -1, -1, head_dim),
            a,
            b_w.unsqueeze(0),
            c_w.unsqueeze(0),
            out,
            D=d[:, None].expand(-1, head_dim),
            dt_bias=dt_bias[:, None].expand(-1, head_dim),
            dt_softplus=True,
            state_batch_indices=torch.tensor([slot], dtype=torch.int32, device=DEV),
            query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32, device=DEV),
            max_spec_len=max_spec_len,
            replayssm_buffer_len=buffer_len,
        )

        # Accepted-prefix outputs must match a plain serial decode.
        expected = []
        for i in range(accepted):
            o = torch.empty(1, nheads, head_dim, device=DEV, dtype=act_dtype)
            selective_state_update(
                truth_state,
                x_w[i][None],
                dt_w[i][None, :, None].expand(-1, -1, head_dim),
                a,
                b_w[i][None],
                c_w[i][None],
                d[:, None].expand(-1, head_dim),
                dt_bias[:, None].expand(-1, head_dim),
                dt_softplus=True,
                out=o,
            )
            expected.append(o[0])
        torch.testing.assert_close(
            out[0][:accepted].float(),
            torch.stack(expected).float(),
            rtol=6e-2,
            atol=2e-1,
            msg=f"step {step}: accepted-prefix output diverged",
        )

        # Host cursor rules, mirroring commit_replayssm_spec_flashinfer.
        if accepted > 0:
            if is_flush:
                new_origin = origin + history
                if new_origin >= ring_len:
                    new_origin -= ring_len
                    saw_wrap = True
                origin, history = new_origin, accepted
                saw_flush = True
            else:
                history += accepted
        is_flush = (history + seq_len) > buffer_len
        assert 0 <= history <= buffer_len
        assert 0 <= origin < ring_len

    assert saw_flush, "the pattern should have flushed"
    assert saw_wrap, "the pattern should have wrapped the ring origin"
