# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the layer math in `specialized_decode/reference.py` is supposed to be,
checked against properties rather than against a second implementation.

A reference is only a yardstick if something holds it straight. Comparing it
to a kernel would be circular -- the kernel is what it exists to measure -- and
comparing it to another library's layer would make that library the real
definition. So the assertions here are the properties the math has: a rotation
preserves a pair's norm, a normalization is blind to its input's scale, a
softmax's weights sum to one, a renormalized route sums to its scaling factor.
Those are true of the function, not of any implementation of it, and a
definition that fails one of them is wrong on its own terms.

Where a property pins a value exactly -- a rotation by zero, a clamp past its
limit, an inverse round trip -- it is asserted exactly. Everything else is
within float64 round-off of the quantity it is derived from, not within a
tolerance picked to make it pass.

Plain torch on the CPU: no device code, no FlyDSL, no GPU.
"""

import math

import pytest
import torch

from tests.models.common.specialized_decode import reference as R

F64 = R.F64


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _rand(*shape: int, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, generator=_gen(seed), dtype=F64) * scale


def _close(got: torch.Tensor, want: torch.Tensor, scale: float = 1.0) -> None:
    """`got` within float64 round-off of `want`, reported as a multiple of it
    rather than as two decimals that print the same."""
    ulps = torch.finfo(F64).eps * scale * want.abs().clamp(min=1.0)
    off = (got.double() - want.double()).abs()
    assert torch.all(off <= 64 * ulps), (
        f"off by up to {(off / ulps).max().item():.1f} units of "
        f"{torch.finfo(F64).eps:.3e} x the value, at {off.argmax().item()}"
    )


# ---------------------------------------------------------------- reductions
def test_ordered_sum_exact_values_do_not_care_about_order():
    """Sums of values a float64 adds exactly -- small integers -- come out
    the same whichever order they are added in, so `order` is doing nothing
    but ordering."""
    terms = torch.arange(8, dtype=F64).expand(3, 8).contiguous()
    forward = R.ordered_sum(terms)
    assert torch.equal(forward, R.ordered_sum(terms, order=range(7, -1, -1)))
    assert torch.equal(forward, terms.sum(-1))


def test_ordered_sum_order_is_load_bearing_once_it_rounds():
    """With a rounding between the adds, the order changes the answer: one
    big term first swallows the small ones, the small ones first survive.
    This is why the order is an argument and not a convention."""
    terms = torch.tensor([[1.0, 2.0**-9, 2.0**-9, 2.0**-9]], dtype=F64)
    big_first = R.ordered_sum(terms, round_to=R.bf16_rne)
    small_first = R.ordered_sum(terms, order=[1, 2, 3, 0], round_to=R.bf16_rne)
    assert big_first.item() == 1.0
    assert small_first.item() > 1.0


@pytest.mark.parametrize("order", [[0, 1], [0, 0, 1, 2], [0, 1, 2, 4]])
def test_ordered_sum_rejects_an_order_that_is_not_a_permutation(order: list[int]):
    """A reordering, not a selection. A dropped or repeated term still
    returns a plausible total, which is the worst thing a yardstick can do,
    so it is rejected rather than summed."""
    with pytest.raises(AssertionError, match="permutation"):
        R.ordered_sum(torch.ones(1, 4, dtype=F64), order=order)


# ---------------------------------------------------------------- RMSNorm
@pytest.mark.parametrize("factor", [2.0**-40, 0.5, 7.0, 2.0**40])
def test_rms_norm_ignores_its_input_scale(factor: float):
    """Without an epsilon the normalization is scale invariant: it divides by
    a quantity that scales with its input."""
    x = _rand(4, 64, seed=1)
    _close(R.rms_norm(x * factor), R.rms_norm(x), scale=4.0)


def test_rms_norm_output_has_unit_rms():
    """Which is what it is for -- before any weight is applied."""
    normed = R.rms_norm(_rand(4, 64, seed=2, scale=3.0))
    _close(normed.square().mean(-1), torch.ones(4, dtype=F64))


def test_rms_norm_weight_is_applied_after_the_normalization():
    x, weight = _rand(4, 64, seed=3), _rand(64, seed=4)
    _close(R.rms_norm(x, weight), R.rms_norm(x) * weight, scale=4.0)


def test_rms_norm_eps_divides_by_the_mean_not_the_sum():
    """`eps` sits next to a mean, so doubling the hidden size of a constant
    row leaves the answer alone. A sum-of-squares formula would not."""
    narrow = torch.full((1, 32), 0.25, dtype=F64)
    wide = torch.full((1, 64), 0.25, dtype=F64)
    eps = 1e-3
    _close(R.rms_norm(narrow, eps=eps)[0, :1], R.rms_norm(wide, eps=eps)[0, :1])


# ---------------------------------------------------------------- Sinkhorn
def test_sinkhorn_without_eps_leaves_the_columns_stochastic():
    """The last normalization is over the columns, so with no epsilon in the
    denominators they sum to exactly one."""
    comb = R.sinkhorn(_rand(5, 4, 4, seed=5, scale=2.0), eps=0.0, repeat=20)
    _close(comb.sum(-2), torch.ones(5, 4, dtype=F64))


def test_sinkhorn_converges_toward_doubly_stochastic():
    """The columns are exact because they go last; the rows are only as close
    as the iteration got, and every extra round brings them closer. Twenty
    rounds is a choice about how close, not a point where it becomes an
    identity, so the assertion is the convergence and not a tolerance."""
    logits = _rand(5, 4, 4, seed=6, scale=2.0)
    off = [
        (R.sinkhorn(logits, eps=0.0, repeat=n).sum(-1) - 1.0).abs().max().item()
        for n in (2, 5, 20, 60)
    ]
    assert off == sorted(off, reverse=True), off
    assert off[-1] < 1e-5, off


def test_sinkhorn_eps_only_ever_shrinks_the_sums():
    """`eps` is added to every denominator, so it can only divide by more:
    the result is near doubly stochastic from below, never above."""
    logits = _rand(5, 4, 4, seed=7, scale=2.0)
    comb = R.sinkhorn(logits, eps=1e-6, repeat=20)
    assert torch.all(comb > 0)
    sums = comb.sum(-2)
    assert torch.all((sums <= 1.0) & (sums > 1.0 - 1e-5))


def test_sinkhorn_eps_is_added_to_the_softmax_not_folded_into_it():
    """A row whose softmax underflows to exactly zero separates the two
    readings of `eps`. Added, it floors the entry and the column it sits in
    still has something to normalize; scaled in as `(1 + eps)` instead, the
    entry stays exactly zero and takes its column's sum down with it. The
    floor is part of the definition, not a guard on it."""
    logits = torch.full((1, 4, 4), -1.0e4, dtype=F64)
    logits[..., 0] = 0.0
    underflowed = torch.softmax(logits, dim=-1)
    assert torch.equal(underflowed[0, :, 1:], torch.zeros(4, 3, dtype=F64))
    comb = R.sinkhorn(logits, eps=1e-6, repeat=20)
    assert torch.all(torch.isfinite(comb))
    assert torch.all(comb > 0.0)


def test_sinkhorn_rejects_a_repeat_below_one():
    """`range(repeat - 1)` is empty at 0 and below, so without the guard
    asking for no passes silently gets one."""
    with pytest.raises(AssertionError, match="column passes"):
        R.sinkhorn(_rand(1, 4, 4, seed=86), eps=0.0, repeat=0)


def test_sinkhorn_one_round_is_softmax_then_one_column_pass():
    """`repeat` counts column passes: one means the row softmax and a single
    column normalization, with nothing after it."""
    logits = _rand(2, 3, 3, seed=8)
    rows = torch.softmax(logits, dim=-1)
    want = rows / rows.sum(-2, keepdim=True)
    _close(R.sinkhorn(logits, eps=0.0, repeat=1), want, scale=4.0)


# ---------------------------------------------------------------- mHC seam
def _seam_kwargs(**over):
    kwargs = dict(
        norm_eps=1e-6,
        rms_eps=1e-6,
        pre_eps=0.0,
        sinkhorn_eps=0.0,
        post_mult=2.0,
        sinkhorn_repeat=20,
    )
    kwargs.update(over)
    return kwargs


def test_mhc_post_with_identity_mixes_is_a_plain_residual_add():
    """Every stream keeps itself and takes all of `x`: the fold degenerates
    to the residual addition it generalizes."""
    x, residual = _rand(3, 16, seed=9), _rand(3, 4, 16, seed=10)
    identity = torch.eye(4, dtype=F64).expand(3, 4, 4)
    post = torch.ones(3, 4, dtype=F64)
    _close(R.mhc_post(x, residual, post, identity), residual + x.unsqueeze(-2), 4.0)


def test_mhc_post_reads_comb_from_the_source_stream():
    """`comb[t, i, j]` is how much of stream i reaches stream j. A single
    one at (0, 1) moves stream 0's content onto stream 1 and empties the
    rest, which an implementation that transposed the mix would get backwards.
    """
    residual = _rand(1, 4, 8, seed=11)
    comb = torch.zeros(1, 4, 4, dtype=F64)
    comb[0, 0, 1] = 1.0
    folded = R.mhc_post(
        torch.zeros(1, 8, dtype=F64), residual, comb.new_zeros(1, 4), comb
    )
    assert torch.equal(folded[0, 1], residual[0, 0])
    assert torch.equal(folded[0, [0, 2, 3]], torch.zeros(3, 8, dtype=F64))


def test_mhc_collapse_selects_a_stream_for_a_one_hot_pre_mix():
    residual = _rand(3, 4, 16, seed=12)
    pre = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=F64).expand(3, 4)
    assert torch.equal(R.mhc_collapse(residual, pre), residual[:, 2])


def test_mhc_delayed_seam_degenerates_to_rms_norm_of_one_stream():
    """Identity comb, no post, a one-hot pre: the seam is RMSNorm of the
    stream the pre mix points at, which is the path a layer takes when the
    residual is handed over settled."""
    x, residual = _rand(3, 16, seed=13), _rand(3, 4, 16, seed=14)
    fn, scale = _rand(24, 64, seed=15), _rand(3, seed=16)
    base, weight = _rand(24, seed=17), _rand(16, seed=18)
    pre = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=F64).expand(3, 4)
    folded, layer_input, *_ = R.mhc_delayed_seam(
        x,
        residual,
        torch.zeros(3, 4, dtype=F64),
        torch.eye(4, dtype=F64).expand(3, 4, 4),
        pre,
        fn,
        scale,
        base,
        weight,
        **_seam_kwargs(norm_eps=0.0),
    )
    assert torch.equal(folded, residual)
    _close(layer_input, R.rms_norm(residual[:, 0], weight), scale=4.0)


def test_mhc_delayed_seam_mixes_come_from_the_folded_residual():
    """Delayed means the mixes a seam publishes are read off the residual
    *after* this sublayer's output went into it, not before -- so folding a
    non-zero `x` has to move them."""
    residual, x = _rand(3, 4, 16, seed=19), _rand(3, 16, seed=20)
    args = (
        torch.full((3, 4), 0.5, dtype=F64),
        R.sinkhorn(_rand(3, 4, 4, seed=21), 0.0, 20),
        torch.full((3, 4), 0.25, dtype=F64),
        _rand(24, 64, seed=22),
        _rand(3, seed=23),
        _rand(24, seed=24),
    )
    kwargs = _seam_kwargs()
    _, _, *quiet = R.mhc_delayed_seam(torch.zeros_like(x), residual, *args, **kwargs)
    _, _, *loud = R.mhc_delayed_seam(x, residual, *args, **kwargs)
    for was, now in zip(quiet, loud):
        assert not torch.allclose(was, now)


def test_mhc_seams_chain_into_each_other():
    """A delayed seam applies the mixes the previous one published, so what a
    seam returns has to be what the next one takes -- same shapes, same
    order, nothing reshaped in between. `seam(y, folded, *mixes, ...)` is the
    whole claim, and it is the one thing no single-seam test can make.
    """
    t, m, h = 3, 4, 16
    fn, scale = _rand(m * (m + 2), m * h, seed=79), _rand(3, seed=80)
    base, residual = _rand(m * (m + 2), seed=81), _rand(t, m, h, seed=82)
    kwargs = _seam_kwargs()
    folded, _, post, comb, pre = R.mhc_delayed_seam(
        _rand(t, h, seed=83),
        residual,
        torch.full((t, m), 0.5, dtype=F64),
        R.sinkhorn(_rand(t, m, m, seed=84), 0.0, 20),
        torch.full((t, m), 0.25, dtype=F64),
        fn,
        scale,
        base,
        **kwargs,
    )
    mixes = (post, comb, pre)
    assert (post.shape, comb.shape, pre.shape) == ((t, m), (t, m, m), (t, m))
    # Which slot holds which mix, settled against `mhc_mixes` by name. Going
    # back to the forwarded tuple instead would be circular -- `pre` and
    # `post` are both [T, M], so a swap lands in a slot of the right shape
    # and every assertion downstream of it agrees with itself.
    mix_kwargs = {k: v for k, v in kwargs.items() if k != "norm_eps"}
    want_pre, want_post, want_comb = R.mhc_mixes(folded, fn, scale, base, **mix_kwargs)
    assert torch.equal(post, want_post)
    assert torch.equal(comb, want_comb)
    assert torch.equal(pre, want_pre)
    y = _rand(t, h, seed=85)
    again, layer_input, *_ = R.mhc_delayed_seam(
        y, folded, *mixes, fn, scale, base, **kwargs
    )
    _close(again, R.mhc_post(y, folded, want_post, want_comb), scale=16.0)
    _close(
        layer_input, R.rms_norm(R.mhc_collapse(again, want_pre), eps=1e-6), scale=16.0
    )


def test_mhc_mixes_stay_in_their_ranges():
    """A pre mix is a sigmoid plus `pre_eps`, so no stream is ever switched
    off; a post mix is a sigmoid times `post_mult`, so the fold is bounded;
    a comb mix is Sinkhorn output, so it is positive."""
    pre, post, comb = R.mhc_mixes(
        _rand(6, 4, 16, seed=25, scale=10.0),
        _rand(24, 64, seed=26, scale=4.0),
        _rand(3, seed=27, scale=4.0),
        _rand(24, seed=28, scale=4.0),
        rms_eps=1e-6,
        pre_eps=1e-3,
        sinkhorn_eps=0.0,
        post_mult=2.0,
        sinkhorn_repeat=20,
    )
    # Non-strict at both ends: a sigmoid of a large enough mix is 1.0 or 0.0
    # exactly, and saturating there is the bound holding rather than failing.
    # At the bottom `pre_eps` is then the whole value, which is what it is
    # for -- a stream the mix has given up on still contributes.
    assert torch.all((pre >= 1e-3) & (pre <= 1.0 + 1e-3))
    assert torch.all((post >= 0.0) & (post <= 2.0))
    assert torch.all(comb > 0.0)


def test_mhc_post_rejects_mixes_of_the_wrong_stream_count():
    """`post_mix` and `comb_mix` are both indexed by stream, and a mix built
    for a different M broadcasts into something of a plausible shape instead
    of stopping."""
    x, residual = _rand(3, 16, seed=90), _rand(3, 4, 16, seed=91)
    comb = torch.eye(4, dtype=F64).expand(3, 4, 4)
    with pytest.raises(AssertionError, match="streams"):
        R.mhc_post(x, residual, torch.ones(3, 2, dtype=F64), comb)
    with pytest.raises(AssertionError, match="streams"):
        R.mhc_post(x, residual, torch.ones(3, 4, dtype=F64), torch.eye(2, dtype=F64))


def test_mhc_mixes_rejects_a_gate_projection_of_the_wrong_height():
    """M (M + 2) rows for M streams. A projection sized for another M slices
    into the wrong row groups rather than failing."""
    residual = _rand(3, 4, 16, seed=92)
    with pytest.raises(AssertionError, match="rows"):
        R.mhc_mixes(
            residual,
            _rand(20, 64, seed=93),
            _rand(3, seed=94),
            _rand(24, seed=95),
            **{k: v for k, v in _seam_kwargs().items() if k != "norm_eps"},
        )


# ---------------------------------------------------------------- RoPE
def _cos_sin(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cos(theta), torch.sin(theta)


def test_rope_by_zero_is_the_identity_exactly():
    """Cos 0 and sin 0 are exact, so nothing is allowed to move -- not even
    by a rounding."""
    x = _rand(2, 8, 64, seed=29)
    cos, sin = torch.ones(32, dtype=F64), torch.zeros(32, dtype=F64)
    assert torch.equal(R.rope_pairs(x, cos, sin), x)


def test_rope_by_a_quarter_turn_swaps_the_pair():
    """At cos 0, sin 1 the rotation is exactly `(e, o) -> (-o, e)`, written
    with the literals rather than with `cos(pi / 2)`, which is not zero."""
    x = _rand(2, 8, 64, seed=30)
    cos, sin = torch.zeros(32, dtype=F64), torch.ones(32, dtype=F64)
    want = torch.stack([-x[..., 1::2], x[..., 0::2]], -1).flatten(-2)
    assert torch.equal(R.rope_pairs(x, cos, sin), want)


def test_rope_preserves_every_pair_norm():
    """A plane rotation per adjacent pair: the pair's length is what it
    cannot change."""
    x = _rand(2, 8, 64, seed=31)
    cos, sin = _cos_sin(_rand(32, seed=32, scale=3.0))
    rotated = R.rope_pairs(x, cos, sin)
    _close(
        rotated.unflatten(-1, (32, 2)).square().sum(-1),
        x.unflatten(-1, (32, 2)).square().sum(-1),
        scale=4.0,
    )


def test_rope_composes_by_adding_the_angles():
    a, b = _rand(32, seed=33, scale=2.0), _rand(32, seed=34, scale=2.0)
    x = _rand(2, 8, 64, seed=35)
    twice = R.rope_pairs(R.rope_pairs(x, *_cos_sin(a)), *_cos_sin(b))
    _close(twice, R.rope_pairs(x, *_cos_sin(a + b)), scale=8.0)


def test_rope_inverse_undoes_the_rotation():
    x = _rand(2, 8, 64, seed=36)
    cos, sin = _cos_sin(_rand(32, seed=37, scale=3.0))
    _close(R.rope_pairs(R.rope_pairs(x, cos, sin), cos, sin, inverse=True), x, 8.0)


def test_rope_rotates_adjacent_pairs_not_split_halves():
    """GPT-J's convention. A single pair rotated on its own must leave every
    other dimension untouched, which the split-half convention would not."""
    x = _rand(1, 8, seed=38)
    cos = torch.ones(4, dtype=F64)
    sin = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=F64)
    moved = R.rope_pairs(x, cos, sin)
    assert torch.equal(moved[..., [0, 1, 4, 5, 6, 7]], x[..., [0, 1, 4, 5, 6, 7]])
    assert not torch.equal(moved[..., 2:4], x[..., 2:4])


def test_rope_rejects_a_table_of_the_wrong_width():
    with pytest.raises(AssertionError, match="pairs"):
        R.rope_pairs(
            _rand(1, 8, seed=39), torch.ones(3, dtype=F64), torch.zeros(3, dtype=F64)
        )


# ---------------------------------------------------------------- MLA
def _softmax_attention(q, keys, scale, sink=None):
    """The same attention written the usual way -- base e, one pass, the sink
    as an extra logit with no value behind it -- so that the base-two,
    chunked, sink-folded definition has something independent to match."""
    logits = (q.double() @ keys.double().t()) * scale
    if sink is not None:
        logits = torch.cat([logits, sink.double().unsqueeze(-1)], -1)
    weights = torch.softmax(logits, -1)[:, : keys.shape[0]]
    return weights @ keys.double()


def test_mla_matches_a_base_e_softmax():
    q, keys = _rand(8, 32, seed=40), _rand(50, 32, seed=41)
    scale = 32.0**-0.5
    _close(R.mla_decode(q, keys, scale), _softmax_attention(q, keys, scale), 64.0)


def test_mla_folds_the_sink_into_the_denominator():
    """The sink takes weight away from the keys without contributing a value,
    so it is a logit in the denominator and nowhere else."""
    q, keys = _rand(8, 32, seed=42), _rand(50, 32, seed=43)
    sink, scale = _rand(8, seed=44, scale=2.0), 32.0**-0.5
    _close(
        R.mla_decode(q, keys, scale, sink=sink),
        _softmax_attention(q, keys, scale, sink=sink),
        64.0,
    )


@pytest.mark.parametrize("chunks", [[50], [16, 16, 16, 2], [1] * 50, [49, 1]])
def test_mla_is_the_same_however_the_keys_are_split(chunks: list[int]):
    """Flash decoding is an arrangement of the arithmetic, not a different
    answer: every split, and every order of combining the partials, agrees to
    round-off with the single pass."""
    q, keys = _rand(8, 32, seed=45), _rand(50, 32, seed=46)
    sink, scale = _rand(8, seed=47), 32.0**-0.5
    want = R.mla_decode(q, keys, scale, sink=sink)
    _close(R.mla_decode(q, keys, scale, sink=sink, chunks=chunks), want, 64.0)
    reverse = list(range(len(chunks) - 1, -1, -1))
    _close(
        R.mla_decode(q, keys, scale, sink=sink, chunks=chunks, order=reverse),
        want,
        64.0,
    )


def test_mla_weights_sum_to_one_without_a_sink():
    """Shifting every key by a constant shifts every logit of a head by the
    same amount and every value by that constant: the output comes back
    shifted by exactly the same constant only if the weights are a partition
    of one. Softmax shift invariance and the weights' sum in one assertion.
    """
    q, keys = _rand(8, 32, seed=48), _rand(50, 32, seed=49)
    shift, scale = _rand(32, seed=50), 32.0**-0.5
    _close(
        R.mla_decode(q, keys + shift, scale),
        R.mla_decode(q, keys, scale) + shift,
        scale=64.0,
    )


def test_a_sink_shrinks_the_output_toward_nothing():
    """A sink adds to the denominator and not to the numerator, so it scales
    the whole head's output by the share of the weight the keys kept:
    `Z / (Z + exp(sink))`. It does not point the output anywhere else.

    Note that this also costs the shift invariance above: the keys' logits
    move with a shift and the sink's does not, so with a sink the weights
    are no longer the same ones.
    """
    q, keys = _rand(8, 32, seed=51), _rand(50, 32, seed=52)
    scale, sink = 32.0**-0.5, _rand(8, seed=53, scale=2.0)
    logits = (q @ keys.t()) * scale
    top = logits.amax(-1)
    partition = torch.exp(logits - top.unsqueeze(-1)).sum(-1)
    kept = partition / (partition + torch.exp(sink - top))
    assert torch.all((kept > 0.0) & (kept < 1.0))
    _close(
        R.mla_decode(q, keys, scale, sink=sink),
        R.mla_decode(q, keys, scale) * kept.unsqueeze(-1),
        scale=64.0,
    )


def test_mla_over_one_key_returns_that_key():
    """A window of one: the only weight is one, whatever the logit."""
    q, keys = _rand(8, 32, seed=54), _rand(1, 32, seed=55)
    _close(R.mla_decode(q, keys, 32.0**-0.5), keys.expand(8, 32), scale=4.0)


def test_mla_over_identical_keys_returns_the_key():
    """Every value the same: the output is that value for any weights that
    sum to one, which is a different route to the same guarantee."""
    key = _rand(1, 32, seed=56)
    out = R.mla_decode(_rand(8, 32, seed=57), key.expand(50, 32).contiguous(), 1.0)
    _close(out, key.expand(8, 32), scale=16.0)


def test_mla_rejects_a_split_that_loses_keys():
    with pytest.raises(AssertionError, match="chunks"):
        R.mla_decode(_rand(2, 8, seed=58), _rand(10, 8, seed=59), 1.0, chunks=[4, 4])


def test_mla_rejects_a_combine_order_that_drops_a_chunk():
    """A chunk missing from `order` leaves both the numerator and the
    denominator, so the output stays finite and normalized-looking and the
    only sign anything went wrong is the answer."""
    q, keys = _rand(2, 8, seed=88), _rand(8, 8, seed=89)
    with pytest.raises(AssertionError, match="permutation"):
        R.mla_decode(q, keys, 1.0, chunks=[4, 4], order=[0])


# ---------------------------------------------------------------- MoE route
def test_route_weights_sum_to_the_scaling_factor():
    ids, weights = R.moe_route(_rand(6, 64, seed=60, scale=2.0), topk=6, scale=1.5)
    assert ids.shape == (6, 6)
    _close(weights.sum(-1), torch.full((6,), 1.5, dtype=F64))


def test_route_without_renormalizing_leaves_the_scores_alone():
    logits = _rand(6, 64, seed=61, scale=2.0)
    ids, weights = R.moe_route(logits, topk=6, renormalize=False)
    _close(weights, R.sqrt_softplus_score(logits).gather(-1, ids))


def test_route_picks_the_highest_scores():
    """Monotone in the logit, so the picks are the largest logits when there
    is no bias to move them."""
    logits = _rand(4, 32, seed=62, scale=2.0)
    ids, _ = R.moe_route(logits, topk=5)
    assert torch.equal(ids.sort(-1).values, logits.topk(5, -1).indices.sort(-1).values)


def test_route_breaks_ties_toward_the_lower_expert_id():
    """Equal logits are reachable -- a shared bias, a saturated softplus --
    so which experts win has to be defined, not left to a sort's stability."""
    ids, weights = R.moe_route(torch.zeros(2, 8, dtype=F64), topk=3, scale=1.5)
    assert torch.equal(ids, torch.tensor([[0, 1, 2], [0, 1, 2]]))
    _close(weights, torch.full((2, 3), 0.5, dtype=F64))


def test_route_bias_moves_the_picks_but_not_the_weights():
    """`e_score_correction_bias` is a load-balancing knob: it chooses the
    experts and then leaves the arithmetic, so the weights are the unbiased
    scores of whatever it chose.

    The winners have to carry a bias that is not zero, or the biased and
    unbiased readings agree by construction and the test sees nothing -- so
    the second assertion is that they would have differed.
    """
    logits = _rand(4, 8, seed=63, scale=2.0)
    scores = R.sqrt_softplus_score(logits)
    bias = torch.where(torch.arange(8) < 4, -100.0, 3.0).double()
    ids, weights = R.moe_route(logits, bias=bias, topk=4, renormalize=False)
    assert torch.all(ids >= 4)
    _close(weights, scores.gather(-1, ids))
    assert not torch.allclose(weights, (scores + bias).gather(-1, ids))


def test_route_over_every_expert_reduces_to_the_dense_weights():
    """Top-k with k the whole expert count is not a route at all: it is the
    scores, normalized."""
    logits = _rand(4, 8, seed=64, scale=2.0)
    scores = R.sqrt_softplus_score(logits)
    ids, weights = R.moe_route(logits, topk=8, scale=1.5)
    dense = (scores / scores.sum(-1, keepdim=True) * 1.5).gather(-1, ids)
    _close(weights, dense, scale=16.0)


def test_route_rejects_a_topk_past_the_experts():
    with pytest.raises(AssertionError, match="topk"):
        R.moe_route(_rand(2, 8, seed=65), topk=9)


# ---------------------------------------------------------------- MoE expert
@pytest.mark.parametrize("gate", [-30.0, -1.0, 0.0, 1.0, 9.999, 10.0, 10.5, 1e4])
def test_swiglu_matches_the_clamped_formula(gate: float):
    """Including at the limit and past it, where the clamp is what the test
    is about."""
    g = torch.full((4,), gate, dtype=F64)
    up = torch.tensor([-1e4, -3.0, 2.0, 1e4], dtype=F64)
    capped = min(gate, 10.0)
    want = capped / (1.0 + math.exp(-capped)) * up.clamp(-10.0, 10.0)
    _close(R.swiglu_clamp(g, up, 10.0), want, scale=16.0)


def test_swiglu_is_flat_in_the_gate_past_the_limit():
    """The gate clamp is above only, so everything past it is one value."""
    up = _rand(4, seed=66)
    at = R.swiglu_clamp(torch.full((4,), 10.0, dtype=F64), up, 10.0)
    for gate in (10.0 + 2.0**-20, 11.0, 1e9):
        assert torch.equal(
            R.swiglu_clamp(torch.full((4,), gate, dtype=F64), up, 10.0), at
        )


def test_swiglu_clamps_the_up_projection_both_ways():
    gate = _rand(4, seed=67)
    inside = R.swiglu_clamp(gate, torch.full((4,), 10.0, dtype=F64), 10.0)
    assert torch.equal(
        R.swiglu_clamp(gate, torch.full((4,), 1e9, dtype=F64), 10.0), inside
    )
    assert torch.equal(
        R.swiglu_clamp(gate, torch.full((4,), -1e9, dtype=F64), 10.0), -inside
    )


def test_moe_expert_is_the_two_projections_around_the_swiglu():
    x = _rand(3, 16, seed=68)
    w_gate, w_up, w_down = (
        _rand(8, 16, seed=69),
        _rand(8, 16, seed=70),
        _rand(16, 8, seed=71),
    )
    want = R.swiglu_clamp(x @ w_gate.t(), x @ w_up.t(), 10.0) @ w_down.t()
    _close(R.moe_expert(x, w_gate, w_up, w_down, 10.0), want, scale=64.0)


def test_moe_combine_weights_each_expert_and_sums_in_order():
    outs, weights = _rand(3, 6, 16, seed=72), _rand(3, 6, seed=73)
    want = (outs * weights.unsqueeze(-1)).sum(-2)
    _close(R.moe_combine(outs, weights), want, scale=16.0)
    _close(R.moe_combine(outs, weights, order=[5, 4, 3, 2, 1, 0]), want, scale=16.0)


def test_moe_combine_over_every_expert_is_the_dense_mixture():
    """Routing to all of them, renormalized, is a convex mixture, so a
    constant expert output comes back unchanged."""
    logits = _rand(3, 8, seed=74, scale=2.0)
    _, weights = R.moe_route(logits, topk=8)
    out = torch.ones(3, 8, 16, dtype=F64)
    _close(R.moe_combine(out, weights), torch.ones(3, 16, dtype=F64), scale=16.0)


# ---------------------------------------------------------------- operands
def test_unshuffle_weight_inverts_the_documented_shuffle():
    """AITER's `shuffle_weight(w, layout=(16, 16))` on distinct values: every
    byte has to come back where it started."""
    n, k = 64, 128
    logical = torch.arange(n * k, dtype=torch.int32).view(n, k)
    shuffled = logical.view(n // 16, 16, k // 16, 16).permute(0, 2, 1, 3).reshape(n, k)
    assert torch.equal(R.unshuffle_weight(shuffled), logical)


def test_unshuffle_weight_keeps_leading_dimensions():
    """A stack of experts' weights unshuffles one expert at a time."""
    w = torch.arange(3 * 32 * 64, dtype=torch.int32).view(3, 32, 64)
    per_expert = torch.stack([R.unshuffle_weight(e) for e in w])
    assert torch.equal(R.unshuffle_weight(w), per_expert)


def test_unshuffle_scale_inverts_the_documented_shuffle():
    """`view(rows / 32, 2, 16, cols / 8, 2, 4).permute(0, 3, 5, 2, 4, 1)`."""
    rows, cols = 64, 256
    logical = torch.arange(rows * cols, dtype=torch.int32).view(rows, cols)
    shuffled = logical.view(rows // 32, 2, 16, cols // 8, 2, 4)
    shuffled = shuffled.permute(0, 3, 5, 2, 4, 1).reshape(rows, cols)
    assert torch.equal(R.unshuffle_scale(shuffled), logical)


def test_unshuffle_scale_puts_a_32_row_block_in_one_run():
    """What the layout is for: one 32-row block's codes -- both 16-row
    halves, all of K -- are one contiguous run of stored bytes, so a few wide
    loads fetch it. Filling the first half of the buffer with one code and
    the second half with another therefore has to unshuffle into the first
    32 rows and the last 32 rows, with no mixing."""
    rows, cols = 64, 64
    half = rows * cols // 2
    stored = torch.cat([torch.zeros(half), torch.ones(half)])
    logical = R.unshuffle_scale(stored.view(rows, cols).to(torch.int32))
    assert torch.equal(logical[:32], torch.zeros(32, cols, dtype=torch.int32))
    assert torch.equal(logical[32:], torch.ones(32, cols, dtype=torch.int32))


@pytest.mark.parametrize("rows, cols", [(16, 8), (17, 16), (32, 24)])
def test_unshuffle_weight_rejects_a_partial_block(rows: int, cols: int):
    with pytest.raises(AssertionError, match="whole blocks"):
        R.unshuffle_weight(torch.zeros(rows, cols, dtype=torch.int32))


def test_unshuffle_scale_rejects_a_partial_block():
    with pytest.raises(AssertionError, match="shuffle_scale"):
        R.unshuffle_scale(torch.zeros(48, 16, dtype=torch.int32))


def test_mx_quantize_rejects_an_fmax_past_the_format():
    """The scale code aims at `fmax` and the rounding saturates at the E4M3
    maximum. Aiming past it clips up to half of every group's range and
    still returns numbers, so the pair is checked rather than trusted."""
    with pytest.raises(AssertionError, match="at most 224"):
        R.mx_quantize(_rand(1, 32, seed=87), 448.0, fnuz=True)


def test_unshuffled_mxfp4_dequantizes_to_the_logical_weight():
    """The two together: a weight stored shuffled and quantized comes back as
    the float64 values its logical layout stands for."""
    n, k = 32, 128
    codes = torch.randint(0, 256, (n, k // 2), generator=_gen(75), dtype=torch.uint8)
    scales = torch.randint(
        100, 150, (n, k // 32), generator=_gen(76), dtype=torch.int32
    )
    shuffled = codes.view(n // 16, 16, (k // 2) // 16, 16).permute(0, 2, 1, 3)
    want = R.dequant_mxfp4(codes, scales)
    got = R.dequant_mxfp4(R.unshuffle_weight(shuffled.reshape(n, k // 2)), scales)
    assert torch.equal(got, want)


def test_mx_quantize_is_idempotent():
    """Its output is by construction on the grid it rounds to, so a second
    pass through it must not move anything."""
    once = R.mx_quantize(_rand(4, 64, seed=77, scale=20.0), 448.0)
    assert torch.equal(R.mx_quantize(once, 448.0), once)


def test_mx_quantize_shares_one_scale_across_a_group():
    """A group's values are all divided by the same power of two, so a value
    next to a large one in its group loses precision and the same value in a
    group of its own does not."""
    x = torch.full((1, 64), 2.0**-8, dtype=F64)
    x[0, 0] = 1000.0
    alone = R.mx_quantize(x, 448.0, group=64)
    split = R.mx_quantize(x, 448.0, group=32)
    assert alone[0, 1].item() != x[0, 1].item()
    assert split[0, 32].item() == x[0, 32].item()


def test_mx_quantize_rejects_a_partial_group():
    with pytest.raises(AssertionError, match="groups"):
        R.mx_quantize(_rand(1, 48, seed=78), 448.0, group=32)
