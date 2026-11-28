# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

current_platform.import_kernels()

VOCAB_SIZES = [32000, 49152, 128256]
BATCH_SIZES = [1, 4, 16]
PLACEHOLDER = -1


def reference_greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1).view(-1)


class TestGreedyArgmax:
    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_exact_match(self, vocab_size: int, batch_size: int):
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        expected = reference_greedy_sample(logits)
        result = torch.ops._C.greedy_argmax(logits)
        torch.testing.assert_close(result, expected)

    def test_single_dominant(self):
        logits = torch.full((1, 50000), -1e9, dtype=torch.float32)
        logits[0, 42] = 100.0
        assert torch.ops._C.greedy_argmax(logits).item() == 42

    def test_negative_logits(self):
        logits = torch.randn(8, 32000, dtype=torch.float32) - 10.0
        expected = reference_greedy_sample(logits)
        result = torch.ops._C.greedy_argmax(logits)
        torch.testing.assert_close(result, expected)


class TestFusedGumbelArgmax:
    def test_distribution_chi_squared(self):
        """Verify sampling distribution via chi-squared goodness of fit."""
        # Small support is enough for chi-squared power; large-vocab bias
        # is covered by test_noise_not_sliding_window (#59786).
        small_vocab = 100
        logits = torch.randn(1, small_vocab, dtype=torch.float32)
        probs = logits.softmax(dim=-1).squeeze(0)

        n_samples = 100_000
        seeds = torch.arange(n_samples, dtype=torch.long) * 7 + 13
        idx = torch.ops._C.fused_gumbel_argmax(
            logits.expand(n_samples, -1).contiguous(), seeds
        )
        counts = torch.bincount(idx, minlength=small_vocab).float()

        expected = probs * n_samples
        mask = expected > 5
        chi2 = ((counts[mask] - expected[mask]) ** 2 / expected[mask]).sum()
        dof = mask.sum().item() - 1
        from scipy.stats import chi2 as chi2_dist

        p_value = 1.0 - chi2_dist.cdf(chi2.item(), dof)
        assert p_value > 0.001, (
            f"Chi-squared test failed: chi2={chi2.item():.1f}, "
            f"dof={dof}, p={p_value:.6f}"
        )

    def test_noise_not_sliding_window(self):
        """Table[(seed+i)&MASK] made consecutive seeds a 1-token shift.

        On uniform logits that yields P(w(s+1) == w(s)-1) ~ 1-2/V. Independent
        per-token noise yields ~1/V. See #59786.
        """
        vocab, n = 128, 2048
        logits = torch.zeros(n, vocab, dtype=torch.float32)
        seeds = torch.arange(n, dtype=torch.long)
        winners = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        shifted = (winners[:-1] - 1) % vocab
        frac = (winners[1:] == shifted).float().mean().item()
        assert frac < 0.05, (
            f"consecutive-seed winners follow a table shift (frac={frac:.3f})"
        )

    def test_zipf_head_matches_softmax(self):
        """Zipf head probability should match softmax (#59786)."""
        vocab, n_samples, batch = 8192, 16384, 256
        logits = -1.1 * torch.arange(1, vocab + 1, dtype=torch.float32).log()
        p0 = torch.softmax(logits.double(), dim=-1)[0].item()
        row = logits.view(1, -1).expand(batch, vocab).contiguous()
        n_top = 0
        for start in range(0, n_samples, batch):
            seeds = torch.arange(start, start + batch, dtype=torch.long)
            tokens = torch.ops._C.fused_gumbel_argmax(row, seeds)
            n_top += int((tokens == 0).sum())
        q0 = n_top / n_samples
        se = (p0 * (1.0 - p0) / n_samples) ** 0.5
        assert abs(q0 - p0) < 8.0 * se, (
            f"P(top-1) kernel={q0:.4f} exact={p0:.4f} se={se:.4f}"
        )

    def test_deterministic_same_seed(self):
        """Same seed produces same result."""
        logits = torch.randn(4, 32000, dtype=torch.float32)
        seeds = torch.tensor([42, 123, 456, 789], dtype=torch.long)
        r1 = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        r2 = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        torch.testing.assert_close(r1, r2)

    def test_different_seeds_differ(self):
        logits = torch.zeros(16, 50000, dtype=torch.float32)
        seeds_a = torch.arange(16, dtype=torch.long)
        seeds_b = torch.arange(16, dtype=torch.long) + 1_000_000
        r_a = torch.ops._C.fused_gumbel_argmax(logits, seeds_a)
        r_b = torch.ops._C.fused_gumbel_argmax(logits, seeds_b)
        assert not torch.equal(r_a, r_b)

    @pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_output_in_range(self, vocab_size: int, batch_size: int):
        logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
        seeds = torch.arange(batch_size, dtype=torch.long)
        result = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        assert result.min() >= 0
        assert result.max() < vocab_size

    def test_samples_only_from_finite_logits(self):
        """top-k/top-p mask with -inf; those IDs must never win."""
        allowed = torch.tensor([3, 17, 99, 1024, 8191], dtype=torch.long)
        logits = torch.full((64, 32000), float("-inf"), dtype=torch.float32)
        logits[:, allowed] = torch.randn(64, allowed.numel())
        tokens = torch.ops._C.fused_gumbel_argmax(
            logits, torch.arange(64, dtype=torch.long)
        )
        assert torch.isin(tokens, allowed).all()

    def test_single_unmasked_token(self):
        logits = torch.full((8, 4096), float("-inf"), dtype=torch.float32)
        logits[:, 123] = 0.0
        seeds = torch.tensor([0, -1, -(2**63), 2**63 - 1, 42, 99, 1000, 7])
        tokens = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        assert (tokens == 123).all()

    def test_tokens_beyond_gumbel_max_cannot_win(self):
        """A gap larger than GUMBEL_MAX is unreachable."""
        logits = torch.zeros(256, 4, dtype=torch.float32)
        logits[:, 1:] = -80.0
        tokens = torch.ops._C.fused_gumbel_argmax(
            logits, torch.arange(256, dtype=torch.long)
        )
        assert (tokens == 0).all()

    def test_tied_huge_logits_not_index_biased(self):
        """Noise must still break ties when logits sit at float32 max."""
        logits = torch.full(
            (4096, 2), torch.finfo(torch.float32).max, dtype=torch.float32
        )
        tokens = torch.ops._C.fused_gumbel_argmax(
            logits, torch.arange(4096, dtype=torch.long)
        )
        frac0 = (tokens == 0).float().mean().item()
        assert set(tokens.tolist()) == {0, 1}, tokens.unique().tolist()
        assert 0.35 < frac0 < 0.65, f"P(index 0)={frac0:.3f} expected ~0.5"

    def test_serial_batch1_matches_batched(self):
        """OpenMP rows must match the batch=1 serial path."""
        logits = torch.randn(8, 2048, dtype=torch.float32)
        seeds = torch.arange(8, dtype=torch.long)
        batched = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        serial = torch.stack(
            [
                torch.ops._C.fused_gumbel_argmax(logits[i : i + 1], seeds[i : i + 1])[0]
                for i in range(8)
            ]
        )
        torch.testing.assert_close(batched, serial)

    def test_noncontiguous_matches_contiguous(self):
        logits = torch.randn(8, 1024, dtype=torch.float32)[:, ::2]
        seeds = torch.arange(16, dtype=torch.long)[::2]
        assert not logits.is_contiguous() and not seeds.is_contiguous()
        expected = torch.ops._C.fused_gumbel_argmax(
            logits.contiguous(), seeds.contiguous()
        )
        actual = torch.ops._C.fused_gumbel_argmax(logits, seeds)
        torch.testing.assert_close(actual, expected)

    def test_empty_batch(self):
        tokens = torch.ops._C.fused_gumbel_argmax(
            torch.empty(0, 7), torch.empty(0, dtype=torch.long)
        )
        assert tokens.shape == (0,) and tokens.dtype == torch.int64

    def test_masked_support_matches_softmax(self):
        """Finite subset after -inf mask should match softmax."""
        from scipy.stats import chisquare

        logits = torch.tensor(
            [0.0, -0.2, -0.7, -1.5, -3.0, float("-inf")], dtype=torch.float32
        )
        n_samples = 20_000
        tokens = torch.ops._C.fused_gumbel_argmax(
            logits.expand(n_samples, -1).contiguous(),
            torch.arange(n_samples, dtype=torch.long),
        )
        counts = torch.bincount(tokens, minlength=logits.numel()).double()
        assert counts[-1] == 0
        probs = logits.double().softmax(-1)
        _, p_value = chisquare(counts[:-1].numpy(), (probs[:-1] * n_samples).numpy())
        assert p_value > 1e-7, f"masked-support p-value: {p_value}"

    def test_zipf_tail_matches_softmax(self):
        """Independent 53-bit Gumbel must not starve the Zipf tail (#59786)."""
        vocab, n_samples, batch = 8192, 16384, 256
        logits = -1.1 * torch.arange(1, vocab + 1, dtype=torch.float32).log()
        p_tail = torch.softmax(logits.double(), dim=-1)[vocab // 2 :].sum().item()
        row = logits.view(1, -1).expand(batch, vocab).contiguous()
        n_tail = 0
        for start in range(0, n_samples, batch):
            seeds = torch.arange(start, start + batch, dtype=torch.long)
            tokens = torch.ops._C.fused_gumbel_argmax(row, seeds)
            n_tail += int((tokens >= vocab // 2).sum())
        q_tail = n_tail / n_samples
        se = (p_tail * (1.0 - p_tail) / n_samples) ** 0.5
        assert abs(q_tail - p_tail) < 8.0 * se, (
            f"P(tail) kernel={q_tail:.4f} exact={p_tail:.4f} se={se:.4f}"
        )


class TestMemory:
    def test_fused_no_intermediate_allocs(self):
        """Verify that the fused kernel does not allocate large intermediates."""
        logits = torch.randn(16, 128256, dtype=torch.float32)
        seeds = torch.arange(16, dtype=torch.long)

        torch.ops._C.fused_gumbel_argmax(logits, seeds)

        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()
        for _ in range(50):
            torch.ops._C.fused_gumbel_argmax(logits, seeds)
        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snap_after.compare_to(snap_before, "lineno")
        total_new_bytes = sum(s.size_diff for s in diff if s.size_diff > 0)
        vocab_bytes = 16 * 128256 * 4
        assert total_new_bytes < vocab_bytes, (
            f"Fused kernel allocated {total_new_bytes} bytes, "
            f"expected < {vocab_bytes} (one intermediate tensor)"
        )


@pytest.mark.parametrize("with_draft_probs", [False, True])
def test_placeholder_draft_is_rejected(with_draft_probs: bool):
    # Request 0 is a normal decode whose second row puts all mass on the last
    # vocab id. Request 1 is padded with placeholder drafts: its first row is
    # the real row (only token 3 allowed), its second row is padding.
    # Reading target_probs at a placeholder id would hit the previous row's
    # last entry (1.0) and accept the placeholder.
    vocab = 16
    target = torch.zeros(4, vocab)
    target[0, 5] = 1.0
    target[1, vocab - 1] = 1.0
    target[2, 3] = 1.0
    target[3, 7] = 1.0
    draft_ids = torch.tensor([5, 5, PLACEHOLDER, PLACEHOLDER])
    cu = torch.tensor([2, 4])
    recovered = torch.tensor([5, vocab - 1, 3, 7])
    bonus = torch.tensor([[9], [11]])
    uniform = torch.full((4,), 0.5, dtype=torch.float32)
    draft_probs = None
    if with_draft_probs:
        # Request 0's second row also gives the last vocab id draft mass, so
        # a misread placeholder would see p / q = 2 and be accepted.
        draft_probs = torch.zeros(4, vocab)
        draft_probs[0, 5] = 1.0
        draft_probs[1, 5] = draft_probs[1, vocab - 1] = 0.5
        draft_probs[2:, 0] = 1.0

    out = torch.full((2, 3), PLACEHOLDER, dtype=torch.int64)
    torch.ops._C.rejection_random_sample_kernel_impl(
        out,
        cu,
        draft_ids,
        draft_probs,
        target,
        bonus,
        recovered,
        uniform,
        None,
        2,
        vocab,
        no_draft_probs=draft_probs is None,
    )

    assert out.tolist() == [[5, vocab - 1, PLACEHOLDER], [3, PLACEHOLDER, PLACEHOLDER]]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _assert_exact_topk_topp(
    src: torch.Tensor,
    out: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    tol: float = 1e-7,
):
    """Assert `out` is the exact top-k/top-p masking of `src`.

    Order is logit descending, lower index first on ties; NaN counts as -inf
    and -0.0 equals +0.0. Top-k keeps `min(k, #logits > -inf)`. Top-p keeps the
    shortest prefix of the top-k set whose FP64 softmax mass reaches p (within
    `tol`); the n +inf logits of a set share all its mass, so the first
    `ceil(p * n)` of them are kept. Kept logits keep their bits, the others
    become -inf, and rows with nothing to mask stay bit-identical.
    """
    ninf = float("-inf")
    batch, vocab = src.shape
    src_bits, out_bits = _bits(src), _bits(out)
    clean = src.masked_fill(src.isnan(), ninf) + 0.0
    finite = (clean > ninf).sum(-1)
    kk = torch.full((batch,), vocab) if k is None else k.long()
    pp = torch.ones(batch, dtype=torch.float64) if p is None else p.double()
    budget = torch.minimum(kk, finite)

    untouched = (kk >= finite) & (pp >= 1)
    assert torch.equal(src_bits[untouched], out_bits[untouched]), "row changed"
    empty = (finite == 0) & ~untouched
    same = src_bits[empty] == out_bits[empty]
    nan_cleared = src[empty].isnan() & (out[empty] == ninf)
    assert (same | nan_cleared).all(), "row without finite logits changed"

    rows = torch.nonzero(~untouched & (finite > 0)).flatten()
    if rows.numel() == 0:
        return
    s, o, bud, pr = clean[rows], out[rows], budget[rows], pp[rows]
    kept = o > ninf
    neg_inf_bits = _bits(torch.tensor([ninf]))
    assert (kept | (out_bits[rows] == neg_inf_bits)).all(), "masked logit not -inf"
    assert (~kept | (out_bits[rows] == src_bits[rows])).all(), "kept logit changed"
    order = s.argsort(dim=-1, descending=True, stable=True)
    count = kept.sum(-1)
    ranks = torch.arange(vocab)
    assert torch.equal(kept.gather(-1, order), ranks < count[:, None]), (
        "kept set is not a prefix of the order"
    )

    lo, hi = bud.clone(), bud.clone()
    by_p = torch.nonzero(pr < 1).flatten()
    if by_p.numel() > 0:
        vals = s[by_p].gather(-1, order[by_p]).double()
        vals = vals.masked_fill(ranks >= bud[by_p, None], ninf)
        posinf = vals.isposinf()
        n_posinf = posinf.sum(-1, keepdim=True)
        cum = torch.where(
            n_posinf > 0,
            posinf.double() / n_posinf.clamp_min(1),
            vals.masked_fill(posinf, 0).softmax(-1),
        ).cumsum(-1)
        target = pr[by_p, None]
        reach = cum >= target - tol
        first = torch.where(reach.any(-1), reach.int().argmax(-1) + 1, bud[by_p])
        before = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], -1)
        last = ((before <= target + tol) & (ranks < bud[by_p, None])).sum(-1)
        lo[by_p] = torch.minimum(first, bud[by_p])
        hi[by_p] = torch.maximum(torch.minimum(last, bud[by_p]), lo[by_p])
    bad = torch.nonzero((count < lo) | (count > hi)).flatten()
    assert bad.numel() == 0, (
        f"kept {count[bad].tolist()} outside [{lo[bad].tolist()}, "
        f"{hi[bad].tolist()}] in rows {rows[bad].tolist()}"
    )


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "cpu_topk_topp_exact"),
    reason="Requires the x86 CPU build of cpu_topk_topp_exact",
)
class TestCpuExactTopkTopp:
    """Exact top-k/top-p masking by `torch.ops._C.cpu_topk_topp_exact`.

    The cases come from the stress suites the kernel was developed against:
    ties and plateaus at the cut, NaN and +-inf, extreme magnitudes, k and
    nucleus sizes around the kernel's candidate buffers, rows whose sampled
    positions avoid the cut, top-p within FP32 ulps of 1, strided inputs and
    workspace reuse across shapes.
    """

    VOCAB = 151936
    # Per-row k and p around the kernel's internal sizes and limits.
    MIXED_K = [1, 2, 20, 255, 256, 257, 16385, 32769, VOCAB - 1, VOCAB, VOCAB + 5]
    MIXED_P = [1e-9, 0.1, 0.5, 0.9, 0.99, 1 - 1e-7, 1.0, 2.0]

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.set_default_device("cpu")
        self.generator = torch.Generator(device="cpu").manual_seed(42)

    def _run(self, logits, k=None, p=None):
        k = None if k is None else torch.tensor(k, dtype=torch.int32)
        p = None if p is None else torch.tensor(p, dtype=torch.float32)
        out = logits.clone()
        torch.ops._C.cpu_topk_topp_exact(out, k, p)
        _assert_exact_topk_topp(logits, out, k, p)
        return out

    def _per_row(self, values, batch):
        idx = torch.randint(0, len(values), (batch,), generator=self.generator)
        return [values[i] for i in idx]

    def _logits(self, kind: str, batch: int, vocab: int) -> torch.Tensor:
        g = self.generator
        if kind == "normal":
            return torch.randn(batch, vocab, generator=g)
        if kind == "zipf":
            base = -1.1 * torch.arange(1, vocab + 1, dtype=torch.float32).log()
            return base[torch.rand(batch, vocab, generator=g).argsort(dim=-1)]
        if kind == "steps":  # ties at every level, so at the cut
            return (torch.randn(batch, vocab, generator=g) * 2).round() / 2
        if kind == "plateau":
            return torch.full((batch, vocab), 1.5)
        if kind == "tiny":  # few distinct floats: huge tie groups
            return 3.0 + torch.randn(batch, vocab, generator=g) * 1e-6
        if kind == "flat":  # distinct logits of nearly equal weight
            return torch.randn(batch, vocab, generator=g) * 1e-6
        if kind == "signed_zero":  # the top is a plateau of +0.0 and -0.0
            x = -torch.rand(batch, vocab, generator=g) - 0.01
            z = torch.rand(batch, vocab, generator=g)
            x[z < 0.02] = 0.0
            x[z > 0.98] = -0.0
            return x
        if kind == "bf16_temperature":  # bf16 logits / T, as the sampler does
            x = (torch.randn(batch, vocab, generator=g) * 3).bfloat16().float()
            return x / 0.7
        if kind == "denormal":
            return torch.randn(batch, vocab, generator=g) * 1e-39
        if kind == "huge":
            return torch.randn(batch, vocab, generator=g) * 1e30
        if kind == "near_max":  # +-FLT_MAX: x - max overflows
            x = 3.4e38 * (1 - torch.rand(batch, vocab, generator=g) * 1e-3)
            return torch.where(torch.rand(batch, vocab, generator=g) < 0.5, -x, x)
        if kind == "dominant_tail":  # 4 logits hold nearly all the mass
            x = torch.randn(batch, vocab, generator=g)
            x[:, -4:] = 18.0
            return x
        if kind == "nan_top":  # NaN where the best logits would be
            x = torch.randn(batch, vocab, generator=g)
            x.scatter_(-1, x.topk(300, -1).indices[:, ::2], float("nan"))
            return x
        if kind == "nan_neginf":  # 1% finite, the rest NaN or -inf
            x = torch.randn(batch, vocab, generator=g)
            z = torch.rand(batch, vocab, generator=g)
            x[z < 0.495] = float("nan")
            x[(z >= 0.495) & (z < 0.99)] = float("-inf")
            return x
        raise ValueError(kind)

    @pytest.mark.parametrize(
        "kind",
        [
            "normal",
            "zipf",
            "steps",
            "plateau",
            "tiny",
            "flat",
            "signed_zero",
            "bf16_temperature",
            "denormal",
            "huge",
            "near_max",
            "dominant_tail",
            "nan_top",
            "nan_neginf",
        ],
    )
    @pytest.mark.parametrize(
        "k,p",
        [(1, None), (257, 0.99), (None, 0.9), (None, 1 - 1e-7), ("mixed", "mixed")],
    )
    def test_distributions(self, kind, k, p):
        batch = 5
        logits = self._logits(kind, batch, self.VOCAB)
        if k is not None:
            k = self._per_row(self.MIXED_K, batch) if k == "mixed" else [k] * batch
        if p is not None:
            p = self._per_row(self.MIXED_P, batch) if p == "mixed" else [p] * batch
        self._run(logits, k, p)

    @pytest.mark.parametrize("kind", ["normal", "steps", "plateau"])
    @pytest.mark.parametrize("p", [None, 0.9])
    def test_k_around_buffer_sizes(self, kind, p):
        """Top-k around the exact finish size (256) and the candidate buffer
        (32768)."""
        k = [147, 148, 255, 256, 257, 16383, 16385, 32767, 32768, 32769, 40000]
        logits = self._logits(kind, len(k), self.VOCAB)
        self._run(logits, k, None if p is None else [p] * len(k))

    @pytest.mark.parametrize("batch", [1, 40])
    @pytest.mark.parametrize("vocab", [151936, 262143])
    def test_nucleus_around_buffer_sizes(self, batch, vocab):
        """Nearly uniform rows with a top-p nucleus of n logits, n around the
        exact finish size and the candidate buffer, up to the full-row path.
        A single row is split over the threads."""
        sizes = [255, 257, 16383, 16385, 32767, 32769, 60000]
        for nuclei in [[n] for n in sizes] if batch == 1 else [None]:
            nuclei = nuclei or self._per_row(sizes, batch)
            logits = self._logits("flat", batch, vocab)
            self._run(logits, None, [n / vocab for n in nuclei])

    @pytest.mark.parametrize(
        "kind,shift",
        [("zipf", 0.0), ("zipf", 3.3), ("zipf", -1.7), ("bf16_temperature", 0.0)],
    )
    @pytest.mark.parametrize("p", [0.99999994, 0.99999988, 0.9999997])
    def test_top_p_within_ulps_of_one(self, kind, shift, p):
        """The target mass lies within a few FP32 ulps of the row mass, which
        must sum the same rounded weights `exp(x - max)` as the cut (a shifted
        max makes `x - max` round)."""
        logits = self._logits(kind, 16, self.VOCAB) + shift
        self._run(logits, None, [p] * 16)
        self._run(logits, [self.VOCAB - 1] * 16, [p] * 16)

    @staticmethod
    def _sampled_positions(vocab: int) -> torch.Tensor:
        """The row positions the narrowing samples (`sample_pos` in
        csrc/cpu/topk_topp_exact.cpp): 256 pivots in each of its first rounds
        and the 1024-logit bracket sample (round 1024)."""

        def pos(t, rnd):
            h = (t * 0x9E3779B1 + rnd * 0x85EBCA77 + 0x632BE5AB) & 0xFFFFFFFF
            h ^= h >> 15
            h = (h * 0x2C1B3C6D) & 0xFFFFFFFF
            h ^= h >> 12
            return (h * vocab) >> 32

        taken = {pos(t, r) for r in range(64) for t in range(256)}
        taken |= {pos(t, 1024) for t in range(1024)}
        return torch.tensor(sorted(taken))

    @pytest.mark.parametrize("value", [-1e4, 1e4, float("-inf"), float("nan")])
    @pytest.mark.parametrize("k,p", [(40000, None), (None, 0.5), (60000, 0.9)])
    def test_sampled_positions_avoid_the_cut(self, value, k, p):
        """Every logit the narrowing samples lies outside the final bracket, so
        the sampled pivots never split it."""
        vocab = 262143
        logits = self._logits("normal" if p is None else "flat", 1, vocab)
        logits = logits.sort().values
        logits[0, self._sampled_positions(vocab)] = value
        self._run(logits, None if k is None else [k], None if p is None else [p])

    @pytest.mark.parametrize(
        "k,p", [(2, None), (5, 0.9), (None, 0.5), (None, 0.7), (None, 0.99)]
    )
    def test_posinf_logits(self, k, p):
        """+inf logits come first and share all the top-p mass."""
        logits = self._logits("normal", 5, 4097)
        logits[0, 7] = float("inf")
        logits[1, [3, 900, 2000]] = float("inf")
        logits[2, [3, 900]] = float("inf")
        logits[2, 100:200] = float("nan")
        logits[3] = float("inf")
        logits[4, ::3] = float("inf")
        k = None if k is None else [k] * 5
        self._run(logits, k, None if p is None else [p] * 5)

    @pytest.mark.parametrize("fill", [float("-inf"), float("nan")])
    @pytest.mark.parametrize("finite", [1, 50, 300])
    @pytest.mark.parametrize("p", [None, 0.9])
    def test_few_finite_logits(self, fill, finite, p):
        """Rows with a handful of finite logits (structured output masks), next
        to rows without any; k below, at and above the finite count."""
        logits = torch.full((5, self.VOCAB), fill)
        for row in range(3):
            idx = torch.randperm(self.VOCAB, generator=self.generator)[:finite]
            logits[row, idx] = torch.randn(finite, generator=self.generator)
        logits[3] = float("-inf")
        logits[4] = float("nan")
        p = None if p is None else [p] * 5
        self._run(logits, [50] * 5, p)
        self._run(logits, [max(finite - 1, 1), finite, finite + 1, 1, 1], p)
        if p is not None:
            self._run(logits, None, p)

    def test_untouched_rows_keep_their_bits(self):
        logits = self._logits("nan_top", 4, 4097)
        out = self._run(logits, [4097, 5000, 4097, 20], [1.0, 1.0, 2.0, 1.0])
        assert torch.equal(_bits(out[:3]), _bits(logits[:3]))

    @pytest.mark.parametrize("vocab", [1, 17, 256, 257, 4097, 65537, 262143])
    @pytest.mark.parametrize("batch", [1, 3, 33, 64])
    def test_shapes(self, vocab, batch):
        """Vocab sizes off the kernel's 16/256 blocking and below its exact
        finish size, batches around the thread count (a split row pass below
        it)."""
        logits = self._logits("zipf", batch, vocab)
        k = self._per_row([1, 20, 255, 257, max(vocab - 1, 1), vocab], batch)
        self._run(logits, k, self._per_row(self.MIXED_P, batch))
        self._run(logits, None, [0.8] * batch)

    @pytest.mark.parametrize("kind", ["zipf", "flat"])
    def test_vocab_above_2_pow_20(self, kind):
        vocab = (1 << 20) + 1
        logits = self._logits(kind, 2, vocab)
        self._run(logits, [40000, vocab], [0.9, 0.5])
        self._run(logits, None, [0.99, 0.3])

    def test_strided_rows(self):
        """A column slice of every other row of a wider tensor: the view gets
        the result of a contiguous copy, and nothing outside it is written."""
        base = torch.randn(14, 4097 + 8, generator=self.generator)
        view = base[::2, 5 : 5 + 4097]
        k = torch.tensor(self._per_row([1, 50, 4097], 7), dtype=torch.int32)
        p = torch.tensor(self._per_row(self.MIXED_P, 7), dtype=torch.float32)
        expected = base.clone()
        contiguous = view.clone()
        torch.ops._C.cpu_topk_topp_exact(contiguous, k, p)
        expected[::2, 5 : 5 + 4097] = contiguous
        torch.ops._C.cpu_topk_topp_exact(view, k, p)
        assert torch.equal(_bits(base), _bits(expected))

    def test_workspace_reuse_across_shapes(self):
        """Growing and shrinking shapes in one process reuse the workspace."""
        shapes = [(1, 1000), (64, 151936), (2, 257), (128, 32000), (1, 262143)]
        shapes += [(257, 4097), (1, 1048575), (5, 300)]
        for batch, vocab in shapes:
            logits = self._logits("zipf", batch, vocab)
            k = self._per_row([1, 20, 300, vocab], batch)
            self._run(logits, k, self._per_row(self.MIXED_P, batch))

    def test_rejects_k_below_one(self):
        logits = torch.randn(2, 1000, generator=self.generator)
        with pytest.raises(RuntimeError, match="k must be >= 1"):
            torch.ops._C.cpu_topk_topp_exact(
                logits, torch.tensor([1, 0], dtype=torch.int32), None
            )

    def test_rejects_strided_columns(self):
        logits = torch.randn(2, 2000, generator=self.generator)[:, ::2]
        with pytest.raises(RuntimeError, match="unit column stride"):
            torch.ops._C.cpu_topk_topp_exact(
                logits, torch.tensor([5, 5], dtype=torch.int32), None
            )

    @pytest.mark.parametrize(
        "vocab,dtype",
        [(151936, torch.float32), (200, torch.float32), (151936, torch.bfloat16)],
    )
    def test_sampler_dispatch(self, vocab, dtype):
        """`apply_top_k_top_p` masks x86 logits with this kernel: FP32 logits in
        place, other dtypes in a returned FP32 copy."""
        logits = self._logits("normal", 4, vocab).to(dtype)
        k = torch.tensor([1, 20, vocab, 50], dtype=torch.int32)
        p = torch.tensor([0.9, 0.5, 0.8, 1.0])
        expected = logits.to(torch.float32, copy=True)
        torch.ops._C.cpu_topk_topp_exact(expected, k, p)
        actual = apply_top_k_top_p(logits.clone(), k, p)
        assert actual.dtype == torch.float32
        assert torch.equal(_bits(actual), _bits(expected))

    def test_concurrent_calls(self):
        """Calls from several threads, which share the op's workspace, each get
        their own exact result."""
        cases = []
        for kind, batch, vocab, k, p in [
            ("flat", 1, 262143, None, 0.8),  # split row pass
            ("zipf", 64, 151936, 20, 0.9),
            ("normal", 3, 4097, 50, None),
            ("steps", 8, 65537, 300, 0.5),
        ]:
            logits = self._logits(kind, batch, vocab)
            k = None if k is None else torch.full((batch,), k, dtype=torch.int32)
            p = None if p is None else torch.full((batch,), p)
            expected = logits.clone()
            torch.ops._C.cpu_topk_topp_exact(expected, k, p)
            cases.append((logits, k, p, expected))

        start = threading.Barrier(2 * len(cases))

        def run(case):
            logits, k, p, expected = case
            start.wait()
            for _ in range(20):
                out = logits.clone()
                torch.ops._C.cpu_topk_topp_exact(out, k, p)
                if not torch.equal(_bits(out), _bits(expected)):
                    return False
            return True

        with ThreadPoolExecutor(2 * len(cases)) as pool:
            assert all(pool.map(run, cases * 2))
