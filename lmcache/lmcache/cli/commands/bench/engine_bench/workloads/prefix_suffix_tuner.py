# SPDX-License-Identifier: Apache-2.0
"""Prefix-suffix tuner workload for ``lmcache bench engine``.

Exercises the tiered KV-cache hierarchy (L0 HBM / L1 DRAM / L2 disk) with a
single sequential workload that can be run unchanged across three baselines:

  Baseline 1 — vanilla vLLM (L0 only).  ``--kv-cache-volume`` set to L0 size,
      ``--psf-thrash`` slightly > 1 forces every pass-2 request to miss L0.

  Baseline 2 — vLLM + LMCache L1 + L2.  ``--kv-cache-volume`` set to L1 size,
      so pass-2 requests miss L1 and hit L2 (prefix only — vanilla prefix
      caching cannot reuse the suffix because it sits behind a random
      breaker).

  Baseline 3 — vLLM + LMCache L1 + L2 + CacheBlend.  Same sizing as Baseline
      2; CacheBlend additionally allows the shared suffix's KV chunks to be
      reused, so both the prefix and the suffix hit cache.

Each request is::

    [prefix_i][random breaker][shared suffix]

with::

  - ``num_prefixes`` distinct prefixes, each starting with a unique ID so
    the prefix hash differs even if the random body collides.
  - A fresh random breaker per request, defeating ordinary prefix caching
    past the prefix boundary and preventing non-CacheBlend reuse of the
    suffix.
  - One shared suffix used by every request.

Two passes run sequentially, one request at a time, in identical order:

  - Engine warmup (single throwaway request, before pass 1): amortizes
    first-request torch.compile / CUDA-graph / connector-handshake cost.
    Stats discarded; the prompt is too small to occupy a chunk-aligned
    region so it does not pollute LMCache hit-rate metrics.
  - Pass 1 (warmup): populates the cache.  Stats discarded.
  - Pass 2 (measured): repeats the same prefix order.  Because LRU evicts
    the next-needed prefix on each pass-2 access, even a 1.05× overflow of
    the targeted tier is enough to ensure every pass-2 request misses that
    tier and falls through to the next one.

Debug Guide
-----------

When per-pass-2 hit rates don't match the analytical model, work through
this checklist in order — the failure modes I hit while debugging this
workload are listed in the order they actually surfaced.

**1. Disable vLLM L0 prefix caching during the measurement run.**
   With ``--enable-prefix-caching`` on vLLM, vLLM serves chunks from its
   own HBM cache without asking LMCache, dropping
   ``lmcache_mp_lookup_requested_tokens_total`` for the chunks vLLM
   already had.  This makes the LMCache hit-rate metric look artificially
   low and obscures the actual L1 behavior.  For *measuring LMCache*,
   start vLLM with ``--no-enable-prefix-caching``.  Re-enable it for end-
   to-end latency runs.

**2. Restart LMCache between every comparison run.**
   Prefixes are deterministic by index (``PREFIX_<8-hex-digits>``).  If a
   previous run stored ``prefix_0``, the next run finds it already in L1
   at pass-1 — pass 1 hits cache instead of cold-storing, biasing the
   pass-2 hit rate upward.  Always restart the LMCache server with a
   fresh L1 between back-to-back runs.

**3. Configure LMCache eviction aggressively for thrash tests.**
   LMCache's L1 eviction is a 1Hz polling background thread that fires
   only when ``usage >= --eviction-trigger-watermark`` and clears
   ``--eviction-ratio`` of contents per cycle.  Defaults
   (``watermark=0.80``, ``ratio=0.20``) only drop usage from 80 % to 60 %
   per fire — leaving 60 % of pass-1 content surviving into pass 2.  For
   tests that expect ``thrash → ~0% hit rate``, start the server with::

       lmcache server --l1-size-gb <SIZE> --eviction-policy LRU \\
           --eviction-trigger-watermark 0.80 \\
           --eviction-ratio 0.99

   The 0.99 ratio means each fire clears nearly the whole cache; combined
   with the workload's built-in 5-second pass-1→pass-2 settle delay, this
   approximates a strict-LRU sequential thrash.

**4. Watch the right metrics.**
   ``GET <lmcache-url>/metrics`` exposes Prometheus counters.  Snapshot
   before and after the run, take the delta::

       lmcache_mp_lookup_requested_tokens_total   # both passes
       lmcache_mp_lookup_hit_tokens_total          # both passes (L1+L2)
       lmcache_mp_l1_eviction_loop_ticks_total     # 1 per loop cycle
       lmcache_mp_l1_eviction_loop_triggered_total # only when above watermark

   ``triggered/ticks`` should be > 0 for any thrash test.  If it's 0, the
   benchmark completed before any eviction fired — see (3).  For blend
   tests, also pull ``lmcache_blend_lookup_{requested,hit}_tokens_total``.

**5. Convert aggregate metrics to per-pass-2 hit rate.**
   The Prometheus counters tally pass-1 *and* pass-2 lookups, so
   ``hit_tokens_total / requested_tokens_total`` is roughly *half* the
   per-pass-2 hit rate (pass 1 contributes cold misses).  The right
   conversion is::

       per_pass2_hit_rate = hit_tokens_total / (requested_tokens_total / 2)

   when pass 1 has ~0 hits (the typical case for a fresh LMCache).  Use
   ``num_prefixes`` from the workload's ``log_config`` output to verify
   the lookup count: ``ticks_total ≈ run_duration_seconds`` and
   ``requested_tokens_total / chunks_per_request / chunk_size_tokens
   == 2 * num_prefixes``.

**6. Check pool sizing matches L1 capacity.**
   The workload sizes ``num_prefixes`` so the pool's *full* per-request
   KV footprint (``num_prefixes * context_length * tokens_per_gb_kvcache``)
   equals ``thrash * 1.05`` GB.  If this doesn't match the L1 size you
   started LMCache with, the test is over- or under-provisioned.  Hit
   ``GET <lmcache-url>/api/status`` and verify
   ``storage_manager.l1_manager.memory_total_bytes / 2**30`` matches your
   ``--psf-thrash`` value.

**7. Per-request CSV breakdown.**
   ``--output-dir <DIR>`` writes ``bench_results.csv`` with TTFT per
   pass-2 request.  Sort by request index — under sequential thrash,
   *early* prefix indices should miss (slow TTFT) and *late* indices
   should hit (fast TTFT).  Uniform TTFT across all requests means the
   pool doesn't actually thrash — usually a sign that pass 1 finished
   before any eviction fired (see step 3 again).
"""

# Standard
from dataclasses import dataclass
import random

# First Party
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.stats import StatsCollector
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload
from lmcache.logging import init_logger

logger = init_logger(__name__)

_BREAKER_TOKENS = 32
_MIN_SUFFIX_TOKENS = 100
_UNIQUE_ID_TOKENS = 4  # rough token count for ``PREFIX_<8-hex-digits>``
_MAX_OUTPUT_TOKENS = 1

# Internal multiplier on ``thrash`` GB to size the prefix pool.  Set to 1.05
# (a 5% overflow) because — with sequential pass-1/pass-2 dispatch and LRU
# eviction — even a 5% overflow is sufficient to evict the next-needed
# prefix on every pass-2 access, ensuring all pass-2 requests miss the
# targeted tier.  See ``docs/design/cli/commands/bench/engine_bench/
# bench-engine.md`` §4.5 for the analysis.
_OVERFLOW_FACTOR = 1.05

# Seconds to sleep between pass 1 (warmup) and pass 2 (measured).  LMCache's
# L1 eviction is a 1Hz polling thread that fires only when usage > watermark
# (default 0.80) — meaning a fast warmup that overflows L1 by a few percent
# may complete before any eviction has actually run.  Without this settle
# delay, pass 2 would find all of pass 1's data still in cache, and the
# user's analytical-model claim "thrash → 0% hit rate" would not hold.
# 5 seconds covers several eviction polls (each evicts ``--eviction-ratio``
# fraction of contents), letting the cache reach steady state under the
# overflow before pass 2 measures.
_EVICTION_SETTLE_SECONDS = 5.0

# Range of token-IDs to sample when generating random body content.  Skip
# the low region (byte fallback / special tokens) and the high tail
# (reserved / added tokens, which are rarely 1-token-clean on round-trip).
# 256–50000 is safe across Llama, Qwen, MiniMax tokenizers.
_TOKEN_ID_LO = 256
_TOKEN_ID_HI = 50000


@dataclass
class PrefixSuffixTunerConfig:
    """Workload-specific config for the prefix-suffix-tuner workload.

    Attributes:
        context_length: Total tokens per request (prefix + breaker + suffix).
        prefix_ratio: Fraction of ``context_length`` allocated to the prefix.
        thrash: Size in GB of the KV-cache tier the workload should overflow
            (L0 / vLLM HBM for Baseline 1; L1 / LMCache DRAM for Baselines
            2 and 3).  The prefix pool is sized to ``thrash *
            _OVERFLOW_FACTOR`` GB internally — i.e., just barely larger than
            the targeted tier — which is sufficient under sequential
            dispatch and LRU to ensure every pass-2 request misses that
            tier.
        num_prefixes: Number of distinct prefixes generated (computed by
            :meth:`resolve` from the cache budget).
        prefix_tokens: Token length of each prefix (computed).
        suffix_tokens: Token length of the shared suffix (computed).
        breaker_tokens: Token length of the random breaker between prefix
            and suffix.
    """

    context_length: int = 8000
    prefix_ratio: float = 0.8
    thrash: float = 20.0
    num_prefixes: int = 1
    prefix_tokens: int = 1
    suffix_tokens: int = 1
    breaker_tokens: int = _BREAKER_TOKENS

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {self.context_length}"
            )
        if not 0.0 < self.prefix_ratio < 1.0:
            raise ValueError(
                f"prefix_ratio must be in (0.0, 1.0), got {self.prefix_ratio}"
            )
        if self.thrash <= 0.0:
            raise ValueError(f"thrash (GB) must be positive, got {self.thrash}")
        if self.num_prefixes < 1:
            raise ValueError(f"num_prefixes must be >= 1, got {self.num_prefixes}")
        if self.prefix_tokens < 1:
            raise ValueError(f"prefix_tokens must be >= 1, got {self.prefix_tokens}")
        if self.suffix_tokens < 1:
            raise ValueError(f"suffix_tokens must be >= 1, got {self.suffix_tokens}")
        if self.breaker_tokens < 1:
            raise ValueError(f"breaker_tokens must be >= 1, got {self.breaker_tokens}")

    @classmethod
    def resolve(
        cls,
        tokens_per_gb_kvcache: int,
        context_length: int = 8000,
        prefix_ratio: float = 0.8,
        thrash: float = 20.0,
        breaker_tokens: int = _BREAKER_TOKENS,
    ) -> "PrefixSuffixTunerConfig":
        """Compute ``num_prefixes`` and token splits from the target tier size.

        ``num_prefixes`` is sized so that the total prefix-pool footprint
        in tokens equals
        ``thrash * _OVERFLOW_FACTOR * tokens_per_gb_kvcache``.  ``thrash``
        is the **size of the targeted KV-cache tier in GB**; the internal
        :data:`_OVERFLOW_FACTOR` (1.05) provides the small overflow needed
        for the LRU invariant to drive every pass-2 access to a miss of
        that tier.

        Args:
            tokens_per_gb_kvcache: Tokens fitting in 1 GB of KV cache for
                the served model (auto-detected from the engine in
                ``parse_args_to_config``; user need not supply directly).
            context_length: Total tokens per request.
            prefix_ratio: Fraction of ``context_length`` allocated to the
                prefix.  Must be strictly between 0 and 1.
            thrash: Size of the targeted KV-cache tier in GB.  Defaults
                to 20.0 GB (typical L0 size for a single H100 with low
                ``--gpu-memory-utilization``).
            breaker_tokens: Token length of the random breaker between
                prefix and suffix.  Defaults to 32.

        Returns:
            A fully-resolved PrefixSuffixTunerConfig.

        Raises:
            ValueError: If ``prefix_ratio`` leaves fewer than
                :data:`_MIN_SUFFIX_TOKENS` for the suffix, or if any field
                fails validation.
        """
        prefix_tokens = max(round(context_length * prefix_ratio), 1)
        suffix_tokens = context_length - prefix_tokens - breaker_tokens
        if suffix_tokens < _MIN_SUFFIX_TOKENS:
            raise ValueError(
                f"suffix_tokens={suffix_tokens} is below minimum "
                f"{_MIN_SUFFIX_TOKENS}; reduce prefix_ratio or "
                f"increase context_length"
            )

        # Size by full context_length, not prefix_tokens: each request
        # stores ``context_length`` tokens worth of KV cache (prefix +
        # breaker + suffix), so the pool's L1 footprint is
        # ``num_prefixes * context_length``, not ``num_prefixes *
        # prefix_tokens``.  Earlier versions divided by prefix_tokens,
        # which made the actual pool footprint exceed the requested target
        # by ``1 / prefix_ratio`` — at ``prefix_ratio=0.5`` the pool was
        # 2x the user's intended target.
        pool_gb = thrash * _OVERFLOW_FACTOR
        num_prefixes = max(
            int(pool_gb * tokens_per_gb_kvcache / context_length),
            1,
        )
        logger.debug(
            "Computed num_prefixes=%d from thrash=%.2f GB (target tier), "
            "_OVERFLOW_FACTOR=%.3f -> pool=%.2f GB, "
            "tokens_per_gb_kvcache=%d, context_length=%d",
            num_prefixes,
            thrash,
            _OVERFLOW_FACTOR,
            pool_gb,
            tokens_per_gb_kvcache,
            context_length,
        )
        return cls(
            context_length=context_length,
            prefix_ratio=prefix_ratio,
            thrash=thrash,
            num_prefixes=num_prefixes,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            breaker_tokens=breaker_tokens,
        )


# ---------------------------------------------------------------------------
# Synthetic data generation (module-level helpers)
# ---------------------------------------------------------------------------


def _try_load_tokenizer(model_name: str | None):
    """Best-effort load of the model's tokenizer.

    Used to generate body content as random valid token-IDs, then decoded
    back to text — guaranteeing both (a) configured token counts ≈ actual
    tokens at the model and (b) different content per call site, so chunk
    content-hashes don't collide across prefixes (which would inflate the
    LMCache hit rate even in non-blend mode).

    Returns ``None`` if ``transformers`` isn't installed or the tokenizer
    can't be loaded; callers fall back to the deterministic ``"hi"``
    filler in that case.
    """
    if model_name is None:
        return None
    try:
        # Third Party
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning("transformers not available; falling back to 'hi' body filler")
        return None
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not load tokenizer for %s (%s); falling back to 'hi' filler",
            model_name,
            e,
        )
        return None


def _generate_random_body(num_tokens: int, tokenizer, rng: random.Random) -> str:
    """Build a body of ``num_tokens`` BPE tokens of unique random content.

    Samples random token-IDs in the safe range and decodes them.  The
    decoded text re-tokenizes to (≈) ``num_tokens`` tokens (small drift
    from BPE merge boundaries) and is *content-unique* per call because
    the IDs are independent draws from ``rng``.

    Falls back to ``"hi"`` filler when ``tokenizer`` is None.
    """
    if tokenizer is None:
        return " ".join(["hi"] * num_tokens)
    ids = [rng.randrange(_TOKEN_ID_LO, _TOKEN_ID_HI) for _ in range(num_tokens)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _generate_prefix(
    index: int,
    num_tokens: int,
    tokenizer,
    rng: random.Random,
) -> str:
    """Generate a prefix with unique-ID header + per-prefix random body.

    The unique-ID header makes the prefix's chained block hash unique per
    ``index`` (so vLLM's chain-hashed prefix cache distinguishes prefixes).
    The random body uses different token-IDs per ``index`` because
    ``rng`` is a per-prefix state, so chunk content-hashes don't collide
    across prefixes — required for non-blend cache hit-rate metrics to be
    meaningful.

    Args:
        index: Zero-based prefix index; encoded into the unique ID.
        num_tokens: Approximate target token length of the full prefix.
        tokenizer: The model's tokenizer, or None for fallback.
        rng: Per-prefix seeded random source.

    Returns:
        Prefix text starting with ``"PREFIX_<8-hex-digits>"``.
    """
    unique_id = f"PREFIX_{index:08x}"
    body_words = max(num_tokens - _UNIQUE_ID_TOKENS, 1)
    body = _generate_random_body(body_words, tokenizer, rng)
    return f"{unique_id} {body}"


def _generate_suffix(num_tokens: int, tokenizer, rng: random.Random) -> str:
    """Generate the single shared suffix used by every request.

    The suffix is bit-identical across all requests by design — its
    content-hash fingerprints are the only ones CacheBlend can reuse
    across the full pool, which is exactly what Baseline 3 measures.

    Args:
        num_tokens: Approximate target token length.
        tokenizer: The model's tokenizer, or None for fallback.
        rng: Seeded random source (deterministic seed → reproducible).

    Returns:
        Suffix text starting with ``"SUFFIX"``.
    """
    body_words = max(num_tokens - 1, 1)
    body = _generate_random_body(body_words, tokenizer, rng)
    return f"SUFFIX {body}"


def _generate_breaker(num_tokens: int, tokenizer, rng: random.Random) -> str:
    """Generate a per-request breaker of fresh random tokens.

    Sampled fresh per request: each request's breaker tokens differ both
    in IDs (defeats chained prefix cache past the prefix boundary) and in
    chunk content-hash (so non-blend caches don't carry breaker hits over
    from earlier requests).

    Args:
        num_tokens: Approximate target token length.
        tokenizer: The model's tokenizer, or None for fallback.
        rng: Seeded random source.  Each call advances state.

    Returns:
        A breaker string with header + random body.
    """
    unique_id = f"BR_{rng.randrange(2**32):08x}"
    body_words = max(num_tokens - _UNIQUE_ID_TOKENS, 1)
    body = _generate_random_body(body_words, tokenizer, rng)
    return f"{unique_id} {body}"


# ---------------------------------------------------------------------------
# Workload class
# ---------------------------------------------------------------------------


class PrefixSuffixTunerWorkload(BaseWorkload):
    """Sequential two-pass workload demonstrating tiered KV-cache reuse.

    Pass 1 (executed in :meth:`warmup`) populates the cache by sending each
    prefix once.  Stats from pass 1 are discarded.  Pass 2 (executed via
    :meth:`step`) repeats the same prefix order one request at a time;
    these are the measured requests.
    """

    def __init__(
        self,
        config: PrefixSuffixTunerConfig,
        request_sender: RequestSender,
        stats_collector: StatsCollector,
        progress_monitor: ProgressMonitor,
        seed: int = 42,
        model_name: str | None = None,
    ) -> None:
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        self._seed = seed

        # Best-effort tokenizer load: when available, body content is
        # generated as random valid token-IDs decoded back to text — that
        # gives both (a) ~exact configured token count at the model (no
        # tokenizer-expansion factor) and (b) content uniqueness per
        # prefix so non-blend cache hit-rate metrics are not inflated by
        # chunk content-hash collisions across prefixes.  Falls back to
        # ``"hi"`` filler if transformers / the tokenizer isn't loadable.
        self._tokenizer = _try_load_tokenizer(model_name)

        # Each prefix gets its own RNG state so per-prefix bodies differ.
        # Constant offsets keep reproducibility while leaving room for the
        # suffix (seed + 1) and breaker (seed + 2) RNGs.
        self._prefixes: list[str] = [
            _generate_prefix(
                i,
                config.prefix_tokens,
                self._tokenizer,
                random.Random(seed + 1000 + i),
            )
            for i in range(config.num_prefixes)
        ]
        self._suffix: str = _generate_suffix(
            config.suffix_tokens, self._tokenizer, random.Random(seed + 1)
        )
        self._breaker_rng = random.Random(seed + 2)

        self._pass2_index = 0

    def log_config(self) -> None:
        """Log key workload config before the benchmark starts."""
        c = self._config
        B = "\033[1m"
        C = "\033[96m"
        Y = "\033[93m"
        R = "\033[0m"
        pool_tokens_millions = c.num_prefixes * c.prefix_tokens / 1_000_000
        print(
            f"{B}{'═' * 50}{R}\n"
            f"{B} Workload: {C}prefix-suffix-tuner{R}\n"
            f"{B}{'─' * 50}{R}\n"
            f"  Context length:    {Y}{c.context_length}{R} tokens\n"
            f"  Prefix tokens:     {Y}{c.prefix_tokens}{R} (ratio={c.prefix_ratio})\n"
            f"  Breaker tokens:    {Y}{c.breaker_tokens}{R} (random per request)\n"
            f"  Suffix tokens:     {Y}{c.suffix_tokens}{R} (shared, deterministic)\n"
            f"  Target tier:       {Y}{c.thrash:.2f} GB{R}"
            f" (overflow x{_OVERFLOW_FACTOR:.2f} = "
            f"{c.thrash * _OVERFLOW_FACTOR:.2f} GB)\n"
            f"  Prefix pool size:  {Y}{c.num_prefixes}{R}\n"
            f"  Pool tokens:       {Y}{pool_tokens_millions:.2f}M{R}\n"
            f"  Total measured:    {Y}{c.num_prefixes}{R} requests "
            f"(pass 2 of 2)\n"
            f"{B}{'═' * 50}{R}"
        )

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(self, prefix_index: int) -> list[dict[str, str]]:
        """Build chat messages for one request.

        The breaker is freshly randomized on every call, so two requests
        for the same prefix produce different prompts past the prefix
        boundary.  Pass 1 and pass 2 therefore use different breakers
        per prefix even though the prefix order is identical.

        Args:
            prefix_index: Index into the generated prefix pool.

        Returns:
            A single-message chat list.
        """
        prefix = self._prefixes[prefix_index]
        breaker = _generate_breaker(
            self._config.breaker_tokens, self._tokenizer, self._breaker_rng
        )
        content = f"{prefix} {breaker} {self._suffix}"
        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # Pass 1 — warmup
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Run pass 1: send each prefix once sequentially to populate cache.

        Before pass 1 starts, sends a single short throwaway request to
        amortize first-request engine overhead (torch.compile JIT-fallback
        paths, CUDA-graph capture for novel batch shapes, vLLM/LMCache
        connector handshake, tokenizer lazy initialization).  This keeps
        the first request of pass 1 on a fully-warmed engine, so that the
        cache-population pass does not include outlier latency from
        first-request overhead.  The throwaway prompt is ~5 tokens — too
        small to occupy a chunk-aligned region — so it does not pollute
        the LMCache hit-rate metrics.

        After dispatching all warmup requests, sleeps for
        :data:`_EVICTION_SETTLE_SECONDS` so LMCache's batched-eviction LRU
        (1Hz background poll, evicts only when usage > watermark) has time
        to actually run.  Without this delay, a fast warmup that overflows
        L1 by a few percent may complete before any eviction fires, and
        pass 2 would then find all of pass 1's data still in cache.
        """
        # Engine warmup: amortize first-request torch.compile / CUDA-graph
        # / connector-handshake cost onto a throwaway request.
        self._progress_monitor.log_message(
            "Engine warmup (1 throwaway request before pass 1)"
        )
        warmup_result = await self._request_sender.send_warmup_request(
            "engine_warmup",
            [{"role": "user", "content": "Hello"}],
        )
        if not warmup_result.successful:
            self._progress_monitor.log_message(
                f"Engine warmup failed: {warmup_result.error}"
            )

        n = self._config.num_prefixes
        self._progress_monitor.log_message(f"Pass 1 (warmup): {n} requests")
        for i in range(n):
            request_id = f"pass1_p{i}"
            messages = self._build_messages(i)
            self._progress_monitor.on_request_sent(request_id)
            self._progress_monitor.log_message(f"Pass 1 dispatched {i + 1}/{n}")
            result = await self._request_sender.send_warmup_request(
                request_id,
                messages,
            )
            if not result.successful:
                self._progress_monitor.log_message(
                    f"Pass 1 {request_id} failed: {result.error}"
                )
        self._progress_monitor.log_message(
            f"Pass 1 complete: {n} prefixes populated; "
            f"settling {_EVICTION_SETTLE_SECONDS:.1f}s for LMCache LRU eviction",
        )
        # Standard
        import asyncio as _asyncio

        await _asyncio.sleep(_EVICTION_SETTLE_SECONDS)

    # ------------------------------------------------------------------
    # Pass 2 — measured benchmark
    # ------------------------------------------------------------------

    async def step(self, time_offset: float) -> float:
        """Send one pass-2 request inline, sequentially.

        Awaiting the request inside ``step`` enforces strict
        one-request-at-a-time dispatch.  Returns ``0.0`` for an immediate
        re-call until the prefix list is exhausted, then ``-1.0``.

        Args:
            time_offset: Seconds since pass 2 started (unused).

        Returns:
            ``0.0`` if more requests remain, ``-1.0`` when pass 2 is
            complete.
        """
        if self._pass2_index >= self._config.num_prefixes:
            return -1.0

        i = self._pass2_index
        self._pass2_index += 1
        request_id = f"pass2_p{i}"
        messages = self._build_messages(i)
        self._progress_monitor.on_request_sent(request_id)
        self._progress_monitor.log_message(
            f"Pass 2 {i + 1}/{self._config.num_prefixes}"
        )
        await self._request_sender.send_request(
            request_id,
            messages,
            max_tokens=_MAX_OUTPUT_TOKENS,
        )
        return 0.0

    def on_request_finished(self, request_id: str, output: str) -> None:
        """No-op — this workload is stateless."""
