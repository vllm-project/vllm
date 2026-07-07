# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: plain encode vs batch_split_message (no cache / with cache).

Compares the end-to-end ``encode`` latency of three tokenizers on synthetic
multi-segment prompts:

  1. plain      -- vLLM's normal HF tokenizer (CachedHfTokenizer.encode)
  2. seg        -- batch_split_message, segment cache OFF
  3. seg_cache  -- batch_split_message, segment cache ON

Synthetic data: for each total-token target we build prompts of N segments
whose lengths are random (sum == target, with jitter). Each segment is, with
probability reuse_prob, an already-seen segment (so the cache can hit) and
otherwise freshly generated -- giving a realistic ~50% hit rate that the script
also measures for real.

This measures ONLY the tokenizer (pure CPU); it does not start the vLLM engine.
It does import vllm to exercise the *real* code paths
(CachedHfTokenizer.encode and BatchSplitMessageTokenizerImpl.encode).

Run on the target machine (same code as the repo):
    python benchmarks/benchmark_batch_split_tokenizer.py
"""

from __future__ import annotations

import os
import random
import statistics
import time

from transformers import AutoConfig, AutoTokenizer

from vllm.tokenizers.batch_split_message import (
    _segment_digest,
    make_batch_split_message_tokenizer,
)
from vllm.tokenizers.hf import get_cached_tokenizer

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL = os.environ.get("MODEL_PATH")
if not MODEL:
    raise SystemExit(
        "Set MODEL_PATH to the tokenizer/model path (or HF/ModelScope id)."
    )
DELIM = "<|im_end|>"
TOTAL_TOKENS = [5000, 10000, 30000, 50000, 100000, 150000, 200000]


def n_segments_for(total: int) -> int:
    # small prompts -> 5 segments, large prompts -> 10 segments
    return 5 if total <= 30000 else 10


WARMUP = 10  # warmup encodes (fixed query, identical across runs)
ITERS = 100  # measured prompts per (total, n_seg) cell
REUSE_PROBS = [0.25, 0.5, 0.75]  # P(reuse existing segment) -> ~hit rates tested
MIN_SEG_TOKENS = 16  # floor per segment so lengths stay sane
SEG_LEN_RATIO = 3.0  # longest/shortest segment length ratio (jitter)
SEED = 1234
ADD_SPECIAL = False  # chat path / genuinely-segmented path
TRUST_REMOTE_CODE = True

# Big enough that eviction never interferes (we measure hits, not eviction).
CACHE_MAX_ENTRIES = 10_000_000
CACHE_MAX_TOKENS = 100_000_000_000

# A fixed warmup prompt reused by every run (warms the Rust thread pool etc.).
WARMUP_QUERY = ("warmup " * 256).strip()


# --------------------------------------------------------------------------- #
# Tokenizer construction (bypass get_tokenizer's lru_cache so seg_cache can be
# a fresh instance per scale; encode logic is identical to the registered mode)
# --------------------------------------------------------------------------- #
def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=TRUST_REMOTE_CODE)
    return get_cached_tokenizer(tok)


def make_seg(segment_cache: bool):
    return make_batch_split_message_tokenizer(
        load_base(),
        split_delimiter=DELIM,
        segment_cache=segment_cache,
        cache_max_entries=CACHE_MAX_ENTRIES,
        cache_max_tokens=CACHE_MAX_TOKENS,
    )


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def split_lengths(total: int, n: int, rng: random.Random) -> list[int]:
    """Split ``total`` into ``n`` parts whose longest/shortest ratio ~= SEG_LEN_RATIO.

    Lengths are proportional to weights drawn from [1, SEG_LEN_RATIO]; the two
    extremes are forced in so max/min lands near the target ratio.
    """
    if n == 1:
        return [total]
    weights = [rng.uniform(1.0, SEG_LEN_RATIO) for _ in range(n)]
    weights[0], weights[1] = 1.0, SEG_LEN_RATIO  # pin the extremes
    rng.shuffle(weights)
    s = sum(weights)
    lens = [max(MIN_SEG_TOKENS, int(total * w / s)) for w in weights]
    # push the rounding remainder onto the longest segment (keeps the min intact)
    lens[lens.index(max(lens))] += total - sum(lens)
    return lens


def make_random_segment(
    base_tok, target_len: int, rng: random.Random, special_ids: set[int]
) -> str:
    """Build a text segment whose *real* token count is ~target_len.

    Random token ids do not form a natural BPE sequence: the text they decode to
    can re-encode to far MORE tokens than were sampled (BPE re-splits the
    fragments). So after decoding we re-encode, trim to target_len, and decode
    back to a stable string -- this keeps each segment near target_len and stops
    the prompt from inflating past the model's max sequence length.
    """
    vocab_size = base_tok.vocab_size
    ids: list[int] = []
    guard = 0
    while len(ids) < target_len and guard < target_len * 4:
        guard += 1
        tid = rng.randrange(vocab_size)
        if tid not in special_ids:
            ids.append(tid)
    text = base_tok.decode(ids).replace(DELIM, " ")
    # re-encode (natural BPE) and trim so the segment can't inflate past target
    reids = base_tok.encode(text, add_special_tokens=False)
    if len(reids) > target_len:
        text = base_tok.decode(reids[:target_len]).replace(DELIM, " ")
    return text


class SegmentFactory:
    """Builds fixed-structure prompts with one segment pool per slot.

    All prompts share the same per-slot lengths (``seg_lens``); slot ``i`` only
    ever holds segments of length ``seg_lens[i]``. A reused segment therefore
    always matches the slot's length, so reuse keeps the prompt's total length
    and length distribution intact (only the *content* varies with the hit
    rate). Each slot reuses an already-seen segment with prob ``reuse_prob``.
    """

    def __init__(
        self, base_tok, rng: random.Random, reuse_prob: float, seg_lens: list[int]
    ):
        self._base = base_tok
        self._rng = rng
        self._reuse_prob = reuse_prob
        self._special = set(base_tok.all_special_ids)
        self._seg_lens = seg_lens
        self._pools: list[list[str]] = [[] for _ in seg_lens]

    def make_prompt(self) -> list[str]:
        segs = []
        for i, length in enumerate(self._seg_lens):
            pool = self._pools[i]
            if pool and self._rng.random() < self._reuse_prob:
                segs.append(self._rng.choice(pool))  # reuse: same slot length
            else:
                seg = make_random_segment(self._base, length, self._rng, self._special)
                pool.append(seg)
                segs.append(seg)
        return segs


def build_prompts(base_tok, total: int, n: int, rng: random.Random, reuse_prob: float):
    seg_lens = split_lengths(total, n, rng)  # fixed structure shared by all prompts
    factory = SegmentFactory(base_tok, rng, reuse_prob, seg_lens)
    prompts, seg_lists = [], []
    for _ in range(ITERS):
        segs = factory.make_prompt()
        seg_lists.append(segs)
        prompts.append(DELIM.join(segs))
    return prompts, seg_lists


def measured_hit_rate(seg_lists) -> float:
    """Replay the prompts in order and compute the real per-segment hit rate
    (a segment hits if its digest was seen in an earlier-encoded prompt)."""
    seen: set[bytes] = set()
    hits = total = 0
    for segs in seg_lists:
        for s in segs:
            d = _segment_digest(s)
            total += 1
            if d in seen:
                hits += 1
            seen.add(d)
    return hits / total if total else 0.0


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def bench(encode_fn, prompts) -> dict:
    for _ in range(WARMUP):
        encode_fn(WARMUP_QUERY)
    times = []
    for p in prompts:
        t0 = time.perf_counter()
        encode_fn(p)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    times.sort()
    return {
        "p50": statistics.median(times),
        "p99": times[min(len(times) - 1, int(len(times) * 0.99))],
        "mean": statistics.fmean(times),
    }


def main():
    print("=== batch_split_message tokenizer benchmark ===")
    print(f"model           : {MODEL}")
    print(f"cpu_count       : {os.cpu_count()}")
    rayon = os.environ.get("RAYON_NUM_THREADS", "(default)")
    toks_par = os.environ.get("TOKENIZERS_PARALLELISM", "(default)")
    print(f"RAYON_NUM_THREADS      : {rayon}")
    print(f"TOKENIZERS_PARALLELISM : {toks_par}")
    print(
        f"warmup={WARMUP} iters={ITERS} reuse_probs={REUSE_PROBS} "
        f"add_special_tokens={ADD_SPECIAL}"
    )

    plain = load_base()
    seg = make_seg(segment_cache=False)
    # Confirm seg really runs the segmented impl (the lossless assert below would
    # pass even if seg were plain, so check the class explicitly).
    print(f"plain class: {type(plain).__name__}")
    print(f"seg   class: {type(seg).__name__}")
    assert "BatchSplitMessage" in type(seg).__name__, (
        "seg is NOT the segmented tokenizer -- make_batch_split_message_tokenizer "
        "did not take effect; benchmark would be meaningless."
    )

    def enc_plain(x):
        return plain.encode(x, add_special_tokens=ADD_SPECIAL)

    def enc_seg(x):
        return seg.encode(x, add_special_tokens=ADD_SPECIAL)

    # Cap test sizes at the model's maximum sequence length: prompts longer than
    # that are never used in practice (and trip HF's "sequence length is longer
    # than the specified maximum" warning).
    max_len = getattr(plain, "model_max_length", None)
    if not (isinstance(max_len, int) and 0 < max_len < 10**8):
        # model_max_length unset (huge sentinel) -> fall back to model config
        try:
            cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=TRUST_REMOTE_CODE)
            max_len = getattr(cfg, "max_position_embeddings", None)
        except Exception:
            max_len = None
    if isinstance(max_len, int) and max_len > 0:
        totals = [t for t in TOTAL_TOKENS if t <= max_len]
        skipped = [t for t in TOTAL_TOKENS if t > max_len]
        print(
            f"model max seq len = {max_len}; testing {totals}"
            + (f"; skipped {skipped} (> max)" if skipped else "")
        )
    else:
        totals = list(TOTAL_TOKENS)
        print("model max seq len unknown; testing all sizes")

    rows = []
    total_cells = len(totals) * len(REUSE_PROBS)
    idx = 0
    for total in totals:
        n = n_segments_for(total)
        for reuse in REUSE_PROBS:
            idx += 1
            print(
                f"[{idx}/{total_cells}] total={total} n={n} reuse={reuse:.0%}:",
                end="",
                flush=True,
            )

            rng = random.Random(SEED)  # same prompts for all three tokenizers
            print(" build", end="", flush=True)
            prompts, seg_lists = build_prompts(plain, total, n, rng, reuse)

            # real total token count (decode->re-encode is not identity)
            real_tok = statistics.median(
                len(plain.encode(p, add_special_tokens=ADD_SPECIAL))
                for p in prompts[:5]
            )

            # correctness: segmented must equal a single encode, token-for-token
            for p in prompts[:5]:
                assert list(enc_seg(p)) == list(enc_plain(p)), (
                    f"LOSSLESS VIOLATION at total={total}"
                )

            hit = measured_hit_rate(seg_lists)

            seg_cache = make_seg(segment_cache=True)  # fresh cache per cell

            def enc_cache(x, _tok=seg_cache):
                return _tok.encode(x, add_special_tokens=ADD_SPECIAL)

            print(" plain", end="", flush=True)
            sp = bench(enc_plain, prompts)
            print(" seg", end="", flush=True)
            ss = bench(enc_seg, prompts)
            print(" cache", end="", flush=True)
            sc = bench(enc_cache, prompts)
            print(
                f" done (p50 ms: plain={sp['p50']:.1f} seg={ss['p50']:.1f} "
                f"cache={sc['p50']:.1f}, hit={hit:.0%})",
                flush=True,
            )
            rows.append((total, n, reuse, int(real_tok), hit, sp, ss, sc))

    def print_table(stat: str) -> None:
        header = (
            f"\n[{stat}] {'total':>8} {'segs':>4} {'reuse':>6} {'real_tok':>9} "
            f"{'hit%':>5} {'plain(ms)':>10} {'seg(ms)':>10} {'cache(ms)':>10} "
            f"{'seg/plain':>10} {'cache/seg':>10}"
        )
        print(header)
        print("-" * (len(header) - 1))
        for total, n, reuse, real_tok, hit, sp, ss, sc in rows:
            print(
                f"      {total:>8} {n:>4} {reuse * 100:>5.0f}% {real_tok:>9} "
                f"{hit * 100:>4.0f}% "
                f"{sp[stat]:>10.2f} {ss[stat]:>10.2f} {sc[stat]:>10.2f} "
                f"{sp[stat] / ss[stat]:>9.2f}x {ss[stat] / sc[stat]:>9.2f}x"
            )

    print_table("p50")
    print_table("p99")
    print(
        "\n(per-encode latency in ms. seg/plain = plain/seg speedup (parallel "
        "gain); cache/seg = seg/cache speedup (cache gain on top of segmenting); "
        "higher = faster, e.g. 1.50x = 50% faster. hit% = measured hit rate.)"
    )


if __name__ == "__main__":
    main()
