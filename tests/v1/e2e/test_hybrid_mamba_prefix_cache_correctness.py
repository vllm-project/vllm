# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output-correctness regression tests for hybrid-Mamba prefix caching.

Reproduces issue #43559: with prefix caching (``mamba_cache_mode="align"``)
and MTP speculative decoding, a cached Mamba block can hold a recurrent
state that does not correspond to the block boundary its hash describes.
Every request that later hits that block resumes from a wrong state and
silently produces corrupted output until the cache is reset. Two trigger
mechanisms are covered, each with its own test:

1. Concurrent cold prefills under a small token budget fragment a prefill
   chunk mid Mamba block; the chunk-end state is cached as the boundary
   snapshot (scheduler ``_mamba_block_aligned_split`` fall-through; fixes
   proposed in #45477 / #47861).
2. Multi-turn reuse of blocks written during (speculative) decode, where
   the eagle-lookahead cache lookup lets the Mamba hit length overrun the
   attention-verified hit by one block (``find_longest_cache_hit``; fixes
   proposed in #47861 / #45614 / #46281).

Each test compares a trigger engine against a control engine with
``enable_prefix_caching=False``, using deterministic greedy needle
recall. The arms necessarily differ in TWO knobs: vLLM forces
``mamba_cache_mode`` back to ``"none"`` whenever prefix caching is off
(``MambaModelConfig.verify_and_update_config``), and align mode rewrites
chunked-prefill boundaries on its own — so the control also schedules
different chunk boundaries and hence different numerics. Together with
vLLM not promising batch-shape-invariant numerics
(``VLLM_BATCH_INVARIANT`` is off), this is why grading is relative to
the control everywhere rather than zero-tolerance: benign
nondeterminism flips recall symmetrically in both directions, while
#43559 corruption is one-sided (wrong only with caching). Both tests print prefix-cache
and spec-decode counters from ``llm.get_metrics()`` and fail loudly if
the run never engaged the prefix cache or the speculator; the cold-race
test additionally proves cache liveness with a >=3-block probe prompt,
because its sub-2-block trigger geometry is legitimately uncacheable on
a fixed tree: with EAGLE/MTP the last cacheable Mamba boundary is pulled
back one block, so prompts shorter than 2 * mamba_block_size (thousands
of tokens on these models) get zero prefix-cache reuse and non-final
chunks must end on a block boundary or defer. That geometry cliff is a
documented cost of correctness (the pre-fix "hits" at this geometry WERE
the corruption); recovering reuse there needs boundary-state retention
plus attention-only eagle re-verification, left as follow-up work.

Corruption is a hard red: the fix for both mechanisms ships together
with these tests, so no xfail marker is carried. Corruption checks
raise ``CorruptionDetected`` so a bug red stays distinguishable from
engagement, control-quality, geometry, or environment failures, which
raise plain assertions or dedicated ``Exception`` subclasses. Geometry
and control-quality guards hard-fail (``GeometryUnsupported`` /
``ControlQualityFailure``) rather than ``pytest.skip``/``pytest.fail``:
the fork-based per-test runner reports in-body skips as PASS, so
geometry drift would silently green the CI step and be
indistinguishable from a healthy run. Measured status: RED on unfixed main
and GREEN on the fixed tree for both arms (large-memory GPUs,
Nemotron-3 Super). The multi-turn Qwen parametrization is likewise a
graded arm: it was observed red once on an unfixed tree (14/32
one-sided wave-2 recall flips — the first reproduction on the issue's
own model family; an earlier batch-1 community harness on the issue
thread could not reproduce on this checkpoint) and green on the fixed
tree, though it was not re-baselined red at this fix's exact base
commit.

These are end-to-end output-level tests and intentionally do not overlap
with the scheduler-level unit tests shipped in #45477/#47861.
"""

import random
import time

import pytest

from vllm import LLM, SamplingParams
from vllm.v1.kv_cache_interface import MambaSpec

from ...utils import (
    create_new_process_for_each_test,
    large_gpu_mark,
    multi_gpu_marks,
)

# Hybrid (full-attention + Mamba/GDN) models with MTP weights — the same
# configuration class as issue #43559, which reports Qwen3.6-35B-A3B; the
# smaller 27B-FP8 sibling keeps the single-GPU parametrization affordable.
_QWEN_PARAM = pytest.param(
    "Qwen/Qwen3.6-27B-FP8",
    1,
    marks=[large_gpu_mark(min_gb=80)],
    id="qwen3.6-27b-fp8",
)
_NEMOTRON_PARAM = pytest.param(
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    4,
    # ~240 GB of BF16 weights across TP4 need H200/GB200-class capacity;
    # min_gb=140 separates H200 (~150 decimal GB) from H100-80GB (~85),
    # which would OOM at engine boot under gpu_memory_utilization=0.8.
    marks=[large_gpu_mark(min_gb=140)] + multi_gpu_marks(num_gpus=4),
    id="nemotron-super-120b-bf16",
)
HYBRID_MTP_MODELS = [_QWEN_PARAM, _NEMOTRON_PARAM]
# Qwen is excluded from the cold-race test: under its trigger geometry the
# APC-off control arm itself loses needle recall (measured wave-1 misses
# 16/16 with prefix caching disabled — an APC-independent misbehavior), so
# the control-quality gate fires and the test cannot grade the prefix
# cache on that model.
COLD_RACE_MODELS = [_NEMOTRON_PARAM]

NUM_SPEC_TOKENS = 2
NUM_IDENTICAL_COLD_PROMPTS = 16
# Small per-step budget so concurrent cold prefills of between-1-and-2-block
# prompts are fragmented mid block (the #45477 failing geometry).
RACE_MAX_BATCHED_TOKENS = 4096
RACE_MAX_NUM_SEQS = 8
NUM_MULTI_TURN_PROMPTS = 32
# Benign (non-bug) recall flips from batch-shape nondeterminism are
# symmetric across arms; corruption is one-sided. Allow this many forward
# flips beyond the observed reverse-direction (control-only) flips.
FLIP_MARGIN = 2

# Prefix for GeometryUnsupported hard failures. These paths deliberately
# do NOT use pytest.skip: under the fork-based per-test runner a Skipped
# raised in the test body is converted to child exit code 0, which the
# parent reports as PASS — geometry drift would silently green the
# CI step and be indistinguishable from a healthy run.
GEOMETRY_UNSUPPORTED = "#43559 geometry unsupported, coverage lost: "


class CorruptionDetected(AssertionError):
    """Raised only by the #43559 corruption checks.

    A failed engagement guard, control-quality gate, geometry assert, or
    an environment/boot fault raises a different type, so triage can never
    mistake it for the corruption signature.

    Exception identity survives ``create_new_process_for_each_test``: the
    fork-based runner used on CUDA cloudpickles the child's exception
    object and re-raises it in the parent. (The spawn-based runner used on
    ROCm/XPU re-wraps failures as ``RuntimeError`` — acceptable, since
    these parametrizations are CUDA-targeted.)
    """


class GeometryUnsupported(Exception):
    """Raised when the resolved geometry cannot express a trigger.

    A hard failure by design (see ``GEOMETRY_UNSUPPORTED``): the fork
    runner reports in-body skips as PASS, and a plain ``Exception`` is
    both fork-wrapper-safe and distinct from ``CorruptionDetected``, so
    lost coverage shows up red with its full diagnostic instead of
    greening the CI step.
    """


class ControlQualityFailure(Exception):
    """Raised when the APC-off control cannot support the probe.

    Not raised via ``pytest.fail``: ``Failed`` derives from
    ``BaseException`` and escapes the fork wrapper's ``except Exception``
    child handler, losing the diagnostic and letting the forked pytest
    session run on. A plain ``Exception`` subclass is cloudpickled and
    re-raised verbatim in the parent, and — not being
    ``CorruptionDetected`` — is never mistaken for the corruption
    signature.
    """


def _check_corruption(condition: bool, message: str) -> None:
    if not condition:
        raise CorruptionDetected(message)


FILLERS = [
    "Routine inspections of the corridor systems must be logged in the "
    "master ledger before the end of each shift without exception.",
    "All personnel are reminded that badge access records are audited "
    "weekly by the compliance office and discrepancies are escalated.",
    "Environmental sensors in every wing report temperature and humidity "
    "readings to the central monitoring desk on an hourly schedule.",
    "Maintenance requests for lighting, ventilation, or plumbing should "
    "be filed through the standard facilities portal before noon.",
    "Emergency assembly points are marked on the floor plans posted "
    "beside every stairwell entrance in the main building.",
    "Contractors must be escorted at all times and their equipment must "
    "be inventoried upon both entry and exit from the site.",
]


def _engine_kwargs(tp_size: int) -> dict:
    return dict(
        tensor_parallel_size=tp_size,
        max_model_len=10240,
        gpu_memory_utilization=0.8,
        # Let vLLM resolve the hybrid (Mamba-aligned) block size instead of
        # the VllmRunner default of 16.
        block_size=None,
        enforce_eager=True,
        enable_chunked_prefill=True,
        # Required for llm.get_metrics().
        disable_log_stats=False,
        # Passed to both arms, but effective only with prefix caching on:
        # MambaModelConfig.verify_and_update_config forces the mode back to
        # "none" when enable_prefix_caching=False, so the control arm also
        # loses align-mode chunk-boundary splitting (see module docstring).
        mamba_cache_mode="align",
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": NUM_SPEC_TOKENS,
        },
    )


def _mamba_block_size(llm: LLM) -> int:
    """Resolved Mamba state-checkpoint granularity in tokens.

    Requires the in-process engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``).
    In ``mamba_cache_mode="align"`` the scheduler aligns prefill chunks to
    ``cache_config.block_size`` (the engine core's MIN over all kv-cache
    groups, attention included), so the trigger geometry is only sound if
    that value equals the MambaSpec block size — verified below,
    hard-failing with a coverage-lost signal on mismatch rather than
    silently mistargeting.
    """
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    mamba_block_sizes = {
        group.kv_cache_spec.block_size
        for group in scheduler.kv_cache_config.kv_cache_groups
        if isinstance(group.kv_cache_spec, MambaSpec)
    }
    assert mamba_block_sizes, f"{llm} is not a hybrid-Mamba model"
    assert len(mamba_block_sizes) == 1, mamba_block_sizes
    block_size = mamba_block_sizes.pop()
    split_block_size = scheduler.cache_config.block_size
    if split_block_size != block_size:
        raise GeometryUnsupported(
            GEOMETRY_UNSUPPORTED + f"align-mode chunk splitting uses "
            f"cache_config.block_size ({split_block_size}), which no "
            f"longer matches the MambaSpec block size ({block_size}); "
            f"the trigger geometry would silently mistarget"
        )
    return block_size


def _counter(llm: LLM, name: str) -> int:
    """Sum matching counters; some deployments suffix the metric name."""
    return sum(
        int(metric.value)
        for metric in llm.get_metrics()
        if name in metric.name and hasattr(metric, "value")
    )


def _settled_delta(llm: LLM, name: str, baseline: int, timeout_s: float = 30.0) -> int:
    """Counter delta vs ``baseline``, waiting out aggregation lag.

    Metric aggregation can lag ``generate()`` returning, so an immediate
    post-generate read may transiently be zero even though the engine did
    the work. Poll until the counter moves or ``timeout_s`` elapses; a
    delta that is still zero after the wait is a real zero.
    """
    deadline = time.monotonic() + timeout_s
    delta = _counter(llm, name) - baseline
    while delta <= 0 and time.monotonic() < deadline:
        time.sleep(1.0)
        delta = _counter(llm, name) - baseline
    return delta


def _assert_cache_and_spec_engaged(
    hits: int, queries: int, drafts: int, arm: str, require_hits: bool
) -> None:
    """An APC+MTP run that exercised nothing would pass vacuously.

    ``require_hits=False`` is for the cold-race trigger waves: with spec
    decode the scheduler reserves the last aligned Mamba boundary for the
    eagle block drop, so prompts shorter than two blocks are expected to
    be legitimately uncacheable once a fix lands. The cold-race test
    separately proves the cache read/write path is live with a >=3-block
    liveness probe, so a regression that silently disables caching still
    cannot pass.
    """
    assert queries > 0, (
        f"[{arm}] prefix cache never consulted (queries={queries}); the "
        f"run cannot validate #43559 semantics"
    )
    assert hits > 0 or not require_hits, (
        f"[{arm}] zero prefix-cache hits (queries={queries}); the run "
        f"cannot validate #43559 semantics"
    )
    assert drafts > 0, (
        f"[{arm}] MTP speculator produced no drafts; the run cannot "
        f"validate #43559 semantics"
    )


def _build_needle_manual(
    tokenizer, target_tokens: int
) -> tuple[str, list[str], int, int]:
    """Build a facility manual with recallable access codes ("needles").

    Pads with filler sentences so the manual lands as close as possible to
    ``target_tokens``, at single-filler (~30 token) granularity for any
    Mamba block size: the total filler count is binary-searched and spread
    across needle slots, instead of padding every slot uniformly. Returns
    (text, codes, num_tokens, num_needles).
    """
    num_needles = max(8, min(40, target_tokens // 70))
    rng = random.Random(20260709)
    codes = [f"{rng.randint(0, 999999):06d}" for _ in range(num_needles)]
    header = (
        "You are the security auditor for the Meridian facility. Below is "
        "the facility manual. Memorize every access code exactly as "
        "written; you will be quizzed on them.\n\n"
    )
    footer = (
        "\nEnd of manual. Answer each question using ONLY the codes from "
        "the manual above.\n\n"
    )

    def build(total_fill: int) -> str:
        base, extra = divmod(total_fill, num_needles)
        parts, fill_idx = [header], 0
        for i in range(num_needles):
            for _ in range(base + (1 if i < extra else 0)):
                parts.append(FILLERS[fill_idx % len(FILLERS)] + "\n")
                fill_idx += 1
            parts.append(
                f"Security fact {i:02d}: the access code for vault-{i:02d} "
                f"is {codes[i]}.\n"
            )
        parts.append(footer)
        return "".join(parts)

    token_counts: dict[int, int] = {}

    def tokens(total_fill: int) -> int:
        if total_fill not in token_counts:
            token_counts[total_fill] = len(tokenizer.encode(build(total_fill)))
        return token_counts[total_fill]

    # Token count is monotone in the filler count: binary-search it.
    lo, hi = 0, 8
    while tokens(hi) < target_tokens and hi < 4096:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tokens(mid) < target_tokens:
            lo = mid
        else:
            hi = mid
    best = min((lo, hi), key=lambda t: abs(tokens(t) - target_tokens))
    return build(best), codes, tokens(best), num_needles


def _needle_question(vault: int) -> str:
    return (
        f"Question: What is the access code for vault-{vault:02d}? "
        f"Reply with only the code.\nAnswer:"
    )


def _recall(texts: list[str], codes: list[str], vaults: list[int]) -> list[int]:
    """Positions whose answer does not contain the queried vault code."""
    return [
        k
        for k, (vault, text) in enumerate(zip(vaults, texts))
        if codes[vault] not in text
    ]


def _one_sided_flips(
    apc_missed: set[int], ctl_missed: set[int]
) -> tuple[list[int], list[int]]:
    """(forward, reverse) recall flips between the APC arm and the control.

    Forward = recalled by the control but missed with caching on (the
    #43559 signature); reverse = the benign opposite direction, which
    calibrates the run's own nondeterminism rate.
    """
    return sorted(apc_missed - ctl_missed), sorted(ctl_missed - apc_missed)


@create_new_process_for_each_test()
@pytest.mark.parametrize("model_name, tp_size", COLD_RACE_MODELS)
def test_cold_concurrent_prefill_mamba_prefix_cache(
    vllm_runner, monkeypatch, model_name, tp_size
):
    """Cold-race arm of #43559 (mechanism addressed by #45477/#47861).

    N identical cold prompts sized between 1 and 2 Mamba blocks are
    prefilled concurrently under a small token budget with APC + MTP. The
    budget fragments some prefills mid block; on an unfixed scheduler the
    fragment-end recurrent state is cached as the block-boundary snapshot,
    so requests hitting the shared prefix diverge and lose needle recall.

    Asserts, all relative to a prefix-caching-off control: (a) the
    identical greedy requests produce no more distinct outputs than the
    control's plus one, (b) needle-recall parity within FLIP_MARGIN, and (c)
    engagement validity — nonzero prefix-cache queries and MTP drafts,
    plus nonzero hits for a >=3-block liveness probe prompt (trigger-wave
    hits themselves may legitimately be zero on a fixed tree, where the
    eagle boundary reservation makes sub-2-block prompts uncacheable).

    Corruption raises ``CorruptionDetected``; engagement and environment
    failures raise plain assertions, geometry drift raises
    ``GeometryUnsupported``, and a weak control raises
    ``ControlQualityFailure``, so triage stays unambiguous.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    kwargs = _engine_kwargs(tp_size)
    kwargs["max_num_batched_tokens"] = RACE_MAX_BATCHED_TOKENS
    kwargs["max_num_seqs"] = RACE_MAX_NUM_SEQS

    sampling = SamplingParams(temperature=0.0, max_tokens=24, stop=["\n"])

    try:
        trigger_runner = vllm_runner(
            model_name,
            enable_prefix_caching=True,
            **kwargs,
        )
    except AssertionError as exc:
        # VllmConfig.validate_block_size rejects align mode at engine boot
        # whenever the resolved block size exceeds max_num_batched_tokens —
        # exactly the geometry whose small budget could not fragment a
        # prefill mid block. Surface it as lost coverage, not a boot error.
        if "must be <= max_num_batched_tokens" not in str(exc):
            raise
        raise GeometryUnsupported(
            GEOMETRY_UNSUPPORTED + f"resolved block size exceeds the "
            f"fragmenting token budget {RACE_MAX_BATCHED_TOKENS} ({exc})"
        ) from exc
    with trigger_runner as runner:
        llm = runner.get_llm()
        block_size = _mamba_block_size(llm)
        tokenizer = llm.get_tokenizer()
        manual, codes, manual_tokens, num_needles = _build_needle_manual(
            tokenizer, target_tokens=int(block_size * 1.35)
        )
        if not block_size * 1.05 < manual_tokens < block_size * 1.95:
            raise GeometryUnsupported(
                GEOMETRY_UNSUPPORTED + f"needle manual ({manual_tokens} "
                f"tokens) not strictly between 1 and 2 Mamba blocks "
                f"({block_size=})"
            )

        # Wave 1: identical cold prompts, prefilled concurrently, no warmup.
        # Probe a needle near the prefix end: recall there is reliable even
        # for models whose recall of distant needles is weak, while state
        # corruption garbles the answer regardless of needle position.
        wave1_vault = num_needles - 2
        wave1 = [manual + _needle_question(wave1_vault)]
        wave1 *= NUM_IDENTICAL_COLD_PROMPTS
        # Wave 2: distinct questions resuming from the wave-1 prefix.
        wave2_vaults = list(range(num_needles))
        wave2 = [manual + _needle_question(i) for i in wave2_vaults]

        wave1_texts = [out.outputs[0].text for out in llm.generate(wave1, sampling)]
        wave2_texts = [out.outputs[0].text for out in llm.generate(wave2, sampling)]
        hits = _counter(llm, "vllm:prefix_cache_hits")
        queries = _counter(llm, "vllm:prefix_cache_queries")
        # Cache-liveness probe: the sub-2-block trigger geometry is expected
        # to be uncacheable on a fixed tree, so prove the cache read/write
        # path is live in this configuration with a >=3-block prompt
        # submitted twice; without this, "zero hits" could also mean the
        # cache was silently disabled and the test would pass vacuously.
        liveness_hits = None
        probe_hits = None
        if 3 * block_size + 512 > kwargs["max_model_len"]:
            # The anti-vacuity gate MUST run: if the >=3-block probe cannot
            # fit, a clean cold-race result would be vacuous (a
            # silently-disabled cache is indistinguishable from a cured one).
            # Hard-fail rather than silently skip.
            raise GeometryUnsupported(
                GEOMETRY_UNSUPPORTED + f"cache-liveness probe "
                f"({3 * block_size + 512} tokens) exceeds "
                f"max_model_len={kwargs['max_model_len']}"
            )
        probe_text, _, _, _ = _build_needle_manual(
            tokenizer, target_tokens=3 * block_size + 128
        )
        probe_sampling = SamplingParams(temperature=0.0, max_tokens=8)
        llm.generate([probe_text + _needle_question(0)], probe_sampling)
        probe_hits = _counter(llm, "vllm:prefix_cache_hits")
        llm.generate([probe_text + _needle_question(1)], probe_sampling)
        # The trigger waves' short stop-bounded decodes may legitimately
        # draft zero tokens on some models; a short unconstrained decode
        # proves the MTP speculator is live in this engine configuration.
        # It also shares the cached manual prefix, and its longer decode
        # forces metric flushes: counters only publish during generation
        # steps, so a short probe's stats may never surface if it is the
        # last generate — the liveness delta is therefore read only after
        # this generate (any post-baseline hit proves the cache read path
        # is live).
        llm.generate(
            [manual + _needle_question(0)] * 4,
            SamplingParams(temperature=0.0, max_tokens=128),
        )
        if probe_hits is not None:
            liveness_hits = _settled_delta(llm, "vllm:prefix_cache_hits", probe_hits)
        drafts = _counter(llm, "spec_decode_num_drafts")
        print(
            f"METRIC vllm:prefix_cache_queries {queries}\n"
            f"METRIC vllm:prefix_cache_hits {hits}\n"
            f"METRIC liveness_probe_hits {liveness_hits}\n"
            f"METRIC vllm:spec_decode_num_drafts {drafts}"
        )

    # Control: same budgets and batch composition, prefix caching off.
    # NOTE: vLLM forces mamba_cache_mode back to "none" when caching is
    # off, so the control also loses align-mode chunk splitting — chunk
    # boundaries (and numerics) differ between arms, hence the relative,
    # flip-budgeted grading below instead of zero tolerance.
    with vllm_runner(model_name, enable_prefix_caching=False, **kwargs) as runner:
        llm = runner.get_llm()
        ctl1_texts = [out.outputs[0].text for out in llm.generate(wave1, sampling)]
        ctl2_texts = [out.outputs[0].text for out in llm.generate(wave2, sampling)]

    # Recall is graded as parity against the APC-off control: models may
    # legitimately fail distant needles (the control fails them too), but
    # any vault the control recalls must also be recalled with APC on.
    ctl1_miss = len(_recall(ctl1_texts, codes, [wave1_vault] * len(ctl1_texts)))
    ctl2_missed = set(_recall(ctl2_texts, codes, wave2_vaults))
    if ctl1_miss == len(ctl1_texts) or len(ctl2_missed) > num_needles // 2:
        raise ControlQualityFailure(
            f"control (APC off) recall too weak (wave-1 misses "
            f"{ctl1_miss}/{len(ctl1_texts)}, wave-2 misses "
            f"{sorted(ctl2_missed)}); the model cannot support this probe, "
            f"so the APC arm result is not interpretable"
        )

    # Engagement is asserted BEFORE the corruption checks, so a run that
    # never engaged APC+MTP fails as an engagement problem instead of
    # reaching a corruption check it had no standing to make.
    _assert_cache_and_spec_engaged(
        hits, queries, drafts, "cold-race", require_hits=False
    )

    # Identical greedy requests are graded against the control's own
    # distinct-output count: batch-shape nondeterminism (not guaranteed
    # invariant by vLLM) may benignly split either arm, but on the unfixed
    # scheduler only the caching arm diverges (asymmetrically). The +1
    # tolerance absorbs a single benign arm-asymmetric split (the arms
    # schedule different chunk boundaries by design); divergence without
    # recall loss is not the #43559 harm, so the margin-protected recall
    # checks below carry the pass/fail weight.
    distinct = set(wave1_texts)
    ctl_distinct = set(ctl1_texts)
    print(
        f"METRIC wave1_distinct_outputs apc={len(distinct)} control={len(ctl_distinct)}"
    )
    _check_corruption(
        len(distinct) <= max(len(ctl_distinct), 1) + 1,
        f"{len(distinct)} distinct outputs for "
        f"{NUM_IDENTICAL_COLD_PROMPTS} identical greedy prompts vs "
        f"{len(ctl_distinct)} in the APC-off control "
        f"(mid-block Mamba state cached as a boundary snapshot, #43559): "
        f"{distinct!r}",
    )
    missed = len(_recall(wave1_texts, codes, [wave1_vault] * len(wave1_texts)))
    _check_corruption(
        missed <= ctl1_miss + FLIP_MARGIN,
        f"wave-1 needle recall failed for {missed}/{len(wave1_texts)} "
        f"identical prompts vs {ctl1_miss} in the APC-off control (#43559)",
    )
    forward, reverse = _one_sided_flips(
        set(_recall(wave2_texts, codes, wave2_vaults)), ctl2_missed
    )
    _check_corruption(
        len(forward) <= len(reverse) + FLIP_MARGIN,
        f"wave-2 needle recall failed for vaults {forward} after resuming "
        f"from the cached prefix while the APC-off control recalled them; "
        f"only {len(reverse)} benign reverse flips {reverse} (#43559)",
    )

    # Vacuous-pass guard, checked AFTER the corruption gates: on a
    # bug-live tree the trigger mechanics themselves can suppress the
    # probe's hits (fragmented cold prefill caching the wrong state plus the
    # spec-decode boundary reservation), so a zero here is only
    # meaningful — a silently-disabled cache — when no corruption fired
    # above. On a fixed tree the >=3-block probe must hit.
    if liveness_hits is not None:
        assert liveness_hits > 0, (
            f"[cold-race] no corruption detected, but zero prefix-cache "
            f"hits for the repeated >=3-block liveness probe "
            f"({liveness_hits=}); the prefix cache is not live, so the "
            f"clean trigger-wave result is vacuous"
        )


@create_new_process_for_each_test()
@pytest.mark.parametrize("model_name, tp_size", HYBRID_MTP_MODELS)
def test_multi_turn_decode_written_mamba_prefix_cache(
    vllm_runner, monkeypatch, model_name, tp_size
):
    """Multi-turn arm of #43559 (mechanism addressed by #47861/#45614/#46281).

    Wave-1 prompts end just below a Mamba block boundary and decode past
    it under MTP, so the boundary state snapshot is written during
    speculative decode. Wave-2 re-asks with prompt = wave-1 prompt +
    output + follow-up, hitting the decode-written blocks at a high hit
    ratio. On an unfixed tree the Mamba hit length can overrun the
    attention-verified hit (eagle-lookahead peek), corrupting recall.

    Asserts needle-recall parity (within FLIP_MARGIN of the benign
    reverse-flip rate) between the APC engine and an APC-off control
    replaying the exact wave-2 strings, plus nonzero wave-2 prefix-cache
    hits and MTP drafts.

    Corruption raises ``CorruptionDetected``; engagement and environment
    failures raise plain assertions, geometry drift raises
    ``GeometryUnsupported``, and a weak control raises
    ``ControlQualityFailure``, so triage stays unambiguous.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    kwargs = _engine_kwargs(tp_size)

    with vllm_runner(
        model_name,
        enable_prefix_caching=True,
        **kwargs,
    ) as runner:
        llm = runner.get_llm()
        block_size = _mamba_block_size(llm)
        tokenizer = llm.get_tokenizer()
        # Prompts end just below the second block boundary; decode crosses
        # it, so its snapshot embeds decode-time (speculative) state.
        manual, codes, manual_tokens, num_needles = _build_needle_manual(
            tokenizer, target_tokens=2 * block_size - 192
        )
        if not block_size < manual_tokens < 2 * block_size - 48:
            raise GeometryUnsupported(
                GEOMETRY_UNSUPPORTED + f"needle manual ({manual_tokens} "
                f"tokens) not just below the second Mamba block boundary "
                f"({block_size=})"
            )
        min_tokens = 2 * block_size - manual_tokens + 96
        # Wave-2 prompts embed wave-1 prompt + output + a follow-up.
        if manual_tokens + min_tokens + 64 + 128 > kwargs["max_model_len"]:
            raise GeometryUnsupported(
                GEOMETRY_UNSUPPORTED + f"wave-2 prompts (~"
                f"{manual_tokens + min_tokens + 64} tokens at "
                f"{block_size=}) would exceed "
                f"max_model_len={kwargs['max_model_len']}"
            )

        wave1 = [
            manual + f"Question: Describe the audit and escalation procedure for "
            f"vault-{i:02d} in detail.\nAnswer:"
            for i in range(NUM_MULTI_TURN_PROMPTS)
        ]
        wave1_sampling = SamplingParams(
            temperature=0.0,
            min_tokens=min_tokens,
            max_tokens=min_tokens + 64,
        )
        wave1_texts = [
            out.outputs[0].text for out in llm.generate(wave1, wave1_sampling)
        ]
        hits_before = _counter(llm, "vllm:prefix_cache_hits")
        queries_before = _counter(llm, "vllm:prefix_cache_queries")

        # Wave 2 resumes from wave-1 prompt + output: the prefix-cache hit
        # covers blocks whose boundary snapshot was written during decode.
        wave2_vaults = [
            (i * 7 + 3) % num_needles for i in range(NUM_MULTI_TURN_PROMPTS)
        ]
        wave2 = [
            prompt + text + "\n\n" + _needle_question(vault)
            for prompt, text, vault in zip(wave1, wave1_texts, wave2_vaults)
        ]
        wave2_sampling = SamplingParams(temperature=0.0, max_tokens=24, stop=["\n"])
        wave2_texts = [
            out.outputs[0].text for out in llm.generate(wave2, wave2_sampling)
        ]
        wave2_hits = _settled_delta(llm, "vllm:prefix_cache_hits", hits_before)
        wave2_queries = _settled_delta(llm, "vllm:prefix_cache_queries", queries_before)
        # Engagement is gated on the cumulative (whole-flow) counters:
        # wave-1 shares the manual prefix and wave-2 resumes wave-1
        # prompts, so any zero here means APC/MTP never engaged at all.
        # The wave-2 deltas are diagnostics, not gates.
        hits = _counter(llm, "vllm:prefix_cache_hits")
        queries = _counter(llm, "vllm:prefix_cache_queries")
        drafts = _counter(llm, "spec_decode_num_drafts")
        print(
            f"METRIC wave2 prefix_cache_queries {wave2_queries}\n"
            f"METRIC wave2 prefix_cache_hits {wave2_hits}\n"
            f"METRIC wave2 hit_ratio "
            f"{(wave2_hits / wave2_queries) if wave2_queries else 0.0:.3f}\n"
            f"METRIC vllm:prefix_cache_queries {queries}\n"
            f"METRIC vllm:prefix_cache_hits {hits}\n"
            f"METRIC vllm:spec_decode_num_drafts {drafts}"
        )

    # Control replays the exact wave-2 strings with prefix caching off.
    # NOTE: vLLM forces mamba_cache_mode back to "none" when caching is
    # off, so chunk boundaries (and numerics) differ between arms, hence
    # the flip-budgeted parity grading below instead of zero tolerance.
    with vllm_runner(model_name, enable_prefix_caching=False, **kwargs) as runner:
        llm = runner.get_llm()
        ctl2_texts = [
            out.outputs[0].text for out in llm.generate(wave2, wave2_sampling)
        ]

    # Parity grading: wave-2 strings embed wave-1 outputs, so a weak or
    # corrupted first turn can legitimately cost the control some recalls;
    # what may not happen (beyond the benign symmetric flip rate) is
    # APC-on missing an answer the control got.
    ctl_missed = set(_recall(ctl2_texts, codes, wave2_vaults))
    if len(ctl_missed) > NUM_MULTI_TURN_PROMPTS // 2:
        raise ControlQualityFailure(
            f"control (APC off) recall too weak (misses "
            f"{sorted(ctl_missed)}); the model cannot support this probe, "
            f"so the APC arm result is not interpretable"
        )
    # Engagement is asserted BEFORE the corruption check, so a run that
    # never engaged APC+MTP fails as an engagement problem instead of
    # reaching a corruption check it had no standing to make. The gate
    # uses the cumulative counters (see above); the wave-2 hit ratio is
    # printed rather than asserted (0.431-0.477 across the fixed-tree
    # calibration runs: 0.46 Nemotron-Super TP4 and 0.43 Qwen TP1 on this
    # tree, 0.477 under the single-lane cures — not calibrated as stable
    # across hardware).
    _assert_cache_and_spec_engaged(
        hits, queries, drafts, "multi-turn", require_hits=True
    )

    forward, reverse = _one_sided_flips(
        set(_recall(wave2_texts, codes, wave2_vaults)), ctl_missed
    )
    _check_corruption(
        len(forward) <= len(reverse) + FLIP_MARGIN,
        f"wave-2 needle recall failed for prompts {forward} when resuming "
        f"from decode-written Mamba blocks that the APC-off control "
        f"answered correctly; only {len(reverse)} benign reverse flips "
        f"{reverse} (#43559)",
    )
