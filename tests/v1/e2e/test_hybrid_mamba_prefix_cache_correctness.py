# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output-correctness regression tests for hybrid-Mamba prefix caching (#43559).

With prefix caching (``mamba_cache_mode="align"``) and MTP speculative
decoding, a cached Mamba block can hold a recurrent state that does not match
the block boundary its hash describes. Requests that later hit that block
resume from the wrong state and silently produce corrupted output. This
covers multi-turn reuse of blocks written during speculative decode
(#47861 / #45614 / #46281).

The cold-prefill trigger (#45477 / #47861) is not covered here. Under align
mode with EAGLE/MTP the last cacheable boundary is pulled back one block, so
the sub-2-block prompt that fragments mid block commits no boundary snapshot
and cannot arm; and at a >2-block geometry, where caching is active, a
cached-vs-uncached comparison is dominated by the recall difference between
resuming and full recomputation, which is far larger than the defect. It is
covered by the scheduler-level unit tests in #45477 / #47861.

The test grades a prefix-caching engine against an
``enable_prefix_caching=False`` control rather than against fixed
expectations. vLLM forces ``mamba_cache_mode`` back to ``"none"`` when prefix
caching is off, so the arms schedule different chunk boundaries and differ
numerically; benign nondeterminism flips recall symmetrically, while #43559
corruption is one-sided.

Geometry drift and a too-weak control hard-fail rather than calling
``pytest.skip``: the fork-based per-test runner reports an in-body skip as
PASS, which would silently green the CI step.

These are end-to-end output-level tests and do not overlap with the
scheduler-level unit tests in #45477 / #47861.
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

# Hybrid (full-attention + Mamba/GDN) models with MTP weights. #43559 reports
# Qwen3.6-35B-A3B; the 27B-FP8 sibling keeps this affordable on one GPU.
_QWEN_PARAM = pytest.param(
    "Qwen/Qwen3.6-27B-FP8",
    1,
    marks=[large_gpu_mark(min_gb=80)],
    id="qwen3.6-27b-fp8",
)
_NEMOTRON_PARAM = pytest.param(
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    4,
    # min_gb=140 separates H200 (~150 decimal GB) from H100-80GB, which OOMs
    # at boot under gpu_memory_utilization=0.8.
    marks=[large_gpu_mark(min_gb=140)] + multi_gpu_marks(num_gpus=4),
    id="nemotron-super-120b-bf16",
)
HYBRID_MTP_MODELS = [_QWEN_PARAM, _NEMOTRON_PARAM]

NUM_SPEC_TOKENS = 2
NUM_MULTI_TURN_PROMPTS = 32
# Allow this many forward recall flips beyond the observed reverse-direction
# flips, which calibrate the run's own nondeterminism.
FLIP_MARGIN = 2

# Prefix for GeometryUnsupported hard failures (see module docstring).
GEOMETRY_UNSUPPORTED = "#43559 geometry unsupported, coverage lost: "


class CorruptionDetected(AssertionError):
    """Raised only by the #43559 corruption checks.

    Every other failure mode raises a different type, so a bug red stays
    distinguishable from an engagement, geometry, or environment failure.
    """


class GeometryUnsupported(Exception):
    """Raised when the resolved geometry cannot express a trigger."""


class ControlQualityFailure(Exception):
    """Raised when the APC-off control cannot support the probe.

    A plain ``Exception`` rather than ``pytest.fail``: ``Failed`` derives from
    ``BaseException`` and escapes the fork wrapper's child handler.
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
        # Let vLLM resolve the Mamba-aligned block size, not VllmRunner's 16.
        block_size=None,
        enforce_eager=True,
        enable_chunked_prefill=True,
        # Required for llm.get_metrics().
        disable_log_stats=False,
        # Effective only with prefix caching on; forced to "none" otherwise.
        mamba_cache_mode="align",
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": NUM_SPEC_TOKENS,
        },
    )


def _mamba_block_size(llm: LLM) -> int:
    """Resolved Mamba state-checkpoint granularity in tokens.

    Requires the in-process engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``).
    Align mode splits chunks on ``cache_config.block_size``, so the trigger
    geometry is only sound while that equals the MambaSpec block size.
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
    """Counter delta vs ``baseline``, polling out metric aggregation lag."""
    deadline = time.monotonic() + timeout_s
    delta = _counter(llm, name) - baseline
    while delta <= 0 and time.monotonic() < deadline:
        time.sleep(1.0)
        delta = _counter(llm, name) - baseline
    return delta


def _assert_cache_and_spec_engaged(
    hits: int, queries: int, drafts: int, arm: str
) -> None:
    """Guard against an APC+MTP run that exercised nothing passing vacuously."""
    assert queries > 0, (
        f"[{arm}] prefix cache never consulted (queries={queries}); the "
        f"run cannot validate #43559 semantics"
    )
    assert hits > 0, (
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

    Binary-searches filler count so the manual lands near ``target_tokens``.
    Returns (text, codes, num_tokens, num_needles).
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

    Forward (control recalled, APC missed) is the #43559 signature; reverse
    calibrates the run's own nondeterminism rate.
    """
    return sorted(apc_missed - ctl_missed), sorted(ctl_missed - apc_missed)


@create_new_process_for_each_test()
@pytest.mark.parametrize("model_name, tp_size", HYBRID_MTP_MODELS)
def test_multi_turn_decode_written_mamba_prefix_cache(
    vllm_runner, monkeypatch, model_name, tp_size
):
    """Multi-turn arm of #43559 (mechanism addressed by #47861 / #45614 / #46281).

    Wave-1 prompts end just below a Mamba block boundary and decode past it
    under MTP, so the boundary snapshot is written during speculative decode.
    Wave-2 re-asks with wave-1 prompt + output + follow-up, hitting those
    blocks. On an unfixed tree the Mamba hit length can overrun the
    attention-verified hit, corrupting recall.

    Grades needle recall against an APC-off control replaying the same wave-2
    strings, plus nonzero prefix-cache hits and MTP drafts.
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
        # Decode crosses the second block boundary, so its snapshot embeds
        # decode-time (speculative) state.
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

        # Wave 2 hits blocks whose boundary snapshot was written during decode.
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
        # Gated on cumulative counters; the wave-2 deltas are diagnostics.
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
    with vllm_runner(model_name, enable_prefix_caching=False, **kwargs) as runner:
        llm = runner.get_llm()
        ctl2_texts = [
            out.outputs[0].text for out in llm.generate(wave2, wave2_sampling)
        ]

    # Wave-2 strings embed wave-1 outputs, so the control may legitimately
    # lose some recalls; what may not happen is APC-on missing one it got.
    ctl_missed = set(_recall(ctl2_texts, codes, wave2_vaults))
    if len(ctl_missed) > NUM_MULTI_TURN_PROMPTS // 2:
        raise ControlQualityFailure(
            f"control (APC off) recall too weak (misses "
            f"{sorted(ctl_missed)}); the model cannot support this probe, "
            f"so the APC arm result is not interpretable"
        )
    # Asserted before the corruption check (see the cold-race arm). The
    # wave-2 hit ratio is printed, not asserted: measured ~0.43-0.48, not
    # calibrated as stable across hardware.
    _assert_cache_and_spec_engaged(hits, queries, drafts, "multi-turn")

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
