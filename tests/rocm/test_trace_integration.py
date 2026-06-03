# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests against existing profiler CSV outputs and Perfetto traces.

Covers TC-4.1 through TC-4.7 from the F2/F3 test plan.

These tests are data-driven: they read the kernel CSVs and trace files
produced by `inference-testing -c <config.yaml>` + `uplift-plan` runs.

Data files expected (set env vars or edit DATA_* constants below):
  IT_BASELINE_DECODE_CSV  — decode_kernels.csv from the NONE allreduce run
  IT_BASELINE_PREFILL_CSV — prefill_kernels.csv from the NONE allreduce run
  IT_FUSED_DECODE_CSV     — decode_kernels.csv from the INT4/fused run
  IT_FUSED_PREFILL_CSV    — prefill_kernels.csv from the INT4/fused run
  IT_BASELINE_TRACE_GZ    — dp0_pp0_tp0_* trace from the NONE allreduce run
  IT_FUSED_TRACE_GZ       — dp0_pp0_tp0_* trace from the INT4/fused run
  IT_BENCH_BASELINE_JSON  — bench_allreduce_none.json
  IT_BENCH_INT4_JSON      — bench_allreduce_int4.json

All paths default to the allreduce_experiment results under this repo.
"""

import csv
import gzip
import os
from pathlib import Path

import pytest
import regex as re

# ---------------------------------------------------------------------------
# Resolve data file paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent.parent  # tests/rocm/ → repo root

_RESULTS = _REPO / "results" / "allreduce_experiment"

BASELINE_DIR = Path(os.environ.get("IT_BASELINE_DIR", str(_RESULTS / "none")))
FUSED_DIR = Path(os.environ.get("IT_FUSED_DIR", str(_RESULTS / "int4")))

BASELINE_DECODE_CSV = Path(
    os.environ.get("IT_BASELINE_DECODE_CSV", str(BASELINE_DIR / "decode_kernels.csv"))
)
BASELINE_PREFILL_CSV = Path(
    os.environ.get("IT_BASELINE_PREFILL_CSV", str(BASELINE_DIR / "prefill_kernels.csv"))
)
FUSED_DECODE_CSV = Path(
    os.environ.get("IT_FUSED_DECODE_CSV", str(FUSED_DIR / "decode_kernels.csv"))
)
FUSED_PREFILL_CSV = Path(
    os.environ.get("IT_FUSED_PREFILL_CSV", str(FUSED_DIR / "prefill_kernels.csv"))
)
BENCH_BASELINE_JSON = Path(
    os.environ.get(
        "IT_BENCH_BASELINE_JSON", str(BASELINE_DIR / "bench_allreduce_none.json")
    )
)
BENCH_INT4_JSON = Path(
    os.environ.get("IT_BENCH_INT4_JSON", str(FUSED_DIR / "bench_allreduce_int4.json"))
)


# Trace files: pick rank-0 TP0 trace from each directory
def _find_trace(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("dp0_pp0_tp0_*.pt.trace.json.gz"))
    return candidates[0] if candidates else None


BASELINE_TRACE_GZ = Path(
    os.environ.get("IT_BASELINE_TRACE_GZ", str(_find_trace(BASELINE_DIR) or ""))
)
FUSED_TRACE_GZ = Path(
    os.environ.get("IT_FUSED_TRACE_GZ", str(_find_trace(FUSED_DIR) or ""))
)


def _skip_if_missing(*paths: Path):
    """Decorator: skip the test if any required data file is missing."""
    missing = [str(p) for p in paths if not p.is_file()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"Data file(s) not found: {', '.join(missing)}",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _rows_matching(rows: list[dict], pattern: str) -> list[dict]:
    """Return rows whose 'name' column contains the given substring."""
    return [r for r in rows if pattern in r.get("name", "")]


def _avg_median_dur(rows: list[dict]) -> float:
    durs = [float(r["dur_median"]) for r in rows if r.get("dur_median")]
    return sum(durs) / len(durs) if durs else 0.0


def _weighted_avg_median_dur(rows: list[dict]) -> float:
    """n_occurences-weighted average of dur_median.

    Handles CSVs where rows aggregate different numbers of kernel invocations
    (e.g. one row per step with n_occurences=1, or one aggregated row with
    n_occurences=255).  Weighting by occurrence count gives a fair per-firing
    average regardless of how the profiler grouped the data.
    """
    total_dur = sum(
        float(r["dur_median"]) * int(r.get("n_occurences", 1))
        for r in rows
        if r.get("dur_median")
    )
    total_occ = sum(int(r.get("n_occurences", 1)) for r in rows if r.get("dur_median"))
    return total_dur / total_occ if total_occ else 0.0


def _grep_trace(
    trace_path: Path, pattern: bytes, max_bytes: int = 8 * 1024 * 1024
) -> int:
    """Count occurrences of a byte pattern in the first max_bytes of a trace."""
    with gzip.open(trace_path, "rb") as f:
        data = f.read(max_bytes)
    return len(re.findall(pattern, data))


# ---------------------------------------------------------------------------
# TC-4.1  F2 fused kernel present in fused prefill trace
# ---------------------------------------------------------------------------

# The fused RMSNorm+quant kernel produced by torch.compile pattern matching
F2_KERNEL_PATTERN = "fused__to_copy_add_gemm_with_dynamic_quant_mean_mul_pow_rsqrt"


@_skip_if_missing(FUSED_PREFILL_CSV)
def test_tc4_1_f2_fused_kernel_in_prefill_csv():
    """TC-4.1: The F2 fused RMSNorm+quant kernel must appear in fused prefill CSV."""
    rows = _read_csv(FUSED_PREFILL_CSV)
    matches = _rows_matching(rows, F2_KERNEL_PATTERN)
    assert len(matches) > 0, (
        f"F2 fused kernel '{F2_KERNEL_PATTERN}' not found in {FUSED_PREFILL_CSV}. "
        f"Available kernels (first 5): {[r['name'] for r in rows[:5]]}"
    )


# ---------------------------------------------------------------------------
# TC-4.2  Standalone rms_norm_kernel absent in fused prefill trace
# ---------------------------------------------------------------------------


@_skip_if_missing(FUSED_PREFILL_CSV)
def test_tc4_2_standalone_rms_norm_absent_in_fused_prefill():
    """TC-4.2: Standalone rms_norm_kernel must be absent when F2 fusion is active."""
    rows = _read_csv(FUSED_PREFILL_CSV)
    rms_rows = _rows_matching(rows, "rms_norm_kernel")
    assert len(rms_rows) == 0, (
        f"Standalone rms_norm_kernel found {len(rms_rows)} time(s) "
        f"in {FUSED_PREFILL_CSV}. "
        "F2 fusion is not eliminating standalone RMSNorm calls."
    )


# ---------------------------------------------------------------------------
# TC-4.3  F3 fused kernel present in fused decode trace
# ---------------------------------------------------------------------------

# The fused RoPE+KV-cache kernel produced by torch.compile pattern matching
F3_KERNEL_PATTERN = "fused_add_clone_copy_expand_index_mul_neg_slice"


@_skip_if_missing(FUSED_DECODE_CSV)
def test_tc4_3_f3_fused_kernel_in_decode_csv():
    """TC-4.3: The F3 fused RoPE+KV-cache kernel must appear in fused decode CSV."""
    rows = _read_csv(FUSED_DECODE_CSV)
    matches = _rows_matching(rows, F3_KERNEL_PATTERN)
    assert len(matches) > 0, (
        f"F3 fused kernel '{F3_KERNEL_PATTERN}' not found in {FUSED_DECODE_CSV}. "
        f"Available kernels (first 5): {[r['name'] for r in rows[:5]]}"
    )


# ---------------------------------------------------------------------------
# TC-4.4  concat_and_cache_mla absent (or minimal) in fused decode trace
# ---------------------------------------------------------------------------


@_skip_if_missing(FUSED_DECODE_CSV)
def test_tc4_4_concat_mla_absent_in_fused_decode():
    """TC-4.4: concat_and_cache_mla should not dominate decode when F3 is active."""
    rows = _read_csv(FUSED_DECODE_CSV)
    concat_rows = _rows_matching(rows, "concat_and_cache_mla")

    # With torch.compile F3 fusion: only 0 or 1 warm-up entries allowed
    assert len(concat_rows) <= 1, (
        f"concat_and_cache_mla found {len(concat_rows)} row(s) in fused decode CSV. "
        "F3 fusion may not be active — unfused KV cache write still present."
    )


# ---------------------------------------------------------------------------
# TC-4.5  AllReduce average duration reduced ≥70% in INT4 vs baseline
# ---------------------------------------------------------------------------

AR_KERNEL_PATTERN = "cross_device_reduce_1stage"


@_skip_if_missing(BASELINE_DECODE_CSV, FUSED_DECODE_CSV)
def test_tc4_5_allreduce_duration_reduced():
    """TC-4.5: INT4 QuickReduce must cut AllReduce median duration by ≥70%.

    Uses n_occurences-weighted average to handle CSVs where one run stores
    one row per decode step (n_occurences=1) while another stores aggregated
    rows (n_occurences=N).  A plain row-count mean would be skewed by this
    difference in aggregation granularity.
    """
    baseline_rows = _read_csv(BASELINE_DECODE_CSV)
    fused_rows = _read_csv(FUSED_DECODE_CSV)

    baseline_ar = _rows_matching(baseline_rows, AR_KERNEL_PATTERN)
    fused_ar = _rows_matching(fused_rows, AR_KERNEL_PATTERN)

    assert baseline_ar, f"No {AR_KERNEL_PATTERN} rows in baseline CSV"
    assert fused_ar, f"No {AR_KERNEL_PATTERN} rows in fused/INT4 CSV"

    baseline_avg = _weighted_avg_median_dur(baseline_ar)
    fused_avg = _weighted_avg_median_dur(fused_ar)

    reduction = (baseline_avg - fused_avg) / baseline_avg
    assert reduction >= 0.70, (
        f"AllReduce duration reduction {reduction * 100:.1f}% < 70% threshold. "
        f"Baseline weighted avg: {baseline_avg:.2f}µs, "
        f"INT4 weighted avg: {fused_avg:.2f}µs. "
        "INT4 QuickReduce may not be active or not reducing latency as expected."
    )


# ---------------------------------------------------------------------------
# TC-4.6  qr_all_reduce kernel present in INT4 Perfetto trace
# ---------------------------------------------------------------------------


@_skip_if_missing(FUSED_TRACE_GZ)
def test_tc4_6_qr_all_reduce_in_int4_trace():
    """TC-4.6: The qr_all_reduce kernel must appear in the INT4/QuickReduce trace."""
    count = _grep_trace(FUSED_TRACE_GZ, b"qr_all_reduce")
    assert count > 0, (
        f"qr_all_reduce not found in {FUSED_TRACE_GZ}. "
        "INT4 QuickReduce kernel is not dispatching."
    )


# ---------------------------------------------------------------------------
# TC-4.7  qr_all_reduce absent from NONE (baseline) Perfetto trace
# ---------------------------------------------------------------------------


@_skip_if_missing(BASELINE_TRACE_GZ)
def test_tc4_7_qr_all_reduce_absent_from_baseline_trace():
    """TC-4.7: The baseline (NONE) trace must NOT contain qr_all_reduce."""
    count = _grep_trace(BASELINE_TRACE_GZ, b"qr_all_reduce")
    assert count == 0, (
        f"qr_all_reduce found {count} time(s) in baseline trace {BASELINE_TRACE_GZ}. "
        "The baseline run should not use INT4 QuickReduce — A/B comparison invalid."
    )


# ---------------------------------------------------------------------------
# TC-6.1  AllReduce A/B benchmark: TPOT ≥9%, TTFT ≥4% improvement
# ---------------------------------------------------------------------------


@_skip_if_missing(BENCH_BASELINE_JSON, BENCH_INT4_JSON)
def test_tc6_1_allreduce_benchmark_improvement():
    """TC-6.1: INT4 QuickReduce must improve TPOT ≥9% and TTFT ≥4% vs NONE."""
    import json

    with open(BENCH_BASELINE_JSON) as f:
        baseline = json.load(f)
    with open(BENCH_INT4_JSON) as f:
        int4 = json.load(f)

    b_tpot = baseline["mean_tpot_ms"]
    f_tpot = int4["mean_tpot_ms"]
    b_ttft = baseline["mean_ttft_ms"]
    f_ttft = int4["mean_ttft_ms"]

    tpot_imp = (b_tpot - f_tpot) / b_tpot * 100
    ttft_imp = (b_ttft - f_ttft) / b_ttft * 100

    assert tpot_imp >= 9.0, (
        f"TPOT improvement {tpot_imp:.1f}% < 9% threshold. "
        f"Baseline: {b_tpot:.1f}ms → INT4: {f_tpot:.1f}ms."
    )
    assert ttft_imp >= 4.0, (
        f"TTFT improvement {ttft_imp:.1f}% < 4% threshold. "
        f"Baseline: {b_ttft:.1f}ms → INT4: {f_ttft:.1f}ms."
    )
