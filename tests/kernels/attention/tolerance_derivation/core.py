# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for ground-truth ROCm AITER tolerance derivation.

Ground-truth here means tolerances are derived from

    atol = max_e( |ref_e - golden_e| + kernel_budget_e - rtol * |ref_e| )

over the test parametrization, without reading the kernel output to *set* the
tolerance. Kernel output is used only for *audits* (is AITER within budget?).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MARGIN = 1.5

# Maps sweep / apriori group names to rocm_aiter_tolerances.py constant names.
GROUP_ALIASES: dict[str, str] = {
    "FA_SINGLE_SEQ": "FA_SINGLE_SEQ",
    "FA_MULTI_BATCH": "FA_MULTI_BATCH",
    "FA_DECODE": "FA_DECODE",
    "FA_DIRECT": "FA_DIRECT",
    "FA_FP8_KV": "FA_FP8_KV",
    "MLA_DECODE": "MLA_DECODE",
    "MLA_DECODE_NONCONTIG": "MLA_DECODE",
    "MLA_H12_DECODE": "MLA_H12_DECODE",
    "MLA_FP8_PREFILL": "MLA_FP8_PREFILL",
    "UNIFIED_MIXED_BATCH": "UNIFIED_MIXED_BATCH",
    "UNIFIED_DECODE": "UNIFIED_DECODE",
    "UNIFIED_PREFILL": "UNIFIED_PREFILL",
    "UNIFIED_FP8_KV": "UNIFIED_FP8_KV",
    "UNIFIED_FP8_QUERY": "UNIFIED_FP8_QUERY",
    "UNIFIED_FP8_QUERY_KV": "UNIFIED_FP8_QUERY_KV",
}


def vllm_repo_root() -> Path:
    """Return the vLLM repository root (parent of ``tests/``)."""
    return Path(__file__).resolve().parents[4]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def normalize_group(name: str) -> str:
    return GROUP_ALIASES.get(name, name)


def record_dtype(rec: dict[str, Any]) -> str:
    if "dtype" in rec:
        return str(rec["dtype"])
    group = normalize_group(rec.get("group", ""))
    if group.startswith("MLA") or group.startswith("FA_DECODE"):
        return "torch.bfloat16"
    return ""


def commit_atol(atol_full_max: float, margin: float = DEFAULT_MARGIN) -> float:
    """Round committed atol to 2 significant figures (same style as existing file)."""
    raw = atol_full_max * margin
    if raw == 0:
        return 0.0
    exp = math.floor(math.log10(abs(raw)))
    sig = 2
    return round(raw, max(-int(exp) + (sig - 1), 0))


@dataclass
class AuditResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def audit_record(rec: dict[str, Any]) -> AuditResult:
    """Kernel/reference sanity checks that do not assume AITER is correct."""
    reasons: list[str] = []

    err_ker = rec.get("err_ker_max")
    err_ref = rec.get("err_ref_max")
    group = rec.get("group", "")
    # For fp32 MLA references, err_ker > err_ref vs fp64 golden is expected
    # (kernel is bf16). Only flag when ref and kernel share working precision.
    if (
        err_ker is not None
        and err_ref is not None
        and not str(group).startswith("MLA")
        and err_ker > err_ref * 1.05
    ):
        reasons.append(
            f"kernel farther from golden than ref "
            f"(err_ker={err_ker:.3g} > err_ref={err_ref:.3g})"
        )

    ker_over_det = rec.get("ker_over_det")
    if ker_over_det is not None and ker_over_det > 1.0:
        reasons.append(f"err_ker exceeds deterministic bound ({ker_over_det:.3f})")

    worst_ratio = rec.get("worst_ratio")
    if worst_ratio is not None and worst_ratio > 1.0:
        reasons.append(f"kernel-ref residual exceeds budget ({worst_ratio:.3f})")

    covers = rec.get("covers")
    if covers is not None and not covers:
        reasons.append("elementwise budget does not cover residual")

    model_gap_rel = rec.get("model_gap_rel")
    if model_gap_rel is not None and model_gap_rel > 1e-3:
        reasons.append(f"reference simulation gap too large ({model_gap_rel:.3g})")

    fa_pass = rec.get("fa_pass_strict")
    if fa_pass is not None and not fa_pass:
        reasons.append("FlashAttention upstream criterion failed")

    return AuditResult(passed=len(reasons) == 0, reasons=reasons)


@dataclass
class GroupSummary:
    group: str
    dtype: str
    n_cases: int
    atol_full_max: float
    atol_measured_max: float
    atol_committed: float
    rtol: float
    err_ker_max: float
    err_ref_max: float
    audit_failures: int
    worst_label: str
    margin: float = DEFAULT_MARGIN

    @property
    def canonical_group(self) -> str:
        return normalize_group(self.group)


def aggregate_group(
    records: list[dict[str, Any]],
    *,
    group: str,
    dtype: str,
    margin: float = DEFAULT_MARGIN,
) -> GroupSummary | None:
    rows = [
        r
        for r in records
        if normalize_group(r.get("group", "")) == normalize_group(group)
        and record_dtype(r) == dtype
    ]
    if not rows:
        return None

    atol_full_max = max(r["atol_full"] for r in rows if "atol_full" in r)
    atol_measured_max = max(
        r.get("atol_measured", 0.0) for r in rows if "atol_measured" in r
    )
    err_ker_max = max(r.get("err_ker_max", 0.0) for r in rows)
    err_ref_max = max(r.get("err_ref_max", 0.0) for r in rows)
    rtol_vals = [r["rtol"] for r in rows if "rtol" in r]
    rtol = statistics.mode(rtol_vals) if rtol_vals else 1e-2

    audit_failures = sum(1 for r in rows if not audit_record(r).passed)
    worst = max(rows, key=lambda r: r.get("atol_full", 0.0))

    return GroupSummary(
        group=group,
        dtype=dtype,
        n_cases=len(rows),
        atol_full_max=atol_full_max,
        atol_measured_max=atol_measured_max,
        atol_committed=commit_atol(atol_full_max, margin),
        rtol=rtol,
        err_ker_max=err_ker_max,
        err_ref_max=err_ref_max,
        audit_failures=audit_failures,
        worst_label=worst.get("label", worst.get("cfg", "?")),
        margin=margin,
    )


def summarize_all(
    records: list[dict[str, Any]], margin: float = DEFAULT_MARGIN
) -> list[GroupSummary]:
    keys: set[tuple[str, str]] = set()
    for r in records:
        g = normalize_group(r.get("group", ""))
        d = record_dtype(r)
        if g and d:
            keys.add((g, d))

    out: list[GroupSummary] = []
    for group, dtype in sorted(keys):
        summary = aggregate_group(records, group=group, dtype=dtype, margin=margin)
        if summary is not None:
            out.append(summary)
    return out


def print_report(summaries: list[GroupSummary]) -> None:
    print(
        f"\n{'group':24} {'dtype':16} {'n':>5} {'atol_full':>10} {'measured':>10} "
        f"{'commit':>10} {'audit_fail':>10}  worst"
    )
    for s in summaries:
        dtype_short = s.dtype.replace("torch.", "")
        print(
            f"{s.canonical_group:24} {dtype_short:16} {s.n_cases:5d} "
            f"{s.atol_full_max:10.3e} {s.atol_measured_max:10.3e} "
            f"{s.atol_committed:10.3e} {s.audit_failures:10d}  {s.worst_label[:40]}"
        )


def summaries_to_json(summaries: list[GroupSummary]) -> list[dict[str, Any]]:
    return [asdict(s) for s in summaries]
