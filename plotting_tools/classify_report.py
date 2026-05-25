"""Classification audit stats for Nsight JSONL / Chrome trace loading."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plotting_tools.classify import (
    COMM_PATTERNS,
    COMPUTE_PATTERNS,
    CONTROL_PATTERNS,
    classify_comm_operation,
    classify_event,
)

_RUNTIME_COMM_HINTS = (
    "nccl",
    "cudamemcpy",
    "cudamemcpyasync",
    "cudamemcpy2d",
)

# Text log: cap per-section listings; JSON keeps full inventories.
_TXT_TOP_N = 40


def _label_key(name: str, cat: str) -> str:
    return f"{cat}|{name[:200]}"


def _full_label_key(kind: str, sub: str, cat: str, name: str) -> str:
    return f"{kind}/{sub}|{cat}|{name[:500]}"


def _sorted_counter_dict(c: Counter) -> dict[str, int]:
    return {k: int(v) for k, v in sorted(c.items(), key=lambda x: (-x[1], x[0]))}


def _nested_by_kind_sub(c: Counter) -> dict[str, dict[str, dict[str, int]]]:
    """Group keys 'kind/sub|cat|name' into kind -> sub -> name -> count."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for key, count in c.items():
        parts = key.split("|", 2)
        if len(parts) < 3:
            continue
        head, cat, name = parts[0], parts[1], parts[2]
        if "/" not in head:
            continue
        kind, sub = head.split("/", 1)
        out.setdefault(kind, {}).setdefault(sub, {})
        label = f"{cat}|{name}" if cat else name
        out[kind][sub][label] = out[kind][sub].get(label, 0) + int(count)
    for kind in out:
        for sub in out[kind]:
            out[kind][sub] = dict(
                sorted(out[kind][sub].items(), key=lambda x: (-x[1], x[0]))
            )
    return out


def is_ambiguous_classification(name: str, cat: str, kind: str, sub: str) -> bool:
    if kind != "control" or sub != "control":
        return False
    s = f"{name} {cat}".lower()
    if cat.lower() == "kernel":
        return False
    if any(k in s for k in CONTROL_PATTERNS + COMM_PATTERNS + COMPUTE_PATTERNS):
        return False
    return True


def is_other_compute_bucket(name: str, cat: str, kind: str, sub: str) -> bool:
    return kind == "compute" and sub == "other_compute" and cat.lower() == "kernel"


def comm_op_for_parsed_event(
    name: str,
    cat: str,
    kind: str,
    *,
    args: dict[str, Any] | None = None,
) -> str | None:
    """Same rules as plots._classify_event_for_comm_breakdown (no event dict)."""
    if kind == "control":
        return classify_comm_operation(name, cat, args=args)
    if kind == "comm":
        return classify_comm_operation(name, cat, args=args)
    return None


def _comm_op_counts_from_labels(labels: Counter) -> dict[str, int]:
    """Derive op totals from comm_op_labels (single source of truth)."""
    out: Counter = Counter()
    for key, n in labels.items():
        op = key.split("|", 1)[0]
        out[op] += int(n)
    return dict(sorted(out.items(), key=lambda x: (-x[1], x[0])))


@dataclass
class ClassificationReport:
    trace_path: str
    jsonl_lines: int = 0
    events_parsed: int = 0
    skipped_rows: Counter = field(default_factory=Counter)
    skipped_runtime_names: Counter = field(default_factory=Counter)
    skipped_osrt_names: Counter = field(default_factory=Counter)
    skipped_nvtx_names: Counter = field(default_factory=Counter)
    iteration_nvtx_ranges: Counter = field(default_factory=Counter)
    comm_nvtx_records: Counter = field(default_factory=Counter)
    comm_nvtx_dedup_removed: int = 0
    comm_nvtx_phases_assigned: int = 0
    classified_kind: Counter = field(default_factory=Counter)
    classified_sub: Counter = field(default_factory=Counter)
    # Full inventory: every loaded event name → count
    classified_labels: Counter = field(default_factory=Counter)
    ambiguous_control: Counter = field(default_factory=Counter)
    other_compute_kernels: Counter = field(default_factory=Counter)
    comm_op_labels: Counter = field(default_factory=Counter)
    unclassified_comm_labels: Counter = field(default_factory=Counter)
    # Events not bucketed by comm breakdown (control, most compute, etc.)
    excluded_from_comm_breakdown: Counter = field(default_factory=Counter)

    @property
    def ambiguous_control_count(self) -> int:
        return sum(self.ambiguous_control.values())

    @property
    def skipped_runtime_count(self) -> int:
        return self.skipped_rows.get("runtime_unmatched", 0)

    def record_parsed_event(
        self,
        name: str,
        cat: str,
        kind: str,
        sub: str,
        *,
        args: dict[str, Any] | None = None,
    ) -> None:
        self.events_parsed += 1
        self.classified_kind[kind] += 1
        self.classified_sub[sub] += 1
        self.classified_labels[_full_label_key(kind, sub, cat, name)] += 1

        if is_ambiguous_classification(name, cat, kind, sub):
            self.ambiguous_control[_label_key(name, cat)] += 1
        if is_other_compute_bucket(name, cat, kind, sub):
            self.other_compute_kernels[name[:500]] += 1

        op = comm_op_for_parsed_event(name, cat, kind, args=args)
        if op is None:
            self.excluded_from_comm_breakdown[
                _full_label_key(kind, sub, cat, name)
            ] += 1
        elif op == "unclassified_comm":
            self.unclassified_comm_labels[_label_key(name, cat)] += 1
            self.comm_op_labels[f"{op}|{_label_key(name, cat)}"] += 1
        else:
            self.comm_op_labels[f"{op}|{_label_key(name, cat)}"] += 1

    def record_comm_breakdown(
        self,
        stats: dict[str, dict[str, float | int]],
        unclassified: Counter | dict[str, int],
    ) -> None:
        """Merge unclassified comm labels from plot breakdown (counts not duplicated)."""
        _ = stats
        if isinstance(unclassified, Counter):
            self.unclassified_comm_labels.update(unclassified)
        else:
            self.unclassified_comm_labels.update(unclassified)

    def inventory_dict(self) -> dict[str, Any]:
        """Full name/API inventory for this trace (no top-N truncation)."""
        return {
            "classified_by_kind_sub_name": _nested_by_kind_sub(self.classified_labels),
            "classified_labels_flat": _sorted_counter_dict(self.classified_labels),
            "comm_op_counts": _comm_op_counts_from_labels(self.comm_op_labels),
            "comm_op_labels": _sorted_counter_dict(self.comm_op_labels),
            "comm_op_counts_note": (
                "Derived from comm_op_labels (one count per event); "
                "do not add record_comm_breakdown stats."
            ),
            "unclassified_comm_labels": _sorted_counter_dict(self.unclassified_comm_labels),
            "excluded_from_comm_breakdown": _sorted_counter_dict(
                self.excluded_from_comm_breakdown
            ),
            "ambiguous_control": _sorted_counter_dict(self.ambiguous_control),
            "other_compute_kernels": _sorted_counter_dict(self.other_compute_kernels),
            "skipped_runtime_names": _sorted_counter_dict(self.skipped_runtime_names),
            "skipped_osrt_names": _sorted_counter_dict(self.skipped_osrt_names),
            "skipped_nvtx_names": _sorted_counter_dict(self.skipped_nvtx_names),
            "iteration_nvtx_ranges": _sorted_counter_dict(self.iteration_nvtx_ranges),
            "comm_nvtx_records": _sorted_counter_dict(self.comm_nvtx_records),
            "skipped_rows": dict(self.skipped_rows),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace_path,
            "jsonl_lines": self.jsonl_lines,
            "events_parsed": self.events_parsed,
            "skipped_rows": dict(self.skipped_rows),
            "skipped_runtime_count": self.skipped_runtime_count,
            "ambiguous_control_count": self.ambiguous_control_count,
            "classified_kind": dict(self.classified_kind),
            "classified_sub": dict(self.classified_sub),
            "inventory": self.inventory_dict(),
        }

    def merge(self, other: ClassificationReport) -> None:
        self.jsonl_lines += other.jsonl_lines
        self.events_parsed += other.events_parsed
        self.skipped_rows.update(other.skipped_rows)
        self.skipped_runtime_names.update(other.skipped_runtime_names)
        self.skipped_osrt_names.update(other.skipped_osrt_names)
        self.skipped_nvtx_names.update(other.skipped_nvtx_names)
        self.iteration_nvtx_ranges.update(other.iteration_nvtx_ranges)
        self.comm_nvtx_records.update(other.comm_nvtx_records)
        self.comm_nvtx_dedup_removed += other.comm_nvtx_dedup_removed
        self.comm_nvtx_phases_assigned += other.comm_nvtx_phases_assigned
        self.classified_kind.update(other.classified_kind)
        self.classified_sub.update(other.classified_sub)
        self.classified_labels.update(other.classified_labels)
        self.ambiguous_control.update(other.ambiguous_control)
        self.other_compute_kernels.update(other.other_compute_kernels)
        self.comm_op_labels.update(other.comm_op_labels)
        self.unclassified_comm_labels.update(other.unclassified_comm_labels)
        self.excluded_from_comm_breakdown.update(other.excluded_from_comm_breakdown)


def merge_reports(reports: list[ClassificationReport]) -> ClassificationReport:
    if not reports:
        return ClassificationReport(trace_path="(empty)")
    merged = ClassificationReport(trace_path="(job merged)")
    for r in reports:
        merged.merge(r)
    return merged


def _format_counter_section(
    lines: list[str],
    title: str,
    counter: Counter,
    *,
    top_n: int = _TXT_TOP_N,
) -> None:
    if not counter:
        lines.append(f"  (none)")
        return
    total = sum(counter.values())
    unique = len(counter)
    lines.append(f"  entries: {unique:,}  events: {total:,}")
    for label, n in counter.most_common(top_n):
        lines.append(f"    {n:>8}  {label}")
    if unique > top_n:
        lines.append(f"    ... +{unique - top_n:,} more in plotting_log.json → inventory")


def format_plotting_log(
    job_dir: str,
    job_meta: dict[str, Any],
    per_trace: list[ClassificationReport],
    *,
    parallel_label: str,
    merged_fabric_comm: dict[str, dict[str, float | int]] | None = None,
    merged_data_movement: dict[str, dict[str, float | int]] | None = None,
    job_inventory: dict[str, Any] | None = None,
    merged_comm: dict[str, dict[str, float | int]] | None = None,
) -> str:
    if merged_fabric_comm is None:
        merged_fabric_comm = merged_comm
    lines: list[str] = [
        "vLLM plotting / classification log",
        "=" * 72,
        f"job_dir: {job_dir}",
        f"parallel: {parallel_label}",
        f"model: {job_meta.get('model', '(unknown)')}",
        f"traces: {len(per_trace)}",
        "",
        "Full API/name inventories (all labels, no truncation): plotting_log.json",
        "  → job_inventory + traces[].inventory",
        "",
    ]
    job_merged = merge_reports(per_trace)

    for rep in per_trace:
        lines.extend([
            f"--- {Path(rep.trace_path).name} ---",
            f"  jsonl_lines: {rep.jsonl_lines:,}",
            f"  events_parsed: {rep.events_parsed:,}",
            f"  skipped_rows: {dict(rep.skipped_rows)}",
            f"  ambiguous_control: {rep.ambiguous_control_count:,}",
            f"  skipped_runtime (not loaded): {rep.skipped_runtime_count:,}",
            f"  kind: {dict(rep.classified_kind)}",
            f"  sub: {dict(rep.classified_sub)}",
            "",
            "  classified_labels (kind/sub|cat|name) top:",
        ])
        _format_counter_section(lines, "", rep.classified_labels, top_n=15)
        lines.append("  comm_op_labels top:")
        _format_counter_section(lines, "", rep.comm_op_labels, top_n=15)
        if rep.unclassified_comm_labels:
            lines.append("  unclassified_comm_labels:")
            _format_counter_section(lines, "", rep.unclassified_comm_labels, top_n=15)
        if rep.iteration_nvtx_ranges:
            lines.append("  iteration_nvtx_ranges (prefill/decode/mixed/idle):")
            _format_counter_section(lines, "", rep.iteration_nvtx_ranges, top_n=15)
        if rep.comm_nvtx_records:
            lines.append("  comm_nvtx_records (inner message markers):")
            _format_counter_section(lines, "", rep.comm_nvtx_records, top_n=15)
        if rep.skipped_runtime_names:
            lines.append("  skipped_runtime_names:")
            _format_counter_section(lines, "", rep.skipped_runtime_names, top_n=15)
        lines.append("")

    lines.extend([
        "JOB TOTALS",
        f"  events_parsed: {job_merged.events_parsed:,}",
        f"  ambiguous_control: {job_merged.ambiguous_control_count:,}",
        f"  skipped_runtime: {job_merged.skipped_runtime_count:,}",
        f"  classified_kind: {dict(job_merged.classified_kind)}",
        f"  classified_sub: {dict(job_merged.classified_sub)}",
        "",
        "JOB — classified_labels (all kinds) top:",
    ])
    _format_counter_section(lines, "", job_merged.classified_labels, top_n=_TXT_TOP_N)
    lines.extend([
        "",
        "JOB — comm_op_counts (derived from comm_op_labels):",
    ])
    _format_counter_section(
        lines,
        "",
        Counter(_comm_op_counts_from_labels(job_merged.comm_op_labels)),
        top_n=_TXT_TOP_N,
    )
    lines.extend(["", "JOB — comm_op_labels (op|cat|name) top:"])
    _format_counter_section(lines, "", job_merged.comm_op_labels, top_n=_TXT_TOP_N)
    if job_merged.unclassified_comm_labels:
        lines.extend(["", "JOB — unclassified_comm_labels:"])
        _format_counter_section(
            lines, "", job_merged.unclassified_comm_labels, top_n=_TXT_TOP_N
        )
    lines.extend(["", "JOB — skipped_runtime_names:"])
    _format_counter_section(lines, "", job_merged.skipped_runtime_names, top_n=_TXT_TOP_N)
    if job_merged.skipped_osrt_names:
        lines.extend(["", "JOB — skipped_osrt_names (not loaded):"])
        _format_counter_section(lines, "", job_merged.skipped_osrt_names, top_n=_TXT_TOP_N)
    if job_merged.iteration_nvtx_ranges:
        lines.extend(["", "JOB — iteration_nvtx_ranges (timeline attribution):"])
        _format_counter_section(
            lines, "", job_merged.iteration_nvtx_ranges, top_n=_TXT_TOP_N
        )
    if job_merged.comm_nvtx_records:
        lines.extend([
            "",
            "JOB — comm_nvtx_records (authoritative message records; "
            "logical tensor bytes):",
        ])
        _format_counter_section(
            lines, "", job_merged.comm_nvtx_records, top_n=_TXT_TOP_N
        )
    if job_merged.skipped_nvtx_names:
        lines.extend(["", "JOB — skipped_nvtx_names (PyTorch/layerwise, not phase tags):"])
        _format_counter_section(lines, "", job_merged.skipped_nvtx_names, top_n=_TXT_TOP_N)

    compat = _compatibility_note(
        job_meta,
        job_merged.classified_kind,
        job_merged.classified_sub,
        job_merged.ambiguous_control,
        job_merged.skipped_runtime_names,
    )
    lines.extend(["", "COMPATIBILITY", compat, ""])

    def _append_breakdown_section(
        heading: str, merged: dict[str, dict[str, float | int]] | None
    ) -> None:
        if not merged:
            return
        lines.extend(["", heading])
        for op, stats in sorted(
            merged.items(),
            key=lambda x: float(x[1].get("dur_us", 0)),
            reverse=True,
        ):
            dur_ms = float(stats.get("dur_us", 0)) / 1000.0
            lines.append(
                f"  {op}: count={stats.get('count', 0)} "
                f"dur_ms={dur_ms:.2f} bytes={stats.get('bytes', 0)}"
            )

    _append_breakdown_section(
        "FABRIC COMM BREAKDOWN (network_collective / network_p2p; dur_us from plots)",
        merged_fabric_comm,
    )
    _append_breakdown_section(
        "DATA MOVEMENT BREAKDOWN (device_copy, host_transfer; not fabric)",
        merged_data_movement,
    )

    if job_inventory:
        n_flat = len(job_inventory.get("classified_labels_flat") or {})
        lines.append(f"\njob_inventory: {n_flat:,} unique classified label keys in JSON")

    return "\n".join(lines) + "\n"


def _compatibility_note(
    job_meta: dict[str, Any],
    kind: Counter,
    sub: Counter,
    amb: Counter,
    skip_rt: Counter,
) -> str:
    tp = job_meta.get("tensor_parallel", 1)
    pp = job_meta.get("pipeline_parallel", 1)
    comm = kind.get("comm", 0)
    notes = [
        f"  TP={tp} PP={pp} — classifier is pattern-based; TP jobs expect "
        "all_reduce/all_gather NCCL and multi-GPU device_id in traces.",
    ]
    if comm == 0:
        notes.append("  WARNING: zero comm events parsed; check NCCL/runtime export.")
    elif tp > 1:
        if comm > 0:
            notes.append(
                f"  OK: {comm:,} comm events — compatible for TP={tp} "
                "(verify all_reduce in fabric_comm_breakdown)."
            )
    if sum(amb.values()) > comm * 0.05 and comm > 0:
        notes.append(
            f"  NOTE: {sum(amb.values()):,} ambiguous control rows — see inventory."
        )
    if sum(skip_rt.values()) > 1000:
        notes.append(
            f"  NOTE: {sum(skip_rt.values()):,} runtime API lines skipped — "
            "see inventory.skipped_runtime_names."
        )
    _ = sub
    return "\n".join(notes)


def write_plotting_log(
    out_dir: Path,
    text: str,
    reports: list[ClassificationReport],
    job_meta: dict[str, Any],
    *,
    merged_fabric_comm: dict[str, dict[str, float | int]] | None = None,
    merged_data_movement: dict[str, dict[str, float | int]] | None = None,
    job_inventory: dict[str, Any] | None = None,
    merged_comm: dict[str, dict[str, float | int]] | None = None,
) -> None:
    if merged_fabric_comm is None:
        merged_fabric_comm = merged_comm
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plotting_log.txt").write_text(text)
    payload: dict[str, Any] = {
        "job_metadata": job_meta,
        "job_inventory": job_inventory or merge_reports(reports).inventory_dict(),
        "traces": [r.to_dict() for r in reports],
    }
    if merged_fabric_comm is not None:
        payload["merged_fabric_comm_ops"] = merged_fabric_comm
        payload["merged_comm_ops"] = merged_fabric_comm
    if merged_data_movement is not None:
        payload["merged_data_movement_ops"] = merged_data_movement
    (out_dir / "plotting_log.json").write_text(json.dumps(payload, indent=2))
