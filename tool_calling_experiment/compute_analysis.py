#!/usr/bin/env python3
"""
compute_analysis.py - Compute metrics for the tool-calling experiment.

Reads from tool_calling_experiment/tool_calling.db, computes per-condition
metrics, populates the condition_metrics table, and prints a comprehensive
stdout summary.

Usage:
    python tool_calling_experiment/compute_analysis.py [--db PATH]
"""

import argparse
import os
import sqlite3
import sys
from collections import Counter, defaultdict

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

SCENE_CLASSES = [
    "nominal",
    "flagger",
    "flooded",
    "incident_zone",
    "mounted_police",
]

DB_DEFAULT = os.path.join(os.path.dirname(__file__), "tool_calling.db")

TOOL_NAME_MAP = {
    "check_scene_prior": "prior",
    "check_confusion_risk": "confusion",
    "check_scene_action_compatibility": "scene_action",
    "check_waypoint_feasibility": "waypoint",
}

TOOL_SHORT_NAMES = ["prior", "confusion", "scene_action", "waypoint"]


def _pct(num: float, denom: float, decimals: int = 1) -> str:
    """Format a percentage string safely."""
    if denom == 0:
        return "N/A"
    return f"{100.0 * num / denom:.{decimals}f}%"


def _safe_div(num: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return num / denom


def _percentile(values: list, p: float) -> float:
    """Compute p-th percentile (0-100) of a list."""
    if not values:
        return 0.0
    sv = sorted(values)
    k = (len(sv) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(sv):
        return sv[-1]
    return sv[f] + (k - f) * (sv[c] - sv[f])


# -------------------------------------------------------------------
# Metric computation per experiment
# -------------------------------------------------------------------


def compute_scene_accuracy(rows: list[dict]) -> float:
    """Fraction where final_scene == scene_type_gt."""
    if not rows:
        return 0.0
    correct = sum(
        1
        for r in rows
        if r["final_scene"] == r["scene_type_gt"]
    )
    return correct / len(rows)


def compute_macro_f1(rows: list[dict]) -> tuple[float, dict]:
    """Per-class precision/recall/F1, then macro average.

    Returns (macro_f1, per_class_dict).
    """
    per_class = {}
    for cls in SCENE_CLASSES:
        tp = sum(
            1
            for r in rows
            if r["final_scene"] == cls
            and r["scene_type_gt"] == cls
        )
        fp = sum(
            1
            for r in rows
            if r["final_scene"] == cls
            and r["scene_type_gt"] != cls
        )
        fn = sum(
            1
            for r in rows
            if r["final_scene"] != cls
            and r["scene_type_gt"] == cls
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(
            2 * precision * recall, precision + recall
        )
        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    f1_sum = sum(c["f1"] for c in per_class.values())
    macro_f1 = f1_sum / len(SCENE_CLASSES)
    return macro_f1, per_class


def compute_revision_stats(rows: list[dict]) -> dict:
    """Compute revision counts and accuracy."""
    total = len(rows)
    n_revised = sum(1 for r in rows if r["was_revised"])
    revised_correct = sum(
        1
        for r in rows
        if r["was_revised"]
        and r["final_scene"] == r["scene_type_gt"]
    )
    return {
        "n_revised": n_revised,
        "revision_rate": _safe_div(n_revised, total),
        "revision_accuracy": _safe_div(
            revised_correct, n_revised
        ),
    }


def compute_flip_analysis(rows: list[dict]) -> dict:
    """Compute saves, breaks, and net improvement."""
    n_saves = 0
    n_breaks = 0
    for r in rows:
        orig_ok = r["original_scene"] == r["scene_type_gt"]
        final_ok = r["final_scene"] == r["scene_type_gt"]
        if not orig_ok and final_ok:
            n_saves += 1
        elif orig_ok and not final_ok:
            n_breaks += 1
    return {
        "n_saves": n_saves,
        "n_breaks": n_breaks,
        "net_improvement": n_saves - n_breaks,
    }


def compute_per_tool_effectiveness(
    conn: sqlite3.Connection,
    experiment_id: str,
    prediction_rows: list[dict],
) -> dict:
    """Per-tool call counts and conditional revision rates.

    For each tool, compute:
      - call_count: total invocations across all samples
      - revision_rate: P(was_revised=1 | tool called)
    """
    revised_samples = {
        r["sample_id"]
        for r in prediction_rows
        if r["was_revised"]
    }

    tool_calls = conn.execute(
        "SELECT sample_id, tool_name "
        "FROM tool_calls WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchall()

    tool_samples: dict[str, set] = defaultdict(set)
    tool_counts: Counter = Counter()
    for sample_id, tool_name in tool_calls:
        short = TOOL_NAME_MAP.get(tool_name, tool_name)
        tool_samples[short].add(sample_id)
        tool_counts[short] += 1

    results = {}
    for short_name in TOOL_SHORT_NAMES:
        samples_w = tool_samples.get(short_name, set())
        call_count = tool_counts.get(short_name, 0)
        rev_given = len(samples_w & revised_samples)
        cond_rev = _safe_div(rev_given, len(samples_w))
        results[short_name] = {
            "call_count": call_count,
            "revision_rate": cond_rev,
            "n_samples_called": len(samples_w),
            "n_revised_given_called": rev_given,
        }
    return results


def compute_latency(rows: list[dict]) -> dict:
    """Compute mean and p95 latency stats."""
    pred_t = [
        r["predict_time_ms"]
        for r in rows
        if r["predict_time_ms"] is not None
    ]
    ver_t = [
        r["verify_time_ms"]
        for r in rows
        if r["verify_time_ms"] is not None
    ]
    tot_t = [
        r["total_time_ms"]
        for r in rows
        if r["total_time_ms"] is not None
    ]
    return {
        "mean_predict_time_ms": _safe_div(
            sum(pred_t), len(pred_t)
        ),
        "mean_verify_time_ms": _safe_div(
            sum(ver_t), len(ver_t)
        ),
        "mean_total_time_ms": _safe_div(
            sum(tot_t), len(tot_t)
        ),
        "p95_total_time_ms": _percentile(tot_t, 95),
    }


def build_confusion_matrix(
    rows: list[dict], field: str = "final_scene"
) -> dict:
    """Build {gt_class: {pred_class: count}}."""
    matrix = {
        gt: {pred: 0 for pred in SCENE_CLASSES}
        for gt in SCENE_CLASSES
    }
    for r in rows:
        gt = r["scene_type_gt"]
        pred = r.get(field, "")
        if gt in SCENE_CLASSES and pred in SCENE_CLASSES:
            matrix[gt][pred] += 1
    return matrix


# -------------------------------------------------------------------
# DB interaction
# -------------------------------------------------------------------


def get_experiments(conn: sqlite3.Connection) -> list[dict]:
    """Fetch all experiments."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM experiments ORDER BY created_at"
    ).fetchall()
    return [dict(r) for r in rows]


def get_predictions(
    conn: sqlite3.Connection, experiment_id: str
) -> list[dict]:
    """Fetch all predictions for an experiment."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def upsert_condition_metrics(
    conn: sqlite3.Connection,
    experiment_id: str,
    metrics: dict,
):
    """Insert or replace condition_metrics row."""
    conn.execute(
        "DELETE FROM condition_metrics WHERE experiment_id = ?",
        (experiment_id,),
    )
    cols = [
        "experiment_id",
        "scene_accuracy",
        "scene_macro_f1",
        "n_revised",
        "revision_rate",
        "revision_accuracy",
        "n_saves",
        "n_breaks",
        "net_improvement",
        "tool_prior_call_count",
        "tool_prior_revision_rate",
        "tool_confusion_call_count",
        "tool_confusion_revision_rate",
        "tool_scene_action_call_count",
        "tool_scene_action_revision_rate",
        "tool_waypoint_call_count",
        "tool_waypoint_revision_rate",
        "mean_predict_time_ms",
        "mean_verify_time_ms",
        "mean_total_time_ms",
        "p95_total_time_ms",
    ]
    vals = [
        experiment_id,
        metrics["scene_accuracy"],
        metrics["scene_macro_f1"],
        metrics["n_revised"],
        metrics["revision_rate"],
        metrics["revision_accuracy"],
        metrics["n_saves"],
        metrics["n_breaks"],
        metrics["net_improvement"],
        metrics.get("tool_prior_call_count", 0),
        metrics.get("tool_prior_revision_rate", 0.0),
        metrics.get("tool_confusion_call_count", 0),
        metrics.get("tool_confusion_revision_rate", 0.0),
        metrics.get("tool_scene_action_call_count", 0),
        metrics.get("tool_scene_action_revision_rate", 0.0),
        metrics.get("tool_waypoint_call_count", 0),
        metrics.get("tool_waypoint_revision_rate", 0.0),
        metrics.get("mean_predict_time_ms", 0.0),
        metrics.get("mean_verify_time_ms", 0.0),
        metrics.get("mean_total_time_ms", 0.0),
        metrics.get("p95_total_time_ms", 0.0),
    ]
    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)
    sql = (
        f"INSERT INTO condition_metrics ({col_str}) "
        f"VALUES ({placeholders})"
    )
    conn.execute(sql, vals)
    conn.commit()


# -------------------------------------------------------------------
# Printing / formatting
# -------------------------------------------------------------------


def print_divider(char: str = "=", width: int = 88):
    print(char * width)


def print_section(title: str):
    print()
    print_divider()
    print(f"  {title}")
    print_divider()


def print_table(
    headers: list[str],
    rows: list[list],
    col_widths: list[int] | None = None,
):
    """Print a simple text table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    header_str = ""
    for i, h in enumerate(headers):
        header_str += str(h).ljust(col_widths[i])
    print(header_str)
    print("-" * sum(col_widths))

    for row in rows:
        row_str = ""
        for i, val in enumerate(row):
            if i < len(col_widths):
                row_str += str(val).ljust(col_widths[i])
        print(row_str)


def format_confusion_matrix(
    matrix: dict, label: str = ""
) -> str:
    """Format confusion matrix as string."""
    lines = []
    if label:
        lines.append(f"  {label}")
    header = "  GT \\ Pred".ljust(22) + "".join(
        c[:8].ljust(10) for c in SCENE_CLASSES
    )
    lines.append(header)
    lines.append(
        "  " + "-" * (20 + 10 * len(SCENE_CLASSES))
    )
    for gt in SCENE_CLASSES:
        row_str = f"  {gt[:18]}".ljust(22)
        for pred in SCENE_CLASSES:
            count = matrix[gt][pred]
            row_str += str(count).ljust(10)
        lines.append(row_str)
    return "\n".join(lines)


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------


def _collect_experiment_metrics(conn, experiments):
    """Compute and store metrics for all experiments.

    Returns (all_metrics, all_predictions, all_per_class).
    """
    all_metrics = {}
    all_predictions = {}
    all_per_class = {}

    for exp in experiments:
        eid = exp["experiment_id"]
        cond = exp["condition_name"]
        pipeline = exp["pipeline"]
        verifier = exp.get("verifier_model", "N/A")

        rows = get_predictions(conn, eid)
        if not rows:
            print(
                f"\n  WARNING: No predictions for "
                f"experiment {eid}, skipping."
            )
            continue

        all_predictions[eid] = rows

        accuracy = compute_scene_accuracy(rows)
        macro_f1, per_class = compute_macro_f1(rows)
        revision = compute_revision_stats(rows)
        flips = compute_flip_analysis(rows)
        tool_eff = compute_per_tool_effectiveness(
            conn, eid, rows
        )
        latency = compute_latency(rows)

        metrics = {
            "condition_name": cond,
            "pipeline": pipeline,
            "verifier_model": verifier,
            "total_samples": len(rows),
            "scene_accuracy": accuracy,
            "scene_macro_f1": macro_f1,
            **revision,
            **flips,
            "per_tool": tool_eff,
            **latency,
        }
        # Flatten per-tool into top-level keys
        for sn in TOOL_SHORT_NAMES:
            cc_key = f"tool_{sn}_call_count"
            rr_key = f"tool_{sn}_revision_rate"
            metrics[cc_key] = tool_eff[sn]["call_count"]
            metrics[rr_key] = tool_eff[sn]["revision_rate"]

        all_metrics[eid] = metrics
        all_per_class[eid] = per_class

        upsert_condition_metrics(conn, eid, metrics)

    return all_metrics, all_predictions, all_per_class


def _print_summary_table(all_metrics):
    """Section 1: per-condition summary table."""
    print_section("1. PER-CONDITION SUMMARY")

    headers = [
        "Experiment",
        "Cond",
        "Pipe",
        "N",
        "Acc",
        "F1",
        "Rev%",
        "RevAcc",
        "Saves",
        "Breaks",
        "Net",
    ]
    rows = []
    for eid, m in all_metrics.items():
        short_eid = eid[:28]
        if len(eid) > 28:
            short_eid += ".."
        rev_acc = (
            f"{m['revision_accuracy']:.3f}"
            if m["n_revised"] > 0
            else "N/A"
        )
        rows.append([
            short_eid,
            m["condition_name"],
            m["pipeline"],
            m["total_samples"],
            f"{m['scene_accuracy']:.4f}",
            f"{m['scene_macro_f1']:.4f}",
            _pct(m["n_revised"], m["total_samples"]),
            rev_acc,
            m["n_saves"],
            m["n_breaks"],
            m["net_improvement"],
        ])
    print_table(headers, rows)


def _print_per_class_f1(all_metrics, all_per_class):
    """Section 2: per-class F1 breakdown."""
    print_section("2. PER-CLASS F1 BREAKDOWN")

    for eid, per_class in all_per_class.items():
        cond = all_metrics[eid]["condition_name"]
        print(f"\n  Condition: {cond} ({eid})")
        headers = [
            "Class", "TP", "FP", "FN",
            "Prec", "Recall", "F1",
        ]
        rows = []
        for cls in SCENE_CLASSES:
            c = per_class[cls]
            rows.append([
                cls,
                c["tp"],
                c["fp"],
                c["fn"],
                f"{c['precision']:.3f}",
                f"{c['recall']:.3f}",
                f"{c['f1']:.3f}",
            ])
        print_table(headers, rows)


def _print_tool_effectiveness(all_metrics):
    """Section 3: per-tool effectiveness."""
    print_section("3. PER-TOOL EFFECTIVENESS")

    headers = [
        "Experiment", "Cond", "Tool",
        "Calls", "Samples", "CondRevRate",
    ]
    rows = []
    for eid, m in all_metrics.items():
        for sn in TOOL_SHORT_NAMES:
            t = m["per_tool"][sn]
            if t["call_count"] > 0:
                rows.append([
                    eid[:23],
                    m["condition_name"],
                    sn,
                    t["call_count"],
                    t["n_samples_called"],
                    f"{t['revision_rate']:.3f}",
                ])
    if rows:
        print_table(headers, rows)
    else:
        print("  No tool calls recorded.")


def _find_comparison_pairs(all_metrics):
    """Find 2B vs 8B pairs by condition name."""
    condition_groups = defaultdict(list)
    for eid, m in all_metrics.items():
        base_cond = m["condition_name"]
        condition_groups[base_cond].append((eid, m))

    pairs = []
    for base_cond, entries in condition_groups.items():
        entry_2b = None
        entry_8b = None
        for eid, m in entries:
            v = str(m.get("verifier_model", "")).lower()
            eid_lower = eid.lower()
            if "2b" in v or "2b" in eid_lower:
                entry_2b = (eid, m)
            elif "8b" in v or "8b" in eid_lower:
                entry_8b = (eid, m)
        if entry_2b and entry_8b:
            pairs.append((base_cond, entry_2b, entry_8b))
    return pairs


def _print_2b_vs_8b(all_metrics):
    """Section 4: 2B vs 8B verifier comparison."""
    print_section("4. 2B vs 8B VERIFIER COMPARISON")

    pairs = _find_comparison_pairs(all_metrics)
    if not pairs:
        print("  No 2B vs 8B comparison pairs found.")
        print(
            "  (Pairs identified by '2b'/'8b' in "
            "verifier model path or experiment ID)"
        )
        return pairs

    headers = ["Condition", "Metric", "2B", "8B", "Delta"]
    rows = []
    compare_defs = [
        ("Accuracy", "scene_accuracy", ".4f"),
        ("Macro F1", "scene_macro_f1", ".4f"),
        ("Rev Rate", "revision_rate", ".3f"),
        ("Rev Acc", "revision_accuracy", ".3f"),
        ("Net Impr", "net_improvement", "d"),
    ]
    for base_cond, (_eid_2b, m_2b), (_eid_8b, m_8b) in pairs:
        for label, key, fmt in compare_defs:
            v2b = m_2b[key]
            v8b = m_8b[key]
            delta = v8b - v2b
            show_cond = base_cond if label == "Accuracy" else ""
            if fmt == "d":
                s2 = str(v2b)
                s8 = str(v8b)
                sd = f"{delta:+d}"
            else:
                s2 = f"{v2b:{fmt}}"
                s8 = f"{v8b:{fmt}}"
                sd = f"{delta:+{fmt}}"
            rows.append([show_cond, label, s2, s8, sd])
        rows.append(["", "", "", "", ""])
    print_table(headers, rows)
    return pairs


def _print_per_class_accuracy(all_metrics, all_predictions):
    """Section 5: per-class accuracy by condition."""
    print_section("5. PER-CLASS ACCURACY BY CONDITION")

    headers = ["Condition"] + SCENE_CLASSES
    rows = []
    for eid, m in all_metrics.items():
        preds = all_predictions[eid]
        class_counts = Counter(
            r["scene_type_gt"] for r in preds
        )
        class_correct: Counter = Counter()
        for r in preds:
            if r["final_scene"] == r["scene_type_gt"]:
                class_correct[r["scene_type_gt"]] += 1
        row_data = [m["condition_name"]]
        for cls in SCENE_CLASSES:
            tot = class_counts.get(cls, 0)
            cor = class_correct.get(cls, 0)
            row_data.append(_pct(cor, tot))
        rows.append(row_data)
    print_table(headers, rows)


def _print_confusion_matrices(all_metrics, all_predictions):
    """Section 6: confusion matrix comparison."""
    print_section(
        "6. CONFUSION MATRIX COMPARISON (ORIG vs FINAL)"
    )

    for eid, m in all_metrics.items():
        preds = all_predictions[eid]
        if m["n_revised"] == 0:
            cond = m["condition_name"]
            print(
                f"\n  {cond}: No revisions, "
                "original == final"
            )
            continue

        orig_cm = build_confusion_matrix(
            preds, field="original_scene"
        )
        final_cm = build_confusion_matrix(
            preds, field="final_scene"
        )

        cond = m["condition_name"]
        print(f"\n  Condition: {cond} ({eid})")
        print()
        print(
            format_confusion_matrix(
                orig_cm, label="ORIGINAL predictions:"
            )
        )
        print()
        print(
            format_confusion_matrix(
                final_cm,
                label="FINAL predictions (after verify):",
            )
        )

        print("\n  Changes (Final - Original):")
        hdr = "  GT \\ Pred".ljust(22) + "".join(
            c[:8].ljust(10) for c in SCENE_CLASSES
        )
        print(hdr)
        sep_len = 20 + 10 * len(SCENE_CLASSES)
        print("  " + "-" * sep_len)
        for gt in SCENE_CLASSES:
            row_str = f"  {gt[:18]}".ljust(22)
            for pred in SCENE_CLASSES:
                d = final_cm[gt][pred] - orig_cm[gt][pred]
                if d == 0:
                    row_str += ".".ljust(10)
                elif d > 0:
                    row_str += f"+{d}".ljust(10)
                else:
                    row_str += f"{d}".ljust(10)
            print(row_str)


def _print_flip_detail(all_metrics, all_predictions):
    """Section 7: flip analysis detail."""
    print_section("7. FLIP ANALYSIS DETAIL")

    for eid, m in all_metrics.items():
        preds = all_predictions[eid]
        if m["n_saves"] == 0 and m["n_breaks"] == 0:
            cond = m["condition_name"]
            print(
                f"\n  {cond}: No flips "
                "(saves=0, breaks=0)"
            )
            continue

        cond = m["condition_name"]
        print(f"\n  Condition: {cond}")
        print(
            f"    Saves (wrong->correct):  "
            f"{m['n_saves']}"
        )
        print(
            f"    Breaks (correct->wrong): "
            f"{m['n_breaks']}"
        )
        print(
            f"    Net improvement:         "
            f"{m['net_improvement']:+d}"
        )

        saves_by = Counter()
        breaks_by = Counter()
        for r in preds:
            orig_ok = (
                r["original_scene"] == r["scene_type_gt"]
            )
            final_ok = (
                r["final_scene"] == r["scene_type_gt"]
            )
            if not orig_ok and final_ok:
                saves_by[r["scene_type_gt"]] += 1
            elif orig_ok and not final_ok:
                breaks_by[r["scene_type_gt"]] += 1

        headers = ["Class", "Saves", "Breaks", "Net"]
        rows = []
        for cls in SCENE_CLASSES:
            s = saves_by.get(cls, 0)
            b = breaks_by.get(cls, 0)
            rows.append([cls, s, b, s - b])
        print_table(headers, rows)

        # Saves breakdown
        print(
            "\n    Saves breakdown "
            "(original_scene -> scene_type_gt):"
        )
        save_trans = Counter()
        for r in preds:
            orig_ok = (
                r["original_scene"] == r["scene_type_gt"]
            )
            final_ok = (
                r["final_scene"] == r["scene_type_gt"]
            )
            if not orig_ok and final_ok:
                key = (
                    r["original_scene"],
                    r["scene_type_gt"],
                )
                save_trans[key] += 1
        for (orig, gt), cnt in save_trans.most_common(10):
            print(f"      {orig} -> {gt}: {cnt}")

        # Breaks breakdown
        print(
            "\n    Breaks breakdown "
            "(orig -> final, GT=orig):"
        )
        break_trans = Counter()
        for r in preds:
            orig_ok = (
                r["original_scene"] == r["scene_type_gt"]
            )
            final_ok = (
                r["final_scene"] == r["scene_type_gt"]
            )
            if orig_ok and not final_ok:
                key = (
                    r["original_scene"],
                    r["final_scene"],
                )
                break_trans[key] += 1
        for (orig, fin), cnt in break_trans.most_common(10):
            print(
                f"      {orig} was correct, "
                f"changed to {fin}: {cnt}"
            )


def _print_latency(all_metrics):
    """Section 8: latency summary."""
    print_section("8. LATENCY SUMMARY")

    headers = [
        "Condition",
        "MeanPred(ms)",
        "MeanVerify(ms)",
        "MeanTotal(ms)",
        "P95Total(ms)",
    ]
    rows = []
    for _eid, m in all_metrics.items():
        def _fmt(val):
            return f"{val:.1f}" if val is not None else "N/A"

        rows.append([
            m["condition_name"],
            _fmt(m["mean_predict_time_ms"]),
            _fmt(m["mean_verify_time_ms"]),
            _fmt(m["mean_total_time_ms"]),
            _fmt(m["p95_total_time_ms"]),
        ])
    print_table(headers, rows)


def _print_ranking(all_metrics):
    """Section 9: condition ranking by macro F1."""
    print_section("9. CONDITION RANKING BY MACRO F1")

    ranked = sorted(
        all_metrics.items(),
        key=lambda x: x[1]["scene_macro_f1"],
        reverse=True,
    )
    headers = [
        "Rank", "Condition", "MacroF1",
        "Accuracy", "NetImpr",
    ]
    rows = []
    for i, (_eid, m) in enumerate(ranked, 1):
        rows.append([
            i,
            m["condition_name"],
            f"{m['scene_macro_f1']:.4f}",
            f"{m['scene_accuracy']:.4f}",
            f"{m['net_improvement']:+d}",
        ])
    print_table(headers, rows)
    return ranked


def _print_decision_summary(all_metrics, ranked, pairs):
    """Section 10: quick decision summary."""
    print_section("10. QUICK DECISION SUMMARY")

    if not ranked:
        print("  No conditions to summarize.")
        return

    _best_eid, best_m = ranked[0]

    print(
        f"  Best condition: {best_m['condition_name']} "
        f"(Macro F1 = {best_m['scene_macro_f1']:.4f})"
    )

    baseline_m = None
    for _eid, m in all_metrics.items():
        if m["condition_name"] == "baseline":
            baseline_m = m
            break

    if baseline_m:
        delta = (
            best_m["scene_macro_f1"]
            - baseline_m["scene_macro_f1"]
        )
        print(
            f"  Baseline Macro F1: "
            f"{baseline_m['scene_macro_f1']:.4f}"
        )
        print(f"  Improvement over baseline: {delta:+.4f}")
    else:
        print("  (No baseline condition found)")

    oracle_ms = [
        m
        for m in all_metrics.values()
        if "oracle" in m["condition_name"]
    ]
    if oracle_ms:
        best_o = max(
            oracle_ms, key=lambda x: x["scene_macro_f1"]
        )
        cond = best_o["condition_name"]
        f1 = best_o["scene_macro_f1"]
        print(f"  Oracle ceiling: {f1:.4f} ({cond})")
        if f1 < 0.70:
            print(
                "  ** WARNING: Oracle ceiling < 0.70 "
                "-- tool-calling may not be viable **"
            )
        else:
            print(
                "  Oracle ceiling >= 0.70 "
                "-- tool-calling has potential"
            )

    if pairs:
        print("\n  2B vs 8B Summary:")
        for base_cond, (_e2, m2), (_e8, m8) in pairs:
            d = m8["scene_macro_f1"] - m2["scene_macro_f1"]
            if d > 0:
                winner = "8B"
            elif d < 0:
                winner = "2B"
            else:
                winner = "tie"
            f1_2b = m2["scene_macro_f1"]
            f1_8b = m8["scene_macro_f1"]
            print(
                f"    {base_cond}: "
                f"2B={f1_2b:.4f} vs 8B={f1_8b:.4f} "
                f"(delta={d:+.4f}, winner={winner})"
            )

    net = best_m["net_improvement"]
    sv = best_m["n_saves"]
    br = best_m["n_breaks"]
    print(
        f"\n  Net improvement of best condition: "
        f"{net:+d} (saves={sv}, breaks={br})"
    )


def run_analysis(db_path: str):
    """Run the full analysis pipeline."""
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    experiments = get_experiments(conn)
    if not experiments:
        print("ERROR: No experiments found in database.")
        conn.close()
        sys.exit(1)

    print()
    print_divider("*")
    print(
        "  TOOL-CALLING VERIFICATION EXPERIMENT "
        "-- ANALYSIS REPORT"
    )
    print_divider("*")
    print(f"  Database: {db_path}")
    print(f"  Experiments found: {len(experiments)}")

    (
        all_metrics,
        all_predictions,
        all_per_class,
    ) = _collect_experiment_metrics(conn, experiments)

    _print_summary_table(all_metrics)
    _print_per_class_f1(all_metrics, all_per_class)
    _print_tool_effectiveness(all_metrics)
    pairs = _print_2b_vs_8b(all_metrics)
    _print_per_class_accuracy(all_metrics, all_predictions)
    _print_confusion_matrices(all_metrics, all_predictions)
    _print_flip_detail(all_metrics, all_predictions)
    _print_latency(all_metrics)
    ranked = _print_ranking(all_metrics)
    _print_decision_summary(all_metrics, ranked, pairs)

    print()
    print_divider("*")
    print("  ANALYSIS COMPLETE")
    print_divider("*")
    print()

    conn.close()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute metrics for tool-calling "
            "verification experiment"
        )
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DB_DEFAULT,
        help="Path to SQLite database",
    )
    args = parser.parse_args()
    run_analysis(args.db)


if __name__ == "__main__":
    main()
