#!/usr/bin/env python3
"""
generate_insights_report.py - Human-readable markdown insight report.

Reads from tool_calling_experiment/tool_calling.db and generates a
comprehensive markdown report at
tool_calling_experiment/TOOL_CALLING_INSIGHTS_REPORT.md.

10 sections covering oracle ceiling, tool effectiveness, 2B vs 8B
comparison, flip analysis, per-class impact, case studies, and
a final go/no-go recommendation.

Usage:
    python tool_calling_experiment/generate_insights_report.py \
        [--db PATH] [--output PATH]
"""

import argparse
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime

DB_DEFAULT = os.path.join(
    os.path.dirname(__file__), "tool_calling.db"
)
OUTPUT_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "TOOL_CALLING_INSIGHTS_REPORT.md",
)

SCENE_CLASSES = [
    "nominal",
    "flagger",
    "flooded",
    "incident_zone",
    "mounted_police",
]


# ---------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------


def _dict_rows(conn, sql, params=()):
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, params)]


def get_experiments(conn):
    return _dict_rows(
        conn,
        "SELECT * FROM experiments ORDER BY created_at",
    )


def get_metrics(conn):
    rows = _dict_rows(
        conn,
        "SELECT cm.*, e.condition_name, e.pipeline, "
        "e.verifier_model "
        "FROM condition_metrics cm "
        "JOIN experiments e "
        "ON cm.experiment_id = e.experiment_id "
        "ORDER BY e.created_at",
    )
    return {r["experiment_id"]: r for r in rows}


def get_predictions(conn, experiment_id):
    return _dict_rows(
        conn,
        "SELECT * FROM predictions "
        "WHERE experiment_id = ?",
        (experiment_id,),
    )


def get_all_predictions(conn):
    rows = _dict_rows(
        conn, "SELECT * FROM predictions"
    )
    by_exp = defaultdict(list)
    for r in rows:
        by_exp[r["experiment_id"]].append(r)
    return by_exp


def get_tool_calls(conn, experiment_id, sample_id):
    return _dict_rows(
        conn,
        "SELECT * FROM tool_calls "
        "WHERE experiment_id = ? AND sample_id = ? "
        "ORDER BY tool_call_order",
        (experiment_id, sample_id),
    )


# ---------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------


def _safe_div(n, d):
    return n / d if d else 0.0


def _pct(n, d, dec=1):
    if d == 0:
        return "N/A"
    return f"{100.0 * n / d:.{dec}f}%"


def _find_pairs(metrics):
    """Find 2B/8B pairs grouped by base condition."""
    groups = defaultdict(list)
    for eid, m in metrics.items():
        groups[m["condition_name"]].append((eid, m))

    pairs = []
    for cond, entries in groups.items():
        e2 = e8 = None
        for eid, m in entries:
            v = str(m.get("verifier_model", "")).lower()
            el = eid.lower()
            if "2b" in v or "2b" in el:
                e2 = (eid, m)
            elif "8b" in v or "8b" in el:
                e8 = (eid, m)
        if e2 and e8:
            pairs.append((cond, e2, e8))
    return pairs


def _md_table(headers, rows):
    """Return a markdown table string."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append(
        "| " + " | ".join("---" for _ in headers) + " |"
    )
    for row in rows:
        lines.append(
            "| " + " | ".join(str(c) for c in row) + " |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------


def _section_1_executive(metrics, pairs, all_preds):
    """Executive Summary."""
    lines = [
        "## 1. Executive Summary",
        "",
    ]

    if not metrics:
        lines.append("No experimental data available.")
        return "\n".join(lines)

    ranked = sorted(
        metrics.items(),
        key=lambda x: x[1].get("scene_macro_f1", 0),
        reverse=True,
    )
    best_eid, best = ranked[0]

    baseline = None
    for _eid, m in metrics.items():
        if m["condition_name"] == "baseline":
            baseline = m
            break

    lines.append(
        f"**Best condition: {best['condition_name']}** "
        f"with Macro F1 = {best.get('scene_macro_f1', 0):.4f} "
        f"and accuracy = "
        f"{best.get('scene_accuracy', 0):.4f}."
    )
    lines.append("")

    if baseline:
        bl_f1 = baseline.get("scene_macro_f1", 0)
        delta = best.get("scene_macro_f1", 0) - bl_f1
        lines.append(
            f"Baseline Macro F1: {bl_f1:.4f}. "
            f"Best improvement over baseline: "
            f"{delta:+.4f}."
        )
    lines.append("")

    # Oracle ceiling
    oracles = [
        m
        for m in metrics.values()
        if "oracle" in m["condition_name"]
    ]
    if oracles:
        best_o = max(
            oracles,
            key=lambda x: x.get("scene_macro_f1", 0),
        )
        o_f1 = best_o.get("scene_macro_f1", 0)
        lines.append(
            f"**Oracle ceiling**: {o_f1:.4f} "
            f"({best_o['condition_name']}). "
        )
        if o_f1 >= 0.70:
            lines.append(
                "The oracle ceiling is above 0.70, "
                "indicating tool-calling has real "
                "potential to improve predictions."
            )
        else:
            lines.append(
                "**Warning**: The oracle ceiling is "
                "below 0.70. Even with perfect tool "
                "information, gains are limited."
            )
    lines.append("")

    # 2B vs 8B quick verdict
    if pairs:
        wins_8b = 0
        wins_2b = 0
        for _cond, (_e2, m2), (_e8, m8) in pairs:
            f2 = m2.get("scene_macro_f1", 0)
            f8 = m8.get("scene_macro_f1", 0)
            if f8 > f2:
                wins_8b += 1
            elif f2 > f8:
                wins_2b += 1
        if wins_8b > wins_2b:
            lines.append(
                f"**2B vs 8B verdict**: 8B verifier wins "
                f"in {wins_8b}/{len(pairs)} conditions. "
                "The larger model is a more effective "
                "tool-calling verifier."
            )
        elif wins_2b > wins_8b:
            lines.append(
                f"**2B vs 8B verdict**: 2B verifier wins "
                f"in {wins_2b}/{len(pairs)} conditions. "
                "The smaller model is surprisingly "
                "competitive."
            )
        else:
            lines.append(
                "**2B vs 8B verdict**: Both verifier "
                "sizes perform similarly across "
                "conditions."
            )
    lines.append("")

    # Net improvement
    net = best.get("net_improvement", 0)
    sv = best.get("n_saves", 0)
    br = best.get("n_breaks", 0)
    lines.append(
        f"Best condition net improvement: "
        f"**{net:+d}** (saves={sv}, breaks={br})."
    )
    lines.append("")

    # Plain language
    lines.append("---")
    lines.append("")
    lines.append(
        "*In plain language*: We tested whether giving "
        "the model access to statistical tools (like "
        "\"how common is this scene type?\") could help "
        "it fix its own mistakes. "
    )
    total = best.get("n_revised", 0)
    if total > 0:
        lines.append(
            f"The model revised {total} predictions. "
            f"Of those, it fixed {sv} mistakes but "
            f"also introduced {br} new errors, "
            f"for a net gain of {net:+d}."
        )
    lines.append("")
    return "\n".join(lines)


def _section_2_oracle(metrics):
    """Oracle Ceiling Analysis."""
    lines = [
        "## 2. Oracle Ceiling Analysis",
        "",
    ]
    oracles = {
        eid: m
        for eid, m in metrics.items()
        if "oracle" in m["condition_name"]
    }
    if not oracles:
        lines.append("No oracle conditions found.")
        return "\n".join(lines)

    lines.append(
        "The oracle condition replaces real tool outputs "
        "with ground-truth information. This measures "
        "the **maximum possible improvement** if tools "
        "were perfect."
    )
    lines.append("")

    headers = [
        "Condition", "Accuracy", "Macro F1",
        "Saves", "Breaks", "Net", "Rev Rate",
    ]
    rows = []
    for eid, m in oracles.items():
        rows.append([
            m["condition_name"],
            f"{m.get('scene_accuracy', 0):.4f}",
            f"{m.get('scene_macro_f1', 0):.4f}",
            m.get("n_saves", 0),
            m.get("n_breaks", 0),
            m.get("net_improvement", 0),
            f"{m.get('revision_rate', 0):.3f}",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    baseline = None
    for m in metrics.values():
        if m["condition_name"] == "baseline":
            baseline = m
            break

    best_o = max(
        oracles.values(),
        key=lambda x: x.get("scene_macro_f1", 0),
    )
    o_f1 = best_o.get("scene_macro_f1", 0)
    if baseline:
        bl_f1 = baseline.get("scene_macro_f1", 0)
        room = o_f1 - bl_f1
        lines.append(
            f"**Room for improvement**: "
            f"{room:+.4f} F1 points "
            f"(from {bl_f1:.4f} to {o_f1:.4f})."
        )
    lines.append("")
    lines.append(
        "*Plain language*: If we had a magic tool that "
        "always told the model the right answer, "
        f"the best F1 we could reach is {o_f1:.4f}. "
        "Any real tool will do worse than this."
    )
    lines.append("")
    return "\n".join(lines)


def _section_3_tool_ranking(metrics, conn):
    """Tool Effectiveness Ranking."""
    lines = [
        "## 3. Tool Effectiveness Ranking",
        "",
        "Which tools help the most? We measure "
        "this by the **conditional revision rate**: "
        "when the model sees a tool's output, "
        "how often does it change its answer?",
        "",
    ]

    tool_stats = defaultdict(
        lambda: {
            "calls": 0,
            "samples": 0,
            "rev": 0,
        }
    )
    for eid, m in metrics.items():
        for short in [
            "prior", "confusion",
            "scene_action", "waypoint",
        ]:
            cc = m.get(f"tool_{short}_call_count", 0)
            rr = m.get(f"tool_{short}_revision_rate", 0)
            if cc > 0:
                tool_stats[short]["calls"] += cc
                tool_stats[short]["samples"] += 1
                tool_stats[short]["rev"] += rr

    if not tool_stats:
        lines.append("No tool call data available.")
        return "\n".join(lines)

    headers = [
        "Tool", "Total Calls",
        "Avg Cond Rev Rate",
    ]
    rows = []
    for tool in sorted(
        tool_stats.keys(),
        key=lambda t: _safe_div(
            tool_stats[t]["rev"],
            tool_stats[t]["samples"],
        ),
        reverse=True,
    ):
        s = tool_stats[tool]
        avg_rr = _safe_div(s["rev"], s["samples"])
        rows.append([
            tool,
            s["calls"],
            f"{avg_rr:.3f}",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")

    # Per-condition breakdown
    lines.append("### Per-Condition Tool Usage")
    lines.append("")
    headers2 = [
        "Condition", "Tool", "Calls", "Rev Rate",
    ]
    rows2 = []
    for eid, m in metrics.items():
        for short in [
            "prior", "confusion",
            "scene_action", "waypoint",
        ]:
            cc = m.get(f"tool_{short}_call_count", 0)
            rr = m.get(f"tool_{short}_revision_rate", 0)
            if cc > 0:
                rows2.append([
                    m["condition_name"],
                    short,
                    cc,
                    f"{rr:.3f}",
                ])
    if rows2:
        lines.append(_md_table(headers2, rows2))
    lines.append("")
    lines.append(
        "*Plain language*: A higher conditional "
        "revision rate means the model pays more "
        "attention to that tool's output. "
        "A tool with many calls but low revision "
        "rate is being ignored."
    )
    lines.append("")
    return "\n".join(lines)


def _section_4_2b_vs_8b(metrics, pairs):
    """2B vs 8B Verifier Comparison."""
    lines = [
        "## 4. 2B vs 8B Verifier Comparison",
        "",
    ]

    if not pairs:
        lines.append(
            "No 2B vs 8B comparison pairs found."
        )
        return "\n".join(lines)

    lines.append(
        "Direct comparison of the 2B and 8B base "
        "models as tool-calling verifiers."
    )
    lines.append("")

    headers = [
        "Condition", "Metric", "2B", "8B", "Winner",
    ]
    rows = []
    compare_items = [
        ("Accuracy", "scene_accuracy", ".4f"),
        ("Macro F1", "scene_macro_f1", ".4f"),
        ("Rev Rate", "revision_rate", ".3f"),
        ("Rev Acc", "revision_accuracy", ".3f"),
        ("Net Impr", "net_improvement", "d"),
    ]
    for cond, (_e2, m2), (_e8, m8) in pairs:
        for label, key, fmt in compare_items:
            v2 = m2.get(key, 0)
            v8 = m8.get(key, 0)
            if fmt == "d":
                s2 = str(v2)
                s8 = str(v8)
            else:
                s2 = f"{v2:{fmt}}"
                s8 = f"{v8:{fmt}}"
            if v8 > v2:
                winner = "8B"
            elif v2 > v8:
                winner = "2B"
            else:
                winner = "tie"
            show_c = cond if label == "Accuracy" else ""
            rows.append([
                show_c, label, s2, s8, winner,
            ])
        rows.append(["---", "---", "---", "---", "---"])
    lines.append(_md_table(headers, rows))
    lines.append("")
    lines.append(
        "*Plain language*: We tested whether a bigger "
        "model (8B parameters) is better at using "
        "tools than a smaller one (2B). The table "
        "above shows who wins for each metric."
    )
    lines.append("")
    return "\n".join(lines)


def _section_5_flips(metrics):
    """Flip Analysis: saves vs breaks."""
    lines = [
        "## 5. Flip Analysis",
        "",
        "A **save** is when the model corrects a wrong "
        "prediction. A **break** is when it changes a "
        "correct prediction to a wrong one. "
        "Net improvement = saves - breaks.",
        "",
    ]

    headers = [
        "Condition", "Saves", "Breaks", "Net",
        "Save Rate", "Break Rate",
    ]
    rows = []
    for eid, m in sorted(
        metrics.items(),
        key=lambda x: x[1].get("net_improvement", 0),
        reverse=True,
    ):
        total = m.get("n_revised", 0)
        sv = m.get("n_saves", 0)
        br = m.get("n_breaks", 0)
        net = m.get("net_improvement", 0)
        rows.append([
            m["condition_name"],
            sv,
            br,
            f"{net:+d}",
            _pct(sv, total) if total else "N/A",
            _pct(br, total) if total else "N/A",
        ])
    lines.append(_md_table(headers, rows))
    lines.append("")
    lines.append(
        "*Plain language*: Think of saves and breaks "
        "like a doctor's second opinion. Saves are "
        "cases where the second opinion caught an "
        "error. Breaks are cases where the second "
        "opinion introduced one."
    )
    lines.append("")
    return "\n".join(lines)


def _section_6_per_class(metrics, all_preds):
    """Per-Class Impact."""
    lines = [
        "## 6. Per-Class Impact",
        "",
        "Which scene classes benefit most from "
        "tool verification?",
        "",
    ]

    # Per-condition, per-class accuracy
    headers = ["Condition"] + SCENE_CLASSES
    rows = []
    for eid, m in metrics.items():
        preds = all_preds.get(eid, [])
        if not preds:
            continue
        gt_counts = Counter(
            r["scene_type_gt"] for r in preds
        )
        correct = Counter()
        for r in preds:
            if r["final_scene"] == r["scene_type_gt"]:
                correct[r["scene_type_gt"]] += 1
        row = [m["condition_name"]]
        for cls in SCENE_CLASSES:
            row.append(
                _pct(
                    correct.get(cls, 0),
                    gt_counts.get(cls, 0),
                )
            )
        rows.append(row)

    lines.append(_md_table(headers, rows))
    lines.append("")
    lines.append(
        "*Plain language*: Each cell shows what "
        "percentage of that class the model got "
        "right. Higher is better. Look for classes "
        "where tool conditions outperform baseline."
    )
    lines.append("")
    return "\n".join(lines)


def _section_7_incident_zone(metrics, all_preds):
    """The Incident Zone Problem deep dive."""
    lines = [
        "## 7. The Incident Zone Problem",
        "",
        "The dominant error in this model is "
        "**over-predicting incident_zone**. The model "
        "predicts incident_zone ~46.8% of the time, "
        "but only ~3.7% of samples are truly "
        "incident_zone. This section analyzes whether "
        "tools fix this bias.",
        "",
    ]

    for eid, m in metrics.items():
        preds = all_preds.get(eid, [])
        if not preds:
            continue

        # Count incident_zone predictions
        orig_iz = sum(
            1
            for r in preds
            if r.get("original_scene") == "incident_zone"
        )
        final_iz = sum(
            1
            for r in preds
            if r.get("final_scene") == "incident_zone"
        )
        gt_iz = sum(
            1
            for r in preds
            if r["scene_type_gt"] == "incident_zone"
        )
        total = len(preds)

        lines.append(
            f"### {m['condition_name']}"
        )
        lines.append("")
        lines.append(
            f"- Original incident_zone predictions: "
            f"{orig_iz} ({_pct(orig_iz, total)})"
        )
        lines.append(
            f"- Final incident_zone predictions: "
            f"{final_iz} ({_pct(final_iz, total)})"
        )
        lines.append(
            f"- Ground truth incident_zone: "
            f"{gt_iz} ({_pct(gt_iz, total)})"
        )
        reduction = orig_iz - final_iz
        lines.append(
            f"- Reduction: {reduction} "
            f"({_pct(reduction, orig_iz)} of original)"
        )
        lines.append("")

    lines.append(
        "*Plain language*: The model sees traffic "
        "cones, barriers, and similar objects and "
        "thinks \"incident zone!\" even when there "
        "is no actual incident. The tools try to "
        "tell the model that incident_zone is rare "
        "and most of the time it is just a normal "
        "scene with some construction equipment."
    )
    lines.append("")
    return "\n".join(lines)


def _section_8_saves(conn, metrics, all_preds):
    """Case studies of successful corrections."""
    lines = [
        "## 8. Samples Where Tools Saved",
        "",
        "Top 5 examples where tool verification "
        "corrected a wrong prediction.",
        "",
    ]

    # Find saves across all conditions, pick top 5
    saves = []
    for eid, m in metrics.items():
        preds = all_preds.get(eid, [])
        for r in preds:
            orig_ok = (
                r.get("original_scene")
                == r["scene_type_gt"]
            )
            final_ok = (
                r.get("final_scene")
                == r["scene_type_gt"]
            )
            if not orig_ok and final_ok:
                saves.append((eid, m["condition_name"], r))

    if not saves:
        lines.append("No saves found.")
        return "\n".join(lines)

    # Take first 5 unique sample_ids
    seen = set()
    examples = []
    for eid, cond, r in saves:
        sid = r["sample_id"]
        if sid not in seen:
            seen.add(sid)
            examples.append((eid, cond, r))
        if len(examples) >= 5:
            break

    for i, (eid, cond, r) in enumerate(examples, 1):
        sid = r["sample_id"]
        lines.append(f"### Save #{i}: Sample {sid}")
        lines.append("")
        lines.append(
            f"- **Condition**: {cond}"
        )
        lines.append(
            f"- **Ground truth**: {r['scene_type_gt']}"
        )
        lines.append(
            f"- **Original prediction**: "
            f"{r.get('original_scene', 'N/A')}"
        )
        lines.append(
            f"- **Final prediction**: "
            f"{r.get('final_scene', 'N/A')}"
        )
        if r.get("revision_reason"):
            lines.append(
                f"- **Reason**: {r['revision_reason']}"
            )

        # Tool call trace
        tcs = get_tool_calls(conn, eid, sid)
        if tcs:
            lines.append("")
            lines.append("**Tool call trace**:")
            lines.append("")
            for tc in tcs:
                name = tc["tool_name"]
                args = tc.get(
                    "tool_arguments_json", "{}"
                )
                result = tc.get(
                    "tool_result_json", "{}"
                )
                lines.append(
                    f"  1. `{name}({args})` "
                    f"-> `{result}`"
                )
                if tc.get("model_revised_after"):
                    old = tc.get("old_value", "?")
                    new = tc.get("new_value", "?")
                    lines.append(
                        f"     Model revised: "
                        f"{old} -> {new}"
                    )
        lines.append("")

    return "\n".join(lines)


def _section_9_breaks(conn, metrics, all_preds):
    """Case studies of harmful corrections."""
    lines = [
        "## 9. Samples Where Tools Broke",
        "",
        "Top 5 examples where tool verification "
        "changed a correct prediction to a wrong one.",
        "",
    ]

    breaks = []
    for eid, m in metrics.items():
        preds = all_preds.get(eid, [])
        for r in preds:
            orig_ok = (
                r.get("original_scene")
                == r["scene_type_gt"]
            )
            final_ok = (
                r.get("final_scene")
                == r["scene_type_gt"]
            )
            if orig_ok and not final_ok:
                breaks.append(
                    (eid, m["condition_name"], r)
                )

    if not breaks:
        lines.append("No breaks found.")
        return "\n".join(lines)

    seen = set()
    examples = []
    for eid, cond, r in breaks:
        sid = r["sample_id"]
        if sid not in seen:
            seen.add(sid)
            examples.append((eid, cond, r))
        if len(examples) >= 5:
            break

    for i, (eid, cond, r) in enumerate(examples, 1):
        sid = r["sample_id"]
        lines.append(f"### Break #{i}: Sample {sid}")
        lines.append("")
        lines.append(f"- **Condition**: {cond}")
        lines.append(
            f"- **Ground truth**: {r['scene_type_gt']}"
        )
        lines.append(
            f"- **Original prediction**: "
            f"{r.get('original_scene', 'N/A')} "
            "(CORRECT)"
        )
        lines.append(
            f"- **Final prediction**: "
            f"{r.get('final_scene', 'N/A')} "
            "(WRONG)"
        )
        if r.get("revision_reason"):
            lines.append(
                f"- **Reason**: {r['revision_reason']}"
            )

        tcs = get_tool_calls(conn, eid, sid)
        if tcs:
            lines.append("")
            lines.append("**Tool call trace**:")
            lines.append("")
            for tc in tcs:
                name = tc["tool_name"]
                args = tc.get(
                    "tool_arguments_json", "{}"
                )
                result = tc.get(
                    "tool_result_json", "{}"
                )
                lines.append(
                    f"  1. `{name}({args})` "
                    f"-> `{result}`"
                )
                if tc.get("model_revised_after"):
                    old = tc.get("old_value", "?")
                    new = tc.get("new_value", "?")
                    lines.append(
                        f"     Model revised: "
                        f"{old} -> {new}"
                    )
        lines.append("")

    lines.append(
        "*Plain language*: These are cases where "
        "the tool gave the model information that "
        "caused it to second-guess a correct answer. "
        "This is the cost of using tools -- sometimes "
        "the extra information is misleading."
    )
    lines.append("")
    return "\n".join(lines)


def _section_10_recommendation(metrics, pairs):
    """Go/no-go recommendation."""
    lines = [
        "## 10. Recommendation",
        "",
    ]

    if not metrics:
        lines.append("Insufficient data for recommendation.")
        return "\n".join(lines)

    # Determine oracle ceiling
    oracles = [
        m
        for m in metrics.values()
        if "oracle" in m["condition_name"]
    ]
    best_oracle_f1 = 0.0
    if oracles:
        best_oracle_f1 = max(
            m.get("scene_macro_f1", 0) for m in oracles
        )

    # Best real (non-oracle, non-baseline) condition
    real = [
        m
        for m in metrics.values()
        if "oracle" not in m["condition_name"]
        and m["condition_name"] != "baseline"
    ]
    best_real = None
    if real:
        best_real = max(
            real,
            key=lambda x: x.get("scene_macro_f1", 0),
        )

    # Decision logic
    if best_oracle_f1 < 0.70:
        lines.append(
            "### Decision: NO-GO (Oracle ceiling too low)"
        )
        lines.append("")
        lines.append(
            f"The oracle ceiling is {best_oracle_f1:.4f} "
            "(below 0.70). Even with perfect tool "
            "information, the model cannot self-correct "
            "effectively enough. **Retraining or "
            "fine-tuning is needed** rather than "
            "post-hoc tool-calling."
        )
    elif best_real and best_real.get("net_improvement", 0) <= 0:
        lines.append(
            "### Decision: CONDITIONAL NO-GO "
            "(Tools not helping)"
        )
        lines.append("")
        lines.append(
            f"Oracle ceiling is {best_oracle_f1:.4f} "
            "(promising), but the best real tool "
            f"condition ({best_real['condition_name']}) "
            f"has net improvement = "
            f"{best_real.get('net_improvement', 0):+d}. "
            "Tools are not providing useful enough "
            "information, or the model is ignoring them."
        )
        lines.append("")
        rev_rate = best_real.get("revision_rate", 0)
        if rev_rate < 0.1:
            lines.append(
                "The model rarely revises "
                f"(revision rate = {rev_rate:.3f}). "
                "Consider: "
            )
            lines.append(
                "- Rule-based post-processing instead "
                "of model-based verification"
            )
            lines.append(
                "- More informative tool outputs"
            )
            lines.append(
                "- Better verification prompts"
            )
        else:
            lines.append(
                "The model revises but introduces "
                "too many errors. Consider:"
            )
            lines.append("- More conservative tools")
            lines.append(
                "- Higher confidence thresholds "
                "for revision"
            )
    else:
        net = (
            best_real.get("net_improvement", 0)
            if best_real
            else 0
        )
        lines.append(
            "### Decision: GO (with caveats)"
        )
        lines.append("")
        if best_real:
            cond = best_real["condition_name"]
            f1 = best_real.get("scene_macro_f1", 0)
            lines.append(
                f"Best real condition: **{cond}** "
                f"(F1={f1:.4f}, net={net:+d})."
            )
        lines.append("")
        lines.append("**Recommended next steps**:")
        lines.append(
            "1. Deploy the best condition in "
            "a shadow mode to validate on "
            "production data"
        )
        lines.append(
            "2. Monitor save/break ratio over time"
        )
        lines.append(
            "3. Consider adding confidence thresholds "
            "to reduce breaks"
        )

    lines.append("")

    # 2B vs 8B recommendation
    if pairs:
        lines.append("### Verifier Size Recommendation")
        lines.append("")
        total_delta = 0.0
        for _cond, (_e2, m2), (_e8, m8) in pairs:
            d = (
                m8.get("scene_macro_f1", 0)
                - m2.get("scene_macro_f1", 0)
            )
            total_delta += d
        avg_delta = total_delta / len(pairs)
        if avg_delta > 0.01:
            lines.append(
                f"The 8B verifier outperforms 2B by "
                f"an average of {avg_delta:+.4f} "
                "Macro F1. **Use the 8B model** "
                "if latency permits."
            )
        elif avg_delta < -0.01:
            lines.append(
                f"The 2B verifier outperforms 8B by "
                f"an average of {-avg_delta:+.4f} "
                "Macro F1. **Use the 2B model** "
                "for faster inference."
            )
        else:
            lines.append(
                "Both verifier sizes perform similarly "
                f"(avg delta = {avg_delta:+.4f}). "
                "**Use the 2B model** for efficiency."
            )
    lines.append("")

    lines.append(
        "*Plain language*: Based on all the evidence, "
        "this is our recommendation for whether to "
        "use tool-calling in the production pipeline. "
        "Think of it like deciding whether to add "
        "a spell-checker to your writing -- "
        "it only helps if it catches more real "
        "mistakes than it creates."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------


def generate_report(db_path, output_path):
    """Generate the full insights report."""
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    metrics = get_metrics(conn)
    all_preds = get_all_predictions(conn)
    pairs = _find_pairs(metrics)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = []

    # Title
    sections.append(
        "# Tool-Calling Verification Experiment "
        "-- Insights Report"
    )
    sections.append("")
    sections.append(f"*Generated: {now}*")
    sections.append("")
    sections.append(
        f"*Database: {db_path}*"
    )
    sections.append(
        f"*Conditions analyzed: {len(metrics)}*"
    )
    sections.append("")

    # Sections
    sections.append(
        _section_1_executive(metrics, pairs, all_preds)
    )
    sections.append(_section_2_oracle(metrics))
    sections.append(
        _section_3_tool_ranking(metrics, conn)
    )
    sections.append(_section_4_2b_vs_8b(metrics, pairs))
    sections.append(_section_5_flips(metrics))
    sections.append(
        _section_6_per_class(metrics, all_preds)
    )
    sections.append(
        _section_7_incident_zone(metrics, all_preds)
    )
    sections.append(
        _section_8_saves(conn, metrics, all_preds)
    )
    sections.append(
        _section_9_breaks(conn, metrics, all_preds)
    )
    sections.append(
        _section_10_recommendation(metrics, pairs)
    )

    report = "\n".join(sections)

    with open(output_path, "w") as f:
        f.write(report)

    conn.close()

    print(f"Insights report written to: {output_path}")
    print("Sections: 10")
    print(f"Total length: {len(report)} characters")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate tool-calling insights "
            "markdown report"
        )
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DB_DEFAULT,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DEFAULT,
        help="Output markdown file path",
    )
    args = parser.parse_args()
    generate_report(args.db, args.output)


if __name__ == "__main__":
    main()
