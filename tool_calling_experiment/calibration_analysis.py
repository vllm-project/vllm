#!/usr/bin/env python3
"""
Calibration Baseline Analysis for SceneIQ Predictions.

Simulates multiple post-processing calibration strategies
on existing model predictions to measure achievable F1 gain
without retraining.

Data: tool_calling.db, table 'predictions' (8,613 rows)
Model: Fine-tuned Qwen3-VL-2B (baseline accuracy 46.9%)
"""

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

DB_PATH = Path(__file__).parent / "tool_calling.db"
OUTPUT_PATH = Path(__file__).parent / "calibration_results.json"

CLASSES = [
    "nominal",
    "flagger",
    "flooded",
    "incident_zone",
    "mounted_police",
]


def load_data():
    """Load predictions from the database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT sample_id, original_scene, scene_type_gt, "
        "original_scene_correct, fine_class, odd_label "
        "FROM predictions"
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def compute_metrics(y_true, y_pred, classes=None):
    """Compute accuracy, per-class P/R/F1, macro F1, confusion."""
    if classes is None:
        classes = CLASSES

    n = len(y_true)
    correct = sum(
        1 for t, p in zip(y_true, y_pred) if t == p
    )
    accuracy = correct / n

    # Confusion matrix: rows = GT, cols = predicted
    cm: dict = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class = {}
    f1_scores = []

    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(
            cm[gt][cls] for gt in classes if gt != cls
        )
        fn = sum(
            cm[cls][pred]
            for pred in classes
            if pred != cls
        )

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * prec * rec / (prec + rec)
            if (prec + rec) > 0
            else 0.0
        )

        support = sum(cm[cls][p] for p in classes)
        pred_count = sum(cm[gt][cls] for gt in classes)
        per_class[cls] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "predicted_count": pred_count,
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)

    # Build confusion matrix as nested dict for JSON
    cm_dict = {}
    for gt_cls in classes:
        cm_dict[gt_cls] = {
            pred_cls: cm[gt_cls][pred_cls]
            for pred_cls in classes
        }

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm_dict,
    }


def net_improvement(y_true, y_pred_baseline, y_pred_new):
    """Count samples saved vs broken by the strategy."""
    saved = 0  # was wrong, now right
    broken = 0  # was right, now wrong
    for t, b, nv in zip(
        y_true, y_pred_baseline, y_pred_new
    ):
        was_correct = t == b
        now_correct = t == nv
        if not was_correct and now_correct:
            saved += 1
        elif was_correct and not now_correct:
            broken += 1
    return saved, broken


def _fmt_delta(val):
    """Format a delta value with sign."""
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}"


def print_metrics(name, metrics, baseline_metrics=None):
    """Pretty-print metrics for a strategy."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)
    acc_pct = metrics["accuracy"] * 100
    print(f"  Accuracy:  {acc_pct:.1f}%", end="")
    if baseline_metrics:
        d = acc_pct - baseline_metrics["accuracy"] * 100
        print(f"  ({_fmt_delta(d)}pp)")
    else:
        print()
    mf1_pct = metrics["macro_f1"] * 100
    print(f"  Macro F1:  {mf1_pct:.1f}%", end="")
    if baseline_metrics:
        d = mf1_pct - baseline_metrics["macro_f1"] * 100
        print(f"  ({_fmt_delta(d)}pp)")
    else:
        print()

    hdr = (
        "  {:<18} {:>7} {:>7} {:>7} {:>8} {:>9}"
    ).format(
        "Class", "Prec", "Recall", "F1", "Support", "Predicted"
    )
    print(f"\n{hdr}")
    print(f"  {'-' * 58}")
    for cls in CLASSES:
        c = metrics["per_class"][cls]
        p = c["precision"] * 100
        r = c["recall"] * 100
        f = c["f1"] * 100
        s = c["support"]
        pc = c["predicted_count"]
        row = f"  {cls:<18} {p:6.1f}% {r:6.1f}%"
        row += f" {f:6.1f}% {s:>8} {pc:>9}"
        print(row)

    # Confusion matrix
    print("\n  Confusion Matrix (rows=GT, cols=Pred):")
    gt_pred_label = "GT \\ Pred"
    header = f"  {gt_pred_label:<18}"
    for c in CLASSES:
        header += f"{c[:8]:>10}"
    print(header)
    print(f"  {'-' * 68}")
    cm = metrics["confusion_matrix"]
    for gt_cls in CLASSES:
        row = f"  {gt_cls:<18}"
        for pred_cls in CLASSES:
            val = cm[gt_cls][pred_cls]
            row += f"{val:>10}"
        print(row)


def print_net_improvement(saved, broken):
    """Print net improvement summary."""
    net = saved - broken
    sign = "+" if net >= 0 else ""
    print(
        f"\n  Net improvement: {saved} saved, "
        f"{broken} broken, net {sign}{net}"
    )


def _f1_delta(m, baseline):
    """Compute macro F1 delta in pp."""
    return (m["macro_f1"] - baseline["macro_f1"]) * 100


def _acc_delta(m, baseline):
    """Compute accuracy delta in pp."""
    return (m["accuracy"] - baseline["accuracy"]) * 100


def _f1_gain_rounded(m, baseline):
    """Compute rounded macro F1 delta."""
    return round(_f1_delta(m, baseline), 1)


def main():  # noqa: C901
    """Run all calibration strategies."""
    print("Loading data...")
    data = load_data()
    n = len(data)
    print(f"Loaded {n} predictions.\n")

    y_true = [d["scene_type_gt"] for d in data]
    y_pred_bl = [d["original_scene"] for d in data]
    fine_cls = [d["fine_class"] for d in data]

    # GT class priors
    gt_counts = Counter(y_true)
    gt_priors = {
        cls: gt_counts[cls] / n for cls in CLASSES
    }

    print("Ground truth class priors:")
    for cls in CLASSES:
        pct = gt_priors[cls] * 100
        print(
            f"  {cls:<18} {gt_counts[cls]:>5}"
            f" ({pct:.1f}%)"
        )

    pred_counts = Counter(y_pred_bl)
    print("\nPredicted class distribution:")
    for cls in CLASSES:
        cnt = pred_counts.get(cls, 0)
        pct = cnt / n * 100
        print(f"  {cls:<18} {cnt:>5} ({pct:.1f}%)")

    all_results: dict = {}

    # ===== BASELINE =====
    bl_m = compute_metrics(y_true, y_pred_bl)
    print_metrics("BASELINE (no correction)", bl_m)
    all_results["baseline"] = bl_m

    # ===== STRATEGY 1: Naive flip all IZ -> nominal =====
    y_s1 = [
        "nominal" if p == "incident_zone" else p
        for p in y_pred_bl
    ]
    s1_m = compute_metrics(y_true, y_s1)
    s1_sv, s1_br = net_improvement(y_true, y_pred_bl, y_s1)
    print_metrics(
        "STRATEGY 1: Naive flip (IZ -> nominal)",
        s1_m,
        bl_m,
    )
    print_net_improvement(s1_sv, s1_br)
    all_results["strategy_1_naive_flip"] = {
        **s1_m,
        "description": "All IZ predictions flipped to nominal",
        "saved": s1_sv,
        "broken": s1_br,
    }

    # ===== STRATEGY 2: Prior-weighted threshold =====
    thresholds = [0.05, 0.10, 0.15, 0.20]
    all_results["strategy_2_prior_threshold"] = {}

    for thresh in thresholds:
        low_prior = [
            cls
            for cls in CLASSES
            if gt_priors[cls] < thresh
        ]
        y_s2 = [
            "nominal" if p in low_prior else p
            for p in y_pred_bl
        ]
        s2_m = compute_metrics(y_true, y_s2)
        s2_sv, s2_br = net_improvement(
            y_true, y_pred_bl, y_s2
        )

        thr_pct = thresh * 100
        label = (
            f"STRATEGY 2: Prior < {thr_pct:.0f}%"
            " -> nominal"
        )
        parts = [
            f"{c} ({gt_priors[c] * 100:.1f}%)"
            for c in low_prior
        ]
        flipped_str = ", ".join(parts) or "none"
        full_label = (
            f"{label}\n  Flipped classes: {flipped_str}"
        )
        print_metrics(full_label, s2_m, bl_m)
        print_net_improvement(s2_sv, s2_br)

        key = f"threshold_{int(thr_pct)}pct"
        all_results["strategy_2_prior_threshold"][key] = {
            **s2_m,
            "threshold": thresh,
            "flipped_classes": low_prior,
            "saved": s2_sv,
            "broken": s2_br,
        }

    # ===== STRATEGY 3: Confusion-aware correction =====
    # Flip IZ -> nominal EXCEPT fine_class == incident_zone
    y_s3 = []
    for i, p in enumerate(y_pred_bl):
        if (
            p == "incident_zone"
            and fine_cls[i] != "incident_zone"
        ):
            y_s3.append("nominal")
        else:
            y_s3.append(p)

    s3_m = compute_metrics(y_true, y_s3)
    s3_sv, s3_br = net_improvement(y_true, y_pred_bl, y_s3)
    print_metrics(
        "STRATEGY 3: Confusion-aware"
        " (IZ->nominal, except FC=IZ)",
        s3_m,
        bl_m,
    )
    print_net_improvement(s3_sv, s3_br)

    iz_with_iz_fc = sum(
        1
        for i in range(n)
        if (
            y_pred_bl[i] == "incident_zone"
            and fine_cls[i] == "incident_zone"
        )
    )
    iz_total = sum(
        1 for p in y_pred_bl if p == "incident_zone"
    )
    print(
        f"\n  IZ preds with fine_class=IZ: "
        f"{iz_with_iz_fc}/{iz_total}"
    )

    all_results["strategy_3_confusion_aware"] = {
        **s3_m,
        "description": (
            "IZ->nominal, except fine_class=IZ (oracle)"
        ),
        "saved": s3_sv,
        "broken": s3_br,
        "iz_pred_with_iz_fineclass": iz_with_iz_fc,
        "iz_pred_total": iz_total,
    }

    # ===== STRATEGY 4: Oracle ceiling (IZ only) =====
    y_s4 = []
    for i, p in enumerate(y_pred_bl):
        if p == "incident_zone":
            y_s4.append(y_true[i])  # oracle: use GT
        else:
            y_s4.append(p)

    s4_m = compute_metrics(y_true, y_s4)
    s4_sv, s4_br = net_improvement(y_true, y_pred_bl, y_s4)
    print_metrics(
        "STRATEGY 4: Oracle ceiling"
        " (perfect IZ corrections)",
        s4_m,
        bl_m,
    )
    print_net_improvement(s4_sv, s4_br)

    total_err = sum(
        1
        for t, p in zip(y_true, y_pred_bl)
        if t != p
    )
    iz_err = sum(
        1
        for t, p in zip(y_true, y_pred_bl)
        if p == "incident_zone" and t != p
    )
    iz_err_pct = round(100 * iz_err / total_err, 1)
    print(f"\n  Total errors: {total_err}")
    print(
        f"  Errors from IZ over-prediction: {iz_err}"
        f" ({iz_err_pct}% of all errors)"
    )

    all_results["strategy_4_oracle_ceiling"] = {
        **s4_m,
        "description": (
            "Oracle: perfect IZ corrections using GT"
        ),
        "saved": s4_sv,
        "broken": s4_br,
        "total_errors_baseline": total_err,
        "errors_from_iz_overprediction": iz_err,
        "pct_errors_from_iz": iz_err_pct,
    }

    # ===== STRATEGY 5: Selective by fine_class =====
    # IZ + nominal_triggers fine_class -> nominal
    y_s5 = []
    for i, p in enumerate(y_pred_bl):
        if (
            p == "incident_zone"
            and fine_cls[i] == "nominal_triggers"
        ):
            y_s5.append("nominal")
        else:
            y_s5.append(p)

    s5_m = compute_metrics(y_true, y_s5)
    s5_sv, s5_br = net_improvement(y_true, y_pred_bl, y_s5)
    print_metrics(
        "STRATEGY 5: Selective"
        " (IZ+nominal_triggers -> nominal)",
        s5_m,
        bl_m,
    )
    print_net_improvement(s5_sv, s5_br)

    # IZ predictions breakdown by fine_class
    iz_by_fc: Counter = Counter()
    iz_correct_by_fc: Counter = Counter()
    for i in range(n):
        if y_pred_bl[i] == "incident_zone":
            fc = fine_cls[i]
            iz_by_fc[fc] += 1
            if y_true[i] == "incident_zone":
                iz_correct_by_fc[fc] += 1

    print("\n  IZ predictions by fine_class:")
    hdr = "  {:<20} {:>7} {:>11} {:>9}".format(
        "fine_class", "Count", "Actually IZ", "Accuracy"
    )
    print(hdr)
    print(f"  {'-' * 52}")
    for fc, cnt in iz_by_fc.most_common():
        corr = iz_correct_by_fc[fc]
        acc = 100 * corr / cnt
        print(
            f"  {fc:<20} {cnt:>7}"
            f" {corr:>11} {acc:>8.1f}%"
        )

    all_results["strategy_5_selective_fineclass"] = {
        **s5_m,
        "description": (
            "IZ+nominal_triggers -> nominal (oracle FC)"
        ),
        "saved": s5_sv,
        "broken": s5_br,
        "iz_by_fineclass": dict(iz_by_fc),
    }

    # ===== SUMMARY COMPARISON TABLE =====
    _print_summary(
        bl_m,
        s1_m,
        s3_m,
        s4_m,
        s5_m,
        all_results,
        y_true,
        y_pred_bl,
        y_s1,
        y_s3,
        y_s4,
        y_s5,
    )

    # ===== KEY INSIGHTS =====
    _print_insights(bl_m, s1_m, s3_m, s4_m, s5_m, all_results)

    # ===== SAVE RESULTS =====
    _save_results(
        all_results, bl_m, s1_m, s3_m, s4_m, s5_m
    )


def _print_summary(
    bl_m,
    s1_m,
    s3_m,
    s4_m,
    s5_m,
    all_results,
    y_true,
    y_pred_bl,
    y_s1,
    y_s3,
    y_s4,
    y_s5,
):
    """Print summary comparison table."""
    sep = "#" * 72
    print(f"\n\n{sep}")
    print("  SUMMARY COMPARISON")
    print(sep)

    s2 = all_results["strategy_2_prior_threshold"]
    rows = [
        ("Baseline", bl_m, 0, 0),
        (
            "S1: Naive flip (IZ->nom)",
            s1_m,
            *net_improvement(y_true, y_pred_bl, y_s1),
        ),
        (
            "S2: Prior <5% -> nom",
            s2["threshold_5pct"],
            s2["threshold_5pct"]["saved"],
            s2["threshold_5pct"]["broken"],
        ),
        (
            "S2: Prior <10% -> nom",
            s2["threshold_10pct"],
            s2["threshold_10pct"]["saved"],
            s2["threshold_10pct"]["broken"],
        ),
        (
            "S2: Prior <15% -> nom",
            s2["threshold_15pct"],
            s2["threshold_15pct"]["saved"],
            s2["threshold_15pct"]["broken"],
        ),
        (
            "S2: Prior <20% -> nom",
            s2["threshold_20pct"],
            s2["threshold_20pct"]["saved"],
            s2["threshold_20pct"]["broken"],
        ),
        (
            "S3: Confusion-aware (oracle)",
            s3_m,
            *net_improvement(y_true, y_pred_bl, y_s3),
        ),
        (
            "S4: Oracle ceiling (IZ)",
            s4_m,
            *net_improvement(y_true, y_pred_bl, y_s4),
        ),
        (
            "S5: Selective (IZ+nt->nom)",
            s5_m,
            *net_improvement(y_true, y_pred_bl, y_s5),
        ),
    ]

    hdr_fmt = (
        "  {:<30} {:>6} {:>6} {:>7} {:>6}"
        " {:>5} {:>5} {:>5}"
    )
    print(
        "\n"
        + hdr_fmt.format(
            "Strategy",
            "Acc",
            "dAcc",
            "MacF1",
            "dF1",
            "Save",
            "Brk",
            "Net",
        )
    )
    print(f"  {'-' * 76}")

    for name, m, sv, br in rows:
        acc = m["accuracy"] * 100
        mf1 = m["macro_f1"] * 100
        d_acc = _acc_delta(m, bl_m)
        d_f1 = _f1_delta(m, bl_m)
        net_val = sv - br
        is_bl = name == "Baseline"
        da = "---" if is_bl else _fmt_delta(d_acc)
        df = "---" if is_bl else _fmt_delta(d_f1)
        ns = "---" if is_bl else _fmt_delta(net_val)
        print(
            f"  {name:<30} {acc:5.1f}% {da:>6}"
            f" {mf1:5.1f}%  {df:>5}"
            f" {sv:>5} {br:>5} {ns:>5}"
        )


def _print_insights(bl_m, s1_m, s3_m, s4_m, s5_m, res):
    """Print key insights section."""
    sep = "#" * 72
    print(f"\n\n{sep}")
    print("  KEY INSIGHTS")
    print(sep)

    candidates = [("S1", s1_m), ("S5", s5_m)]
    best_name, best_m = max(
        candidates, key=lambda x: x[1]["macro_f1"]
    )
    gain = _f1_delta(best_m, bl_m)

    bl_acc = bl_m["accuracy"] * 100
    bl_f1 = bl_m["macro_f1"] * 100
    print(
        f"\n  1. BASELINE: {bl_acc:.1f}% accuracy,"
        f" {bl_f1:.1f}% macro F1"
    )

    b_acc = best_m["accuracy"] * 100
    b_f1 = best_m["macro_f1"] * 100
    print(f"\n  2. BEST PRACTICAL STRATEGY: {best_name}")
    print(
        f"     Accuracy: {b_acc:.1f}%,"
        f" Macro F1: {b_f1:.1f}%"
    )
    print(f"     F1 gain: {_fmt_delta(gain)}pp")

    s4_acc = s4_m["accuracy"] * 100
    s4_f1 = s4_m["macro_f1"] * 100
    print("\n  3. ORACLE CEILING (correcting only IZ):")
    print(
        f"     Accuracy: {s4_acc:.1f}%,"
        f" Macro F1: {s4_f1:.1f}%"
    )
    print("     MAX achievable by fixing IZ only.")

    oracle = res["strategy_4_oracle_ceiling"]
    iz_e = oracle["errors_from_iz_overprediction"]
    tot_e = oracle["total_errors_baseline"]
    iz_pct = oracle["pct_errors_from_iz"]
    acc_recov = _acc_delta(s4_m, bl_m)
    print("\n  4. ERROR BUDGET:")
    print(
        f"     - {iz_e} of {tot_e} errors"
        f" ({iz_pct}%) from IZ over-prediction"
    )
    print(
        f"     - Fixing IZ recovers up to"
        f" {_fmt_delta(acc_recov)}pp accuracy"
    )

    s1_g = _f1_delta(s1_m, bl_m)
    s3_g = _f1_delta(s3_m, bl_m)
    s4_g = _f1_delta(s4_m, bl_m)
    print(
        "\n  5. F1 GAIN FROM PURE POST-PROCESSING:"
    )
    print(
        f"     - Simple rule-based:"
        f" {_fmt_delta(s1_g)}pp macro F1"
    )
    print(
        f"     - Oracle fine_class:"
        f" {_fmt_delta(s3_g)}pp macro F1"
    )
    print(
        f"     - Oracle ceiling (IZ):"
        f" {_fmt_delta(s4_g)}pp macro F1"
    )
    print()


def _save_results(
    all_results, bl_m, s1_m, s3_m, s4_m, s5_m
):
    """Save all results to JSON."""
    candidates = [("S1", s1_m), ("S5", s5_m)]
    best_name, best_m = max(
        candidates, key=lambda x: x[1]["macro_f1"]
    )

    oracle = all_results["strategy_4_oracle_ceiling"]
    all_results["summary"] = {
        "baseline_accuracy": bl_m["accuracy"],
        "baseline_macro_f1": bl_m["macro_f1"],
        "best_practical_strategy": best_name,
        "best_practical_accuracy": best_m["accuracy"],
        "best_practical_macro_f1": best_m["macro_f1"],
        "oracle_ceiling_accuracy": s4_m["accuracy"],
        "oracle_ceiling_macro_f1": s4_m["macro_f1"],
        "pct_errors_from_iz": oracle["pct_errors_from_iz"],
        "max_f1_gain_simple_pp": _f1_gain_rounded(
            s1_m, bl_m
        ),
        "max_f1_gain_oracle_fineclass": _f1_gain_rounded(
            s3_m, bl_m
        ),
        "max_f1_gain_oracle_ceiling": _f1_gain_rounded(
            s4_m, bl_m
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
