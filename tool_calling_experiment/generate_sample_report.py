#!/usr/bin/env python3
"""
generate_sample_report.py - Per-sample JSONL report for tool-calling.

Reads from tool_calling_experiment/tool_calling.db and generates a JSONL
file at tool_calling_experiment/sample_report_data.jsonl with one JSON
object per sample, aggregating results across all experimental conditions.

Usage:
    python tool_calling_experiment/generate_sample_report.py [--db PATH]
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict

DB_DEFAULT = os.path.join(
    os.path.dirname(__file__), "tool_calling.db"
)
OUTPUT_DEFAULT = os.path.join(
    os.path.dirname(__file__), "sample_report_data.jsonl"
)


def get_all_experiments(conn):
    """Fetch all experiments as dicts keyed by experiment_id."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM experiments ORDER BY created_at"
    ).fetchall()
    return {r["experiment_id"]: dict(r) for r in rows}


def get_all_predictions(conn):
    """Fetch all predictions grouped by sample_id.

    Returns dict: sample_id -> list of prediction dicts.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY sample_id"
    ).fetchall()
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r["sample_id"]].append(dict(r))
    return by_sample


def get_tool_calls_for_sample(conn, experiment_id, sample_id):
    """Fetch tool calls for a specific experiment+sample.

    Returns list of dicts sorted by tool_call_order.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tool_calls "
        "WHERE experiment_id = ? AND sample_id = ? "
        "ORDER BY tool_call_order",
        (experiment_id, sample_id),
    ).fetchall()
    return [dict(r) for r in rows]


def _parse_json_safe(text):
    """Parse JSON string, returning raw string on failure."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


def _build_condition_entry(pred, tool_calls_list):
    """Build a condition entry for one prediction row."""
    entry = {}

    if pred.get("original_scene"):
        entry["original_scene"] = pred["original_scene"]
    if pred.get("final_scene"):
        entry["final_scene"] = pred["final_scene"]
        entry["scene_correct"] = (
            pred["final_scene"] == pred["scene_type_gt"]
        )
    if pred.get("original_long_action"):
        entry["original_long_action"] = pred[
            "original_long_action"
        ]
    if pred.get("original_lat_action"):
        entry["original_lat_action"] = pred[
            "original_lat_action"
        ]
    if pred.get("final_long_action"):
        entry["final_long_action"] = pred[
            "final_long_action"
        ]
    if pred.get("final_lat_action"):
        entry["final_lat_action"] = pred["final_lat_action"]

    entry["was_revised"] = bool(pred.get("was_revised"))
    entry["scene_was_revised"] = bool(
        pred.get("scene_was_revised")
    )

    if pred.get("revision_reason"):
        entry["revision_reason"] = pred["revision_reason"]

    # Include original correctness for flip analysis
    entry["original_scene_correct"] = (
        pred.get("original_scene") == pred.get("scene_type_gt")
    )

    # Tool call traces
    if tool_calls_list:
        tc_entries = []
        for tc in tool_calls_list:
            tc_entry = {
                "tool": tc["tool_name"],
                "args": _parse_json_safe(
                    tc["tool_arguments_json"]
                ),
                "result": _parse_json_safe(
                    tc["tool_result_json"]
                ),
            }
            if tc.get("model_revised_after"):
                tc_entry["revised_after"] = True
                if tc.get("revised_field"):
                    tc_entry["revised_field"] = tc[
                        "revised_field"
                    ]
                if tc.get("old_value"):
                    tc_entry["old_value"] = tc["old_value"]
                if tc.get("new_value"):
                    tc_entry["new_value"] = tc["new_value"]
            tc_entries.append(tc_entry)
        entry["tool_calls"] = tc_entries

    # Timing
    if pred.get("predict_time_ms") is not None:
        entry["predict_time_ms"] = pred["predict_time_ms"]
    if pred.get("verify_time_ms") is not None:
        entry["verify_time_ms"] = pred["verify_time_ms"]

    return entry


def _compute_diagnostic_tags(
    sample_conditions, scene_type_gt, fine_class
):
    """Compute diagnostic tags for a sample.

    Args:
        sample_conditions: dict of condition_name -> entry
        scene_type_gt: ground truth scene type
        fine_class: fine class label

    Returns list of tag strings.
    """
    tags = []

    # Collect correctness info across conditions
    any_save = False
    any_break = False
    any_revised_correct = False
    any_revised_incorrect = False
    any_ignored_warning = False
    final_scenes = set()

    # Track 2B vs 8B by base condition
    cond_2b = {}
    cond_8b = {}

    for cond_name, entry in sample_conditions.items():
        scene_correct = entry.get("scene_correct", None)
        was_revised = entry.get("was_revised", False)
        orig_correct = entry.get(
            "original_scene_correct", None
        )
        final_scene = entry.get("final_scene")

        if final_scene:
            final_scenes.add(final_scene)

        # Save / break detection
        if orig_correct is False and scene_correct is True:
            any_save = True
        if orig_correct is True and scene_correct is False:
            any_break = True

        # Revised correctly / incorrectly
        if was_revised and scene_correct is True:
            any_revised_correct = True
        if was_revised and scene_correct is False:
            any_revised_incorrect = True

        # Ignored tool warning detection
        tool_calls = entry.get("tool_calls", [])
        has_high_risk = False
        for tc in tool_calls:
            result = tc.get("result")
            if isinstance(result, dict):
                risk = result.get("risk_level", "")
                err_rate = result.get("error_rate", 0)
                if risk == "high" or (
                    isinstance(err_rate, (int, float))
                    and err_rate > 0.3
                ):
                    has_high_risk = True
        if has_high_risk and not was_revised:
            any_ignored_warning = True

        # Classify 2B vs 8B
        cn_lower = cond_name.lower()
        if "2b" in cn_lower:
            base = cn_lower.replace("_2b", "").replace(
                "2b_", ""
            )
            cond_2b[base] = scene_correct
        elif "8b" in cn_lower:
            base = cn_lower.replace("_8b", "").replace(
                "8b_", ""
            )
            cond_8b[base] = scene_correct

    if any_save:
        tags.append("tool_saved")
    if any_break:
        tags.append("tool_broke")
    if any_revised_correct:
        tags.append("revised_correctly")
    if any_revised_incorrect:
        tags.append("revised_incorrectly")
    if any_ignored_warning:
        tags.append("ignored_tool_warning")
    if len(final_scenes) <= 1 and len(final_scenes) > 0:
        tags.append("all_conditions_agree")

    # Incident zone overcorrection
    for _cond_name, entry in sample_conditions.items():
        orig = entry.get("original_scene")
        final = entry.get("final_scene")
        if orig == "incident_zone" and final != orig:
            tags.append("incident_zone_overcorrected")
            break

    # Nominal triggers subclass
    if fine_class == "nominal_triggers":
        tags.append("nominal_triggers")

    # 2B vs 8B comparison
    eight_better = False
    two_better = False
    for base_cond in set(cond_2b.keys()) & set(
        cond_8b.keys()
    ):
        c2 = cond_2b[base_cond]
        c8 = cond_8b[base_cond]
        if c8 is True and c2 is not True:
            eight_better = True
        if c2 is True and c8 is not True:
            two_better = True
    if eight_better:
        tags.append("8b_better_than_2b")
    if two_better:
        tags.append("2b_better_than_8b")

    return tags


def generate_report(db_path, output_path):
    """Generate per-sample JSONL report."""
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    experiments = get_all_experiments(conn)
    predictions_by_sample = get_all_predictions(conn)

    if not predictions_by_sample:
        print("ERROR: No predictions found in database.")
        conn.close()
        sys.exit(1)

    # Build experiment_id -> condition_name mapping
    eid_to_cond = {}
    for eid, exp in experiments.items():
        cond = exp["condition_name"]
        verifier = str(exp.get("verifier_model", ""))
        # Append verifier size to condition name
        if "2b" in verifier.lower() or "2b" in eid.lower():
            cond_key = f"{cond}_2b"
        elif (
            "8b" in verifier.lower()
            or "8b" in eid.lower()
        ):
            cond_key = f"{cond}_8b"
        else:
            cond_key = cond
        eid_to_cond[eid] = cond_key

    n_samples = 0
    n_tags = defaultdict(int)

    with open(output_path, "w") as f:
        for sample_id in sorted(predictions_by_sample.keys()):
            preds = predictions_by_sample[sample_id]

            # Get common ground truth from first pred
            first = preds[0]
            gt_scene = first.get("scene_type_gt", "")
            fine_class = first.get("fine_class", "")
            chum_uri = first.get("chum_uri", "")

            conditions = {}
            for pred in preds:
                eid = pred["experiment_id"]
                cond_name = eid_to_cond.get(eid, eid)

                tc_list = get_tool_calls_for_sample(
                    conn, eid, sample_id
                )
                entry = _build_condition_entry(
                    pred, tc_list
                )
                conditions[cond_name] = entry

            diag_tags = _compute_diagnostic_tags(
                conditions, gt_scene, fine_class
            )

            record = {
                "sample_id": sample_id,
                "chum_uri": chum_uri,
                "scene_type_gt": gt_scene,
                "fine_class": fine_class,
                "conditions": conditions,
                "diagnostic_tags": diag_tags,
            }

            f.write(json.dumps(record) + "\n")
            n_samples += 1
            for tag in diag_tags:
                n_tags[tag] += 1

    conn.close()

    # Print summary
    print(f"Sample report written to: {output_path}")
    print(f"Total samples: {n_samples}")
    print(f"Conditions per sample: {len(eid_to_cond)}")
    print()
    print("Diagnostic tag distribution:")
    for tag, count in sorted(
        n_tags.items(), key=lambda x: -x[1]
    ):
        pct = 100.0 * count / n_samples if n_samples else 0
        print(f"  {tag}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-sample JSONL report"
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
        help="Output JSONL file path",
    )
    args = parser.parse_args()
    generate_report(args.db, args.output)


if __name__ == "__main__":
    main()
