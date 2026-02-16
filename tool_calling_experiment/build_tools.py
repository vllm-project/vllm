#!/usr/bin/env python3
"""Build tool statistics JSON from the self-consistency DB.

Queries the self-consistency SQLite database to extract:
- Ground-truth class frequencies
- Confusion matrix (predicted vs ground truth)
- Scene-action co-occurrence counts
- Waypoint statistics per scene-action combination

Saves results to ``tool_calling_experiment/tool_stats.json``.

Usage:
    python tool_calling_experiment/build_tools.py \
        [--sc-db PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
from collections import defaultdict

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SC_DB = os.path.join(
    os.path.dirname(_DIR),
    "self_consistency_experiment",
    "self_consistency.db",
)
DEFAULT_OUTPUT = os.path.join(_DIR, "tool_stats.json")


def discover_schema(
    conn: sqlite3.Connection,
) -> dict[str, str]:
    """Print and return table schemas from the DB."""
    cursor = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table'"
    )
    schemas: dict[str, str] = {}
    for name, sql in cursor.fetchall():
        schemas[name] = sql or ""
        print(f"--- {name} ---")
        print(sql)
        print()
    return schemas


def _find_column(
    conn: sqlite3.Connection,
    table: str,
    candidates: list[str],
) -> str | None:
    """Return the first column in *candidates* that exists."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    cols = {row[1] for row in cursor.fetchall()}
    for c in candidates:
        if c in cols:
            return c
    return None


def build_class_frequencies(
    conn: sqlite3.Connection,
    table: str,
    gt_col: str,
) -> dict[str, float]:
    """Compute ground-truth class frequency distribution."""
    cursor = conn.execute(
        f"SELECT {gt_col}, COUNT(*) FROM {table} "  # noqa: S608
        f"GROUP BY {gt_col}"
    )
    counts: dict[str, int] = {}
    total = 0
    for scene, cnt in cursor.fetchall():
        if scene is None:
            continue
        counts[str(scene).strip()] = cnt
        total += cnt
    freqs = {
        k: round(v / total, 4) if total else 0.0
        for k, v in counts.items()
    }
    print(f"Class frequencies (n={total}): {freqs}")
    return freqs


def build_confusion_pairs(
    conn: sqlite3.Connection,
    table: str,
    gt_col: str,
    pred_col: str,
) -> dict[str, dict[str, object]]:
    """Build confusion-pair data from predictions vs GT."""
    # Get full confusion matrix
    cursor = conn.execute(
        f"SELECT {gt_col}, {pred_col}, COUNT(*) "  # noqa: S608
        f"FROM {table} GROUP BY {gt_col}, {pred_col}"
    )
    matrix: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    gt_totals: dict[str, int] = defaultdict(int)
    for gt, pred, cnt in cursor.fetchall():
        if gt is None or pred is None:
            continue
        gt_s = str(gt).strip()
        pred_s = str(pred).strip()
        matrix[gt_s][pred_s] += cnt
        gt_totals[gt_s] += cnt

    # For each class, find the most-confused-with class
    pairs: dict[str, dict[str, object]] = {}
    for gt_class, preds in matrix.items():
        # Most frequent misclassification
        errors = {
            p: c for p, c in preds.items() if p != gt_class
        }
        if not errors:
            continue
        worst = max(errors, key=lambda p: errors[p])
        err_count = errors[worst]
        total = gt_totals[gt_class]
        rate = round(err_count / total, 3) if total else 0.0
        if rate < 0.05:
            continue  # skip classes with low confusion
        pairs[gt_class] = {
            "confused_with": worst,
            "error_rate": rate,
            "error_count": err_count,
            "total": total,
            "note": (
                f"{rate:.1%} of {gt_class} scenes are "
                f"misclassified as {worst}."
            ),
        }
    print(f"Confusion pairs: {list(pairs.keys())}")
    return pairs


def build_cooccurrence(
    conn: sqlite3.Connection,
    table: str,
    gt_scene_col: str,
    long_col: str,
    lat_col: str,
) -> dict[str, dict[str, int]]:
    """Build scene-action co-occurrence from GT labels."""
    cursor = conn.execute(
        f"SELECT {gt_scene_col}, "  # noqa: S608
        f"{long_col}, {lat_col}, COUNT(*) "
        f"FROM {table} "
        f"GROUP BY {gt_scene_col}, {long_col}, {lat_col}"
    )
    coocc: dict[str, dict[str, int]] = defaultdict(dict)
    for scene, la, lat, cnt in cursor.fetchall():
        if scene is None:
            continue
        scene_s = str(scene).strip()
        la_s = str(la).strip() if la else "null"
        lat_s = str(lat).strip() if lat else "null"
        key = f"{la_s}|{lat_s}"
        coocc[scene_s][key] = cnt
    print(
        "Co-occurrence scenes: "
        f"{list(coocc.keys())}"
    )
    return dict(coocc)


def build_waypoint_stats(
    conn: sqlite3.Connection,
    table: str,
    gt_scene_col: str,
    long_col: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Build waypoint statistics per scene-action pair.

    Looks for columns containing waypoint data (x/y deltas).
    Returns nested dict: scene -> action -> stat_dict.
    """
    # Try to find waypoint columns
    cursor = conn.execute(f"PRAGMA table_info({table})")
    all_cols = [row[1] for row in cursor.fetchall()]
    wp_x_candidates = [
        c
        for c in all_cols
        if "waypoint" in c.lower() and "x" in c.lower()
    ]
    wp_y_candidates = [
        c
        for c in all_cols
        if "waypoint" in c.lower() and "y" in c.lower()
    ]

    # Also check for generated_text that might contain
    # waypoints
    if not wp_x_candidates:
        print(
            "No waypoint columns found. "
            "Returning empty stats."
        )
        print(f"  Available columns: {all_cols}")
        return {}

    wp_x_col = wp_x_candidates[0]
    wp_y_col = (
        wp_y_candidates[0] if wp_y_candidates else None
    )
    print(
        f"Using waypoint columns: x={wp_x_col}, "
        f"y={wp_y_col}"
    )

    cols = f"{gt_scene_col}, {long_col}, {wp_x_col}"
    if wp_y_col:
        cols += f", {wp_y_col}"

    cursor = conn.execute(
        f"SELECT {cols} FROM {table} "  # noqa: S608
        f"WHERE {wp_x_col} IS NOT NULL"
    )
    rows = cursor.fetchall()

    # Group by scene+action
    grouped: dict[
        tuple[str, str], list[tuple[float, float]]
    ] = defaultdict(list)
    for row in rows:
        scene = str(row[0]).strip() if row[0] else "unknown"
        action = str(row[1]).strip() if row[1] else "null"
        x_val = row[2]
        y_val = row[3] if wp_y_col else 0.0
        if x_val is not None:
            with contextlib.suppress(ValueError, TypeError):
                grouped[(scene, action)].append(
                    (float(x_val), float(y_val or 0.0))
                )

    # Compute stats
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for (scene, action), points in grouped.items():
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        n = len(xs)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        x_var = sum((x - x_mean) ** 2 for x in xs) / n
        y_var = sum((y - y_mean) ** 2 for y in ys) / n
        x_std = x_var**0.5
        y_std = y_var**0.5
        if scene not in stats:
            stats[scene] = {}
        stats[scene][action] = {
            "x_mean": round(x_mean, 4),
            "x_std": round(x_std, 4),
            "x_min": round(min(xs), 4),
            "x_max": round(max(xs), 4),
            "y_mean": round(y_mean, 4),
            "y_std": round(y_std, 4),
            "y_min": round(min(ys), 4),
            "y_max": round(max(ys), 4),
            "count": n,
        }
    print(f"Waypoint stats scenes: {list(stats.keys())}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build tool statistics from "
        "self-consistency DB"
    )
    parser.add_argument(
        "--sc-db",
        default=DEFAULT_SC_DB,
        help="Path to self-consistency SQLite DB",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to write tool_stats.json",
    )
    args = parser.parse_args()

    if not os.path.exists(args.sc_db):
        print(
            f"WARNING: Self-consistency DB not found "
            f"at {args.sc_db}"
        )
        print(
            "Writing empty tool_stats.json with "
            "hardcoded fallbacks."
        )
        with open(args.output, "w") as f:
            json.dump({}, f, indent=2)
        print(f"Wrote empty stats to {args.output}")
        return

    conn = sqlite3.connect(
        f"file:{args.sc_db}?mode=ro", uri=True
    )
    try:
        print("=== Schema Discovery ===")
        schemas = discover_schema(conn)

        if not schemas:
            print("ERROR: No tables found in DB.")
            sys.exit(1)

        # Find the main predictions/results table
        # Common table names to check
        table_candidates = [
            "predictions",
            "results",
            "samples",
            "inference_results",
        ]
        table = None
        for t in table_candidates:
            if t in schemas:
                table = t
                break
        if table is None:
            # Use the first non-internal table
            for t in schemas:
                if not t.startswith("sqlite_"):
                    table = t
                    break
        if table is None:
            print("ERROR: Could not find data table.")
            sys.exit(1)
        print(f"\nUsing table: {table}")

        # Find relevant columns
        gt_scene = _find_column(
            conn,
            table,
            [
                "scene_type_gt",
                "gt_scene",
                "ground_truth_scene",
                "scene_gt",
                "scene_type",
            ],
        )
        pred_scene = _find_column(
            conn,
            table,
            [
                "predicted_scene",
                "pred_scene",
                "scene_pred",
                "original_scene",
                "scene",
            ],
        )
        long_action = _find_column(
            conn,
            table,
            [
                "long_action_gt",
                "gt_long_action",
                "longitudinal_action",
                "long_action",
            ],
        )
        lat_action = _find_column(
            conn,
            table,
            [
                "lat_action_gt",
                "gt_lat_action",
                "lateral_action",
                "lat_action",
            ],
        )

        print(f"  GT scene col: {gt_scene}")
        print(f"  Pred scene col: {pred_scene}")
        print(f"  Long action col: {long_action}")
        print(f"  Lat action col: {lat_action}")

        output: dict[str, object] = {}

        # Build class frequencies
        if gt_scene:
            output["class_frequencies"] = (
                build_class_frequencies(
                    conn, table, gt_scene
                )
            )

        # Build confusion pairs
        if gt_scene and pred_scene:
            output["confusion_pairs"] = (
                build_confusion_pairs(
                    conn, table, gt_scene, pred_scene
                )
            )

        # Build co-occurrence
        if gt_scene and long_action and lat_action:
            output["cooccurrence"] = build_cooccurrence(
                conn,
                table,
                gt_scene,
                long_action,
                lat_action,
            )

        # Build waypoint stats
        if gt_scene and long_action:
            output["waypoint_stats"] = build_waypoint_stats(
                conn, table, gt_scene, long_action
            )

        # Write output
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote tool stats to {args.output}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
