#!/usr/bin/env python3
"""Prediction phase for the tool-calling verification experiment.

Two modes:
1. ``--reuse-greedy``: Copy greedy baseline results from the
   self-consistency DB into the tool_calling DB.
2. Default: Run fine-tuned model inference (n=1, temp=0).

Usage:
    python tool_calling_experiment/run_prediction.py \
        --reuse-greedy \
        [--db-path PATH] \
        [--self-consistency-db PATH]

    python tool_calling_experiment/run_prediction.py \
        --model-path /workspace/vllm/models/checkpoint \
        --gpu-id 0 \
        [--db-path PATH]
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(_DIR, "tool_calling.db")
DEFAULT_SC_DB = os.path.join(
    os.path.dirname(_DIR),
    "self_consistency_experiment",
    "self_consistency.db",
)
DEFAULT_MODEL = os.path.join(
    os.path.dirname(_DIR), "models", "checkpoint"
)

VALID_SCENES = {
    "nominal",
    "flooded",
    "incident_zone",
    "mounted_police",
    "flagger",
}


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------


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


def _discover_sc_schema(
    conn: sqlite3.Connection,
) -> dict[str, str]:
    """Discover schema of the self-consistency DB."""
    cursor = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table'"
    )
    schemas: dict[str, str] = {}
    for name, sql in cursor.fetchall():
        schemas[name] = sql or ""
    return schemas


def _parse_scene_from_text(text: str) -> str | None:
    """Parse scene type from generated text."""
    if not text:
        return None
    lower = text.lower()
    for scene in VALID_SCENES:
        pat = rf"\b{re.escape(scene)}\b"
        if re.search(pat, lower):
            return scene
    return None


# ----------------------------------------------------------
# Reuse-greedy mode
# ----------------------------------------------------------


def reuse_greedy(
    db_path: str, sc_db_path: str
) -> None:
    """Copy greedy baseline from self-consistency DB."""
    if not os.path.exists(sc_db_path):
        print(
            f"ERROR: Self-consistency DB not found: "
            f"{sc_db_path}"
        )
        sys.exit(1)

    sc_conn = sqlite3.connect(
        f"file:{sc_db_path}?mode=ro", uri=True
    )
    sc_conn.row_factory = sqlite3.Row

    try:
        # Discover schema
        schemas = _discover_sc_schema(sc_conn)
        print("Self-consistency DB tables:")
        for name in schemas:
            print(f"  {name}")

        # Find the right table
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
            for t in schemas:
                if not t.startswith("sqlite_"):
                    table = t
                    break
        if table is None:
            print("ERROR: No data table found.")
            sys.exit(1)
        print(f"Using table: {table}")

        # Find columns
        col_map = {
            "sample_id": _find_column(
                sc_conn,
                table,
                ["sample_id", "id", "idx"],
            ),
            "chum_uri": _find_column(
                sc_conn,
                table,
                ["chum_uri", "uri", "image_path"],
            ),
            "pred_scene": _find_column(
                sc_conn,
                table,
                [
                    "predicted_scene",
                    "pred_scene",
                    "scene_pred",
                    "scene",
                ],
            ),
            "pred_long": _find_column(
                sc_conn,
                table,
                [
                    "predicted_long_action",
                    "pred_long_action",
                    "long_action_pred",
                    "long_action",
                ],
            ),
            "pred_lat": _find_column(
                sc_conn,
                table,
                [
                    "predicted_lat_action",
                    "pred_lat_action",
                    "lat_action_pred",
                    "lat_action",
                ],
            ),
            "gt_scene": _find_column(
                sc_conn,
                table,
                [
                    "scene_type_gt",
                    "gt_scene",
                    "ground_truth_scene",
                    "scene_gt",
                ],
            ),
            "gt_long": _find_column(
                sc_conn,
                table,
                [
                    "long_action_gt",
                    "gt_long_action",
                ],
            ),
            "gt_lat": _find_column(
                sc_conn,
                table,
                [
                    "lat_action_gt",
                    "gt_lat_action",
                ],
            ),
            "odd_label": _find_column(
                sc_conn,
                table,
                ["odd_label", "odd"],
            ),
            "fine_class": _find_column(
                sc_conn,
                table,
                ["fine_class", "fine"],
            ),
            "location": _find_column(
                sc_conn,
                table,
                ["location", "loc"],
            ),
            "generated_text": _find_column(
                sc_conn,
                table,
                [
                    "generated_text",
                    "output",
                    "response",
                    "text",
                ],
            ),
            "temperature": _find_column(
                sc_conn,
                table,
                ["temperature", "temp"],
            ),
            "sample_index": _find_column(
                sc_conn,
                table,
                [
                    "sample_index",
                    "sample_idx",
                    "n_index",
                ],
            ),
        }

        print("Column mapping:")
        for k, v in col_map.items():
            print(f"  {k} -> {v}")

        # Build query for greedy predictions
        # Greedy = temperature 0 or first sample
        select_cols = []
        for key, col in col_map.items():
            if col is not None:
                select_cols.append(f"{col} AS {key}")

        if not select_cols:
            print("ERROR: No usable columns found.")
            sys.exit(1)

        query = f"SELECT {', '.join(select_cols)} FROM {table}"  # noqa: E501, S608

        # Filter for greedy (temp=0 or sample_index=0)
        conditions = []
        if col_map["temperature"] is not None:
            conditions.append(
                f"{col_map['temperature']} = 0"
            )
        elif col_map["sample_index"] is not None:
            conditions.append(
                f"{col_map['sample_index']} = 0"
            )

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        print(f"Query: {query}")
        cursor = sc_conn.execute(query)
        rows = cursor.fetchall()
        print(f"Found {len(rows)} greedy predictions")

        if not rows:
            print(
                "ERROR: No greedy predictions found. "
                "Check the query."
            )
            sys.exit(1)

    finally:
        sc_conn.close()

    # Write to tool_calling DB
    tc_conn = sqlite3.connect(db_path)
    try:
        # Create baseline experiment record
        tc_conn.execute(
            "INSERT OR REPLACE INTO experiments "
            "(experiment_id, condition_name, pipeline, "
            "predictor_model, temperature_predict, "
            "total_samples, status, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "baseline",
                "baseline",
                "A",
                DEFAULT_MODEL,
                0.0,
                len(rows),
                "completed",
                "Greedy baseline copied from "
                "self-consistency DB",
            ),
        )

        inserted = 0
        for row in rows:
            r = dict(row)
            sid = r.get("sample_id", inserted)
            pred_scene = r.get("pred_scene")
            gt_scene = r.get("gt_scene")
            scene_correct = (
                int(pred_scene == gt_scene)
                if pred_scene and gt_scene
                else None
            )

            tc_conn.execute(
                "INSERT OR REPLACE INTO predictions "
                "(experiment_id, sample_id, chum_uri, "
                "original_scene, original_long_action, "
                "original_lat_action, "
                "original_generated_text, "
                "original_scene_correct, "
                "final_scene, final_long_action, "
                "final_lat_action, "
                "final_scene_correct, "
                "was_revised, scene_was_revised, "
                "scene_type_gt, long_action_gt, "
                "lat_action_gt, odd_label, "
                "fine_class, location) "
                "VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "baseline",
                    sid,
                    r.get("chum_uri"),
                    pred_scene,
                    r.get("pred_long"),
                    r.get("pred_lat"),
                    r.get("generated_text"),
                    scene_correct,
                    # final = original for baseline
                    pred_scene,
                    r.get("pred_long"),
                    r.get("pred_lat"),
                    scene_correct,
                    0,  # was_revised
                    0,  # scene_was_revised
                    gt_scene,
                    r.get("gt_long"),
                    r.get("gt_lat"),
                    r.get("odd_label"),
                    r.get("fine_class"),
                    r.get("location"),
                ),
            )
            inserted += 1

        tc_conn.commit()
        print(
            f"Inserted {inserted} baseline predictions "
            f"into {db_path}"
        )

        # Verify
        count = tc_conn.execute(
            "SELECT COUNT(*) FROM predictions "
            "WHERE experiment_id='baseline'"
        ).fetchone()[0]
        correct = tc_conn.execute(
            "SELECT SUM(original_scene_correct) "
            "FROM predictions "
            "WHERE experiment_id='baseline'"
        ).fetchone()[0]
        print(f"Verification: {count} rows")
        if correct is not None and count > 0:
            print(
                f"Baseline scene accuracy: "
                f"{correct}/{count} = "
                f"{correct / count:.4f}"
            )

    finally:
        tc_conn.close()


# ----------------------------------------------------------
# Fresh inference mode
# ----------------------------------------------------------


def run_inference(
    db_path: str,
    model_path: str,
    gpu_id: int,
) -> None:
    """Run fresh greedy inference with fine-tuned model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import vllm after setting CUDA_VISIBLE_DEVICES
    from vllm import LLM, SamplingParams  # type: ignore  # noqa: E402

    print(f"Loading model from {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        seed=42,
    )

    # Load dataset
    # Assumes MDS format dataset
    dataset_path = os.path.join(
        os.path.dirname(_DIR), "models", "dataset"
    )
    if not os.path.exists(dataset_path):
        print(
            f"ERROR: Dataset not found at "
            f"{dataset_path}"
        )
        print("Use --reuse-greedy instead.")
        sys.exit(1)

    try:
        from streaming import StreamingDataset  # type: ignore
    except ImportError:
        print(
            "ERROR: mosaicml-streaming not installed. "
            "Install with: pip install mosaicml-streaming"
        )
        sys.exit(1)

    dataset = StreamingDataset(
        local=dataset_path, shuffle=False
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Create experiment record
    tc_conn = sqlite3.connect(db_path)
    exp_id = "baseline"
    tc_conn.execute(
        "INSERT OR REPLACE INTO experiments "
        "(experiment_id, condition_name, pipeline, "
        "predictor_model, temperature_predict, "
        "total_samples, status, description) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            exp_id,
            "baseline",
            "A",
            model_path,
            0.0,
            len(dataset),
            "running",
            "Fresh greedy inference with "
            "fine-tuned model",
        ),
    )
    tc_conn.commit()

    # Build prompts
    messages_list = []
    sample_meta = []
    for i in range(len(dataset)):
        sample = dataset[i]
        # Extract image and metadata
        image = sample.get("image")
        meta = {
            "sample_id": i,
            "chum_uri": sample.get("chum_uri", ""),
            "scene_type_gt": sample.get(
                "scene_type", ""
            ),
            "long_action_gt": sample.get(
                "long_action", ""
            ),
            "lat_action_gt": sample.get(
                "lat_action", ""
            ),
            "odd_label": sample.get("odd_label", ""),
            "fine_class": sample.get("fine_class", ""),
            "location": sample.get("location", ""),
        }
        sample_meta.append(meta)
        prompt_text = sample.get("prompt", "")
        msg: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        if image is not None:
            msg[0]["content"].insert(
                0,
                {
                    "type": "image_url",
                    "image_url": {"url": image},
                },
            )
        messages_list.append(msg)

    # Run inference in batches
    start_time = time.time()
    outputs = llm.chat(
        messages_list,
        sampling_params=sampling_params,
    )
    wall_time = time.time() - start_time

    # Store results
    inserted = 0
    for idx, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        meta = sample_meta[idx]
        pred_scene = _parse_scene_from_text(text)
        gt_scene = meta["scene_type_gt"]
        scene_correct = (
            int(pred_scene == gt_scene)
            if pred_scene and gt_scene
            else None
        )
        tc_conn.execute(
            "INSERT OR REPLACE INTO predictions "
            "(experiment_id, sample_id, chum_uri, "
            "original_scene, original_generated_text, "
            "original_scene_correct, "
            "final_scene, final_scene_correct, "
            "was_revised, scene_was_revised, "
            "scene_type_gt, long_action_gt, "
            "lat_action_gt, odd_label, "
            "fine_class, location) "
            "VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                exp_id,
                meta["sample_id"],
                meta["chum_uri"],
                pred_scene,
                text,
                scene_correct,
                pred_scene,
                scene_correct,
                0,
                0,
                gt_scene,
                meta["long_action_gt"],
                meta["lat_action_gt"],
                meta["odd_label"],
                meta["fine_class"],
                meta["location"],
            ),
        )
        inserted += 1

    # Update experiment
    tc_conn.execute(
        "UPDATE experiments SET "
        "status='completed', "
        "total_wall_time_s=?, "
        "predict_wall_time_s=?, "
        "throughput_samples_per_s=? "
        "WHERE experiment_id=?",
        (
            wall_time,
            wall_time,
            inserted / wall_time if wall_time > 0 else 0,
            exp_id,
        ),
    )
    tc_conn.commit()
    tc_conn.close()

    print(
        f"Inserted {inserted} predictions in "
        f"{wall_time:.1f}s"
    )


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run prediction phase for "
        "tool-calling experiment"
    )
    parser.add_argument(
        "--reuse-greedy",
        action="store_true",
        help="Copy greedy baseline from "
        "self-consistency DB",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB,
        help="Path to tool_calling DB",
    )
    parser.add_argument(
        "--self-consistency-db",
        default=DEFAULT_SC_DB,
        help="Path to self-consistency DB",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use",
    )
    args = parser.parse_args()

    if args.reuse_greedy:
        reuse_greedy(args.db_path, args.self_consistency_db)
    else:
        run_inference(
            args.db_path, args.model_path, args.gpu_id
        )


if __name__ == "__main__":
    main()
