#!/usr/bin/env python3
"""Create the SQLite database for the tool-calling experiment.

Usage:
    python tool_calling_experiment/create_db.py [--db-path PATH]

Creates the DB at tool_calling_experiment/tool_calling.db by default.
"""

import argparse
import os
import sqlite3

_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(_DIR, "tool_calling.db")

SCHEMA_SQL = """\
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT,
    condition_name TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    predictor_model TEXT,
    verifier_model TEXT,
    tools_enabled TEXT,
    temperature_predict REAL DEFAULT 0.0,
    temperature_verify REAL DEFAULT 0.0,
    seed INTEGER DEFAULT 42,
    total_samples INTEGER,
    total_wall_time_s REAL,
    predict_wall_time_s REAL,
    verify_wall_time_s REAL,
    throughput_samples_per_s REAL,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL
        REFERENCES experiments(experiment_id),
    sample_id INTEGER NOT NULL,
    chum_uri TEXT,
    original_scene TEXT,
    original_long_action TEXT,
    original_lat_action TEXT,
    original_generated_text TEXT,
    original_scene_correct INTEGER,
    final_scene TEXT,
    final_long_action TEXT,
    final_lat_action TEXT,
    final_generated_text TEXT,
    final_scene_correct INTEGER,
    was_revised INTEGER,
    scene_was_revised INTEGER,
    revision_reason TEXT,
    scene_type_gt TEXT,
    long_action_gt TEXT,
    lat_action_gt TEXT,
    odd_label TEXT,
    fine_class TEXT,
    location TEXT,
    original_flipped_correct INTEGER,
    original_flipped_incorrect INTEGER,
    predict_time_ms REAL,
    verify_time_ms REAL,
    total_time_ms REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(experiment_id, sample_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_exp
    ON predictions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_predictions_revised
    ON predictions(experiment_id, was_revised);
CREATE INDEX IF NOT EXISTS idx_predictions_flip
    ON predictions(
        original_flipped_correct,
        original_flipped_incorrect
    );

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    sample_id INTEGER NOT NULL,
    tool_call_order INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    tool_arguments_json TEXT,
    tool_result_json TEXT,
    model_revised_after INTEGER,
    revised_field TEXT,
    old_value TEXT,
    new_value TEXT,
    created_at TEXT NOT NULL
        DEFAULT (datetime('now')),
    UNIQUE(
        experiment_id, sample_id, tool_call_order
    )
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_exp
    ON tool_calls(experiment_id, sample_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_name
    ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_revised
    ON tool_calls(model_revised_after);

CREATE TABLE IF NOT EXISTS condition_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    scene_accuracy REAL,
    scene_macro_f1 REAL,
    n_revised INTEGER,
    revision_rate REAL,
    revision_accuracy REAL,
    n_saves INTEGER,
    n_breaks INTEGER,
    net_improvement INTEGER,
    tool_prior_call_count INTEGER,
    tool_prior_revision_rate REAL,
    tool_confusion_call_count INTEGER,
    tool_confusion_revision_rate REAL,
    tool_scene_action_call_count INTEGER,
    tool_scene_action_revision_rate REAL,
    tool_waypoint_call_count INTEGER,
    tool_waypoint_revision_rate REAL,
    mean_predict_time_ms REAL,
    mean_verify_time_ms REAL,
    mean_total_time_ms REAL,
    p95_total_time_ms REAL,
    created_at TEXT NOT NULL
        DEFAULT (datetime('now')),
    UNIQUE(experiment_id)
);
"""


def create_database(db_path: str) -> None:
    """Create the tool_calling database with full schema."""
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        print(f"Database created at: {db_path}")
        cursor = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables: {tables}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create tool_calling experiment database"
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to create the database "
        f"(default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()
    create_database(args.db_path)


if __name__ == "__main__":
    main()
