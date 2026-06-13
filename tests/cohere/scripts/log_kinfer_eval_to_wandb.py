# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-process a finished Bee W&B run into a persistent trend run.

Bee logs only artifact tables (no scalars). This script:
  1. Finds the Bee run by display name.
  2. Downloads each ``*_metrics`` artifact table and extracts the primary score.
  3. Appends ONE scalar step to the stable per-(model, device) persistent run
     so the workspace shows a clean nightly time-series chart.

The persistent run ID is stable across nightlies (``resume="allow"``), so
selecting the B200 and MI300X runs in the W&B workspace automatically gives
two lines on the same chart.

Run ID prefix: ``vllm-ci-3``. Bump in run_kinfer_eval.py if a clean slate
is needed (deleted run IDs are tombstoned permanently by W&B).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import regex as re

logger = logging.getLogger(__name__)

WANDB_BASE_URL = "https://cohere.wandb.io"
WANDB_ENTITY = "cohere"
WANDB_PROJECT = "cohere-vllm-ci-nightly-evals"

# Maps the short label extracted from a task key to a W&B metric path.
_TASK_METRIC_KEY: dict[str, str] = {
    "python": "eval/code/HumanEval_Python",
    "mbpp": "eval/code/MBPP",
    "arc-c": "eval/llh/ARC-C",
    "boolq": "eval/llh/BoolQ",
    "winogrande": "eval/llh/WinoGrande",
    "openbookqa": "eval/llh/OpenBookQA",
    "gsm8k": "eval/gen/GSM8k",
    "triviaqa": "eval/gen/TriviaQA",
    "leaderboardbigbenchhard": "eval/gen/BIGBench",
}

# Tasks that produce one score per context length rather than a single scalar.
# Maps task label → {column_name → W&B metric path}.
# RULER/NIAH reports match_rate_at_<ctx> columns plus a task_average.
_NIAH_CTX_COLS: dict[str, str] = {
    "match_rate_at_4096": "4k",
    "match_rate_at_8192": "8k",
    "match_rate_at_32768": "32k",
    "match_rate_at_49152": "49k",
}
_MULTI_SCORE_TASKS: dict[str, dict[str, str]] = {
    "niah_multikey_2": {
        col: f"eval/lc/NIAH_multikey/{suffix}" for col, suffix in _NIAH_CTX_COLS.items()
    },
    "niah_single_2": {
        col: f"eval/lc/NIAH_single/{suffix}" for col, suffix in _NIAH_CTX_COLS.items()
    },
}

_NON_TASK_KEYS = frozenset(["debug_metrics", "summary_metrics"])
_PRIMARY_PREFIX = "1_"
_FALLBACK_COLS = [
    "accuracy",
    "exact_match",
    "pass_at_1",
    "pass@1",
    "maj@01",  # GSM8k greedy (majority-vote@1)
    "task_average",  # fallback for tasks with a single average score
    "f1",
    "score",
]

_KNOWN_DEVICES = ("b200", "mi300x", "h100", "h200", "gb200", "a100")


def _label(task_key: str) -> str:
    """Return a short, unique label used to look up the W&B metric path.

    Normal case: ``HumanEvalPack_Python_metrics`` → ``python`` (last token).
    Numeric-suffix case: ``RulerRetrieval_niah_multikey_2_metrics``
        → ``niah_multikey_2`` (everything after the CamelCase class prefix,
        joined with underscores), so that the two NIAH variants don't collide.
    """
    name = task_key.removesuffix("_metrics")
    parts = name.split("_")
    if parts and parts[-1].isdigit():
        # Skip the leading CamelCase class token (e.g. "RulerRetrieval") and
        # join the remainder to produce a collision-free label.
        return "_".join(p.lower() for p in parts[1:])[:40]
    return parts[-1].lower()[:40]


def _primary_score(columns: list[str], row: list) -> float | None:
    idx = {c: i for i, c in enumerate(columns)}
    for i, col in enumerate(columns):
        if col.startswith(_PRIMARY_PREFIX):
            v = row[i] if i < len(row) else None
            if isinstance(v, (int, float)):
                return float(v)
    for col in _FALLBACK_COLS:
        col_idx = idx.get(col)
        if col_idx is not None:
            v = row[col_idx] if col_idx < len(row) else None
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _download_table(api: Any, artifact_path: str) -> tuple[list[str], list]:
    with tempfile.TemporaryDirectory() as tmp:
        art = api.artifact(artifact_path)
        art.download(tmp)
        files = list(Path(tmp).rglob("*.table.json"))
        if not files:
            raise FileNotFoundError(f"No .table.json in {artifact_path}")
        data = json.loads(files[0].read_text())
    rows: list[list] = data.get("data", [])
    return data.get("columns", []), (rows[0] if rows else [])


def _extract_metrics(api: Any, bee_run: Any) -> dict[str, float]:
    run_id = bee_run.id
    task_keys = [
        k
        for k in bee_run.summary.keys()  # noqa: SIM118 — wandb Summary.__iter__ is not dict-like
        if k.endswith("_metrics") and k not in _NON_TASK_KEYS
    ]
    metrics: dict[str, float] = {}
    for task_key in task_keys:
        artifact_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/run-{run_id}-{task_key}:latest"
        try:
            cols, row = _download_table(api, artifact_path)
            label = _label(task_key)
            col_idx = {c: i for i, c in enumerate(cols)}
            if label in _MULTI_SCORE_TASKS:
                # Log one metric per context length (e.g. NIAH 4k/8k/32k/49k).
                for col, wandb_key in _MULTI_SCORE_TASKS[label].items():
                    i = col_idx.get(col)
                    if (
                        i is not None
                        and i < len(row)
                        and isinstance(row[i], (int, float))
                    ):
                        metrics[wandb_key] = float(row[i])
                        logger.info(
                            "  ✓ %s[%s] → %s = %.4f", task_key, col, wandb_key, row[i]
                        )
                    else:
                        logger.warning("  ✗ %s — missing column %s", task_key, col)
            else:
                score = _primary_score(cols, row)
                if score is None:
                    logger.warning("  ✗ %s — no primary score column", task_key)
                    continue
                wandb_key = _TASK_METRIC_KEY.get(label, f"eval/other/{label}")
                metrics[wandb_key] = score
                logger.info("  ✓ %s → %s = %.4f", task_key, wandb_key, score)
        except Exception as exc:
            logger.warning("  skipping %s: %s", task_key, exc)
    return metrics


def log_eval(
    bee_run_name: str,
    model: str,
    device: str | None = None,
    cluster: str | None = None,
    tp: int | None = None,
) -> None:
    """Extract scores from a finished Bee run and append one step to the
    persistent per-(model, device) W&B trend run.

    The timestamp is extracted from the bee run name (YYYYMMDD-HHMM).  All
    GPU jobs in the same nightly share an identical run_timestamp injected by
    the parent workflow, so they automatically land at the same x position.
    Falls back to date-only (YYYYMMDD) for run names that lack a time component.
    """
    if not os.environ.get("WANDB_API_KEY"):
        logger.warning(
            "WANDB_API_KEY not set — skipping W&B logging for '%s'.", bee_run_name
        )
        return

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping W&B logging.")
        return

    os.environ.setdefault("WANDB_BASE_URL", WANDB_BASE_URL)
    api = wandb.Api(overrides={"base_url": WANDB_BASE_URL})

    # Locate the Bee run by display name (most recent match).
    # Exclude our own persistent trend runs (job_type="kinfer-eval") so the
    # search never accidentally picks up the run we previously wrote to.
    # W&B indexing can lag a few minutes after a run finishes, so retry.
    _LOOKUP_ATTEMPTS = 5
    _LOOKUP_DELAY_S = 30
    bee_run = None
    for attempt in range(_LOOKUP_ATTEMPTS):
        candidates = list(
            api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={
                    "display_name": bee_run_name,
                    "jobType": {"$ne": "kinfer-eval"},
                },
                order="-created_at",
                per_page=5,
            )
        )
        bee_run = next((r for r in candidates if r.name == bee_run_name), None)
        if bee_run is not None:
            break
        if attempt < _LOOKUP_ATTEMPTS - 1:
            logger.warning(
                "Bee run '%s' not yet indexed in W&B (attempt %d/%d) \
                — retrying in %ds. "
                "Filter returned: %s",
                bee_run_name,
                attempt + 1,
                _LOOKUP_ATTEMPTS,
                _LOOKUP_DELAY_S,
                [r.name for r in candidates] or "nothing",
            )
            time.sleep(_LOOKUP_DELAY_S)
    if bee_run is None:
        logger.warning(
            "Bee run '%s' not found in W&B after %d attempts — skipping.",
            bee_run_name,
            _LOOKUP_ATTEMPTS,
        )
        return

    metrics = _extract_metrics(api, bee_run)
    if not metrics:
        logger.warning("No metrics extracted from '%s' — skipping.", bee_run_name)
        return

    # Extract the timestamp from the bee run name.  All GPU jobs in the same
    # nightly receive an identical run_timestamp from the parent workflow, so
    # B200 and MI300X always land at the same x position.  Two runs on the
    # same day get distinct timestamps and therefore distinct x positions.
    m_full = re.search(r"(\d{8}-\d{4})", bee_run_name)
    if m_full:
        run_date = m_full.group(1)
        commit_date_unix = (
            datetime.strptime(run_date, "%Y%m%d-%H%M")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    else:
        m_date = re.search(r"(\d{8})", bee_run_name)
        run_date = (
            m_date.group(1) if m_date else datetime.now(timezone.utc).strftime("%Y%m%d")
        )
        commit_date_unix = (
            datetime.strptime(run_date, "%Y%m%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

    effective_device = device or next(
        (g for g in _KNOWN_DEVICES if model.endswith(f"-{g}")), model.split("-")[-1]
    )

    from run_kinfer_eval import _WANDB_RUN_ID_PREFIX  # type: ignore[import]

    # Normalise the model key for the persistent run ID: strip any variant suffix
    # that follows the device (e.g. "mls-base-fp8-h100-llh" → "mls-base-fp8-h100")
    # so that all hardware variants of the same model share one W&B trend run.
    m_dev = re.search(rf"^(.+-{re.escape(effective_device)})(?:-.+)?$", model)
    model_for_id = m_dev.group(1) if m_dev else model

    # Derive display name from the normalised key so the variant suffix (-llh,
    # -gen, …) never leaks into the W&B run name.
    base_model = (
        model_for_id[: -(len(effective_device) + 1)]
        if model_for_id.endswith(f"-{effective_device}")
        else model_for_id
    )

    run_id = f"{_WANDB_RUN_ID_PREFIX}-{model_for_id}"[:64]
    run_name = f"{base_model} ({effective_device})"

    logger.info(
        "Logging %d metrics to '%s' (id=%s, date=%s)",
        len(metrics),
        run_name,
        run_id,
        run_date,
    )

    trend_run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        id=run_id,
        name=run_name,
        resume="allow",
        config={
            "model": base_model,
            "device": effective_device,
            "cluster": cluster or "",
            "tp": tp or 0,
        },
        job_type="kinfer-eval",
        tags=[f"model:{base_model}", f"device:{effective_device}"],
        settings=wandb.Settings(base_url=WANDB_BASE_URL),
        reinit="finish_previous",
    )
    assert trend_run is not None
    # Use commit_date_unix as the x-axis so the charts show calendar dates and
    # so B200/MI300X runs from the same nightly land at the same x position.
    wandb.define_metric("commit_date_unix")
    wandb.define_metric("eval/*", step_metric="commit_date_unix")
    wandb.log({"commit_date_unix": commit_date_unix, **metrics})
    trend_run.finish()
    logger.info("Run '%s' updated.", run_name)
