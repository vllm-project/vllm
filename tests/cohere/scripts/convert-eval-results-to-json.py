# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import datetime
import glob
import json
import logging
import os
import subprocess
from pathlib import Path

import regex as re

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_DIR = SCRIPT_DIR.parent.parent.parent
MODEL_PATH_PREFIX = os.environ.get("MODEL_PATH_PREFIX")
TASK_KEY_MAP_PATH = SCRIPT_DIR.parent / "configs" / "bee_tasks" / "task_key_map.json"


def extract_model_name(file_name: str) -> str | None:
    file_name = os.path.basename(file_name)
    match = re.match(r"^(.+)_tp\d+_[0-9a-fA-F\-]+_summary_metrics\.jsonl$", file_name)
    if match:
        return match.group(1)
    else:
        return None


def _sanitize_eval_results(eval_results: dict) -> dict:
    """Normalize metric keys that may contain path separators or spaces.

    Bee likelihood tasks can emit metric names such as
    ``1_overall/accuracy`` or ``2_social sciences/accuracy`` directly into
    summary_metrics.jsonl when TOML summary_metrics sections are absent.
    Replace ``/`` and spaces with ``_`` so that downstream JSON-path
    consumers and comparison logic are not confused by these characters.
    """
    return {k.replace("/", "_").replace(" ", "_"): v for k, v in eval_results.items()}


def _bee_task_id_from_metrics_suffix(task_and_est: str, estimator_name: str) -> str:
    """
    task_and_est is the part after '_metrics_' in our copied filename, i.e. bee's
    metrics basename without .jsonl
    (e.g. 'HeuristicMathGeneration.aime_2025_aa_c5-3a30t_fp8_tp1').

    bee writes ``metrics_{task.name}_{estimator.name}.jsonl`` and
    ``estimator.name`` defaults to the ``--model`` value
    (see ``hive.base.BaseEstimator.set_default_name``), so we strip that
    dynamically rather than relying on a fixed literal.
    """
    s = task_and_est
    if s.startswith("metrics_"):
        s = s[len("metrics_") :]
    suffix = "_" + estimator_name
    if s.endswith(suffix):
        s = s[: -len(suffix)]
    return s


def _load_task_key_map() -> dict[str, str]:
    """Load the bee-task-ID-to-canonical-key mapping from task_key_map.json."""
    with open(TASK_KEY_MAP_PATH) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _canonical_ci_task_key(bee_task_id: str) -> str | None:
    """
    Map bee task.name (possibly with CI/template suffixes) to keys aligned with
    eval_results / ground_truths.json. The mapping is loaded from
    task_key_map.json; pattern order matters if patterns overlap.
    """
    task_key_map = _load_task_key_map()
    for pattern, key in task_key_map.items():
        if re.search(pattern, bee_task_id):
            return key
    return None


def _merge_per_task_metrics(folder_path: str | Path, combined_data: list[dict]) -> None:
    """Merge per-task bee metrics into the combined summary data.

    Bee (the eval harness) writes a ``metrics_*.jsonl`` file per task alongside
    the ``summary_metrics.jsonl`` file.  ``run-bee-eval.sh`` copies these into
    the results folder with the naming convention::

        {model_name}_tp{N}_{run_uuid}_metrics_{bee_metrics_basename}.jsonl

    Each JSONL row contains aggregated counters:
      * ``usage/response`` -- total generated tokens (sum over all samples)
      * ``usage/thinking`` -- total thinking tokens (sum over all samples; 0 if absent)
      * ``sample_timing/total`` -- wall-clock sample time in seconds (sum)
      * ``num_samples`` -- number of evaluated samples

    This function parses those files, maps bee task IDs to canonical CI task
    keys (via ``task_key_map.json``), and attaches per-task totals and averages
    for token usage and timing to the corresponding model entry in
    *combined_data*.
    """
    by_model = {d["model"]: d for d in combined_data}
    usage_total_by_task: dict[str, dict[str, float]] = {}
    thinking_total_by_task: dict[str, dict[str, float]] = {}
    timing_total_by_task: dict[str, dict[str, float]] = {}
    num_samples_by_task: dict[str, dict[str, int]] = {}

    for path in glob.glob(os.path.join(folder_path, "*.jsonl")):
        file_name = os.path.basename(path)
        # Per-task metrics files are copied by run-bee-eval.sh as:
        # {model_name}_tp{N}_{run_uuid}_metrics_{bee_metrics_basename}.jsonl
        match = re.match(
            r"^(.+)_(tp\d+)_[0-9a-fA-F\-]+_metrics_(.+)\.jsonl$", file_name
        )
        if not match:
            continue
        model_name = match.group(1)
        served_model_name = f"{model_name}_{match.group(2)}"
        task_and_est = match.group(3)
        if model_name not in by_model:
            continue
        bee_task_id = _bee_task_id_from_metrics_suffix(
            task_and_est, estimator_name=served_model_name
        )
        canonical = _canonical_ci_task_key(bee_task_id)
        if canonical is None:
            logger.warning(
                "Unrecognized bee task ID %r in file %s -- skipping",
                bee_task_id,
                file_name,
            )
            continue

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                num = row.get("num_samples")

                if num is not None:
                    num_samples_by_task.setdefault(model_name, {})
                    num_samples_by_task[model_name][canonical] = (
                        num_samples_by_task[model_name].get(canonical, 0) + num
                    )

                ur = row.get("usage/response")
                if ur is not None:
                    usage_total_by_task.setdefault(model_name, {})
                    usage_total_by_task[model_name][canonical] = usage_total_by_task[
                        model_name
                    ].get(canonical, 0.0) + float(ur)

                ut = float(row.get("usage/thinking", 0))
                thinking_total_by_task.setdefault(model_name, {})
                thinking_total_by_task[model_name][canonical] = (
                    thinking_total_by_task[model_name].get(canonical, 0.0) + ut
                )

                st = row.get("sample_timing/total")
                if st is not None:
                    timing_total_by_task.setdefault(model_name, {})
                    timing_total_by_task[model_name][canonical] = timing_total_by_task[
                        model_name
                    ].get(canonical, 0.0) + float(st)

    # Compute per-task averages from accumulated totals and sample counts.
    usage_avg_by_task: dict[str, dict[str, float]] = {}
    thinking_avg_by_task: dict[str, dict[str, float]] = {}
    timing_avg_by_task: dict[str, dict[str, float]] = {}

    all_models = (
        set(usage_total_by_task)
        | set(thinking_total_by_task)
        | set(timing_total_by_task)
    )
    for model_name in all_models:
        total_n = num_samples_by_task.get(model_name, {})
        for key, total in usage_total_by_task.get(model_name, {}).items():
            n = total_n.get(key, 0)
            if n > 0:
                usage_avg_by_task.setdefault(model_name, {})[key] = total / n
        for key, total in thinking_total_by_task.get(model_name, {}).items():
            n = total_n.get(key, 0)
            if n > 0:
                thinking_avg_by_task.setdefault(model_name, {})[key] = total / n
        for key, total in timing_total_by_task.get(model_name, {}).items():
            n = total_n.get(key, 0)
            if n > 0:
                timing_avg_by_task.setdefault(model_name, {})[key] = total / n

    for model_name, data in by_model.items():
        if model_name in usage_total_by_task:
            data["usage_response_tokens_by_task"] = usage_total_by_task[model_name]
        if model_name in usage_avg_by_task:
            data["usage_response_tokens_avg_by_task"] = usage_avg_by_task[model_name]
        if model_name in thinking_total_by_task:
            data["usage_thinking_tokens_by_task"] = thinking_total_by_task[model_name]
        if model_name in thinking_avg_by_task:
            data["usage_thinking_tokens_avg_by_task"] = thinking_avg_by_task[model_name]
        if model_name in timing_total_by_task:
            data["sample_timing_total_by_task"] = timing_total_by_task[model_name]
        if model_name in timing_avg_by_task:
            data["sample_timing_avg_by_task"] = timing_avg_by_task[model_name]


def main(folder_path: str | Path, output_json_file: str | Path) -> None:
    if not MODEL_PATH_PREFIX:
        raise ValueError("MODEL_PATH_PREFIX environment variable is required.")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    )
    branch_name = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("utf-8")
        .strip()
    )

    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))

    combined_data: list[dict] = []

    for jsonl_file in jsonl_files:
        model_name = extract_model_name(jsonl_file)
        if model_name is None:
            continue
        checkpoint_path = f"{MODEL_PATH_PREFIX.rstrip('/')}/{model_name}"
        with open(jsonl_file) as f:
            for line in f:
                parsed = _sanitize_eval_results(json.loads(line))
                model_exists = False
                for data in combined_data:
                    if data.get("model") == model_name:
                        model_exists = True
                        data["eval_results"].update(parsed)
                        break
                if not model_exists:
                    data = {}
                    data["model"] = model_name
                    data["checkpoint_path"] = checkpoint_path
                    data["timestamp"] = timestamp
                    data["commit"] = commit_hash
                    data["branch"] = branch_name
                    data["eval_results"] = parsed
                    combined_data.append(data)

    _merge_per_task_metrics(folder_path, combined_data)

    with open(output_json_file, "w") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combine JSONL eval results into a single JSON file"
    )
    parser.add_argument(
        "--folder_path",
        default=VLLM_DIR / "results",
        required=False,
        help="Path to the folder containing JSONL eval results",
    )
    parser.add_argument(
        "--output_json_file",
        default=VLLM_DIR / "results" / "eval_results_summary.json",
        required=False,
        help="Path to the output summary JSON file",
    )
    args = parser.parse_args()

    main(args.folder_path, args.output_json_file)
