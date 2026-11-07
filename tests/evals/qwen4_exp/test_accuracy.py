# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GSM8K and AIME25 accuracy evaluation for Qwen3.8-Flash-Next-FP8."""

import shlex
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from tests.utils import RemoteOpenAIServer


def run_evalscope(
    eval_config: dict[str, Any],
    base_url: str,
    work_dir: Path,
    datasets: list[str] | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    from evalscope.run import run_task

    generation_config = dict(eval_config["generation_config"])
    if seed is not None:
        generation_config["seed"] = seed

    reports = run_task(
        {
            "model": eval_config["model_name"],
            "api_url": base_url,
            "api_key": "EMPTY_TOKEN",
            "datasets": datasets or list(eval_config["datasets"]),
            "eval_batch_size": eval_config.get("eval_batch_size", 32),
            "generation_config": generation_config,
            "work_dir": str(work_dir),
            "no_timestamp": True,
        }
    )
    if not isinstance(reports, dict):
        raise TypeError(f"Expected EvalScope reports to be a dict, got {type(reports)}")

    return {dataset: float(report.score) for dataset, report in reports.items()}


def _datasets_needing_retry(
    eval_config: dict[str, Any],
    score_history: dict[str, list[float]],
    next_trial: int,
) -> list[str]:
    retry_datasets = []
    for dataset, metric_config in eval_config["datasets"].items():
        minimum_score = metric_config["metric_threshold"] - metric_config.get(
            "tolerance", 0.0
        )
        if (
            next_trial < metric_config.get("max_score_trials", 1)
            and fmean(score_history[dataset]) < minimum_score
        ):
            retry_datasets.append(dataset)
    return retry_datasets


def test_qwen4_exp_accuracy(config_filename: Path, tmp_path: Path):
    eval_config = yaml.safe_load(config_filename.read_text(encoding="utf-8"))
    server_args = shlex.split(eval_config.get("server_args", ""))
    server_args.extend(["--trust-remote-code", "--disable-uvicorn-access-log"])

    model_name = eval_config["model_name"]
    print(f"Starting Qwen4Exp evaluation for model: {model_name}")
    print(f"Datasets: {', '.join(eval_config['datasets'])}")
    print(f"Server args: {' '.join(server_args)}")

    with RemoteOpenAIServer(
        model_name,
        server_args,
        env_dict=eval_config.get("env"),
        max_wait_seconds=eval_config.get("startup_max_wait_seconds", 1800),
    ) as remote_server:
        scores = run_evalscope(eval_config, remote_server.url_for("v1"), tmp_path)
        score_history = {dataset: [score] for dataset, score in scores.items()}

        max_trials = max(
            config.get("max_score_trials", 1)
            for config in eval_config["datasets"].values()
        )
        for trial in range(1, max_trials):
            retry_datasets = _datasets_needing_retry(eval_config, score_history, trial)
            if not retry_datasets:
                break

            base_seed = eval_config["generation_config"].get("seed")
            if type(base_seed) is not int:
                raise ValueError("max_score_trials requires an integer generation seed")
            retry_seed = base_seed + trial
            print(f"Retrying {', '.join(retry_datasets)} with seed {retry_seed}")
            retry_scores = run_evalscope(
                eval_config,
                remote_server.url_for("v1"),
                tmp_path / f"trial-{trial + 1}",
                datasets=retry_datasets,
                seed=retry_seed,
            )
            for dataset, score in retry_scores.items():
                score_history[dataset].append(score)

    for dataset, metric_config in eval_config["datasets"].items():
        trial_scores = score_history[dataset]
        score = fmean(trial_scores)
        threshold = metric_config["metric_threshold"]
        tolerance = metric_config.get("tolerance", 0.0)
        minimum_score = threshold - tolerance

        print(
            f"{dataset}: measured={score:.4f}, expected={threshold:.4f}, "
            f"tolerance={tolerance:.4f}, trials={trial_scores}"
        )
        assert score >= minimum_score, (
            f"{dataset} score too low: {score:.4f} < {threshold:.4f} - "
            f"{tolerance:.4f} = {minimum_score:.4f}"
        )


def test_qwen4_exp_accuracy_retries_failed_dataset(monkeypatch, tmp_path: Path):
    eval_config = {
        "model_name": "test-model",
        "datasets": {
            "gsm8k": {"metric_threshold": 0.96},
            "aime25": {
                "metric_threshold": 0.90,
                "tolerance": 0.05,
                "max_score_trials": 2,
            },
        },
        "generation_config": {"seed": 1236},
    }
    config_filename = tmp_path / "config.yaml"
    config_filename.write_text(yaml.safe_dump(eval_config), encoding="utf-8")

    eval_results = iter(
        [
            {"gsm8k": 0.98, "aime25": 25 / 30},
            {"aime25": 26 / 30},
        ]
    )
    calls = []

    def fake_run_evalscope(eval_config, base_url, work_dir, datasets=None, seed=None):
        calls.append((datasets, seed, work_dir))
        return next(eval_results)

    class FakeRemoteOpenAIServer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def url_for(self, path):
            return f"http://test/{path}"

    test_module = sys.modules[__name__]
    monkeypatch.setattr(test_module, "RemoteOpenAIServer", FakeRemoteOpenAIServer)
    monkeypatch.setattr(test_module, "run_evalscope", fake_run_evalscope)

    test_qwen4_exp_accuracy(config_filename, tmp_path)

    assert calls == [
        (None, None, tmp_path),
        (["aime25"], 1237, tmp_path / "trial-2"),
    ]
