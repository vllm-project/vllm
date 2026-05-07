# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MRCR long-context accuracy test.

Usage:
    pytest -s -v tests/evals/mrcr/test_mrcr_correctness.py \
        --config-list-file=configs/models-small.txt
"""

import shlex

import yaml

from tests.utils import RemoteOpenAIServer

from .mrcr_eval import evaluate_mrcr


def _split_host_port(url: str, default_port: int = 8000) -> tuple[str, int]:
    if "://" in url:
        url = url.split("://", 1)[1]
    host_port = url.split("/", 1)[0]
    if ":" in host_port:
        host, p = host_port.split(":", 1)
        return f"http://{host}", int(p)
    return f"http://{host_port}", default_port


def test_mrcr_correctness(config_filename):
    cfg = yaml.safe_load(config_filename.read_text(encoding="utf-8"))

    server_args = shlex.split(cfg.get("server_args", ""))
    server_args += ["--trust-remote-code", "--disable-uvicorn-access-log"]

    print(
        f"MRCR eval for {cfg['model_name']} (threshold {cfg['match_ratio_threshold']})"
    )

    with RemoteOpenAIServer(
        cfg["model_name"],
        server_args,
        env_dict=cfg.get("env"),
        max_wait_seconds=cfg.get("startup_max_wait_seconds", 600),
    ) as server:
        host, port = _split_host_port(server.url_for("v1"))
        results = evaluate_mrcr(
            model_name=cfg.get("model_name"),
            num_samples=cfg.get("num_samples", 40),
            needles=cfg.get("needles", [2, 4, 8]),
            max_prompt_tokens=cfg.get("max_prompt_tokens"),
            max_tokens=cfg.get("max_tokens", 2048),
            host=host,
            port=port,
            concurrency=cfg.get("concurrency", 8),
            extra_body=cfg.get("extra_body"),
        )

    threshold = cfg["match_ratio_threshold"]
    tol = cfg.get("tolerance", 0.05)

    print(f"  match_ratio:     {results['match_ratio']:.4f}")
    print(f"  prefix_hit_rate: {results['prefix_hit_rate']:.4f}")
    for k, v in results["per_needle"].items():
        print(f"  {k}: {v:.4f}")

    failures: list[str] = []
    if isinstance(threshold, dict):
        for n, expected in threshold.items():
            key = f"match_ratio_n{int(n)}"
            measured = results["per_needle"].get(key)
            if measured is None:
                failures.append(f"{key}: no samples collected")
            elif measured < expected - tol:
                failures.append(f"{key}: {measured:.4f} < {expected:.4f} - {tol:.4f}")
    else:
        measured = results["match_ratio"]
        if measured < threshold - tol:
            failures.append(
                f"match_ratio: {measured:.4f} < {threshold:.4f} - {tol:.4f}"
            )

    assert not failures, "MRCR thresholds failed: " + "; ".join(failures)
