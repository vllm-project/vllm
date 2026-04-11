# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified accuracy + performance evaluation runner for vLLM.

Starts a managed vLLM server (or connects to an existing one), runs
lm_eval for accuracy metrics, polls /metrics for performance data, and
appends a single structured JSON record to a JSONL output file.
"""

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import regex as re
import requests
from lm_eval import simple_evaluate

from vllm.eval.metrics_collector import VLLMMetricsCollector


class BenchmarkServerManager:
    """Manages a vllm serve subprocess for the duration of an eval run.

    Handles port selection, process-group isolation, health-check polling,
    and graceful shutdown (SIGTERM -> SIGKILL -> orphan cleanup).

    Usage::

        with BenchmarkServerManager(model, vllm_args) as mgr:
            base = mgr.url_root  # "http://127.0.0.1:<port>"
            v1 = mgr.url_for("v1")
    """

    def __init__(
        self,
        model: str,
        vllm_serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
        max_wait_seconds: float = 600.0,
        host: str = "127.0.0.1",
    ) -> None:
        self.model = model
        self.host = host
        self.port = _get_open_port()
        self.max_wait_seconds = max_wait_seconds

        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if env_dict:
            env.update(env_dict)

        cmd = [
            "vllm",
            "serve",
            model,
            "--host",
            host,
            "--port",
            str(self.port),
            *vllm_serve_args,
        ]
        print(f"[BenchmarkServerManager] Starting: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )

    def __enter__(self) -> "BenchmarkServerManager":
        self._wait_for_server(timeout=self.max_wait_seconds)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pid = self._proc.pid
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, OSError):
            pgid = None

        if pgid is not None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(pgid, signal.SIGTERM)
                print(f"[BenchmarkServerManager] SIGTERM -> pgid {pgid}")
        else:
            self._proc.terminate()

        try:
            self._proc.wait(timeout=15)
            print(f"[BenchmarkServerManager] Server pid={pid} stopped gracefully")
        except subprocess.TimeoutExpired:
            print("[BenchmarkServerManager] SIGTERM timed out, sending SIGKILL")
            if pgid is not None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(pgid, signal.SIGKILL)
            else:
                self._proc.kill()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _kill_orphaned_children(pid)

    @property
    def url_root(self) -> str:
        return f"http://{self.host}:{self.port}"

    def url_for(self, *parts: str) -> str:
        return self.url_root + "/" + "/".join(parts)

    def _wait_for_server(self, timeout: float) -> None:
        health_url = self.url_for("health")
        start = time.time()
        while True:
            try:
                if requests.get(health_url, timeout=5).status_code == 200:
                    print(f"[BenchmarkServerManager] Server ready at {self.url_root}")
                    return
            except Exception:
                pass
            rc = self._proc.poll()
            if rc is not None and rc != 0:
                raise RuntimeError(
                    f"[BenchmarkServerManager] Server exited with code {rc} "
                    "before becoming healthy."
                )
            if time.time() - start > timeout:
                raise RuntimeError(
                    f"[BenchmarkServerManager] Server did not become healthy "
                    f"within {timeout}s."
                )
            time.sleep(1.0)


def _get_open_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _kill_orphaned_children(parent_pid: int) -> None:
    """Best-effort cleanup of any lingering child processes."""
    try:
        import psutil

        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for child in children:
            print(
                f"[BenchmarkServerManager] Killing orphaned child "
                f"pid={child.pid} name={child.name()}"
            )
            child.kill()
        psutil.wait_procs(children, timeout=5)
    except Exception as exc:
        print(f"[BenchmarkServerManager] Orphan cleanup failed: {exc}")
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(parent_pid, signal.SIGKILL)


def _collect_environment() -> dict:
    """Collect environment info via vllm.collect_env and return a structured dict."""
    try:
        from vllm.collect_env import get_env_info

        env = get_env_info()
    except Exception as exc:
        return {"error": f"Could not collect environment info: {exc}"}

    def _parse_lines(text: str | None) -> list[str]:
        if not text:
            return []
        return [line.strip() for line in text.strip().splitlines() if line.strip()]

    def _parse_env_vars(text: str | None) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in _parse_lines(text):
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
        return result

    def _parse_cpu_info(text: str | None) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in _parse_lines(text):
            key, sep, val = line.partition(":")
            if sep:
                result[key.strip()] = val.strip()
        return result

    def _parse_build_flags(text: str | None) -> dict[str, str]:
        result: dict[str, str] = {}
        if not text:
            return result
        for segment in text.split(";"):
            key, sep, val = segment.strip().partition(":")
            if sep:
                result[key.strip()] = val.strip()
        return result

    return {
        "system_info": {
            "os": env.os,
            "gcc_version": env.gcc_version,
            "clang_version": env.clang_version,
            "cmake_version": env.cmake_version,
            "libc_version": env.libc_version,
        },
        "pytorch_info": {
            "torch_version": env.torch_version,
            "is_debug_build": env.is_debug_build,
            "cuda_compiled_version": env.cuda_compiled_version,
            "hip_compiled_version": env.hip_compiled_version,
        },
        "python_env": {
            "python_version": env.python_version,
            "python_platform": env.python_platform,
        },
        "cuda_gpu_info": {
            "is_cuda_available": env.is_cuda_available,
            "cuda_runtime_version": env.cuda_runtime_version,
            "cuda_module_loading": env.cuda_module_loading,
            "nvidia_gpu_models": env.nvidia_gpu_models,
            "nvidia_driver_version": env.nvidia_driver_version,
            "cudnn_version": env.cudnn_version,
            "hip_runtime_version": env.hip_runtime_version,
            "miopen_runtime_version": env.miopen_runtime_version,
            "is_xnnpack_available": env.is_xnnpack_available,
            "caching_allocator_config": env.caching_allocator_config,
        },
        "cpu_info": _parse_cpu_info(env.cpu_info),
        "pip_packages": _parse_lines(env.pip_packages),
        "conda_packages": _parse_lines(env.conda_packages),
        "vllm_info": {
            "rocm_version": env.rocm_version,
            "vllm_version": env.vllm_version,
            "vllm_build_flags": _parse_build_flags(env.vllm_build_flags),
            "gpu_topo": _parse_gpu_topo(env.gpu_topo),
        },
        "env_vars": _parse_env_vars(env.env_vars),
    }


def _sanitize_model_name(model: str) -> str:
    """Return a filesystem-safe stem from a model ID.

    Example: "meta-llama/Meta-Llama-3-8B" -> "meta-llama__Meta-Llama-3-8B"
    """
    name = model.replace("/", "__")
    name = re.sub(r'[<>:"|?*\\\s]', "_", name)
    return name


def _parse_gpu_topo(text: str | None) -> dict:
    """Parse ROCm SMI GPU topology report into structured matrices.

    Returns a dict with keys: weight_matrix, hops_matrix,
    link_type_matrix, numa_nodes. Empty dict if text is None.
    """
    if not text:
        return {}

    SECTION_MAP = {
        "Weight": "weight_matrix",
        "Hops": "hops_matrix",
        "Link Type": "link_type_matrix",
        "Numa": "numa_nodes",
    }

    result: dict = {}
    current_section: str | None = None
    col_headers: list[str] = []
    current_matrix: dict = {}

    def _flush() -> None:
        nonlocal current_matrix, col_headers
        if current_section and current_matrix:
            result[current_section] = current_matrix
        current_matrix = {}
        col_headers = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if (
            len(stripped) > 10
            and stripped.startswith("=====")
            and stripped.endswith("=====")
        ):
            title = re.sub(r"=+", "", stripped).strip()
            new_section = next((v for k, v in SECTION_MAP.items() if k in title), None)
            if new_section != current_section:
                _flush()
                current_section = new_section
            continue

        if not stripped or current_section is None:
            continue

        if current_section in ("weight_matrix", "hops_matrix", "link_type_matrix"):
            if raw_line[0] in (" ", "\t") and "GPU" in stripped:
                col_headers = stripped.split()
            elif re.match(r"^GPU\d+\s", raw_line):
                parts = stripped.split()
                if len(parts) < 2 or not col_headers:
                    continue
                row_gpu = parts[0]
                values = parts[1 : len(col_headers) + 1]
                if len(values) != len(col_headers):
                    continue
                row: dict = {}
                for col, val in zip(col_headers, values):
                    if current_section in ("weight_matrix", "hops_matrix"):
                        try:
                            row[col] = int(val)
                        except ValueError:
                            row[col] = val
                    else:
                        row[col] = val
                current_matrix[row_gpu] = row

        elif current_section == "numa_nodes":
            m = re.match(r"GPU\[(\d+)\]\s*:\s*\(Topology\)\s*(.+?):\s*(.+)", stripped)
            if m:
                gpu_id = f"GPU{m.group(1)}"
                prop = m.group(2).strip().lower().replace(" ", "_")
                val_str = m.group(3).strip()
                try:
                    node_val: int | str = int(val_str)
                except ValueError:
                    node_val = val_str
                current_matrix.setdefault(gpu_id, {})[prop] = node_val

    _flush()
    return result


def _run_lm_eval(
    v1_url: str,
    model: str,
    tasks: str,
    num_concurrent: int = 1,
    num_fewshot: int | None = None,
    limit: float | None = None,
) -> dict:
    """Run lm_eval via its Python API and return the results dict."""
    model_args = (
        f"model={model},"
        f"base_url={v1_url}/completions,"
        f"tokenized_requests=False,"
        f"num_concurrent={num_concurrent}"
    )
    print(
        f"[UnifiedEvalRunner] lm_eval simple_evaluate: "
        f"tasks={tasks}, num_concurrent={num_concurrent}, limit={limit}"
    )
    return simple_evaluate(
        model="local-completions",
        model_args=model_args,
        tasks=tasks.split(","),
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=False,
    )


class UnifiedEvalRunner:
    """Orchestrates accuracy (lm_eval) and performance (/metrics) collection
    from a single vLLM workload. Results are appended as JSONL."""

    def __init__(
        self,
        model: str,
        tasks: str,
        vllm_args: list[str] | None = None,
        num_fewshot: int | None = None,
        limit: float | None = None,
        poll_interval: float = 2.0,
        output_jsonl: str = "./eval_results/results.jsonl",
        max_wait_seconds: float = 600.0,
        num_concurrent_requests: int = 1,
        run_config: dict | None = None,
    ) -> None:
        self.model = model
        self.tasks = tasks
        self.vllm_args = vllm_args or []
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.poll_interval = poll_interval
        self.output_jsonl = Path(output_jsonl)
        self.max_wait_seconds = max_wait_seconds
        self.run_config = run_config or {}
        self.num_concurrent_requests = num_concurrent_requests

    def run(self) -> dict:
        """Start a vLLM server, run eval, collect metrics, shut down."""
        print("[UnifiedEvalRunner] Collecting environment info...")
        environment = _collect_environment()

        with BenchmarkServerManager(
            self.model,
            self.vllm_args,
            max_wait_seconds=self.max_wait_seconds,
        ) as server:
            return self._run_with_server(server.url_for("v1"), environment)

    def run_against_existing_server(self, base_url: str) -> dict:
        """Run against an already-running vLLM server."""
        print("[UnifiedEvalRunner] Collecting environment info...")
        environment = _collect_environment()
        v1_url = base_url.rstrip("/") + "/v1"
        return self._run_with_server(v1_url, environment)

    def _run_with_server(self, v1_url: str, environment: dict) -> dict:
        collector = VLLMMetricsCollector(
            v1_url.replace("/v1", ""),
            poll_interval=self.poll_interval,
        )
        collector.start()
        print(
            f"[UnifiedEvalRunner] Metrics collector started "
            f"(polling every {self.poll_interval}s)"
        )

        time.sleep(self.poll_interval + 0.5)

        try:
            eval_results = _run_lm_eval(
                v1_url,
                self.model,
                self.tasks,
                num_concurrent=self.num_concurrent_requests,
                num_fewshot=self.num_fewshot,
                limit=self.limit,
            )
        finally:
            collector.stop()

        performance = collector.get_summary()
        record = self._build_record(eval_results, performance, environment)
        self._append_jsonl(record)
        self._print_summary(record)
        return record

    def _build_record(
        self,
        eval_results: dict,
        performance: dict,
        environment: dict,
    ) -> dict:
        return {
            "metadata": {
                "run_id": str(uuid.uuid4()),
                "model": self.model,
                "tasks": self.tasks,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "config": self.run_config,
            "accuracy": eval_results.get("results", eval_results),
            "performance": performance,
            "environment": environment,
        }

    def _append_jsonl(self, record: dict) -> None:
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n[UnifiedEvalRunner] Result appended to: {self.output_jsonl}")

    def _print_summary(self, record: dict) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print("UNIFIED EVALUATION REPORT")
        print(sep)
        meta = record["metadata"]
        print(f"Run ID   : {meta['run_id']}")
        print(f"Model    : {meta['model']}")
        print(f"Tasks    : {meta['tasks']}")
        print(f"Time     : {meta['timestamp']}")

        cfg = record.get("config", {})
        if cfg:
            print("\n--- Config ---")
            for k, v in cfg.items():
                if v is not None:
                    print(f"  {k}: {v}")

        print("\n--- Accuracy ---")
        accuracy = record["accuracy"]
        if "raw_stdout" in accuracy:
            print("  (raw lm_eval output captured; table parse produced no rows)")
        elif isinstance(accuracy, dict):
            for task_name, metrics in accuracy.items():
                if isinstance(metrics, dict):
                    print(f"  {task_name}:")
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            print(f"    {k}: {v}")

        print("\n--- Performance ---")
        for k, v in record["performance"].items():
            if k == "error":
                print(f"  WARNING: {v}")
            else:
                print(f"  {k}: {v}")

        env_info = record["environment"]
        if "error" not in env_info:
            vllm = env_info.get("vllm_info", {})
            sys_info = env_info.get("system_info", {})
            print("\n--- Environment (summary) ---")
            print(f"  vLLM version : {vllm.get('vllm_version', 'N/A')}")
            print(f"  ROCm version : {vllm.get('rocm_version', 'N/A')}")
            print(f"  OS           : {sys_info.get('os', 'N/A')}")
        print(sep)


@dataclass
class EvalConfig:
    """Config for the optional accuracy eval pass in `vllm bench serve`."""

    backend: str = "lm_eval"
    """Which eval harness to use. One of `lm_eval` or `gpt_oss`."""

    tasks: str | None = None
    """Comma-separated task names. e.g. `gsm8k,hellaswag` or `gpqa,aime25`."""

    limit: float | None = None
    """Cap samples per task. Default None means full dataset. See also `num_samples`."""

    num_samples: float | None = None
    """Friendly alias for `limit`. If both are set, `num_samples` wins."""

    num_fewshot: int | None = None
    """Few-shot examples (lm_eval only). Default None uses the task default."""

    max_tokens: int | None = None
    """Cap the per-request `max_tokens` sent to the server."""

    reasoning_effort: str | None = None
    """gpt_oss reasoning effort: `low`, `medium`, or `high`."""

    output: str | None = None
    """JSONL output path. Defaults to eval_results/<model>.jsonl."""


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Register `--eval` as a JSON/dot-notation argument."""
    g = parser.add_argument_group("accuracy eval (optional, requires --eval)")
    g.add_argument(
        "--eval",
        type=json.loads,
        default=None,
        help="Eval configuration as JSON or via dot-notation. "
        "Examples: `--eval.tasks gsm8k --eval.num_samples 1024` or "
        "`--eval.backend gpt_oss --eval.tasks gpqa --eval.reasoning_effort low`. "
        "Accepted keys: backend (lm_eval|gpt_oss), tasks, num_samples (or limit), "
        "num_fewshot, max_tokens, reasoning_effort, output.",
    )


def score_lm_eval_offline(
    instances: list[Any],
    task_dict: dict,
    generated_texts: list[str],
) -> dict:
    """Score lm_eval instances against captured responses without HTTP.

    Populates each instance's resps, applies task filters, and aggregates
    per-task accuracy metrics. Returns {task_name: {metric: value, ...}}.
    """
    import collections as _collections
    import math as _math

    # lm_eval resps: list of responses for each generation attempt.
    # With a single generation, each instance gets [text].
    for instance, text in zip(instances, generated_texts):
        instance.resps = [text]

    results: dict = {}
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, tuple):
            _, task_obj = task_obj
        task_obj.apply_filters()
        eval_docs = task_obj.eval_docs
        agg = task_obj.aggregation()
        results[task_name] = {}

        # Each filter must be aggregated separately to avoid mixing scores
        # across different filter strategies (e.g. strict vs flexible).
        filter_keys = (
            list(task_obj.instances[0].filtered_resps) if task_obj.instances else []
        )
        for filter_key in filter_keys:
            filter_vals: dict = _collections.defaultdict(list)
            for instance in task_obj.instances:
                if instance.request_type != "generate_until":
                    continue
                doc = eval_docs[instance.doc_id]
                per_doc = task_obj.process_results(
                    doc, [instance.filtered_resps[filter_key]]
                )
                for metric, val in per_doc.items():
                    filter_vals[metric].append(val)
            for metric in filter_vals:
                if metric in agg:
                    vals = filter_vals[metric]
                    agg_val = agg[metric](vals)
                    results[task_name][f"{metric},{filter_key}"] = agg_val
                    n = len(vals)
                    if n > 1:
                        variance = sum((v - agg_val) ** 2 for v in vals) / (n - 1)
                        stderr = _math.sqrt(variance / n)
                    else:
                        stderr = float("nan")
                    results[task_name][f"{metric}_stderr,{filter_key}"] = stderr

    return results


def score_gpt_oss_offline(
    task_evals: dict,
    row_index: list[tuple[str, int]],
    generated_texts: list[str],
) -> dict:
    """Score gpt_oss eval rows against captured chat completions.

    The `sampler(...)` step inside each gpt_oss `Eval.__call__` is
    replaced with reading the pre-captured `generated_text`; the
    per-task results are then aggregated via gpt_oss's own
    `aggregate_results` so the metrics match `python -m gpt_oss.evals`.
    """
    from gpt_oss.evals import report
    from gpt_oss.evals.types import SingleEvalResult

    per_task: dict[str, list[tuple[int, str]]] = {name: [] for name in task_evals}
    for (task_name, row_idx), text in zip(row_index, generated_texts):
        per_task.setdefault(task_name, []).append((row_idx, text))

    results: dict = {}
    for task_name, eval_obj in task_evals.items():
        per_row = per_task.get(task_name, [])
        single_results: list[SingleEvalResult] = []
        for row_idx, response_text in per_row:
            row = eval_obj.examples[row_idx]
            single_results.append(_score_gpt_oss_row(task_name, row, response_text))
        agg = report.aggregate_results(single_results)
        task_metrics = dict(agg.metrics or {})
        if agg.score is not None:
            task_metrics["score"] = float(agg.score)
        results[task_name] = task_metrics
    return results


def _score_gpt_oss_row(task_name: str, row: dict, response_text: str):
    # Modified version of `__call__` in each upstream eval class:
    # `GPQAEval`, `AIME25Eval`, `BasicEval` from `gpt_oss.evals`.
    from gpt_oss.evals.types import SingleEvalResult

    if task_name == "gpqa":
        from gpt_oss.evals.abcd_grader import extract_abcd

        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in row["permutation"]]
        correct_index = choices.index(row["Correct Answer"])
        correct_answer = "ABCD"[correct_index]
        extracted = extract_abcd(response_text)
        score = 1.0 if extracted == correct_answer else 0.0
        return SingleEvalResult(
            html=None,
            score=score,
            convo=None,
            metrics={"chars": len(response_text)},
        )

    if task_name == "aime25":
        from gpt_oss.evals.aime_eval import extract_boxed_text

        extracted = extract_boxed_text(response_text)
        try:
            extracted_int = int(extracted)
        except (ValueError, TypeError):
            extracted_int = None
        correct_int = int(row["answer"])
        score = 1.0 if extracted_int == correct_int else 0.0
        return SingleEvalResult(
            html=None,
            score=score,
            convo=None,
            metrics={"chars": len(response_text)},
        )

    if task_name == "basic":
        score = 1.0 if len(response_text) > 0 else 0.0
        return SingleEvalResult(
            html=None,
            score=score,
            convo=None,
            metrics={"chars": len(response_text)},
        )

    raise ValueError(f"Unknown gpt_oss task '{task_name}'")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register all vllm eval arguments on parser."""
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        type=str,
        help="Comma-separated lm_eval task names, e.g. hellaswag,gsm8k.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        metavar="N",
        help="Tensor-parallel size (default: 1).",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        metavar="N",
        help="Pipeline-parallel size (default: 1).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        metavar="N",
        help="Maximum sequence length override.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model weight dtype (default: auto).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        metavar="FRAC",
        help="GPU memory utilisation fraction (default: 0.9).",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=600.0,
        metavar="SECS",
        help="Maximum seconds to wait for the server to become healthy (default: 600).",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        metavar="N",
        help="Number of few-shot examples (lm_eval --num_fewshot).",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        metavar="N",
        help="Limit samples per task, useful for quick smoke tests.",
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=int,
        default=1,
        dest="num_concurrent_requests",
        metavar="N",
        help="Number of concurrent requests lm_eval sends to the server (default: 1).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        metavar="SECS",
        help="Seconds between /metrics scrapes (default: 2.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="JSONL output path. Results are appended. "
        "Defaults to ./eval_results/<model>.jsonl.",
    )
    parser.add_argument(
        "--existing-server",
        type=str,
        default=None,
        metavar="URL",
        help="Connect to a running vLLM server instead of starting one, "
        "e.g. http://localhost:8000",
    )


def main(args: argparse.Namespace) -> None:
    """Entry point for the vllm eval CLI subcommand."""
    vllm_args = [
        "--tensor-parallel-size",
        str(args.tp),
        "--pipeline-parallel-size",
        str(args.pp),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.max_model_len:
        vllm_args += ["--max-model-len", str(args.max_model_len)]

    output_jsonl = args.output or str(
        Path("./eval_results") / f"{_sanitize_model_name(args.model)}.jsonl"
    )

    run_config = {
        "num_concurrent_requests": args.num_concurrent_requests,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "tensor_parallel_size": args.tp,
        "pipeline_parallel_size": args.pp,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }

    runner = UnifiedEvalRunner(
        model=args.model,
        tasks=args.tasks,
        vllm_args=vllm_args,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        poll_interval=args.poll_interval,
        output_jsonl=output_jsonl,
        max_wait_seconds=args.max_wait_seconds,
        num_concurrent_requests=args.num_concurrent_requests,
        run_config=run_config,
    )

    if args.existing_server:
        runner.run_against_existing_server(args.existing_server)
    else:
        runner.run()


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Unified accuracy + performance evaluation for vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_cli_args(_parser)
    main(_parser.parse_args())
