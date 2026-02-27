#!/usr/bin/env python3
"""Robust experiment runner utility for Phase 2 tool-calling experiments.

Provides:
    - RobustServer: vLLM server with health monitoring and auto-restart
    - run_experiment: batch runner with health checks and incremental saves
    - load_samples: sample loader from self-consistency DB
    - SYSTEM_PROMPT: shared system prompt for all experiments

All Phase 2 experiment scripts import from this module.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

import requests  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are analyzing a driving scene image. "
    "The image is 504x336 pixels. "
    "When specifying pixel coordinates, keep x in range 0-503 and y in range 0-335. "
    "The waypoint grid is 63x63 (row 0=top, col 0=left)."
)


# ---------------------------------------------------------------------------
# RobustServer
# ---------------------------------------------------------------------------
class RobustServer:
    """VLLMServer with built-in health monitoring and auto-restart."""

    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        port: int,
        max_model_len: int = 8192,
    ) -> None:
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.port = port
        self.max_model_len = max_model_len
        self.proc: subprocess.Popen | None = None
        self.url = f"http://localhost:{port}"

    def _warm_fsx_cache(self) -> None:
        """Pre-read safetensor files to warm FSx page cache. Cuts load from 15min to <2min."""
        import glob
        safetensors = glob.glob(os.path.join(self.model_path, "*.safetensors"))
        if not safetensors:
            return
        print(f"  Warming FSx cache for {len(safetensors)} shards...")
        for sf in sorted(safetensors):
            size_gb = os.path.getsize(sf) / (1024**3)
            print(f"    Reading {os.path.basename(sf)} ({size_gb:.1f}G)...")
            with open(sf, "rb") as f:
                while f.read(1024 * 1024):  # 1MB chunks
                    pass
        print(f"  FSx cache warmed.")

    def start(self, timeout: int = 300) -> None:
        """Start server, wait for healthy. Uses file lock to prevent simultaneous FSx reads."""
        import fcntl
        lock_path = "/tmp/vllm_server_start.lock"
        print(f"  Acquiring FSx load lock...")
        lock_fd = open(lock_path, "w")  # noqa: SIM115
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        print(f"  Lock acquired.")
        self._lock_fd = lock_fd
        # Pre-warm FSx cache before starting server
        self._warm_fsx_cache()
        print(f"  Starting server...")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # Remove /workspace/vllm from PYTHONPATH so installed vllm is used
        pp = env.get("PYTHONPATH", "")
        parts = [p for p in pp.split(":") if p and "/workspace/vllm" not in p]
        env["PYTHONPATH"] = ":".join(parts)

        vllm_bin = "/home/mketkar/.local/bin/vllm"
        cmd = [
            vllm_bin, "serve", self.model_path,
            "--trust-remote-code",
            "--max-model-len", str(self.max_model_len),
            "--enforce-eager",
            "--port", str(self.port),
            "--gpu-memory-utilization", "0.8",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ]

        log_path = f"/tmp/vllm_robust_{self.port}.log"
        print(f"  Starting server: model={os.path.basename(self.model_path)}, "
              f"GPU={self.gpu_id}, port={self.port}")
        print(f"  Log: {log_path}")
        log_file = open(log_path, "w")  # noqa: SIM115
        self.proc = subprocess.Popen(
            cmd, env=env, cwd="/tmp",
            stdout=log_file, stderr=subprocess.STDOUT,
        )

        for i in range(timeout):
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"Server died during startup (exit {self.proc.returncode}). "
                    f"Check log: {log_path}"
                )
            try:
                r = requests.get(f"{self.url}/health", timeout=2)
                if r.status_code == 200:
                    print(f"  Server ready on port {self.port} "
                          f"(GPU {self.gpu_id}) in {i}s")
                    # Release FSx lock so next server can start loading
                    if hasattr(self, '_lock_fd') and self._lock_fd:
                        import fcntl
                        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                        self._lock_fd.close()
                        self._lock_fd = None
                        print(f"  FSx load lock released.")
                    return
            except Exception:
                pass
            time.sleep(1)

        # Release lock on failure too
        if hasattr(self, '_lock_fd') and self._lock_fd:
            import fcntl
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None
        raise RuntimeError(f"Server timeout after {timeout}s. Check log: {log_path}")

    def is_healthy(self) -> bool:
        """Check if server responds to /health."""
        try:
            return requests.get(f"{self.url}/health", timeout=5).status_code == 200
        except Exception:
            return False

    def ensure_healthy(self) -> None:
        """Check health, auto-restart if dead."""
        if not self.is_healthy():
            print(f"  WARNING: Server on port {self.port} is dead. Restarting...")
            self.stop()
            time.sleep(5)
            self.start()

    def stop(self) -> None:
        """Terminate the server process."""
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0,
        max_tokens: int = 1024,
        tool_choice: str = "auto",
    ) -> dict[str, Any] | None:
        """Single chat request with retry on failure.

        Returns the message dict from the first choice, or None on failure.
        """
        payload: dict[str, Any] = {
            "model": self.model_path,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.url}/v1/chat/completions",
                    json=payload,
                    timeout=120,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]
                elif resp.status_code == 400:
                    print(f"  400 error: {resp.text[:200]}")
                    return None  # Don't retry 400s
                else:
                    print(f"  HTTP {resp.status_code}, retrying...")
            except requests.Timeout:
                print(f"  Timeout on attempt {attempt + 1}, retrying...")
            except Exception as e:
                print(f"  Error: {e}, checking health...")
                self.ensure_healthy()

        return None  # All retries failed


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    task_id: str,
    server: RobustServer,
    samples: list[dict[str, Any]],
    prompt_fn: Any,
    tools: dict[str, Any],
    tool_defs: list[dict[str, Any]],
    tool_choice: str = "auto",
    max_rounds: int = 5,
    save_path: str | None = None,
) -> list[dict[str, Any]]:
    """Run an experiment task with health monitoring.

    Args:
        task_id: string identifier for this task
        server: RobustServer instance
        samples: list of sample dicts (must have sample_id, image_path, etc.)
        prompt_fn: function(sample) -> (system_prompt, user_prompt)
        tools: dict of name -> callable
        tool_defs: list of OpenAI tool definitions
        tool_choice: "auto" or "required"
        max_rounds: max tool call rounds
        save_path: where to save results JSON

    Returns:
        list of result dicts
    """
    from orchestrator import ToolCallingOrchestrator  # type: ignore[import-not-found]

    results: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        # Health check every 10 samples
        if i % 10 == 0 and i > 0:
            server.ensure_healthy()
            n_changed = sum(1 for r in results if r.get("changed_mind"))
            print(f"  [{task_id}] {i}/{len(samples)} done, "
                  f"{n_changed} revised")

        sys_prompt, user_prompt = prompt_fn(sample)

        orch = ToolCallingOrchestrator(
            server_url=server.url,
            tools=tools,
            tool_definitions=tool_defs,
            max_tool_rounds=max_rounds,
            temperature=0,
            max_tokens=1024,
        )

        try:
            result = orch.run(
                sample.get("image_path"),
                sys_prompt,
                user_prompt,
                tool_choice=tool_choice,
            )
            result["sample_id"] = sample.get("sample_id")
            result["gt_scene"] = sample.get("scene_type_gt")
            result["original_pred"] = sample.get("predicted_scene")
            results.append(result)
        except Exception as e:
            print(f"  [{task_id}] Sample {i} failed: {e}")
            results.append({
                "sample_id": sample.get("sample_id"),
                "error": str(e),
            })

        # Save incrementally every 25 samples
        if save_path and (i + 1) % 25 == 0:
            _save(save_path, task_id, results)

    # Final save
    if save_path:
        _save(save_path, task_id, results)

    print(f"  [{task_id}] COMPLETE: {len(results)} samples")
    return results


def _save(path: str, task_id: str, results: list[dict[str, Any]]) -> None:
    """Save results, merging with existing file."""
    existing: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing[task_id] = {
        "n_samples": len(results),
        "n_errors": sum(1 for r in results if "error" in r),
        "results": results,
    }

    with open(path, "w") as f:
        json.dump(existing, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Sample loader
# ---------------------------------------------------------------------------
def load_samples(
    db_path: str,
    category: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Load sample predictions from self-consistency DB.

    Categories:
        false_iz: predicted incident_zone but GT is nominal
        true_iz: predicted incident_zone and GT is incident_zone
        correct: predicted scene matches GT
        other_errors: predicted != GT, not false IZ
        mixed: 50 wrong + 50 correct
        all: all samples
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if category == "false_iz":
        rows = conn.execute(
            "SELECT * FROM predictions "
            "WHERE predicted_scene='incident_zone' AND scene_type_gt='nominal' "
            "LIMIT ?",
            (limit,),
        ).fetchall()
    elif category == "true_iz":
        rows = conn.execute(
            "SELECT * FROM predictions "
            "WHERE predicted_scene='incident_zone' AND scene_type_gt='incident_zone' "
            "LIMIT ?",
            (limit,),
        ).fetchall()
    elif category == "correct":
        rows = conn.execute(
            "SELECT * FROM predictions "
            "WHERE predicted_scene=scene_type_gt "
            "LIMIT ?",
            (limit,),
        ).fetchall()
    elif category == "other_errors":
        rows = conn.execute(
            "SELECT * FROM predictions "
            "WHERE predicted_scene!=scene_type_gt "
            "AND predicted_scene!='incident_zone' "
            "LIMIT ?",
            (limit,),
        ).fetchall()
    elif category == "mixed":
        wrong = conn.execute(
            "SELECT * FROM predictions "
            "WHERE predicted_scene!=scene_type_gt LIMIT ?",
            (limit // 2,),
        ).fetchall()
        correct = conn.execute(
            "SELECT * FROM predictions "
            "WHERE predicted_scene=scene_type_gt LIMIT ?",
            (limit // 2,),
        ).fetchall()
        rows = list(wrong) + list(correct)
    else:
        rows = conn.execute(
            "SELECT * FROM predictions LIMIT ?",
            (limit,),
        ).fetchall()

    conn.close()

    # Convert to dicts and load images
    from visual_tools import load_sample_image  # type: ignore[import-not-found]

    samples: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        try:
            d["image_path"] = load_sample_image(d.get("sample_id", 0))
            samples.append(d)
        except Exception as e:
            print(f"  Failed to load image for sample {d.get('sample_id')}: {e}")

    return samples
