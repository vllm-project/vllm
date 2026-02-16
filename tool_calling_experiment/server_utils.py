#!/usr/bin/env python3
"""Utilities for managing vLLM server and making batched requests.

The vLLM LLM() Python API does not work in this environment due to a
V1 engine subprocess CUDA initialisation failure.  Instead, we start
``vllm serve`` as a subprocess, wait for it to become healthy, and
talk to it via the OpenAI-compatible HTTP API.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests  # type: ignore[import-not-found]


class VLLMServer:
    """Manage a vLLM server process."""

    def __init__(
        self,
        model_path: str,
        port: int = 8199,
        gpu_id: int = 0,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.8,
        enable_tools: bool = True,
    ) -> None:
        self.model_path = model_path
        self.port = port
        self.gpu_id = gpu_id
        self.max_model_len = max_model_len
        self.gpu_mem = gpu_memory_utilization
        self.enable_tools = enable_tools
        self.proc: subprocess.Popen[bytes] | None = None
        self.base_url = f"http://localhost:{port}"

    # ----------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------

    def start(self, timeout: int = 360) -> None:
        """Start vLLM server and block until the /health endpoint responds."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # Remove /workspace/vllm from PYTHONPATH so the installed
        # vllm package is used instead of the source tree.
        pp = env.get("PYTHONPATH", "")
        parts = [
            p for p in pp.split(":") if p and "/workspace/vllm" not in p
        ]
        env["PYTHONPATH"] = ":".join(parts)

        vllm_bin = "/home/mketkar/.local/bin/vllm"
        cmd = [
            vllm_bin,
            "serve",
            self.model_path,
            "--trust-remote-code",
            "--max-model-len",
            str(self.max_model_len),
            "--enforce-eager",
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(self.gpu_mem),
        ]
        if self.enable_tools:
            cmd.extend(
                [
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "hermes",
                ]
            )

        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd="/tmp",
        )

        # Poll /health until ready
        for elapsed in range(timeout):
            try:
                r = requests.get(
                    f"{self.base_url}/health", timeout=2
                )
                if r.status_code == 200:
                    print(
                        f"Server ready in {elapsed}s on port "
                        f"{self.port}"
                    )
                    return
            except Exception:
                pass
            time.sleep(1)
            # Check whether the process exited
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"Server died during startup (exit code "
                    f"{self.proc.returncode})"
                )

        raise RuntimeError(
            f"Server failed to start within {timeout}s"
        )

    def is_healthy(self) -> bool:
        """Check if the server is responding to /health."""
        try:
            r = requests.get(
                f"{self.base_url}/health", timeout=5
            )
            return r.status_code == 200
        except Exception:
            return False

    def restart(self, timeout: int = 360) -> None:
        """Stop and restart the server."""
        print("Restarting vLLM server...")
        self.stop()
        time.sleep(5)
        self.start(timeout=timeout)

    def stop(self) -> None:
        """Terminate the server process."""
        if self.proc is not None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None

    # ----------------------------------------------------------
    # Chat requests
    # ----------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0,
        max_tokens: int = 512,
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        """Make a single ``/v1/chat/completions`` request.

        Returns the ``message`` object from the first choice.
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

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    def batch_chat(
        self,
        message_list: list[list[dict[str, Any]]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0,
        max_tokens: int = 512,
        max_workers: int = 16,
        tool_choice: str = "auto",
    ) -> list[tuple[dict[str, Any] | None, str | None]]:
        """Issue many chat requests concurrently via a thread pool.

        Parameters
        ----------
        message_list:
            A list where each element is a conversation (list of
            message dicts) for one sample.
        tools:
            Tool definitions to pass with every request.
        max_workers:
            Number of concurrent HTTP request threads.

        Returns
        -------
        A list of ``(response_message, error_string)`` tuples in the
        same order as *message_list*.  On success ``error_string`` is
        ``None``; on failure ``response_message`` is ``None``.
        """
        results: list[tuple[dict[str, Any] | None, str | None]] = [
            (None, None)
        ] * len(message_list)

        def _do_request(
            idx: int, messages: list[dict[str, Any]]
        ) -> tuple[int, dict[str, Any] | None, str | None]:
            try:
                msg = self.chat(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tool_choice=tool_choice,
                )
                return idx, msg, None
            except Exception as exc:
                return idx, None, str(exc)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_do_request, i, msgs)
                for i, msgs in enumerate(message_list)
            ]
            for done, future in enumerate(
                as_completed(futures), 1
            ):
                idx, result, error = future.result()
                results[idx] = (result, error)
                if done % 500 == 0:
                    print(
                        f"  Progress: {done}/{len(message_list)}"
                    )

        return results

    # ----------------------------------------------------------
    # Context manager
    # ----------------------------------------------------------

    def __enter__(self) -> VLLMServer:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
