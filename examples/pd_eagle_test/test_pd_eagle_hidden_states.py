#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test P/D disaggregation + EAGLE with hidden states transfer.

Flow:
  1. Prefill instance computes KV cache AND hidden states
  2. Both are transferred to the decode instance via RDMA (NixlConnector)
  3. EAGLE uses the transferred hidden states for warm-up (no recompute)

Usage:
    python examples/pd_eagle_test/test_pd_eagle_hidden_states.py
"""

import argparse
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE_DIR = "yuhuili/EAGLE-LLaMA3.1-Instruct-8B"


# ── Utilities ──────────────────────────────────────────────────────────


def is_port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def find_available_port(start: int, exclude: set[int]) -> int:
    for port in range(start, start + 100):
        if port not in exclude and is_port_available(port):
            return port
    raise RuntimeError(f"No available port found starting from {start}")


def tail_file(path: Path, lines: int = 50) -> str:
    if not path or not path.exists():
        return ""
    try:
        with open(path) as f:
            return "".join(f.readlines()[-lines:])
    except Exception:
        return ""


# ── Test Harness ───────────────────────────────────────────────────────


class PDEagleHiddenStatesTest:
    def __init__(self, args):
        self.args = args
        self.model = args.model
        self.eagle_dir = args.eagle_dir
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        used: set[int] = set()

        def alloc(start: int) -> int:
            p = find_available_port(start, used)
            used.add(p)
            return p

        self.prefill_port = alloc(8100)
        self.decode_port = alloc(8200)
        self.prefill_nixl_port = alloc(5560)
        self.decode_nixl_port = alloc(5660)
        self.proxy_port = alloc(8192)

        self.prefill_proc = None
        self.decode_proc = None
        self.proxy_proc = None

        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, lambda *_: (self.cleanup(), sys.exit(0)))
        signal.signal(signal.SIGTERM, lambda *_: (self.cleanup(), sys.exit(0)))

    def cleanup(self):
        print("\nCleaning up...")
        for name, proc in [
            ("prefill", self.prefill_proc),
            ("decode", self.decode_proc),
            ("proxy", self.proxy_proc),
        ]:
            if proc and proc.poll() is None:
                print(f"  Stopping {name} (pid {proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("Done")

    def start_vllm(
        self, role: str, port: int, nixl_port: int, extra_args: list[str] | None = None
    ) -> subprocess.Popen:
        log_file = self.log_dir / f"{role}.log"

        env = os.environ.copy()
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(nixl_port)
        # Ensure the repo root is on PYTHONPATH so the EngineCore child
        # process can import our custom connector module (examples.…).
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        env["PYTHONPATH"] = repo_root + (
            ":" + env["PYTHONPATH"] if "PYTHONPATH" in env else ""
        )

        kv_config = {
            "kv_connector": "NixlConnectorWithAux",
            "kv_connector_module_path": (
                "examples.pd_eagle_test.nixl_connector_with_aux"
            ),
            "kv_role": "kv_both",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "transfer_aux": "true",
                "aux_max_seq_len": "2048",
            },
        }

        cmd = [
            "chg",
            "run",
            "--gpus",
            "1",
            "--",
            "vllm",
            "serve",
            self.model,
            "--port",
            str(port),
            "--enforce-eager",
            "--block-size",
            "128",
            "--gpu-memory-utilization",
            "0.8",
            "--max-model-len",
            "2048",
            "--no-enable-chunked-prefill",
            "--kv-transfer-config",
            json.dumps(kv_config),
        ]
        if extra_args:
            cmd.extend(extra_args)

        print(f"  {role}: port={port}, nixl_port={nixl_port}")
        print(f"    cmd: {' '.join(cmd)}")

        with open(log_file, "w") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        return proc

    def wait_for_server(
        self, name: str, port: int, proc: subprocess.Popen, timeout: int = 300
    ) -> bool:
        start = time.time()
        log_file = self.log_dir / f"{name}.log"

        while time.time() - start < timeout:
            if proc.poll() is not None:
                print(f"  {name}: CRASHED (exit code: {proc.returncode})")
                print(f"\n=== Last 50 lines of {name} log ===")
                print(tail_file(log_file, 50))
                return False

            try:
                resp = urllib.request.urlopen(
                    f"http://localhost:{port}/health", timeout=2
                )
                if resp.status == 200:
                    print(f"  {name}: Ready!")
                    return True
            except (
                urllib.error.URLError,
                ConnectionRefusedError,
                TimeoutError,
                OSError,
            ):
                pass

            print(".", end="", flush=True)
            time.sleep(3)

        print(f"  {name}: TIMEOUT after {timeout}s")
        print(f"\n=== Last 50 lines of {name} log ===")
        print(tail_file(log_file, 50))
        return False

    def start_proxy(self):
        repo_root = Path(__file__).parent.parent.parent
        proxy_script = (
            repo_root / "tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"
        )

        cmd = [
            sys.executable,
            str(proxy_script),
            "--port",
            str(self.proxy_port),
            "--prefiller-hosts",
            "localhost",
            "--prefiller-ports",
            str(self.prefill_port),
            "--decoder-hosts",
            "localhost",
            "--decoder-ports",
            str(self.decode_port),
        ]

        log_file = self.log_dir / "proxy.log"
        with open(log_file, "w") as log:
            self.proxy_proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT
            )
        print(f"  proxy: port={self.proxy_port}")

    def send_request(self, prompt: str, max_tokens: int = 50) -> dict | None:
        url = f"http://localhost:{self.proxy_port}/v1/completions"
        data = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        ).encode()

        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            return json.loads(resp.read().decode())
        except Exception as e:
            print(f"  Request failed: {e}")
            return None

    def get_metrics(self, port: int) -> str | None:
        """Fetch Prometheus metrics from a vLLM instance."""
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5)
            return resp.read().decode()
        except Exception:
            return None

    def parse_spec_decode_metrics(self, metrics_text: str) -> dict:
        """Extract speculative decoding metrics."""
        result = {}
        for line in metrics_text.splitlines():
            if line.startswith("#"):
                continue
            if "spec_decode" in line.lower() or "draft" in line.lower():
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    result[parts[0].strip()] = parts[1].strip()
        return result

    def run(self):
        print("=" * 70)
        print("  P/D + EAGLE with hidden states transfer")
        print(f"  Model: {self.model}")
        print(f"  EAGLE: {self.eagle_dir}")
        print("=" * 70)
        print("Logs:")
        print(f"  prefill: {self.log_dir / 'prefill.log'}")
        print(f"  decode:  {self.log_dir / 'decode.log'}")
        print(f"  proxy:   {self.log_dir / 'proxy.log'}")
        print()

        # ── Launch instances ──────────────────────────────────────────

        print("Starting vLLM instances...")

        # Prefill: target model only, no EAGLE
        self.prefill_proc = self.start_vllm(
            "prefill", self.prefill_port, self.prefill_nixl_port
        )

        # Decode: target model + EAGLE
        spec_config = json.dumps(
            {
                "method": "eagle",
                "model": self.eagle_dir,
                "num_speculative_tokens": 3,
            }
        )
        self.decode_proc = self.start_vllm(
            "decode",
            self.decode_port,
            self.decode_nixl_port,
            extra_args=["--speculative-config", spec_config],
        )

        # ── Wait for health ───────────────────────────────────────────

        print("\nWaiting for servers...")
        for name, port, proc in [
            ("prefill", self.prefill_port, self.prefill_proc),
            ("decode", self.decode_port, self.decode_proc),
        ]:
            print(f"  Waiting for {name}...", end="", flush=True)
            if not self.wait_for_server(name, port, proc):
                self.cleanup()
                return 1

        print("\nStarting proxy...")
        self.start_proxy()
        time.sleep(5)

        # ── Send test requests ────────────────────────────────────────

        prompts = [
            "The theory of general relativity states that",
            "In machine learning, gradient descent works by",
            "The Python programming language was created by",
            "Explain how a transformer neural network processes",
            "The capital of Japan is Tokyo, which is known for",
        ]

        print(f"\nSending {len(prompts)} test requests...")
        results = []
        for i, prompt in enumerate(prompts):
            result = self.send_request(prompt, max_tokens=50)
            if result and "choices" in result:
                text = result["choices"][0]["text"]
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "?")
                completion_tokens = usage.get("completion_tokens", "?")
                print(f"  [{i + 1}/{len(prompts)}] OK: {prompt[:40]}...")
                print(
                    f"    prompt_tokens={prompt_tokens}"
                    f"  completion_tokens={completion_tokens}"
                )
                print(f"    → {text[:80]}...")
                results.append(result)
            else:
                print(f"  [{i + 1}/{len(prompts)}] FAIL: {prompt[:40]}...")
                results.append(None)
            time.sleep(1)

        success = sum(1 for r in results if r is not None)
        print(f"\n  Results: {success}/{len(prompts)} requests succeeded")

        # ── Check decode metrics ──────────────────────────────────────

        print("\nChecking decode instance metrics...")
        metrics = self.get_metrics(self.decode_port)
        if metrics:
            spec_metrics = self.parse_spec_decode_metrics(metrics)
            if spec_metrics:
                print("  Spec decode metrics:")
                for k, v in spec_metrics.items():
                    print(f"    {k}: {v}")
            else:
                print("  (no spec decode metrics found)")
        else:
            print("  (could not fetch metrics)")

        # ── Summary ───────────────────────────────────────────────────

        print("\n" + "=" * 70)
        print(f"  Requests: {success}/{len(prompts)} succeeded")
        print(f"  Logs: {self.log_dir}")
        print("=" * 70)

        self.cleanup()
        return 0 if success == len(prompts) else 1


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Test P/D + EAGLE with hidden states transfer"
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--eagle-dir", default=EAGLE_DIR)
    parser.add_argument("--log-dir", default="/tmp/pd_eagle_hs_test")
    args = parser.parse_args()

    test = PDEagleHiddenStatesTest(args)
    sys.exit(test.run())


if __name__ == "__main__":
    main()
