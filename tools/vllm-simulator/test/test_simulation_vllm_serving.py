"""
Test vLLM server-mode simulation.

1. Spawns vLLM simulation server (launch_server.py) as subprocess
2. Sends requests via OpenAI-compatible /v1/completions API
3. Validates that requests complete and timing is reasonable

Two modes are covered:
- BLOCKING: raw requests against the server, engine sleeps the predicted
  GPU span per step (test_vllm_serving_blocking).
- NON_BLOCKING: the hijacked `vllm bench serve` client
  (vllm_simulator.simulation.bench_serving) replays the traffic logically
  through simulation.created_time, and the benchmark result is substituted
  with the backend-dumped simulation metrics
  (test_vllm_serving_non_blocking).
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import requests

os.environ["VLLM_SIMULATOR_CONFIG_PATH"] = (
    os.path.dirname(__file__) + "/assets/config_vllm.json"
)
os.environ["VLLM_SIMULATOR_OUTPUT_MODE"] = "BLOCKING"

MODEL_PATH = "Qwen/Qwen3-8B"
SERVER_PORT = 18100


class VLLMServingRunner:
    def __init__(self, model: str, port: int = SERVER_PORT, env_overrides: dict | None = None, **extra_args):
        cmd = [
            sys.executable,
            "-m",
            "vllm_simulator.simulation.vllm.launch_server",
            "--model", model,
            "--port", str(port),
            "--dtype", "float16",
            "--load-format", "dummy",
            "--enforce-eager",
            "--block-size", "16",
            "--gpu-memory-utilization", "0.9",
            "--max-model-len", "8192",
            "--num-gpu-blocks-override", "1000000",
        ]
        for k, v in extra_args.items():
            flag = "--" + k.replace("_", "-")
            if v is True:
                cmd.append(flag)
            elif v is False:
                pass
            else:
                cmd.extend([flag, str(v)])

        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.server_proc = subprocess.Popen(
            cmd, env=env, preexec_fn=os.setsid
        )

        # Wait for server to be ready
        dur = 0
        while dur < 120:
            try:
                r = requests.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1)
            dur += 1
        raise RuntimeError("Failed to start vLLM simulation server.")

    def completions(self, prompt: str, max_tokens: int = 10) -> dict:
        """Send a single completion request."""
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "ignore_eos": True,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def shutdown(self):
        if not self.server_proc or self.server_proc.poll() is not None:
            return
        os.killpg(self.server_proc.pid, signal.SIGTERM)
        try:
            self.server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.server_proc.pid, signal.SIGKILL)
            self.server_proc.wait()
        self.server_proc = None


def test_vllm_serving_blocking():
    """Test vLLM server with BLOCKING mode simulation."""
    runner = VLLMServingRunner(model=MODEL_PATH)

    try:
        # Send a few requests sequentially
        results = []
        for i in range(3):
            t0 = time.time()
            result = runner.completions(prompt=f"Hello world {i}", max_tokens=5)
            elapsed = time.time() - t0
            results.append((result, elapsed))

        # Validate responses
        for i, (result, elapsed) in enumerate(results):
            assert "choices" in result, f"Request {i}: no choices in response"
            assert len(result["choices"]) > 0, f"Request {i}: empty choices"
            text = result["choices"][0]["text"]
            assert len(text) > 0, f"Request {i}: empty text output"
            # In BLOCKING mode, each request should take some time due to time.sleep
            # With AIConfigurator predictor, each token takes real inference time
            print(
                f"  Request {i}: output_tokens={result['usage']['completion_tokens']}, "
                f"elapsed={elapsed:.3f}s"
            )

        print("\n[PASS] test_vllm_serving_blocking passed!")
    finally:
        runner.shutdown()


def test_vllm_serving_non_blocking():
    """NON_BLOCKING-mode serving driven by the hijacked `vllm bench serve` client.

    The client (vllm_simulator.simulation.bench_serving) does not pace
    requests in real time; each request carries simulation.created_time and
    the server replays the traffic against its virtual clock. The client then
    substitutes the backend-dumped simulation metrics into the result.
    """
    num_prompts = 8
    request_rate = 10
    output_dir = tempfile.mkdtemp(prefix="sim_non_blocking_output_")
    result_dir = tempfile.mkdtemp(prefix="sim_non_blocking_result_")

    runner = VLLMServingRunner(
        model=MODEL_PATH,
        port=SERVER_PORT + 1,
        env_overrides={
            "VLLM_SIMULATOR_OUTPUT_MODE": "NON_BLOCKING",
            "VLLM_SIMULATOR_OUTPUT_DIR": output_dir,
        },
        # No --profiler-config needed: the EngineArgs hook internally
        # defaults profiler="cuda" so /start_profile & /stop_profile (the
        # sim round markers) are registered.
    )

    try:
        env = os.environ.copy()
        env["VLLM_SIMULATOR_OUTPUT_DIR"] = output_dir
        cmd = [
            sys.executable,
            "-m",
            "vllm_simulator.simulation.bench_serving",
            "--backend", "openai",
            "--base-url", runner.base_url,
            "--model", MODEL_PATH,
            "--dataset-name", "random",
            "--num-prompts", str(num_prompts),
            "--random-input-len", "64",
            "--random-output-len", "8",
            "--request-rate", str(request_rate),
            "--seed", "0",
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--save-result",
            "--result-filename", "bench_result.json",
            "--result-dir", result_dir,
            "--disable-tqdm",
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        assert proc.returncode == 0, f"bench_serving failed:\n{proc.stderr}"

        # The backend stats dump must have been found and substituted.
        assert " Simulation Metrics " in proc.stdout, (
            "simulation metrics were not substituted into the benchmark result"
        )

        with open(os.path.join(result_dir, "bench_result.json")) as f:
            result = json.load(f)

        # Simulation-substituted fields.
        assert result["completed"] == num_prompts
        assert result["num_requests"] == num_prompts
        assert result["duration"] > 0
        assert result["output_throughput"] > 0
        assert result["mean_ttft_ms"] > 0
        assert result["mean_e2el_ms"] > 0
        # Sim-only extra stats prove the result came from the backend dump.
        assert "prompt_tokens_local_cache_hit_ratio" in result

        # Backend per-request stats: all requests replayed by created_time.
        stats_path = os.path.join(output_dir, "request.jsonl")
        with open(stats_path) as f:
            req_stats = [json.loads(line) for line in f if line.strip()]
        assert len(req_stats) == num_prompts
        # request_rate=10 -> logical arrivals spread over ~0.8 virtual seconds.
        assert any(r["created_time"] > 0 for r in req_stats), (
            "expected non-zero created_time for rate-limited replay"
        )
        assert all(
            r["created_time"] >= 0 and len(r["gen_token_latencies"]) > 0
            for r in req_stats
        )

        print("\n[PASS] test_vllm_serving_non_blocking passed!")
    finally:
        runner.shutdown()


if __name__ == "__main__":
    test_vllm_serving_blocking()
    test_vllm_serving_non_blocking()
