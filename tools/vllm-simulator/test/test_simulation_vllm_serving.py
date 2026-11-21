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
  with the simulation metrics computed from the server's per-request stats
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

MODEL_PATH = "Qwen/Qwen3-8B"
SERVER_PORT = 18100


class VLLMServingRunner:
    def __init__(
        self,
        model: str,
        port: int = SERVER_PORT,
        env_overrides: dict | None = None,
        **extra_args,
    ):
        cmd = [
            sys.executable,
            "-m",
            "vllm_simulator.simulation.vllm.launch_server",
            "--model",
            model,
            "--port",
            str(port),
            "--dtype",
            "float16",
            "--load-format",
            "dummy",
            "--enforce-eager",
            "--block-size",
            "16",
            "--gpu-memory-utilization",
            "0.9",
            "--max-model-len",
            "8192",
            "--num-gpu-blocks-override",
            "1000000",
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
        self.server_proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)

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
        self.shutdown()
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
    # Scoped to the server process: a module-level os.environ write would
    # leak into in-process engines of other test modules in the same session.
    runner = VLLMServingRunner(
        model=MODEL_PATH, env_overrides={"VLLM_SIMULATOR_OUTPUT_MODE": "BLOCKING"}
    )

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
            # With AISimulate predictor, each token takes real inference time
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
    computes metrics from stats returned with each non-streaming completion
    or chat response, with no separate stats endpoint.
    """
    num_prompts = 8
    request_rate = 10
    result_dir = tempfile.mkdtemp(prefix="sim_non_blocking_result_")

    runner = VLLMServingRunner(
        model=MODEL_PATH,
        port=SERVER_PORT + 1,
        env_overrides={"VLLM_SIMULATOR_OUTPUT_MODE": "NON_BLOCKING"},
    )

    try:
        for backend in ("openai", "openai-chat"):
            filename = f"bench_{backend}.json"
            cmd = [
                sys.executable,
                "-m",
                "vllm_simulator.simulation.bench_serving",
                "--backend",
                backend,
                "--endpoint",
                (
                    "/v1/chat/completions"
                    if backend == "openai-chat"
                    else "/v1/completions"
                ),
                "--base-url",
                runner.base_url,
                "--model",
                MODEL_PATH,
                "--dataset-name",
                "random",
                "--num-prompts",
                str(num_prompts),
                "--random-input-len",
                "64",
                "--random-output-len",
                "8",
                "--request-rate",
                str(request_rate),
                "--seed",
                "0",
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--metric-percentiles",
                "50,75,99",
                "--num-warmups",
                "2",
                "--ready-check-timeout-sec",
                "60",
                "--ignore-eos",
                "--save-result",
                "--save-detailed",
                "--result-filename",
                filename,
                "--result-dir",
                result_dir,
                "--disable-tqdm",
            ]
            if backend == "openai-chat":
                cmd.extend(["--goodput", "ttft:10000", "tpot:10000", "e2el:10000"])
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            assert proc.returncode == 0, f"{backend} failed:\n{proc.stderr}"
            assert " Simulation Metrics " in proc.stdout
            with open(os.path.join(result_dir, filename)) as f:
                result = json.load(f)

            # The readiness check and warmups must not count toward results.
            assert result["completed"] == result["num_requests"] == num_prompts
            assert result["total_output_tokens"] == num_prompts * 8
            assert result["duration"] > 0
            assert result["output_throughput"] > 0
            assert result["mean_ttft_ms"] > 0
            assert result["mean_e2el_ms"] > 0
            assert result["p75_ttft_ms"] >= result["p50_ttft_ms"]
            assert "prompt_tokens_local_cache_hit_ratio" in result
            if backend == "openai-chat":
                assert (
                    abs(result["request_goodput"] - result["request_throughput"]) < 1e-6
                )
            else:
                assert result["request_goodput"] is None
            assert len(result["ttfts"]) == num_prompts
            assert (
                abs(sum(result["ttfts"]) / num_prompts * 1000 - result["mean_ttft_ms"])
                < 1e-6
            )

        # The previous round reset the virtual clock, so this non-streaming
        # request is replayed from its own created_time.
        created_time = 0.5
        resp = requests.post(
            f"{runner.base_url}/v1/completions",
            json={
                "model": MODEL_PATH,
                "prompt": "Hello world",
                "max_tokens": 4,
                "ignore_eos": True,
                "vllm_xargs": {
                    "simulation": json.dumps(
                        {"created_time": created_time, "total_request": 1}
                    ),
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        stats = resp.json()["kv_transfer_params"]["simulation"]
        assert stats["created_time"] == created_time
        assert len(stats["gen_token_latencies"]) == 4
        assert stats["last_event_time"] > created_time

        print("\n[PASS] test_vllm_serving_non_blocking passed!")
    finally:
        runner.shutdown()


if __name__ == "__main__":
    test_vllm_serving_blocking()
    test_vllm_serving_non_blocking()
