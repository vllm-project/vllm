#!/usr/bin/env python3
"""
Comprehensive benchmarking script for vLLM serving performance.

This script starts vLLM servers with different attention backends and runs
benchmark_serving.py against them with various configurations.

Usage:
    python benchmark_serving_runner.py --model /path/to/model --tp-size 4
    
Environment variables:
    MODEL_PATH: Default model path
    TP_SIZE: Default tensor parallel size  
    OUTPUT_PATH: Output directory for results
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


class VLLMServerManager:
    """Manages vLLM server lifecycle."""

    def __init__(self, model_path: str, tp_size: int, port: int = 8000):
        self.model_path = model_path
        self.tp_size = tp_size
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.log_file: Optional[str] = None

    def start_server(self, backend_env: Dict[str, str],
                     backend_name: str) -> bool:
        """Start vLLM server with specified backend environment."""
        print(f"Starting {backend_name} server on port {self.port}...")

        # Set up environment
        env = os.environ.copy()
        env.update(backend_env)

        # Prepare command
        cmd = [
            "vllm", "serve", self.model_path, "--host", "0.0.0.0", "--port",
            str(self.port), "--tensor-parallel-size",
            str(self.tp_size), "--quantization", "modelopt",
            "--kv-cache-dtype", "fp8", "--disable-log-requests",
            "--disable-log-stats", "--trust-remote-code"
        ]

        # Start server
        self.log_file = f"{backend_name.lower()}_server.log"
        with open(self.log_file, 'w') as log:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group
            )

        # Wait for server to be ready
        return self._wait_for_ready(timeout=300)

    def _wait_for_ready(self, timeout: int = 60) -> bool:
        """Wait for server to be ready to accept requests."""
        print(f"Waiting for server to be ready on port {self.port}...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://localhost:{self.port}/v1/models", timeout=5)
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except (requests.exceptions.RequestException,
                    requests.exceptions.Timeout):
                pass

            print(f"Waiting... ({int(time.time() - start_time)}s elapsed)")
            time.sleep(2)

        print(f"Server failed to start within {timeout} seconds")
        return False

    def stop_server(self):
        """Stop the vLLM server."""
        if self.process:
            print("Stopping server...")
            try:
                # Kill the process group to ensure all child processes are terminated
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                # Force kill if graceful shutdown fails
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            finally:
                self.process = None

        # Clean up any remaining processes
        try:
            subprocess.run(["pkill", "-f", "vllm serve"], check=False)
        except FileNotFoundError:
            pass

        time.sleep(3)  # Give time for cleanup


class BenchmarkRunner:
    """Runs benchmark_serving.py with various configurations."""

    def __init__(self, model_path: str, output_path: str, port: int = 8000):
        self.model_path = model_path
        self.output_path = Path(output_path)
        self.port = port
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run_benchmark(self, config: Dict) -> bool:
        """Run a single benchmark configuration."""
        concurrency = config['concurrency']
        input_len = config['input_len']
        output_len = config['output_len']
        backend_name = config['backend_name']
        num_requests = concurrency * 10

        result_file = self.output_path / f"{concurrency}-{input_len}-{output_len}-{backend_name.lower()}.json"

        print(
            f"Running benchmark: concurrency={concurrency}, input_len={input_len}, "
            f"output_len={output_len}, backend={backend_name}")

        cmd = [
            "python", "benchmarks/benchmark_serving.py", "--backend", "vllm",
            "--model", self.model_path, "--host", "0.0.0.0", "--port",
            str(self.port), "--dataset-name", "random", "--num-prompts",
            str(num_requests), "--random-input-len",
            str(input_len), "--random-output-len",
            str(output_len), "--max-concurrency",
            str(concurrency), "--result-filename",
            str(result_file), "--save-result"
        ]

        try:
            result = subprocess.run(cmd,
                                    capture_output=True,
                                    text=True,
                                    timeout=1200)
            if result.returncode == 0:
                print(f"✓ Benchmark completed: {result_file}")
                return True
            else:
                print(f"✗ Benchmark failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"✗ Benchmark timed out")
            return False


def generate_test_configurations() -> List[Dict]:
    """Generate test configurations for benchmarking."""
    concurrency_levels = [4, 8]
    output_lengths = [100, 200]
    input_lengths = [100, 2000, 8000, 16000, 32000, 64000]

    configs = []
    for concurrency in concurrency_levels:
        for output_len in output_lengths:
            for input_len in input_lengths:
                configs.append({
                    'concurrency': concurrency,
                    'input_len': input_len,
                    'output_len': output_len
                })

    return configs


def run_backend_benchmarks(server_manager: VLLMServerManager,
                           benchmark_runner: BenchmarkRunner,
                           backend_config: Tuple[Dict[str, str], str],
                           test_configs: List[Dict]) -> List[bool]:
    """Run all benchmarks for a specific backend."""
    backend_env, backend_name = backend_config

    # Start server for this backend
    if not server_manager.start_server(backend_env, backend_name):
        print(f"Failed to start {backend_name} server")
        return [False] * len(test_configs)

    results = []
    try:
        # Run all test configurations
        for config in test_configs:
            config['backend_name'] = backend_name
            success = benchmark_runner.run_benchmark(config)
            results.append(success)

            if not success:
                print(
                    f"Skipping remaining tests for {backend_name} due to failure"
                )
                break

    finally:
        # Always stop the server
        server_manager.stop_server()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run vLLM serving benchmarks")
    parser.add_argument("--model",
                        default=os.getenv("MODEL_PATH",
                                          "/scratch/usr/quantized_model"),
                        help="Model path")
    parser.add_argument("--tp-size",
                        type=int,
                        default=int(os.getenv("TP_SIZE", "4")),
                        help="Tensor parallel size")
    parser.add_argument("--output-path",
                        default=os.getenv("OUTPUT_PATH", "."),
                        help="Output directory for results")
    parser.add_argument("--port",
                        type=int,
                        default=8000,
                        help="Port for vLLM server")
    parser.add_argument("--backends",
                        nargs="+",
                        choices=["tke", "flash-attn", "flashinfer"],
                        default=["tke", "flash-attn"],
                        help="Backends to test")

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        return 1

    # Backend configurations
    backend_configs = {
        "tke": ({
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_ATTENTION_BACKEND": "TKE"
        }, "TKE"),
        "flash-attn": ({
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN_VLLM_V1"
        }, "Flash-Attn"),
        "flashinfer": ({
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER_VLLM_V1"
        }, "FlashInfer")
    }

    test_configs = generate_test_configurations()

    print(f"Running {len(test_configs)} test configurations per backend")
    print(f"Testing backends: {args.backends}")
    print(f"Model: {args.model}")
    print(f"TP Size: {args.tp_size}")
    print(f"Output Path: {args.output_path}")
    print()

    # Initialize managers
    server_manager = VLLMServerManager(args.model, args.tp_size, args.port)
    benchmark_runner = BenchmarkRunner(args.model, args.output_path, args.port)

    # Run benchmarks for each backend
    all_results = {}
    for backend_name in args.backends:
        if backend_name not in backend_configs:
            print(f"Warning: Unknown backend '{backend_name}', skipping")
            continue

        print(f"\n{'='*60}")
        print(
            f"Running benchmarks for {backend_configs[backend_name][1]} backend"
        )
        print(f"{'='*60}")

        results = run_backend_benchmarks(server_manager, benchmark_runner,
                                         backend_configs[backend_name],
                                         test_configs)

        all_results[backend_name] = results
        successful = sum(results)
        total = len(results)
        print(
            f"\n{backend_configs[backend_name][1]} Results: {successful}/{total} successful"
        )

    # Print final summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for backend_name, results in all_results.items():
        successful = sum(results)
        total = len(results)
        success_rate = (successful / total * 100) if total > 0 else 0
        backend_display = backend_configs[backend_name][1]
        print(
            f"{backend_display:12}: {successful:3}/{total:3} successful ({success_rate:5.1f}%)"
        )

    print(f"\nResults saved to: {args.output_path}")

    # Return non-zero exit code if any benchmarks failed
    total_failed = sum(
        len(results) - sum(results) for results in all_results.values())
    return min(total_failed, 1)


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        # Clean up any remaining processes
        try:
            subprocess.run(["pkill", "-f", "vllm serve"], check=False)
        except FileNotFoundError:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
