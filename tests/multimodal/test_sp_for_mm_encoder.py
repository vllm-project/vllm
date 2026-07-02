#!/usr/bin/env python3
"""
E2E Test Script for Sequence Parallelism Performance Comparison

This script:
1. Starts baseline vLLM server (TP=2, SP=False)
2. Runs benchmark and extracts metrics
3. Shuts down baseline server
4. Starts SP-enabled vLLM server (TP=2, SP=True)
5. Runs benchmark and extracts metrics
6. Compares results and generates report
"""

import subprocess
import time
import re
import json
import signal
import sys
from typing import Dict, Optional, List
from dataclasses import dataclass
import threading
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark results"""
    successful_requests: int = 0
    failed_requests: int = 0
    max_concurrency: float = 0.0
    request_rate: float = 0.0
    benchmark_duration: float = 0.0
    total_input_tokens: int = 0
    total_generated_tokens: int = 0
    request_throughput: float = 0.0
    output_token_throughput: float = 0.0
    peak_output_token_throughput: float = 0.0
    peak_concurrent_requests: float = 0.0
    total_token_throughput: float = 0.0
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0


class VLLMServerManager:
    """Manage vLLM server lifecycle"""
    
    def __init__(self, model_path: str, tp_size: int = 2, enable_sp: bool = False, port: int = 8000):
        self.model_path = model_path
        self.tp_size = tp_size
        self.enable_sp = enable_sp
        self.port = port
        self.process = None
        self.log_file = f"vllm_server_{'sp' if enable_sp else 'baseline'}.log"
        
    def start(self) -> bool:
        """Start vLLM server"""
        if self.process and self.process.poll() is None:
            logger.warning("Server already running")
            return True
            
        env = os.environ.copy()
        env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
            
        cmd = [
            "vllm", "serve", self.model_path,
            "--dtype", "bfloat16",
            "--max-model-len", "16384",
            "--tensor_parallel_size", str(self.tp_size),
            "--limit-mm-per-prompt", '{"image": 3, "video": 0}',
            "--port", str(self.port)
        ]
        
        if self.enable_sp:
            cmd.append("--enable-mm-encoder-sp")
            logger.info("Starting vLLM server with Sequence Parallelism enabled...")
        else:
            logger.info("Starting baseline vLLM server (TP=2, SP=False)...")
        
        # Redirect output to log file
        with open(self.log_file, 'w') as f:
            self.process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid if os.name != 'nt' else None,
                env=env
            )
        
        # Wait for server to be ready
        return self._wait_for_ready()
    
    def _wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for server to be ready by checking logs"""
        logger.info("Waiting for server to be ready...")
        start_time = time.time()
        ready_patterns = [
            r"Uvicorn running on",
            r"Application startup complete",
            r"INFO:.*Application startup complete",
        ]
        
        while time.time() - start_time < timeout:
            try:
                with open(self.log_file, 'r') as f:
                    content = f.read()
                    for pattern in ready_patterns:
                        if re.search(pattern, content):
                            logger.info(f"Server ready in {time.time() - start_time:.1f}s")
                            time.sleep(5)  # extra wait for stability
                            return True
            except FileNotFoundError:
                pass
            time.sleep(2)
        
        logger.error(f"Server failed to start within {timeout}s")
        return False
    
    def stop(self):
        """Stop vLLM server"""
        if self.process is None:
            return
            
        logger.info("Stopping vLLM server...")
        
        # Send SIGINT to allow graceful shutdown
        try:
            if os.name != 'nt':
                os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            else:
                self.process.terminate()
        except ProcessLookupError:
            pass
            
        # Wait for process to terminate
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        
        self.process = None
        logger.info("Server stopped")
        time.sleep(3)  # Wait for port release


class BenchmarkRunner:
    """Run vLLM benchmark and parse results"""
    
    def __init__(self, model_path: str, num_prompts: int = 100, max_concurrency: int = 10, port: int = 8000):
        self.model_path = model_path
        self.num_prompts = num_prompts
        self.max_concurrency = max_concurrency
        self.port = port
        self.output_file = "benchmark_output.txt"
        
    def run(self) -> Optional[BenchmarkMetrics]:
        """Run benchmark and return metrics"""
        logger.info("Starting benchmark...")
        
        cmd = [
            "vllm", "bench", "serve",
            "--backend", "openai-chat",
            "--model", self.model_path,
            "--port", str(self.port),
            "--endpoint", "/v1/chat/completions",
            "--dataset-name", "random-mm",
            "--num-prompts", str(self.num_prompts),
            "--max-concurrency", str(self.max_concurrency),
            "--random-prefix-len", "25",
            "--random-input-len", "300",
            "--random-output-len", "40",
            "--random-range-ratio", "0.2",
            "--random-mm-base-items-per-request", "2",
            "--random-mm-limit-mm-per-prompt", '{"image": 3, "video": 0}',
            "--random-mm-bucket-config", '{(256, 256, 1): 0.25, (720, 720, 1): 0.25, (1080, 1080, 1): 0.25, (2080, 2080, 1): 0.25}',
            "--request-rate", "inf",
            "--ignore-eos",
            "--seed", "42"
        ]
        
        try:
            # Run benchmark and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            output = result.stdout
            logger.info("Benchmark completed successfully")
            
            # Save output for debugging
            with open(self.output_file, 'w') as f:
                f.write(output)
            
            return self._parse_benchmark_output(output)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Benchmark failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return None
    
    def _parse_benchmark_output(self, output: str) -> BenchmarkMetrics:
        """Parse benchmark output and extract metrics"""
        metrics = BenchmarkMetrics()
        
        # Define regex patterns
        patterns = {
            'successful_requests': r"Successful requests:\s*(\d+)",
            'failed_requests': r"Failed requests:\s*(\d+)",
            'max_concurrency': r"Maximum request concurrency:\s*(\d+)",
            'request_rate': r"Request rate configured \(RPS\):\s*([\d.]+)",
            'benchmark_duration': r"Benchmark duration \(s\):\s*([\d.]+)",
            'total_input_tokens': r"Total input tokens:\s*(\d+)",
            'total_generated_tokens': r"Total generated tokens:\s*(\d+)",
            'request_throughput': r"Request throughput \(req/s\):\s*([\d.]+)",
            'output_token_throughput': r"Output token throughput \(tok/s\):\s*([\d.]+)",
            'peak_output_token_throughput': r"Peak output token throughput \(tok/s\):\s*([\d.]+)",
            'peak_concurrent_requests': r"Peak concurrent requests:\s*([\d.]+)",
            'total_token_throughput': r"Total token throughput \(tok/s\):\s*([\d.]+)",
            'mean_ttft_ms': r"Mean TTFT \(ms\):\s*([\d.]+)",
            'median_ttft_ms': r"Median TTFT \(ms\):\s*([\d.]+)",
            'p99_ttft_ms': r"P99 TTFT \(ms\):\s*([\d.]+)",
            'mean_tpot_ms': r"Mean TPOT \(ms\):\s*([\d.]+)",
            'median_tpot_ms': r"Median TPOT \(ms\):\s*([\d.]+)",
            'p99_tpot_ms': r"P99 TPOT \(ms\):\s*([\d.]+)",
            'mean_itl_ms': r"Mean ITL \(ms\):\s*([\d.]+)",
            'median_itl_ms': r"Median ITL \(ms\):\s*([\d.]+)",
            'p99_itl_ms': r"P99 ITL \(ms\):\s*([\d.]+)",
        }
        
        for attr, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                value_str = match.group(1)
                # Convert to appropriate type
                if attr in ['successful_requests', 'failed_requests', 'total_input_tokens', 
                           'total_generated_tokens']:
                    value = int(value_str)
                else:
                    value = float(value_str)
                setattr(metrics, attr, value)
            else:
                logger.warning(f"Could not find metric: {attr}")
        
        return metrics


def print_comparison(baseline: BenchmarkMetrics, sp: BenchmarkMetrics):
    """Print comparison between baseline and SP"""
    if not baseline or not sp:
        logger.error("Missing metrics data")
        return
    
    logger.info("\n" + "="*80)
    logger.info("SEQUENCE PARALLELISM PERFORMANCE COMPARISON")
    logger.info("="*80)
    
    # Header
    print("\n{:<35} {:>20} {:>20} {:>15}".format(
        "Metric", "Baseline (TP=2)", "With SP (TP=2)", "Speedup"
    ))
    print("-"*92)
    
    # Compare metrics
    metrics_to_compare = [
        ('Request Throughput (req/s)', 'request_throughput', True),
        ('Output Token Throughput (tok/s)', 'output_token_throughput', True),
        ('Peak Output Token Throughput (tok/s)', 'peak_output_token_throughput', True),
        ('Total Token Throughput (tok/s)', 'total_token_throughput', True),
        ('Mean TTFT (ms)', 'mean_ttft_ms', False),
        ('Median TTFT (ms)', 'median_ttft_ms', False),
        ('P99 TTFT (ms)', 'p99_ttft_ms', False),
        ('Mean TPOT (ms)', 'mean_tpot_ms', False),
        ('Median TPOT (ms)', 'median_tpot_ms', False),
        ('P99 TPOT (ms)', 'p99_tpot_ms', False),
        ('Mean ITL (ms)', 'mean_itl_ms', False),
        ('Median ITL (ms)', 'median_itl_ms', False),
        ('P99 ITL (ms)', 'p99_itl_ms', False),
        ('Benchmark Duration (s)', 'benchmark_duration', False),
    ]
    
    summary = {}
    for label, attr, higher_better in metrics_to_compare:
        baseline_val = getattr(baseline, attr, 0)
        sp_val = getattr(sp, attr, 0)
        
        if baseline_val > 0 and sp_val > 0:
            if higher_better:
                speedup = sp_val / baseline_val
                change_symbol = "▲" if speedup > 1 else "▼"
            else:
                speedup = baseline_val / sp_val
                change_symbol = "▲" if speedup > 1 else "▼"
            
            summary[label] = {
                'baseline': baseline_val,
                'sp': sp_val,
                'speedup': speedup,
                'symbol': change_symbol
            }
            
            print("{:<35} {:>20.2f} {:>20.2f} {:>14.2f}x {}".format(
                label, baseline_val, sp_val, speedup, change_symbol
            ))
        else:
            print("{:<35} {:>20} {:>20} {:>15}".format(
                label, "N/A", "N/A", "N/A"
            ))
    
    # Overall summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    throughput_improvement = summary.get('Total Token Throughput (tok/s)', {}).get('speedup', 0)
    if throughput_improvement > 0:
        logger.info(f"✓ Overall throughput speedup: {throughput_improvement:.2f}x")
        
        if throughput_improvement > 1.05:
            logger.info(f"✓ Sequence Parallelism shows significant improvement (+{(throughput_improvement-1)*100:.1f}%)")
        elif throughput_improvement > 0.95:
            logger.info("≈ Sequence Parallelism performance is similar to baseline")
        else:
            logger.warning(f"✗ Sequence Parallelism shows degradation ({(1-throughput_improvement)*100:.1f}%)")
    
    # Generate JSON report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen3-VL-30B-A3B-Instruct",
        "tensor_parallel_size": 2,
        "baseline": {
            "enable_sp": False,
            "metrics": {k: getattr(baseline, k) for k in baseline.__annotations__.keys()}
        },
        "with_sp": {
            "enable_sp": True,
            "metrics": {k: getattr(sp, k) for k in sp.__annotations__.keys()}
        },
        "speedup": {k: v['speedup'] for k, v in summary.items()}
    }
    
    with open('sp_benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("\nDetailed report saved to: sp_benchmark_report.json")


def cleanup_on_exit(*args):
    """Cleanup function for graceful exit"""
    logger.info("\nCleaning up...")
    # Kill any remaining vLLM processes
    try:
        subprocess.run(['pkill', '-f', 'vllm serve'], check=False)
    except:
        pass
    sys.exit(0)


def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup_on_exit)
    signal.signal(signal.SIGTERM, cleanup_on_exit)
    
    # Configuration
    MODEL_PATH = "models/Qwen3-VL-30B-A3B-Instruct"
    NUM_PROMPTS = 100
    MAX_CONCURRENCY = 10
    TP_SIZE = 2
    PORT = 3456
    
    # Test Sequence Parallelism
    sp_test = [
        ('Baseline', False),
        ('Sequence Parallelism', True)
    ]
    
    results = []
    
    for test_name, enable_sp in sp_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*60}\n")
        
        # Start server
        server = VLLMServerManager(MODEL_PATH, TP_SIZE, enable_sp, PORT)
        if not server.start():
            logger.error(f"Failed to start {test_name} server")
            continue
        
        # Run benchmark
        benchmark = BenchmarkRunner(MODEL_PATH, NUM_PROMPTS, MAX_CONCURRENCY, PORT)
        metrics = benchmark.run()
        
        if metrics is None:
            logger.error(f"Benchmark failed for {test_name}")
            server.stop()
            continue
        
        results.append((test_name, metrics))
        
        # Stop server
        server.stop()
        
        # Add wait between tests
        logger.info("Waiting before next test...")
        time.sleep(10)
    
    # Compare results
    if len(results) == 2:
        baseline_metrics = results[0][1]
        sp_metrics = results[1][1]
        print_comparison(baseline_metrics, sp_metrics)
    else:
        logger.error("Not enough benchmark results to compare")
    
    cleanup_on_exit()


if __name__ == "__main__":
    main()