#!/usr/bin/env python3
"""
vLLM Profiling Workload Generator

This module provides tools to generate custom requests and schedule plans
for profiling vLLM performance across Prefill and Decode stages.

Factors considered:
- C: Number of input tokens/prompt
- M: Number of KV Cache blocks for precomputed tokens
- B: Batch size
- block_size: KV Cache block size
"""

import time
import random
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Generator
from collections import defaultdict
import argparse
import logging

from vllm import LLMEngine, SamplingParams, RequestOutput
from vllm.config import SchedulerConfig, CacheConfig
from vllm.core.scheduler import Scheduler, SchedulingBudget
from vllm.sequence import SequenceGroup, SequenceData, SequenceStatus
from vllm.utils import FlexibleArgumentParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProfilingConfig:
    """Configuration for profiling workload generation."""

    # Core parameters
    C: int  # Number of input tokens/prompt
    M: int  # Number of KV Cache blocks for precomputed tokens
    B: int  # Batch size
    block_size: int  # KV Cache block size

    # Profiling parameters
    num_warmup_runs: int = 3
    num_measurement_runs: int = 10
    max_model_len: int = 8192
    enable_chunked_prefill: bool = True

    # Advanced scheduling parameters
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_partial_prefills: int = 4
    scheduling_policy: str = "fcfs"

    # Output parameters
    save_results: bool = True
    results_file: str = "profiling_results.json"

    def __post_init__(self):
        """Validate and set default values."""
        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = self.B * self.C

        if self.max_num_seqs is None:
            self.max_num_seqs = self.B * 2  # Allow some headroom

        # Validate parameters
        if self.C <= 0 or self.M <= 0 or self.B <= 0 or self.block_size <= 0:
            raise ValueError(
                "All parameters (C, M, B, block_size) must be positive"
            )

        if self.C > self.max_model_len:
            raise ValueError(
                f"C ({self.C}) cannot exceed max_model_len ({self.max_model_len})"
            )

        if self.M * self.block_size < self.C:
            logger.warning(
                f"M * block_size ({self.M * self.block_size}) < C ({self.C}). "
                f"This may cause issues with KV cache management."
            )


@dataclass
class ProfilingResult:
    """Results from a profiling run."""

    # Configuration
    config: ProfilingConfig

    # Timing results
    prefill_latencies: List[float]  # milliseconds
    decode_latencies: List[float]  # milliseconds
    total_latencies: List[float]  # milliseconds

    # Throughput results
    tokens_per_second: float
    requests_per_second: float

    # Resource utilization
    gpu_memory_usage: Optional[float] = None
    cpu_memory_usage: Optional[float] = None

    # Scheduling statistics
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    avg_batch_size: float = 0.0
    avg_tokens_per_batch: float = 0.0

    # Metadata
    timestamp: float = None
    model_name: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["config"] = asdict(self.config)
        return result

    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        return {
            "config": {
                "C": self.config.C,
                "M": self.config.M,
                "B": self.config.B,
                "block_size": self.config.block_size,
            },
            "prefill_stats": {
                "mean_latency_ms": sum(self.prefill_latencies)
                / len(self.prefill_latencies),
                "min_latency_ms": min(self.prefill_latencies),
                "max_latency_ms": max(self.prefill_latencies),
                "p95_latency_ms": sorted(self.prefill_latencies)[
                    int(0.95 * len(self.prefill_latencies))
                ],
                "p99_latency_ms": sorted(self.prefill_latencies)[
                    int(0.99 * len(self.prefill_latencies))
                ],
            },
            "decode_stats": {
                "mean_latency_ms": sum(self.decode_latencies)
                / len(self.decode_latencies),
                "min_latency_ms": min(self.decode_latencies),
                "max_latency_ms": max(self.decode_latencies),
                "p95_latency_ms": sorted(self.decode_latencies)[
                    int(0.95 * len(self.decode_latencies))
                ],
                "p99_latency_ms": sorted(self.decode_latencies)[
                    int(0.99 * len(self.decode_latencies))
                ],
            },
            "throughput": {
                "tokens_per_second": self.tokens_per_second,
                "requests_per_second": self.requests_per_second,
            },
        }


class ProfilingWorkloadGenerator:
    """Generates custom workloads for vLLM profiling."""

    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.tokenizer = None  # Will be set when engine is provided

    def generate_prompt(self, num_tokens: int) -> str:
        """Generate a prompt with exactly num_tokens tokens."""
        if self.tokenizer is None:
            # Fallback: generate dummy tokens
            return " ".join([f"token_{i}" for i in range(num_tokens)])

        # Generate meaningful text that tokenizes to approximately num_tokens
        base_text = "The quick brown fox jumps over the lazy dog. "
        words = base_text.split()

        # Calculate how many repetitions we need
        tokens_per_repetition = len(self.tokenizer.encode(base_text))
        repetitions_needed = max(1, num_tokens // tokens_per_repetition)

        prompt = base_text * repetitions_needed

        # Trim to exact token count
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > num_tokens:
            tokens = tokens[:num_tokens]
            prompt = self.tokenizer.decode(tokens)
        elif len(tokens) < num_tokens:
            # Pad with additional words
            additional_words = ["word"] * (num_tokens - len(tokens))
            prompt += " " + " ".join(additional_words)

        return prompt

    def create_scheduler_config(self) -> SchedulerConfig:
        """Create a custom scheduler configuration for profiling."""
        return SchedulerConfig(
            runner_type="generate",
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_num_seqs=self.config.max_num_seqs,
            max_model_len=self.config.max_model_len,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            max_num_partial_prefills=self.config.max_num_partial_prefills,
            policy=self.config.scheduling_policy,
            num_lookahead_slots=0,  # Disable for profiling
        )

    def create_cache_config(self) -> CacheConfig:
        """Create a custom cache configuration for profiling."""
        # Calculate required GPU blocks based on M and B
        num_gpu_blocks = max(
            self.config.M * self.config.B, 1024
        )  # Minimum 1024 blocks

        return CacheConfig(
            block_size=self.config.block_size,
            gpu_memory_utilization=0.9,
            swap_space=4.0,
            cache_dtype="auto",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_gpu_blocks // 2,  # Half for CPU offloading
        )

    def generate_prefill_workload(
        self, num_requests: int
    ) -> List[Tuple[str, SamplingParams]]:
        """Generate prefill workload with specified number of requests."""
        workload = []

        for i in range(num_requests):
            # Generate prompt with exactly C tokens
            prompt = self.generate_prompt(self.config.C)

            # Create sampling parameters for prefill
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for profiling
                max_tokens=1,  # Just one token for prefill measurement
                top_p=1.0,
                top_k=1,
            )

            workload.append((prompt, sampling_params))

        return workload

    def generate_decode_workload(
        self, num_requests: int, precomputed_tokens: int = 0
    ) -> List[Tuple[str, SamplingParams]]:
        """Generate decode workload with precomputed tokens."""
        workload = []

        for i in range(num_requests):
            # Generate prompt with precomputed tokens
            prompt = self.generate_prompt(precomputed_tokens)

            # Create sampling parameters for decode
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for profiling
                max_tokens=1,  # One token at a time for decode measurement
                top_p=1.0,
                top_k=1,
            )

            workload.append((prompt, sampling_params))

        return workload

    def generate_mixed_workload(
        self, prefill_ratio: float = 0.5
    ) -> List[Tuple[str, SamplingParams, str]]:
        """Generate mixed workload with both prefill and decode requests."""
        workload = []

        num_prefill = int(self.config.B * prefill_ratio)
        num_decode = self.config.B - num_prefill

        # Generate prefill requests
        for i in range(num_prefill):
            prompt = self.generate_prompt(self.config.C)
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                top_p=1.0,
                top_k=1,
            )
            workload.append((prompt, sampling_params, "prefill"))

        # Generate decode requests (with some precomputed tokens)
        precomputed_tokens = min(
            self.config.M * self.config.block_size, self.config.C
        )
        for i in range(num_decode):
            prompt = self.generate_prompt(precomputed_tokens)
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,
                top_p=1.0,
                top_k=1,
            )
            workload.append((prompt, sampling_params, "decode"))

        # Shuffle to simulate realistic workload
        random.shuffle(workload)

        return workload


class ProfilingRunner:
    """Runs profiling workloads and collects metrics."""

    def __init__(self, engine: LLMEngine, config: ProfilingConfig):
        self.engine = engine
        self.config = config
        self.generator = ProfilingWorkloadGenerator(config)

        # Set tokenizer for prompt generation
        if hasattr(engine, "tokenizer"):
            self.generator.tokenizer = engine.tokenizer

        # Metrics collection
        self.prefill_latencies = []
        self.decode_latencies = []
        self.total_latencies = []
        self.batch_sizes = []
        self.tokens_per_batch = []

    def run_prefill_profiling(self) -> ProfilingResult:
        """Run prefill-only profiling."""
        logger.info(
            f"Running prefill profiling with C={self.config.C}, B={self.config.B}"
        )

        # Warmup runs
        for i in range(self.config.num_warmup_runs):
            logger.info(f"Warmup run {i + 1}/{self.config.num_warmup_runs}")
            self._run_single_batch("prefill")

        # Measurement runs
        for i in range(self.config.num_measurement_runs):
            logger.info(
                f"Measurement run {i + 1}/{self.config.num_measurement_runs}"
            )
            start_time = time.time()

            # Run prefill batch
            batch_latency = self._run_single_batch("prefill")

            end_time = time.time()
            total_latency = (end_time - start_time) * 1000  # Convert to ms

            self.prefill_latencies.append(batch_latency)
            self.total_latencies.append(total_latency)

        return self._create_result("prefill")

    def run_decode_profiling(self) -> ProfilingResult:
        """Run decode-only profiling."""
        logger.info(
            f"Running decode profiling with M={self.config.M}, B={self.config.B}"
        )

        # Warmup runs
        for i in range(self.config.num_warmup_runs):
            logger.info(f"Warmup run {i + 1}/{self.config.num_warmup_runs}")
            self._run_single_batch("decode")

        # Measurement runs
        for i in range(self.config.num_measurement_runs):
            logger.info(
                f"Measurement run {i + 1}/{self.config.num_measurement_runs}"
            )
            start_time = time.time()

            # Run decode batch
            batch_latency = self._run_single_batch("decode")

            end_time = time.time()
            total_latency = (end_time - start_time) * 1000  # Convert to ms

            self.decode_latencies.append(batch_latency)
            self.total_latencies.append(total_latency)

        return self._create_result("decode")

    def run_mixed_profiling(self) -> ProfilingResult:
        """Run mixed prefill/decode profiling."""
        logger.info(
            f"Running mixed profiling with C={self.config.C}, M={self.config.M}, B={self.config.B}"
        )

        # Warmup runs
        for i in range(self.config.num_warmup_runs):
            logger.info(f"Warmup run {i + 1}/{self.config.num_warmup_runs}")
            self._run_mixed_batch()

        # Measurement runs
        for i in range(self.config.num_measurement_runs):
            logger.info(
                f"Measurement run {i + 1}/{self.config.num_measurement_runs}"
            )
            start_time = time.time()

            # Run mixed batch
            prefill_latency, decode_latency = self._run_mixed_batch()

            end_time = time.time()
            total_latency = (end_time - start_time) * 1000  # Convert to ms

            self.prefill_latencies.append(prefill_latency)
            self.decode_latencies.append(decode_latency)
            self.total_latencies.append(total_latency)

        return self._create_result("mixed")

    def _run_single_batch(self, batch_type: str) -> float:
        """Run a single batch of requests and measure latency."""
        if batch_type == "prefill":
            workload = self.generator.generate_prefill_workload(self.config.B)
        else:  # decode
            workload = self.generator.generate_decode_workload(self.config.B)

        # Add requests to engine
        request_ids = []
        for i, (prompt, sampling_params) in enumerate(workload):
            request_id = f"{batch_type}_{i}_{int(time.time() * 1000)}"
            self.engine.add_request(request_id, prompt, sampling_params)
            request_ids.append(request_id)

        # Measure batch execution time
        start_time = time.time()

        # Run until all requests are finished
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for output in outputs:
                if output.finished:
                    logger.debug(f"Request {output.request_id} finished")

        end_time = time.time()
        batch_latency = (end_time - start_time) * 1000  # Convert to ms

        # Record batch statistics
        self.batch_sizes.append(len(request_ids))
        self.tokens_per_batch.append(self.config.C * len(request_ids))

        return batch_latency

    def _run_mixed_batch(self) -> Tuple[float, float]:
        """Run a mixed batch with both prefill and decode requests."""
        workload = self.generator.generate_mixed_workload()

        # Separate prefill and decode requests
        prefill_requests = []
        decode_requests = []

        for prompt, sampling_params, req_type in workload:
            request_id = f"{req_type}_{int(time.time() * 1000)}"
            if req_type == "prefill":
                prefill_requests.append((request_id, prompt, sampling_params))
            else:
                decode_requests.append((request_id, prompt, sampling_params))

        # Add all requests to engine
        all_request_ids = []
        for request_id, prompt, sampling_params in (
            prefill_requests + decode_requests
        ):
            self.engine.add_request(request_id, prompt, sampling_params)
            all_request_ids.append(request_id)

        # Measure execution time
        start_time = time.time()

        # Track when prefill and decode requests finish
        prefill_finished = False
        decode_finished = False
        prefill_end_time = None
        decode_end_time = None

        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            current_time = time.time()

            for output in outputs:
                if output.finished:
                    request_id = output.request_id
                    if (
                        request_id.startswith("prefill")
                        and not prefill_finished
                    ):
                        prefill_finished = True
                        prefill_end_time = current_time
                    elif (
                        request_id.startswith("decode") and not decode_finished
                    ):
                        decode_finished = True
                        decode_end_time = current_time

        end_time = time.time()

        # Calculate latencies
        prefill_latency = (
            (prefill_end_time - start_time) * 1000 if prefill_end_time else 0
        )
        decode_latency = (
            (decode_end_time - start_time) * 1000 if decode_end_time else 0
        )

        # Record batch statistics
        self.batch_sizes.append(len(all_request_ids))
        self.tokens_per_batch.append(
            sum(
                [
                    self.config.C
                    if req_id.startswith("prefill")
                    else self.config.M * self.config.block_size
                    for req_id in all_request_ids
                ]
            )
        )

        return prefill_latency, decode_latency

    def _create_result(self, profiling_type: str) -> ProfilingResult:
        """Create profiling result from collected metrics."""
        # Calculate throughput
        total_tokens = sum(self.tokens_per_batch)
        total_time = sum(self.total_latencies) / 1000  # Convert back to seconds
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        total_requests = sum(self.batch_sizes)
        requests_per_second = (
            total_requests / total_time if total_time > 0 else 0
        )

        # Create result
        result = ProfilingResult(
            config=self.config,
            prefill_latencies=self.prefill_latencies,
            decode_latencies=self.decode_latencies,
            total_latencies=self.total_latencies,
            tokens_per_second=tokens_per_second,
            requests_per_second=requests_per_second,
            num_prefill_requests=len(
                [x for x in self.prefill_latencies if x > 0]
            ),
            num_decode_requests=len(
                [x for x in self.decode_latencies if x > 0]
            ),
            avg_batch_size=sum(self.batch_sizes) / len(self.batch_sizes)
            if self.batch_sizes
            else 0,
            avg_tokens_per_batch=sum(self.tokens_per_batch)
            / len(self.tokens_per_batch)
            if self.tokens_per_batch
            else 0,
            model_name=getattr(self.engine, "model_name", "unknown"),
        )

        # Save results if requested
        if self.config.save_results:
            self._save_results(result, profiling_type)

        return result

    def _save_results(self, result: ProfilingResult, profiling_type: str):
        """Save profiling results to file."""
        filename = f"{profiling_type}_{self.config.results_file}"

        with open(filename, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Results saved to {filename}")

        # Print summary
        summary = result.get_summary_stats()
        logger.info(f"Profiling Summary ({profiling_type}):")
        logger.info(
            f"  Config: C={summary['config']['C']}, M={summary['config']['M']}, "
            f"B={summary['config']['B']}, block_size={summary['config']['block_size']}"
        )
        logger.info(
            f"  Prefill P95: {summary['prefill_stats']['p95_latency_ms']:.2f}ms"
        )
        logger.info(
            f"  Decode P95: {summary['decode_stats']['p95_latency_ms']:.2f}ms"
        )
        logger.info(
            f"  Throughput: {summary['throughput']['tokens_per_second']:.2f} tokens/sec"
        )


def create_profiling_engine(
    config: ProfilingConfig, model_name: str = "microsoft/DialoGPT-medium"
) -> LLMEngine:
    """Create a vLLM engine configured for profiling."""
    from vllm import EngineArgs

    # Create engine arguments
    engine_args = EngineArgs(
        model=model_name,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        max_model_len=config.max_model_len,
        enable_chunked_prefill=config.enable_chunked_prefill,
        max_num_partial_prefills=config.max_num_partial_prefills,
        scheduling_policy=config.scheduling_policy,
        block_size=config.block_size,
        gpu_memory_utilization=0.9,
        swap_space=4.0,
        disable_log_stats=True,  # Disable for cleaner profiling
    )

    # Create engine
    engine = LLMEngine.from_engine_args(engine_args)
    return engine


def main():
    """Main function for running profiling workloads."""
    parser = argparse.ArgumentParser(
        description="vLLM Profiling Workload Generator"
    )

    # Core parameters
    parser.add_argument(
        "--C", type=int, required=True, help="Number of input tokens/prompt"
    )
    parser.add_argument(
        "--M", type=int, required=True, help="Number of KV Cache blocks"
    )
    parser.add_argument("--B", type=int, required=True, help="Batch size")
    parser.add_argument(
        "--block-size", type=int, default=16, help="KV Cache block size"
    )

    # Profiling parameters
    parser.add_argument(
        "--num-warmup", type=int, default=3, help="Number of warmup runs"
    )
    parser.add_argument(
        "--num-measurements",
        type=int,
        default=10,
        help="Number of measurement runs",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192, help="Maximum model length"
    )

    # Profiling type
    parser.add_argument(
        "--profiling-type",
        choices=["prefill", "decode", "mixed"],
        default="mixed",
        help="Type of profiling to run",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Model to use for profiling",
    )

    # Output
    parser.add_argument(
        "--results-file",
        type=str,
        default="profiling_results.json",
        help="Results file name",
    )

    args = parser.parse_args()

    # Create configuration
    config = ProfilingConfig(
        C=args.C,
        M=args.M,
        B=args.B,
        block_size=args.block_size,
        num_warmup_runs=args.num_warmup,
        num_measurement_runs=args.num_measurements,
        max_model_len=args.max_model_len,
        results_file=args.results_file,
    )

    # Create engine
    logger.info("Creating vLLM engine...")
    engine = create_profiling_engine(config, args.model)

    # Create runner
    runner = ProfilingRunner(engine, config)

    # Run profiling
    logger.info(f"Starting {args.profiling_type} profiling...")

    if args.profiling_type == "prefill":
        result = runner.run_prefill_profiling()
    elif args.profiling_type == "decode":
        result = runner.run_decode_profiling()
    else:  # mixed
        result = runner.run_mixed_profiling()

    # Print final summary
    summary = result.get_summary_stats()
    print("\n" + "=" * 60)
    print("PROFILING RESULTS SUMMARY")
    print("=" * 60)
    print(
        f"Configuration: C={summary['config']['C']}, M={summary['config']['M']}, "
        f"B={summary['config']['B']}, block_size={summary['config']['block_size']}"
    )
    print(f"Profiling Type: {args.profiling_type}")
    print(f"Model: {args.model}")
    print()
    print("Latency Statistics (ms):")
    print(
        f"  Prefill - Mean: {summary['prefill_stats']['mean_latency_ms']:.2f}, "
        f"P95: {summary['prefill_stats']['p95_latency_ms']:.2f}, "
        f"P99: {summary['prefill_stats']['p99_latency_ms']:.2f}"
    )
    print(
        f"  Decode  - Mean: {summary['decode_stats']['mean_latency_ms']:.2f}, "
        f"P95: {summary['decode_stats']['p95_latency_ms']:.2f}, "
        f"P99: {summary['decode_stats']['p99_latency_ms']:.2f}"
    )
    print()
    print("Throughput:")
    print(f"  Tokens/sec: {summary['throughput']['tokens_per_second']:.2f}")
    print(f"  Requests/sec: {summary['throughput']['requests_per_second']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
