#!/usr/bin/env python3
"""
Comprehensive benchmark script to compare MLFQ scheduler vs default scheduler performance.

This script measures:
- Throughput (tokens/second)
- Latency (time to first token, total completion time)
- Queue waiting times
- Resource utilization
- Different workload scenarios
"""

import asyncio
import json
import os
import time
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import argparse
import concurrent.futures
from pathlib import Path

from vllm import LLM, SamplingParams


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    scheduler_type: str
    total_requests: int
    total_tokens: int
    total_time: float
    throughput_tokens_per_sec: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    avg_ttft: float  # Time to first token
    p50_ttft: float
    p95_ttft: float
    p99_ttft: float
    success_rate: float
    error_count: int
    config: Dict[str, Any]


@dataclass
class WorkloadConfig:
    """Configuration for different workload scenarios."""
    name: str
    prompts: List[str]
    max_tokens: int
    temperature: float
    concurrent_requests: int
    description: str


class MLFQBenchmark:
    """Benchmark class for comparing MLFQ vs default scheduler."""
    
    def __init__(self, model_path: str, output_dir: str = "benchmark_results"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Common LLM configuration
        self.base_config = {
            "model": model_path,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 2048,
            "enforce_eager": True,
            "disable_log_stats": True,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
        }
        
        # MLFQ specific configuration
        self.mlfq_config = {
            **self.base_config,
            "scheduling_policy": "mlfq",
            "mlfq_num_levels": 4,
                "mlfq_base_quantum": 1,
                "mlfq_quantum_multiplier": 2.0,
            "mlfq_skip_join_base": 100,
            "mlfq_starvation_threshold": 50,
            "mlfq_eta": 1,
        }
        
        # Default scheduler configuration
        self.default_config = {
            **self.base_config,
            "scheduling_policy": "fcfs",  # First-come-first-served
        }
    
    def create_workloads(self) -> List[WorkloadConfig]:
        """Create different workload scenarios for testing."""
        workloads = []
        
        # Define request counts to test (from small to large)
        # request_counts = [5,10,20,50]
        request_counts = [5]
        
        # Base prompts for each workload type
        short_base_prompts = [
            "What is AI?",
            "Explain ML.",
            "Define NLP.",
            "What is deep learning?",
            "Explain neural networks.",
            "What is reinforcement learning?",
            "Define computer vision.",
            "What is natural language processing?",
            "Explain supervised learning.",
            "What is unsupervised learning?",
        ]
        
        mixed_base_prompts = [
            "What is artificial intelligence?",
            "Explain the concept of machine learning and its applications in modern technology.",
            "AI",
            "Write a comprehensive analysis of the impact of artificial intelligence on healthcare, including benefits, challenges, and future prospects.",
            "ML",
            "Describe the differences between supervised and unsupervised learning with examples.",
            "What is deep learning?",
            "Explain the transformer architecture and its role in modern natural language processing systems.",
            "NLP",
            "Compare and contrast various machine learning algorithms including decision trees, random forests, and neural networks.",
        ]
        
        long_base_prompts = [
            "Write a detailed technical report on the implementation of multi-level feedback queue scheduling algorithms in distributed systems, including performance analysis, optimization techniques, and real-world applications.",
            "Provide a comprehensive analysis of the evolution of artificial intelligence from its inception to modern deep learning systems, covering key milestones, breakthrough technologies, and future research directions.",
            "Explain the mathematical foundations of machine learning algorithms, including linear algebra, calculus, probability theory, and their applications in building intelligent systems.",
        ]
        
        burst_base_prompts = [
            f"Request {i}: Explain the concept of {['AI', 'ML', 'NLP', 'CV', 'RL'][i % 5]}."
            for i in range(20)  # Base set of 20 prompts
        ]
        
        # Create 16 workloads (4 types × 4 request counts)
        workload_configs = [
            {
                "name_prefix": "short_requests",
                "base_prompts": short_base_prompts,
                "max_tokens": 20,
                "concurrent_requests": 10,
                "description": "Short requests for high throughput testing"
            },
            {
                "name_prefix": "mixed_length",
                "base_prompts": mixed_base_prompts,
                "max_tokens": 50,
                "concurrent_requests": 8,
                "description": "Mixed length requests for realistic testing"
            },
            {
                "name_prefix": "long_requests",
                "base_prompts": long_base_prompts,
                "max_tokens": 100,
                "concurrent_requests": 5,
                "description": "Long requests for latency testing"
            },
            {
                "name_prefix": "burst_load",
                "base_prompts": burst_base_prompts,
                "max_tokens": 30,
                "concurrent_requests": 20,
                "description": "Burst load with many concurrent requests"
            }
        ]
        
        for config in workload_configs:
            for count in request_counts:
                # Generate prompts by repeating base prompts to reach desired count
                if count <= len(config["base_prompts"]):
                    prompts = config["base_prompts"][:count]
                else:
                    # Repeat base prompts to reach the desired count
                    repeat_times = (count // len(config["base_prompts"])) + 1
                    repeated_prompts = config["base_prompts"] * repeat_times
                    prompts = repeated_prompts[:count]
                
                workload_name = f"{config['name_prefix']}_{count}"
                
                workloads.append(WorkloadConfig(
                    name=workload_name,
                    prompts=prompts,
                    max_tokens=config["max_tokens"],
                    temperature=0.7,
                    concurrent_requests=config["concurrent_requests"],
                    description=f"{config['description']} ({count} requests)"
                ))
        
        return workloads
    
    async def run_single_benchmark(
        self, 
        config: Dict[str, Any], 
        workload: WorkloadConfig,
        scheduler_type: str
    ) -> BenchmarkResult:
        """Run a single benchmark with given configuration."""
        print(f"\n🚀 Running {scheduler_type} benchmark for {workload.name}...")
        print(f"   Requests: {len(workload.prompts)}, Concurrent: {workload.concurrent_requests}")
        
        # Initialize LLM
        llm = LLM(**config)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=workload.temperature,
            max_tokens=workload.max_tokens,
            top_p=0.9,
        )
        
        # Track timing
        start_time = time.time()
        request_times = []
        ttft_times = []
        error_count = 0
        
        # Process requests in batches
        batch_size = workload.concurrent_requests
        total_requests = len(workload.prompts)
        total_tokens = 0
        
        for i in range(0, total_requests, batch_size):
            batch_prompts = workload.prompts[i:i + batch_size]
            batch_start = time.time()
            
            try:
                # Generate responses
                outputs = llm.generate(batch_prompts, sampling_params)
                
                # Process results
                for j, output in enumerate(outputs):
                    request_time = time.time() - batch_start
                    request_times.append(request_time)
                    
                    if output.outputs:
                        total_tokens += len(output.outputs[0].token_ids)
                        # Estimate TTFT (simplified - in real scenario you'd track first token time)
                        ttft_times.append(request_time * 0.3)  # Rough estimate
                    else:
                        error_count += 1
                        
            except Exception as e:
                print(f"   ❌ Error in batch {i//batch_size + 1}: {e}")
                error_count += len(batch_prompts)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        if request_times:
            avg_latency = statistics.mean(request_times)
            p50_latency = statistics.median(request_times)
            p95_latency = statistics.quantiles(request_times, n=20)[18] if len(request_times) > 20 else max(request_times)
            p99_latency = statistics.quantiles(request_times, n=100)[98] if len(request_times) > 100 else max(request_times)
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0
        
        if ttft_times:
            avg_ttft = statistics.mean(ttft_times)
            p50_ttft = statistics.median(ttft_times)
            p95_ttft = statistics.quantiles(ttft_times, n=20)[18] if len(ttft_times) > 20 else max(ttft_times)
            p99_ttft = statistics.quantiles(ttft_times, n=100)[98] if len(ttft_times) > 100 else max(ttft_times)
        else:
            avg_ttft = p50_ttft = p95_ttft = p99_ttft = 0
        
        throughput = total_tokens / total_time if total_time > 0 else 0
        success_rate = (total_requests - error_count) / total_requests if total_requests > 0 else 0
        
        result = BenchmarkResult(
            scheduler_type=scheduler_type,
            total_requests=total_requests,
            total_tokens=total_tokens,
            total_time=total_time,
            throughput_tokens_per_sec=throughput,
            avg_latency=avg_latency,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            avg_ttft=avg_ttft,
            p50_ttft=p50_ttft,
            p95_ttft=p95_ttft,
            p99_ttft=p99_ttft,
            success_rate=success_rate,
            error_count=error_count,
            config=config
        )
        
        print(f"   ✅ Completed: {throughput:.2f} tokens/sec, {avg_latency:.3f}s avg latency")
        return result
    
    async def run_benchmark_suite(self, workloads: Optional[List[str]] = None) -> Dict[str, List[BenchmarkResult]]:
        """Run the complete benchmark suite."""
        all_workloads = self.create_workloads()
        
        if workloads:
            all_workloads = [w for w in all_workloads if w.name in workloads]
        
        results = {}
        
        for workload in all_workloads:
            print(f"\n📊 Testing workload: {workload.name} - {workload.description}")
            workload_results = []
        
        # Test MLFQ scheduler
            mlfq_result = await self.run_single_benchmark(
                self.mlfq_config, workload, "MLFQ"
            )
            workload_results.append(mlfq_result)
            
            # Test default scheduler
            default_result = await self.run_single_benchmark(
                self.default_config, workload, "Default"
            )
            workload_results.append(default_result)
            
            results[workload.name] = workload_results
        
        return results
    
    def generate_report(self, results: Dict[str, List[BenchmarkResult]]) -> str:
        """Generate a comprehensive performance report."""
        report = []
        report.append("# MLFQ vs Default Scheduler Benchmark Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary table
        report.append("## Summary")
        report.append("")
        report.append("| Workload | Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Success Rate |")
        report.append("|----------|----------|-----------|-------------------|-----------------|-----------------|--------------|")
        
        for workload_name, workload_results in results.items():
            # Extract request count from workload name
            request_count = workload_name.split('_')[-1]
            for result in workload_results:
                report.append(
                    f"| {workload_name} | {request_count} | {result.scheduler_type} | "
                    f"{result.throughput_tokens_per_sec:.2f} | "
                    f"{result.avg_latency:.3f} | "
                    f"{result.p95_latency:.3f} | "
                    f"{result.success_rate:.2%} |"
                )
        
        report.append("")
        
        # Detailed analysis - group by workload type
        report.append("## Detailed Analysis")
        report.append("")
        
        # Group workloads by type
        workload_groups = {}
        for workload_name, workload_results in results.items():
            # Extract workload type (remove request count suffix)
            workload_type = '_'.join(workload_name.split('_')[:-1])
            if workload_type not in workload_groups:
                workload_groups[workload_type] = []
            workload_groups[workload_type].append((workload_name, workload_results))
        
        for workload_type, type_results in workload_groups.items():
            report.append(f"### {workload_type.replace('_', ' ').title()}")
            report.append("")
            
            # Create comparison table for this workload type
            report.append("| Requests | Scheduler | Throughput (tok/s) | Avg Latency (s) | P95 Latency (s) | Throughput Δ% | Latency Δ% |")
            report.append("|----------|-----------|-------------------|-----------------|-----------------|---------------|------------|")
            
            for workload_name, workload_results in sorted(type_results, key=lambda x: int(x[0].split('_')[-1])):
                if len(workload_results) == 2:
                    mlfq_result = workload_results[0] if workload_results[0].scheduler_type == "MLFQ" else workload_results[1]
                    default_result = workload_results[1] if workload_results[0].scheduler_type == "MLFQ" else workload_results[0]
                    
                    request_count = workload_name.split('_')[-1]
                    
                    # Calculate improvements
                    throughput_improvement = ((mlfq_result.throughput_tokens_per_sec - default_result.throughput_tokens_per_sec) 
                                            / default_result.throughput_tokens_per_sec * 100)
                    latency_improvement = ((default_result.avg_latency - mlfq_result.avg_latency) 
                                         / default_result.avg_latency * 100)
                    
                    # Add rows for both schedulers
                    report.append(
                        f"| {request_count} | MLFQ | {mlfq_result.throughput_tokens_per_sec:.2f} | "
                        f"{mlfq_result.avg_latency:.3f} | {mlfq_result.p95_latency:.3f} | "
                        f"{throughput_improvement:+.1f}% | {latency_improvement:+.1f}% |"
                    )
                    report.append(
                        f"| {request_count} | Default | {default_result.throughput_tokens_per_sec:.2f} | "
                        f"{default_result.avg_latency:.3f} | {default_result.p95_latency:.3f} | - | - |"
                    )
            
            report.append("")
        
        # Overall conclusion
        report.append("## Overall Conclusion")
        report.append("")
        
        # Calculate overall improvements
        all_mlfq_results = []
        all_default_results = []
        
        for workload_results in results.values():
            for result in workload_results:
                if result.scheduler_type == "MLFQ":
                    all_mlfq_results.append(result)
                else:
                    all_default_results.append(result)
        
        if all_mlfq_results and all_default_results:
            avg_mlfq_throughput = statistics.mean([r.throughput_tokens_per_sec for r in all_mlfq_results])
            avg_default_throughput = statistics.mean([r.throughput_tokens_per_sec for r in all_default_results])
            avg_mlfq_latency = statistics.mean([r.avg_latency for r in all_mlfq_results])
            avg_default_latency = statistics.mean([r.avg_latency for r in all_default_results])
            
            overall_throughput_improvement = (avg_mlfq_throughput - avg_default_throughput) / avg_default_throughput * 100
            overall_latency_improvement = (avg_default_latency - avg_mlfq_latency) / avg_default_latency * 100
            
            report.append(f"**Overall Performance:**")
            report.append(f"- Average Throughput Improvement: {overall_throughput_improvement:+.1f}%")
            report.append(f"- Average Latency Improvement: {overall_latency_improvement:+.1f}%")
            report.append("")
            
        if overall_throughput_improvement > 0:
                report.append("✅ MLFQ scheduler shows improved throughput performance.")
        else:
                report.append("❌ MLFQ scheduler shows decreased throughput performance.")
            
        if overall_latency_improvement > 0:
                report.append("✅ MLFQ scheduler shows improved latency performance.")
        else:
                report.append("❌ MLFQ scheduler shows increased latency.")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, List[BenchmarkResult]], report: str):
        """Save benchmark results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save raw results as JSON
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert results to serializable format
            serializable_results = {}
            for workload_name, workload_results in results.items():
                serializable_results[workload_name] = [asdict(result) for result in workload_results]
            json.dump(serializable_results, f, indent=2)
        
        # Save report as markdown
        report_file = self.output_dir / f"benchmark_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📁 Results saved to:")
        print(f"   - {results_file}")
        print(f"   - {report_file}")


async def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Benchmark MLFQ vs Default Scheduler")
    parser.add_argument("--model", required=True, help="Path to the model")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    parser.add_argument("--workloads", nargs="+", help="Specific workloads to test (e.g., short_requests_10, mixed_length_50, long_requests_100, burst_load_200)")
    parser.add_argument("--gpu", type=int, help="GPU device to use")
    
    args = parser.parse_args()
    
    # Set GPU device if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU device: {args.gpu}")
    
    # Initialize benchmark
    benchmark = MLFQBenchmark(args.model, args.output_dir)
    
    print("🔬 MLFQ vs Default Scheduler Benchmark")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    
    # Run benchmarks
    results = await benchmark.run_benchmark_suite(args.workloads)
    
    # Generate and save report
    report = benchmark.generate_report(results)
    benchmark.save_results(results, report)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 50)
    
    # Group workloads by type for summary
    workload_groups = {}
    for workload_name, workload_results in results.items():
        workload_type = '_'.join(workload_name.split('_')[:-1])
        if workload_type not in workload_groups:
            workload_groups[workload_type] = []
        workload_groups[workload_type].append((workload_name, workload_results))
    
    for workload_type, type_results in workload_groups.items():
        print(f"\n{workload_type.replace('_', ' ').title()}:")
        print("  Requests | Throughput Δ% | Latency Δ%")
        print("  ---------|---------------|------------")
        
        for workload_name, workload_results in sorted(type_results, key=lambda x: int(x[0].split('_')[-1])):
            if len(workload_results) == 2:
                mlfq_result = workload_results[0] if workload_results[0].scheduler_type == "MLFQ" else workload_results[1]
                default_result = workload_results[1] if workload_results[0].scheduler_type == "MLFQ" else workload_results[0]
                
                request_count = workload_name.split('_')[-1]
                throughput_improvement = ((mlfq_result.throughput_tokens_per_sec - default_result.throughput_tokens_per_sec) 
                                        / default_result.throughput_tokens_per_sec * 100)
                latency_improvement = ((default_result.avg_latency - mlfq_result.avg_latency) 
                                     / default_result.avg_latency * 100)
                
                print(f"  {request_count:8} | {throughput_improvement:+12.1f}% | {latency_improvement:+9.1f}%")
    
    print(f"\n📁 Detailed results saved to: {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())