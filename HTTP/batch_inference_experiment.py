#!/usr/bin/env python3
"""
Batch Inference Experiment Script for vLLM

This script allows you to experiment with different parallel configurations
(tensor parallel, pipeline parallel, data parallel) and batch inference
parameters to test performance and scalability.

Usage Examples:
    # Basic tensor parallel experiment
    python batch_inference_experiment.py \
        --model "meta-llama/Llama-2-7b-chat-hf" \
        --tensor-parallel-size 2 \
        --batch-sizes 1,4,8,16 \
        --seq-lens 512,1024,2048

    # Tensor + Pipeline parallel experiment
    python batch_inference_experiment.py \
        --model "meta-llama/Llama-2-7b-chat-hf" \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 2 \
        --batch-sizes 1,2,4,8 \
        --seq-lens 512,1024

    # Data parallel experiment
    python batch_inference_experiment.py \
        --model "meta-llama/Llama-2-7b-chat-hf" \
        --data-parallel-size 2 \
        --batch-sizes 1,4,8,16 \
        --seq-lens 512,1024

    # Multi-node experiment (run on each node)
    python batch_inference_experiment.py \
        --model "meta-llama/Llama-2-7b-chat-hf" \
        --tensor-parallel-size 4 \
        --pipeline-parallel-size 2 \
        --data-parallel-size 2 \
        --node-rank 0 \
        --node-size 2 \
        --master-addr "192.168.1.100" \
        --master-port 29500
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import statistics
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import get_open_port


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    # Parallel configuration
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    
    # Batch configuration
    batch_size: int = 1
    seq_len: int = 512
    
    # Model configuration
    model: str = "meta-llama/Llama-2-7b-chat-hf"
    max_model_len: int = 4096
    dtype: str = "auto"  # type: ignore
    
    # Performance configuration
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.8
    
    # Sampling configuration
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 128
    
    # Distributed configuration
    distributed_executor_backend: str = "ray"
    node_rank: int = 0
    node_size: int = 1
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    
    # Experiment configuration
    num_runs: int = 3
    warmup_runs: int = 1
    trust_remote_code: bool = False
    enforce_eager: bool = False
    seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float
    latency_ms: float
    memory_usage_gb: float
    gpu_utilization: float
    run_time_seconds: float
    tokens_generated: int
    requests_processed: int


class BatchInferenceExperiment:
    """Main class for running batch inference experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.llm = None
        self.results: List[ExperimentResult] = []
        
    def setup_distributed_environment(self):
        """Setup distributed environment if needed."""
        if self.config.data_parallel_size > 1:
            if self.config.master_addr is None:
                self.config.master_addr = "127.0.0.1"
            if self.config.master_port is None:
                self.config.master_port = get_open_port()
                
            # Set environment variables for data parallel
            os.environ["VLLM_DP_RANK"] = str(self.config.node_rank)
            os.environ["VLLM_DP_RANK_LOCAL"] = str(self.config.node_rank)
            os.environ["VLLM_DP_SIZE"] = str(self.config.data_parallel_size)
            os.environ["VLLM_DP_MASTER_IP"] = self.config.master_addr
            os.environ["VLLM_DP_MASTER_PORT"] = str(self.config.master_port)
    
    def initialize_llm(self):
        """Initialize the vLLM LLM instance."""
        print(f"Initializing LLM with config: {self.config}")
        
        # Calculate total GPUs needed
        total_gpus = (self.config.tensor_parallel_size * 
                     self.config.pipeline_parallel_size * 
                     self.config.data_parallel_size)
        
        print(f"Total GPUs required: {total_gpus}")
        print(f"Parallel config: TP={self.config.tensor_parallel_size}, "
              f"PP={self.config.pipeline_parallel_size}, "
              f"DP={self.config.data_parallel_size}")
        
        self.llm = LLM(
            model=self.config.model,
            tensor_parallel_size=self.config.tensor_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            distributed_executor_backend=self.config.distributed_executor_backend,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype=self.config.dtype,
            trust_remote_code=self.config.trust_remote_code,
            enforce_eager=self.config.enforce_eager,
            seed=self.config.seed,
        )
        
        print("LLM initialized successfully!")
    
    def generate_prompts(self, batch_size: int, seq_len: int) -> List[str]:
        """Generate test prompts for the experiment."""
        # Create diverse prompts with different lengths
        base_prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
            "Machine learning is a subset of",
            "The best way to learn programming is",
            "Climate change is caused by",
            "The internet was invented by",
            "Python is a programming language that",
            "Artificial intelligence will",
        ]
        
        # Repeat prompts to reach batch size
        prompts = []
        for i in range(batch_size):
            prompt = base_prompts[i % len(base_prompts)]
            # Add some variation to make prompts different lengths
            if seq_len > 50:
                # Extend prompt to target length
                extension = f" and this is additional text to reach the target sequence length of {seq_len} tokens. " * (seq_len // 50)
                prompt += extension[:seq_len * 4]  # Rough approximation
            prompts.append(prompt)
        
        return prompts
    
    def measure_gpu_metrics(self) -> Dict[str, float]:
        """Measure GPU utilization and memory usage."""
        try:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
                return {
                    "memory_allocated_gb": memory_allocated,
                    "memory_reserved_gb": memory_reserved,
                    "gpu_utilization": 0.0  # Would need nvidia-ml-py for actual utilization
                }
            else:
                return {"memory_allocated_gb": 0.0, "memory_reserved_gb": 0.0, "gpu_utilization": 0.0}
        except Exception as e:
            print(f"Warning: Could not measure GPU metrics: {e}")
            return {"memory_allocated_gb": 0.0, "memory_reserved_gb": 0.0, "gpu_utilization": 0.0}
    
    def run_single_experiment(self, batch_size: int, seq_len: int) -> ExperimentResult:
        """Run a single experiment with given batch size and sequence length."""
        print(f"\nRunning experiment: batch_size={batch_size}, seq_len={seq_len}")
        
        # Generate prompts
        prompts = self.generate_prompts(batch_size, seq_len)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
        
        # Warmup runs
        for i in range(self.config.warmup_runs):
            print(f"  Warmup run {i+1}/{self.config.warmup_runs}")
            _ = self.llm.generate(prompts, sampling_params)
        
        # Actual experiment runs
        run_times = []
        total_tokens = 0
        total_requests = 0
        
        for run in range(self.config.num_runs):
            print(f"  Run {run+1}/{self.config.num_runs}")
            
            # Measure GPU metrics before
            gpu_metrics_before = self.measure_gpu_metrics()
            
            # Run inference
            start_time = time.time()
            outputs = self.llm.generate(prompts, sampling_params)
            end_time = time.time()
            
            # Measure GPU metrics after
            gpu_metrics_after = self.measure_gpu_metrics()
            
            run_time = end_time - start_time
            run_times.append(run_time)
            
            # Count tokens and requests
            for output in outputs:
                total_tokens += len(output.outputs[0].token_ids)
                total_requests += 1
            
            print(f"    Run time: {run_time:.3f}s")
        
        # Calculate statistics
        avg_run_time = statistics.mean(run_times)
        total_tokens_avg = int(total_tokens / self.config.num_runs)
        total_requests_avg = int(total_requests / self.config.num_runs)
        
        # Calculate throughput
        throughput_tokens_per_sec = total_tokens_avg / avg_run_time
        throughput_requests_per_sec = total_requests_avg / avg_run_time
        latency_ms = avg_run_time * 1000
        
        # Use average memory usage
        memory_usage_gb = (gpu_metrics_before["memory_allocated_gb"] + 
                          gpu_metrics_after["memory_allocated_gb"]) / 2
        
        # Create a new config instance with the specific batch_size and seq_len for this experiment
        experiment_config = ExperimentConfig(
            tensor_parallel_size=self.config.tensor_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            data_parallel_size=self.config.data_parallel_size,
            batch_size=batch_size,
            seq_len=seq_len,
            model=self.config.model,
            max_model_len=self.config.max_model_len,
            dtype=self.config.dtype,  # type: ignore
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            distributed_executor_backend=self.config.distributed_executor_backend,
            node_rank=self.config.node_rank,
            node_size=self.config.node_size,
            master_addr=self.config.master_addr,
            master_port=self.config.master_port,
            num_runs=self.config.num_runs,
            warmup_runs=self.config.warmup_runs,
            trust_remote_code=self.config.trust_remote_code,
            enforce_eager=self.config.enforce_eager,
            seed=self.config.seed,
        )
        
        result = ExperimentResult(
            config=experiment_config,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            throughput_requests_per_sec=throughput_requests_per_sec,
            latency_ms=latency_ms,
            memory_usage_gb=memory_usage_gb,
            gpu_utilization=gpu_metrics_after["gpu_utilization"],
            run_time_seconds=avg_run_time,
            tokens_generated=total_tokens_avg,
            requests_processed=total_requests_avg,
        )
        
        print(f"  Results: {throughput_tokens_per_sec:.2f} tokens/sec, "
              f"{throughput_requests_per_sec:.2f} requests/sec, "
              f"{latency_ms:.2f}ms latency")
        
        return result
    
    def run_experiments(self, batch_sizes: List[int], seq_lens: List[int]):
        """Run experiments for all combinations of batch sizes and sequence lengths."""
        print(f"\nStarting experiments with {len(batch_sizes)} batch sizes and {len(seq_lens)} sequence lengths")
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                try:
                    result = self.run_single_experiment(batch_size, seq_len)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error in experiment batch_size={batch_size}, seq_len={seq_len}: {e}")
                    continue
    
    def save_results(self, output_file: str):
        """Save experiment results to a JSON file."""
        results_data = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert config to dict for JSON serialization
            result_dict["config"] = asdict(result.config)
            results_data.append(result_dict)
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def print_summary(self):
        """Print a summary of all experiment results."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        # Group results by configuration
        config_groups = {}
        for result in self.results:
            key = (result.config.tensor_parallel_size, 
                   result.config.pipeline_parallel_size, 
                   result.config.data_parallel_size)
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(result)
        
        for (tp, pp, dp), results in config_groups.items():
            print(f"\nParallel Config: TP={tp}, PP={pp}, DP={dp}")
            print("-" * 60)
            print(f"{'Batch':<6} {'Seq Len':<8} {'Tokens/sec':<12} {'Requests/sec':<15} {'Latency(ms)':<12} {'Memory(GB)':<12}")
            print("-" * 60)
            
            for result in sorted(results, key=lambda r: (r.config.batch_size, r.config.seq_len)):
                print(f"{result.config.batch_size:<6} {result.config.seq_len:<8} "
                      f"{result.throughput_tokens_per_sec:<12.2f} "
                      f"{result.throughput_requests_per_sec:<15.2f} "
                      f"{result.latency_ms:<12.2f} "
                      f"{result.memory_usage_gb:<12.2f}")
        
        print("\n" + "="*80)


def parse_batch_sizes(batch_sizes_str: str) -> List[int]:
    """Parse batch sizes from comma-separated string."""
    return [int(x.strip()) for x in batch_sizes_str.split(',')]


def parse_seq_lens(seq_lens_str: str) -> List[int]:
    """Parse sequence lengths from comma-separated string."""
    return [int(x.strip()) for x in seq_lens_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Batch Inference Experiment Script for vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Parallel configuration
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1,
                       help="Pipeline parallel size")
    parser.add_argument("--data-parallel-size", "-dp", type=int, default=1,
                       help="Data parallel size")
    
    # Batch configuration
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16",
                       help="Comma-separated list of batch sizes to test")
    parser.add_argument("--seq-lens", type=str, default="512,1024,2048",
                       help="Comma-separated list of sequence lengths to test")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                       help="Model name or path")
    parser.add_argument("--max-model-len", type=int, default=4096,
                       help="Maximum model length")
    parser.add_argument("--dtype", type=str, default="auto",
                       help="Model data type")
    
    # Performance configuration
    parser.add_argument("--max-num-seqs", type=int, default=256,
                       help="Maximum number of sequences")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192,
                       help="Maximum number of batched tokens")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                       help="GPU memory utilization")
    
    # Sampling configuration
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                       help="Top-p sampling parameter")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Maximum tokens to generate")
    
    # Distributed configuration
    parser.add_argument("--distributed-executor-backend", type=str, default="ray",
                       choices=["ray", "mp", "external_launcher"],
                       help="Distributed executor backend")
    parser.add_argument("--node-rank", type=int, default=0,
                       help="Node rank for distributed setup")
    parser.add_argument("--node-size", type=int, default=1,
                       help="Total number of nodes")
    parser.add_argument("--master-addr", type=str, default=None,
                       help="Master node address")
    parser.add_argument("--master-port", type=int, default=None,
                       help="Master node port")
    
    # Experiment configuration
    parser.add_argument("--num-runs", type=int, default=3,
                       help="Number of runs per experiment")
    parser.add_argument("--warmup-runs", type=int, default=1,
                       help="Number of warmup runs")
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code")
    parser.add_argument("--enforce-eager", action="store_true",
                       help="Enforce eager mode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Output configuration
    parser.add_argument("--output-file", type=str, default="batch_inference_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Parse batch sizes and sequence lengths
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    seq_lens = parse_seq_lens(args.seq_lens)
    
    # Create experiment configuration
    config = ExperimentConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        model=args.model,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        distributed_executor_backend=args.distributed_executor_backend,
        node_rank=args.node_rank,
        node_size=args.node_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    
    # Create and run experiment
    experiment = BatchInferenceExperiment(config)
    
    try:
        # Setup distributed environment
        experiment.setup_distributed_environment()
        
        # Initialize LLM
        experiment.initialize_llm()
        
        # Run experiments
        experiment.run_experiments(batch_sizes, seq_lens)
        
        # Print summary
        experiment.print_summary()
        
        # Save results
        experiment.save_results(args.output_file)
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        raise
    finally:
        # Cleanup
        if experiment.llm is not None:
            del experiment.llm


if __name__ == "__main__":
    main() 