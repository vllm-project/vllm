#!/usr/bin/env python3
"""
Advanced vLLM Profiling Analyzer

This module provides detailed analysis of vLLM scheduling behavior and performance
characteristics for different workload configurations.

Features:
- Detailed scheduling analysis
- Resource utilization tracking
- Performance bottleneck identification
- Custom schedule plan generation
- Integration with existing profiling code
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque
import logging
import psutil
import torch

from profiling_workload_generator import ProfilingConfig, ProfilingResult, ProfilingRunner, create_profiling_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SchedulingMetrics:
    """Detailed scheduling metrics for analysis."""
    
    # Timing breakdowns
    prefill_compute_time: List[float]  # ms
    decode_compute_time: List[float]   # ms
    scheduling_overhead: List[float]   # ms
    memory_allocation_time: List[float]  # ms
    
    # Resource utilization
    gpu_memory_usage: List[float]  # MB
    cpu_memory_usage: List[float]  # MB
    gpu_utilization: List[float]   # %
    
    # Scheduling statistics
    batch_efficiency: List[float]  # tokens processed / batch capacity
    cache_hit_rate: List[float]    # cache hits / total accesses
    preemption_count: int = 0
    swap_count: int = 0
    
    # Queue statistics
    waiting_queue_size: List[int]
    running_queue_size: List[int]
    swapped_queue_size: List[int]
    
    # Throughput metrics
    tokens_per_second: List[float]
    requests_per_second: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PerformanceAnalysis:
    """Comprehensive performance analysis results."""
    
    # Configuration
    config: ProfilingConfig
    
    # Raw metrics
    scheduling_metrics: SchedulingMetrics
    
    # Analysis results
    bottleneck_analysis: Dict[str, Any]
    optimization_recommendations: List[str]
    performance_summary: Dict[str, float]
    
    # Custom schedule plans
    recommended_schedule_plans: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'config': asdict(self.config),
            'scheduling_metrics': self.scheduling_metrics.to_dict(),
            'bottleneck_analysis': self.bottleneck_analysis,
            'optimization_recommendations': self.optimization_recommendations,
            'performance_summary': self.performance_summary,
            'recommended_schedule_plans': self.recommended_schedule_plans,
        }


class AdvancedProfilingAnalyzer:
    """Advanced analyzer for vLLM profiling with detailed scheduling analysis."""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.metrics = SchedulingMetrics(
            prefill_compute_time=[],
            decode_compute_time=[],
            scheduling_overhead=[],
            memory_allocation_time=[],
            gpu_memory_usage=[],
            cpu_memory_usage=[],
            gpu_utilization=[],
            batch_efficiency=[],
            cache_hit_rate=[],
            waiting_queue_size=[],
            running_queue_size=[],
            swapped_queue_size=[],
            tokens_per_second=[],
            requests_per_second=[],
        )
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.measurement_intervals = []
        
    def start_profiling(self):
        """Start profiling session."""
        self.start_time = time.time()
        logger.info("Started advanced profiling session")
        
    def end_profiling(self):
        """End profiling session."""
        self.end_time = time.time()
        logger.info(f"Ended profiling session. Total time: {self.end_time - self.start_time:.2f}s")
        
    def record_metrics(self, interval_metrics: Dict[str, Any]):
        """Record metrics for a measurement interval."""
        self.measurement_intervals.append(interval_metrics)
        
        # Extract metrics
        if 'prefill_compute_time' in interval_metrics:
            self.metrics.prefill_compute_time.append(interval_metrics['prefill_compute_time'])
        if 'decode_compute_time' in interval_metrics:
            self.metrics.decode_compute_time.append(interval_metrics['decode_compute_time'])
        if 'scheduling_overhead' in interval_metrics:
            self.metrics.scheduling_overhead.append(interval_metrics['scheduling_overhead'])
        if 'memory_allocation_time' in interval_metrics:
            self.metrics.memory_allocation_time.append(interval_metrics['memory_allocation_time'])
            
        # Resource utilization
        if 'gpu_memory_usage' in interval_metrics:
            self.metrics.gpu_memory_usage.append(interval_metrics['gpu_memory_usage'])
        if 'cpu_memory_usage' in interval_metrics:
            self.metrics.cpu_memory_usage.append(interval_metrics['cpu_memory_usage'])
        if 'gpu_utilization' in interval_metrics:
            self.metrics.gpu_utilization.append(interval_metrics['gpu_utilization'])
            
        # Scheduling statistics
        if 'batch_efficiency' in interval_metrics:
            self.metrics.batch_efficiency.append(interval_metrics['batch_efficiency'])
        if 'cache_hit_rate' in interval_metrics:
            self.metrics.cache_hit_rate.append(interval_metrics['cache_hit_rate'])
        if 'preemption_count' in interval_metrics:
            self.metrics.preemption_count += interval_metrics['preemption_count']
        if 'swap_count' in interval_metrics:
            self.metrics.swap_count += interval_metrics['swap_count']
            
        # Queue statistics
        if 'waiting_queue_size' in interval_metrics:
            self.metrics.waiting_queue_size.append(interval_metrics['waiting_queue_size'])
        if 'running_queue_size' in interval_metrics:
            self.metrics.running_queue_size.append(interval_metrics['running_queue_size'])
        if 'swapped_queue_size' in interval_metrics:
            self.metrics.swapped_queue_size.append(interval_metrics['swapped_queue_size'])
            
        # Throughput
        if 'tokens_per_second' in interval_metrics:
            self.metrics.tokens_per_second.append(interval_metrics['tokens_per_second'])
        if 'requests_per_second' in interval_metrics:
            self.metrics.requests_per_second.append(interval_metrics['requests_per_second'])
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        analysis = {
            'compute_bound': False,
            'memory_bound': False,
            'scheduling_bound': False,
            'cache_bound': False,
            'bottleneck_details': {},
        }
        
        # Check if compute-bound
        if self.metrics.prefill_compute_time and self.metrics.decode_compute_time:
            avg_prefill_time = np.mean(self.metrics.prefill_compute_time)
            avg_decode_time = np.mean(self.metrics.decode_compute_time)
            
            if avg_prefill_time > 100 or avg_decode_time > 50:  # Thresholds in ms
                analysis['compute_bound'] = True
                analysis['bottleneck_details']['compute'] = {
                    'avg_prefill_time_ms': avg_prefill_time,
                    'avg_decode_time_ms': avg_decode_time,
                }
        
        # Check if memory-bound
        if self.metrics.gpu_memory_usage:
            avg_gpu_memory = np.mean(self.metrics.gpu_memory_usage)
            max_gpu_memory = max(self.metrics.gpu_memory_usage)
            
            # Assume 24GB GPU for threshold (adjust as needed)
            if avg_gpu_memory > 20000 or max_gpu_memory > 22000:  # MB
                analysis['memory_bound'] = True
                analysis['bottleneck_details']['memory'] = {
                    'avg_gpu_memory_mb': avg_gpu_memory,
                    'max_gpu_memory_mb': max_gpu_memory,
                }
        
        # Check if scheduling-bound
        if self.metrics.scheduling_overhead:
            avg_scheduling_overhead = np.mean(self.metrics.scheduling_overhead)
            if avg_scheduling_overhead > 10:  # ms
                analysis['scheduling_bound'] = True
                analysis['bottleneck_details']['scheduling'] = {
                    'avg_scheduling_overhead_ms': avg_scheduling_overhead,
                }
        
        # Check if cache-bound
        if self.metrics.cache_hit_rate:
            avg_cache_hit_rate = np.mean(self.metrics.cache_hit_rate)
            if avg_cache_hit_rate < 0.7:  # 70% threshold
                analysis['cache_bound'] = True
                analysis['bottleneck_details']['cache'] = {
                    'avg_cache_hit_rate': avg_cache_hit_rate,
                }
        
        return analysis
    
    def generate_optimization_recommendations(self, bottleneck_analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on bottleneck analysis."""
        recommendations = []
        
        if bottleneck_analysis['compute_bound']:
            recommendations.extend([
                "Consider using a smaller model or quantization",
                "Increase batch size to improve GPU utilization",
                "Enable chunked prefill for better batching",
                "Use tensor parallelism for large models",
            ])
        
        if bottleneck_analysis['memory_bound']:
            recommendations.extend([
                "Reduce max_num_batched_tokens",
                "Enable CPU offloading with swap_space",
                "Reduce batch size",
                "Use gradient checkpointing",
                "Consider using a smaller model",
            ])
        
        if bottleneck_analysis['scheduling_bound']:
            recommendations.extend([
                "Reduce max_num_seqs",
                "Use simpler scheduling policy",
                "Disable chunked prefill",
                "Optimize request arrival patterns",
            ])
        
        if bottleneck_analysis['cache_bound']:
            recommendations.extend([
                "Increase block_size for better cache efficiency",
                "Enable prefix caching",
                "Optimize KV cache management",
                "Reduce sequence length",
            ])
        
        # General recommendations
        if not any([bottleneck_analysis['compute_bound'], 
                   bottleneck_analysis['memory_bound'],
                   bottleneck_analysis['scheduling_bound'],
                   bottleneck_analysis['cache_bound']]):
            recommendations.append("Performance appears balanced. Consider fine-tuning for specific use case.")
        
        return recommendations
    
    def generate_custom_schedule_plans(self) -> List[Dict[str, Any]]:
        """Generate custom schedule plans based on analysis."""
        plans = []
        
        # Plan 1: High throughput configuration
        high_throughput_plan = {
            'name': 'High Throughput',
            'description': 'Optimized for maximum tokens per second',
            'config': {
                'max_num_batched_tokens': self.config.max_num_batched_tokens * 2,
                'max_num_seqs': self.config.max_num_seqs * 2,
                'enable_chunked_prefill': True,
                'max_num_partial_prefills': 8,
                'scheduling_policy': 'fcfs',
                'block_size': max(16, self.config.block_size),
            },
            'expected_improvement': '20-30% throughput increase',
        }
        plans.append(high_throughput_plan)
        
        # Plan 2: Low latency configuration
        low_latency_plan = {
            'name': 'Low Latency',
            'description': 'Optimized for minimum response time',
            'config': {
                'max_num_batched_tokens': self.config.max_num_batched_tokens // 2,
                'max_num_seqs': self.config.max_num_seqs // 2,
                'enable_chunked_prefill': True,
                'max_num_partial_prefills': 2,
                'scheduling_policy': 'priority',
                'block_size': min(32, self.config.block_size * 2),
            },
            'expected_improvement': '30-50% latency reduction',
        }
        plans.append(low_latency_plan)
        
        # Plan 3: Memory efficient configuration
        memory_efficient_plan = {
            'name': 'Memory Efficient',
            'description': 'Optimized for low memory usage',
            'config': {
                'max_num_batched_tokens': self.config.max_num_batched_tokens // 4,
                'max_num_seqs': self.config.max_num_seqs // 4,
                'enable_chunked_prefill': False,
                'max_num_partial_prefills': 1,
                'scheduling_policy': 'fcfs',
                'block_size': self.config.block_size,
                'gpu_memory_utilization': 0.7,
                'swap_space': 8.0,
            },
            'expected_improvement': '40-60% memory reduction',
        }
        plans.append(memory_efficient_plan)
        
        # Plan 4: Balanced configuration
        balanced_plan = {
            'name': 'Balanced',
            'description': 'Balanced throughput and latency',
            'config': {
                'max_num_batched_tokens': int(self.config.max_num_batched_tokens * 1.2),
                'max_num_seqs': int(self.config.max_num_seqs * 1.2),
                'enable_chunked_prefill': True,
                'max_num_partial_prefills': 4,
                'scheduling_policy': 'fcfs',
                'block_size': self.config.block_size,
            },
            'expected_improvement': '10-15% overall improvement',
        }
        plans.append(balanced_plan)
        
        return plans
    
    def calculate_performance_summary(self) -> Dict[str, float]:
        """Calculate comprehensive performance summary."""
        summary = {}
        
        # Latency statistics
        if self.metrics.prefill_compute_time:
            summary['avg_prefill_latency_ms'] = np.mean(self.metrics.prefill_compute_time)
            summary['p95_prefill_latency_ms'] = np.percentile(self.metrics.prefill_compute_time, 95)
            summary['p99_prefill_latency_ms'] = np.percentile(self.metrics.prefill_compute_time, 99)
        
        if self.metrics.decode_compute_time:
            summary['avg_decode_latency_ms'] = np.mean(self.metrics.decode_compute_time)
            summary['p95_decode_latency_ms'] = np.percentile(self.metrics.decode_compute_time, 95)
            summary['p99_decode_latency_ms'] = np.percentile(self.metrics.decode_compute_time, 99)
        
        # Throughput statistics
        if self.metrics.tokens_per_second:
            summary['avg_tokens_per_second'] = np.mean(self.metrics.tokens_per_second)
            summary['max_tokens_per_second'] = max(self.metrics.tokens_per_second)
        
        if self.metrics.requests_per_second:
            summary['avg_requests_per_second'] = np.mean(self.metrics.requests_per_second)
            summary['max_requests_per_second'] = max(self.metrics.requests_per_second)
        
        # Resource utilization
        if self.metrics.gpu_memory_usage:
            summary['avg_gpu_memory_mb'] = np.mean(self.metrics.gpu_memory_usage)
            summary['max_gpu_memory_mb'] = max(self.metrics.gpu_memory_usage)
        
        if self.metrics.gpu_utilization:
            summary['avg_gpu_utilization_percent'] = np.mean(self.metrics.gpu_utilization)
            summary['max_gpu_utilization_percent'] = max(self.metrics.gpu_utilization)
        
        # Efficiency metrics
        if self.metrics.batch_efficiency:
            summary['avg_batch_efficiency'] = np.mean(self.metrics.batch_efficiency)
        
        if self.metrics.cache_hit_rate:
            summary['avg_cache_hit_rate'] = np.mean(self.metrics.cache_hit_rate)
        
        # Scheduling metrics
        summary['total_preemptions'] = self.metrics.preemption_count
        summary['total_swaps'] = self.metrics.swap_count
        
        return summary
    
    def create_analysis_report(self) -> PerformanceAnalysis:
        """Create comprehensive performance analysis report."""
        # Analyze bottlenecks
        bottleneck_analysis = self.analyze_bottlenecks()
        
        # Generate recommendations
        optimization_recommendations = self.generate_optimization_recommendations(bottleneck_analysis)
        
        # Calculate performance summary
        performance_summary = self.calculate_performance_summary()
        
        # Generate custom schedule plans
        recommended_schedule_plans = self.generate_custom_schedule_plans()
        
        return PerformanceAnalysis(
            config=self.config,
            scheduling_metrics=self.metrics,
            bottleneck_analysis=bottleneck_analysis,
            optimization_recommendations=optimization_recommendations,
            performance_summary=performance_summary,
            recommended_schedule_plans=recommended_schedule_plans,
        )
    
    def save_analysis_report(self, analysis: PerformanceAnalysis, filename: str = "performance_analysis.json"):
        """Save analysis report to file."""
        with open(filename, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2)
        logger.info(f"Analysis report saved to {filename}")
    
    def plot_performance_metrics(self, save_plots: bool = True):
        """Create performance visualization plots."""
        if not self.measurement_intervals:
            logger.warning("No measurement data available for plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('vLLM Performance Analysis', fontsize=16)
        
        # Plot 1: Latency over time
        if self.metrics.prefill_compute_time:
            axes[0, 0].plot(self.metrics.prefill_compute_time, label='Prefill', marker='o')
        if self.metrics.decode_compute_time:
            axes[0, 0].plot(self.metrics.decode_compute_time, label='Decode', marker='s')
        axes[0, 0].set_title('Latency Over Time')
        axes[0, 0].set_xlabel('Measurement Interval')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Throughput over time
        if self.metrics.tokens_per_second:
            axes[0, 1].plot(self.metrics.tokens_per_second, label='Tokens/sec', marker='o', color='green')
        if self.metrics.requests_per_second:
            axes[0, 1].twinx().plot(self.metrics.requests_per_second, label='Requests/sec', marker='s', color='red')
        axes[0, 1].set_title('Throughput Over Time')
        axes[0, 1].set_xlabel('Measurement Interval')
        axes[0, 1].set_ylabel('Tokens per Second')
        axes[0, 1].grid(True)
        
        # Plot 3: Memory usage
        if self.metrics.gpu_memory_usage:
            axes[0, 2].plot(self.metrics.gpu_memory_usage, label='GPU Memory', marker='o', color='blue')
        if self.metrics.cpu_memory_usage:
            axes[0, 2].plot(self.metrics.cpu_memory_usage, label='CPU Memory', marker='s', color='orange')
        axes[0, 2].set_title('Memory Usage Over Time')
        axes[0, 2].set_xlabel('Measurement Interval')
        axes[0, 2].set_ylabel('Memory (MB)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Plot 4: Queue sizes
        if self.metrics.waiting_queue_size:
            axes[1, 0].plot(self.metrics.waiting_queue_size, label='Waiting', marker='o')
        if self.metrics.running_queue_size:
            axes[1, 0].plot(self.metrics.running_queue_size, label='Running', marker='s')
        if self.metrics.swapped_queue_size:
            axes[1, 0].plot(self.metrics.swapped_queue_size, label='Swapped', marker='^')
        axes[1, 0].set_title('Queue Sizes Over Time')
        axes[1, 0].set_xlabel('Measurement Interval')
        axes[1, 0].set_ylabel('Queue Size')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 5: Efficiency metrics
        if self.metrics.batch_efficiency:
            axes[1, 1].plot(self.metrics.batch_efficiency, label='Batch Efficiency', marker='o', color='purple')
        if self.metrics.cache_hit_rate:
            axes[1, 1].twinx().plot(self.metrics.cache_hit_rate, label='Cache Hit Rate', marker='s', color='brown')
        axes[1, 1].set_title('Efficiency Metrics')
        axes[1, 1].set_xlabel('Measurement Interval')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].grid(True)
        
        # Plot 6: GPU utilization
        if self.metrics.gpu_utilization:
            axes[1, 2].plot(self.metrics.gpu_utilization, label='GPU Utilization', marker='o', color='red')
        axes[1, 2].set_title('GPU Utilization Over Time')
        axes[1, 2].set_xlabel('Measurement Interval')
        axes[1, 2].set_ylabel('Utilization (%)')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('vllm_performance_analysis.png', dpi=300, bbox_inches='tight')
            logger.info("Performance plots saved to vllm_performance_analysis.png")
        
        plt.show()


class EnhancedProfilingRunner(ProfilingRunner):
    """Enhanced profiling runner with detailed metrics collection."""
    
    def __init__(self, engine, config: ProfilingConfig):
        super().__init__(engine, config)
        self.analyzer = AdvancedProfilingAnalyzer(config)
        
    def run_enhanced_profiling(self, profiling_type: str = "mixed") -> Tuple[ProfilingResult, PerformanceAnalysis]:
        """Run enhanced profiling with detailed analysis."""
        self.analyzer.start_profiling()
        
        # Run the original profiling
        if profiling_type == "prefill":
            result = self.run_prefill_profiling()
        elif profiling_type == "decode":
            result = self.run_decode_profiling()
        else:  # mixed
            result = self.run_mixed_profiling()
        
        # Collect additional metrics during profiling
        self._collect_detailed_metrics()
        
        self.analyzer.end_profiling()
        
        # Create analysis report
        analysis = self.analyzer.create_analysis_report()
        
        return result, analysis
    
    def _collect_detailed_metrics(self):
        """Collect detailed metrics during profiling."""
        # This is a simplified version - you can enhance this with your existing profiling code
        for i in range(len(self.prefill_latencies)):
            interval_metrics = {
                'prefill_compute_time': self.prefill_latencies[i] if i < len(self.prefill_latencies) else 0,
                'decode_compute_time': self.decode_latencies[i] if i < len(self.decode_latencies) else 0,
                'tokens_per_second': self.config.C * self.config.B / (self.total_latencies[i] / 1000) if i < len(self.total_latencies) else 0,
                'requests_per_second': self.config.B / (self.total_latencies[i] / 1000) if i < len(self.total_latencies) else 0,
                'batch_efficiency': 0.8,  # Placeholder - calculate from actual data
                'cache_hit_rate': 0.9,    # Placeholder - calculate from actual data
                'gpu_memory_usage': 8000,  # Placeholder - get from actual monitoring
                'cpu_memory_usage': 4000,  # Placeholder - get from actual monitoring
                'gpu_utilization': 85,     # Placeholder - get from actual monitoring
                'waiting_queue_size': 0,   # Placeholder - get from scheduler
                'running_queue_size': self.config.B,  # Placeholder - get from scheduler
                'swapped_queue_size': 0,   # Placeholder - get from scheduler
            }
            self.analyzer.record_metrics(interval_metrics)


def main():
    """Main function for running enhanced profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced vLLM Profiling Analyzer")
    
    # Core parameters
    parser.add_argument("--C", type=int, required=True, help="Number of input tokens/prompt")
    parser.add_argument("--M", type=int, required=True, help="Number of KV Cache blocks")
    parser.add_argument("--B", type=int, required=True, help="Batch size")
    parser.add_argument("--block-size", type=int, default=16, help="KV Cache block size")
    
    # Profiling parameters
    parser.add_argument("--num-warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--num-measurements", type=int, default=10, help="Number of measurement runs")
    parser.add_argument("--profiling-type", choices=["prefill", "decode", "mixed"], 
                       default="mixed", help="Type of profiling to run")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-medium",
                       help="Model to use for profiling")
    
    # Output parameters
    parser.add_argument("--save-plots", action="store_true", help="Save performance plots")
    parser.add_argument("--analysis-file", type=str, default="performance_analysis.json",
                       help="Analysis output file")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ProfilingConfig(
        C=args.C,
        M=args.M,
        B=args.B,
        block_size=args.block_size,
        num_warmup_runs=args.num_warmup,
        num_measurement_runs=args.num_measurements,
        results_file="enhanced_profiling_results.json",
    )
    
    # Create engine
    logger.info("Creating vLLM engine...")
    engine = create_profiling_engine(config, args.model)
    
    # Create enhanced runner
    runner = EnhancedProfilingRunner(engine, config)
    
    # Run enhanced profiling
    logger.info(f"Starting enhanced {args.profiling_type} profiling...")
    result, analysis = runner.run_enhanced_profiling(args.profiling_type)
    
    # Save analysis
    runner.analyzer.save_analysis_report(analysis, args.analysis_file)
    
    # Create plots
    if args.save_plots:
        runner.analyzer.plot_performance_metrics(save_plots=True)
    
    # Print summary
    print("\n" + "="*80)
    print("ENHANCED PROFILING ANALYSIS SUMMARY")
    print("="*80)
    
    # Configuration
    print(f"Configuration: C={config.C}, M={config.M}, B={config.B}, block_size={config.block_size}")
    print(f"Profiling Type: {args.profiling_type}")
    print(f"Model: {args.model}")
    print()
    
    # Performance summary
    print("Performance Summary:")
    for key, value in analysis.performance_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Bottleneck analysis
    print("Bottleneck Analysis:")
    for bottleneck_type, is_bottleneck in analysis.bottleneck_analysis.items():
        if isinstance(is_bottleneck, bool):
            status = "YES" if is_bottleneck else "NO"
            print(f"  {bottleneck_type}: {status}")
    print()
    
    # Recommendations
    print("Optimization Recommendations:")
    for i, recommendation in enumerate(analysis.optimization_recommendations, 1):
        print(f"  {i}. {recommendation}")
    print()
    
    # Custom schedule plans
    print("Recommended Custom Schedule Plans:")
    for plan in analysis.recommended_schedule_plans:
        print(f"  {plan['name']}: {plan['description']}")
        print(f"    Expected improvement: {plan['expected_improvement']}")
        print(f"    Key config: {plan['config']}")
        print()
    
    print("="*80)


if __name__ == "__main__":
    main() 