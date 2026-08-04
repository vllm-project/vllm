#!/usr/bin/env python3
"""
LMCache Controller ZMQ Benchmark Tool - CLI Entry Point

This tool performs load testing on LMCache Controller using ZMQ interface
to measure message throughput, latency, and system performance.

Test operations:
- BatchedKVOperationMsg: admit/evict messages via PUSH socket
- RegisterMsg/DeRegisterMsg/HeartbeatMsg: worker lifecycle messages
"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Dict, List
import argparse
import asyncio
import json
import multiprocessing
import statistics

# First Party
from lmcache.logging import init_logger
from lmcache.tools.controller_benchmark.benchmark import (
    BenchmarkResults,
    OperationStats,
    ZMQControllerBenchmark,
)
from lmcache.tools.controller_benchmark.config import ZMQBenchmarkConfig

logger = init_logger(__name__)


def run_single_process(config: ZMQBenchmarkConfig) -> BenchmarkResults:
    """Run benchmark in a single process and return results"""
    benchmark = ZMQControllerBenchmark(config)
    asyncio.run(benchmark.run_benchmark())
    benchmark.print_results()
    return benchmark.get_results()


def aggregate_results(
    results_list: List[BenchmarkResults], operations: Dict[str, float]
) -> BenchmarkResults:
    """Aggregate results from multiple processes"""
    aggregated = BenchmarkResults()

    if not results_list:
        return aggregated

    # Sum up totals
    aggregated.total_requests = sum(r.total_requests for r in results_list)
    aggregated.total_messages = sum(r.total_messages for r in results_list)
    aggregated.total_time = max(r.total_time for r in results_list)
    aggregated.overall_rps = sum(r.overall_rps for r in results_list)
    aggregated.overall_qps = sum(r.overall_qps for r in results_list)

    # Aggregate memory usage
    for r in results_list:
        aggregated.memory_usage.extend(r.memory_usage)

    # Aggregate per-operation stats
    for op_name in operations.keys():
        op_stats_list = [
            r.operations[op_name] for r in results_list if op_name in r.operations
        ]
        if op_stats_list:
            # Sum QPS and RPS
            total_qps = sum(s.qps for s in op_stats_list)
            total_rps = sum(s.rps for s in op_stats_list)

            # Average latencies (weighted by RPS would be more accurate,
            # but simple average is acceptable)
            min_latencies = [s.min_latency for s in op_stats_list if s.min_latency > 0]
            max_latencies = [s.max_latency for s in op_stats_list if s.max_latency > 0]
            p95_latencies = [s.p95_latency for s in op_stats_list if s.p95_latency > 0]

            aggregated.operations[op_name] = OperationStats(
                qps=total_qps,
                rps=total_rps,
                avg_latency=(
                    sum(s.avg_latency * s.rps for s in op_stats_list) / total_rps
                    if total_rps > 0
                    else 0.0
                ),
                min_latency=min(min_latencies) if min_latencies else 0.0,
                max_latency=max(max_latencies) if max_latencies else 0.0,
                p95_latency=(
                    sum(s.p95_latency * s.rps for s in op_stats_list) / total_rps
                    if total_rps > 0 and p95_latencies
                    else 0.0
                ),
                errors=sum(s.errors for s in op_stats_list),
            )

    return aggregated


def print_aggregated_results(
    results: BenchmarkResults,
    config: ZMQBenchmarkConfig,
    num_processes: int,
):
    """Print aggregated benchmark results from all processes"""
    print("\n" + "=" * 80)
    print(
        "LMCache Controller ZMQ Benchmark - AGGREGATED RESULTS (%d processes)"
        % num_processes
    )
    print("=" * 80)

    print("\nConfiguration:")
    print("  Controller URL: %s" % config.controller_pull_url)
    print("  Duration: %d seconds" % config.duration)
    print("  Batch Size: %d" % config.batch_size)
    print("  Operations: %s" % config.operations)
    print(
        "  Instances per process: %d, Workers: %d, Locations: %d, Keys: %d"
        % (
            config.num_instances,
            config.num_workers,
            config.num_locations,
            config.num_keys,
        )
    )
    print("  Total Instances: %d" % (config.num_instances * num_processes))

    print("\nAggregated Performance:")
    print("  Total Requests: %d" % results.total_requests)
    print("  Total Messages: %d" % results.total_messages)
    print("  Total Time: %.2fs" % results.total_time)
    print("  Overall RPS (Requests/sec): %.2f" % results.overall_rps)
    print("  Overall QPS (Messages/sec): %.2f" % results.overall_qps)

    print("\nPer-Operation Performance (Aggregated):")
    for op_name in config.operations.keys():
        if op_name in results.operations:
            stats = results.operations[op_name]
            print("  %s:" % op_name)
            print("    RPS (Requests/sec): %.2f" % stats.rps)
            print("    QPS (Messages/sec): %.2f" % stats.qps)
            print(
                "    Latency - Avg: %.3fms, Min: %.3fms, Max: %.3fms, P95: %.3fms"
                % (
                    stats.avg_latency * 1000,
                    stats.min_latency * 1000,
                    stats.max_latency * 1000,
                    stats.p95_latency * 1000,
                )
            )
            print("    Errors: %d" % stats.errors)

    print("\nSystem Metrics (All Processes):")
    if results.memory_usage:
        avg_memory = statistics.mean(results.memory_usage)
        max_memory = max(results.memory_usage)
        print("  Memory Usage - Avg: %.1f%%, Max: %.1f%%" % (avg_memory, max_memory))

    print("=" * 80)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="LMCache Controller ZMQ Benchmark Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--controller-host",
        type=str,
        default="127.0.0.1",
        help="Controller host address",
    )

    parser.add_argument(
        "--monitor-ports",
        type=str,
        default='{"pull":8100,"reply":8101}',
        help='Monitor ports in JSON format, e.g. {"pull":8100,"reply":8101}',
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of KV operations per batch message",
    )

    parser.add_argument(
        "--operations",
        type=str,
        default="admit:70,evict:25,heartbeat:5",
        help="Operation distribution (name:percentage comma-separated)",
    )

    parser.add_argument(
        "--num-instances",
        type=int,
        default=10,
        help="Number of instances to simulate per process",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers per instance",
    )

    parser.add_argument(
        "--num-locations",
        type=int,
        default=1,
        help="Number of storage locations",
    )

    parser.add_argument(
        "--num-keys",
        type=int,
        default=10000,
        help="Number of unique keys",
    )

    parser.add_argument(
        "--num-hashes",
        type=int,
        default=100,
        help="Number of hashes for P2P lookup operations",
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of concurrent benchmark processes",
    )

    parser.add_argument(
        "--no-register-first",
        action="store_true",
        help="Skip pre-registering workers before benchmark",
    )

    args = parser.parse_args()

    # Parse monitor ports from JSON
    try:
        monitor_ports = json.loads(args.monitor_ports)
        pull_port = monitor_ports.get("pull", 8100)
        reply_port = monitor_ports.get("reply")
        heartbeat_port = monitor_ports.get("heartbeat")
    except json.JSONDecodeError as e:
        logger.error("Failed to parse monitor-ports JSON: %s", e)
        raise ValueError("Invalid monitor-ports format") from e

    # Convert 0.0.0.0 to 127.0.0.1 for client connections
    client_host = (
        "127.0.0.1" if args.controller_host == "0.0.0.0" else args.controller_host
    )
    controller_pull_url = f"{client_host}:{pull_port}"
    controller_reply_url = f"{client_host}:{reply_port}" if reply_port else None
    controller_heartbeat_url = (
        f"{client_host}:{heartbeat_port}" if heartbeat_port else None
    )

    # Parse operations
    operations = {}
    for op_str in args.operations.split(","):
        if ":" in op_str:
            name, percentage = op_str.split(":", 1)
            operations[name.strip()] = float(percentage.strip())

    num_processes = args.num_processes

    # Create a base config dict
    base_config_kwargs = {
        "controller_pull_url": controller_pull_url,
        "controller_reply_url": controller_reply_url,
        "controller_heartbeat_url": controller_heartbeat_url,
        "duration": args.duration,
        "batch_size": args.batch_size,
        "operations": operations,
        "num_instances": args.num_instances,
        "num_workers": args.num_workers,
        "num_locations": args.num_locations,
        "num_keys": args.num_keys,
        "num_hashes": args.num_hashes,
        "register_first": not args.no_register_first,
        "num_processes": num_processes,
    }

    try:
        if num_processes == 1:
            # Single process mode
            config = ZMQBenchmarkConfig(**base_config_kwargs, process_id=0)
            run_single_process(config)
        else:
            # Multi-process mode
            logger.info(
                "Starting multi-process benchmark with %d processes", num_processes
            )
            configs = [
                ZMQBenchmarkConfig(**base_config_kwargs, process_id=i)
                for i in range(num_processes)
            ]
            # Use multiprocessing pool to run benchmarks in parallel
            with multiprocessing.Pool(processes=num_processes) as pool:
                results_list = pool.map(run_single_process, configs)

            # Aggregate and print combined results
            aggregated = aggregate_results(results_list, operations)
            print_aggregated_results(aggregated, configs[0], num_processes)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error("Benchmark failed: %s", e)
        raise e


if __name__ == "__main__":
    main()
