# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared memory buffer for cross-process metrics aggregation.

This module provides a thread-safe shared memory buffer that allows multiple
API server processes to aggregate their metrics for unified logging.
"""

import time
from ctypes import c_double, c_int64
from multiprocessing import Array, Lock
from typing import Any


class SharedStatsBuffer:
    """Thread-safe shared memory buffer for cross-server metrics.

    This buffer is created in the main process and shared across all API server
    processes. Each server records its stats to its own slot in the buffer,
    and the primary server (index 0) aggregates and logs the totals.

    Args:
        num_servers: Number of API server processes.
    """

    def __init__(self, num_servers: int):
        self.num_servers = num_servers

        # Cumulative counters (reset after each log interval)
        self.prompt_tokens = Array(c_int64, num_servers)
        self.generation_tokens = Array(c_int64, num_servers)
        self.preemptions = Array(c_int64, num_servers)
        self.corrupted_reqs = Array(c_int64, num_servers)

        # Snapshot values (not cumulative)
        self.running_reqs = Array(c_int64, num_servers)
        self.waiting_reqs = Array(c_int64, num_servers)
        self.kv_cache_usage = Array(c_double, num_servers)
        self.cpu_cache_usage = Array(c_double, num_servers)

        # Timestamps for staleness detection
        self.last_update_time = Array(c_double, num_servers)

        # Lock for atomic operations
        self.lock = Lock()

    def record(self, server_idx: int, **stats: Any) -> None:
        """Record stats from a specific server.

        Args:
            server_idx: Index of the server (0 to num_servers-1).
            **stats: Keyword arguments containing stat values.
                Cumulative stats: prompt_tokens, generation_tokens,
                    preemptions, corrupted_reqs
                Snapshot stats: running_reqs, waiting_reqs,
                    kv_cache_usage, cpu_cache_usage
        """
        with self.lock:
            # Cumulative counters (add to existing value)
            if "prompt_tokens" in stats:
                self.prompt_tokens[server_idx] += stats["prompt_tokens"]
            if "generation_tokens" in stats:
                self.generation_tokens[server_idx] += stats["generation_tokens"]
            if "preemptions" in stats:
                self.preemptions[server_idx] += stats["preemptions"]
            if "corrupted_reqs" in stats:
                self.corrupted_reqs[server_idx] += stats["corrupted_reqs"]

            # Snapshot values (replace existing value)
            if "running_reqs" in stats:
                self.running_reqs[server_idx] = stats["running_reqs"]
            if "waiting_reqs" in stats:
                self.waiting_reqs[server_idx] = stats["waiting_reqs"]
            if "kv_cache_usage" in stats:
                self.kv_cache_usage[server_idx] = stats["kv_cache_usage"]
            if "cpu_cache_usage" in stats:
                self.cpu_cache_usage[server_idx] = stats["cpu_cache_usage"]

            # Update timestamp
            self.last_update_time[server_idx] = time.monotonic()

    def get_aggregated_stats(self, reset_counters: bool = True) -> dict[str, Any]:
        """Get aggregated stats across all servers.

        Args:
            reset_counters: If True, reset cumulative counters after reading.

        Returns:
            Dictionary containing aggregated stats:
                - total_prompt_tokens: Sum across all servers
                - total_generation_tokens: Sum across all servers
                - total_preemptions: Sum across all servers
                - total_corrupted_reqs: Sum across all servers
                - total_running_reqs: Sum of currently running requests
                - total_waiting_reqs: Sum of currently waiting requests
                - avg_kv_cache_usage: Average KV cache usage
                - avg_cpu_cache_usage: Average CPU cache usage
        """
        with self.lock:
            # Sum cumulative counters (convert to list for sum())
            total_prompt_tokens = sum(list(self.prompt_tokens))  # type: ignore
            total_generation_tokens = sum(
                list(self.generation_tokens)  # type: ignore
            )
            total_preemptions = sum(list(self.preemptions))  # type: ignore
            total_corrupted_reqs = sum(list(self.corrupted_reqs))  # type: ignore

            # Sum snapshot values
            total_running_reqs = sum(list(self.running_reqs))  # type: ignore
            total_waiting_reqs = sum(list(self.waiting_reqs))  # type: ignore

            # Average cache usage (only count servers with non-zero values)
            kv_values = list(self.kv_cache_usage)  # type: ignore
            cpu_values = list(self.cpu_cache_usage)  # type: ignore
            active_kv_servers = sum(1 for x in kv_values if x > 0)
            active_cpu_servers = sum(1 for x in cpu_values if x > 0)

            avg_kv_cache = (
                sum(kv_values) / active_kv_servers if active_kv_servers > 0 else 0.0
            )
            avg_cpu_cache = (
                sum(cpu_values) / active_cpu_servers if active_cpu_servers > 0 else 0.0
            )

            stats = {
                "total_prompt_tokens": total_prompt_tokens,
                "total_generation_tokens": total_generation_tokens,
                "total_preemptions": total_preemptions,
                "total_corrupted_reqs": total_corrupted_reqs,
                "total_running_reqs": total_running_reqs,
                "total_waiting_reqs": total_waiting_reqs,
                "avg_kv_cache_usage": avg_kv_cache,
                "avg_cpu_cache_usage": avg_cpu_cache,
            }

            if reset_counters:
                # Reset cumulative counters
                for i in range(self.num_servers):
                    self.prompt_tokens[i] = 0
                    self.generation_tokens[i] = 0
                    self.preemptions[i] = 0
                    self.corrupted_reqs[i] = 0

            return stats

    def reset(self) -> None:
        """Reset all counters to zero."""
        with self.lock:
            for i in range(self.num_servers):
                self.prompt_tokens[i] = 0
                self.generation_tokens[i] = 0
                self.preemptions[i] = 0
                self.corrupted_reqs[i] = 0
                self.running_reqs[i] = 0
                self.waiting_reqs[i] = 0
                self.kv_cache_usage[i] = 0.0
                self.cpu_cache_usage[i] = 0.0
                self.last_update_time[i] = 0.0
