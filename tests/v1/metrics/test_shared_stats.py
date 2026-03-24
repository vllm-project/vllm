# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for shared stats buffer used in multi-server metrics aggregation."""

import multiprocessing
import time

from vllm.v1.metrics.shared_stats import SharedStatsBuffer


class TestSharedStatsBuffer:
    """Test SharedStatsBuffer for cross-process metrics aggregation."""

    def test_basic_record_and_aggregation(self):
        """Test basic recording and aggregation across multiple servers."""
        buffer = SharedStatsBuffer(num_servers=3)

        # Record stats from three servers
        buffer.record(0, prompt_tokens=100, generation_tokens=200, running_reqs=5)
        buffer.record(1, prompt_tokens=150, generation_tokens=250, running_reqs=3)
        buffer.record(2, prompt_tokens=50, generation_tokens=100, running_reqs=2)

        # Get aggregated stats without resetting
        stats = buffer.get_aggregated_stats(reset_counters=False)

        assert stats["total_prompt_tokens"] == 300
        assert stats["total_generation_tokens"] == 550
        assert stats["total_running_reqs"] == 10
        assert stats["total_waiting_reqs"] == 0

    def test_cumulative_vs_snapshot(self):
        """Test that cumulative counters accumulate while snapshot values replace."""
        buffer = SharedStatsBuffer(num_servers=2)

        # First recording
        buffer.record(0, prompt_tokens=100, running_reqs=5)
        buffer.record(0, prompt_tokens=50, running_reqs=3)  # Second call

        stats = buffer.get_aggregated_stats(reset_counters=False)

        # Cumulative counter should add
        assert stats["total_prompt_tokens"] == 150  # 100 + 50

        # Snapshot should replace (not accumulate)
        assert stats["total_running_reqs"] == 3  # Latest value, not 8

    def test_reset_counters(self):
        """Test that reset_counters=True clears cumulative counters."""
        buffer = SharedStatsBuffer(num_servers=2)

        buffer.record(0, prompt_tokens=100, generation_tokens=200)
        buffer.record(1, prompt_tokens=150, generation_tokens=250)

        # Get and reset
        stats1 = buffer.get_aggregated_stats(reset_counters=True)
        assert stats1["total_prompt_tokens"] == 250
        assert stats1["total_generation_tokens"] == 450

        # After reset, should be zero
        stats2 = buffer.get_aggregated_stats(reset_counters=False)
        assert stats2["total_prompt_tokens"] == 0
        assert stats2["total_generation_tokens"] == 0

    def test_kv_cache_averaging(self):
        """Test that KV cache usage is averaged correctly."""
        buffer = SharedStatsBuffer(num_servers=3)

        buffer.record(0, kv_cache_usage=0.5)
        buffer.record(1, kv_cache_usage=0.7)
        buffer.record(2, kv_cache_usage=0.6)

        stats = buffer.get_aggregated_stats(reset_counters=False)

        # Average should be (0.5 + 0.7 + 0.6) / 3 = 0.6
        assert abs(stats["avg_kv_cache_usage"] - 0.6) < 0.001

    def test_partial_kv_cache_usage(self):
        """Test averaging when only some servers report KV cache usage."""
        buffer = SharedStatsBuffer(num_servers=3)

        # Only two servers report KV cache usage
        buffer.record(0, kv_cache_usage=0.5)
        buffer.record(1, kv_cache_usage=0.7)
        # Server 2 doesn't report (defaults to 0.0)

        stats = buffer.get_aggregated_stats(reset_counters=False)

        # Should average only non-zero values: (0.5 + 0.7) / 2 = 0.6
        assert abs(stats["avg_kv_cache_usage"] - 0.6) < 0.001

    def test_multiprocess_access(self):
        """Test that the buffer works correctly across multiple processes."""

        def worker_task(buffer, server_idx, num_iterations):
            """Worker function that records stats."""
            for i in range(num_iterations):
                buffer.record(
                    server_idx,
                    prompt_tokens=10,
                    generation_tokens=20,
                    running_reqs=server_idx + 1,
                )
                time.sleep(0.001)  # Small delay

        num_servers = 3
        num_iterations = 10
        buffer = SharedStatsBuffer(num_servers)

        # Spawn worker processes
        processes = []
        for i in range(num_servers):
            p = multiprocessing.Process(
                target=worker_task,
                args=(buffer, i, num_iterations),
            )
            processes.append(p)
            p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Verify aggregated results
        stats = buffer.get_aggregated_stats(reset_counters=False)

        # Each server recorded 10 iterations of 10 prompt tokens
        # 3 servers * 10 iterations * 10 tokens = 300
        assert stats["total_prompt_tokens"] == 300

        # Each server recorded 10 iterations of 20 generation tokens
        # 3 servers * 10 iterations * 20 tokens = 600
        assert stats["total_generation_tokens"] == 600

        # Running reqs should be sum of latest values: 1 + 2 + 3 = 6
        assert stats["total_running_reqs"] == 6

    def test_reset_all(self):
        """Test reset() method clears all counters."""
        buffer = SharedStatsBuffer(num_servers=2)

        buffer.record(
            0,
            prompt_tokens=100,
            generation_tokens=200,
            running_reqs=5,
            kv_cache_usage=0.7,
        )
        buffer.record(
            1,
            prompt_tokens=150,
            generation_tokens=250,
            running_reqs=3,
            kv_cache_usage=0.5,
        )

        # Reset everything
        buffer.reset()

        # All values should be zero
        stats = buffer.get_aggregated_stats(reset_counters=False)
        assert stats["total_prompt_tokens"] == 0
        assert stats["total_generation_tokens"] == 0
        assert stats["total_running_reqs"] == 0
        assert stats["total_waiting_reqs"] == 0
        assert stats["avg_kv_cache_usage"] == 0.0

    def test_preemptions_and_corrupted(self):
        """Test tracking of preemptions and corrupted requests."""
        buffer = SharedStatsBuffer(num_servers=2)

        buffer.record(0, preemptions=5, corrupted_reqs=2)
        buffer.record(1, preemptions=3, corrupted_reqs=1)

        stats = buffer.get_aggregated_stats(reset_counters=False)

        assert stats["total_preemptions"] == 8
        assert stats["total_corrupted_reqs"] == 3
