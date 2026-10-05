"""
Note on Multi-Rank Aggregation (TP > 1):
Telemetry recorded here (transfer_duration, post_duration, bytes_transferred, num_descriptors)
is recorded independently per Tensor Parallel (TP) rank.

During aggregate(), observations across all ranks are concatenated into a combined pool via list.extend().
Consequently, reduce() computes summary metrics across all ranks combined:
- 'Num successful transfers' is the total count across all ranks (not per-rank).
- 'Avg MB per transfer' and percentiles (P90) reflect the combined distribution across all ranks.
- 'Throughput (MB/s)' represents total_MB_all_ranks / total_time_all_ranks (average per-rank throughput).
"""
