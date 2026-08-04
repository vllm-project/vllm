# Chunk Statistics Example

This example demonstrates how to use LMCache's chunk statistics feature to track and analyze KV cache chunk reuse patterns.

## Overview

Chunk statistics provides insights into cache efficiency by tracking:
- Total chunks processed
- Unique chunks encountered
- Duplicate chunks (cache hits)
- Reuse rate (duplicate/total ratio)

## Prerequisites

- LMCache installed with vLLM integration
- A model for testing (e.g., `/data1/deepseek/DeepSeek-V2-Lite-Chat`)

## Examples

### Example 1: Memory Bloom Filter Strategy

The memory bloom filter strategy uses a probabilistic data structure for efficient duplicate detection with minimal memory overhead.

#### Configuration

See `memory_bloom_filter.yaml` for the configuration file.

#### Running the Example

```bash
# Start vLLM with chunk statistics enabled
LMCACHE_CONFIG_FILE=memory_bloom_filter.yaml \
PYTHONHASHSEED=0 \
python3 -m vllm.entrypoints.cli.main serve <model_path> \
--load-format dummy \
-tp 2 \
--trust-remote-code \
--served-model-name vllm_cpu_offload \
--gpu-memory-utilization 0.5 \
--max-num-seqs 64 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

#### Query Statistics

```bash
# Get current statistics (default port: 6999 for scheduler)
curl http://localhost:6999/chunk_statistics/status

# Pretty print JSON output
curl http://localhost:6999/chunk_statistics/status | jq .

# Start statistics collection (if not auto-started)
curl -X POST http://localhost:6999/chunk_statistics/start

# Stop statistics collection
curl -X POST http://localhost:6999/chunk_statistics/stop

# Reset statistics
curl -X POST http://localhost:6999/chunk_statistics/reset
```

#### Expected Output

```json
{
  "enabled": true,
  "total_requests": 3,
  "timing": {
    "lookup_time_seconds": 0.044486284255981445,
    "record_statistics_time_seconds": 6.246566772460938e-05,
    "check_exit_conditions_time_seconds": 5.7220458984375e-06,
    "total_time_seconds": 0.04455447196960449,
    "overhead_time_seconds": 6.818771362304688e-05,
    "overhead_percentage": 0.1530434782608696
  },
  "total_chunks": 12,
  "unique_chunks": 9,
  "duplicate_chunks": 3,
  "reuse_rate": 0.25,
  "async_queue": {
    "enabled": true,
    "capacity": 100000,
    "current_size": 0,
    "max_size_reached": 0,
    "full_blocks": 0,
    "utilization": 0.0
  },
  "bloom_filter": {
    "size_mb": 11.426279067993164,
    "hash_count": 6,
    "item_count": 9,
    "bits_set": 54,
    "fill_rate": 5.633768549952377e-07,
    "expected_elements": 10000000,
    "false_positive_rate": 0.01
  },
  "timestamp": 1763026696.7670634,
  "auto_exit_enabled": false,
  "auto_exit_timeout_hours": 0.0,
  "auto_exit_target_unique_chunks": null
}
```

### Example 2: File Hash Strategy

The file hash strategy writes chunk hashes to disk for exact tracking and offline analysis.

#### Configuration

See `file_hash.yaml` for the configuration file.

#### Running the Example

```bash
# Start vLLM with file hash strategy
LMCACHE_CONFIG_FILE=file_hash.yaml \
PYTHONHASHSEED=0 \
python3 -m vllm.entrypoints.cli.main serve <model_path> \
--load-format dummy \
-tp 2 \
--trust-remote-code \
--served-model-name vllm_cpu_offload \
--gpu-memory-utilization 0.5 \
--max-num-seqs 64 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

#### Analyze Collected Data

Use the provided Python script to analyze the collected chunk hashes:

```bash
# Analyze chunk hashes from default directory
python analyze_chunk_hashes.py --input-dir /tmp/lmcache_chunk_statistics

# Export results to JSON file
python analyze_chunk_hashes.py --input-dir /tmp/lmcache_chunk_statistics --output analysis_results.json
```

### Example 3: Auto-Stop Configuration

This example demonstrates automatic stopping based on time or chunk count.

#### Configuration

See `auto_stop.yaml` for the configuration file.

#### Running the Example

```bash
# Statistics will automatically stop after configured time or chunk count
LMCACHE_CONFIG_FILE=auto_stop.yaml \
PYTHONHASHSEED=0 \
python3 -m vllm.entrypoints.cli.main serve <model_path> \
--load-format dummy \
-tp 2 \
--trust-remote-code \
--served-model-name vllm_cpu_offload \
--gpu-memory-utilization 0.5 \
--max-num-seqs 64 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## Configuration Options

### Memory Bloom Filter Strategy

| Option | Default | Description |
|--------|---------|-------------|
| `chunk_statistics_mem_bf_expected_chunks` | 20000000 | Expected number of chunks for capacity planning |
| `chunk_statistics_mem_bf_false_positive_rate` | 0.01 | Target false positive rate (1%) |

### File Hash Strategy

| Option | Default | Description |
|--------|---------|-------------|
| `chunk_statistics_file_output_dir` | `/tmp/lmcache_chunk_statistics` | Directory for storing chunk hash files |
| `chunk_statistics_file_rotation_size` | 104857600 | File size threshold for rotation (100MB) |
| `chunk_statistics_file_max_count` | 100 | Maximum number of files to keep |

## Understanding the Metrics

### Reuse Rate

The reuse rate indicates cache efficiency:
- **0.0**: No cache reuse (all chunks are unique)
- **0.5**: 50% of chunks are duplicates
- **0.9**: 90% of chunks are duplicates (high cache efficiency)

### Bloom Filter Metrics

- **size_mb**: Memory used by the bloom filter
- **fill_rate**: Percentage of bits set in the bloom filter (0.0 to 1.0)
- **false_positive_rate**: Configured target false positive rate

### Async Queue Metrics

- **capacity**: Maximum queue size
- **current_size**: Current number of items in queue
- **max_size_reached**: Peak queue size observed
- **full_blocks**: Number of times the queue was full
- **utilization**: Current queue utilization (0.0 to 1.0)

## Best Practices

1. **Choose the Right Strategy:**
   - Use `memory_bloom_filter` for real-time monitoring with minimal overhead
   - Use `file_hash` for exact tracking and offline analysis

2. **Tune Bloom Filter Parameters:**
   - Set `expected_chunks` based on your workload size
   - Lower `false_positive_rate` increases memory usage but improves accuracy

3. **Monitor Memory Usage:**
   - Track `bloom_filter_size_mb` to ensure it fits in available memory
   - Adjust `expected_chunks` if memory usage is too high

4. **File Rotation:**
   - Configure appropriate `file_rotation_size` to balance file size and count
   - Set `file_max_count` to prevent unlimited disk usage

## Troubleshooting

### Statistics Not Updating

**Problem:** Statistics remain at zero or don't update.

**Solution:**
- Verify `enable_chunk_statistics` is set to `true`
- Check that statistics collection is started
- Ensure requests are being processed

### High Memory Usage

**Problem:** Bloom filter consuming too much memory.

**Solution:**
- Reduce `chunk_statistics_mem_bf_expected_chunks`
- Increase `chunk_statistics_mem_bf_false_positive_rate`
- Consider switching to `file_hash` strategy

### File System Full

**Problem:** Disk space exhausted with file hash strategy.

**Solution:**
- Reduce `chunk_statistics_file_max_count`
- Decrease `chunk_statistics_file_rotation_size`
- Implement external log rotation

## Additional Resources

- [Chunk Statistics Documentation](../../docs/source/production/observability/chunk_statistics.rst)
- [Internal API Server Documentation](../../docs/source/production/observability/internal_api_server.rst)
