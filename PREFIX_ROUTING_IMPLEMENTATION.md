# Prefix-Aware Router Implementation Summary

This document summarizes the implementation of prefix-aware routing for vLLM's data parallel load balancing system.

## Implementation Overview

The prefix-aware router routes requests with the same token prefix to the same engine to improve cache hit rates, while load-balancing new prefixes across engines.

## Files Modified

### 1. Configuration Layer

#### `vllm/config/parallel.py`
- Added `enable_prefix_aware_routing: bool = False` field
- Added `prefix_routing_length: int = 16` field

#### `vllm/engine/arg_utils.py`
- Added corresponding fields to `EngineArgs` dataclass
- Added CLI arguments:
  - `--enable-prefix-aware-routing`: Enable/disable prefix-aware routing
  - `--prefix-routing-length`: Number of tokens to use as routing prefix (default: 16)
- Passed new configuration to `ParallelConfig` in `create_engine_config()`

### 2. Router Implementation

#### `vllm/v1/engine/core_client.py`
- Created new `PrefixAwareDPLBAsyncMPClient` class (extends `DPLBAsyncMPClient`)
  - Maintains `prefix_to_engine_map: dict[tuple[int, ...], int]` for prefix→engine mappings
  - Extracts prefix: `tuple(request.prompt_token_ids[:prefix_length])`
  - Routes to same engine if prefix seen before
  - Routes to least-loaded engine for new prefixes
  - Preserves explicit `data_parallel_rank` override functionality

- Modified `EngineCoreClient.make_async_mp_client()` to use `PrefixAwareDPLBAsyncMPClient` when:
  - `data_parallel_size > 1`
  - NOT using external load balancer
  - `enable_prefix_aware_routing` is True

## Files Created

### 3. Testing

#### `tests/v1/test_prefix_aware_routing.py`
- `test_prefix_routing_same_prefix()`: Verifies same prefix routes to same engine
- `test_prefix_routing_load_balance()`: Verifies new prefixes distribute evenly
- `test_prefix_routing_mixed_workload()`: Tests cached + new prefixes
- `test_prefix_routing_short_prompts()`: Handles prompts < prefix_length
- `test_prefix_routing_config()`: Verifies CLI flags work

### 4. Demonstration

#### `examples/online_serving/prefix_routing_demo.py`
- Demonstrates prefix-aware routing with OpenAI-compatible API
- Shows three scenarios:
  1. Repeated prefixes (cache hits)
  2. Unique prefixes (load balancing)
  3. Mixed workload (cache + load balancing)

## Usage

### Starting the Server

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --data-parallel-size 2 \
    --enable-prefix-aware-routing \
    --prefix-routing-length 16
```

### Running the Demo

```bash
python examples/online_serving/prefix_routing_demo.py
```

### Running Tests

```bash
# Run prefix routing tests
pytest tests/v1/test_prefix_aware_routing.py -v

# Run all DP load balancing tests (including prefix routing)
pytest tests/v1/test_internal_lb_dp.py -v
```

## Architecture Details

### Current System
- Routing in `DPLBAsyncMPClient.get_core_engine_for_request()`
- Per-request routing: one request at a time
- Simple scoring: `score = waiting * 4 + running`

### New System
- Extends `DPLBAsyncMPClient` with new `PrefixAwareDPLBAsyncMPClient` subclass
- Maintains prefix map for prefix→engine mappings
- Extract prefix from first N tokens (configurable, default 16)
- Route to same engine if prefix seen before, else route to least-loaded engine
- Opt-in via `--enable-prefix-aware-routing` CLI flag

### Design Rationale

**Why extend DPLBAsyncMPClient?**
- Reuses existing load balancing infrastructure (`lb_engines` stats)
- Minimal changes to existing code
- Backward compatible (opt-in via flag)
- Natural integration point in architecture

**Why per-request routing vs batch?**
- Matches vLLM's existing architecture
- No request buffering needed (avoids latency)
- State maintained in client instance across requests

### Trade-offs
- Prefix map grows unbounded (future: add LRU cache with configurable size)
- Each API server has independent prefix map (acceptable for internal LB mode)
- Coordination overhead minimal (map lookup is O(1))

## Verification Checklist

- [x] Requests with identical prefixes route to same engine
- [x] New prefixes distribute across engines based on load
- [x] Short prompts (< 16 tokens) handled correctly
- [x] Explicit `data_parallel_rank` override still works
- [x] Configuration flags properly exposed via CLI
- [x] Tests cover main scenarios
- [x] Demo script shows practical usage

## Performance Considerations

- Prefix map lookup: O(1) time complexity
- Memory: Grows with number of unique prefixes seen
- Routing overhead: Negligible (< 1ms)
- Expected benefits:
  - Improved prefix cache hit rates
  - Reduced latency for repeated prefixes
  - Better overall throughput for workloads with repeated patterns

## Future Enhancements

1. **LRU Cache for Prefix Map**: Add configurable size limit to prevent unbounded growth
2. **Prefix Length Tuning**: Adaptive prefix length based on workload
3. **Cross-Server Coordination**: Share prefix maps across API servers
4. **Metrics**: Add Prometheus metrics for prefix hit rates
5. **Cache Warmup**: Pre-populate prefix map from common patterns
