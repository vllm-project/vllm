# Stats-Based Health Check Implementation

## Overview

This document describes the implementation of an efficient health check mechanism for vLLM V1 that monitors scheduler statistics instead of executing dummy batches.

## Problem Statement

The original health check pathway only verified that control-plane processes were alive (Ray executor and GPU workers). It did not validate that scheduling or model execution was making forward progress, so a wedged worker could still report as "healthy."

While forcing `execute_dummy_batch` would catch execution stalls, every probe burned a full forward pass on the accelerator, adding unnecessary overhead.

## Solution

The vLLM V1 engine already emits rich scheduler statistics on every iteration, including:
- **step_counter**: A monotonic counter incremented on each engine step
- **current_wave**: Wave number for data-parallel coordination
- **iteration_timestamp**: Timestamp from iteration stats
- **Request counts**: Number of waiting and running requests

These stats are published by the `EngineCore` over ZMQ, aggregated by the `DPCoordinator` (in data-parallel setups), and consumed by `AsyncMPClient` instances.

By observing whether the `step_counter` advances between health checks, we can determine if the engine core is progressing without executing extra work.

## Architecture

### Components

1. **HealthStateTracker** (`vllm/v1/engine/async_llm.py:48-91`)
   - Tracks the last observed `step_counter` and update timestamp
   - Provides `update(step_counter)` method to record progress
   - Provides `is_healthy()` method to check if progress is being made
   - Configurable stall timeout (default: 60 seconds)

2. **AsyncLLM Integration** (`vllm/v1/engine/async_llm.py`)
   - Instantiates `HealthStateTracker` in `__init__()` (line 198-201)
   - Updates tracker in `output_handler()` when scheduler stats arrive (line 574-576)
   - Enhanced `check_health()` method queries tracker for health status (line 707-730)

3. **Configuration** (`vllm/envs.py`)
   - `VLLM_HEALTH_CHECK_STALL_TIMEOUT`: Environment variable to configure timeout
   - Default: 60.0 seconds
   - Can be tuned based on workload characteristics

### Data Flow

```
EngineCore.step()
  └─> Emits SchedulerStats (with step_counter)
       └─> Published via ZMQ
            └─> [Optional] DPCoordinator aggregates (for DP>1)
                 └─> AsyncMPClient consumes stats
                      └─> AsyncLLM.output_handler() receives EngineCoreOutputs
                           └─> HealthStateTracker.update(step_counter)
                                └─> AsyncLLM.check_health() queries tracker
```

### Health Check Logic

```python
class HealthStateTracker:
    def update(self, step_counter: int, current_wave: int,
               num_waiting_reqs: int, num_running_reqs: int):
        # Update request counts
        self.num_waiting_reqs = num_waiting_reqs
        self.num_running_reqs = num_running_reqs

        # Detect forward progress: wave advanced OR step_counter advanced within same wave
        made_progress = False
        if current_wave > self.last_current_wave:
            # Wave advanced (even if step_counter reset to 0)
            made_progress = True
        elif current_wave == self.last_current_wave and step_counter > self.last_step_counter:
            # Same wave, step_counter advanced
            made_progress = True

        if made_progress:
            self.last_step_counter = step_counter
            self.last_current_wave = current_wave
            self.last_update_time = time.monotonic()  # Use monotonic for elapsed time

    def is_healthy(self):
        # Idle engine (no active requests)? Always healthy
        if self.num_waiting_reqs == 0 and self.num_running_reqs == 0:
            return True

        # Active requests - check for progress
        elapsed = time.monotonic() - self.last_update_time
        return elapsed < self.stall_timeout
```

The tracker distinguishes between these states:
1. **Idle**: No active requests (waiting + running = 0) → Always healthy
   - Includes initial state before any stats received
2. **Active with progress**: Requests in-flight AND (wave advancing OR step_counter advancing) → Healthy
3. **Stalled**: Requests in-flight but NEITHER wave NOR step_counter advancing for > timeout → **Unhealthy**

**Critical Design Details:**

- **Wave Tracking**: The tracker monitors `current_wave` from scheduler stats to correctly handle counter resets. When a data-parallel wave completes, `step_counter` resets to 0. This is legitimate progress, not a stall.
  - Example: `(wave=1, step=100)` → `(wave=2, step=0)` is **progress**
  - Example: `(wave=1, step=100)` → `(wave=1, step=0)` is **NOT progress** (anomaly)

- **Monotonic Time**: Uses `time.monotonic()` instead of `time.time()` to measure elapsed time, avoiding issues with system clock adjustments.

This design ensures:
- An idle engine never reports as unhealthy
- A stalled engine with active requests is detected
- Wave-based counter resets are correctly recognized as progress
- System clock adjustments don't affect health checks

## Benefits

1. **Zero Overhead**: Reuses existing stats infrastructure without additional work
2. **Detects Real Stalls**: Catches actual scheduling/execution wedges
3. **Backward Compatible**: Falls back to process-alive checks if stats unavailable
4. **Configurable**: Timeout adjustable via environment variable
5. **Works with Data Parallelism**: Coordinator already aggregates stats from all engines

## Configuration

Set the stall timeout via environment variable:

```bash
export VLLM_HEALTH_CHECK_STALL_TIMEOUT=120.0  # 2 minutes
```

Or programmatically:

```python
import vllm.envs as envs
envs.VLLM_HEALTH_CHECK_STALL_TIMEOUT = 120.0
```

## Testing

Unit tests are provided in `tests/v1/engine/test_health_check.py`:
- Initial state is healthy before initialization
- Tracker becomes initialized after first update
- Remains healthy with progress
- Becomes unhealthy after timeout
- Ignores non-advancing counters
- Custom timeout values work correctly

Run tests with:
```bash
pytest tests/v1/engine/test_health_check.py -v
```

## Edge Cases

### Idle Engines
If the engine is idle (no active requests), the tracker explicitly checks `num_waiting_reqs == 0 && num_running_reqs == 0` and always reports healthy, regardless of whether the `step_counter` is advancing. This prevents false positives when the engine is legitimately idle.

### Stats Delivery Delays
The default 60-second timeout is generous enough to handle network delays. Stats are normally published every 100ms by the coordinator, so even significant delays won't cause false positives.

### Data Parallel Deployments
In data-parallel setups (DP>1), the coordinator aggregates stats from all engines. The health tracker monitors progress based on any advancing step_counter from any engine, ensuring the system as a whole is making progress.

### Process Death
The existing `errored` property still checks if the engine process is alive. The health check first validates the process is alive, then checks for forward progress. This provides defense in depth.

## Implementation Files

- `vllm/envs.py`: Added `VLLM_HEALTH_CHECK_STALL_TIMEOUT` configuration
- `vllm/v1/engine/async_llm.py`:
  - Added `HealthStateTracker` class
  - Integrated into `AsyncLLM.__init__()`
  - Updated `output_handler()` to feed stats
  - Enhanced `check_health()` method
- `tests/v1/engine/test_health_check.py`: Comprehensive unit tests

## Future Enhancements

1. **Per-Engine Tracking**: In DP setups, track health per engine rank
2. **Metrics**: Expose health check status via Prometheus metrics
3. **Adaptive Timeout**: Automatically adjust timeout based on request patterns
4. **History Tracking**: Track step_counter velocity to predict stalls earlier

## References

- Design Doc: `docs/design/hybrid_kv_cache_manager.md` (original proposal)
- Scheduler Stats: `vllm/v1/metrics/stats.py`
- Coordinator: `vllm/v1/engine/coordinator.py`
- Core Client: `vllm/v1/engine/core_client.py`
