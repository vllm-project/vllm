# Weighted Fair Queuing (WFQ) Scheduler Design Document

**Author**: Ishrith Gowda
**Date**: 2025-12-28
**Status**: Design Phase
**Target**: vLLM v1 Scheduler

---

## 1. Executive Summary

This document outlines the design for adding Weighted Fair Queuing (WFQ) scheduling policy to vLLM's v1 scheduler architecture. WFQ provides fairness guarantees by preventing large requests from starving smaller ones, while maintaining backward compatibility with existing FCFS and Priority scheduling policies.

**Key Benefits:**
- Fairness: Guarantees proportional resource allocation based on request weights
- Starvation Prevention: Ensures all requests make progress regardless of size
- Flexibility: User-configurable weights per request
- Performance: Target 30-40% reduction in latency variance with < 5% throughput overhead

---

## 2. Architecture Analysis

### 2.1 Current Scheduler Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Scheduler                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              SchedulerInterface                         │ │
│  │  - add_request()                                        │ │
│  │  - schedule() → SchedulerOutput                         │ │
│  │  - update_from_output()                                 │ │
│  └────────────────────────────────────────────────────────┘ │
│           │                                                  │
│           ▼                                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │          RequestQueue (ABC)                             │ │
│  │  - add_request()                                        │ │
│  │  - pop_request()                                        │ │
│  │  - peek_request()                                       │ │
│  │  - prepend_request()  (for preemption)                  │ │
│  │  - remove_request()                                     │ │
│  └────────────────────────────────────────────────────────┘ │
│           │                                                  │
│           ├─────────────┬─────────────┬──────────────┐      │
│           ▼             ▼             ▼              ▼      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │ FCFSRequest  │ │ PriorityReq  │ │  WFQRequest  │  NEW  │
│  │    Queue     │ │    Queue     │ │    Queue     │       │
│  │              │ │              │ │              │       │
│  │ - deque      │ │ - heap       │ │ - heap       │       │
│  │   based      │ │   (priority, │ │   (virtual   │       │
│  │              │ │    arrival)  │ │    finish)   │       │
│  └──────────────┘ └──────────────┘ └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Request Lifecycle

```
Request Arrival → add_request()
                      │
                      ▼
              Queue (FCFS/Priority/WFQ)
                      │
                      ▼
              schedule() ← Token Budget Check
                      │       KV Cache Check
                      ▼
              Selected Requests → SchedulerOutput
                      │
                      ▼
              Model Execution
                      │
                      ▼
              update_from_output()
                      │
                      ├─ Finished → Remove
                      └─ Preempted → prepend_request()
```

### 2.3 Key Files and Responsibilities

| File | Responsibility | Modifications Needed |
|------|---------------|---------------------|
| `vllm/v1/core/sched/scheduler.py` | Main scheduler logic | ❌ None (uses factory) |
| `vllm/v1/core/sched/request_queue.py` | Queue interface & implementations | ✅ Add WFQ enum, factory |
| `vllm/v1/core/sched/wfq_queue.py` | WFQ implementation | ✅ New file |
| `vllm/v1/request.py` | Request data model | ✅ Add weight attribute |
| `vllm/config/scheduler.py` | Configuration | ✅ Add policy="wfq" |

---

## 3. Design Decisions

### 3.1 Why WFQ Over Alternatives?

| Policy | Pros | Cons | Verdict |
|--------|------|------|---------|
| **WFQ** | Simple virtual time, proven fair, LLM-friendly | Requires weight tuning | ✅ **Selected** |
| Deficit Round Robin | Good for packet networks | Complex deficit tracking, quantum tuning | ❌ Too complex |
| Multi-Level Feedback | Adaptive priorities | Many parameters, hard to tune | ❌ 3 weeks insufficient |
| Shortest Job First | Simple, optimal avg latency | Starvation possible, no fairness | ❌ Not fair |

### 3.2 Virtual Time Algorithm

**Core Principle:** Each request tracks virtual start/finish times. Scheduler selects request with smallest virtual_finish_time.

```
virtual_start = max(global_virtual_time, request_arrival_time)
virtual_finish = virtual_start + (tokens_needed / weight)
global_virtual_time += tokens_scheduled / weight
```

**Why this works:**
- Higher weight → smaller virtual_finish → scheduled earlier
- Equal weights → degenerates to FCFS
- Prevents starvation: all requests eventually have smallest virtual_finish

### 3.3 Integration with vLLM Features

| Feature | Integration Strategy | Risk |
|---------|---------------------|------|
| Chunked Prefill | Update virtual_finish after partial scheduling | Low |
| Preemption | Preserve virtual times on prepend_request() | Medium |
| Prefix Caching | Virtual time independent of cache hits | Low |
| Spec Decoding | Weight applies to total tokens (prompt + spec) | Low |
| LoRA | Weight per request, independent of adapter | Low |

---

## 4. Detailed Component Design

### 4.1 WFQRequestQueue Class

```python
# File: vllm/v1/core/sched/wfq_queue.py

class WFQRequestQueue(RequestQueue):
    """Weighted Fair Queuing scheduler for LLM requests.

    Implements virtual time-based fairness using a min-heap ordered by
    virtual_finish_time. Requests with higher weights receive proportionally
    more resources.
    """

    def __init__(self, default_weight: float = 1.0) -> None:
        self._heap: list[Request] = []
        self._virtual_time: float = 0.0
        self._default_weight = default_weight

    def add_request(self, request: Request) -> None:
        """Add request and compute virtual start/finish times."""
        # Initialize weight if not set
        if not hasattr(request, 'weight') or request.weight <= 0:
            request.weight = self._default_weight

        # Compute virtual times
        request.virtual_start_time = max(
            self._virtual_time,
            request.arrival_time
        )
        tokens_needed = self._estimate_tokens_needed(request)
        request.virtual_finish_time = (
            request.virtual_start_time + (tokens_needed / request.weight)
        )

        # Insert into heap (min-heap by virtual_finish_time)
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop request with smallest virtual_finish_time."""
        if not self._heap:
            raise IndexError("pop from empty heap")

        request = heapq.heappop(self._heap)

        # Advance global virtual time
        self._virtual_time = max(
            self._virtual_time,
            request.virtual_start_time
        )

        return request

    # ... (other methods following RequestQueue ABC)
```

### 4.2 Request Model Extensions

```python
# File: vllm/v1/request.py

class Request:
    def __init__(
        self,
        request_id: str,
        # ... existing parameters ...
        weight: float = 1.0,  # NEW
    ) -> None:
        # ... existing initialization ...

        # WFQ-specific attributes
        self.weight = weight
        self.virtual_start_time: float = 0.0
        self.virtual_finish_time: float = 0.0
```

**Note:** Adding `weight` parameter maintains backward compatibility (default=1.0).

### 4.3 Configuration Extensions

```python
# File: vllm/config/scheduler.py

SchedulerPolicy = Literal["fcfs", "priority", "wfq"]  # Add "wfq"

@config
@dataclass
class SchedulerConfig:
    # ... existing fields ...

    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:
    - "fcfs": First Come First Served
    - "priority": Priority-based (lower priority value = earlier)
    - "wfq": Weighted Fair Queuing (fairness-aware)
    """
```

### 4.4 Factory Pattern Update

```python
# File: vllm/v1/core/sched/request_queue.py

class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
    WFQ = "wfq"  # NEW

def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.WFQ:
        from vllm.v1.core.sched.wfq_queue import WFQRequestQueue
        return WFQRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
```

---

## 5. Request Comparison for Heap

Current Request comparison (from reading source):
```python
# In Request class
def __lt__(self, other: "Request") -> bool:
    # Priority queue: compare by (priority, arrival_time)
    if self.priority != other.priority:
        return self.priority < other.priority
    return self.arrival_time < other.arrival_time
```

**WFQ Extension:**
```python
def __lt__(self, other: "Request") -> bool:
    """Compare requests for heap operations.

    For WFQ: compare by virtual_finish_time
    Fallback: original (priority, arrival_time) comparison
    """
    # WFQ: compare by virtual_finish_time if set
    if (hasattr(self, 'virtual_finish_time') and
        hasattr(other, 'virtual_finish_time')):
        if self.virtual_finish_time != other.virtual_finish_time:
            return self.virtual_finish_time < other.virtual_finish_time

    # Fallback to original comparison
    if self.priority != other.priority:
        return self.priority < other.priority
    return self.arrival_time < other.arrival_time
```

---

## 6. Edge Cases and Handling

| Edge Case | Scenario | Solution |
|-----------|----------|----------|
| **Zero Weight** | User sets weight=0 | Validate weight > 0, use default if invalid |
| **Very Large Weight** | weight=1000000 | No issue, just earlier scheduling |
| **Preemption** | Request returns after preemption | Preserve virtual_finish_time in prepend_request() |
| **Empty Queue** | pop/peek from empty queue | Raise IndexError (consistent with Priority queue) |
| **Partial Prefill** | Request scheduled with partial tokens | Recompute virtual_finish after each iteration |
| **Equal Virtual Finish** | Two requests same virtual_finish_time | Fallback to arrival_time (from __lt__) |
| **Virtual Time Overflow** | Long-running system | Use float64, overflow at ~10^308 (not practical) |

---

## 7. Testing Strategy

### 7.1 Unit Tests
```python
# tests/v1/core/test_wfq_scheduler.py

def test_wfq_virtual_time_monotonic():
    """Virtual time should never decrease."""

def test_wfq_fairness_equal_weights():
    """Equal weights should behave like FCFS."""

def test_wfq_fairness_weighted():
    """Higher weight should get more tokens."""

def test_wfq_preemption_preserves_virtual_time():
    """Preempted request maintains virtual_finish_time."""

def test_wfq_empty_queue_raises():
    """pop/peek on empty queue raises IndexError."""
```

### 7.2 Integration Tests
```python
def test_wfq_with_chunked_prefill():
    """WFQ works correctly with partial token scheduling."""

def test_wfq_with_prefix_caching():
    """Cache hits don't affect virtual time calculations."""
```

### 7.3 Benchmarking
```python
# benchmarks/scheduler_policies/benchmark_wfq.py

def benchmark_fairness(num_requests=10000):
    """Measure Jain's Fairness Index."""

def benchmark_overhead(num_requests=10000):
    """Measure WFQ overhead vs FCFS."""
```

---

## 8. Performance Analysis

### 8.1 Complexity Analysis

| Operation | FCFS | Priority | WFQ |
|-----------|------|----------|-----|
| add_request | O(1) | O(log n) | O(log n) |
| pop_request | O(1) | O(log n) | O(log n) |
| prepend_request | O(1) | O(log n) | O(log n) |
| remove_request | O(n) | O(n) | O(n) |

**Verdict:** WFQ has same complexity as Priority queue (both use heap).

### 8.2 Expected Performance

**Throughput:** < 5% degradation vs FCFS (heap operations are fast)
**Fairness:** 30-40% improvement in Jain's Index
**Latency (p50):** < 10% increase (fair distribution may delay some)
**Latency (p99):** **Improvement** (no long tail from starvation)

---

## 9. Critical Questions

1. **Preemption Semantics**: When a request is preempted and returns via `prepend_request()`, should we:
   - (A) Preserve original virtual_finish_time? ✅ **Preferred** (maintains fairness)
   - (B) Recompute virtual_finish_time? (could penalize preempted requests)

2. **Partial Prefill Updates**: After scheduling partial tokens, should we:
   - (A) Recompute virtual_finish based on remaining tokens? ✅ **Preferred**
   - (B) Keep original virtual_finish? (doesn't account for work done)

3. **Weight Bounds**: Should we enforce min/max weight limits?
   - (A) No limits (trust users) ✅ **Preferred** (simpler)
   - (B) Enforce 0.1 <= weight <= 10.0 (prevent abuse)

4. **Backward Compatibility**: How to handle requests from API without weight?
   - (A) Default weight=1.0 ✅ **Implemented**
   - (B) Reject requests (too strict)

5. **Virtual Time Units**: Should virtual time be:
   - (A) Dimensionless (current design) ✅ **Preferred**
   - (B) In units of seconds (adds complexity)

6. **Integration with Priority**: What if request has both priority AND weight?
   - (A) WFQ policy ignores priority field ✅ **Preferred** (clean separation)
   - (B) Combine both (weight * priority_factor) (too complex)

7. **Spec Decoding**: For speculative tokens, should weight apply to:
   - (A) Total tokens (prompt + accepted_spec) ✅ **Preferred**
   - (B) Only prompt tokens (inconsistent)

8. **Token Estimation**: How to estimate tokens_needed for virtual_finish?
   - (A) prompt_tokens + max_tokens ✅ **Current design**
   - (B) Use sampling_params.min_tokens (more accurate but complex)

9. **KV Cache Interaction**: Should cache hits affect virtual time?
   - (A) No, virtual time based on total tokens ✅ **Preferred** (simpler)
   - (B) Yes, reduce tokens_needed by cache hits (complex)

10. **Heap Reordering**: After partial scheduling, should we re-heapify?
    - (A) Yes, always maintain heap property ✅ **Required for correctness**
    - (B) No, only on next pop (incorrect)

---

## 10. Implementation Checklist

### Phase 1: Core Implementation
- [ ] Create `wfq_queue.py` with WFQRequestQueue class
- [ ] Add `weight` attribute to Request class
- [ ] Update `SchedulingPolicy` enum
- [ ] Update `create_request_queue()` factory
- [ ] Extend Request.__lt__() for virtual_finish_time comparison

### Phase 2: Configuration
- [ ] Update `SchedulerConfig.policy` type to include "wfq"
- [ ] Add validation for policy value
- [ ] Update docstrings

### Phase 3: Testing
- [ ] Unit tests for WFQRequestQueue (15-20 tests)
- [ ] Integration tests with Scheduler (5-10 tests)
- [ ] Benchmarking suite

### Phase 4: Documentation
- [ ] Algorithm explanation
- [ ] Configuration guide
- [ ] Performance characteristics
- [ ] Migration guide (FCFS → WFQ)

### Phase 5: PR Preparation
- [ ] Code review and cleanup
- [ ] Type checking (mypy)
- [ ] Linting (ruff)
- [ ] Commit message preparation
- [ ] PR description with benchmarks

---

## 11. References

1. **WFQ Original Paper**: Demers, A., Keshav, S., & Shenker, S. (1989). "Analysis and Simulation of a Fair Queueing Algorithm"
2. **vLLM Paper**: Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention"
3. **vLLM Codebase**: https://github.com/vllm-project/vllm

---

## 12. Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-12-28 | 0.1 | Initial design document | Ishrith Gowda |

---

**End of Design Document**
