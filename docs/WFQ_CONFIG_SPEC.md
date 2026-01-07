# WFQ Configuration Specification

**Date**: 2025-12-28
**Author**: Ishrith Gowda

---

## 1. Configuration Changes

### 1.1 SchedulerConfig Extensions

**File**: `vllm/config/scheduler.py`

**Current Policy Type:**
```python
SchedulerPolicy = Literal["fcfs", "priority"]
```

**Updated Policy Type:**
```python
SchedulerPolicy = Literal["fcfs", "priority", "wfq"]
```

**Current policy field:**
```python
@config
@dataclass
class SchedulerConfig:
    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:
    - "fcfs" means first come first served, i.e. requests are handled in order
    of arrival.
    - "priority" means requests are handled based on given priority (lower
    value means earlier handling) and time of arrival deciding any ties)."""
```

**Updated policy field:**
```python
@config
@dataclass
class SchedulerConfig:
    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:
    - "fcfs": First Come First Served - requests handled in order of arrival
    - "priority": Priority-based - lower priority value handled first, with
      arrival time as tiebreaker
    - "wfq": Weighted Fair Queuing - requests scheduled based on virtual finish
      time, providing fairness guarantees proportional to request weights"""
```

### 1.2 Why No Additional WFQ Config Fields?

**Decision**: Do NOT add WFQ-specific configuration fields (e.g., `wfq_default_weight`)

**Rationale:**
1. **Simplicity**: Additional config increases complexity
2. **API-driven**: Weight should be per-request (set by client), not global
3. **Backward Compatibility**: Default weight=1.0 is sufficient
4. **Consistency**: FCFS and Priority policies have no extra config

**Alternative Considered:**
```python
# REJECTED: Too complex, not needed
wfq_default_weight: float = 1.0
wfq_min_weight: float = 0.1
wfq_max_weight: float = 10.0
```

---

## 2. Request Model Extensions

### 2.1 New Attributes

**File**: `vllm/v1/request.py`

**Add to `Request.__init__()`:**
```python
def __init__(
    self,
    request_id: str,
    prompt_token_ids: list[int] | None,
    sampling_params: SamplingParams | None,
    pooling_params: PoolingParams | None,
    eos_token_id: int | None,
    client_index: int = 0,
    arrival_time: float | None = None,
    prompt_embeds: torch.Tensor | None = None,
    mm_features: list[MultiModalFeatureSpec] | None = None,
    lora_request: Optional["LoRARequest"] = None,
    cache_salt: str | None = None,
    priority: int = 0,
    trace_headers: Mapping[str, str] | None = None,
    block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
    weight: float = 1.0,  # NEW: WFQ weight
) -> None:
    # ... existing initialization ...

    # WFQ-specific attributes (initialized after existing attrs)
    self.weight = weight
    self.virtual_start_time: float = 0.0
    self.virtual_finish_time: float = 0.0
```

**Attribute Specifications:**

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `weight` | `float` | `1.0` | WFQ scheduling weight (higher = more resources) |
| `virtual_start_time` | `float` | `0.0` | Virtual time when request started |
| `virtual_finish_time` | `float` | `0.0` | Virtual time when request will finish |

### 2.2 Validation Strategy

**Option A: No validation** (Trust client, raise errors on invalid usage)
- **Selected**: Simpler, consistent with other attributes (priority, arrival_time)

**Option B: Validate in __init__()**
```python
# REJECTED: Too restrictive
if weight <= 0:
    raise ValueError(f"weight must be positive, got {weight}")
if weight > 1000:
    logger.warning(f"Very large weight {weight} may cause issues")
```

**Rationale**: Let WFQRequestQueue handle validation when needed

### 2.3 Backward Compatibility

**Concern**: Existing code creates Request without `weight` parameter

**Solution**: Default parameter `weight=1.0` ensures backward compatibility

**Test**:
```python
# Old code (still works)
req = Request(request_id="1", prompt_token_ids=[1,2,3], ...)

# New code (also works)
req = Request(request_id="1", prompt_token_ids=[1,2,3], weight=2.0, ...)
```

---

## 3. Request Comparison Update

### 3.1 Current __lt__ Implementation

```python
def __lt__(self, other: "Request") -> bool:
    """Compare two requests based on priority, arrival time, and request ID."""
    if self.priority != other.priority:
        return self.priority < other.priority
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time
    if self.request_id != other.request_id:
        return self.request_id < other.request_id
    return id(self) < id(other)
```

### 3.2 Updated __lt__ for WFQ Support

```python
def __lt__(self, other: "Request") -> bool:
    """Compare two requests based on scheduling policy.

    For WFQ: Compare by virtual_finish_time (if set)
    Otherwise: Compare by (priority, arrival_time, request_id)

    This allows the same Request class to work with FCFS, Priority, and WFQ
    schedulers without modification.
    """
    # WFQ: Compare by virtual_finish_time if both requests have it set
    # (virtual_finish_time > 0.0 indicates WFQ has initialized it)
    if (hasattr(self, 'virtual_finish_time') and
        hasattr(other, 'virtual_finish_time') and
        self.virtual_finish_time > 0.0 and
        other.virtual_finish_time > 0.0):
        if self.virtual_finish_time != other.virtual_finish_time:
            return self.virtual_finish_time < other.virtual_finish_time

    # Fallback to original comparison (priority, arrival_time, request_id)
    if self.priority != other.priority:
        return self.priority < other.priority
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time
    if self.request_id != other.request_id:
        return self.request_id < other.request_id
    return id(self) < id(other)
```

**Key Design Decision:**
- Check `hasattr()` for forward/backward compatibility
- Check `> 0.0` to distinguish initialized vs uninitialized
- Fallback to original comparison for FCFS/Priority queues

---

## 4. API Integration

### 4.1 How Clients Specify Weight

**Current API (EngineCoreRequest):**
```python
@dataclass
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams | None
    priority: int = 0
    # ... other fields
```

**Extended API:**
```python
@dataclass
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams | None
    priority: int = 0
    weight: float = 1.0  # NEW
    # ... other fields
```

**Request.from_engine_core_request() update:**
```python
@classmethod
def from_engine_core_request(
    cls,
    request: EngineCoreRequest,
    block_hasher: Callable[["Request"], list["BlockHash"]] | None,
) -> "Request":
    return cls(
        request_id=request.request_id,
        # ... existing fields ...
        priority=request.priority,
        weight=getattr(request, 'weight', 1.0),  # Backward compatible
        # ... rest of fields ...
    )
```

### 4.2 Client Usage Example

```python
# Python client
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m", scheduler_policy="wfq")

# Low priority interactive request (high weight = more resources)
outputs = llm.generate(
    prompts=["Hello"],
    sampling_params=SamplingParams(max_tokens=50),
    weight=2.0,  # NEW: Get more resources
)

# Batch processing request (low weight = fewer resources)
outputs = llm.generate(
    prompts=["Long document..."],
    sampling_params=SamplingParams(max_tokens=500),
    weight=0.5,  # NEW: Be more patient
)
```

---

## 5. Type Annotations

Following vLLM conventions (use `list[T]` not `List[T]`):

```python
# Correct (Python 3.9+ style)
self.weight: float = 1.0
self.virtual_start_time: float = 0.0
self.virtual_finish_time: float = 0.0

# Incorrect (old typing module style) - DO NOT USE
from typing import Optional
self.weight: Optional[float] = None
```

---

## 6. Summary of Changes

| File | Change | Lines Added | Risk |
|------|--------|-------------|------|
| `vllm/config/scheduler.py` | Add "wfq" to SchedulerPolicy | ~2 | Low |
| `vllm/config/scheduler.py` | Update policy docstring | ~5 | Low |
| `vllm/v1/request.py` | Add weight parameter | ~1 | Low |
| `vllm/v1/request.py` | Add WFQ attributes | ~3 | Low |
| `vllm/v1/request.py` | Update __lt__ method | ~10 | Medium |
| `vllm/v1/engine.py` | Add weight to EngineCoreRequest | ~1 | Low |
| **Total** | **22 lines** | **Low-Med** |

**Migration Path**: Zero-change migration (all defaults maintain current behavior)

---

## 7. Open Questions

1. **Should weight be mutable?**
   - No, treat as immutable after initialization (like priority)

2. **Should we log warnings for unusual weights?**
   - No, trust users. WFQRequestQueue can validate if needed

3. **Should weight be included in Request repr/str?**
   - Optional enhancement, not required for MVP

4. **Should virtual times be reset on request reuse?**
   - N/A - Requests are not reused in current architecture

---

**End of Configuration Specification**
