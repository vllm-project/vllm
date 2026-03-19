# RFC: Dynamic LoRA GPU Capacity with Resolver Plugin Integration

- **Author**: Chen Wang (wangchen615)
- **Status**: Draft
- **Created**: 2026-03-19
- **Related**: [LoRA Resolver Plugins](../lora_resolver_plugins.md), [In-Place LoRA Reloading](../../features/lora.md#in-place-lora-reloading)

---

## Summary

This RFC proposes dynamically adjusting the number of LoRA adapters resident in GPU memory at runtime, controlled via the LoRA resolver plugin interface. Rather than fixing `max_loras` at server startup, the system responds to GPU memory pressure signals — including from external memory managers like [kvcached](https://github.com/ovg-project/kvcached) — to grow or shrink GPU LoRA capacity between batches. Combined with in-place LoRA reloading, this enables full runtime control over both *how many* and *which* LoRA adapters occupy GPU memory without restarting vLLM.

---

## Motivation

### Current Limitations

`max_loras` is a static configuration set at server startup. It controls how many LoRA adapter weight tensors are pre-allocated on the GPU. This creates a fundamental tension:

- **Set too low**: Heavy multi-LoRA workloads queue up; popular LoRAs compete for a small number of slots, increasing latency.
- **Set too high**: GPU memory is wasted on idle LoRA slots when only a few adapters are active, memory that could otherwise be used for KV cache.

### The Opportunity

Two recent vLLM capabilities, when combined, make dynamic LoRA capacity tractable:

1. **LoRA Resolver Plugins** ([docs](../lora_resolver_plugins.md)): Plugins can resolve and load LoRA adapters at request time without server restarts. The resolver is a natural policy point for deciding which LoRAs should be in GPU.

2. **In-Place LoRA Reloading** (`load_inplace=True` on `LoRARequest`): A LoRA already in a GPU slot can be hot-swapped without freeing the slot first, enabling low-overhead adapter rotation.

Additionally, systems like [kvcached](https://github.com/ovg-project/kvcached) dynamically shrink and grow the KV cache on the GPU based on load. When kvcached releases GPU memory (e.g., during low traffic), that memory could be used to load more LoRA adapters. When kvcached reclaims memory (e.g., traffic spike), LoRA slots should be freed to make room.

### Use Cases

- **Bursty multi-LoRA serving**: Scale up GPU slots when many distinct LoRAs are needed; scale down during quiet periods to free memory for KV cache.
- **kvcached co-deployment**: LoRA capacity and KV cache capacity share GPU memory cooperatively rather than competing with fixed allocations.
- **Hot-adapter rotation**: Evict cold LoRAs and swap in hot ones without server restart using `load_inplace=True`.
- **Cost efficiency**: Run more LoRA variants on the same GPU by dynamically time-sharing GPU memory between KV cache and LoRA weights.

---

## Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        vLLM Engine                          │
│                                                             │
│  ┌──────────────┐    post-batch     ┌───────────────────┐   │
│  │  Scheduler   │ ─────────────── ▶ │  _maybe_resize_   │   │
│  │              │                   │  lora_slots()     │   │
│  └──────────────┘                   └────────┬──────────┘   │
│                                              │               │
│                                    ┌─────────▼──────────┐   │
│                                    │   LoRAResolver      │   │
│                                    │ .get_desired_slots()│   │
│                                    └─────────┬──────────┘   │
│                                              │               │
│                                    ┌─────────▼──────────┐   │
│                                    │ WorkerLoRAManager   │   │
│                                    │ .resize_lora_slots()│   │
│                                    └─────────┬──────────┘   │
│                                              │ collective    │
│                              ┌───────────────┼──────────┐   │
│                              ▼               ▼          ▼   │
│                           Worker0         Worker1    ...     │
│                        (TP rank 0)     (TP rank 1)          │
└─────────────────────────────────────────────────────────────┘

         ┌──────────────────────────────────────┐
         │         External Memory Manager       │
         │  (kvcached / KV Cache Manager)        │
         │                                       │
         │  notify(MEMORY_FREED, bytes) ─────────┼──▶ LoRAMemoryNotifier
         │  notify(MEMORY_CLAIMED, bytes) ───────┼──▶ (triggers resize eval)
         └──────────────────────────────────────┘
```

### Design Principles

1. **No static upper bound on LoRA GPU memory**: GPU tensors for LoRA weights are allocated and freed dynamically; there is no `max_loras_limit` pre-allocation.
2. **Policy in plugin, mechanism in core**: The *when* and *how many* decision lives in the `LoRAResolver` plugin. The *how* (reallocation, eviction, TP coordination) lives in vLLM core.
3. **Between-batch only**: All resizing happens strictly between batches to avoid corrupting kernel index metadata during inference.
4. **Safe bounds**: User-configurable `min_loras` and `max_loras` act as floor and ceiling, preventing runaway allocation.
5. **Watermark-based triggering**: Resizing is triggered by GPU memory utilization crossing configurable thresholds, not by batch count.

---

## Detailed Implementation

### 1. `LoRAConfig` Changes

**File**: `vllm/config/lora.py`

Add three new fields:

```python
@dataclass
class LoRAConfig:
    # Existing fields (unchanged)
    max_loras: int = 1
    max_cpu_loras: Optional[int] = None
    max_lora_rank: int = 16
    # ...

    # New fields
    min_loras: int = 1
    """Minimum number of LoRA GPU slots. Acts as floor for dynamic resizing.
    Must be >= 1. Only meaningful when dynamic_lora_slots=True."""

    dynamic_lora_slots: bool = False
    """Enable dynamic resizing of GPU LoRA slots at runtime.
    When True, max_loras becomes the initial value and upper bound.
    Requires a LoRAResolver that implements get_desired_lora_slots()."""

    lora_mem_high_watermark: float = 0.8
    """GPU memory utilization above which LoRA slots are proactively reduced.
    Range: 0.0 - 1.0. Only used when dynamic_lora_slots=True."""

    lora_mem_low_watermark: float = 0.5
    """GPU memory utilization below which LoRA slots may be expanded.
    Range: 0.0 - 1.0. Only used when dynamic_lora_slots=True."""

    def __post_init__(self):
        # Existing validation ...
        if self.dynamic_lora_slots:
            if self.min_loras < 1:
                raise ValueError("min_loras must be >= 1")
            if self.min_loras > self.max_loras:
                raise ValueError("min_loras must be <= max_loras")
            if not (0.0 < self.lora_mem_low_watermark
                    < self.lora_mem_high_watermark < 1.0):
                raise ValueError(
                    "lora_mem_low_watermark must be less than "
                    "lora_mem_high_watermark, both in (0, 1)")
```

---

### 2. `LoRAResolver` Interface Extension

**File**: `vllm/lora/resolver.py`

Extend the abstract base class with an optional method:

```python
class LoRAResolver(ABC):

    @abstractmethod
    async def resolve_lora(
        self,
        base_model_name: str,
        lora_name: str,
    ) -> Optional[LoRARequest]:
        """Resolve a LoRA adapter by name and return a LoRARequest."""
        ...

    async def get_desired_lora_slots(
        self,
        current_slots: int,
        active_loras: list[str],
        free_gpu_memory_bytes: int,
        total_gpu_memory_bytes: int,
    ) -> Optional[int]:
        """
        Optional. Return the desired number of GPU LoRA slots, or None to
        keep the current value.

        Called by the engine between batches when dynamic_lora_slots=True.
        The returned value is clamped to [min_loras, max_loras] by the engine.

        Args:
            current_slots: Currently active GPU LoRA slot count.
            active_loras: Names of LoRAs currently occupying GPU slots.
            free_gpu_memory_bytes: Available GPU memory (from
                torch.cuda.mem_get_info()).
            total_gpu_memory_bytes: Total GPU memory.

        Returns:
            Desired slot count, or None to leave unchanged.
        """
        return None
```

**Backward compatibility**: `get_desired_lora_slots` has a default `return None` implementation, so existing `LoRAResolver` implementations continue to work without changes.

---

### 3. `LoRAMemoryNotifier` (New File)

**File**: `vllm/lora/memory_notifier.py`

A lightweight notification interface for external memory managers:

```python
import asyncio
from enum import Enum
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class GPUMemoryEvent(Enum):
    MEMORY_FREED = "freed"      # External system released GPU memory
    MEMORY_CLAIMED = "claimed"  # External system is about to allocate memory


class LoRAMemoryNotifier:
    """
    Interface for external GPU memory managers (e.g., kvcached) to notify
    the vLLM LoRA subsystem of memory availability changes.

    External systems call notify() when they free or claim GPU memory.
    vLLM evaluates whether to resize LoRA slots in response.

    Usage (from kvcached or similar):
        notifier = LoRAMemoryNotifier.get_instance()
        notifier.notify(GPUMemoryEvent.MEMORY_FREED, bytes_freed)
    """

    _instance: Optional["LoRAMemoryNotifier"] = None

    def __init__(self):
        self._callbacks: list[Callable[[GPUMemoryEvent, int], None]] = []
        self._pending_event: Optional[tuple[GPUMemoryEvent, int]] = None

    @classmethod
    def get_instance(cls) -> "LoRAMemoryNotifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_resize_callback(
        self,
        cb: Callable[[GPUMemoryEvent, int], None],
    ) -> None:
        """Register a callback to be invoked on memory events."""
        self._callbacks.append(cb)

    def notify(self, event: GPUMemoryEvent, bytes_delta: int) -> None:
        """
        Called by external memory manager to signal a memory change.

        Args:
            event: MEMORY_FREED or MEMORY_CLAIMED
            bytes_delta: Approximate bytes freed or claimed
        """
        logger.debug(
            "LoRAMemoryNotifier: event=%s bytes_delta=%d",
            event.value,
            bytes_delta,
        )
        self._pending_event = (event, bytes_delta)
        for cb in self._callbacks:
            try:
                cb(event, bytes_delta)
            except Exception:
                logger.exception("LoRA memory resize callback failed")

    def consume_pending_event(
        self,
    ) -> Optional[tuple[GPUMemoryEvent, int]]:
        """Consume and return any pending event (called by engine loop)."""
        event = self._pending_event
        self._pending_event = None
        return event
```

---

### 4. Per-Layer Tensor Reallocation

**File**: `vllm/lora/layers/base_linear.py`

Add `reallocate_lora_weights()` to `BaseLinearLayerWithLoRA`:

```python
def reallocate_lora_weights(self, new_slots: int) -> None:
    """
    Reallocate stacked LoRA tensors for new_slots GPU slots.
    Copies weights for slots that survive the resize.
    Called between batches only.
    """
    if not hasattr(self, 'lora_a_stacked') or self.lora_a_stacked is None:
        return

    current_slots = self.lora_a_stacked[0].shape[0]
    surviving_slots = min(current_slots, new_slots)

    new_lora_a = []
    new_lora_b = []

    for i, (a, b) in enumerate(
            zip(self.lora_a_stacked, self.lora_b_stacked)):
        new_a = torch.zeros(
            new_slots, *a.shape[1:], dtype=a.dtype, device=a.device)
        new_b = torch.zeros(
            new_slots, *b.shape[1:], dtype=b.dtype, device=b.device)
        # Copy surviving slot weights
        new_a[:surviving_slots].copy_(a[:surviving_slots])
        new_b[:surviving_slots].copy_(b[:surviving_slots])
        new_lora_a.append(new_a)
        new_lora_b.append(new_b)

    # Explicit delete to release GPU memory before allocating new tensors
    del self.lora_a_stacked
    del self.lora_b_stacked
    torch.cuda.empty_cache()

    self.lora_a_stacked = tuple(new_lora_a)
    self.lora_b_stacked = tuple(new_lora_b)
```

The same pattern applies to other LoRA layer types:
- `vllm/lora/layers/column_parallel_linear.py`
- `vllm/lora/layers/row_parallel_linear.py`
- `vllm/lora/layers/logits_processor.py`

---

### 5. `LoRAModelManager.resize_lora_slots()`

**File**: `vllm/lora/model_manager.py`

Add to both `LoRAModelManager` and `LRUCacheLoRAModelManager`:

```python
@property
def lora_slots(self) -> int:
    return self._lora_slots

def resize_lora_slots(self, new_slots: int) -> None:
    """
    Dynamically resize the number of GPU LoRA slots.

    If shrinking: LRU-evicts active LoRAs down to new_slots.
    If growing: allocates additional slots (no LoRAs loaded into them yet).
    Reallocates all per-layer stacked weight tensors.

    Must be called between batches only.

    Args:
        new_slots: Target number of GPU LoRA slots. Must be >= 1.
    """
    if new_slots == self._lora_slots:
        return
    if new_slots < 1:
        raise ValueError(f"new_slots must be >= 1, got {new_slots}")

    logger.info(
        "Resizing LoRA GPU slots: %d -> %d", self._lora_slots, new_slots)

    # Evict active LoRAs if shrinking
    if new_slots < self._lora_slots:
        while len(self._active_adapters) > new_slots:
            evicted_id = self._get_lru_active_adapter_id()
            self.deactivate_adapter(evicted_id)
            logger.debug("Evicted LoRA %d to free GPU slot", evicted_id)

    # Reallocate per-layer tensors
    for module in self.modules.values():
        if hasattr(module, 'reallocate_lora_weights'):
            module.reallocate_lora_weights(new_slots)

    # Resize slot-to-id mapping
    old_mapping = self.lora_index_to_id[:]
    self.lora_index_to_id = old_mapping[:new_slots]
    while len(self.lora_index_to_id) < new_slots:
        self.lora_index_to_id.append(None)

    self._lora_slots = new_slots

    # Re-load surviving active LoRAs into their slots
    for idx, lora_id in enumerate(self.lora_index_to_id):
        if lora_id is not None and lora_id in self._registered_adapters:
            lora_model = self._registered_adapters[lora_id]
            self._load_adapter_to_slot(lora_model, idx)
```

---

### 6. `WorkerLoRAManager` Exposure

**File**: `vllm/lora/worker_manager.py`

Expose `resize_lora_slots` as a public method callable via RPC:

```python
class WorkerLoRAManager:

    def resize_lora_slots(self, new_slots: int) -> None:
        """Resize GPU LoRA slots. Called by engine via collective RPC."""
        self._adapter_manager.resize_lora_slots(new_slots)

    @property
    def lora_slots(self) -> int:
        return self._adapter_manager.lora_slots
```

---

### 7. Engine Integration

**File**: `vllm/v1/engine/llm_engine.py` (or equivalent async engine)

Hook into the post-batch path:

```python
class LLMEngine:

    def __init__(self, ...):
        # Existing init ...
        if self.lora_config and self.lora_config.dynamic_lora_slots:
            self._setup_dynamic_lora()

    def _setup_dynamic_lora(self) -> None:
        """Register memory notifier callback for external systems."""
        notifier = LoRAMemoryNotifier.get_instance()
        notifier.register_resize_callback(self._on_memory_event)

    def _on_memory_event(
        self,
        event: GPUMemoryEvent,
        bytes_delta: int,
    ) -> None:
        """Called by external memory manager (e.g., kvcached)."""
        # Flag for evaluation on next batch boundary
        self._lora_resize_pending = True

    def _maybe_resize_lora_slots(self) -> None:
        """
        Evaluate and apply LoRA slot resize between batches.
        Called after each batch step when dynamic_lora_slots=True.
        """
        if not (self.lora_config and self.lora_config.dynamic_lora_slots):
            return

        # Check for pending external memory event
        notifier = LoRAMemoryNotifier.get_instance()
        pending = notifier.consume_pending_event()
        if pending is None and not self._lora_resize_pending:
            return
        self._lora_resize_pending = False

        free, total = torch.cuda.mem_get_info()
        utilization = 1.0 - free / total
        current = self.executor.collective_rpc_single(
            "get_lora_slots")[0]

        cfg = self.lora_config

        # Watermark-based initial decision
        if utilization > cfg.lora_mem_high_watermark:
            desired: Optional[int] = max(cfg.min_loras, current - 1)
        elif utilization < cfg.lora_mem_low_watermark:
            desired = min(cfg.max_loras, current + 1)
        else:
            desired = None  # In green zone — defer to resolver

        # Ask resolver for fine-grained policy
        resolver_desired = None
        for resolver in LoRAResolverRegistry.get_all_resolvers():
            active_loras = self._get_active_lora_names()
            resolver_desired = asyncio.run_coroutine_threadsafe(
                resolver.get_desired_lora_slots(
                    current_slots=current,
                    active_loras=active_loras,
                    free_gpu_memory_bytes=free,
                    total_gpu_memory_bytes=total,
                ),
                self._event_loop,
            ).result(timeout=1.0)
            if resolver_desired is not None:
                break

        # Resolver overrides watermark decision if provided
        if resolver_desired is not None:
            desired = resolver_desired

        if desired is None or desired == current:
            return

        # Clamp to configured bounds
        desired = max(cfg.min_loras, min(cfg.max_loras, desired))

        if desired != current:
            self._broadcast_resize_lora_slots(desired)

    def _broadcast_resize_lora_slots(self, new_slots: int) -> None:
        """Coordinate resize across all TP workers via collective RPC."""
        self.executor.collective_rpc("resize_lora_slots", args=(new_slots,))
        logger.info("LoRA GPU slots resized to %d", new_slots)
```

---

### 8. Tensor Parallel Coordination

**File**: `vllm/v1/worker/gpu_worker.py`

Each TP worker exposes `resize_lora_slots` as an RPC-callable:

```python
class Worker:

    def resize_lora_slots(self, new_slots: int) -> None:
        """RPC target: resize LoRA slots on this worker."""
        if self.lora_manager is not None:
            self.lora_manager.resize_lora_slots(new_slots)

    def get_lora_slots(self) -> int:
        """RPC target: return current LoRA slot count."""
        if self.lora_manager is not None:
            return self.lora_manager.lora_slots
        return 0
```

All TP workers are called via `collective_rpc`, ensuring they resize in lockstep before the next batch is dispatched.

---

### 9. CudaGraph Handling

**File**: `vllm/v1/cudagraph_dispatcher.py`

```python
def _get_lora_cases(self) -> list[int]:
    if self.lora_config is None:
        return [0]
    if self.lora_config.dynamic_lora_slots:
        # Dynamic slot sizes invalidate captured graphs (tensor addresses
        # change on reallocation). Disable LoRA cudagraph specialization.
        logger.warning(
            "dynamic_lora_slots=True: disabling LoRA cudagraph "
            "specialization. This may reduce throughput slightly.")
        return [0]
    # Existing logic unchanged
    ...
```

---

## What Lives in Core vs Plugin

### Core vLLM (this RFC)

| Component | Responsibility |
|---|---|
| `LoRAConfig` fields | `min_loras`, `dynamic_lora_slots`, watermark thresholds |
| `LoRAModelManager.resize_lora_slots()` | GPU tensor reallocation + LRU eviction |
| `BaseLinearLayerWithLoRA.reallocate_lora_weights()` | Per-layer tensor realloc |
| `WorkerLoRAManager.resize_lora_slots()` | RPC-exposed worker method |
| `LoRAResolver.get_desired_lora_slots()` | Extended interface (default no-op) |
| `LoRAMemoryNotifier` | Pub/sub interface for external memory managers |
| Engine post-batch hook | `_maybe_resize_lora_slots()` |
| TP collective RPC | Lockstep resize across all workers |
| CudaGraph bypass | When `dynamic_lora_slots=True` |

### Plugin Package (e.g., `vllm-dynamic-lora-plugin`)

| Component | Responsibility |
|---|---|
| Custom `LoRAResolver` subclass | Implements `get_desired_lora_slots()` with load-aware policy |
| kvcached integration | Calls `LoRAMemoryNotifier.notify()` on memory events |
| Policy logic | Decides slot counts based on request rates, queue depths, memory |

---

## Example Plugin Implementation

A plugin that integrates with kvcached and scales slots based on request pressure:

```python
# vllm_dynamic_lora_plugin/resolver.py

from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.lora.request import LoRARequest
from vllm.lora.memory_notifier import LoRAMemoryNotifier, GPUMemoryEvent


class DynamicCapacityLoRAResolver(LoRAResolver):

    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> Optional[LoRARequest]:
        # Delegate to filesystem or HF resolver
        ...

    async def get_desired_lora_slots(
        self,
        current_slots: int,
        active_loras: list[str],
        free_gpu_memory_bytes: int,
        total_gpu_memory_bytes: int,
    ) -> Optional[int]:
        """Scale slots based on number of distinct active LoRAs + memory."""
        util = 1.0 - free_gpu_memory_bytes / total_gpu_memory_bytes
        n_active = len(active_loras)

        # Want at least as many slots as distinct active LoRAs,
        # but back off under memory pressure
        if util > 0.85:
            return max(1, n_active - 1)
        elif util < 0.5 and n_active >= current_slots:
            return current_slots + 1
        return None  # No change needed


# kvcached integration
def register_kvcached_notifier():
    """Called by kvcached after shrinking KV cache."""
    notifier = LoRAMemoryNotifier.get_instance()
    # kvcached calls this when it frees memory:
    notifier.notify(GPUMemoryEvent.MEMORY_FREED, bytes_freed)


def register():
    resolver = DynamicCapacityLoRAResolver()
    LoRAResolverRegistry.register_resolver(
        "DynamicCapacityResolver", resolver)
```

---

## Interaction with In-Place LoRA Reloading

In-place LoRA reloading (`load_inplace=True` on `LoRARequest`) hot-swaps a LoRA in an existing GPU slot without freeing and re-allocating the slot. Combined with dynamic slot sizing:

1. **Scale down**: `resize_lora_slots(n - 1)` → evicts LRU LoRA, frees a slot's GPU memory.
2. **Swap**: Resolver returns `LoRARequest(load_inplace=True)` for a new LoRA → replaces a slot's weights in-place (low overhead, no reallocation).
3. **Scale up**: `resize_lora_slots(n + 1)` → allocates a new slot from freed GPU memory.

This gives the resolver fine-grained control over both capacity and which adapters are resident.

---

## Failure Modes and Safety

| Scenario | Handling |
|---|---|
| Resize during batch | Not possible — resize only triggered post-batch |
| Resize to 0 | Clamped to `min_loras` (≥ 1) |
| Resize above `max_loras` | Clamped to `max_loras` |
| OOM during tensor realloc | PyTorch OOM exception propagated; current slots unchanged (best-effort rollback) |
| TP worker resize failure | Collective RPC failure — engine logs error and retains current slots |
| Resolver timeout | `get_desired_lora_slots` call has 1s timeout; skipped if exceeded |
| kvcached notifies while batch running | Event queued; consumed at next batch boundary |

---

## Configuration Reference

```bash
# Enable dynamic LoRA slots
--lora-extra-config '{"dynamic_lora_slots": true, "min_loras": 1, "max_loras": 8}'

# Or via Python
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    lora_extra_config={
        "max_loras": 8,           # Upper bound
        "min_loras": 1,           # Lower bound
        "dynamic_lora_slots": True,
        "lora_mem_high_watermark": 0.80,  # Shrink above 80% GPU utilization
        "lora_mem_low_watermark": 0.50,   # Grow below 50% GPU utilization
    }
)
```

---

## Files to Modify

| File | Change Type | Description |
|---|---|---|
| `vllm/config/lora.py` | Modify | Add `min_loras`, `dynamic_lora_slots`, watermark fields |
| `vllm/lora/resolver.py` | Modify | Add `get_desired_lora_slots()` default method |
| `vllm/lora/memory_notifier.py` | **New** | `LoRAMemoryNotifier` + `GPUMemoryEvent` |
| `vllm/lora/model_manager.py` | Modify | Add `resize_lora_slots()`, `_lora_slots` property |
| `vllm/lora/layers/base_linear.py` | Modify | Add `reallocate_lora_weights()` |
| `vllm/lora/layers/column_parallel_linear.py` | Modify | Add `reallocate_lora_weights()` |
| `vllm/lora/layers/row_parallel_linear.py` | Modify | Add `reallocate_lora_weights()` |
| `vllm/lora/layers/logits_processor.py` | Modify | Add `reallocate_lora_weights()` |
| `vllm/lora/worker_manager.py` | Modify | Expose `resize_lora_slots()` as RPC target |
| `vllm/v1/engine/llm_engine.py` | Modify | Add `_maybe_resize_lora_slots()` post-batch hook |
| `vllm/v1/worker/gpu_worker.py` | Modify | Add `resize_lora_slots()` + `get_lora_slots()` RPC methods |
| `vllm/v1/cudagraph_dispatcher.py` | Modify | Disable LoRA cudagraph when `dynamic_lora_slots=True` |

---

## Open Questions

1. **Resize frequency throttling**: Should there be a minimum time between resizes (e.g., no more than once per second) to avoid thrashing?
2. **Metrics**: Should current `lora_slots` be exposed as a Prometheus gauge so operators can observe dynamic behavior?
3. **PP (Pipeline Parallelism)**: This RFC covers TP coordination. PP adds another dimension — is PP in scope?
4. **Rollback on OOM**: On `torch.cuda.OutOfMemoryError` during reallocation, should we attempt to restore the previous slot count, or just log and keep whatever partial state exists?
5. **kvcached API**: Should `LoRAMemoryNotifier` be the stable integration point, or should we coordinate with the kvcached project on a richer bidirectional interface?
