# SAE Eviction Policy Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SAE (Session-Aware Eviction) as a third `CachePolicy` alongside LRU and ARC in vLLM's CPU offload, selectable via `kv_connector_extra_config["eviction_policy"] = "sae"`, and add four per-policy cache-effectiveness counters (`cpu_block_lookup / hit / miss / block_eviction`) with a `policy` label emitted by all three policies.

**Architecture:** SAE implements the existing `CachePolicy` ABC — no new interface methods. Session boundaries are reconstructed from the call sequence (`insert` after `touch`/`evict`/`remove`/`clear` opens a new session; consecutive `insert`s join it). Ghost scores accumulate per-key on `get` calls; periodic decay runs every `decay_interval`-th `get`. The admission gate lives inside `evict(n, protected)`: it computes the would-be new session's score from the ghost sum of `protected` and returns `None` (declining eviction) when that score is worse than the worst incumbent's. Counter plumbing lives in `CPUOffloadingManager` and emits via labelled counters on the existing `OffloadingConnectorStats` pipeline — no changes to `OffloadingConnectorStats` or `OffloadPromMetrics`.

**Tech Stack:** Python 3.10+, `vllm.v1.kv_offload.cpu.policies` (existing `CachePolicy` ABC), `vllm.v1.kv_offload.cpu.manager.CPUOffloadingManager`, `vllm.v1.kv_offload.cpu.spec.CPUOffloadingSpec`, `pytest`.

## Global Constraints

- Python line length 88; docstrings Google style (`Args:`/`Returns:`/`Raises:`).
- Zero behavioral change for `LRUCachePolicy` and `ARCCachePolicy`.
- No new methods on the `CachePolicy` ABC. SAE uses only `get / insert / remove / touch / evict / clear / mark_evictable / mark_non_evictable`.
- All SAE tunables live under `kv_connector_extra_config`; no env-var or TOML config layer.
- Fail-fast validation at `CPUOffloadingSpec.__init__`; all violations raise `ValueError` naming the offending key.
- Every commit must be signed off (`git commit -s`). Include `Assisted-by: Claude` trailer per repo AGENTS.md.
- Run `pre-commit run` on changed files before every commit.
- Use `uv` / `.venv/bin/python` for Python commands — never system `python3` or bare `pip`.

## File Structure

**New files:**
- `vllm/v1/kv_offload/cpu/policies/sae.py` — `SAECachePolicy` class (~280 lines).
- `tests/v1/kv_offload/cpu/policies/__init__.py` — empty test package init.
- `tests/v1/kv_offload/cpu/policies/test_sae_policy.py` — SAE algorithm unit tests.
- `tests/v1/kv_offload/cpu/test_manager_policy_metrics.py` — parametrized counter tests across all three policies.
- `tests/v1/kv_offload/cpu/test_spec_config_validation.py` — `CPUOffloadingSpec` config validation tests.

**Modified files:**
- `vllm/v1/kv_offload/cpu/manager.py` — register `"sae"` in `_CACHE_POLICIES`; widen `cache_policy` Literal; add `policy_kwargs` kwarg; add 4 counter deltas and emit them in `get_stats()`.
- `vllm/v1/kv_offload/cpu/spec.py` — validate `eviction_policy`; extract/validate `sae_*` tunables; add 4 counter definitions; pass `policy_kwargs`; log startup INFO line.
- `docs/features/disagg_prefill.md` or the closest existing CPU-offload doc — small addition noting `"sae"` is an accepted value.

**Untouched:**
- `vllm/v1/kv_offload/cpu/policies/base.py`, `.../lru.py`, `.../arc.py`.
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py` — `OffloadingConnectorStats` and `OffloadPromMetrics` handle labelled counters already.

## Interfaces Established By This Plan

```python
# vllm/v1/kv_offload/cpu/policies/sae.py

class SAECachePolicy(CachePolicy):
    """Session-Aware Eviction cache policy."""

    def __init__(
        self,
        cache_capacity: int,
        *,
        decay_interval: int = 500,
        decay_factor: float = 0.9,
        ghost_hit_weight: float = 12.0,
        ghost_miss_weight: float = 1.0,
        ghost_norm: float = 12.0,
    ) -> None: ...

    # Inherited CachePolicy interface — no new methods.
    def get(self, key: OffloadKey) -> BlockStatus | None: ...
    def insert(self, key: OffloadKey, block: BlockStatus) -> None: ...
    def remove(self, key: OffloadKey) -> None: ...
    def touch(self, keys: Iterable[OffloadKey]) -> None: ...
    def evict(self, n: int, protected: set[OffloadKey]) -> list[tuple[OffloadKey, BlockStatus]] | None: ...
    def clear(self) -> None: ...
    def mark_evictable(self, key: OffloadKey) -> None: ...
    def mark_non_evictable(self, key: OffloadKey) -> None: ...
```

```python
# vllm/v1/kv_offload/cpu/manager.py — added

_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
    "sae": SAECachePolicy,  # new
}

class CPUOffloadingManager(OffloadingManager):
    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc", "sae"] = "lru",  # widened
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        policy_kwargs: dict[str, Any] | None = None,  # new
    ): ...
```

```python
# Counter names emitted by CPUOffloadingManager.get_stats()

CPU_BLOCK_LOOKUP = "vllm:cpu_block_lookup_total"
CPU_BLOCK_HIT = "vllm:cpu_block_hit_total"
CPU_BLOCK_MISS = "vllm:cpu_block_miss_total"
CPU_BLOCK_EVICTION = "vllm:block_eviction_total"
# All four carry a "policy" label.
```

---

## Task 1: SAE policy skeleton — construction, empty state, `get` for a missing key

**Files:**
- Create: `vllm/v1/kv_offload/cpu/policies/sae.py`
- Create: `tests/v1/kv_offload/cpu/policies/__init__.py` (empty)
- Test: `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

**Interfaces:**
- Consumes: `CachePolicy`, `BlockStatus`, `OffloadKey` from existing base module.
- Produces: `SAECachePolicy(cache_capacity, *, decay_interval, decay_factor, ghost_hit_weight, ghost_miss_weight, ghost_norm)` with `.get(key) -> BlockStatus | None`.

- [ ] **Step 1: Create the test package init**

Run: `mkdir -p tests/v1/kv_offload/cpu/policies`

Create `tests/v1/kv_offload/cpu/policies/__init__.py` with a single comment:

```python
# SPDX-License-Identifier: Apache-2.0
```

- [ ] **Step 2: Write the failing test for construction + missing-key lookup**

Create `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.kv_offload.base import OffloadKey, make_offload_key
from vllm.v1.kv_offload.cpu.policies.sae import SAECachePolicy


def key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def test_construction_and_missing_key_returns_none():
    policy = SAECachePolicy(cache_capacity=4)
    assert policy.get(key(1)) is None
```

- [ ] **Step 3: Run test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py::test_construction_and_missing_key_returns_none -v`
Expected: `ImportError` (module `sae` not found).

- [ ] **Step 4: Create the minimal SAE module**

Create `vllm/v1/kv_offload/cpu/policies/sae.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-Aware Eviction (SAE) cache policy for CPU offload.

Ported from the out-of-tree ``sae_kv_offload`` plugin package. Two
documented semantic differences from that reference:

1. Session boundaries are reconstructed from the call sequence
   (``insert`` after ``touch``/``evict``/``remove``/``clear`` opens a
   new session; consecutive ``insert`` calls join it) rather than
   taken from a batch of block hashes handed to ``prepare_store``.
2. Per-batch position weighting on ``get`` is dropped because the
   current scheduler calls ``manager.lookup`` one key at a time.
"""

from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class SAECachePolicy(CachePolicy):
    """Session-Aware Eviction cache policy.

    See the module docstring for the two semantic differences from
    the reference algorithm.
    """

    def __init__(
        self,
        cache_capacity: int,
        *,
        decay_interval: int = 500,
        decay_factor: float = 0.9,
        ghost_hit_weight: float = 12.0,
        ghost_miss_weight: float = 1.0,
        ghost_norm: float = 12.0,
    ) -> None:
        self._cache_capacity = cache_capacity
        self._decay_interval = decay_interval
        self._decay_factor = decay_factor
        self._ghost_hit_weight = ghost_hit_weight
        self._ghost_miss_weight = ghost_miss_weight
        self._ghost_norm = ghost_norm

        self._blocks: dict[OffloadKey, BlockStatus] = {}
        self._sid_to_keys: dict[int, list[OffloadKey]] = {}
        self._key_to_sid: dict[OffloadKey, int] = {}
        self._sid_stats: dict[int, dict[str, float | int]] = {}
        self._key_ghost: dict[OffloadKey, float] = {}
        self._evictable_keys: OrderedDict[OffloadKey, None] = OrderedDict()

        self._logical_timer: int = 0
        self._sid_counter: int = 0
        self._lookup_count: int = 0
        self._open_sid: int | None = None
        self._last_event: str = "init"

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self._blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        raise NotImplementedError

    @override
    def remove(self, key: OffloadKey) -> None:
        raise NotImplementedError

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        raise NotImplementedError

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        raise NotImplementedError

    @override
    def clear(self) -> None:
        raise NotImplementedError
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py::test_construction_and_missing_key_returns_none -v`
Expected: PASS.

- [ ] **Step 6: Run pre-commit and fix any lint**

Run: `pre-commit run --files vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/__init__.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py`
Fix any reported issues.

- [ ] **Step 7: Commit**

```bash
git add vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/__init__.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py
git commit -s -m "$(cat <<'EOF'
feat(kv_offload): scaffold SAECachePolicy skeleton

Skeleton class implementing CachePolicy with construction and
missing-key lookup only. Remaining methods raise NotImplementedError
and will be filled in by subsequent tasks.

Assisted-by: Claude
EOF
)"
```

---

## Task 2: `insert` and `remove` — session boundary detection

**Files:**
- Modify: `vllm/v1/kv_offload/cpu/policies/sae.py`
- Test: `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

**Interfaces:**
- Consumes: `SAECachePolicy` skeleton from Task 1.
- Produces: `insert` and `remove` behavior; `_open_sid`, `_last_event`, `_sid_to_keys`, `_key_to_sid`, `_sid_stats` maintained correctly.

- [ ] **Step 1: Write failing tests**

Append to `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`:

```python
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus


def make_block(block_id: int) -> BlockStatus:
    return BlockStatus(block_id)


def test_first_insert_opens_new_session():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    assert policy._open_sid == 0
    assert policy._sid_to_keys == {0: [key(1)]}
    assert policy._key_to_sid == {key(1): 0}


def test_consecutive_inserts_join_open_session():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.insert(key(2), make_block(1))
    assert policy._open_sid == 0
    assert policy._sid_to_keys == {0: [key(1), key(2)]}


def test_remove_closes_open_session_and_cleans_state():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.remove(key(1))
    assert policy._open_sid is None
    assert policy._sid_to_keys == {}
    assert policy._key_to_sid == {}
    assert policy.get(key(1)) is None


def test_insert_after_remove_opens_new_session():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.remove(key(1))
    policy.insert(key(2), make_block(1))
    assert policy._open_sid == 1  # sid_counter incremented
    assert policy._sid_to_keys == {1: [key(2)]}


def test_insert_seeds_initial_hits_from_ghost_sum():
    policy = SAECachePolicy(cache_capacity=4, ghost_norm=2.0)
    policy._key_ghost[key(1)] = 4.0
    policy._key_ghost[key(2)] = 6.0
    policy.insert(key(1), make_block(0))
    policy.insert(key(2), make_block(1))
    # (4.0 + 6.0) / 2.0 = 5.0
    assert policy._sid_stats[0]["hits"] == 5.0
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v`
Expected: 4 failures (NotImplementedError) plus 1 pass (missing-key from Task 1).

- [ ] **Step 3: Implement `insert` and `remove`**

Replace the `insert` and `remove` stubs in `vllm/v1/kv_offload/cpu/policies/sae.py` with:

```python
    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        if self._last_event != "insert" or self._open_sid is None:
            sid = self._sid_counter
            self._sid_counter += 1
            self._open_sid = sid
            self._sid_to_keys[sid] = []
            self._sid_stats[sid] = {
                "hits": 0.0,
                "last_touch": self._logical_timer,
                "start_pos": 0,
            }
        sid = self._open_sid
        self._sid_to_keys[sid].append(key)
        self._key_to_sid[key] = sid
        self._blocks[key] = block
        seed = self._key_ghost.get(key, 0.0) / self._ghost_norm
        self._sid_stats[sid]["hits"] += seed
        self._last_event = "insert"

    @override
    def remove(self, key: OffloadKey) -> None:
        block = self._blocks.pop(key, None)
        if block is None:
            self._last_event = "remove"
            self._open_sid = None
            return
        sid = self._key_to_sid.pop(key, None)
        if sid is not None:
            seq = self._sid_to_keys.get(sid)
            if seq is not None and key in seq:
                seq.remove(key)
            if not seq:
                self._sid_to_keys.pop(sid, None)
                self._sid_stats.pop(sid, None)
        self._evictable_keys.pop(key, None)
        self._open_sid = None
        self._last_event = "remove"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v`
Expected: 5 passes.

- [ ] **Step 5: Run pre-commit**

Run: `pre-commit run --files vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

- [ ] **Step 6: Commit**

```bash
git add vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py
git commit -s -m "$(cat <<'EOF'
feat(kv_offload): SAE insert/remove with session boundary detection

First insert opens a session; consecutive inserts join it; touch/
evict/remove/clear close it. initial_hits is seeded from ghost sum
incrementally per insert.

Assisted-by: Claude
EOF
)"
```

---

## Task 3: `touch` and `clear` — session close and hit accounting

**Files:**
- Modify: `vllm/v1/kv_offload/cpu/policies/sae.py`
- Test: `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

**Interfaces:**
- Consumes: session state from Task 2.
- Produces: `touch(keys)` bumps `hits` and `last_touch`; `clear()` resets all state; both close the open session.

- [ ] **Step 1: Write failing tests**

Append to `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`:

```python
def test_touch_bumps_hits_and_last_touch_and_closes_session():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.insert(key(2), make_block(1))
    policy.touch([key(1), key(2)])
    sid = 0
    assert policy._sid_stats[sid]["hits"] == 2.0
    assert policy._sid_stats[sid]["last_touch"] == 1
    assert policy._open_sid is None
    # Next insert opens a fresh session
    policy.insert(key(3), make_block(2))
    assert policy._open_sid == 1


def test_touch_ignores_unknown_keys():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.touch([key(1), key(99)])
    assert policy._sid_stats[0]["hits"] == 1.0


def test_clear_resets_all_state():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.touch([key(1)])
    policy._key_ghost[key(2)] = 5.0
    policy.clear()
    assert policy._blocks == {}
    assert policy._sid_to_keys == {}
    assert policy._key_to_sid == {}
    assert policy._sid_stats == {}
    assert policy._key_ghost == {}
    assert policy._evictable_keys == OrderedDict()
    assert policy._open_sid is None
    assert policy._last_event == "clear"
```

Add `from collections import OrderedDict` to the test file imports if not already there.

- [ ] **Step 2: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v`
Expected: 3 new failures (NotImplementedError on touch/clear).

- [ ] **Step 3: Implement `touch` and `clear`**

Replace stubs in `vllm/v1/kv_offload/cpu/policies/sae.py`:

```python
    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        self._logical_timer += 1
        touched_sids: set[int] = set()
        for k in keys:
            sid = self._key_to_sid.get(k)
            if sid is None:
                continue
            stats = self._sid_stats[sid]
            stats["hits"] = stats["hits"] + 1
            stats["last_touch"] = self._logical_timer
            touched_sids.add(sid)
        self._open_sid = None
        self._last_event = "touch"

    @override
    def clear(self) -> None:
        self._blocks.clear()
        self._sid_to_keys.clear()
        self._key_to_sid.clear()
        self._sid_stats.clear()
        self._key_ghost.clear()
        self._evictable_keys.clear()
        self._logical_timer = 0
        self._sid_counter = 0
        self._lookup_count = 0
        self._open_sid = None
        self._last_event = "clear"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v`
Expected: all pass.

- [ ] **Step 5: Run pre-commit**

Run: `pre-commit run --files vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

- [ ] **Step 6: Commit**

```bash
git add vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py
git commit -s -m "$(cat <<'EOF'
feat(kv_offload): SAE touch/clear

touch bumps per-session hits and last_touch; clear resets state.
Both close the currently-open session.

Assisted-by: Claude
EOF
)"
```

---

## Task 4: `get` overrides — ghost score accumulation and periodic decay

**Files:**
- Modify: `vllm/v1/kv_offload/cpu/policies/sae.py`
- Test: `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

**Interfaces:**
- Consumes: state from Tasks 1–3.
- Produces: `get(key)` accumulates ghost scores; decay runs every `decay_interval`-th call, scaling `hits` and `_key_ghost` by `decay_factor` and pruning non-resident ghost entries below `0.01`.

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
def test_get_hit_accumulates_ghost_score():
    policy = SAECachePolicy(cache_capacity=4, ghost_hit_weight=3.0)
    policy.insert(key(1), make_block(0))
    policy.get(key(1))
    policy.get(key(1))
    assert policy._key_ghost[key(1)] == 6.0


def test_get_miss_accumulates_ghost_score():
    policy = SAECachePolicy(cache_capacity=4, ghost_miss_weight=0.5)
    policy.get(key(1))
    policy.get(key(1))
    assert policy._key_ghost[key(1)] == 1.0


def test_decay_runs_every_interval_and_prunes_low_ghosts():
    policy = SAECachePolicy(
        cache_capacity=4,
        decay_interval=3,
        decay_factor=0.5,
        ghost_hit_weight=1.0,
        ghost_miss_weight=1.0,
    )
    policy.insert(key(1), make_block(0))
    policy._sid_stats[0]["hits"] = 10.0
    policy._key_ghost[key(2)] = 0.05  # non-resident
    policy._key_ghost[key(1)] = 4.0   # resident
    # 3 gets triggers one decay tick
    policy.get(key(1))
    policy.get(key(1))
    policy.get(key(1))
    assert policy._sid_stats[0]["hits"] == 5.0
    # resident key(1) accumulated 3 hits before decay: (4.0 + 3.0) * 0.5 = 3.5
    assert policy._key_ghost[key(1)] == 3.5
    # non-resident key(2) with score 0.05 -> 0.025 < 0.01? no, still 0.025 >= 0.01
    # Adjust: set higher threshold test
    assert key(2) in policy._key_ghost


def test_decay_prunes_below_threshold():
    policy = SAECachePolicy(
        cache_capacity=4,
        decay_interval=1,
        decay_factor=0.1,
    )
    policy._key_ghost[key(99)] = 0.05  # non-resident
    policy.get(key(1))  # triggers decay; 0.05 * 0.1 = 0.005 < 0.01
    assert key(99) not in policy._key_ghost
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v -k "get_hit or get_miss or decay"`
Expected: 4 failures (assertion / missing behavior).

- [ ] **Step 3: Implement `get` override with ghost accumulation and decay**

Replace the current `get` in `vllm/v1/kv_offload/cpu/policies/sae.py`:

```python
    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        block = self._blocks.get(key)
        if block is not None and block.is_ready:
            self._key_ghost[key] = (
                self._key_ghost.get(key, 0.0) + self._ghost_hit_weight
            )
        else:
            self._key_ghost[key] = (
                self._key_ghost.get(key, 0.0) + self._ghost_miss_weight
            )
        self._lookup_count += 1
        if self._lookup_count % self._decay_interval == 0:
            self._run_decay()
        self._last_event = "get"
        return block

    def _run_decay(self) -> None:
        for stats in self._sid_stats.values():
            stats["hits"] = stats["hits"] * self._decay_factor
        for k in list(self._key_ghost):
            new_score = self._key_ghost[k] * self._decay_factor
            if k not in self._blocks and new_score < 0.01:
                del self._key_ghost[k]
            else:
                self._key_ghost[k] = new_score
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v`
Expected: all pass.

- [ ] **Step 5: Run pre-commit**

Run: `pre-commit run --files vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

- [ ] **Step 6: Commit**

```bash
git add vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py
git commit -s -m "$(cat <<'EOF'
feat(kv_offload): SAE get with ghost score accumulation and decay

Every get() call adds ghost_hit_weight (resident+ready) or
ghost_miss_weight (otherwise) to _key_ghost. Every decay_interval
calls, session hits and ghost scores decay by decay_factor and
non-resident entries below 0.01 are pruned.

Assisted-by: Claude
EOF
)"
```

---

## Task 5: `mark_evictable` / `mark_non_evictable` — evictable-key tracking

**Files:**
- Modify: `vllm/v1/kv_offload/cpu/policies/sae.py`
- Test: `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

**Interfaces:**
- Consumes: skeleton from Task 1 (`_evictable_keys` field).
- Produces: `mark_evictable(key)` adds to `_evictable_keys`; `mark_non_evictable(key)` removes.

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
def test_mark_evictable_adds_to_evictable_set():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.mark_evictable(key(1))
    assert key(1) in policy._evictable_keys


def test_mark_non_evictable_removes_from_evictable_set():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), make_block(0))
    policy.mark_evictable(key(1))
    policy.mark_non_evictable(key(1))
    assert key(1) not in policy._evictable_keys


def test_mark_non_evictable_missing_key_is_safe():
    policy = SAECachePolicy(cache_capacity=4)
    # Should not raise
    policy.mark_non_evictable(key(99))
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v -k "mark_"`
Expected: 3 failures (base-class no-op means `_evictable_keys` is untouched, so `key(1) in policy._evictable_keys` is False after `mark_evictable`).

- [ ] **Step 3: Add overrides**

Append to `SAECachePolicy` in `vllm/v1/kv_offload/cpu/policies/sae.py`:

```python
    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self._evictable_keys[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        self._evictable_keys.pop(key, None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v`
Expected: all pass.

- [ ] **Step 5: Run pre-commit and commit**

Run: `pre-commit run --files vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

```bash
git add vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py
git commit -s -m "$(cat <<'EOF'
feat(kv_offload): SAE mark_evictable / mark_non_evictable

Tracks keys with ref_cnt == 0 in an OrderedDict for eviction
candidate scans.

Assisted-by: Claude
EOF
)"
```

---

## Task 6: `evict` — worst-first session walk with admission gate

**Files:**
- Modify: `vllm/v1/kv_offload/cpu/policies/sae.py`
- Test: `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

**Interfaces:**
- Consumes: session state, ghost scores, `_evictable_keys` from Tasks 2–5.
- Produces: `evict(n, protected)` returns a list of `n` (key, block) tuples selected worst-session-first from idle, non-protected blocks; returns `None` when the admission gate denies or when `n` cannot be satisfied.

- [ ] **Step 1: Write failing tests**

Append to the test file:

```python
def score_of(policy: SAECachePolicy, sid: int) -> float:
    stats = policy._sid_stats[sid]
    pos_bonus = 30000.0 / (1.0 + stats["start_pos"] / 8.0)
    freq_bonus = stats["hits"] * 1500.0
    return stats["last_touch"] + freq_bonus + pos_bonus


def _make_ready_block(block_id: int) -> BlockStatus:
    b = BlockStatus(block_id)
    b.ref_cnt = 0
    return b


def test_evict_returns_empty_when_n_zero():
    policy = SAECachePolicy(cache_capacity=4)
    assert policy.evict(0, set()) == []


def test_evict_returns_none_when_insufficient_idle_blocks():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), _make_ready_block(0))
    policy.mark_evictable(key(1))
    # Only 1 idle block, but need 2
    assert policy.evict(2, set()) is None


def test_evict_walks_sessions_worst_first():
    policy = SAECachePolicy(cache_capacity=4)
    # Session 0 (worst): low hits
    policy.insert(key(1), _make_ready_block(0))
    policy.mark_evictable(key(1))
    policy.touch([])  # close session 0 without bumping hits
    # Session 1 (best): high hits
    policy.insert(key(2), _make_ready_block(1))
    policy.mark_evictable(key(2))
    policy._sid_stats[1]["hits"] = 1000.0
    # Evict 1 -> should come from session 0
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert len(evicted) == 1
    assert evicted[0][0] == key(1)


def test_evict_skips_protected_keys():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), _make_ready_block(0))
    policy.mark_evictable(key(1))
    assert policy.evict(1, {key(1)}) is None


def test_evict_skips_non_evictable_keys():
    policy = SAECachePolicy(cache_capacity=4)
    b = BlockStatus(0)
    b.ref_cnt = 1  # not evictable (ref_cnt != 0)
    policy.insert(key(1), b)
    # Never marked evictable
    assert policy.evict(1, set()) is None


def test_evict_removes_evicted_keys_from_all_state():
    policy = SAECachePolicy(cache_capacity=4)
    policy.insert(key(1), _make_ready_block(0))
    policy.mark_evictable(key(1))
    evicted = policy.evict(1, set())
    assert evicted is not None
    assert key(1) not in policy._blocks
    assert key(1) not in policy._key_to_sid
    assert key(1) not in policy._evictable_keys
    # sid 0 was emptied, so it should be gone
    assert 0 not in policy._sid_to_keys


def test_evict_admission_gate_denies_when_new_session_score_below_worst():
    policy = SAECachePolicy(cache_capacity=4, ghost_norm=1.0)
    # Incumbent session with very high score
    policy.insert(key(1), _make_ready_block(0))
    policy.mark_evictable(key(1))
    policy._sid_stats[0]["hits"] = 10000.0
    # Would-be new session has 0 ghost score -> gate denies
    result = policy.evict(1, {key(2)})
    assert result is None


def test_evict_admission_gate_allows_when_new_session_score_above_worst():
    policy = SAECachePolicy(cache_capacity=4, ghost_norm=1.0)
    policy.insert(key(1), _make_ready_block(0))
    policy.mark_evictable(key(1))
    # Incumbent has low hits
    policy._sid_stats[0]["hits"] = 0.0
    # New session has strong ghost score
    policy._key_ghost[key(2)] = 100.0
    result = policy.evict(1, {key(2)})
    assert result is not None
    assert len(result) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v -k "evict"`
Expected: multiple failures (NotImplementedError).

- [ ] **Step 3: Implement `evict`**

Replace the `evict` stub in `vllm/v1/kv_offload/cpu/policies/sae.py`:

```python
    def _score(self, sid: int) -> float:
        stats = self._sid_stats[sid]
        pos_bonus = 30000.0 / (1.0 + stats["start_pos"] / 8.0)
        freq_bonus = stats["hits"] * 1500.0
        return stats["last_touch"] + freq_bonus + pos_bonus

    def _admission_gate_allows(self, protected: set[OffloadKey]) -> bool:
        if not self._sid_stats:
            return True
        ghost_sum = sum(self._key_ghost.get(k, 0.0) for k in protected)
        new_hits = ghost_sum / self._ghost_norm
        new_score = self._logical_timer + new_hits * 1500.0 + 30000.0
        worst_sid = min(self._sid_stats.keys(), key=self._score)
        return new_score >= self._score(worst_sid)

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        self._open_sid = None
        self._last_event = "evict"
        if n == 0:
            return []
        if not self._admission_gate_allows(protected):
            return None

        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        # Walk sessions worst-first
        sorted_sids = sorted(self._sid_stats.keys(), key=self._score)
        for sid in sorted_sids:
            if len(candidates) == n:
                break
            seq = self._sid_to_keys.get(sid, [])
            # Walk from the tail of the session backwards
            for k in reversed(seq):
                if len(candidates) == n:
                    break
                if k in protected:
                    continue
                if k not in self._evictable_keys:
                    continue
                block = self._blocks.get(k)
                if block is None or block.ref_cnt != 0:
                    continue
                candidates.append((k, block))

        if len(candidates) < n:
            return None

        # Apply evictions
        for k, _ in candidates:
            sid = self._key_to_sid.pop(k, None)
            if sid is not None:
                seq = self._sid_to_keys.get(sid)
                if seq is not None and k in seq:
                    seq.remove(k)
                if seq is not None and not seq:
                    self._sid_to_keys.pop(sid, None)
                    self._sid_stats.pop(sid, None)
            self._blocks.pop(k, None)
            self._evictable_keys.pop(k, None)
        return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v`
Expected: all pass.

- [ ] **Step 5: Run pre-commit**

Run: `pre-commit run --files vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

- [ ] **Step 6: Commit**

```bash
git add vllm/v1/kv_offload/cpu/policies/sae.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py
git commit -s -m "$(cat <<'EOF'
feat(kv_offload): SAE evict with admission gate and worst-first walk

evict(n, protected) runs the admission gate (returns None when the
would-be new session's score is below the worst incumbent's) and
otherwise walks sessions sorted by SAE's score function worst-first,
yielding idle non-protected keys from each session's tail until n
are collected.

Assisted-by: Claude
EOF
)"
```

---

## Task 7: Register SAE in `_CACHE_POLICIES` and add `policy_kwargs`

**Files:**
- Modify: `vllm/v1/kv_offload/cpu/manager.py`
- Test: `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

**Interfaces:**
- Consumes: `SAECachePolicy` from Tasks 1–6.
- Produces: `CPUOffloadingManager(cache_policy="sae", policy_kwargs={"decay_interval": ...}, ...)` constructs a manager whose `_policy` is `SAECachePolicy` with the specified tunables.

- [ ] **Step 1: Write failing test**

Append to `tests/v1/kv_offload/cpu/policies/test_sae_policy.py`:

```python
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


def test_cpu_offloading_manager_accepts_sae_policy_and_kwargs():
    mgr = CPUOffloadingManager(
        num_blocks=4,
        cache_policy="sae",
        policy_kwargs={"decay_interval": 42, "decay_factor": 0.5},
    )
    assert isinstance(mgr._policy, SAECachePolicy)
    assert mgr._policy._decay_interval == 42
    assert mgr._policy._decay_factor == 0.5


def test_cpu_offloading_manager_defaults_still_work_for_lru():
    mgr = CPUOffloadingManager(num_blocks=4, cache_policy="lru")
    assert mgr._policy is not None
    # LRU still works — sanity check
    from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
    assert isinstance(mgr._policy, LRUCachePolicy)
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py::test_cpu_offloading_manager_accepts_sae_policy_and_kwargs -v`
Expected: ValueError ("Unknown cache policy: 'sae'") or TypeError (unexpected `policy_kwargs`).

- [ ] **Step 3: Modify `CPUOffloadingManager`**

Edit `vllm/v1/kv_offload/cpu/manager.py`:

At the top of the file, replace the import block for policies:

```python
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.cpu.policies.sae import SAECachePolicy
```

Replace the `_CACHE_POLICIES` mapping:

```python
_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
    "sae": SAECachePolicy,
}
```

Replace the `__init__` signature and policy construction (currently around lines 46-65):

```python
    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc", "sae"] = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        policy_kwargs: dict[str, object] | None = None,
    ):
        self.medium: str = CPULoadStoreSpec.medium()
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = _CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy: {cache_policy!r}. "
                f"Supported: {list(_CACHE_POLICIES)}"
            )
        kwargs = policy_kwargs or {}
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks, **kwargs)
        self._policy_name: str = cache_policy
        # Track the number of blocks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_blocks: int = 0
```

(Leave the rest of `__init__` — `store_threshold`, `max_tracker_size`, `counts` — unchanged.)

Add the required import at the top if not present:

```python
from typing import Any, Literal
```

(Check the file — `Literal` should already be imported; add `Any` if missing.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v`
Expected: all pass.

- [ ] **Step 5: Run existing CPUOffloadingManager tests to confirm no regression**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_manager.py -v`
Expected: all pass.

- [ ] **Step 6: Run pre-commit**

Run: `pre-commit run --files vllm/v1/kv_offload/cpu/manager.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py`

- [ ] **Step 7: Commit**

```bash
git add vllm/v1/kv_offload/cpu/manager.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py
git commit -s -m "$(cat <<'EOF'
feat(kv_offload): register SAE in _CACHE_POLICIES and add policy_kwargs

CPUOffloadingManager now accepts cache_policy="sae" and forwards
policy_kwargs to the CachePolicy constructor. LRU/ARC ignore the
kwargs (default empty dict).

Assisted-by: Claude
EOF
)"
```

---

## Task 8: Four per-policy counters — manager-side tallies and stats emission

**Files:**
- Modify: `vllm/v1/kv_offload/cpu/manager.py`
- Modify: `vllm/v1/kv_offload/cpu/common.py`
- Create: `tests/v1/kv_offload/cpu/test_manager_policy_metrics.py`

**Interfaces:**
- Consumes: `CPUOffloadingManager` from Task 7.
- Produces: `get_stats()` emits `vllm:cpu_block_lookup_total`, `vllm:cpu_block_hit_total`, `vllm:cpu_block_miss_total`, `vllm:block_eviction_total` — each with `labelvalues=(policy_name,)`. Deltas reset each call.

- [ ] **Step 1: Locate `CPUOffloadingMetrics` and add four new constants**

Read `vllm/v1/kv_offload/cpu/common.py` to find the `CPUOffloadingMetrics` class. Add four new class attributes (adjacent to the existing `CPU_CACHE_USAGE_PERC` and `STORES_SKIPPED`):

```python
    CPU_BLOCK_LOOKUP = "vllm:cpu_block_lookup_total"
    CPU_BLOCK_HIT = "vllm:cpu_block_hit_total"
    CPU_BLOCK_MISS = "vllm:cpu_block_miss_total"
    BLOCK_EVICTION = "vllm:block_eviction_total"
```

- [ ] **Step 2: Write failing counter-emission tests**

Create `tests/v1/kv_offload/cpu/test_manager_policy_metrics.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


def key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


REQ = ReqContext(req_id="test")


@pytest.mark.parametrize("policy", ["lru", "arc", "sae"])
def test_stats_emit_four_counters_with_policy_label(policy: str):
    mgr = CPUOffloadingManager(num_blocks=4, cache_policy=policy)
    mgr.on_new_request(REQ)
    # Two misses
    assert mgr.lookup(key(1), REQ) == LookupResult.MISS
    assert mgr.lookup(key(2), REQ) == LookupResult.MISS

    stats = mgr.get_stats()
    data = stats.data["data"]
    assert data.get(CPUOffloadingMetrics.CPU_BLOCK_LOOKUP, {}).get((policy,)) == 2
    assert data.get(CPUOffloadingMetrics.CPU_BLOCK_HIT, {}).get((policy,)) == 0
    assert data.get(CPUOffloadingMetrics.CPU_BLOCK_MISS, {}).get((policy,)) == 2
    assert data.get(CPUOffloadingMetrics.BLOCK_EVICTION, {}).get((policy,)) == 0


@pytest.mark.parametrize("policy", ["lru", "arc", "sae"])
def test_stats_hits_plus_misses_equals_lookups(policy: str):
    mgr = CPUOffloadingManager(num_blocks=4, cache_policy=policy)
    mgr.on_new_request(REQ)
    # Fill one block so a hit is possible
    ps = mgr.prepare_store([key(1)], REQ)
    assert ps is not None
    mgr.complete_store([key(1)], REQ, success=True)
    # 1 hit, 2 misses -> 3 lookups
    mgr.lookup(key(1), REQ)
    mgr.lookup(key(2), REQ)
    mgr.lookup(key(3), REQ)

    stats = mgr.get_stats()
    data = stats.data["data"]
    lookups = data.get(CPUOffloadingMetrics.CPU_BLOCK_LOOKUP, {}).get((policy,), 0)
    hits = data.get(CPUOffloadingMetrics.CPU_BLOCK_HIT, {}).get((policy,), 0)
    misses = data.get(CPUOffloadingMetrics.CPU_BLOCK_MISS, {}).get((policy,), 0)
    assert lookups == hits + misses


@pytest.mark.parametrize("policy", ["lru", "arc", "sae"])
def test_stats_deltas_reset_each_call(policy: str):
    mgr = CPUOffloadingManager(num_blocks=4, cache_policy=policy)
    mgr.on_new_request(REQ)
    mgr.lookup(key(1), REQ)  # MISS
    mgr.get_stats()  # flush
    stats = mgr.get_stats()
    data = stats.data["data"]
    # No new activity → counter for this policy label should not appear or be 0
    assert data.get(CPUOffloadingMetrics.CPU_BLOCK_LOOKUP, {}).get((policy,), 0) == 0
```

- [ ] **Step 3: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_manager_policy_metrics.py -v`
Expected: multiple failures (`AttributeError` on the constants, or the counters aren't emitted).

- [ ] **Step 4: Add counter tallies to `CPUOffloadingManager`**

In `vllm/v1/kv_offload/cpu/manager.py`, add these fields at the end of `__init__`:

```python
        self._lookups_delta: int = 0
        self._hits_delta: int = 0
        self._misses_delta: int = 0
        self._evictions_delta: int = 0
```

Modify `lookup()` (currently around lines 116-130). After the existing lookup logic that returns `LookupResult.MISS`, `HIT`, `HIT_PENDING`, or `RETRY`, wrap it so we count. Change the method body to:

```python
    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            result = LookupResult.MISS
        elif not block.is_ready:
            result = LookupResult.HIT_PENDING
        else:
            result = LookupResult.HIT
        # Counter tallies (HIT_PENDING counts as HIT; RETRY is not emitted here).
        if result == LookupResult.MISS:
            self._lookups_delta += 1
            self._misses_delta += 1
        elif result in (LookupResult.HIT, LookupResult.HIT_PENDING):
            self._lookups_delta += 1
            self._hits_delta += 1
        return result
```

Modify `prepare_store()` — after `evicted = self._policy.evict(...)` returns a non-None list, tally the count. Locate the block around line 201-203 and change:

```python
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None
            self._evictions_delta += len(evicted)
```

Modify `get_stats()` — after the existing counter emissions, add the four new counters:

```python
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        # Compute cache usage.
        num_used = (
            self._num_allocated_blocks
            - len(self._free_list)
            - self._num_evictable_cache_blocks
        )
        usage = num_used / self._num_blocks if self._num_blocks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        # Per-policy cache effectiveness counters.
        policy_label = (self._policy_name,)
        stats.increase_counter(
            CPUOffloadingMetrics.CPU_BLOCK_LOOKUP,
            self._lookups_delta,
            labelvalues=policy_label,
        )
        stats.increase_counter(
            CPUOffloadingMetrics.CPU_BLOCK_HIT,
            self._hits_delta,
            labelvalues=policy_label,
        )
        stats.increase_counter(
            CPUOffloadingMetrics.CPU_BLOCK_MISS,
            self._misses_delta,
            labelvalues=policy_label,
        )
        stats.increase_counter(
            CPUOffloadingMetrics.BLOCK_EVICTION,
            self._evictions_delta,
            labelvalues=policy_label,
        )
        self._lookups_delta = 0
        self._hits_delta = 0
        self._misses_delta = 0
        self._evictions_delta = 0
        return stats
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_manager_policy_metrics.py -v`
Expected: all pass.

- [ ] **Step 6: Run existing manager tests for regression**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_manager.py tests/v1/kv_offload/cpu/policies/test_sae_policy.py -v`
Expected: all pass.

- [ ] **Step 7: Run pre-commit**

Run: `pre-commit run --files vllm/v1/kv_offload/cpu/manager.py vllm/v1/kv_offload/cpu/common.py tests/v1/kv_offload/cpu/test_manager_policy_metrics.py`

- [ ] **Step 8: Commit**

```bash
git add vllm/v1/kv_offload/cpu/manager.py vllm/v1/kv_offload/cpu/common.py tests/v1/kv_offload/cpu/test_manager_policy_metrics.py
git commit -s -m "$(cat <<'EOF'
feat(kv_offload): per-policy cache effectiveness counters

CPUOffloadingManager now tallies lookups/hits/misses/evictions per
call cycle and emits them via get_stats() as four labelled Prometheus
counters (vllm:cpu_block_lookup_total, cpu_block_hit_total,
cpu_block_miss_total, block_eviction_total), each carrying a
"policy" label so all three policies (lru/arc/sae) surface uniformly
on a single dashboard. HIT_PENDING counts as a hit; RETRY does not
increment lookups.

Assisted-by: Claude
EOF
)"
```

---

## Task 9: `CPUOffloadingSpec` — validation, tunable extraction, counter metadata, startup log

**Files:**
- Modify: `vllm/v1/kv_offload/cpu/spec.py`
- Create: `tests/v1/kv_offload/cpu/test_spec_config_validation.py`

**Interfaces:**
- Consumes: `CPUOffloadingManager(cache_policy, policy_kwargs)` from Tasks 7–8.
- Produces: `CPUOffloadingSpec.__init__` validates `eviction_policy` in `{"lru","arc","sae"}`, extracts `sae_*` tunables when policy is `sae`, rejects `sae_*` keys under non-SAE policies, adds four counter definitions to `build_metric_definitions`, logs `"CPU offload: eviction_policy=<name>"` at INFO.

- [ ] **Step 1: Write failing spec-validation tests**

Create `tests/v1/kv_offload/cpu/test_spec_config_validation.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import KVTransferConfig, VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.cpu.policies.sae import SAECachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def make_vllm_config(extra_config: dict) -> VllmConfig:
    kv_transfer = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 1024 * 1024,
            **extra_config,
        },
    )
    return VllmConfig(kv_transfer_config=kv_transfer)


def make_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(num_blocks=0, kv_cache_tensors=[])


def test_unknown_eviction_policy_raises():
    with pytest.raises(ValueError, match="eviction_policy"):
        CPUOffloadingSpec(
            make_vllm_config({"eviction_policy": "bogus"}),
            make_kv_cache_config(),
        )


def test_sae_key_under_non_sae_policy_raises():
    with pytest.raises(ValueError, match="sae_decay_interval"):
        CPUOffloadingSpec(
            make_vllm_config({
                "eviction_policy": "lru",
                "sae_decay_interval": 500,
            }),
            make_kv_cache_config(),
        )


def test_out_of_range_decay_factor_raises():
    with pytest.raises(ValueError, match="sae_decay_factor"):
        CPUOffloadingSpec(
            make_vllm_config({
                "eviction_policy": "sae",
                "sae_decay_factor": 1.5,
            }),
            make_kv_cache_config(),
        )


def test_valid_sae_config_stores_kwargs():
    spec = CPUOffloadingSpec(
        make_vllm_config({
            "eviction_policy": "sae",
            "sae_decay_interval": 250,
        }),
        make_kv_cache_config(),
    )
    assert spec.eviction_policy == "sae"
    assert spec._sae_policy_kwargs["decay_interval"] == 250


def test_get_manager_returns_sae_policy_when_selected():
    spec = CPUOffloadingSpec(
        make_vllm_config({"eviction_policy": "sae"}),
        make_kv_cache_config(),
    )
    mgr = spec.get_manager()
    assert isinstance(mgr._policy, SAECachePolicy)


def test_default_policy_still_lru_when_not_specified():
    spec = CPUOffloadingSpec(
        make_vllm_config({}),
        make_kv_cache_config(),
    )
    assert spec.eviction_policy == "lru"
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_spec_config_validation.py -v`
Expected: multiple failures (no validation, no `_sae_policy_kwargs` attribute).

- [ ] **Step 3: Modify `CPUOffloadingSpec.__init__`**

In `vllm/v1/kv_offload/cpu/spec.py`, add the import at the top:

```python
from vllm.logger import init_logger

logger = init_logger(__name__)
```

Add these module-level constants near the top of the file (after imports):

```python
_SUPPORTED_POLICIES = ("lru", "arc", "sae")

_SAE_TUNABLE_DEFAULTS: dict[str, tuple[type, object]] = {
    "sae_decay_interval": (int, 500),
    "sae_decay_factor": (float, 0.9),
    "sae_ghost_hit_weight": (float, 12.0),
    "sae_ghost_miss_weight": (float, 1.0),
    "sae_ghost_norm": (float, 12.0),
}


def _validate_sae_tunables(extra_config: dict) -> dict[str, object]:
    """Extract and validate SAE tunables from extra_config.

    Returns:
        A dict of ``SAECachePolicy`` constructor kwargs
        (``decay_interval``, ``decay_factor``, ``ghost_hit_weight``,
        ``ghost_miss_weight``, ``ghost_norm``).

    Raises:
        ValueError: on out-of-range values, naming the offending key.
    """
    kwargs: dict[str, object] = {}
    for key, (expected_type, default) in _SAE_TUNABLE_DEFAULTS.items():
        raw = extra_config.get(key, default)
        try:
            value = expected_type(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key}={raw!r} is not a valid {expected_type.__name__}"
            ) from exc

        if key == "sae_decay_interval" and value < 1:
            raise ValueError(f"{key}={value} must be >= 1")
        if key == "sae_decay_factor" and not (0.0 < value <= 1.0):
            raise ValueError(f"{key}={value} must satisfy 0.0 < x <= 1.0")
        if key == "sae_ghost_hit_weight" and value < 0.0:
            raise ValueError(f"{key}={value} must be >= 0.0")
        if key == "sae_ghost_miss_weight" and value < 0.0:
            raise ValueError(f"{key}={value} must be >= 0.0")
        if key == "sae_ghost_norm" and value <= 0.0:
            raise ValueError(f"{key}={value} must be > 0.0")

        # Strip "sae_" prefix for the CachePolicy constructor kwarg name.
        kwargs[key[len("sae_"):]] = value
    return kwargs
```

Modify `CPUOffloadingSpec.__init__`. Locate the current line that reads `self.eviction_policy = self.extra_config.get("eviction_policy", "lru")` (around line 106) and expand it into:

```python
        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        if self.eviction_policy not in _SUPPORTED_POLICIES:
            raise ValueError(
                f"eviction_policy={self.eviction_policy!r} is not supported. "
                f"Supported: {list(_SUPPORTED_POLICIES)}"
            )

        offending_sae_keys = [
            k for k in self.extra_config if k.startswith("sae_")
        ]
        if self.eviction_policy != "sae" and offending_sae_keys:
            raise ValueError(
                f"SAE-specific keys {offending_sae_keys!r} are set but "
                f"eviction_policy={self.eviction_policy!r} is not 'sae'."
            )

        self._sae_policy_kwargs: dict[str, object] = (
            _validate_sae_tunables(self.extra_config)
            if self.eviction_policy == "sae"
            else {}
        )

        logger.info(
            "CPU offload: eviction_policy=%s", self.eviction_policy
        )
```

Modify `get_manager` to pass `policy_kwargs`:

```python
    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
                policy_kwargs=self._sae_policy_kwargs,
            )
        return self._manager
```

Modify `build_metric_definitions` to add the four counter definitions:

```python
    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_BLOCK_LOOKUP: OffloadingCounterMetadata(
                documentation=(
                    "Total CPU KV cache lookup calls. Sum of hits and misses "
                    "(HIT_PENDING counts as a hit; RETRY is not counted)."
                ),
                labelnames=("policy",),
            ),
            CPUOffloadingMetrics.CPU_BLOCK_HIT: OffloadingCounterMetadata(
                documentation=(
                    "Total CPU KV cache lookup hits, labelled by eviction "
                    "policy (lru/arc/sae). HIT_PENDING counts as a hit."
                ),
                labelnames=("policy",),
            ),
            CPUOffloadingMetrics.CPU_BLOCK_MISS: OffloadingCounterMetadata(
                documentation=(
                    "Total CPU KV cache lookup misses, labelled by eviction "
                    "policy (lru/arc/sae)."
                ),
                labelnames=("policy",),
            ),
            CPUOffloadingMetrics.BLOCK_EVICTION: OffloadingCounterMetadata(
                documentation=(
                    "Total CPU KV cache blocks evicted, labelled by eviction "
                    "policy (lru/arc/sae)."
                ),
                labelnames=("policy",),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions
```

- [ ] **Step 4: Verify `OffloadingCounterMetadata` supports `labelnames`**

Read `vllm/v1/kv_offload/base.py` and confirm `OffloadingCounterMetadata` accepts a `labelnames` field. If it does not, add:

```python
@dataclass
class OffloadingCounterMetadata(OffloadingMetricMetadata):
    documentation: str
    labelnames: tuple[str, ...] = ()
```

(and the same to any related metadata dataclasses where relevant). If `OffloadingMetricMetadata` already has `labelnames`, no change needed.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_spec_config_validation.py tests/v1/kv_offload/cpu/test_manager_policy_metrics.py -v`
Expected: all pass.

- [ ] **Step 6: Run full CPU offload test suite for regression**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/ -v`
Expected: all pass.

- [ ] **Step 7: Run pre-commit**

Run: `pre-commit run --files vllm/v1/kv_offload/cpu/spec.py vllm/v1/kv_offload/base.py tests/v1/kv_offload/cpu/test_spec_config_validation.py`

- [ ] **Step 8: Commit**

```bash
git add vllm/v1/kv_offload/cpu/spec.py vllm/v1/kv_offload/base.py tests/v1/kv_offload/cpu/test_spec_config_validation.py
git commit -s -m "$(cat <<'EOF'
feat(kv_offload): SAE config validation + counter definitions in spec

CPUOffloadingSpec now validates eviction_policy in {lru,arc,sae},
rejects sae_* keys when the active policy is not sae, extracts and
range-validates SAE tunables, and logs the active policy at INFO.
Four labelled counter definitions are added to
build_metric_definitions so the counters emitted by
CPUOffloadingManager land on /metrics with a `policy` label.

Assisted-by: Claude
EOF
)"
```

---

## Task 10: Docs — mention SAE in the CPU offload doc page

**Files:**
- Modify: an existing CPU-offload docs file (see Step 1 to locate).

- [ ] **Step 1: Find the current CPU offload doc**

Run: `grep -l "eviction_policy\|CPUOffloadingSpec\|kv_connector_extra_config" docs/ -r 2>/dev/null | head -10`

Pick the doc page that discusses `eviction_policy` today (most likely under `docs/features/` or `docs/design/`). If no such page exists, add a small block to `docs/features/disagg_prefill.md` or the closest match.

- [ ] **Step 2: Add the SAE mention**

Add a short paragraph noting that `"sae"` is a supported `eviction_policy` value alongside `"lru"` and `"arc"`, with a table of the SAE tunables and their defaults:

```markdown
### Eviction policy

`kv_connector_extra_config["eviction_policy"]` selects the CPU
offload eviction policy. Supported values: `"lru"` (default),
`"arc"`, `"sae"`.

**SAE (Session-Aware Eviction)** tracks per-session hit history
and biases eviction toward sessions with weaker recent access
patterns. Tunables (all under `kv_connector_extra_config`):

| Key                     | Default | Notes                                        |
|-------------------------|---------|----------------------------------------------|
| `sae_decay_interval`    | 500     | Lookups between decay ticks (`>= 1`).         |
| `sae_decay_factor`      | 0.9     | Scale factor per decay tick (`0.0 < x <= 1`). |
| `sae_ghost_hit_weight`  | 12.0    | Ghost score bump on hits (`>= 0`).            |
| `sae_ghost_miss_weight` | 1.0     | Ghost score bump on misses (`>= 0`).          |
| `sae_ghost_norm`        | 12.0    | Divisor when seeding new session hits (`> 0`).|

All three policies emit four cache-effectiveness counters on
`/metrics`, labelled by `policy`:
`vllm:cpu_block_lookup_total`, `vllm:cpu_block_hit_total`,
`vllm:cpu_block_miss_total`, `vllm:block_eviction_total`.
```

- [ ] **Step 3: Run pre-commit and commit**

Run: `pre-commit run --files <the doc file you edited>`

```bash
git add <the doc file>
git commit -s -m "$(cat <<'EOF'
docs(kv_offload): document SAE eviction policy and per-policy counters

Adds a subsection covering "sae" as an eviction_policy choice, its
tunables, and the four labelled counters emitted by all three
policies.

Assisted-by: Claude
EOF
)"
```

---

## Task 11: End-to-end smoke — spec constructs, manager runs, counters flow through

**Files:**
- Create: `tests/v1/kv_offload/cpu/test_sae_end_to_end.py`

- [ ] **Step 1: Write an end-to-end test**

Create `tests/v1/kv_offload/cpu/test_sae_end_to_end.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test: spec → manager → policy → counters."""
from vllm.config import KVTransferConfig, VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.policies.sae import SAECachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def _key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def test_sae_end_to_end_smoke():
    kv_transfer = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 1024 * 1024,
            "eviction_policy": "sae",
            "sae_decay_interval": 100,
        },
    )
    vllm_config = VllmConfig(kv_transfer_config=kv_transfer)
    kv_cache_config = KVCacheConfig(num_blocks=0, kv_cache_tensors=[])

    spec = CPUOffloadingSpec(vllm_config, kv_cache_config)
    assert spec.eviction_policy == "sae"

    mgr = spec.get_manager()
    assert isinstance(mgr._policy, SAECachePolicy)
    assert mgr._policy._decay_interval == 100

    req = ReqContext(req_id="smoke")
    mgr.on_new_request(req)
    assert mgr.lookup(_key(1), req) == LookupResult.MISS

    stats = mgr.get_stats()
    data = stats.data["data"]
    assert data[CPUOffloadingMetrics.CPU_BLOCK_LOOKUP][("sae",)] == 1
    assert data[CPUOffloadingMetrics.CPU_BLOCK_MISS][("sae",)] == 1
```

- [ ] **Step 2: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/cpu/test_sae_end_to_end.py -v`
Expected: PASS.

- [ ] **Step 3: Final regression run — full kv_offload suite**

Run: `.venv/bin/python -m pytest tests/v1/kv_offload/ -v`
Expected: all pass.

- [ ] **Step 4: Run pre-commit and commit**

Run: `pre-commit run --files tests/v1/kv_offload/cpu/test_sae_end_to_end.py`

```bash
git add tests/v1/kv_offload/cpu/test_sae_end_to_end.py
git commit -s -m "$(cat <<'EOF'
test(kv_offload): end-to-end smoke for SAE policy path

Constructs CPUOffloadingSpec with eviction_policy=sae, retrieves
the manager, issues one lookup, and verifies the four counters
land on the stats payload with the "sae" policy label.

Assisted-by: Claude
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:**
  - SAE as a `CachePolicy` sibling of LRU/ARC → Tasks 1–7.
  - Session boundary reconstruction (`insert` after `touch`/`evict`/`remove`/`clear`) → Task 2.
  - Ghost score accumulation and decay → Task 4.
  - Admission gate inside `evict` → Task 6.
  - Worst-first eviction walk → Task 6.
  - No changes to `CachePolicy` ABC → verified: only `sae.py` and `manager.py` (registration) are touched among policy files. `base.py`, `lru.py`, `arc.py` untouched.
  - Four per-policy counters with `policy` label → Task 8, exposed via spec metadata in Task 9.
  - Fail-fast validation of `eviction_policy` and SAE tunables → Task 9.
  - `sae_*` keys under non-SAE policy → single `ValueError` → Task 9.
  - INFO log line at startup → Task 9.
  - `extra_config`-only config surface (no env vars, no TOML) → Task 9.
  - No LRU/ARC behavior change → confirmed by re-running `test_manager.py` in Tasks 7 and 9.
- **Placeholder scan:** no TBDs, no "similar to Task N", no "add error handling" — all code is explicit.
- **Type/name consistency:** `_policy_name` (str), `_sae_policy_kwargs` (dict), `SAECachePolicy` constructor kwargs (`decay_interval`, `decay_factor`, `ghost_hit_weight`, `ghost_miss_weight`, `ghost_norm`) — consistent across Tasks 7, 8, 9. Counter names on `CPUOffloadingMetrics` are used consistently in Tasks 8, 9, 11.

---

Plan complete and saved to [docs/superpowers/plans/2026-07-01-sae-eviction-policy-integration.md](docs/superpowers/plans/2026-07-01-sae-eviction-policy-integration.md).
