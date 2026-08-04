# `lmcache bench engine` — Design & Extension Guide

**Status:** Implemented  |  **Date:** 2026-05-05

## Overview

`lmcache bench engine` runs sustained benchmarks against an OpenAI-compatible
inference engine. It ships five workload types that exercise different
caching patterns, each controlling its own request scheduling while shared
modules handle request sending, stats collection, and real-time progress
display.

If required arguments (`--engine-url`, `--workload`, and either
`--tokens-per-gb-kvcache` or `--lmcache-url`) are missing, the command drops
into a guided **interactive TUI** to fill them in. `--config FILE` loads a
previously-exported JSON config and skips the TUI; `--no-interactive` errors
out instead of prompting; `--export-config FILE` writes the resolved config
to JSON and exits without running the benchmark.

```bash
# Long-document Q&A (semaphore-controlled concurrency)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload long-doc-qa --tokens-per-gb-kvcache 6000

# Multi-round chat (QPS-controlled dispatch)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload multi-round-chat --tokens-per-gb-kvcache 6000 \
    --mrc-qps 2.0 --mrc-duration 120

# Random prefill (all requests at once, 1-token output)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload random-prefill --tokens-per-gb-kvcache 6000 \
    --rp-num-requests 100

# Long-doc permutator (blended-prefix cache reuse stress test)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload long-doc-permutator --tokens-per-gb-kvcache 6000 \
    --ldp-num-contexts 5 --ldp-num-permutations 20

# Prefix-suffix tuner (tiered KV-cache demonstrator, sequential 2-pass)
lmcache bench engine --engine-url http://localhost:8000 \
    --workload prefix-suffix-tuner --lmcache-url http://localhost:8080 \
    --psf-context-length 8000 --psf-prefix-ratio 0.8 --psf-thrash 100
```

---

## 1. Architecture

### File Layout

```
lmcache/cli/commands/bench/
├── __init__.py                    # BenchCommand (CLI registration + orchestrator)
└── engine_bench/
    ├── __init__.py                # Package marker
    ├── config.py                  # EngineBenchConfig, auto-detection helpers
    ├── stats.py                   # RequestResult, StatsCollector, FinalStats
    ├── request_sender.py          # RequestSender (async streaming)
    ├── progress.py                # ProgressMonitor (real-time terminal display)
    ├── interactive/               # Guided TUI for missing-arg resolution
    │   ├── __init__.py            # run_interactive() entry point
    │   ├── schema.py              # Field schema + workload-specific items
    │   ├── state.py               # InteractiveState (load/save JSON, merge CLI args)
    │   ├── terminal.py            # Terminal rendering primitives
    │   └── config.json            # Static schema for interactive prompts
    └── workloads/
        ├── __init__.py            # create_workload() factory
        ├── base.py                # BaseWorkload (ABC with run loop)
        ├── long_doc_permutator.py # LongDocPermutatorConfig + LongDocPermutatorWorkload
        ├── long_doc_qa.py         # LongDocQAConfig + LongDocQAWorkload
        ├── multi_round_chat.py    # MultiRoundChatConfig + Session + MultiRoundChatWorkload
        ├── prefix_suffix_tuner.py # PrefixSuffixTunerConfig + PrefixSuffixTunerWorkload
        └── random_prefill.py      # RandomPrefillConfig + RandomPrefillWorkload
```

### Module Dependency Graph

```
BenchCommand (orchestrator)
  ├── config.py          → EngineBenchConfig
  ├── stats.py           → StatsCollector
  ├── progress.py        → ProgressMonitor
  ├── request_sender.py  → RequestSender
  └── workloads/
       ├── __init__.py   → create_workload() factory
       └── base.py       → BaseWorkload (used by all concrete workloads)
```

All concrete workloads depend on `BaseWorkload`, `RequestSender`,
`StatsCollector`, and `ProgressMonitor` — but never on each other.

---

## 2. Core Modules

### 2.1 `config.py` — Configuration

```python
@dataclass
class EngineBenchConfig:
    engine_url: str
    model: str                    # auto-detected if not provided
    workload: str                 # "long-doc-qa", "multi-round-chat", "random-prefill"
    kv_cache_volume_gb: float
    tokens_per_gb_kvcache: int    # auto-resolved via --lmcache-url or explicit
    seed: int
    output_dir: str
    export_csv: bool
    export_json: bool
    quiet: bool
```

Key functions:

| Function | Purpose |
|----------|---------|
| `parse_args_to_config(args) -> EngineBenchConfig` | Converts CLI args to fully-resolved config |
| `auto_detect_model(engine_url) -> str` | Fetches model ID from `/v1/models` |
| `resolve_tokens_per_gb(lmcache_url, model_name) -> int` | Queries LMCache `/status` for `cache_size_per_token * world_size` |

`EngineBenchConfig` contains only general parameters. Workload-specific
configs live in their own modules and are resolved by the workload factory.

### 2.2 `stats.py` — Stats Collection

```python
@dataclass
class RequestResult:
    request_id: str
    successful: bool
    ttft: float                   # seconds (time to first token)
    request_latency: float        # seconds (total request time)
    num_input_tokens: int
    num_output_tokens: int
    decode_speed: float           # tokens/second
    submit_time: float            # absolute timestamp
    first_token_time: float       # absolute timestamp
    finish_time: float            # absolute timestamp
    error: str                    # empty if successful

@dataclass
class AggregatedStats:
    total_requests: int
    successful_requests: int
    failed_requests: int
    elapsed_time: float
    mean_ttft_ms: float
    mean_decode_speed: float
    mean_request_latency_ms: float
    input_throughput: float       # tokens/second
    output_throughput: float      # tokens/second
    total_input_tokens: int
    total_output_tokens: int

@dataclass
class FinalStats(AggregatedStats):
    p50_ttft_ms: float            # plus p90, p99
    p50_decode_speed: float       # plus p90, p99
    p50_request_latency_ms: float # plus p90, p99
```

`StatsCollector` is **thread-safe** (uses `threading.Lock`):

| Method | Called by | Description |
|--------|-----------|-------------|
| `on_request_finished(result)` | RequestSender callback | Records a completed request |
| `get_current_stats() -> AggregatedStats` | ProgressMonitor (every 1s) | Returns a snapshot for live display |
| `get_final_stats() -> FinalStats` | Orchestrator (after benchmark) | Computes percentiles |
| `reset()` | BaseWorkload (between warmup/benchmark) | Clears warmup stats |
| `export_csv(path)` | Orchestrator | Writes per-request CSV |
| `export_json(path, config)` | Orchestrator | Writes summary JSON |

### 2.3 `request_sender.py` — Async Streaming

```python
OnFinishedCallback = Callable[[RequestResult, str], None]

class RequestSender:
    def __init__(self, engine_url, model, completions_mode=False, on_finished=[])
    async def send_request(self, request_id, messages, max_tokens=128) -> RequestResult
    async def send_warmup_request(self, request_id, messages, max_tokens=1) -> RequestResult
    async def close(self) -> None
```

- Uses `AsyncOpenAI` for streaming chat/completions.
- Measures TTFT, decode speed, total latency per request.
- Extracts token counts from server usage reports.
- After each request (success or failure), invokes all `on_finished`
  callbacks with `(RequestResult, response_text)`.

Each `send_request` call is a **self-contained coroutine** — concurrency is
controlled externally by the workload (semaphore, QPS, or fire-all-at-once).

### 2.4 `progress.py` — Real-Time Display

```python
class ProgressMonitor:
    def __init__(self, stats_collector, quiet=False)
    def start(self) -> None          # starts daemon thread
    def stop(self) -> None           # stops thread, prints final state
    def on_request_sent(request_id)  # increments in-flight count
    def on_request_finished(request_id, successful)  # decrements in-flight count
    def log_message(message)         # adds to rolling log (last 5 lines)
```

Runs a daemon thread that redraws every second using ANSI cursor control.
Reads aggregated stats from `StatsCollector.get_current_stats()`. Tracks
in-flight count and rolling log messages internally. No-op when `quiet=True`.

### 2.5 `workloads/base.py` — Base Workload

```python
class BaseWorkload(ABC):
    def __init__(self, request_sender, stats_collector, progress_monitor)

    # --- Must implement ---
    @abstractmethod async def warmup(self) -> None
    @abstractmethod async def step(self, time_offset: float) -> float
    @abstractmethod def log_config(self) -> None
    @abstractmethod def on_request_finished(self, request_id: str, output: str) -> None

    # --- Provided by base class ---
    def run(self) -> None                         # entry point (blocks)
    def request_finished(self, result, text)      # thread-safe queue bridge
```

**`run()` loop** (in base class):

```
log_config()           ← print workload config (before progress monitor starts)
warmup()               ← workload-specific warmup
stats_collector.reset()
loop:
    drain_finished_queue() → on_request_finished()
    next_wakeup = step(time_offset)
    if next_wakeup < 0: break
    sleep until next_wakeup
drain_finished_queue()   ← final drain
```

**`step()` contract:**

- Returns the **absolute time offset** (from benchmark start) when the loop
  should call `step()` again. The loop sleeps until that time.
- Returns a **negative value** to signal the workload is complete.
- The loop calls `_drain_finished_queue()` before each `step()`, which
  calls `on_request_finished()` for any completed requests.

**Callback bridge — `request_finished()`:**

This method matches the `OnFinishedCallback` signature and is registered
on `RequestSender._on_finished` by the orchestrator. It enqueues
`(request_id, response_text)` onto a `queue.Queue`. The loop thread drains
this queue and calls `on_request_finished()` from a single thread, so
workload implementations do not need to handle cross-thread concerns.

### 2.6 `workloads/__init__.py` — Factory

```python
def create_workload(config, args, request_sender, stats_collector, progress_monitor) -> BaseWorkload
```

Dispatches on `config.workload` string to the appropriate workload module.
Resolves workload-specific config from `args`, constructs the workload
instance, and returns it ready to `run()`.

---

## 3. End-to-End Flow

The orchestrator in `engine_bench.command.run_engine_bench()` wires everything together:

```
0. _resolve_args(args)            → argparse.Namespace
     (a) --config FILE            → load InteractiveState, merge CLI overrides
     (b) --no-interactive / --export-config → error if required args missing
     (c) interactive TUI          → if any required arg is missing
     (d) pass through             → if all required args present
1. parse_args_to_config(args)     → EngineBenchConfig
   (--export-config: write JSON and return without running)
2. StatsCollector()
3. ProgressMonitor(stats_collector, quiet)
4. RequestSender(engine_url, model)
5. create_workload(config, args, sender, collector, monitor) → workload
6. Wire callbacks on sender:
     - stats_collector.on_request_finished(result)
     - progress_monitor.on_request_finished(request_id, successful)
     - workload.request_finished(result, response_text)
7. workload.log_config()          → print config to terminal
8. progress_monitor.start()       → start live display
9. workload.run()                 → blocks until benchmark complete
10. progress_monitor.stop()
11. request_sender.close()
12. Emit final metrics (CLI metrics system)
13. Export CSV / JSON
14. sys.exit(1) if any failures
```

### Callback Wiring

```
Workload.step()
    │
    ├── send_request()  ──────────────────┐
    │                                     │
    └── progress_monitor.on_request_sent()│
                                          ▼
                                   RequestSender
                                   (streams SSE, collects stats)
                                          │
                              on_finished callbacks:
                              ├── stats_collector.on_request_finished(result)
                              ├── progress_monitor.on_request_finished(id, ok)
                              └── workload.request_finished(result, text)
                                          │
                                          ▼
                                   finished_queue
                                          │
                              loop drains → workload.on_request_finished(id, text)
```

---

## 4. Existing Workloads

### 4.1 `long-doc-qa` — Long Document Q&A

Tests KV cache reuse by asking repeated questions over long documents.

**Config** (`LongDocQAConfig`):

| Field | CLI arg | Default | Description |
|-------|---------|---------|-------------|
| `document_length` | `--ldqa-document-length` | 10000 | Tokens per document |
| `query_per_document` | `--ldqa-query-per-document` | 2 | Questions per document |
| `num_documents` | computed | — | `kv_cache_volume * tokens_per_gb / document_length` |
| `shuffle_policy` | `--ldqa-shuffle-policy` | `random` | `random` or `tile` |
| `num_inflight_requests` | `--ldqa-num-inflight-requests` | 3 | Max concurrent requests |

**Behavior:**

- **Warmup:** Sends each document once sequentially (`max_tokens=1`)
- **Dispatch:** Semaphore-controlled — `step()` acquires semaphore, fires
  async task, returns `0.0` (immediate re-call). Semaphore released when
  task completes.
- **`on_request_finished`:** No-op (stateless).
- **Termination:** Returns `-1.0` when schedule exhausted and all tasks done.

### 4.2 `multi-round-chat` — Multi-Round Chat

Simulates concurrent chat users with growing conversation history.

**Config** (`MultiRoundChatConfig`):

| Field | CLI arg | Default | Description |
|-------|---------|---------|-------------|
| `shared_prompt_length` | `--mrc-shared-prompt-length` | 2000 | System prompt tokens |
| `chat_history_length` | `--mrc-chat-history-length` | 10000 | Pre-filled history tokens |
| `user_input_length` | `--mrc-user-input-length` | 50 | Tokens per query |
| `output_length` | `--mrc-output-length` | 200 | Max tokens per response |
| `qps` | `--mrc-qps` | 1.0 | Queries per second |
| `duration` | `--mrc-duration` | 60.0 | Benchmark duration (seconds) |
| `num_concurrent_users` | computed | — | `kv_cache_volume * tokens_per_gb / (prompt + history)` |

**Behavior:**

- **Warmup:** Sends one request per session sequentially (`max_tokens=1`)
- **Dispatch:** QPS-controlled — `step()` dispatches at `1/qps` intervals
  using round-robin session scheduling. Returns `global_index * interval`.
  If the target session is busy, returns `time_offset + 0.01` to retry
  after queue drain.
- **`on_request_finished`:** **Stateful** — records the response in the
  session's conversation history via `Session.record_answer()`, which
  marks the session as ready for its next request.
- **Termination:** Returns `-1.0` when `time_offset >= duration` and all
  pending tasks are complete.

**Session state:** Each `Session` holds a system prompt, pre-filled history,
and a growing list of `(query, answer)` exchanges. `build_messages()` constructs
the full OpenAI message list including all prior context.

### 4.3 `random-prefill` — Prefill Speed Testing

Tests raw prefill throughput by firing all requests simultaneously.

**Config** (`RandomPrefillConfig`):

| Field | CLI arg | Default | Description |
|-------|---------|---------|-------------|
| `request_length` | `--rp-request-length` | 10000 | Tokens per request |
| `num_requests` | `--rp-num-requests` | 50 | Number of requests |

**Behavior:**

- **Warmup:** None.
- **Dispatch:** Fire-all-at-once — first `step()` dispatches all requests
  as concurrent async tasks with `max_tokens=1`, returns `0.0`. Subsequent
  `step()` calls wait via `asyncio.wait(FIRST_COMPLETED)`.
- **`on_request_finished`:** No-op (stateless).
- **Termination:** Returns `-1.0` when all tasks are complete.

### 4.4 `long-doc-permutator` — Blended Cache-Reuse Stress Test

Stresses **blended** KV cache reuse — not just prefix reuse — by sending
permutations of a fixed set of context documents. Each request is:

```
[System Prompt] + [Doc_i1] + [Doc_i2] + ... + [Doc_iN]
```

where `(i1, …, iN)` is one permutation of the `N` contexts. Most permutations
share *some* chunks with prior requests but rarely the same prefix, exercising
chunk-level cache lookup and eviction.

**Config** (`LongDocPermutatorConfig`):

| Field | CLI arg | Default | Description |
|-------|---------|---------|-------------|
| `num_contexts` | `--ldp-num-contexts` | 5 | Number of unique context documents (`N`) |
| `context_length` | `--ldp-context-length` | 5000 | Tokens per context |
| `system_prompt_length` | `--ldp-system-prompt-length` | 1000 | Shared system prompt tokens (`0` disables) |
| `num_permutations` | `--ldp-num-permutations` | 10 | Distinct permutations to send (capped at `N!`) |
| `vocab_size` | (none — hardcoded in factory) | 8000 | Vocabulary pool size for synthetic context generation |
| `num_inflight_requests` | `--ldp-num-inflight-requests` | 1 | Max concurrent in-flight requests |

**Stress axes** (each config field tunes one):

| Axis | Knob |
|------|------|
| Blended-context boundaries | `num_contexts` |
| Eviction pressure | `num_permutations` |
| Chunk homogeneity (hash collisions) | `vocab_size` |
| Prefix domination | `system_prompt_length` |
| Concurrency | `num_inflight_requests` |

**Behavior:**

- **Data generation:** Builds a deterministic vocab pool of pseudo-words,
  generates `num_contexts` distinct contexts (each seeded independently so
  token sequences truly diverge), and enumerates permutations.
- **Permutation enumeration:** For small `N`, iterates `itertools.permutations`
  and truncates. When `N!` is much larger than `num_permutations * 10`, samples
  random permutations into a `set` to avoid exhausting an enormous search
  space. Returns all `N!` permutations when `num_permutations >= N!`.
- **Warmup:** A single dummy request (`max_tokens=1`) to prime the engine.
- **Dispatch:** Semaphore-controlled — `step()` acquires the semaphore, fires
  an async task with the next permutation, returns `0.0` for immediate
  re-call. Once all permutations are dispatched, awaits remaining tasks via
  `asyncio.wait(FIRST_COMPLETED)`.
- **`on_request_finished`:** No-op (stateless).
- **Termination:** Returns `-1.0` when the request list is exhausted and all
  pending tasks have completed.

**`run()` override:** Unlike the other workloads, `LongDocPermutatorWorkload`
overrides `BaseWorkload.run()` to close `RequestSender`'s async HTTP client
inside the same `asyncio.run()` call as the benchmark loop. `asyncio.run()`
closes the loop on exit, which would orphan any open `httpx` connections;
closing the client here ensures clean teardown. The orchestrator's subsequent
`asyncio.run(request_sender.close())` then finds nothing to close and
completes without error.

### 4.5 `prefix-suffix-tuner` — Tiered KV-Cache Demonstrator

A single sequential workload designed to be run **unchanged** across three
LMCache configurations to demonstrate the value of each cache tier:

| Baseline | LMCache config | Targeted overflow | Expected pass-2 hits |
|----------|---------------|-------------------|----------------------|
| 1 | vanilla vLLM (L0 only) | L0 (HBM) | none — every request a cold prefill |
| 2 | vLLM + LMCache L1 + L2 | L1 (DRAM) | L2 prefix hits (suffix recomputed) |
| 3 | vLLM + LMCache L1 + L2 + CacheBlend | L1 (DRAM) | L2 prefix hits + CacheBlend suffix hits |

The user picks `--psf-thrash` to match the size of the tier they want to
overflow (L0 size for Baseline 1, L1 size for Baselines 2 and 3). The
workload itself does not need to know which baseline it is running — the
internal `_OVERFLOW_FACTOR` (1.05) sizes the pool slightly larger than the
named target, and sequential dispatch + LRU does the rest.

`--kv-cache-volume` is unused by this workload (it remains required for
other workloads that size themselves around a user-provided GB budget).

**Request layout:**

```
[prefix_i with unique-ID][random breaker][shared suffix]
```

- `num_prefixes` distinct prefixes — each begins with `PREFIX_<8-hex-digits>`
  so the prefix's chained block hash differs from every other prefix.
- A **fresh random breaker** per request (32 tokens by default), defeating
  ordinary prefix caching past the prefix boundary and preventing
  non-CacheBlend reuse of the suffix.
- A **single shared suffix**, deterministic and bit-identical across every
  request — the only entry CacheBlend can reuse.

**Synthetic body generation** uses a vocabulary pool of pseudo-words
(consonant-vowel patterns + numeric suffix, e.g. `"boko42"`), shared by all
prefixes / suffix / breakers but sampled with a *different* per-component
RNG offset. Mirrors `long_doc_permutator`'s approach. This guarantees:

- **CacheBlend correctness**: each prefix samples a different random
  sequence, so chunk-level content fingerprints don't collide across
  prefixes and inflate the blend hit rate. The shared suffix is the *only*
  bit-identical chunk surface CacheBlend can reuse — which is what the
  workload measures.
- **Predictable token counts**: pseudo-words tokenize to ~2 BPE tokens on
  most modern tokenizers (vs. ~3 for raw 6-digit numbers), so the actual
  prompt length is closer to the configured `context_length`.

**Config** (`PrefixSuffixTunerConfig`):

| Field | CLI arg | Default | Description |
|-------|---------|---------|-------------|
| `context_length` | `--psf-context-length` | 8000 | Total tokens per request (prefix + breaker + suffix) |
| `prefix_ratio` | `--psf-prefix-ratio` | 0.8 | Fraction of context allocated to the prefix; must be in (0.0, 1.0) |
| `thrash` | `--psf-thrash` | 20.0 | **Size in GB of the targeted KV-cache tier** (L0 for Baseline 1, L1 for Baselines 2 and 3). Pool footprint is `thrash * _OVERFLOW_FACTOR` GB. |
| `num_prefixes` | (computed) | — | `floor(thrash * _OVERFLOW_FACTOR * tokens_per_gb / prefix_tokens)` |
| `prefix_tokens` | (computed) | — | `round(context_length * prefix_ratio)` |
| `suffix_tokens` | (computed) | — | `context_length - prefix_tokens - breaker_tokens`; errors if `< 100` |
| `breaker_tokens` | (hardcoded) | 32 | Random breaker length |
| `_OVERFLOW_FACTOR` | (module constant) | 1.05 | How much to overflow the targeted tier. Hardcoded at 1.05 because the LRU invariant proves that a 5% overflow is sufficient under sequential same-order replay. |

**Behavior:**

- **Concurrency:** Strictly sequential, one in-flight request at a time —
  `step()` awaits each request inline. No semaphore, no concurrent tasks.
- **Pass 1 (warmup):** Sends each prefix once in pool order using
  `send_warmup_request` (`max_tokens=1`). Stats are discarded by the base
  class's `_run_async` after warmup.
- **Pass 2 (measured):** Sends each prefix once **in identical pool order**
  with `max_tokens=1`. These are the requests captured in final stats.
- **Termination:** `step()` returns `-1.0` once `pass2_index` reaches
  `num_prefixes`.

**Why 1.05× is enough:** With sequential dispatch and LRU eviction in any
single tier of capacity `K`:

- After pass 1 of `N = 1.05K` prefixes, the `0.05K` oldest accesses have
  been evicted; L1 holds prefixes `[0.05K..1.05K-1]` in LRU order.
- Pass 2 access of prefix `0` misses (it was evicted), and serving it
  evicts the LRU = prefix `0.05K` — *the very next prefix pass 2 will need*.
- This pattern continues for the whole pass: every access misses the
  targeted tier and the LRU it evicts is exactly the prefix needed next.

So the workload does not need to overprovision by 2× or more; even a 5%
overflow is sufficient to drive every measured request to the next tier
down (Baseline 1 → cold prefill, Baseline 2/3 → L2).

**Pass-1 vs pass-2 breakers:** The breaker is freshly randomized on every
`_build_messages()` call, so pass 1 and pass 2 use different breakers per
prefix. This makes the suffix unreachable by ordinary prefix caching even
within a single benchmark run — exactly the case CacheBlend is designed to
handle, and exactly what Baseline 3 should improve over Baseline 2.

---

## 5. Adding a New Workload

### Step 1: Create the workload module

Create `workloads/my_workload.py` with:

```python
from dataclasses import dataclass
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload

@dataclass
class MyWorkloadConfig:
    """Workload-specific config fields with defaults."""
    my_param: int = 100

    def __post_init__(self) -> None:
        # Validate all fields
        if self.my_param <= 0:
            raise ValueError(f"my_param must be positive, got {self.my_param}")

    @classmethod
    def resolve(cls, kv_cache_volume_gb, tokens_per_gb_kvcache, **kwargs):
        """Compute derived fields from the KV cache budget + CLI args."""
        # Example: compute a count from the cache budget
        computed_count = max(1, int(kv_cache_volume_gb * tokens_per_gb_kvcache / kwargs["my_param"]))
        return cls(my_param=kwargs["my_param"], ...)


class MyWorkload(BaseWorkload):
    def __init__(self, config, request_sender, stats_collector, progress_monitor, seed=42):
        super().__init__(request_sender, stats_collector, progress_monitor)
        self._config = config
        # ... generate data, build schedule, etc.

    def log_config(self) -> None:
        """Print workload config. Called BEFORE progress monitor starts."""
        print(f"Workload: my-workload\n  my_param: {self._config.my_param}")

    async def warmup(self) -> None:
        """Run warmup requests (or no-op)."""

    async def step(self, time_offset: float) -> float:
        """Dispatch logic. Return next wakeup time, or negative when done."""

    def on_request_finished(self, request_id: str, output: str) -> None:
        """Handle completed request. No-op for stateless, or record state."""
```

### Step 2: Register in the factory

In `workloads/__init__.py`, add the import and dispatch:

```python
from lmcache.cli.commands.bench.engine_bench.workloads.my_workload import (
    MyWorkloadConfig, MyWorkload,
)

_WORKLOAD_NAMES = (..., "my-workload")

def create_workload(...):
    ...
    if config.workload == "my-workload":
        wl_config = MyWorkloadConfig.resolve(
            kv_cache_volume_gb=config.kv_cache_volume_gb,
            tokens_per_gb_kvcache=config.tokens_per_gb_kvcache,
            my_param=args.mw_my_param,
        )
        return MyWorkload(wl_config, request_sender, stats_collector, progress_monitor, seed=config.seed)
    ...
```

### Step 3: Add CLI args

In `engine_bench/command.py`, inside `register_engine_parser()`:

1. Add `"my-workload"` to the `--workload` choices list.
2. Add a new argument group with **prefixed** arg names:

```python
mw_group = parser.add_argument_group("my-workload workload options")
mw_group.add_argument("--mw-my-param", type=int, default=100, help="...")
```

All workload-specific args must be prefixed with a short workload identifier
(e.g., `ldqa-`, `ldp-`, `mrc-`, `psf-`, `rp-`, `mw-`) to avoid name collisions.

### Step 4: Add tests

Create `tests/cli/commands/bench/engine_bench/workloads/test_my_workload.py`
with tests for:

- Config validation (`__post_init__` raises on invalid values)
- Config resolution (`resolve()` computes derived fields correctly)
- Workload data generation
- `warmup()` behavior (async)
- `step()` dispatch logic and return values (async)
- `on_request_finished()` behavior
- `run()` end-to-end with mocked `RequestSender`

Add factory tests to `test_create_workload.py`.

### Key Design Constraints

- **`step()` must not block indefinitely.** It should dispatch or wait
  briefly and return. The loop handles sleeping between calls.
- **`on_request_finished()` runs on the loop thread** (via queue drain),
  not the async sender thread. No locking needed within the workload.
- **`log_config()` prints via `print()`**, not `log_message()`, because
  it runs before the progress monitor starts. Use ANSI colors for
  readability.
- **Use `progress_monitor.log_message()`** for all runtime logging during
  the benchmark to avoid corrupting the terminal display.
- **Warmup stats are discarded** — `stats_collector.reset()` is called
  after warmup, so warmup request metrics don't affect final results.

---

## 6. Tests

```bash
# All bench tests (~190 tests)
pytest -xvs tests/cli/commands/bench/

# Specific workload
pytest -xvs tests/cli/commands/bench/engine_bench/workloads/test_long_doc_qa.py
pytest -xvs tests/cli/commands/bench/engine_bench/workloads/test_multi_round_chat.py
pytest -xvs tests/cli/commands/bench/engine_bench/workloads/test_random_prefill.py

# Factory
pytest -xvs tests/cli/commands/bench/engine_bench/workloads/test_create_workload.py

# CLI registration + orchestrator
pytest -xvs tests/cli/commands/bench/test_bench_command.py
```
