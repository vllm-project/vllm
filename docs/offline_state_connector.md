# OfflineState KV Connector: Bloom-Filter Cooperative Caching for vLLM

## Overview

The OfflineState connector adds **decentralized KV cache discovery** to vLLM using
bloom filters exchanged between peer instances. It eliminates the centralized
metadata server that existing distributed KV cache solutions (llm-d, LMCache,
Mooncake) require, replacing it with a compact probabilistic data structure that
each node maintains locally.

**Key idea:** Each vLLM instance periodically broadcasts a bloom filter summarizing
its cached block hashes. Peers merge received filters and can answer "does any
peer have block X?" in **~0.5 microseconds** with a local memory lookup — no
network round-trip required.

Based on the OfflineState design from *"Cuckoo for Clients: Disaggregated
Cuckoo Hashing"* thesis, adapted for vLLM's KV cache block discovery.

---

## Architecture

### Three-Tier Lookup

```
Request arrives at Node N
           │
           ▼
   ┌──────────────────┐
   │  Tier 1: Local   │   Cost: ~0.1 us
   │  BlockPool check  │   vLLM's existing prefix cache
   └────────┬─────────┘
            │ miss
            ▼
   ┌──────────────────┐
   │  Tier 2: Bloom   │   Cost: ~0.5 us
   │  Filter lookup    │   Check merged peer bloom filter
   └────────┬─────────┘
            │ miss
            ▼
   ┌──────────────────┐
   │  Tier 3: Miss    │   Cost: ~10,000 us
   │  Recompute        │   Full KV cache computation
   └──────────────────┘
```

- **Tier 1** is handled by vLLM's existing `BlockPool` / `KVCacheCoordinator`
  before the connector is called. `num_computed_tokens` reflects local hits.
- **Tier 2** is this connector's contribution. The merged bloom filter (bitwise
  OR of all peer filters) answers "does any peer have this block?" in local
  memory. If yes, we identify which specific peer and initiate a transfer.
- **Tier 3** is the fallback — recompute the KV cache from scratch.

### Sync Protocol

```
  Node 0                    Node 1                    Node 2
    │                         │                         │
    │  rebuild local bloom    │  rebuild local bloom    │
    │  from LRU cache keys    │  from LRU cache keys    │
    │                         │                         │
    ├───── PUB bloom ────────►│◄──── PUB bloom ─────────┤
    │◄──── PUB bloom ─────────┤───── PUB bloom ────────►│
    │                         │                         │
    │  merge all peer blooms  │  merge all peer blooms  │
    │  (bitwise OR)           │  (bitwise OR)           │
    │                         │                         │
    └─── sleep(interval) ─────┴─── sleep(interval) ─────┘
```

Each node runs a background thread that periodically:
1. Rebuilds the local bloom filter from current LRU cache keys
2. Publishes the local bloom via ZMQ PUB socket
3. Receives peer blooms via ZMQ SUB socket
4. Computes the merged bloom (bitwise OR of all received filters)

Default sync interval: 100ms. The bloom filter tolerates staleness — hit rate
is stable regardless of sync interval (only false positives increase).

---

## Files

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/distributed/kv_transfer/kv_connector/v1/bloom_filter.py` | 213 | Auto-scaling bloom filter with numpy bitarray |
| `vllm/distributed/kv_transfer/kv_connector/v1/peer_discovery.py` | 322 | Decentralized peer discovery via ZMQ PUB/SUB |
| `vllm/distributed/kv_transfer/kv_connector/v1/offline_state_connector.py` | 362 | KVConnectorBase_V1 implementation |
| `tests/distributed/test_offline_state.py` | 524 | 23 unit tests (all passing) |
| `benchmarks/benchmark_offline_state.py` | 967 | Evaluation benchmark with 7 plot types |

### Modified Files

| File | Change |
|------|--------|
| `vllm/distributed/kv_events.py` | Added `BloomFilterSync` event type (lines 96-102) |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | Registered `OfflineStateConnector` (lines 217-221) |

---

## Code Structure

### `BloomFilter` (`bloom_filter.py`)

Numpy-backed bloom filter optimized for KV cache block discovery.

```python
class BloomFilter:
    def __init__(self, expected_items=10000, fp_rate=0.01, auto_scale_clients=1):
        # Auto-scaling: n = expected_items * ceil(sqrt(auto_scale_clients))
        # Optimal sizing: m = -(n * ln(p)) / (ln(2)^2)
        # Optimal hashes: k = (m/n) * ln(2)

    def add(self, block_hash: int) -> None: ...
    def contains(self, block_hash: int) -> bool: ...
    def rebuild_from_keys(self, keys: Iterable[int]) -> None: ...

    @staticmethod
    def bitwise_or(filters: list['BloomFilter']) -> 'BloomFilter': ...

    def to_bytes(self) -> bytes: ...
    @classmethod
    def from_bytes(cls, data: bytes) -> 'BloomFilter': ...

    @property
    def theoretical_fpr(self) -> float: ...
    @property
    def fill_rate(self) -> float: ...
    @property
    def size_bytes(self) -> int: ...
```

**Auto-scaling formula:** When `auto_scale_clients=N`, the filter capacity is
scaled by `ceil(sqrt(N))`. This prevents FPR saturation when N peer blooms are
merged via bitwise OR, because OR-merging increases the fill rate. The sqrt
factor provides the minimum scaling needed to keep the merged FPR below the
target.

**Hashing:** Uses murmurhash-style integer mixing with `k` independent seeds.
Each seed produces a bit position via `hash(seed ^ block_hash) % size_bits`.

### `BloomFilterPeerDiscovery` (`peer_discovery.py`)

Manages local cache state, peer bloom filters, and the background sync thread.

```python
class BloomFilterPeerDiscovery:
    def __init__(self, node_id, num_nodes, sync_interval_ms=100.0,
                 bloom_fp_rate=0.01, max_cache_entries=10000,
                 zmq_base_port=15600): ...

    def register_block(self, block_hash: int) -> None: ...
    def register_blocks(self, block_hashes: list[int]) -> None: ...
    def unregister_block(self, block_hash: int) -> None: ...

    def find_peer_with_block(self, block_hash: int) -> int | None: ...

    def start(self) -> None: ...  # Start background sync thread + ZMQ
    def stop(self) -> None: ...   # Graceful shutdown
    def force_sync(self) -> None: ...  # Immediate sync (for testing)
```

**Local cache:** `OrderedDict` used as LRU cache. When at capacity, the least
recently used entry is evicted. The bloom filter is rebuilt periodically from
current cache keys (handles evictions that bloom filters can't remove).

**ZMQ topology:** Each node binds a PUB socket on `zmq_base_port + node_id` and
SUB-connects to all other nodes' PUB ports. Messages are `BloomFilterMessage`
structs serialized with msgspec msgpack.

### `OfflineStateConnector` (`offline_state_connector.py`)

Implements `KVConnectorBase_V1` — vLLM's standard interface for distributed KV
cache connectors.

**Scheduler-side methods:**

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    """Three-tier prefix discovery.

    Walks through uncomputed prefix blocks, computes block hashes from
    token IDs, and checks the merged peer bloom filter. Returns the
    number of additional tokens that can be loaded from a peer.
    """

def update_state_after_alloc(self, request, blocks, num_external_tokens):
    """Register newly allocated blocks in the bloom filter for
    peer discovery by other nodes."""

def build_connector_meta(self, scheduler_output):
    """Package peer location and block hashes into metadata
    for worker-side KV transfer."""
```

**Worker-side methods:**

```python
def start_load_kv(self, forward_context, **kwargs):
    """Initiate KV cache transfer from discovered peer.
    (Stub — full RDMA transfer for production deployment.)"""

def wait_for_layer_load(self, layer_name): ...
def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs): ...
def wait_for_save(self): ...
```

**Configuration** via `kv_connector_extra_config`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sync_interval_ms` | 100.0 | Bloom filter sync interval in milliseconds |
| `bloom_fp_rate` | 0.01 | Target false positive rate |
| `max_cache_entries` | 10000 | Max blocks tracked in local LRU cache |
| `num_nodes` | 1 | Number of nodes in cluster |
| `zmq_base_port` | 15600 | Base port for ZMQ PUB/SUB |

---

## Usage

### Starting vLLM with OfflineState

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --kv-connector OfflineStateConnector \
    --kv-connector-extra-config '{
        "num_nodes": 4,
        "sync_interval_ms": 100,
        "bloom_fp_rate": 0.01,
        "max_cache_entries": 10000,
        "zmq_base_port": 15600
    }' \
    --enable-prefix-caching
```

Each instance in the cluster should use a unique `engine_id` (used to derive
`node_id`). All instances must be reachable on their ZMQ ports.

### Running Tests

```bash
# Create a Python 3.12+ venv with minimal deps
python3.12 -m venv /tmp/vllm-venv
source /tmp/vllm-venv/bin/activate
pip install numpy msgspec pyzmq pytest

# Run the 23 unit tests
pytest tests/distributed/test_offline_state.py -v --noconftest
```

### Running the Evaluation Benchmark

```bash
pip install matplotlib  # for plot generation

python benchmarks/benchmark_offline_state.py \
    --output-dir plots/ \
    --n-ops 50000 \
    --cache-capacity 512
```

Generates 7 PNG plots and a `benchmark_results.json` in the output directory.

---

## Evaluation Results

### Hit Rate by Strategy (Zipf theta=0.9, 8 nodes, 50K ops)

| Strategy | Local Hits | Peer Hits | Combined | Avg Latency |
|----------|-----------|-----------|----------|-------------|
| Local-only (vLLM default) | 31.2% | — | 31.2% | 6,877 us |
| Centralized (llm-d model) | 31.7% | 19.0% | 50.7% | 4,989 us |
| Event-based (LMCache model) | 31.7% | 19.0% | 50.7% | 5,042 us |
| **OfflineState** | **31.7%** | **19.0%** | **50.7%** | **4,939 us** |

### Hit Rate by Strategy (Shared System Prompt, 8 nodes)

| Strategy | Combined Hit Rate | Avg Latency |
|----------|-------------------|-------------|
| Local-only | 95.8% | 418 us |
| Centralized | 99.7% | 82 us |
| **OfflineState** | **99.7%** | **32 us** |

### Cluster Scaling (Zipf theta=0.9)

| Nodes | Local-only | OfflineState | Improvement |
|-------|-----------|--------------|-------------|
| 1 | 32.2% | 32.2% | — |
| 4 | 31.4% | 44.2% | +12.8pp |
| 8 | 31.2% | 50.7% | +19.5pp |
| 16 | 30.0% | 57.1% | +27.1pp |
| 64 | 25.3% | 66.7% | +41.4pp |

### Bloom Filter Overhead

| Cache Entries | 1 Node | 8 Nodes | 64 Nodes |
|--------------|--------|---------|----------|
| 1,000 | 9.4 KB | 28.1 KB | 74.9 KB |
| 10,000 | 93.6 KB | 280.8 KB | 748.8 KB |
| 50,000 | 468.0 KB | 1,404 KB | 3,744 KB |

FPR stays well below the 1% target at all cluster scales due to auto-scaling.

---

## Plots

The benchmark generates 7 evaluation plots:

1. **`01_hitrate_vs_zipf.png`** — Cache hit rate vs workload skewness (Zipf theta)
2. **`02_latency_vs_zipf.png`** — Discovery latency vs workload skewness
3. **`03_hitrate_vs_cluster.png`** — Hit rate vs cluster size (1-64 nodes)
4. **`04_latency_vs_cluster.png`** — Discovery latency vs cluster size
5. **`05_bloom_sizing.png`** — Bloom filter size and empirical FPR vs cluster scale
6. **`06_sync_interval.png`** — Hit rate and false positives vs sync interval
7. **`07_hitrate_breakdown.png`** — Three-tier hit rate breakdown (stacked bar)

---

## Key Design Decisions

1. **Bloom filter over consistent hashing:** Bloom filters tolerate node
   failures gracefully (stale filter = slightly higher FPR). Consistent hashing
   requires rebalancing on node changes.

2. **Periodic sync over event streaming:** Periodic O(N) sync replaces
   continuous O(mutations) event streaming. One bloom filter broadcast every
   100ms is far cheaper than streaming every block store/evict event.

3. **Auto-scaling formula:** `n = entries * ceil(sqrt(nodes))` prevents FPR
   saturation when merging peer blooms. The sqrt factor is the minimum scaling
   to maintain the target FPR after bitwise-OR of N filters.

4. **LRU + periodic rebuild:** Bloom filters can't remove elements. The local
   LRU cache is the source of truth; the bloom filter is rebuilt periodically
   from current LRU keys, automatically handling evictions.

5. **No centralized coordinator:** Every node is equal. The system tolerates
   node failures, network partitions, and uneven load without requiring leader
   election or consensus.

---

## References

- Fan et al., "Summary Cache: A Scalable Wide-Area Web Cache Sharing Protocol"
  (IEEE/ACM TON, 2000) — original bloom filter for cooperative web caching
- Mooncake (FAST'25 Best Paper) — centralized KV cache for LLM inference
- LMCache — event-based distributed KV cache for vLLM
- llm-d — centralized indexer for distributed inference
- KVDirect — discovery overhead is 18-23% of TTFT (IEEE Cloud 2025)
- Preble (ICLR'25) — 85-97% shared tokens in production LLM workloads
