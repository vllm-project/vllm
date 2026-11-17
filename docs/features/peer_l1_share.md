# Peer L1 KV cache sharing

`PeerL1Connector` reuses prompt-prefix KV directly from another data-parallel
worker's GPU memory. It helps DP+EP MoE deployments when requests from one
session reach different ranks and would otherwise recompute the same prefix.
It is an optional cache: a miss falls back to local computation.

Workers on the same host use CUDA/HIP IPC. Each rank independently looks up reusable prefixes on its peers.

## Requirements

- A MoE model with prefix caching enabled on every participating worker.
- TP=PP=DCP=PCP=1; scale with DP and optionally expert parallelism.
- Identical weights, tokenizer, position encoding, dtype and KV geometry.
- A C++17 compiler and pthread for the native control service. Set `CXX` if needed.
  Its build cache is under `$VLLM_CACHE_ROOT/peer_l1`.

On ROCm, set these before launching either path so copies retain high priority:

```bash
export DEBUG_HIP_IGNORE_STREAM_PRIORITY=0
export ROCPROFILER_QUEUE_INTERPOSITION=1
```

Legacy profiler interception, including some profiling modes, can override
queue priority. Check the connector's startup warning if copies delay compute.

## Same-instance IPC

Enable the connector on all DP ranks of the instance:

```bash
vllm serve ibm-research/PowerMoE-3b \
  --data-parallel-size 2 --enable-expert-parallel --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"PeerL1Connector","kv_role":"kv_both"}'
```

IPC participants must share a process failure domain. A failed IPC load stops
execution rather than releasing source rows while a copy might still read them.

## Limits and metrics

The connector requires a shared process failure domain. Sleep mode, fabric
LoadStore and composition with L2 connectors are unsupported.

Metrics use the `vllm:peer_l1_` prefix with vLLM's per-engine labels. Lookups,
hit tokens, granted rows, busy replies, truncations and wait histograms describe
reader decisions. Owner pin/window/reserve gauges describe admission headroom;
copy-byte and GPU-time histograms describe completed load steps.

IPC handles and the local control service belong to a trusted deployment.
See the [ownership notes](../design/peer_l1_share.md) for lifetime invariants.
