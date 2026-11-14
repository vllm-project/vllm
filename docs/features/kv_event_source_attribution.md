# KV cache event source attribution

KV cache events can include their logical producer. Enable the optional
`enable_source_attribution` flag in `--kv-events-config`:

```bash
vllm serve <model> --kv-events-config \
  '{"enable_kv_cache_events":true,"enable_source_attribution":true,"publisher":"zmq","endpoint":"tcp://*:5557"}'
```

The flag defaults to `false`. With it disabled, event encoding and connector
aggregation are unchanged. With it enabled, `BlockStored`, `BlockRemoved` and
`AllBlocksCleared` maps contain an optional inline `source` map:

```json
{"source_id":"connector/0/1","kind":"connector","connector_name":"OffloadingConnector"}
```

`cache_manager` (`kind: native`) identifies the native KV cache manager.
`connector` identifies a single connector. MultiConnector children append their
zero-based configured index, recursively: `connector/0`, `connector/1`,
`connector/0/1`. Inactive children do not change these paths. Repeated connector
types have distinct paths. `connector_name` is the configured factory name, not
a unique identity. An opaque plugin composite is one source unless it implements
`take_events_with_source` to expose its children.

## Identity and lifecycle

Paths identify configured occurrences for one scheduler lifetime with fixed
connector topology. They are stable across batches and replay within that
lifetime. Pair paths with the publisher connection/lifetime and
`data_parallel_rank`; two ranks or engines may use identical paths.
Reordering or replacing connectors starts a new identity lifetime. There is no
wire epoch in the current event protocol: restarting at the same endpoint can
produce exactly the same source and payload. Consumers must reset their state
on an engine restart using deployment lifecycle information. Paths alone cannot
detect a missed restart or prove continuity across it.

Source identifies the reporter. It does not describe a storage domain, sharing,
cache accessibility, namespace, retrieval granularity or exclusive ownership.
Different sources may report the same physical cache, and one source may report
several media. Consumers still need deployment configuration and the existing
cache contract to interpret residency and reference counts.

## Event and consumer compatibility

Attribution is applied on the scheduler after worker quorum/aggregation and
before publication. It preserves payloads, event order and multiplicity. Events
retained by connectors are not mutated. Producer-supplied source fields are
replaced by framework metadata at this boundary. Workers and existing connector
`take_events()` APIs continue to use their existing event representation.
MultiConnector worker event transport requires the separate upstream work in
[PR #62637](https://github.com/vllm-project/vllm/pull/62637); attribution does not
supply missing transport.

A source on `AllBlocksCleared` identifies who emitted it; it does **not** narrow
or redefine the effect of the existing clear contract. In particular, do not
infer that only records with the same source should be cleared.

The default wire encoding is unchanged. New decoders accept absent or null
`source` as unknown. Existing msgspec map decoders ignore the new field; strict
consumers that reject unknown fields need an update before enabling the flag.
The `/kv_event_sources` response also retains its legacy fields when attribution
is disabled; `enable_source_attribution: true` is reported only when enabled.
Registered custom publishers keep their existing constructor arguments when
attribution is disabled. Opting in forwards `enable_source_attribution=True`.
Replay preserves the original encoded payload including the inline metadata.
Event structs gain one optional reference slot even with the flag disabled.
The flag adds a shallow event copy per published event and a descriptor per
visited connector path per drain; it does not retain a residency index or copy
token/hash lists.

Pending publisher identity and snapshots work in
[PR #57789](https://github.com/vllm-project/vllm/pull/57789) requires a source-aware
recorder before combining snapshots with attribution: its current scope keys
merge sources and synthesized events omit source. Snapshot bootstrap must not
silently discard this metadata. This feature does not introduce a competing
publisher identity or change that PR's snapshot/clear semantics.
