# KV Event Metadata

KV events can report the logical producer of a cache event (`source`) and the
sharing scope of the reported resource (`sharing_scope`). Both fields are
**disabled by default**. Enable them explicitly through the existing
`--kv-events-config` JSON argument:

```bash
vllm serve <model> \
  --enable-prefix-caching \
  --kv-events-config '{"enable_kv_cache_events":true,"enable_kv_event_metadata":true,"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558"}'
```

To keep publishing events with the legacy event fields, omit
`enable_kv_event_metadata` or set it to `false`:

```bash
vllm serve <model> \
  --enable-prefix-caching \
  --kv-events-config '{"enable_kv_cache_events":true,"enable_kv_event_metadata":false,"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558"}'
```

| KV events enabled | Metadata enabled | Publication |
| --- | --- | --- |
| `false` | Any value | Events remain disabled. |
| `true` | Omitted or `false` | `source` and `sharing_scope` are completely omitted, including producer-supplied values. |
| `true` | `true` | Source is attached; known store/remove scope is included. Unknown scope is omitted. |

The setting is fixed at initialization. Set it to `false` and restart to restore
the legacy event format. When upgrading from a version that published metadata
automatically, explicitly set it to `true` to retain that metadata. The existing runtime KV event
configuration query reports its actual value together with resolved publisher
endpoints, including ports selected by `tcp://*:0` and data-parallel rank port
offsets. No additional discovery endpoint is introduced.

## Event fields and compatibility

The three event types keep their existing fields. `BlockStored` and
`BlockRemoved` can carry both `source` and `sharing_scope`.
`AllBlocksCleared` can carry only `source`; it has no sharing scope. There is no
new `kind` field or batch field, and the transport frames and positional batch
format are unchanged.

The following JSON objects are **partial event examples** with metadata enabled;
required payload fields are omitted here for clarity:

```json
[
  {"type":"BlockStored","medium":"GPU","source":{"source_id":"cache_manager"},"sharing_scope":"instance"},
  {"type":"BlockRemoved","medium":"CPU","source":{"source_id":"connector/0","connector_name":"OffloadingConnector"},"sharing_scope":"instance"},
  {"type":"AllBlocksCleared","source":{"source_id":"cache_manager"}}
]
```

With metadata disabled, these same events retain their existing payload fields
and omit the two metadata keys entirely. Sending the keys with `null` values is
not the disabled-mode format. Decoders can still accept old events and explicit
`null` metadata as unknown values.

Strict consumers that reject unknown event fields can continue using default
mode. Update those consumers before opting in. Tolerant consumers can ignore
the new map fields. Replay retains the original encoded bytes for the chosen
mode; it does not add, remove, or reconstruct metadata. This does not promise
identical timestamps or sequence numbers across separate runs.

## Logical producer identity

`source` contains a required, nonempty `source_id` and an optional
`connector_name` taken from the configured connector name:

| Producer configuration | Source ID |
| --- | --- |
| Native cache manager | `cache_manager` (no connector name) |
| One connector | `connector/0` |
| MultiConnector children | `connector/0`, `connector/1`, etc. |
| Nested child of the second MultiConnector entry | `connector/1/0`, `connector/1/1`, etc. |

Consumers must treat the complete `source_id` as an opaque identifier.
Indices refer to configuration positions. Repeated connector types have
separate identities, and a child that contributes no events keeps its position.
An opaque plugin composite is one source unless it exposes the supported
MultiConnector child boundary; vLLM does not inspect arbitrary plugin internals.
Framework attribution overrides a producer-supplied source without changing the
producer's original event object.

The complete identity context is the publisher runtime, data-parallel (DP)
rank, and source ID. Source IDs are not storage pool identifiers, physical
worker identifiers, or unique event IDs. Equal event payloads from separate
sources remain separate announcements, including when metadata is disabled.
IDs may be reused after restart or configuration changes. Consumers must reset
their source context as part of deployment lifecycle handling; there is no
automatic cross-restart detection.

A source on `AllBlocksCleared` identifies the reporter. It does not narrow the
existing clear operation to that source or change its invalidation contract.

This attribution does not add worker-event forwarding to MultiConnector.
Worker-originated events still depend on the connector's existing transport;
source paths alone do not establish that a child event reaches the publisher.

## Resource sharing scope

`sharing_scope` describes the reported resource independently of its source:

| Resource | Scope declaration |
| --- | --- |
| Native GPU prefix cache | `instance` |
| Native primary CPU offload cache | `instance` |
| Mooncake Store successful store | `shared` |
| Filesystem / object-store offload tier | Optional `instance` or `shared` tier configuration |
| Resource without an explicit declaration | Unknown; key omitted |

Resource declarations are retained internally in either publication mode. A
filesystem path, bucket name, `CPU`/`STORAGE` medium, or `LOCAL`/`REMOTE` locality
does not establish scope. `LOCAL` and `shared` can coexist. `ownership` continues
to identify an offload tier; it is independent of source and scope.

`shared` is a deployment property, not authorization, a globally unique storage
domain, or a guarantee of continued backend residency. Consumers must associate
reports with the actual deployed storage domain separately.

The metadata switch controls only `source` and `sharing_scope`. It is independent
of offloading's `self_describing_kv_events`, which selects full versus placeholder
event payloads. It does not change block/chunk matching, removal hashes,
reference counting, or invalidation completeness. See
[native KV offloading](kv_offloading_usage.md) for tier configuration and
[Mooncake Store](mooncake_store_connector_usage.md) for its store-event boundary.
