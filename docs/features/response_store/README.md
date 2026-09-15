# Responses Session Store

The Responses session store keeps the tokenized conversation history of a
Responses session on the server. When several requests share the same
`x-session-id`, follow-up requests can append only their new input to the stored
history instead of rendering and tokenizing the whole conversation again.

The store is disabled by default and is enabled per server with
`--responses-store-config '{"enabled": true}'`.

## Features

- **Two storage tiers**: an always-on in-memory tier and an optional SQLite disk
  tier that can restore evicted sessions. Disk writes are batched by a background
  writer, so saving never blocks request completion.
- **Per-tier retention**: independent idle TTLs and capacity watermarks for the
  memory and disk tiers. Cleanup removes expired records first and then the oldest
  records, bounded by a per-batch candidate and byte budget.
- **Eviction safety**: a memory copy is evicted only after a same-version,
  readable copy is validated on disk, so eviction never removes the only
  recoverable history.
- **Encryption at rest**: disk token payloads are encrypted and authenticated with
  AES-GCM, using either an ephemeral process-local key or a caller-provided key
  file. Incremental writes append ciphertext frames without decrypting existing
  history.
- **Automatic key rotation**: with a key file, the whole database is re-encrypted
  and the active key replaced atomically, and an interrupted rotation is recovered
  on the next startup.
- **Incremental rendering**: models whose chat template supports incremental
  rendering append new input to the reused history. A DeepSeek V4 tokenizer
  adaptation is included.
- **All request modes**: token collection and saving cover streaming,
  non-streaming, and background Responses requests, including incremental tool
  follow-up turns.

## Requirements

- The model's chat template must support incremental rendering. Harmony models do
  not support cross-request incremental token reuse.
- Requests that set `use_store` or `use_incremental_token` must send the
  `x-session-id` header.
- File-key mode additionally requires an explicit persistent disk path, a
  writable key file and its parent directory, and a single API server process.

## Configuration

### CLI arguments

| Argument | Description |
| --- | --- |
| `--responses-store-config` | Store settings as a JSON object. Use `{"enabled": true}` to enable the store. |
| `--responses-store-disk-path` | SQLite file path. Defaults to a process-local file in the temp directory. |
| `--responses-store-key-file` | File containing a Base64-encoded 32-byte AES-256 key. Preserves key metadata across restarts and enables automatic key rotation. |

### `--responses-store-config` reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable the store. |
| `disk_enabled` | bool | `true` | Enable the SQLite disk tier. Set to `false` for memory-only storage. |
| `memory_capacity_mb` | int | `1024` | Memory tier capacity in MiB. |
| `disk_capacity_mb` | int | `10240` | Disk tier capacity in MiB. |
| `memory_low_watermark` / `memory_high_watermark` | float | `0.4` / `0.8` | Memory-tier capacity ratios. Cleanup starts at the high watermark and stops at the low watermark. |
| `disk_low_watermark` / `disk_high_watermark` | float | `0.4` / `0.9` | Disk-tier capacity ratios, with the same start/stop behavior. |
| `memory_ttl_seconds` | int | `1800` | Memory idle TTL in seconds. `0` disables expiration for the tier. |
| `disk_ttl_seconds` | int | `36000` | Disk idle TTL in seconds. `0` disables expiration for the tier. |
| `cleanup_interval_seconds` | float | `300` | Interval between cleanup passes. |
| `cleanup_max_candidates` | int | `256` | Maximum number of sessions a single cleanup batch may process. |
| `cleanup_max_bytes_mb` | int | `2048` | Maximum bytes a single cleanup batch may reclaim, in MiB. |
| `num_shards` | int | `64` | Number of locks used to serialize tier operations for the same session. |
| `disk_write_interval_seconds` | float | `0.05` | Merge window that batches concurrent disk writes into one transaction. |

!!! note
    Capacity limits trigger cleanup; they do not reject writes.

### Examples

Enable a disk-backed store with a persistent key file:

```bash
mkdir -p response-store-data
(umask 077; openssl rand -base64 32 > response-store-data/key)

vllm serve /path/to/model \
  --served-model-name response-store-demo \
  --responses-store-config '{"enabled":true}' \
  --responses-store-disk-path ./response-store-data/responses.sqlite3 \
  --responses-store-key-file ./response-store-data/key
```

Keep everything in memory by disabling the disk tier and omitting the key file and
disk path:

```bash
vllm serve /path/to/model \
  --served-model-name response-store-demo \
  --responses-store-config '{"enabled":true,"disk_enabled":false}'
```

## Usage

### Save and reuse a session

Two request fields control the store. Both default to `false` and both require the
`x-session-id` header.

| Field | Description |
| --- | --- |
| `use_store` | Save the token IDs added by this request to the session store after generation completes. |
| `use_incremental_token` | Reuse the tokenized history stored for `x-session-id` and append only the new input. |

Send the full context on the first request and enable token storage:

```bash
curl http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'x-session-id: example-session' \
  -d '{
    "model": "response-store-demo",
    "input": "Remember that the project name is Demo.",
    "use_store": true
  }'
```

After the first request completes, send only the new input with the same session ID.
Enable both reuse and storage to retain the new turn:

```bash
curl http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'x-session-id: example-session' \
  -d '{
    "model": "response-store-demo",
    "input": "What is the project name?",
    "use_incremental_token": true,
    "use_store": true
  }'
```

!!! note
    `use_store` saves session token IDs. It is separate from the Responses API
    `store` field, which controls storage of response objects.

### Request behavior

| Condition | Result |
| --- | --- |
| Stored history is unavailable for `use_incremental_token` | HTTP 500 with error type `incremental_context_miss` |
| Session store is not configured on the server | HTTP 503 with error type `session_store_unavailable` |
| `x-session-id` header is missing | HTTP 400 |
| `use_incremental_token` with a Harmony model | HTTP 400 |

On `incremental_context_miss`, resend the full context with
`use_incremental_token=false`, keeping `use_store=true` to rebuild the session.

## Encryption and restart behavior

The key file must contain a Base64-encoded 32-byte AES-256 key and must exist
before startup. File-key mode requires an explicit persistent disk path and a
single API server process. The file and its parent directory must be writable so
the service can stage and atomically replace the key during rotation.

The service checks for rotation every hour, with a default rotation interval of
90 days. Rotation generates a new key, re-encrypts the database token payloads,
and replaces the active key file. Pending-key state supports recovery from an
interrupted rotation.

**Session data is cleared when the store is initialized, including after a server
restart.** File-key mode preserves key metadata, not session history across
restarts, so clients must resend their full context after a restart. Without a key
file, the store uses an ephemeral process-local key.

## Metrics

Each cleanup pass logs an `event=response_store_metrics` record that reports the
cleanup duration, per-tier usage and freed bytes, and the encryption operation
count, plaintext bytes, duration, and throughput accumulated since the previous
pass.
