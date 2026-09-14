# Responses Session Store

The Responses session store saves token IDs for reuse across requests with the
same `x-session-id`. It allows supported models to append new input to an existing
tokenized history without rendering and tokenizing that history again.
The feature is disabled by default.

## Features

- Memory storage with an optional SQLite disk tier and batched disk writes.
- Independent idle TTLs and capacity watermarks for each tier. Cleanup prioritizes
  expired records, then older records, with limits on candidates and bytes per batch.
- Version checks and disk-copy validation before normal memory eviction.
- AES-GCM encryption and authentication of disk token payloads, with either an
  ephemeral key or a caller-provided key file.
- Automatic database-wide key rotation and interrupted-rotation recovery when
  using a key file.
- Token collection and saving for streaming, non-streaming, and background
  Responses requests, including incremental tool follow-up turns.

Cleanup and key-rotation schedulers run as asyncio tasks. The application
lifespan starts these tasks and closes the store during shutdown.

## Enable the store

For a new store, create a directory and generate a key once. The following
examples use a POSIX shell and require OpenSSL for key generation.

```bash
mkdir -p response-store-data
(umask 077; openssl rand -base64 32 > response-store-data/key)
```

Start a model whose chat template supports incremental rendering:

```bash
vllm serve /path/to/model \
  --served-model-name response-store-demo \
  --responses-store-config '{"enabled":true}' \
  --responses-store-disk-path ./response-store-data/responses.sqlite3 \
  --responses-store-key-file ./response-store-data/key
```

`--responses-store-config` accepts a JSON object. The disk path and key file
remain separate CLI arguments. Common defaults are:

| Configuration field | Default |
| --- | --- |
| `enabled` | `false` |
| `disk_enabled` | `true` |
| `memory_capacity_mb` / `disk_capacity_mb` | `1024` / `10240` |
| `memory_low_watermark` / `memory_high_watermark` | `0.4` / `0.8` |
| `disk_low_watermark` / `disk_high_watermark` | `0.4` / `0.9` |
| `memory_ttl_seconds` / `disk_ttl_seconds` | `1800` / `36000` |
| `cleanup_interval_seconds` | `300` |

Capacity values use MiB. TTLs and intervals use seconds; a TTL of `0` disables
expiration for that tier. To use only memory, set `disk_enabled` to `false` and
omit the key-file argument.

## Save and reuse a session

The request fields `use_store` and `use_incremental_token` both default to
`false`. Enabling either field requires the `x-session-id` header.

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

After the first request completes, send only the new input with the same
session ID. Enable both reuse and storage to retain the new turn:

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

`use_store` saves session token IDs. It is separate from the existing Responses
API `store` field, which controls storage of response objects.

When history is unavailable, the request returns HTTP 500 with error type
`incremental_context_miss`. Retry with the full context and
`use_incremental_token=false`. An unconfigured session store returns HTTP 503;
a missing session header returns HTTP 400.

## Encryption and restart behavior

The key file must contain a Base64-encoded, 32-byte AES-256 key. It must exist
before startup. File-key mode requires an explicit persistent disk path and a
single API server process. The file and its parent directory must be writable
so the service can stage and atomically replace the key during rotation.

The service checks for rotation every hour, with a default rotation interval of
90 days. Rotation generates a new key, re-encrypts the database token payloads,
and replaces the active key file. Pending-key state supports recovery from an
interrupted rotation.

**Session data is cleared when the store is initialized, including after a
server restart.** File-key mode preserves key metadata, not session history
across restarts. Clients must resend their full context after a restart.
Without a key file, the store uses an ephemeral process-local key.

## Limitations

- Incremental rendering requires support in the model's chat template. The
  implementation includes a DeepSeek V4 tokenizer adaptation. Harmony models
  do not support cross-request incremental token reuse.
- Expiration and eviction can make a session unavailable. Clients must handle
  `incremental_context_miss` and retain the context needed to rebuild a session.
- Disk writes are batched; request completion is not a durability guarantee.
- Performance improvements have not been quantified by benchmarks.

## Implementation and tests

The implementation lives in
[`vllm/entrypoints/openai/responses/store/`](../../../vllm/entrypoints/openai/responses/store/).
Responses request handling collects and reuses tokens, and the launcher
application lifespan owns the storage service.

With the project's development environment configured, run the focused tests
from the repository root:

```bash
uv run --no-sync .venv/bin/python -m pytest \
  tests/entrypoints/launchers/test_cli_args.py \
  tests/entrypoints/launchers/test_responses_store_lifespan.py \
  tests/entrypoints/openai/responses/test_store_config.py \
  tests/entrypoints/openai/responses/test_store_recovery.py \
  tests/entrypoints/openai/responses/test_serving_responses.py
```

These tests cover configuration, service lifecycle, key recovery and session
clearing, and Responses rendering, including incremental input and tool turns.
See the [contribution guide](../../contributing/README.md) for environment setup
and validation requirements.
