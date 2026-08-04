# Q Ring Buffer

> Worker-side capture of per-layer **query** tensors, offloaded to LMCache
> through the paged-KV machinery. Runs inside vLLM (the producer), not the
> consumer SDK.

## Goal

Attention query (Q) tensors are transient. To persist them for later reuse
(retrieved via the [`qcache`](context.md) SDK), the worker stages each layer's
Q into a GPU **ring buffer** laid out like a paged KV cache, then stores whole
blocks to the LMCache MP server over the existing STORE path — keyed under a
query-specific model name `<model>##query`.

This is driven by the vLLM MP connector's `QRingBufferCapture` and gated by the
`transfer_intermediate_tensors` flag. Nothing here is called by SDK users.

## Components

- **`QRingBuffer`**: a GPU tensor of logical shape
  `[num_layers, num_blocks, block_size, hidden_dim]`, paged in blocks so it
  reuses LMCache's paged-KV transfer kernels. Per-layer views are exposed as
  `tensors = {"lmcache_q_layer_{i}": ...}`.
  - `allocate(n)`: allocate `n` free block ids, or `None` if too few are free.
  - `free(block_ids)`: freeing block IDs.
  - `num_free_blocks()`: getting the number of free blocks.
  - `scatter(layer_index, query, ring_slots)`: write one layer's flattened
    query rows into the ring at the given slot ids (`-1` slots are dropped).

- **`QRingBufferCapture`**: hooks the connector's forward lifecycle:
  - `setup_q_ring(kv_caches, kv_cache_config, vllm_config)`: pick the
    attention layers, calculate ring size, and register it (at KV register).
  - `save_q_layer(layer_name, metadata, **kwargs)`: per attention layer:
    build the per-step plan on the first layer, then scatter that layer's Q.
  - `batched_submit_qstore_requests(event)`: at forward exit, consume the
    plan and submit one Q store per request (reset when there is none).

### Row-to-token attribution

Under continuous batching a step's query tensor concatenates rows from many
requests, and a request's row count is its **scheduled** token count for the
step — which need not equal its store op's chunk-aligned token count (prompt
tails past the last chunk boundary still produce rows, and the batch's row
order need not match the connector metadata's request order). The plan
therefore never assigns rows positionally. Instead it matches rows to op
tokens through `attn_metadata.slot_mapping`: row `r` writes its KV to GPU slot
`slot_mapping[r]`, and op token `i` lives in
`block_ids[i // block_size] * block_size + i % block_size` (op block lists are
pre-sliced to `[start, end)`). Each GPU slot is written at most once per step,
so a full intersection is a bijection between an op's tokens and its rows; an
op whose tokens are only partially present in the step (e.g. computed in an
earlier chunked-prefill iteration) is skipped **individually**, without
affecting other requests in the step, and non-STORE requests are simply
ignored rather than disabling the whole step's capture.

- **`QRingBufferAdapter`**: owns the ring's interaction with LMCache:
  - `register_q_ring(...)`: allocate the `QRingBuffer` and register it via
    `transfer_ctx.register_q` (`REGISTER_Q_CACHE`) under `q_model_name`.
  - `submit_q_store_request(request_id, op, ring_block_ids, event, ...)`:
    build the key from token ids, send `STORE_Q`, track the future + blocks.
  - `reclaim_finished_q_stores()`: free ring blocks once a store completes
    (called from the connector's `get_finished`).
  - `shutdown_q_ring()`: send `UNREGISTER_Q_CACHE` on teardown.

## Lifecycle

1. **Register**: `setup_q_ring` → `register_q_ring` → `REGISTER_Q_CACHE`; the
   server's `QStoreModule` builds the Q cache context.
2. **Capture**: each attention layer's `save_q_layer` scatters Q into the
   ring blocks reserved for that step's store requests.
3. **Store**: `batched_submit_qstore_requests` sends `STORE_Q` per request;
   the server copies the ring blocks to CPU exactly like a KV store.
4. **Reclaim**: `reclaim_finished_q_stores` frees blocks as stores finish.
5. **Shutdown**: `shutdown_q_ring` sends `UNREGISTER_Q_CACHE`.

## Cache addressing

Q shares the KV store path but under `q_model_name = "<model>##query"`, so Q
and KV objects never collide. The Q ring registers under the worker's own
`instance_id` (the same id as its KV cache), disambiguated by that model name.

## Current Limitations

- **CUDA / lmcache-driven transport only** — `register_q` / `submit_q_store`
  are unimplemented on the engine-driven (CPU) path.
- **Only transferring Q of prefill step**: if `allocate` cannot reserve enough
  blocks for a step, that step's Q capture is skipped (blocks freed, not
  queued). This happens when decoding. Currently acceptable because the SDK is
  used in an offline manner.
