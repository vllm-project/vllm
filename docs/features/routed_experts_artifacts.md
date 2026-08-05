# Routed-experts artifacts

vLLM can return the routed expert IDs selected for each prompt and generated
token. The data is captured by Model Runner V2 and stored in shared memory by
an Artifact Connector. Full blocks use the same content hashes as the KV cache,
so prefix-cache and KV-offload hits can reuse the corresponding routing data.

## Usage

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <model> \
  --enable-prefix-caching \
  --enable-return-routed-experts
```

The OpenAI-compatible completion and chat responses include `routed_experts`
as a base64-encoded NumPy array. Its shape is
`(num_returned_tokens, num_moe_layers, num_experts_per_token)`.

Native CPU KV offload can be enabled at the same time:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <model> \
  --enable-prefix-caching \
  --enable-return-routed-experts \
  --kv-offloading-size 8 \
  --kv-offloading-backend native
```

Without a KV connector, the SHM capacity defaults to the local GPU KV-cache
capacity. With a KV connector, set the SHM capacity explicitly:

```bash
--artifact-config '{"max_shm_bytes": 1073741824}'
```

The store fails closed if a KV hit refers to an artifact that has been evicted;
increase `max_shm_bytes` in that case. Artifact reuse never reduces or changes
KV-cache hits.

## Current constraints

- Model Runner V2 is required.
- Pipeline and context parallelism are not supported.
- KV transfer must use `kv_role=kv_both`; P/D disaggregation is not supported.
- Normal streaming output is supported. Resumable streaming input is rejected.
- The SHM directory must be under `/dev/shm` and must be shared by the output
  worker and EngineCore processes.
