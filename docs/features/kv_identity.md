# KV Identity

Engines that exchange KV cache blocks, through NIXL disaggregated prefill or a shared offloading tier, must compute the same KV bytes for the same token ids. The KV identity is one value per engine that says so: engines with equal identities compute the same KV for the same token ids, up to floating-point rounding order.

The identity covers the target model:

| Component | Content |
| - | - |
| `weights` | Digest of the checkpoint files the model was loaded from: each file's path in its checkpoint folder and its SHA-256 |
| `config` | Digest of the model and KV cache configuration that changes the KV: the model's Hugging Face config after `hf_overrides`, dtype, quantization, `max_model_len`, `cache_dtype`, `kv_cache_dtype_skip_layers` and the Mamba cache dtypes, among others |
| `code` | The vLLM version |
| `libraries` | The Transformers version |

Settings that change only placement or rounding order are not part of it, so engines may still differ in them: tensor parallel size, block size, KV cache layout, compilation and CUDA graph settings, memory limits, the scheduler, and the speculative decoding configuration. The connectors check layout, block size and draft model compatibility themselves.

## Weights

For a model in the Hugging Face cache, the SHA-256 of each weight file is known from the cache layout, so identifying the weights costs nothing at startup. A local copy of the same files has the same identity.

Weights outside the Hugging Face cache are identified by their path and revision, which cannot tell two checkpoints at the same path apart. Pass `--kv-identity-hash-local-weights` to hash them instead. One worker hashes them in the background while the weights load, so the hash mostly reads what the load already brought into memory; startup waits only for whatever hashing outlasts the load. The `weights_source` component says which applies: `hf_cache`, `hashed`, `path`, or `dummy` for `--load-format dummy`.

Weights loaded by the Run:ai streamer, Tensorizer, ModelExpress and sharded-state loaders are always identified by path.

## Reading the identity

- The `kv_identity` label of `vllm:cache_config_info` carries the identity, so a router can pair engines whose labels are equal.
- `GET /kv_identity` returns the identity and its components for each data-parallel rank, so a mismatch names its cause:

```json
{
  "0": {
    "kv_identity": "5c1e...",
    "components": {
      "weights": "9f2a...",
      "config": "77b0...",
      "code": "vllm==0.20.0",
      "libraries": "transformers==5.18.0",
      "weights_source": "hf_cache"
    }
  }
}
```

The engine also logs the identity and its components at startup.

## Enforcing it

Enforcement is opt-in. Set `enforce_kv_identity` in `kv_connector_extra_config` on every engine that exchanges KV:

```bash
--kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"enforce_kv_identity": true}}'
```

- **NixlConnector** adds the identity to its handshake compatibility hash, so engines with different identities refuse to pair. See [What must match between P and D](nixl_connector_compatibility.md#what-must-match-between-p-and-d).
- **OffloadingConnector** adds the identity to the storage namespace of its secondary tiers and to the P2P peer fingerprint, so engines with different identities neither read each other's files nor connect as P2P peers. Enabling it moves the namespace, so blocks stored before are no longer found.

!!! note
    The identity is computed once at startup. Weights changed at runtime, through `reload_weights`, weight updates or sleep mode, do not update it yet.
