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

- `GET /kv_identity` returns the current identity and its components for each data-parallel rank, so a router can pair engines whose identities are equal and a mismatch names its cause:

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

The `kv_identity` label of `vllm:cache_config_info` carries the identity the engine started with, and the engine logs the identity and its components whenever it changes.

## Enforcing it

Enforcement is opt-in. Set `enforce_kv_identity` in `kv_connector_extra_config` on every engine that exchanges KV:

```bash
--kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"enforce_kv_identity": true}}'
```

- **NixlConnector** adds the identity to its handshake compatibility hash, so engines with different identities refuse to pair. See [What must match between P and D](nixl_connector_compatibility.md#what-must-match-between-p-and-d).
- **OffloadingConnector** adds the identity to the storage namespace of its secondary tiers and to the P2P peer fingerprint, so engines with different identities neither read each other's files nor connect as P2P peers. Enabling it moves the namespace, so blocks stored before are no longer found.

## Weight changes

The identity follows weights changed at runtime. While the engine cannot identify its weights, the identity is `unknown-<random>`, which matches no other engine; `weights_source` is then `unknown`.

| Change | Identity |
| - | - |
| `reload_weights` from files, with no `weights_iterator` | Identified by the files, when they are in the Hugging Face cache or hashed with `--kv-identity-hash-local-weights` and every weight was reloaded; otherwise unknown |
| `reload_weights` with a `weights_iterator`, `reset_weights`, sleep level 2 | Unknown |
| A weight update, from `start_weight_update` to `finish_weight_update` | Unknown |
| `finish_weight_update` with a `weight_version`, or `update_weight_version` | The last files loaded and the version; `weights_source` is `weight_version` |
| Sleep level 1, draft model updates | Unchanged |

A weight version identifies the weights only as well as the caller assigns it: engines that loaded the same files and set the same version match, so use a version for exactly one set of weights. Setting a version while a weight update is incomplete does not change the identity.

With `enforce_kv_identity`, before publishing a new identity the engine resets the caches computed with the previous weights: the prefix cache, which preempts running requests, the connector's cache, and the encoder cache. If the reset fails, for example while requests wait for KV from a remote instance, the identity stays unknown until the next weight change or version. Then:

- NixlConnector advertises the new identity in its handshake, and instances check each other again before their next transfer.
- OffloadingConnector stores and looks up files under the new identity, and P2P peers reconnect and are checked again. While the identity is unknown, secondary tiers are neither read nor written.

Change weights while generation is paused, as in the RL weight-transfer API, so that no KV is computed while the weights change. Methods of a custom `--worker-extension-cls` that change weights are not detected: pause generation, change the weights, and set the weight version before resuming.

The identity assumes that LoRA adapters do not change under the same name and path, since cached KV is keyed by them.
