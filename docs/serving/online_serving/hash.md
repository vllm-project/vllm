# Prefix hashes

`POST /hash` renders a prompt and returns the running engine's prefix-cache
hashes without generating tokens or allocating KV blocks. Start the Python
server with `--enable-scale-out` and prefix caching enabled. The endpoint uses
the server's API-key authentication. It requires a running engine; the standalone
render server cannot determine the engine's resolved cache geometry.

Send an OpenAI chat-completion or completion request unchanged:

```bash
curl http://localhost:8000/hash \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hello"}]}'
```

For pretokenized requests, send `model`, `token_ids`, and optional `cache_salt`
and `features`. `features` has the same `mm_hashes` and `mm_placeholders` shape
as `/render`; tensor data is unnecessary for hashing. Raw multimodal inputs
use chat messages so vLLM performs placeholder expansion. Completion batches
return one entry in `prompts` per prompt.

The response contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | Wire contract version, currently `1`. |
| `model` | Primary served base-model name. LoRA identity is included in the hashes. |
| `hash_algorithm` | The engine's prefix-cache hash algorithm. |
| `hash_encoding` | `hex`: full hash bytes encoded as hexadecimal strings. |
| `hash_block_size` | Tokens between consecutive hash boundaries. |
| `num_hash_lookahead_tokens` | Additional input tokens needed to compute each hash. |
| `groups` | Cache-group indices, kinds, effective block sizes and sliding windows. |
| `prompts` | Input token counts and ordered `block_hashes` for each prompt. |

Only complete hash blocks with sufficient lookahead are returned. A group's
full-block hashes are every `group.block_size / hash_block_size` element of the
chain, starting at that stride minus one. Group block sizes include DCP scaling.
Partial cache checkpoints may use intermediate hashes from the same chain.

Hashes include the engine's seed, LoRA identity, cache salt, multimodal
identifiers and positions, and configured lookahead. The frontend uses vLLM's
shared request hasher with the initialized `PrefixHashConfig` received from the
engine at startup, including its exact first-parent hash and cache-group geometry.
With multiple DP engines behind an API server, all ranks must supply the same
hash configuration and geometry. Use compatible
hash settings across replicas, including a shared `PYTHONHASHSEED` when using
non-cryptographic algorithms. Different versions or serving configurations can
produce different hashes. Consumers must use the metadata in each response.

KV events export hash bytes, or the low 64 bits when
`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` is enabled. A consumer of integer events
must apply that same conversion to `/hash` results. Match events using the
base model, group index and hash; a hash alone does not identify a model.

Hashing reports content identity, not residency. Cache state comes from KV
events, including removals and resets. Missing events still require the
consumer's replay or resynchronization policy.

The endpoint supports decoder-only OpenAI chat, completions, and token inputs.
It does not support prompt embeddings, stateful Responses, Anthropic requests,
or raw `content_parts` on token inputs. Hashing runs in a frontend thread without
a per-request engine RPC. Measure its preprocessing and hashing cost for the
intended workload.
