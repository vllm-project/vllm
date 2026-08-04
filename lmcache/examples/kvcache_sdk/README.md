# KV Cache SDK Examples

These examples show how to use the Python SDK to store and retrieve
KV cache tensors through the LMCache MP HTTP API. The SDK path is memory-first:
applications can retrieve a tensor, pair it with new token metadata, and store
it again without writing the tensor through a local storage format. Store will
return False if LMCache already have the KV Cache with the same token sequence.

## End-to-end vLLM flow

First, start up the vLLM and LMCache server by running commands listed in the
first cell of `e2e_kv_edit.ipynb`.

Then, starting from cell 2 of `e2e_kv_edit.ipynb`, it is doing below experiment
flow:
1. Send a source prompt to vLLM so the normal connector stores KV in LMCache.
2. Retrieve the source KV cache into an in-memory tensor with `retrieve()`.
3. Build a target token-ID prompt: the same length as the source prompt and
   identical apart from a few different synthetic leading tokens.
4. Store the source KV under the target prefix with `store()`.
5. Send the target token IDs to vLLM so the target prefix hits the remapped KV.
6. Print retrieve counts, latencies, response previews, and whether the source
   and target outputs match.

The target prompt starts with different token IDs, so it does not rely on a
serving-engine local prefix match. Because the prompts are identical apart from
those leading tokens, reusing the source KV reconstructs the same final context
for the target request, which should produce the same deterministic output.

The core SDK pattern used by the end-to-end example is:

```python
import lmcache.sdk.kvcache as lmc_sdk

ctx = lmc_sdk.connect(
    url="tcp://localhost:6555",        # ZMQ message queue
    http_url="http://localhost:8080",  # HTTP config / status
    model_name="...",
)

kv = lmc_sdk.retrieve(ctx, tokens=source_tokens)
if kv is not None:
    lmc_sdk.store(ctx, kv=kv, tokens=target_tokens)

lmc_sdk.close(ctx)
```

### Requirements

- An LMCache MP server running with HTTP enabled.
- A model already registered with that server. Check `/status` for the
  registered `model_name`, `chunk_size`, layer count, dtype, and hidden dim.
- A homogeneous `KV_2LTD` layout.

Use `--cache-salt` on all commands when storing and retrieving from a
non-default namespace.
