# Async Reinforcement Learning

## Overview

In a standard RL training loop, generation and training happen sequentially: the policy generates rollouts, then training runs on those rollouts, and the cycle repeats. During generation the training accelerators sit idle, and vice versa.

The **one-off pipelining** approach separates the generation and training phases into two parallel coroutines, allowing the model to generate new samples while simultaneously training on previously generated data. This can lead to better GPU utilization and greater training throughput.

However, this overlap introduces a complication: weights must be updated in the inference engine mid-flight, while requests may still be in progress.

## The Pause and Resume API

To safely update weights while the inference engine is running, vLLM provides `pause_generation` and `resume_generation` methods. These let the trainer coordinate a clean window for weight synchronization without losing in-flight work.

### pause_generation

```python
await engine.pause_generation(mode="keep", clear_cache=True)
```

The `mode` parameter controls how in-flight requests are handled, and whether
new requests are admitted while paused:

| Mode | In-flight requests | New requests |
| ---- | ------------------ | ------------ |
| `"abort"` | Aborted immediately, returning partial results (default) | Rejected with HTTP 503 until resume |
| `"wait"` | Running requests finish before the pause completes; queued ones run after resume | Rejected with HTTP 503 until resume |
| `"keep"` | Frozen in the queue; they resume when `resume_generation` is called | Accepted and queued |

`"abort"` and `"wait"` treat the pause as a generation boundary: from the moment
the pause starts (for `"wait"`, including while in-flight requests drain) until
`resume_generation` returns, the engine rejects new requests from every API server
rather than carrying them across the boundary. Clients should retry on 503. A
request racing the pause call itself may instead be treated as in-flight, and
input to an already-open streaming-input session is still accepted. Other pauses,
such as `sleep`, keep queueing new requests.

The `clear_cache` parameter controls whether to clear the KV cache and prefix cache after pausing.

`clear_cache=True` also resets a configured KV connector's cache. To clear only the local caches,
pause with `clear_cache=False`, then call
`reset_prefix_cache(reset_running_requests=True, reset_connector=False)`. Block hashes do not cover
the weights, so keep the connector's cache only if the weights do not change, or give requests a
`cache_salt` per weight version.

### resume_generation

```python
await engine.resume_generation()
```

Resumes the scheduler after a pause. Any requests frozen with `mode="keep"` will continue generating.

Wait for `pause_generation()` to finish before resuming, since a data-parallel MoE engine can reject an early resume. Once generation resumes, calling `resume_generation()` again has no effect.

If `resume_generation` is called while a `pause_generation` call is still waiting to complete (for example a `"wait"` pause that is still draining), that pause call fails with `Resumed before the pause completed.` (HTTP 500 over the HTTP endpoint) instead of returning later as if the pause had held.

### HTTP Endpoints

With `VLLM_SERVER_DEV_MODE=1`, the vLLM HTTP server exposes the same functionality via:

- `POST /pause?mode=keep` - Pause generation
- `POST /resume` - Resume generation
- `POST /abort_requests` - Abort in-flight requests without pausing the scheduler (send `{}` to abort all, or `{"request_ids": [...]}`)
- `POST /update_weight_version` - Set the `weight_version` label without changing weights (send `{"new_version": "v8"}`)
- `GET /weight_info` - Return the latest committed `weight_version`

!!! note "Data Parallelism"
    When using data parallelism with vLLM's **internal load balancer** (i.e. `data_parallel_backend="ray"`), pause and resume are handled automatically across all DP ranks -- a single call is sufficient. When using an **external load balancer** (i.e. multiple independent vLLM instances behind a proxy), you must send pause and resume requests to **every** engine instance individually before and after the weight update, and set the weight version on every instance too.

### Prefill/decode disaggregation

With [disaggregated prefill](../features/disagg_prefill.md), every prefill (P) and decode (D) instance is paused and resumed on its own, as with an external load balancer. For continuous rollouts, pause each instance with `mode="keep", clear_cache=False`, in any order:

- The pause completes once the instance has nothing left to compute. It does not wait for KV transfers: a P keeps the KV blocks a D has not read yet, and the transfer finishes after the pause or after resume.
- A paused instance keeps stepping without running the model while transfers are in flight, so it can keep renewing KV leases and receiving KV. A pause therefore stops model execution, not KV cache changes.
- While only D is paused, P keeps the blocks of every request D holds. If P keeps serving new requests, its KV cache can fill up until D resumes, so pause P too.
- With `mode="abort"`, and on `POST /abort_requests` with no request ids, P stops keeping the blocks it keeps only for a remote reader (NIXL pull and Mooncake connectors) instead of waiting for the read. On Mooncake, reads that have not started fail at once, and the blocks are freed when the reads in flight end; on NIXL they are freed at once and a read is not revoked. Pause D with `mode="abort"` too, so that D does not read those requests' KV later. Blocks a connector keeps for local work, such as an offloading connector's async save, are still waited for.
- When D aborts a request whose KV is still loading, Mooncake tells P to drop that KV, so the load fails quickly instead of waiting for P (which may be paused).

`clear_cache=True` on a P/D pair needs every block back. With `mode="keep"`, the kept requests do not wait for their KV transfers:

- P stops keeping the blocks of finished prefills for D, as with `mode="abort"`. D's later reads of them fail and are handled by `kv_load_failure_policy`.
- On D, every request whose KV comes from P becomes a local prefill: it stays queued and, after resume, D prefills it with the new weights. The pause waits only for loads already in flight, then discards their KV. On Mooncake, D also tells P to drop the KV of these requests.

`mode="wait"` with `clear_cache=True` waits for the transfers to finish.

## Typical Async RL Flow

A typical async RL loop with weight syncing looks like this:

1. Start generating rollouts from the current policy
2. Once trainer has new weights to update to, pause generation with `mode="keep"`
3. Sync the updated weights from the trainer to the inference engine (see [Weight Transfer](weight_transfer/README.md)) and set their weight version
4. Resume generation -- in-flight requests continue with the new weights
5. Repeat

The key insight is that requests paused with `mode="keep"` will produce tokens from the **old** weights before the pause and tokens from the **new** weights after resume. The `clear_cache` parameter controls whether the KV cache is invalidated during the pause. When `clear_cache=True`, previously cached key-value entries are discarded, so all tokens generated after resume will be computed entirely with the new weights. When `clear_cache=False`, existing KV cache entries are retained, meaning some tokens in context may still reflect the old weights (stale KV cache).

## Weight Versions in Outputs

To tell which tokens came from which weights, set a weight-version label while generation is paused, after the new weights are loaded: pass `weight_version` to `finish_weight_update`, or call `update_weight_version` (`POST /update_weight_version` over HTTP). Labels are strings you choose, such as the trainer step; vLLM doesn't order or check them.

The final output of every sequence then carries `weight_versions`, one span per label over its output-token indexes. A request paused with `mode="keep"` across one update gets two spans:

```python
completion = final_output.outputs[0]
for span in completion.weight_versions:
    # The tokens in [span.start, span.end) were sampled by weights `span.version`.
    print(span.version, completion.token_ids[span.start : span.end])
```

The offsets cover the whole output, so with streamed delta outputs, collect the tokens before slicing. `/inference/v1/generate` returns the same list on each choice; see [Tokens In <> Tokens Out API](../serving/online_serving/token_in_token_out.md#weight-versions).

`/v1/completions` and `/v1/chat/completions` return it on each choice when the request sets `return_weight_versions: true`. The offsets count the choice's output tokens, which `return_token_ids` returns as `token_ids`, and a streamed choice gets the list on its last chunk.

An aborted request returns spans over its partial output. If you resubmit that output as part of a new prompt, the new request's spans start again at 0, so shift them by the number of tokens already generated before joining the two lists.

## Example

The [async RLHF example](../../examples/rl/rlhf_async_new_apis.py) demonstrates this pattern with `vllm.AsyncLLMEngine`, NCCL weight transfer, and mid-flight pause/resume with validation.
