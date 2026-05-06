# Routed Experts Replay

## Overview

Routed experts replay captures which MoE (Mixture of Experts) experts process each token during inference and returns this information alongside the generated text. This is essential for **reinforcement learning (RL) training pipelines** (such as GRPO and RLHF) where the training step needs to reconstruct expert routing decisions from the inference pass.

When enabled, each API response includes:

- **`prompt_routed_experts`**: A `[prompt_len, num_moe_layers, top_k]` array of expert IDs for the prompt tokens (at the response level, shared across completions).
- **`routed_experts`**: A `[gen_len, num_moe_layers, top_k]` array of expert IDs for the generated tokens (per completion).

For example, a model with 40 MoE layers and top-22 routing that processes a 100-token prompt and generates 50 tokens would return:

- `prompt_routed_experts`: shape `[100, 40, 22]`
- `routed_experts`: shape `[50, 40, 22]`

Each value is an int16 expert ID in the range `[0, num_experts)`.

## Quickstart

### OpenAI API Server

```bash
vllm serve <MODEL> \
    --enable-return-routed-experts \
    --tensor-parallel-size 4 \
    --enable-expert-parallel
```

Then query the `/v1/completions` endpoint as usual. The response includes routing data:

```python
import requests

resp = requests.post("http://localhost:8000/v1/completions", json={
    "model": "<MODEL>",
    "prompt": "Explain quantum computing.",
    "max_tokens": 64,
    "temperature": 0.0,
}).json()

# Generation routing (per completion choice)
gen_routing = resp["choices"][0]["routed_experts"]  # [gen_len, layers, top_k]

# Prompt routing (shared across all choices)
prompt_routing = resp["prompt_routed_experts"]      # [prompt_len, layers, top_k]

print(f"Prompt routing shape: [{len(prompt_routing)}, "
      f"{len(prompt_routing[0])}, {len(prompt_routing[0][0])}]")
print(f"Gen routing shape: [{len(gen_routing)}, "
      f"{len(gen_routing[0])}, {len(gen_routing[0][0])}]")
```

### Python SDK (Offline Inference)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="<MODEL>",
    enable_return_routed_experts=True,
    tensor_parallel_size=4,
    enable_expert_parallel=True,
)

outputs = llm.generate(
    ["Explain quantum computing."],
    SamplingParams(temperature=0, max_tokens=64),
)

result = outputs[0]

# Prompt routing: numpy array, shape [prompt_len, num_moe_layers, top_k]
prompt_routing = result.prompt_routed_experts
print(f"Prompt routing: {prompt_routing.shape}, dtype={prompt_routing.dtype}")

# Generation routing: numpy array, shape [gen_len, num_moe_layers, top_k]
gen_routing = result.outputs[0].routed_experts
print(f"Gen routing: {gen_routing.shape}, dtype={gen_routing.dtype}")
```

## Output Format

### `CompletionOutput.routed_experts`

- **Type**: `numpy.ndarray` (Python SDK) or `list[list[list[int]]]` (JSON API)
- **Shape**: `[gen_len, num_moe_layers, top_k]`
- **Dtype**: `int16`
- **Content**: Expert IDs for **generated tokens only**. `gen_len` matches the number of generated tokens (i.e., `usage.completion_tokens` or fewer).

### `RequestOutput.prompt_routed_experts`

- **Type**: `numpy.ndarray` (Python SDK) or `list[list[list[int]]]` (JSON API)
- **Shape**: `[prompt_len, num_moe_layers, top_k]`
- **Dtype**: `int16`
- **Content**: Expert IDs for **prompt tokens only**. `prompt_len` matches `usage.prompt_tokens`. This field lives on the request-level response (not per-choice), because prompt routing is shared across all completions when `n > 1`.

### Why Separate Prompt and Generation Routing?

When a request has multiple completions (`n > 1`), each completion shares the same prompt but produces different generated text. Storing prompt routing once on the `RequestOutput` (rather than duplicating it on every `CompletionOutput`) avoids redundant data. For RL training, the consumer typically needs:

1. The prompt routing (once) to reconstruct the forward pass for the shared prefix.
2. The per-completion generation routing to reconstruct each completion's forward pass.

## Architecture

### Data Flow

```text
Forward Pass                    Async D2H Pipeline              Output
─────────────                   ──────────────────              ──────
FusedMoE layer                  After forward pass:             On request finish:
writes topk_ids    ──────►     D2H copy to pinned     ──────►  Extract from host cache
to device buffer               staging buffer                   Split at prompt_len
(L, N, K) int16                 (via CUDA stream)                Trim gen to output len
                                Scatter to per-request           Serialize to API response
                                host cache (numpy)
```

### Device Cache

A pre-allocated GPU buffer with layout `(L, N, K)` where:

- `L` = number of MoE layers
- `N` = `max_num_batched_tokens`
- `K` = `num_experts_per_tok` (top-k)

The `(L, N, K)` layout ensures that `buffer[layer_id]` gives a contiguous `(N, K)` view per layer. Each `FusedMoE` layer gets a persistent reference to its slice via `module._routing_replay_out = buffer[layer_id]`.

**Dtype**: `int16` — sufficient for expert IDs (max ~512 experts in practice) and half the memory of `int32`.

### Host Cache

Per-request numpy arrays for accumulating routing data across decode steps. Each request gets a lazily allocated `(seq_len, L, K)` int16 buffer that grows as the sequence lengthens. Buffers are freed when a request completes.

### Async D2H Pipeline

After each forward pass, the model runner issues a non-blocking device-to-host copy on a dedicated CUDA stream:

1. **Copy**: `pinned_staging[:, :total_tokens, :].copy_(device_buffer[:, :total_tokens, :])` on a separate stream, recorded with a CUDA event.
2. **Scatter** (deferred to next step): On the *next* forward pass, synchronize the event (effectively free — an entire forward pass has elapsed) and scatter the staging data into per-request host cache buffers using the token positions.

This design ensures the D2H copy overlaps with the next forward pass, minimizing GPU stall time.

### CUDA Graph Compatibility

CUDA graph compatibility requires two mechanisms:

1. **Persistent tensor attribute**: Each `FusedMoE` layer stores a reference to its buffer slice as `module._routing_replay_out`. Because `torch.compile` captures module attributes by reference, graph replay always writes to the live buffer — not a stale snapshot.

2. **Static marking**: Both the full `(L, N, K)` buffer and each per-layer `(N, K)` view are marked with `cudagraph_mark_tensor_static()`. This prevents CUDA graphs from snapshot/restore behavior that would zero the buffer on replay.

### Multi-Node Support

On multi-node tensor-parallel setups, all TP ranks allocate a device buffer (required for symmetric CUDA graph structure), but only TP rank 0 runs the D2H pipeline and host cache. Routing data flows from the model runner through `ModelRunnerOutput` via Ray DAG to the scheduler — no shared memory or file locks needed.

### Routing Capture Path

For the **non-monolithic (Triton) kernel path** (e.g., BF16 MoE), routing is captured after `select_experts()` in the MoE runner:

```python
routing_replay_out = getattr(layer, "_routing_replay_out", None)
topk_weights, topk_ids = self.router.select_experts(...)

if routing_replay_out is not None:
    routing_replay_out[:topk_ids.shape[0]].copy_(topk_ids.to(torch.int16))
```

For the **monolithic kernel path** (e.g., FP8/MXFP8 via FlashInfer), `routing_replay_out` is threaded through the `apply_monolithic()` call chain and FlashInfer writes expert IDs directly during routing inside the fused kernel.

### MTP (Multi-Token Prediction) Handling

With MTP speculative decoding, the model captures routing for all tokens including speculative ones that may later be rejected. When a request finishes, the generation routing is trimmed to match the actual number of accepted output tokens:

```python
num_gen = self.detokenizer.num_output_tokens()
if gen_routed_experts.shape[0] > num_gen and num_gen > 0:
    gen_routed_experts = gen_routed_experts[:num_gen]
```

This ensures the routing array length always matches the token IDs in the response.

## Design Decisions

### Why Replace SharedMemory with Device Cache?

The previous implementation used `multiprocessing.SharedMemory` with `fcntl` file locking to transfer routing data from GPU workers to the scheduler. This approach had fundamental problems:

- **Multi-node**: `SharedMemory` is node-local. On multi-node TP setups (required for 400B+ parameter models), the scheduler on node 0 cannot read shared memory from workers on other nodes.
- **Performance**: Synchronous `.cpu().numpy()` D2H transfers block the GPU. File-based locking adds further overhead.
- **CUDA graphs**: The callback-based capture mechanism bakes tensor references at trace time, causing stale data on graph replay.

The device cache approach solves all three: data flows through Ray DAG (works multi-node), D2H is async (non-blocking), and persistent tensor attributes work with CUDA graphs.

### Why `(L, N, K)` Layout Instead of `(N, L, K)`?

FlashInfer's `routing_replay_out` parameter expects a contiguous `(N, K)` tensor per layer. With `(L, N, K)` layout, `buffer[layer_id]` gives a contiguous `(N, K)` view with zero-copy slicing. The previous `(N, L, K)` layout would require non-contiguous indexing or an explicit copy.

### Why int16 Instead of int32?

Expert IDs are small integers (typically 0-255 for models with up to 256 experts). `int16` supports up to 32,767 experts — far more than any current model — while halving GPU memory usage and D2H bandwidth compared to `int32`.

### Why Split Prompt and Generation Routing?

RL training pipelines process prompt and generation routing separately:

- Prompt routing reconstructs the shared forward pass for the input.
- Generation routing reconstructs each sampled trajectory.

With `n > 1` completions, all completions share the same prompt routing. Duplicating it per completion would waste memory proportional to `n * prompt_len * L * K`. Instead, `prompt_routed_experts` is stored once on `RequestOutput` and shared.

### Why Async D2H Instead of Synchronous Copy?

A synchronous `.cpu()` call forces the GPU to drain its command queue before the copy can begin, stalling the pipeline. The async approach:

1. Issues the copy on a separate CUDA stream (non-blocking to the main compute stream).
2. Defers the host-side scatter to the *next* step, by which time the copy has finished.

This means the D2H transfer overlaps entirely with the next forward pass, adding near-zero latency to the critical path.

### Why All TP Ranks Get a Device Buffer?

CUDA graph capture records the exact sequence of kernel calls and their arguments. If only rank 0 had a device buffer, the `FusedMoE` layer would take a different code path on rank 0 vs. other ranks (one writes to a buffer, others don't). This asymmetry causes different CUDA graph structures across ranks, which can lead to NCCL deadlocks during collective operations inside the graph. Giving all ranks a real buffer ensures symmetric graph structure. Only rank 0 does the D2H copy and host cache management.

## Performance

Routing replay adds a small overhead from the device buffer writes and async D2H copies. On tested configurations:

- **Throughput overhead** (random data, ISL=1024, OSL=1024): **~2%**
- **Memory overhead** (int16 buffer, 40 layers, 8192 tokens, top-22): **~14 MB per GPU**
- **Accuracy impact** (GSM8K): **Zero** (pass@1 identical with and without routing replay)

The overhead is dominated by the per-layer `.copy_()` during the forward pass. The async D2H pipeline runs entirely in the background.

## Supported Configurations

| Configuration                            | Supported                                                 |
|------------------------------------------|-----------------------------------------------------------|
| BF16 Triton MoE (non-monolithic)         | Yes                                                       |
| FP8/MXFP8 FlashInfer MoE (monolithic)    | Yes (requires FlashInfer with `routing_replay_out`)       |
| CUDA graphs                              | Yes                                                       |
| Multi-node tensor parallelism            | Yes                                                       |
| Data parallelism (DP)                    | Yes                                                       |
| Expert parallelism (EP)                  | Yes                                                       |
| Prefix caching                           | Yes (cached positions marked with `-1` sentinel)          |
| MTP speculative decoding                 | Yes (gen routing trimmed to accepted tokens)              |
| `n > 1` (multiple completions)           | Yes (prompt routing shared, gen routing per-completion)   |

## Limitations

- **Streaming**: Routing data is only available when the request finishes (not streamed incrementally).
- **V1 engine only**: Routing replay is implemented for the vLLM V1 engine.
- **Preempted requests**: When a request is preempted by the scheduler (and later resumed via re-prefill), any routing already accumulated in the worker's host cache for that request is dropped without being emitted. The consumer sees `routed_experts=None` for the resumed request with no other signal. Partial-rollout and async-RL pipelines that rely on routing for preempted requests should either disable preemption (`--no-enable-chunked-prefill` / sufficient KV headroom) or reconstruct routing on the resumed prefill.
- **Async scheduling**: Not supported; rejected at config time. The worker-side stop predicate reads `req_state.output_token_ids[-1]`, which under async scheduling is the placeholder `-1` until `AsyncGPUModelRunnerOutput` resolves the real sampled token, so EOS / stop-token finishes would silently drop routing. Use sync scheduling (the default when `--enable-return-routed-experts` is set, or set explicitly with the appropriate scheduler config).
- **Sequence parallelism / naive DP MoE dispatch**: Not supported on the FusedMoE layer; rejected at bind time. SP shards `topk_ids` along dim 0 across the TP group so each rank only captures `1/sp_size` of the rows; naive DP dispatch all-gathers tokens across DP ranks before routing, so `topk_ids.shape[0]` exceeds the per-rank buffer size. Both raise `NotImplementedError` from `bind_routing_capture_to_model`.
- **Pipeline / prefill-context / decode-context parallelism**: Not yet validated; rejected at config time.

## CLI Reference

| Flag                               | Description                                                            |
|------------------------------------|------------------------------------------------------------------------|
| `--enable-return-routed-experts`   | Enable routing replay capture and return expert IDs in API responses.  |

## API Reference

### Completions (`/v1/completions`)

**Response-level field:**

| Field                     | Type                                | Description                                                                 |
|---------------------------|-------------------------------------|-----------------------------------------------------------------------------|
| `prompt_routed_experts`   | `list[list[list[int]]]` or `null`   | Expert IDs for prompt tokens. Shape: `[prompt_len, num_moe_layers, top_k]`. |

**Choice-level field:**

| Field              | Type                                | Description                                                                   |
|--------------------|-------------------------------------|-------------------------------------------------------------------------------|
| `routed_experts`   | `list[list[list[int]]]` or `null`   | Expert IDs for generated tokens. Shape: `[gen_len, num_moe_layers, top_k]`.   |

### Chat Completions (`/v1/chat/completions`)

Same fields as above on `ChatCompletionResponse` and `ChatCompletionResponseChoice`.

### Python SDK

| Object               | Field                     | Type                     | Description                 |
|----------------------|---------------------------|--------------------------|-----------------------------|
| `RequestOutput`      | `prompt_routed_experts`   | `np.ndarray` or `None`   | `[prompt_len, L, K]` i16    |
| `CompletionOutput`   | `routed_experts`          | `np.ndarray` or `None`   | `[gen_len, L, K]` int16     |
