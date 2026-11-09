# Nemotron Elastic serving

Serve two widths of a BF16 Nemotron Elastic checkpoint from one weight
allocation. The default request uses `full`. The dimensions come from the
checkpoint configuration, including any `--hf-overrides`:

- `full`: top-level `hidden_size` and `moe_intermediate_size`.
- `sliced`: `elastic_hidden_size` and `elastic_moe_intermediate_size` inside
  `elastic_thinking`.

Set `elastic_phase` on each request:

| Value | Behavior |
| --- | --- |
| `full` (default) | Use the full model for the entire request. |
| `sliced` | Use the smaller slice for the entire request. |
| `auto` | Start sliced, then switch to full after generating the checkpoint's `elastic_think_end_token_id` (`</think>`). |

Omitting `elastic_phase` always uses `full`. A marker in the prompt or chat history
does not trigger a switch. If an `auto` request never generates the marker, it
stays sliced. Requests with different profiles can be active together; the
scheduler alternates between their batches.

This is an experimental implementation for BF16 Nemotron-H checkpoints
trained for width slicing. It does not turn an ordinary Nemotron checkpoint into
an Elastic model. Single-GPU and tensor-parallel serving are supported.

## Start the server

Install this branch using vLLM's [source installation instructions](../getting_started/installation/gpu.md).
The public [NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16)
needs an explicit configuration overlay to enable runtime slicing. Its published
configuration does not contain the `elastic_*` fields consumed by this implementation.
The overlay below uses the 12B slice dimensions from the model's public
[model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16/blob/4bcc4326801e1ef6b8ec1c281edd69cb2e041964/README.md)
and [slicing script](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16/blob/4bcc4326801e1ef6b8ec1c281edd69cb2e041964/zero_shot_slicing.py).
The published tokenizer maps token 13 to `</think>`.

```bash
vllm serve nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16 \
  --revision 4bcc4326801e1ef6b8ec1c281edd69cb2e041964 \
  --hf-overrides '{"elastic_use_lazy_slicing":true,"elastic_thinking":{"elastic_hidden_size":1920,"elastic_moe_intermediate_size":960},"elastic_think_end_token_id":13}' \
  --served-model-name nemotron-elastic \
  --dtype bfloat16 --tensor-parallel-size 2 \
  --max-model-len 8192 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.88 \
  --no-async-scheduling --moe-backend triton \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[1,2,4,8]}' \
  --enable-prefix-caching --enable-prompt-tokens-details
```

The full weights, cache and runtime buffers must fit across the selected GPUs.
Use the dimensions and marker provided by your checkpoint when serving another
Elastic model; ordinary Nemotron checkpoints are not trained for width slicing.
The full model's attention, Mamba and expert dimensions must support the chosen
tensor parallel size. For debugging, replace `--compilation-config` with
`--enforce-eager`.
For checkpoints with `use_bias: true`, Mamba `n_groups` must be divisible by the
tensor parallel size; the native replicated-group input-bias loader is not
supported.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-elastic",
    "messages": [{"role":"user","content":"What is 2 plus 3?"}],
    "elastic_phase": "auto",
    "temperature": 0,
    "max_tokens": 256
  }'
```

`elastic_phase` also works on `/v1/completions`. For the Python `LLM` API, use
`SamplingParams(extra_args={"elastic_phase": "sliced"})`. The API field takes
precedence over an `elastic_phase` value in `vllm_xargs`. Automatic switching
requires `elastic_phase: auto`; there is no configurable alternate default.
Only `full`, `sliced`, and `auto` are accepted. Invalid values, or using the setting
on a non-Elastic model, are rejected.
Beam search and resumable input sessions are not supported. Streaming responses
are supported.

## What is shared

The `sliced` profile reads slices of the full weights. Attention head counts
and Mamba state dimensions stay fixed, so a request can change width without
reallocating its cache. Decode CUDA graphs are captured separately for each width.
Prefill runs eagerly.

Tensor parallelism preserves the same global prefix of the checkpoint's expert
weights. Each rank uses the part of that prefix in its full-weight shard, which
can be empty. The reduced expert output includes all ranks' contributions, and
each rank retains its existing cache allocation when the request changes width.

Prefix caching uses one namespace for both profiles. A `full` request can therefore
reuse state created by a `sliced` request, and vice versa. This is deliberate mixed-
width inference: the reused state is **not numerically equivalent** to recomputing
that prefix entirely at the destination width. Validate answer quality on your
workload. Use `--no-enable-prefix-caching` to disable reuse between requests; the
state within an `auto` request is still shared across its width change.
After preemption, missing tokens are recomputed with the request's current slice.
An `auto` request that has switched to `full` stays on `full`, including during a
full cache rebuild. It does not replay its earlier `sliced` phase. Recomputed
state can therefore differ from the state before preemption.

The implementation currently requires Model Runner V2, synchronous scheduling,
and Triton MoE. Pipeline/data/expert parallelism, quantization, LoRA, speculative
decoding and disaggregated serving are outside its supported scope. The full
checkpoint must fit across the tensor-parallel ranks even when every request
chooses `sliced`.
Auxiliary outputs (`--enable-return-routed-experts`) are not supported with Elastic,
including requests that use `full`.

## Verify the implementation

Run the CPU tests with `.venv/bin/python -m pytest tests/v1/core/test_elastic.py`. The small GPU tests
create a tiny hybrid checkpoint and compare logits with separately materialized
Transformers models. They cover both widths, the transition, cross-request reuse,
snapshot isolation, cancellation and recovery after preemption, in eager mode
and with decode graphs. Run the single-GPU cases with:

```bash
.venv/bin/python -m pytest tests/models/language/generation/test_nemotron_elastic.py -m 'hybrid_model and not distributed' -v
```

On two GPUs, run the tensor-parallel cases, including partial and empty expert
shards, with:

```bash
.venv/bin/python -m pytest tests/models/language/generation/test_nemotron_elastic.py -m 'distributed(num_gpus=2)' -v
```

The serving smoke test needs the real checkpoint and enough GPU memory for the
full weights, cache and runtime buffers. For example, on two GPUs:

```bash
ELASTIC_TEST_TP_SIZE=2 \
VLLM_GPU_SYNC_CHECK=error \
.venv/bin/python -m pytest tests/models/language/generation/test_nemotron_elastic.py -k switching_and_shared_cache -v
```

It checks the generated-marker transition, fixed modes, concurrent requests and
cache reuse in eager mode and with decode graphs. Set `ELASTIC_TEST_TP_SIZE=1`
when the full checkpoint and runtime allocations fit on one GPU.
The test uses the public checkpoint and overlay above by default. Set
`ELASTIC_TEST_MODEL` to use another checkpoint with its own `elastic_*` metadata. The tiny-checkpoint tests run
in hybrid-model and distributed-model CI; the real-checkpoint smoke test is
manual. These checks do not
establish task-level quality or a throughput improvement.
