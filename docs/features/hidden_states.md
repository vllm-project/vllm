# Returning the final prompt hidden state

A normal generation instance can optionally return the final hidden state at the
last prompt position. It does not need a pooling runner, a KV connector, or
speculative extraction configuration.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3.5-4B")
normal = SamplingParams(max_tokens=128)
inline = SamplingParams(
    max_tokens=1,
    extra_args={"kv_transfer_params": {"return_inline": True}},
)

llm.generate("Tell me a story.", normal)
result = llm.generate("Describe a forest.", inline)[0]
vector = result.kv_transfer_params["hidden_states"]
llm.generate("Continue the story.", normal)
```

On the Python and Rust `/v1/chat/completions` and `/v1/completions` endpoints, add
`"kv_transfer_params": {"return_inline": true}` with `max_tokens=1`, `n=1`, and
`stream=false`. Ordinary requests retain their usual token budgets and streaming.
Offline batches may mix ordinary and opted-in requests.

The response's `kv_transfer_params` contains:

- `hidden_states`: one list of `hidden_size` finite floats;
- `token_position`: the zero-based last position of the processed prompt;
- `layer_id`: the text decoder's number of layers;
- `representation`: `"post_final_norm"`.

This is the model output used to compute the first next-token logits, after the
decoder's final normalization. It is **not** the state of the newly generated
token. Chat templates, assistant-generation boundaries, thinking settings, and
image processing determine the prompt positions. No L2 normalization is applied.

The initial implementation accepts native Qwen2 and Qwen3.5 models on CPU, CUDA,
and ROCm using Model Runner V2, with pipeline and context parallel sizes of one.
Opted-in requests are rejected when the engine selects Model Runner V1, including
automatic fallbacks, or when speculative decoding or a KV connector is configured. Streaming,
multiple HTTP prompts, `n>1`, beam search, resumable input, and the Responses API
are unsupported for extraction. Rust checks engine support using the startup
handshake; engines that do not advertise support reject inline requests.

Only opted-in, completed prefill rows are copied. Model Runner V2 includes the copy
in its existing token-output synchronization; the scheduler returns the vector
with that step's token and releases the request normally. There is no additional
hidden-state cache, file export, or deferred connector-output lifecycle. Missing,
wrong-sized, or nonfinite vectors fail the request.

The payload and extra copy are O(hidden_size) per opted-in request. Serialization
and transfer still cost time, including for ordinary requests sharing its batch.
