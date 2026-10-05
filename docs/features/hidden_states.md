# Returning the final prompt hidden state

This document describes native support proposed in
[vLLM PR #57185](https://github.com/vllm-project/vllm/pull/57185). Use a build
containing that implementation. The independently installable
[NuMind extension](https://github.com/numindai/vLLM-Last-Hidden-State) provides
an alternative for stock vLLM 0.30.0, with narrower interfaces and different
runtime support; it is not required by the native feature described here.

A normal generation instance can optionally return the final hidden state at the
last prompt position. It does not need a pooling runner, a KV connector, or
speculative extraction configuration.

Select Model Runner V2 before starting the Python process:

```sh
export VLLM_USE_V2_MODEL_RUNNER=1
```

On CPU, V2 also requires a working Triton CPU backend. A CPU installation that
falls back to Model Runner V1 cannot serve hidden-state extraction requests.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3.5-4B")
normal = SamplingParams(max_tokens=128)
inline = SamplingParams(
    max_tokens=1,
    extra_args={"kv_transfer_params": {"return_last_hidden_state": True}},
)

llm.generate("Tell me a story.", normal)
result = llm.generate("Describe a forest.", inline)[0]
vector = result.kv_transfer_params["last_hidden_state"]
llm.generate("Continue the story.", normal)
```

On the Python and Rust `/v1/chat/completions` and `/v1/completions` endpoints, add
`"kv_transfer_params": {"return_last_hidden_state": true}` with `max_tokens=1`,
`n=1`, and `stream=false`. Ordinary requests retain their usual token budgets and streaming.
Offline batches may mix ordinary and opted-in requests.

For example, with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    messages=[{"role": "user", "content": "Describe a forest."}],
    max_completion_tokens=1,
    n=1,
    stream=False,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "kv_transfer_params": {"return_last_hidden_state": True},
    },
)
vector = response.model_dump()["kv_transfer_params"]["last_hidden_state"]
```

The response's `kv_transfer_params` contains:

- `last_hidden_state`: one list of `hidden_size` finite floats;
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

## Related work and standalone API

Prompt-state representations are studied in
[PromptReps](https://aclanthology.org/2024.emnlp-main.250/) and
[E5-V](https://arxiv.org/abs/2407.12580). This API exposes the model state; it
neither trains an embedding model nor guarantees retrieval quality.

The standalone package uses the same `return_last_hidden_state` flag and
`last_hidden_state` vector field. It exposes Python chat on Qwen3.5 with one
local worker and returns no position/layer metadata. Its MTP support does not
remove the native feature's speculation restriction. Keep runtime validation
for the two implementations separate.
