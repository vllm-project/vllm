# Bidirectional Attention Spans

Some decoder models are trained with parts of the prompt attending to each other in both directions, for example decision models that read a state before a question about it. vLLM serves such models with bidirectional spans: per-request token ranges whose tokens see every other token of the same range, on the layers the model applies them to. Everything else stays causal.

Spans work with prefix caching, chunked prefill and tensor parallelism. A checkpoint opts in through its config, and needs no deployment flags.

## Semantics

A request carries `bidirectional_spans`, a list of half-open `[start, end)` token ranges over its prompt. The ranges are sorted and disjoint, with `0 <= start < end <= len(prompt_token_ids)`. One-token spans attend exactly as causal attention does, so vLLM drops them, and equivalent span lists are served and cached identically.

On a layer that applies spans, query token `i` attends to key token `j` if `j <= i`, or if `i` and `j` lie in the same span. On other layers, and for tokens outside every span, the model's usual mask holds. Generated tokens are never in a span. Layers that apply spans take no multimodal PrefixLM image ranges: an image inside a span is already bidirectional there.

## Which models support spans

A model supports spans when its implementation does (it implements [SupportsBidirectionalSpans][vllm.model_executor.models.interfaces.SupportsBidirectionalSpans]) and the checkpoint declares them in its text config:

```json
{
  "bidirectional_span_layer_types": ["full_attention"],
  "bidirectional_span_max_length": 1024
}
```

- `bidirectional_span_layer_types` lists the layer types that apply spans. Only `full_attention` layers can apply spans for now.
- `bidirectional_span_max_length` is the longest span the checkpoint was trained with. Longer spans are refused.

Gemma 4 (`Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration` and the unified variant) implements spans on its full-attention layers. vLLM refuses to start a checkpoint that declares spans on an architecture that does not implement them.

A checkpoint that declares spans is never served causally by accident: every request must set `bidirectional_spans`, and an empty list states a causal request. Requests without the field are refused, so requests that cannot set it, such as text prompts and chat conversations, cannot be served, and the structured decisions endpoints are off for such a checkpoint.

## Offline inference

Pass the spans with a token-id prompt, to `LLM.generate` or to the pooling methods (`LLM.embed`, `LLM.classify`, `LLM.encode`):

```python
from vllm import LLM, SamplingParams, TokensPrompt

llm = LLM("path/to/checkpoint")
prompt = TokensPrompt(
    prompt_token_ids=token_ids,
    bidirectional_spans=[(0, 1024), (1024, 1536)],
)
outputs = llm.generate(prompt, SamplingParams(max_tokens=1))
```

Spans index prompt tokens. A `TokensPrompt` with spans therefore cannot also carry multimodal data or be truncated, since both move token positions. Text prompts and prompt embeddings cannot carry spans.

[examples/features/bidirectional_spans/bidirectional_spans_offline.py](../../examples/features/bidirectional_spans/bidirectional_spans_offline.py) serves a prompt and a grown version of it, and shows the grown prompt reusing the cached spans.

## Online serving

The Completions API (`/v1/completions`) and the completion-style pooling APIs (`/classify`, `/pooling`, `/v1/embeddings`) take `bidirectional_spans` with token-id prompts. For a list of token-id prompts, give one list of spans per prompt:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "path/to/checkpoint",
        "prompt": [2, 105, 2364, 107, 818, 6524],
        "bidirectional_spans": [[0, 4]],
        "max_tokens": 1
    }'
```

Chat requests refuse the field, since a chat client cannot know the token positions of the rendered prompt. On a checkpoint that declares spans, a completion without spans and every Chat Completions, Responses or Messages request get a 400 before any stream starts. Other span errors found by the engine arrive as an error event of the stream.

These do not take spans, so a checkpoint that declares them cannot serve them:

- the Chat Completions, Responses and Messages APIs,
- scoring and reranking,
- `/v1/completions/render`, which refuses the field, and `/inference/v1/generate`, whose requests carry no spans,
- `echo` with `logprobs`, which asks for prompt logprobs,
- embeddings with chunked processing, which refuse the field,
- the Rust frontend, which refuses the field and whose requests the engine then refuses for lack of it.

## Spans over multimodal prompts

Placeholder expansion decides where an image's tokens are, so spans over a multimodal prompt are set on the rendered engine input, after expansion, by the code that renders the prompt:

```python
from vllm.renderers.inputs.preprocess import parse_model_prompt

prompt = {"prompt": text, "multi_modal_data": {"image": image}}
(engine_input,) = llm.renderer.render_cmpl(
    [parse_model_prompt(llm.model_config, prompt)]
)
num_tokens = len(engine_input["prompt_token_ids"])
engine_input["bidirectional_spans"] = [(0, state_end), (state_end, num_tokens)]
llm.llm_engine.add_request("request-0", engine_input, SamplingParams(max_tokens=1))
outputs = []
while llm.llm_engine.has_unfinished_requests():
    outputs.extend(output for output in llm.llm_engine.step() if output.finished)
```

Every multimodal item must lie inside one span. Caller-chosen media IDs (`multi_modal_uuids`) are refused with spans, because the prefix-cache keys of a span identify its items by their content hash.

## Prefix caching

A token in a span depends on every token of its span, including the tokens after it. vLLM therefore adds each span's bounds and a digest of its tokens and multimodal items to the hash of every KV block the span overlaps. A block is reused only by a request with the same tokens, items and span up to the span's end, so a request never reads KV computed with other span bounds or other tokens in its span. In particular:

- A repeated prompt, or a new question after the same spans, reuses all of their blocks.
- A prompt that grows (a state that gains a step) reuses the full blocks of every span that ends before the new tokens, and recomputes the span they land in. Splitting a long prefix into fixed-size spans, for example of 1024 tokens, bounds the recomputation per step.
- Two prompts with the same tokens but different spans share no blocks from the first span on.

Reuse is per full KV block. With a speculative decoding method whose block hashes cover lookahead tokens, a block also depends on the first tokens after it, so the block before a span that starts at a block boundary, and the last block of a span followed by other tokens, are not reused.

External KV connectors must load KV that is valid under a request's spans. vLLM refuses to start a checkpoint that declares spans with a connector that does not declare it (`KVConnectorBase_V1.supports_bidirectional_spans`). The offloading connectors that `--kv-offloading-size` selects with the native backend look blocks up by these hashes and are supported. Disaggregated prefill connectors are not.

## Scheduling

All of a span's uncomputed tokens run in one step, because each of them attends to the whole span. The scheduler never ends a prefill chunk inside a span: a chunk that would end inside one ends at its start, including when the encoder budget or cache cannot take an image of the span this step, and a request that has entered a span (after a prefix-cache hit that ends there) waits for a step with budget to finish it. A waiting request that cannot start its span keeps its place at the head of the queue, including under `--cache-aware-admission-window`, so that later requests do not keep taking the budget or encoder cache space it needs. A running request whose span waits for encoder cache space stops new admissions for the step. A running request that waits for the token budget does not, since later requests are scheduled after it. With spans close to the step budget, a waiting span request can hold the queue until the running batch leaves it enough budget. Chunked prefill stays on for the tokens outside spans.

Each span must fit one step: at most `--max-num-batched-tokens` tokens, less the slots a speculative decoding method reserves, and at most `--max-num-scheduled-tokens` and `--long-prefill-token-threshold` when they are set. Longer spans are refused, and so are spans whose multimodal items do not fit the encoder cache (`--encoder-cache-size`) together.

## Limitations

vLLM refuses to start a checkpoint that declares spans with:

- context parallelism,
- `--kv-sharing-fast-prefill`,
- a `cudagraph_mode` that captures prefill steps in full CUDA graphs (`FULL`),
- micro-batching (`--enable-dbo`, `--ubatch-size` above 1),
- multi-module MTP speculative decoding,
- a KV connector that does not declare span support, which includes disaggregated prefill,
- an attention backend that does not apply PrefixLM ranges, such as `CPU_ATTN`.

Spans are refused on requests with streaming input or prompt logprobs. Sliding-window layers and hybrid (Mamba) models do not apply spans yet.

Layers that apply spans run on a backend that applies PrefixLM ranges. When that is the composite `TRITON_FLASH_ATTN` backend, as on H100, it does not support full CUDA graphs with speculative decoding, so vLLM falls back to piecewise CUDA graphs (`cudagraph_mode=PIECEWISE`) and logs a warning at startup.
