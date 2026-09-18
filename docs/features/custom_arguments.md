# Custom Arguments

You can use vLLM *custom arguments* to pass in arguments which are not part of the vLLM `SamplingParams` and REST API specifications. Adding or removing a vLLM custom argument does not require recompiling vLLM, since the custom arguments are passed in as a dictionary.

Custom arguments can be useful if, for example, you want to use a [custom logits processor](./custom_logitsprocs.md) without modifying the vLLM source code.

!!! note
    Make sure your custom logits processor have implemented `validate_params` for custom arguments. Otherwise, invalid custom arguments can cause unexpected behaviour.

## Offline Custom Arguments

Custom arguments passed to `SamplingParams.extra_args` as a `dict` will be visible to any code which has access to `SamplingParams`:

``` python
SamplingParams(extra_args={"your_custom_arg_name": 67})
```

This allows arguments which are not already part of `SamplingParams` to be passed into `LLM` as part of a request.

## Online Custom Arguments

The OpenAI-compatible REST API and Anthropic-compatible `/v1/messages` endpoint allow custom arguments to be passed to the vLLM server via `vllm_xargs`. The example below integrates custom arguments into a vLLM REST API request:

``` bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        ...
        "vllm_xargs": {"your_custom_arg": 67}
    }'
```

Furthermore, OpenAI SDK users can access `vllm_xargs` via the `extra_body` argument:

``` python
batch = await client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    ...,
    extra_body={
        "vllm_xargs": {
            "your_custom_arg": 67
        }
    }
)
```

!!! note
    `vllm_xargs` is assigned to `SamplingParams.extra_args` under the hood, so code which uses `SamplingParams.extra_args` is compatible with both offline and online scenarios.

## Built-in no-repeat n-gram argument

The GPU model runner recognizes `no_repeat_ngram_size` without requiring a
custom logits processor. A positive integer prevents the generated output from
repeating an n-gram of that size; an omitted value disables the feature. Prompt
tokens are not included in the n-gram history. `window_size` limits how many
recent output tokens are searched, and `whitelist_token_ids` exempts tokens
which must remain available, such as OCR grounding markers. The `ngram_size`
alias is accepted for compatibility with existing OCR model integrations. The
legacy alias preserves the existing OCR processor's `ngram_size=1` no-op;
canonical `no_repeat_ngram_size=1` applies the standard unigram constraint.

``` python
SamplingParams(
    extra_args={
        "no_repeat_ngram_size": 100,
        "window_size": 100,
        "whitelist_token_ids": [128821, 128822],
    }
)
```

The equivalent online request uses:

``` json
{
    "vllm_xargs": {
        "no_repeat_ngram_size": 100,
        "window_size": 100,
        "whitelist_token_ids": [128821, 128822]
    }
}
```

This native path applies the constraint to the whole persistent batch on the
GPU. It avoids the per-request Python processing and model-runner fallback that
would result from loading a custom logits processor. Speculative decoding is
not currently supported with this argument.
