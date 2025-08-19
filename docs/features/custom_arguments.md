# vLLM Custom Arguments

You can use vLLM *custom arguments* to enable [custom logits processors](./custom_logitsprocs.md) and vLLM plugins to receive request arguments which are not hard-coded into vLLM's interface.

Custom arguments passed to `SamplingParams.extra_args` as a `dict` will be visible to any code which has access to `SamplingParams`:

``` python
SamplingParams(...,
               extra_args={"your_custom_arg_name": 67})
```

This allows arguments which are not already part of `SamplingParams` to be passed into vLLM.

The vLLM REST API allows custom arguments to be passed to the vLLM server via `vllm_xargs`; under the hood `vllm_xargs` is transferred directly into `SamplingParams.extra_args`. The example below integrates custom arguments into a vLLM REST API request:

``` bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0,
        "vllm_xargs": {"your_custom_arg": 67}
    }'
```

Furthermore, OpenAI SDK users can access `vllm_xargs` via the `extra_body` argument:

``` python
batch = await client.completions.create(
    model=model_name,
    prompt=prompt,
    extra_body={
        "vllm_xargs": {
            "your_custom_arg": 67
        }
    }
)
```
