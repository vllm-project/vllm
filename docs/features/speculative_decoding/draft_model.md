# Draft Models

The following code configures vLLM in an offline mode to use speculative decoding with a draft model, speculating 5 tokens at a time.

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
        "method": "draft_model",
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

To perform the equivalent launch in online mode, use the following server-side code:

```bash
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --seed 42 \
    -tp 1 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.8 \
    --speculative-config '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 5, "method": "draft_model"}'
```

The code used to request completions as a client remains unchanged:

??? code

    ```python
    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # Completion API
    stream = False
    completion = client.completions.create(
        model=model,
        prompt="The future of AI is",
        echo=False,
        n=1,
        stream=stream,
    )

    print("Completion results:")
    if stream:
        for c in completion:
            print(c)
    else:
        print(completion)
    ```

## FLy verification

[FLy](https://arxiv.org/abs/2511.22972) is an approximate verification policy
that can override a native rejection at an ambiguous position when the following
native acceptance decisions remain aligned. It is intentionally lossy: unlike
standard speculative decoding, it does not preserve the target distribution.

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "draft_model",
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 8,
        "rejection_sample_method": "fly",
        "fly_window_size": 6,
        "fly_entropy_threshold": 0.3,
    },
)
```

`fly_window_size` is the number of subsequent tokens checked and must be
smaller than `num_speculative_tokens`. The entropy gate uses the top three
probabilities after target-side temperature, top-k, and top-p processing. FLy
supports greedy requests, target-only acceptance with the default greedy draft
sampling, and probability-ratio acceptance with
`draft_sample_method="probabilistic"`.

Besides `draft_model`, FLy accepts the hidden-state drafting methods (`eagle`,
`eagle3`, `mtp`, `dflash`, `dspark`) and their parallel drafting. It requires a
GPU, a shared target/draft vocabulary, and the V1 model runner: configurations
that default to the V2 model runner fall back to V1 when FLy is enabled, and
configurations that require V2 are rejected.

## Draft Model Method with heterogeneous vocabs

  By default, vLLM requires the draft and target models to share the same vocabulary. Setting `use_heterogeneous_vocab: true` enables the **Token-Level Intersection (TLI)** algorithm, which allows draft models from a different model family with a different tokenizer.
  
  Currently, `use_heterogeneous_vocab` requires `draft_sample_method='greedy'` (the default). Probabilistic draft sampling is not yet supported and will be added in a
  future release.

  ```python
  from vllm import LLM, SamplingParams

  llm = LLM(
      model="Qwen/Qwen3-8B",
      speculative_config={                               
          "method": "draft_model",
          "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
          "num_speculative_tokens": 3,
          "use_heterogeneous_vocab": True,
      },
      gpu_memory_utilization=0.5,
  )
outputs = llm.generate(prompts，sampling_params)

for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! warning
    Note: Please use `--speculative-config` to set all configurations related
    to speculative decoding. The previous method of specifying the model
    through `--speculative-model` and adding related parameters such as
    `--num-speculative-tokens` separately has been deprecated. For supported
    keys and examples, see the [`--speculative-config` schema](README.md#--speculative-config-schema).
