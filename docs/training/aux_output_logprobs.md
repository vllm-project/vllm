# Auxiliary Logprobs for RL

The AuxOutput connector can retain generated-token logprobs by KV block and
replay them for requests that opt in. This is useful when an RL rollout needs
the inference-side sampling logprobs while reusing a locally cached prefix.
The returned values keep the normal `logprobs` response shape; prompt logprobs
are not part of this feature.

Enable the connector on the engine:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    enable_prefix_caching=True,
    aux_output_config={"enable_return_logprobs": True},
)
outputs = llm.generate(
    ["The prompt"],
    SamplingParams(
        max_tokens=8,
        logprobs=5,
        extra_args={"aux_output_enabled": True},
    ),
)
token_logprobs = outputs[0].outputs[0].logprobs
```

The `aux_output_enabled` request flag opts that request into the auxiliary
path. It requires `logprobs` to be set and rejects prompt logprobs, full
vocabulary requests (`logprobs=-1`), non-raw `logprobs_mode`, and online weight
updates. The engine must use Model Runner V2 and prefix caching. Logprobs-only
configuration supports dense and MoE models.

This is separate from [prompt token ID logprobs](prompt_token_id_logprobs.md),
which scores caller-provided candidate IDs during prompt processing.
