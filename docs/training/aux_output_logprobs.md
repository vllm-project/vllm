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

The same configuration is available on the command line:

```bash
vllm serve facebook/opt-125m \
    --enable-prefix-caching \
    --aux-output-config '{"enable_return_logprobs": true}'
```

The `aux_output_enabled` request flag opts that request into the auxiliary
path. It requires `logprobs` to be set and rejects prompt logprobs, full
vocabulary requests (`logprobs=-1`), non-raw `logprobs_mode`, and online weight
updates.

Enabling `enable_return_logprobs` requires Model Runner V2
(`VLLM_USE_V2_MODEL_RUNNER=1` when vLLM would otherwise fall back to V1) and
prefix caching.

This is separate from [prompt token ID logprobs](prompt_token_id_logprobs.md),
which scores caller-provided candidate IDs during prompt processing.
