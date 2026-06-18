# Orthrus

Orthrus is a block-diffusion speculative decoding method for Orthrus/Qwen3
checkpoints. It reuses the loaded target model as the drafter and runs a
separate diffusion attention path for the speculative block, so no separate
draft model repository is required.

Use `"method": "orthrus"` in `speculative_config` and set
`num_speculative_tokens` to the number of tokens to draft per step:

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="chiennv/Orthrus-Qwen3-1.7B",
    speculative_config={
        "method": "orthrus",
        "num_speculative_tokens": 3,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

For online serving, pass the same configuration through
`--speculative-config`:

```bash
vllm serve chiennv/Orthrus-Qwen3-1.7B \
    --speculative-config '{"method": "orthrus", "num_speculative_tokens": 3}'
```

Orthrus uses parallel drafting internally. The scheduler reserves KV lookahead
slots for the diffusion block, but the Orthrus drafter batch is separate from
the target verification batch and does not append extra compute slots to the
target batch.

!!! note
    Orthrus support is intended for checkpoints whose architecture maps to
    `OrthrusForCausalLM` or `OrthrusLM`. Generic Qwen3 checkpoints should use
    another speculative decoding method unless they include the Orthrus
    diffusion weights expected by the model implementation.
