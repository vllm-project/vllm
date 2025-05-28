# Model Resolution

vLLM loads HuggingFace-compatible models by inspecting the `architectures` field in `config.json` of the model repository
and finding the corresponding implementation that is registered to vLLM.
Nevertheless, our model resolution may fail for the following reasons:

- The `config.json` of the model repository lacks the `architectures` field.
- Unofficial repositories refer to a model using alternative names which are not recorded in vLLM.
- The same architecture name is used for multiple models, creating ambiguity as to which model should be loaded.

To fix this, explicitly specify the model architecture by passing `config.json` overrides to the `hf_overrides` option.
For example:

```python
from vllm import LLM

model = LLM(
    model="cerebras/Cerebras-GPT-1.3B",
    hf_overrides={"architectures": ["GPT2LMHeadModel"]},  # GPT-2
)
```

Our [list of supported models][supported-models] shows the model architectures that are recognized by vLLM.
