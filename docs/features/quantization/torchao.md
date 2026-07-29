# TorchAO

TorchAO is an architecture optimization library for PyTorch, it provides high performance dtypes, optimization techniques and kernels for inference and training, featuring composability with native PyTorch features like torch.compile, FSDP etc.. Some benchmark numbers can be found [here](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks).

We recommend installing the latest torchao nightly with

```bash
# Install the latest TorchAO nightly build
# Choose the CUDA version that matches your system (cu126, cu128, etc.)
pip install \
    --pre torchao>=10.0.0 \
    --index-url https://download.pytorch.org/whl/nightly/cu126
```

## Quantizing HuggingFace Models

You can quantize your own huggingface model with torchao, e.g. [transformers](https://huggingface.co/docs/transformers/main/en/quantization/torchao) and [diffusers](https://huggingface.co/docs/diffusers/en/quantization/torchao), and save the checkpoint to huggingface hub like [this](https://huggingface.co/jerryzh168/llama3-8b-int8wo) with the following example code:

??? code

    ```Python
    import torch
    from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
    from torchao.quantization import Int8WeightOnlyConfig

    model_name = "meta-llama/Meta-Llama-3-8B"
    quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_text = "What are we having for dinner?"
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    hub_repo = # YOUR HUB REPO ID
    tokenizer.push_to_hub(hub_repo)
    quantized_model.push_to_hub(hub_repo, safe_serialization=False)
    ```

Alternatively, you can use the [TorchAO Quantization space](https://huggingface.co/spaces/medmekk/TorchAO_Quantization) for quantizing models with a simple UI.

## Quantizing at Load Time

TorchAO can also quantize an FP16/BF16 checkpoint while vLLM loads it. For
example, the following uses the packed INT4 weight-only kernel:

```python
import json

from torchao.core.config import config_to_dict
from torchao.quantization import Int4WeightOnlyConfig
from vllm import LLM

config = Int4WeightOnlyConfig(
    group_size=128,
    int4_packing_format="tile_packed_to_4d",
)
llm = LLM(
    "RWKV/RWKV7-Goose-World2.8-1.5B-HF",
    dtype="bfloat16",
    quantization="torchao",
    hf_overrides={
        "quantization_config_dict_json": json.dumps(config_to_dict(config))
    },
)
```

For models with an untied output embedding, TorchAO also quantizes the
`ParallelLMHead`; leaving a large LM head unquantized can otherwise dominate
both memory use and decode time.
