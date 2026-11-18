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

## Kernels

On CUDA, int4 weights in tinygemm layout (`Int4WeightOnlyConfig(int4_packing_format="tile_packed_to_4d")`, as in [`pytorch/Qwen3-8B-INT4`](https://huggingface.co/pytorch/Qwen3-8B-INT4)) run on [Humming](https://github.com/vllm-project/humming) instead of tinygemm, which torchao tunes for batch size 1. Other torchao tensors run on torchao's own kernels.
