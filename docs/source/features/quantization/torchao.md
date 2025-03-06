(torchao)=

# TorchAO

TorchAO is an architecture optimization library for PyTorch, it provides high performance dtypes, optimization techniques and kernels for inference and training, featuring composability with native PyTorch features like torch.compile, FSDP etc.. Some benchmark numbers can be found [here](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks).

We recommend installing the latest torchao nightly with

```console
pip install --pre torchao>=10.0.0 --index-url https://download.pytorch.org/whl/nightly/cu124  # or other cuda versions like cu121

You can quantize your own huggingface model with torchao, e.g. [transformers](https://huggingface.co/docs/transformers/main/en/quantization/torchao) and [diffusers](https://huggingface.co/docs/diffusers/en/quantization/torchao), and save the checkpoint to huggingface hub like [this](https://huggingface.co/jerryzh168/llama3-8b-int8wo) either with the following example code:

```python
import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import Int8WeightOnlyConfig

model_name = "meta-llama/Meta-Llama-3-8B"
quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

hub_repo = # YOUR HUB REPO ID
tokenizer.push_to_hub(hub_repo)
quantized_model.push_to_hub(hub_repo, safe_serialization=False)
```

or with [spaces](https://huggingface.co/spaces/medmekk/TorchAO_Quantization).

We can then test latency for the quantized model with vllm:

```console
python benchmarks/benchmark_latency.py --input-len 256 --output-len 256 --model jerryzh168/llama3-8b-int8wo --batch-size 1 --quantization torchao --compilation-config 3
```
