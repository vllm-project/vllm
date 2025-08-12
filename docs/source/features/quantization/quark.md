(quark)=

# AMD QUARK

Quantization can effectively reduce memory and bandwidth usage, accelerate computation and improve
throughput while with minimal accuracy loss. vLLM can leverage [Quark](https://quark.docs.amd.com/latest/),
the flexible and powerful quantization toolkit, to produce performant quantized models to run on AMD GPUs. Quark has specialized support for quantizing large language models with weight,
activation and kv-cache quantization and cutting-edge quantization algorithms like
AWQ, GPTQ, Rotation and SmoothQuant.

## Quark Installation

Before quantizing models, you need to install Quark. The latest release of Quark can be installed with pip:

```console
pip install amd-quark
```

You can refer to [Quark installation guide](https://quark.docs.amd.com/latest/install.html)
for more installation details.

## Quantization Process

After installing Quark, we will use an example to illustrate how to use Quark.  
The Quark quantization process can be listed for 5 steps as below:

1. Load the model
2. Prepare the calibration dataloader
3. Set the quantization configuration
4. Quantize the model and export
5. Evaluation in vLLM

### 1. Load the Model

Quark uses [Transformers](https://huggingface.co/docs/transformers/en/index)
to fetch model and tokenizer.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-2-70b-chat-hf"
MAX_SEQ_LEN = 512

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, model_max_length=MAX_SEQ_LEN)
tokenizer.pad_token = tokenizer.eos_token
```

### 2. Prepare the Calibration Dataloader

Quark uses the [PyTorch Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
to load calibration data. For more details about how to use calibration datasets efficiently, please refer
to [Adding Calibration Datasets](https://quark.docs.amd.com/latest/pytorch/calibration_datasets.html).

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

BATCH_SIZE = 1
NUM_CALIBRATION_DATA = 512

# Load the dataset and get calibration data.
dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
text_data = dataset["text"][:NUM_CALIBRATION_DATA]

tokenized_outputs = tokenizer(text_data, return_tensors="pt",
    padding=True, truncation=True, max_length=MAX_SEQ_LEN)
calib_dataloader = DataLoader(tokenized_outputs['input_ids'],
    batch_size=BATCH_SIZE, drop_last=True)
```

### 3. Set the Quantization Configuration

We need to set the quantization configuration, you can check
[quark config guide](https://quark.docs.amd.com/latest/pytorch/user_guide_config_description.html)
for further details. Here we use FP8 per-tensor quantization on weight, activation,
kv-cache and the quantization algorithm is AutoSmoothQuant.

:::{note}
Note the quantization algorithm needs a JSON config file and the config file is located in
[Quark Pytorch examples](https://quark.docs.amd.com/latest/pytorch/pytorch_examples.html),
under the directory `examples/torch/language_modeling/llm_ptq/models`. For example,
AutoSmoothQuant config file for Llama is
`examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json`.
:::

```python
from quark.torch.quantization import (Config, QuantizationConfig,
                                     FP8E4M3PerTensorSpec,
                                     load_quant_algo_config_from_file)

# Define fp8/per-tensor/static spec.
FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max",
    is_dynamic=False).to_quantization_spec()

# Define global quantization config, input tensors and weight apply FP8_PER_TENSOR_SPEC.
global_quant_config = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
    weight=FP8_PER_TENSOR_SPEC)

# Define quantization config for kv-cache layers, output tensors apply FP8_PER_TENSOR_SPEC.
KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC
kv_cache_layer_names_for_llama = ["*k_proj", "*v_proj"]
kv_cache_quant_config = {name :
    QuantizationConfig(input_tensors=global_quant_config.input_tensors,
                       weight=global_quant_config.weight,
                       output_tensors=KV_CACHE_SPEC)
    for name in kv_cache_layer_names_for_llama}
layer_quant_config = kv_cache_quant_config.copy()

# Define algorithm config by config file.
LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE =
    'examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json'
algo_config = load_quant_algo_config_from_file(LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE)

EXCLUDE_LAYERS = ["lm_head"]
quant_config = Config(
    global_quant_config=global_quant_config,
    layer_quant_config=layer_quant_config,
    kv_cache_quant_config=kv_cache_quant_config,
    exclude=EXCLUDE_LAYERS,
    algo_config=algo_config)
```

### 4. Quantize the Model and Export

Then we can apply the quantization. After quantizing, we need to freeze the
quantized model first before exporting. Note that we need to export model with format of
HuggingFace `safetensors`, you can refer to
[HuggingFace format exporting](https://quark.docs.amd.com/latest/pytorch/export/quark_export_hf.html)
for more exporting format details.

```python
import torch
from quark.torch import ModelQuantizer, ModelExporter
from quark.torch.export import ExporterConfig, JsonExporterConfig

# Apply quantization.
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)

# Freeze quantized model to export.
freezed_model = quantizer.freeze(model)

# Define export config.
LLAMA_KV_CACHE_GROUP = ["*k_proj", "*v_proj"]
export_config = ExporterConfig(json_export_config=JsonExporterConfig())
export_config.json_export_config.kv_cache_group = LLAMA_KV_CACHE_GROUP

EXPORT_DIR = MODEL_ID.split("/")[1] + "-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant"
exporter = ModelExporter(config=export_config, export_dir=EXPORT_DIR)
with torch.no_grad():
    exporter.export_safetensors_model(freezed_model,
        quant_config=quant_config, tokenizer=tokenizer)
```

### 5. Evaluation in vLLM

Now, you can load and run the Quark quantized model directly through the LLM entrypoint:

```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant",
          kv_cache_dtype='fp8',quantization='quark')
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
print("\nGenerated Outputs:\n" + "-" * 60)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt:    {prompt!r}")
    print(f"Output:    {generated_text!r}")
    print("-" * 60)
```

Or, you can use `lm_eval` to evaluate accuracy:

```console
$ lm_eval --model vllm \
  --model_args pretrained=Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant,kv_cache_dtype='fp8',quantization='quark' \
  --tasks gsm8k
```

## Quark Quantization Script
In addition to the example of Python API above, Quark also offers a
[quantization script](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html)
to quantize large language models more conveniently. It supports quantizing models with variety
of different quantization schemes and optimization algorithms. It can export the quantized model
and run evaluation tasks on the fly. With the script, the example above can be:

```console
python3 quantize_quark.py --model_dir meta-llama/Llama-2-70b-chat-hf \
                          --output_dir /path/to/output \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --quant_algo autosmoothquant \
                          --num_calib_data 512 \
                          --model_export hf_format \
                          --tasks gsm8k
```
