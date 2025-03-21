(quark)=

# QUARK

Quantization can effectively reduce memory and bandwidth usage, accelerate computation and improve throughput while with minimal accuracy loss.  
vLLM can leverage [Quark](https://quark.docs.amd.com/latest/), the flexible and powerful quantization toolkit, to produce performant quantized models to run on GPUs of AMD or Nvidia.   
Quark has specialized support for quantizing large language models with weight, activation and kv-cache quantization  
and along with the cutting-edge quantization algorithms like SmoothQuant, AWQ, and GPTQ.

## Quark Installation

Before quantizing model, you need to install Quark. Please refer to [Quark installation guide](https://quark.docs.amd.com/latest/install.html).

## Quantization Process

After installing Quark, we will use an example to illustrate how to use Quark.  
The Quark quantization process can be listed for 5 steps as below:

1. Loading the model
2. Preparing calibration dataloader
3. Set the quantization configuration
4. Quantize the model and export
5. Evaluating accuracy in vLLM

### 1. Loading the Model

Quark use [Transformers](https://huggingface.co/docs/transformers/en/index) to fetch model and tokenizer.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-2-70b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
    model_max_length=seq_len, padding_side="left")
```

### 2. Preparing Calibration Dataloader

Calibration is a crucial step in post-training quantization.  
Usually, you need calibration to ensures that the reduced-precision representation (e.g., INT8 or FP8) maintains model accuracy.  
Quark use the [PyTorch Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) to load calibration data.
For more details, please refer to [Adding Calibration Datasets](https://quark.docs.amd.com/latest/pytorch/calibration_datasets.html).

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

BATCH_SIZE = 1
NUM_CALIBRATION_DATA = 512
MAX_SEQUENCE_LENGTH = 512

dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
text_data = dataset["text"][:NUM_CALIBRATION_DATA]
tokenized_outputs = tokenizer(text_data, return_tensors="pt",
    padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH)
calib_dataloader = DataLoader(tokenized_outputs['input_ids'],
    batch_size=BATCH_SIZE, drop_last=True)
```

### 3. Set the Quantization Configuration

We need to set the quantization config, you can check [config user guide](https://quark.docs.amd.com/latest/pytorch/user_guide_config_description.html) for further details.  
Here we use FP8 per-tensor quantization on weight, activate, kv-cache and the quantization algorithm is autosmoothquant.

```python
from quark.torch.quantization import Config, QuantizationConfig, 
                                     FP8E4M3PerTensorSpec,
                                     load_quant_algo_config_from_file

FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max",
    is_dynamic=False).to_quantization_spec()                                          
global_quant_config = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
    weight=FP8_PER_TENSOR_SPEC)
KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC
kv_cache_layer_names_for_llama = ["*k_proj", "*v_proj"]
kv_cache_quant_config = {name :
    QuantizationConfig(input_tensors=global_quant_config.input_tensors,
                       weight=global_quant_config.weight,
                       output_tensors=KV_CACHE_SPEC)
    for name in kv_cache_layer_names_for_llama}
layer_quant_config = kv_cache_quant_config.copy()
LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE = './llama_autosmoothquant_config.json'
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

Then we can apply the quantization. After quantizing, we need freeze the quantized model first before exporting.  
Here we export model with format of HuggingFace `safetensors`.
You can refer to [HuggingFace format exporting](https://quark.docs.amd.com/latest/pytorch/export/quark_export_hf.html) for more details. 

```python
from quark.torch import ModelQuantizer, ModelExporter
from quark.torch.export import ExporterConfig, JsonExporterConfig

quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)

freezed_model = quantizer.freeze(model)

LLAMA_KV_CACHE_GROUP = ["*k_proj", "*v_proj"]
export_config = ExporterConfig(json_export_config=JsonExporterConfig())
export_config.json_export_config.kv_cache_group = LLAMA_KV_CACHE_GROUP
EXPORT_DIR = MODEL_ID.split("/")[1] + "-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant"
exporter = ModelExporter(config=export_config, export_dir=EXPORT_DIR)
with torch.no_grad():
    exporter.export_safetensors_model(freezed_model,
        quant_config=quant_config, tokenizer=tokenizer, custom_mode='fp8')
```

### 5. Evaluating Accuracy

Now, you can load and run the quantized model directly through the LLM entrypoint:

```python
from vllm import LLM
llm = LLM(model="Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant",
          quantization='quark')
```

Or, you can use `lm_eval` to evaluate accuracy:

```console
$ lm_eval --model vllm \
  --model_args pretrained="model=Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant, \
                           quantization=quark" \
  --tasks gsm8k,wikitext \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

## Use Quark Example Quantization Script
In addition to the example of Python API above, Quark also has [example quantization script](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html) to quantize language models in a more convenient way.  
So, the example above can be:
```console
python3 quantize_quark.py --model_dir /path/to/Llama-2-7b-chat-hf  \
                          --output_dir /path/to/output \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --custom_mode fp8 \
                          --quant_algo autosmoothquant \
                          --num_calib_data 512 \
                          --model_export hf_format \
                          --skip_evaluation
```
Also, you can use this script to quantize different models with different quantization schemes and different optimization algorithms.  
Here is an example of Llama-3.1-8B-Instruct quantization, weight is int8 per-channel, activation is int8 per-tensor,  
kv-cache is int8 per-tensor and use advanced algorithms
combination of rotation and smoothquant.  
It will run gsm8k and wikitext evaluation tasks directly to evaluate the quantized model.

```console
python3 quantize_quark.py --model_dir /path/to/Llama-3.1-8B-Instruct \
                          --output_dir /path/to/output \
                          --quant_scheme w_int8_per_channel_a_int8_per_tensor_sym \
                          --kv_cache_dtype int8_per_tensor_static \
                          --pre_quantization_optimization rotation \
                          --pre_quantization_optimization smoothquant \
                          --num_calib_data 512 \
                          --model_export hf_format \
                          --tasks gsm8k,wikitext
```
