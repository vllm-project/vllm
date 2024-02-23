# FP8 KV Cache Extraction Tool

This utility extracts the KV cache scaling factors from a quantized HF (Hugging Face) model. The extracted scaling factors are saved to a JSON file, which can later be used by vLLM (variable-length language model) during runtime. This tool is particularly useful when the KV cache data type is FP8 and is intended for use on ROCm (AMD GPU) platforms.

## Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Hugging Face Transformers
- Hugging Face Hub
- AMMO 

Before using this tool, you'll need to follow these steps:

1. Install all necessary prerequisites and dependencies. 
2. Convert HF model into a quantized HF model. 
3. Extract KV Cache Scaling Factors from quantized HF model.
4. Load KV Cache Scaling Factors into VLLM.

### 2. Convert HF model into a quantized HF model.
Note: The following steps are adapted from the [TensorRT-LLM repository](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md).
## APIs

[`quantize.py`](3rdparty/quantizer/quantize.py) uses the quantization toolkit  (AMMO) to calibrate the PyTorch models and export TensorRT-LLM checkpoints. Each TensorRT-LLM checkpoint contains a config file (in .json format) and one or several rank weight files (in .safetensors format). See this [`doc`](../../docs/source/new_workflow.md) for more details on the TensorRT-LLM checkpoint format.

The detailed quantization toolkit (AMMO) conversion guide for FP8 can be found [here](https://github.com/ROCm/vllm-fp8/blob/fp8_kv/3rdparty/README.md).

### 3. Extract KV Cache Scaling Factors from quantized HF model.
extract_scales.py can be utilized to extract the KV cache scaling factors from your quantized HF model, however at the moment, this tool exclusively supports LLaMa models. It is also important to note the following:
1. **File Structure**: The utility operates under the assumption that all parameters, including KV cache scaling factors, corresponding to a particular Tensor Parallelism (TP) rank are stored in a single file. These files must adhere to a specific naming convention where the TP rank is immediately identified after a specific keyword (e.g., "rank") in the filename.

2. **TP Decomposition**: The utility assumes consistency between the TP decomposition employed by the quantizer tool and that used by vLLM.

3. **AMMO Compatibility**: Currently, the generated KV cache scaling factors for AMMO remain uniform across all TP ranks.

```python
# prerequisites:
# - Quantized HF LLaMa model 
python3 3rdparty/quantizer/extract_scales.py --help
Useage: extract_scales.py [-h] --quantized_model QUANTIZED_MODEL [--cache_dir CACHE_DIR] [--load_format {auto,safetensors,npz,pt}] [--revision REVISION] [--output_dir OUTPUT_DIR] [--output_name OUTPUT_NAME] [--tp_size TP_SIZE]

KV Scale Extraction Example

optional arguments:
--quantized_model: Specify either the local path to, or name of, a quantized HF model. It is expected that the quantization format is FP8_E4M3, for use on ROCm (AMD GPU).
Optional arguments:

--cache_dir: Specify a cache directory to use in the event of a HF model download. (Default: None)
--load_format: Specify the format of the model's tensor files containing the KV cache scaling factors. (Choices: auto, safetensors, npz, pt; Default: auto)
--revision: Specify the model's revision number. (Default: None)
--output_dir: Specify the output directory. By default the KV cache scaling factors will be saved in the model directory. (Default: None)
--output_name: Specify the output filename. (Default: kv_cache_scales.json)
--tp_size: Specify the tensor-parallel (TP) size that the quantized model should correspond to. If specified, during KV cache scaling factor extraction the observed TP size will be checked against this and an error will be raised if there is a mismatch. (Default: None)

Example:
python3 3rdparty/quantizer/extract_scales.py --model <QUANTIZED_MODEL_DIR> --tp_size --output_dir <PATH_TO_OUTPUT_DIR>

### 4. Load KV Cache Scaling Factors into VLLM.
# prerequisites:
# -  LLaMa kv_cache_scales.json file

python3 benchmarks/benchmark_throughput.py --help 
usage: benchmark_throughput.py [-h] [--backend {vllm,hf,mii}] [--dataset DATASET] [--input-len INPUT_LEN] [--output-len OUTPUT_LEN] [--model MODEL]
                               [--tokenizer TOKENIZER] [--quantization {awq,gptq,squeezellm,None}] [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--n N]
                               [--use-beam-search] [--num-prompts NUM_PROMPTS] [--seed SEED] [--hf-max-batch-size HF_MAX_BATCH_SIZE] [--trust-remote-code]
                               [--max-model-len MAX_MODEL_LEN] [--dtype {auto,half,float16,bfloat16,float,float32}] [--enforce-eager] [--kv-cache-dtype {auto,fp8}]
                               [--kv-cache-scales-path KV_CACHE_SCALES_PATH]

Benchmark Throughput Example  
optional arguments:
  -h, --help            show this help message and exit
  --backend {vllm,hf,mii}
  --dataset DATASET     Path to the dataset.
  --input-len INPUT_LEN
                        Input prompt length for each request
  --output-len OUTPUT_LEN
                        Output length for each request. Overrides the output length from the dataset.
  --model MODEL
  --tokenizer TOKENIZER
  --quantization {awq,gptq,squeezellm,None}, -q {awq,gptq,squeezellm,None}
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
  --n N                 Number of generated sequences per prompt.
  --use-beam-search
  --num-prompts NUM_PROMPTS
                        Number of prompts to process.
  --seed SEED
  --hf-max-batch-size HF_MAX_BATCH_SIZE
                        Maximum batch size for HF backend.
  --trust-remote-code   trust remote code from huggingface
  --max-model-len MAX_MODEL_LEN
                        Maximum length of a sequence (including prompt and output). If None, will be derived from the model.
  --dtype {auto,half,float16,bfloat16,float,float32}
                        data type for model weights and activations. The "auto" option will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16
                        models.
  --enforce-eager       enforce eager execution
  --kv-cache-dtype {auto,fp8}
                        Data type for kv cache storage. If "auto", will use model data type. FP8_E5M2 (without scaling) is only supported on cuda version greater than
                        11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for common inference criteria.
  --kv-cache-scales-path KV_CACHE_SCALES_PATH
                        Path to the JSON files containing the KV cache scaling factors. This should generally be supplied, when KV cache dtype is FP8. Otherwise, KV cache
                        scaling factors default to 1.0, which may cause accuracy issues. FP8_E5M2 (without scaling) is only supported on cuda version greater than 11.8.
                        On ROCm (AMD GPU), FP8_E4M3 is instead supported for common inference criteria.
Example:
python3 benchmarks/benchmark_throughput.py --input-len <n> --output-len <n> -tp <n> --kv-cache-dtype fp8 --kv-cache-scales-path </path/to/kv_cache_scales.json>
