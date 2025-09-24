# FP8 INC

vLLM supports FP8 (8-bit floating point) weight and activation quantization using Intel® Neural Compressor (INC) on Intel® Gaudi® 2 and Intel® Gaudi® 3 AI accelerators.
Currently, quantization is validated only in Llama models.

Intel Gaudi supports quantization of various modules and functions, including, but not limited to `Linear`, `KVCache`, `Matmul` and `Softmax`. For more information, please refer to:
[Supported Modules\\Supported Functions\\Custom Patched Modules](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#supported-modules).

!!! note
    Measurement files are required to run quantized models with vLLM on Gaudi accelerators. The FP8 model calibration procedure is described in the [vLLM HPU extension](https://github.com/HabanaAI/vllm-hpu-extension/tree/main/calibration/README.md) package.

!!! note
    `QUANT_CONFIG` is an environment variable that points to the measurement or quantization [JSON config file](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#supported-json-config-file-options).
    The measurement configuration file is used during the calibration procedure to collect measurements for a given model. The quantization configuration is used during inference.

## Run Online Inference Using FP8

Once you've completed the model calibration process and collected the measurements, you can run FP8 inference with vLLM using the following command:

```bash
export QUANT_CONFIG=/path/to/quant/config/inc/meta-llama-3.1-405b-instruct/maxabs_measure_g3.json
vllm serve meta-llama/Llama-3.1-405B-Instruct --quantization inc --kv-cache-dtype fp8_inc --tensor_paralel_size 8
```

!!! tip
    If you are just prototyping or testing your model with FP8, you can use the `VLLM_SKIP_WARMUP=true` environment variable to disable the warmup stage, which can take a long time. However, we do not recommend disabling this feature in production environments as it causes a significant performance drop.

!!! tip
    When using FP8 models, you may experience timeouts caused by the long compilation time of FP8 operations. To mitigate this problem, you can use the below environment variables:
    `VLLM_ENGINE_ITERATION_TIMEOUT_S` - to adjust the vLLM server timeout. You can set the value in seconds, e.g., 600 equals 10 minutes.
    `VLLM_RPC_TIMEOUT` - to adjust the RPC protocol timeout used by the OpenAI-compatible API. This value is in microseconds, e.g., 600000 equals 10 minutes.

## Run Offline Inference Using FP8

To run offline inference (after completing the model calibration process):

* Set the "QUANT_CONFIG" environment variable to point to a JSON configuration file with QUANTIZE mode.
* Pass `quantization=inc` and `kv_cache_dtype=fp8_inc` as parameters to the `LLM` object.
* Call shutdown method of the model_executor at the end of the run.

```python
from vllm import LLM
llm = LLM("llama3.1/Meta-Llama-3.1-8B-Instruct", quantization="inc", kv_cache_dtype="fp8_inc")
...
# Call llm.generate on the required prompts and sampling params.
...
llm.llm_engine.model_executor.shutdown()
```

## Device for the Model's Weights Uploading

The unquantized weights are first loaded onto the CPU, then quantized and transferred to the target device (HPU) for model execution.
This reduces the device memory footprint of model weights, as only quantized weights are stored in the device memory.
