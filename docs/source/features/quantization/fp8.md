(fp8)=

# FP8 W8A8

vLLM supports FP8 (8-bit floating point) weight and activation quantization using hardware acceleration on GPUs such as Nvidia H100 and AMD MI300x.
Currently, only Hopper and Ada Lovelace GPUs are officially supported for W8A8.
Ampere GPUs are supported for W8A16 (weight-only FP8) utilizing Marlin kernels.
Quantization of models with FP8 allows for a 2x reduction in model memory requirements and up to a 1.6x improvement in throughput with minimal impact on accuracy.

Please visit the HF collection of [quantized FP8 checkpoints of popular LLMs ready to use with vLLM](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127).

The FP8 types typically supported in hardware have two distinct representations, each useful in different scenarios:

- **E4M3**: Consists of 1 sign bit, 4 exponent bits, and 3 bits of mantissa. It can store values up to +/-448 and `nan`.
- **E5M2**: Consists of 1 sign bit, 5 exponent bits, and 2 bits of mantissa. It can store values up to +/-57344, +/- `inf`, and `nan`. The tradeoff for the increased dynamic range is lower precision of the stored values.

:::{note}
FP8 computation is supported on NVIDIA GPUs with compute capability > 8.9 (Ada Lovelace, Hopper).
FP8 models will run on compute capability > 8.0 (Ampere) as weight-only W8A16, utilizing FP8 Marlin.
:::

## Installation

To produce performant FP8 quantized models with vLLM, you'll need to install the [llm-compressor](https://github.com/vllm-project/llm-compressor/) library:

```console
pip install llmcompressor
```

## Quantization Process

The quantization process involves three main steps:

1. Loading the model
2. Applying quantization
3. Evaluating accuracy in vLLM

### 1. Loading the Model

Load your model and tokenizer using the standard `transformers` AutoModel classes:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Applying Quantization

For FP8 quantization, we can recover accuracy with simple RTN quantization. We recommend targeting all `Linear` layers using the `FP8_DYNAMIC` scheme, which uses:

- Static, per-channel quantization on the weights
- Dynamic, per-token quantization on the activations

Since simple RTN does not require data for weight quantization and the activations are quantized dynamically, we do not need any calibration data for this quantization flow.

```python
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model: Meta-Llama-3-8B-Instruct-FP8-Dynamic
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

### 3. Evaluating Accuracy

Install `vllm` and `lm-evaluation-harness` for evaluation:

```console
pip install vllm lm-eval==0.4.4
```

Load and run the model in `vllm`:

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
result = model.generate("Hello my name is")
print(result[0].outputs[0].text)
```

Evaluate accuracy with `lm_eval` (for example on 250 samples of `gsm8k`):

:::{note}
Quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.
:::

```console
$ MODEL=$PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic
$ lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250
```

Here's an example of the resulting scores:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
|     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|
```

## Troubleshooting and Support

If you encounter any issues or have feature requests, please open an issue on the [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) GitHub repository.

## Online Dynamic Quantization

Dynamic quantization of an original precision BF16/FP16 model to FP8 can be achieved with vLLM without any calibration data required. You can enable the feature by specifying `--quantization="fp8"` in the command line or setting `quantization="fp8"` in the LLM constructor.

In this mode, all Linear modules (except for the final `lm_head`) have their weights quantized down to FP8_E4M3 precision with a per-tensor scale. Activations have their minimum and maximum values calculated during each forward pass to provide a dynamic per-tensor scale for high accuracy. As a result, latency improvements are limited in this mode.

```python
from vllm import LLM
model = LLM("facebook/opt-125m", quantization="fp8")
# INFO 06-10 17:55:42 model_runner.py:157] Loading model weights took 0.1550 GB
result = model.generate("Hello, my name is")
print(result[0].outputs[0].text)
```

:::{warning}
Currently, we load the model at original precision before quantizing down to 8-bits, so you need enough memory to load the whole model.
:::
