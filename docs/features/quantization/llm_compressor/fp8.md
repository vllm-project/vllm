# FP8 W8A8

vLLM supports FP8 (8-bit floating point) weight and activation quantization using hardware acceleration on GPUs such as Nvidia H100 and AMD MI300x.
Ada Lovelace, Hopper, and Blackwell GPUs are supported for W8A8.
Turing/Ampere GPUs are supported for W8A16 (weight-only FP8) utilizing Marlin kernels.
Quantization of models with FP8 allows for a 2x reduction in model memory requirements and up to a 1.6x improvement in throughput with minimal impact on accuracy.

Please visit the HF collection of [quantized FP8 checkpoints of popular LLMs ready to use with vLLM](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127).

The FP8 types typically supported in hardware have two distinct representations, each useful in different scenarios:

- **E4M3**: Consists of 1 sign bit, 4 exponent bits, and 3 bits of mantissa. It can store values up to +/-448 and `nan`.
- **E5M2**: Consists of 1 sign bit, 5 exponent bits, and 2 bits of mantissa. It can store values up to +/-57344, +/- `inf`, and `nan`. The tradeoff for the increased dynamic range is lower precision of the stored values.

!!! note
    FP8 computation is supported on NVIDIA GPUs with compute capability >= 8.9 (Ada Lovelace, Hopper, Blackwell).
    FP8 models will run on compute capability >= 7.5 (Turing) as weight-only W8A16, utilizing FP8 Marlin.

!!! note "GEMM kernel selection"
    vLLM picks an FP8 GEMM kernel automatically at load time and logs a
    `Selected <kernel> for <module>` line at startup. For block-quantized
    checkpoints on CUDA it tries, in order: a FlashInfer/DeepGEMM hybrid
    (Hopper only), DeepGEMM, CUTLASS, Marlin, Triton, Humming, then a
    PyTorch fallback. GPUs without native FP8 support (e.g. Turing/Ampere)
    land on weight-only (W8A16) Marlin.

    If inference hangs with no error, try `VLLM_USE_DEEP_GEMM=0` or
    `--linear-backend cutlass`.

    `--linear-backend` only affects quantized linear layers; MoE experts
    use the separate `--moe-backend`. An explicit backend that is not
    supported on your hardware raises an error rather than falling back.
    The full list is documented under
    [`KernelConfig`](../../../cli/serve.md#kernelconfig) in the CLI
    reference and shown by `vllm serve --help=KernelConfig`.

## Native Triton per-token FP8 on ROCm

On ROCm GPUs with native FP8 support, `--linear-backend triton` selects the
native Triton backend for dynamic per-token activation scales and per-channel
weight scales. It supports BF16 and FP16 output. With the default
`--linear-backend auto`, this backend is considered after higher-priority
compatible ROCm backends and before Torch RowWise.

```bash
vllm serve RedHatAI/gemma-4-12B-it-FP8-Dynamic \
  --language-model-only --max-model-len 32768 \
  --max-num-batched-tokens 8192 --linear-backend triton
```

Tuning files are optional. FP8-capable RDNA GPUs use a hardware-aware launch
heuristic when no tuned configuration applies. The Radeon AI PRO R9700 BF16
tuning tables apply measured configurations only at their exact token counts;
intermediate counts and entries without a reliable measured gain retain the
heuristic. Here, token count means the rows processed by the current linear
operation, including any graph padding, rather than the full context length.

### Reproducing per-token FP8 tuning

`benchmarks/kernels/benchmark_w8a8_per_token_fp8.py` tunes per-token FP8 GEMMs
with BF16 output on FP8-capable RDNA GPUs. Select a model workload from
`benchmarks/kernels/weight_shapes.py`, its tensor-parallel size, and the GPUs
to use as parallel tuning workers. The script uses matrix dimensions and TP
split information only; model configuration and weights are not loaded.
Expert kernels are separate.

```bash
uv run --no-project --python .venv/bin/python .venv/bin/python \
  benchmarks/kernels/benchmark_w8a8_per_token_fp8.py \
  --model RedHatAI/gemma-4-12B-it-FP8-Dynamic \
  --tp-size 1 --gpus 0 1 --save-path ./fp8-configs
```

`--tp-size` (alias `-tp`) describes the model's tensor parallelism and defaults
to 1. It is independent of the number of tuning workers. Standard catalog entries
divide the recorded TP axis; layouts requiring other partitioning use explicit
per-TP matrix shapes from `TP_WEIGHT_SHAPES`.
`--gpus` defaults to all visible GPUs; selected GPUs must be the same model.

To cover TP2 as well, run the same command with `--tp-size 2`:

```bash
uv run --no-project --python .venv/bin/python .venv/bin/python \
  benchmarks/kernels/benchmark_w8a8_per_token_fp8.py \
  --model RedHatAI/gemma-4-12B-it-FP8-Dynamic \
  --tp-size 2 --gpus 0 1 --save-path ./fp8-configs
```

This example has six unique shapes at each TP size, with eleven shapes in their
union. The benchmark also accepts the existing Llama, Mistral, Qwen, and other
matrix workloads in the shared catalog. Add new matrix workloads to that file.

Staged search is the default: 232 base tile/warp/stage choices plus installed
configurations and the heuristic tile, refinement of the best four choices,
then seven shuffled paired rounds for eight finalists against the installed
config and production heuristic. A winner must reduce median latency by more
than 2% and 0.15 microseconds and win at least five rounds. Selected results pass
full Torch and sampled FP32 checks on two input seeds. Token counts are fixed
at powers of two from 1 through 32768.
Shapes outside the specialized heuristic's layout requirements use the same
shared-kernel fallback as serving. Only the Torch reference is zero-padded when
its alignment requirements demand it; timed operands retain their actual shapes.

`--save-path` writes runtime-compatible JSON tables directly into that directory,
with full rankings, paired timings, matrix workloads, reference tables, and
source hashes under `measurements/`. Each worker owns complete shape tables,
so parallel workers do not overwrite each other's results. Omit `--save-path`
to print results without writing tuning output files. Review generated tables
before copying them into `vllm/model_executor/layers/quantization/utils/configs`.

The search uses hot operands and captured graphs. It excludes activation
quantization, attention, and TP communication; fresh winners can vary with
measurement conditions. Validate candidates against the existing tables and
heuristic with representative weight working sets before replacing profiles.
The reported tables also underwent independent runtime validation after the
search. These commands repeat the search procedure, not whole-model serving
measurements or byte-identical historical results.

## Installation

To produce performant FP8 quantized models with vLLM, you'll need to install the [llm-compressor](https://github.com/vllm-project/llm-compressor/) library:

```bash
(venv-llm-compressor) pip install llmcompressor
```

Additionally, install `vllm` and `lm-evaluation-harness` for evaluation:

```bash
(venv-vllm) pip install vllm "lm-eval[api]>=0.4.12"
```

Please use separate environments for vLLM and llm-compressor as they might not work together.

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
    MODEL_ID,
    device_map="auto",
    dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Applying Quantization

For FP8 quantization, we can recover accuracy with simple RTN quantization. We recommend targeting all `Linear` layers using the `FP8_DYNAMIC` scheme, which uses:

- Static, per-channel quantization on the weights
- Dynamic, per-token quantization on the activations

Since simple RTN does not require data for weight quantization and the activations are quantized dynamically, we do not need any calibration data for this quantization flow.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)

# Save the model: Meta-Llama-3-8B-Instruct-FP8-Dynamic
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

### 3. Evaluating Accuracy

Load and run the model in `vllm`:

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
result = llm.generate("Hello my name is")
print(result[0].outputs[0].text)
```

Evaluate accuracy with `lm_eval` (for example on 250 samples of `gsm8k`):

!!! note
    Quantized models can be sensitive to the presence of the `bos` token. `lm_eval` does not add a `bos` token by default, so make sure to include the `add_bos_token=True` argument when running your evaluations.

```bash
MODEL=$PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250
```

Here's an example of the resulting scores:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
| --- |------:| -------------- |-----:| --------- | - |----:| - |-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
|     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|
```

## Troubleshooting and Support

If you encounter any issues or have feature requests, please open an issue on the [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) GitHub repository.

## Online Dynamic Quantization

Dynamic quantization of an original precision BF16/FP16 model to FP8 can be achieved with vLLM without any calibration data required. You can enable the feature by specifying `--quantization="fp8_per_tensor"` in the command line or setting `quantization="fp8_per_tensor"` in the LLM constructor.

In this mode, all Linear modules (except for the final `lm_head`) have their weights quantized down to FP8_E4M3 precision with a per-tensor scale. Activations have their minimum and maximum values calculated during each forward pass to provide a dynamic per-tensor scale for high accuracy. As a result, latency improvements are limited in this mode.

```python
from vllm import LLM

llm = LLM("facebook/opt-125m", quantization="fp8_per_tensor")
# INFO 06-10 17:55:42 model_runner.py:157] Loading model weights took 0.1550 GB
result = llm.generate("Hello, my name is")
print(result[0].outputs[0].text)
```
