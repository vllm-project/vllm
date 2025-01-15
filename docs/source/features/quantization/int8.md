(int8)=

# INT8 W8A8

vLLM supports quantizing weights and activations to INT8 for memory savings and inference acceleration.
This quantization method is particularly useful for reducing model size while maintaining good performance.

Please visit the HF collection of [quantized INT8 checkpoints of popular LLMs ready to use with vLLM](https://huggingface.co/collections/neuralmagic/int8-llms-for-vllm-668ec32c049dca0369816415).

```{note}
INT8 computation is supported on NVIDIA GPUs with compute capability > 7.5 (Turing, Ampere, Ada Lovelace, Hopper).
```

## Prerequisites

To use INT8 quantization with vLLM, you'll need to install the [llm-compressor](https://github.com/vllm-project/llm-compressor/) library:

```console
pip install llmcompressor
```

## Quantization Process

The quantization process involves four main steps:

1. Loading the model
2. Preparing calibration data
3. Applying quantization
4. Evaluating accuracy in vLLM

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

### 2. Preparing Calibration Data

When quantizing activations to INT8, you need sample data to estimate the activation scales.
It's best to use calibration data that closely matches your deployment data.
For a general-purpose instruction-tuned model, you can use a dataset like `ultrachat`:

```python
from datasets import load_dataset

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load and preprocess the dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)
```

### 3. Applying Quantization

Now, apply the quantization algorithms:

```python
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

# Configure the quantization algorithms
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save the compressed model
SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-Dynamic-Per-Token"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

This process creates a W8A8 model with weights and activations quantized to 8-bit integers.

### 4. Evaluating Accuracy

After quantization, you can load and run the model in vLLM:

```python
from vllm import LLM
model = LLM("./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token")
```

To evaluate accuracy, you can use `lm_eval`:

```console
$ lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

```{note}
Quantized models can be sensitive to the presence of the `bos` token. Make sure to include the `add_bos_token=True` argument when running evaluations.
```

## Best Practices

- Start with 512 samples for calibration data (increase if accuracy drops)
- Use a sequence length of 2048 as a starting point
- Employ the chat template or instruction template that the model was trained with
- If you've fine-tuned a model, consider using a sample of your training data for calibration

## Troubleshooting and Support

If you encounter any issues or have feature requests, please open an issue on the `vllm-project/llm-compressor` GitHub repository.
