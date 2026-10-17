# INT8 W4A8

vLLM supports quantizing weights to INT4 and activations to INT8 for memory savings and inference acceleration.
This quantization method is particularly useful for reducing model size while maintaining good performance.

## Prerequisites

To use INT8 W4A8 quantization with vLLM, you'll need to install the [llm-compressor](https://github.com/vllm-project/llm-compressor/) library.

```bash
(venv-llm-compressor) pip install llmcompressor
```

Additionally, install `vllm` and `lm-evaluation-harness` for evaluation:

```bash
(venv-vllm) pip install vllm "lm-eval[api]>=0.4.12"
```

Please use separate environments for vLLM and llm-compressor as they might not work together.

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
    MODEL_ID,
    dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. Preparing Calibration Data

When quantizing activations to INT8 and weights to INT4, you need sample data to estimate the activation scales.
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

Now, apply the quantization algorithms.

The following recipes create W4A8 models (int4 weights, int8 activations). On Arm® CPUs, this is accelerated through [KleidiAI](https://github.com/ARM-software/kleidiai).

Use groupwise for best accuracy, and channelwise for best inference performance.

=== "Groupwise"

    ```python
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    # Configure the quantization algorithms
    recipe = [
        GPTQModifier(
            targets="Linear",
            scheme="W4A8",
            ignore=["lm_head"],
            dampening_frac=0.01
        ),
    ]

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Save the compressed model: Meta-Llama-3-8B-Instruct-W4A8-G128-Dynamic-Per-Token
    SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A8-G128-Dynamic-Per-Token"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

=== "Channelwise"

    ```python
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from compressed_tensors.quantization import QuantizationStrategy, QuantizationType

    scheme = {
        "targets": ["Linear"],
        "weights": {
            "num_bits": 4,
            "type": QuantizationType.INT,
            "strategy": QuantizationStrategy.CHANNEL,
            "symmetric": True,
            "dynamic": False,
            "group_size": None,
        },
        "input_activations": {
            "num_bits": 8,
            "type": QuantizationType.INT,
            "strategy": QuantizationStrategy.TOKEN,
            "dynamic": True,
            "symmetric": False,
            "observer": None,
        },
        "output_activations": None,
    }

    recipe = [
        GPTQModifier(
            targets="Linear",
            config_groups={"group_0": scheme},
            ignore=["lm_head"],
            dampening_frac=0.01,
        ),
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Save the compressed model: Meta-Llama-3-8B-Instruct-W4A8-Channelwise-Dynamic-Per-Token
    SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A8-Channelwise-Dynamic-Per-Token"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

### 4. Evaluating Accuracy

=== "Groupwise"

    After quantization, you can load and run the model in vLLM:

    ```python
    from vllm import LLM

    llm = LLM("./Meta-Llama-3-8B-Instruct-W4A8-G128-Dynamic-Per-Token")
    ```

    To evaluate accuracy, you can use `lm_eval`:

    ```bash
    lm_eval --model vllm \
        --model_args pretrained="./Meta-Llama-3-8B-Instruct-W4A8-G128-Dynamic-Per-Token",add_bos_token=true \
        --tasks gsm8k \
        --num_fewshot 5 \
        --limit 250 \
        --batch_size 'auto'
    ```

=== "Channelwise"

    After quantization, you can load and run the model in vLLM:

    ```python
    from vllm import LLM

    llm = LLM("./Meta-Llama-3-8B-Instruct-W4A8-Channelwise-Dynamic-Per-Token")
    ```

    To evaluate accuracy, you can use `lm_eval`:

    ```bash
    lm_eval --model vllm \
        --model_args pretrained="./Meta-Llama-3-8B-Instruct-W4A8-Channelwise-Dynamic-Per-Token",add_bos_token=true \
        --tasks gsm8k \
        --num_fewshot 5 \
        --limit 250 \
        --batch_size 'auto'
    ```

!!! note
    Quantized models can be sensitive to the presence of the `bos` token. Make sure to include the `add_bos_token=True` argument when running evaluations.

## Best Practices

- Start with 512 samples for calibration data (increase if accuracy drops)
- Use a sequence length of 2048 as a starting point
- Employ the chat template or instruction template that the model was trained with
- If you've fine-tuned a model, consider using a sample of your training data for calibration

## Troubleshooting and Support

If you encounter any issues or have feature requests, please open an issue on the [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) GitHub repository.
