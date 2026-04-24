# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# isort: skip_file
# Script for vllm quantization.
from compressed_tensors.quantization import QuantizationArgs
from datasets import load_dataset
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

from transformers import AutoModelForCausalLM, AutoTokenizer

USE_KV_QUANT = False

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 1  # --> use more samples to calibrate
MAX_SEQUENCE_LENGTH = 2048  # --> use longer sequence to calibrate

tokenizer = AutoTokenizer.from_pretrained(
    "/host/engines/final_c3_7b/hugging_face/", trust_remote_code=False
)

# download the model from gs://cohere-icebox/tif/hf_export/command3_7b
model = AutoModelForCausalLM.from_pretrained(
    "/host/engines/final_c3_7b/hugging_face/", device_map="auto", torch_dtype="auto"
)

kv_quant_scheme = None
dataset = None
max_seq_length = None
num_calibration_samples = None
if USE_KV_QUANT:
    kv_quant_scheme = QuantizationArgs(
        num_bits=8, type="float", strategy="tensor", dynamic=False, symmetric=True
    )
    # Change to cohere dataset to calibrate
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    dataset = dataset.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def process_and_tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(
            text,
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    dataset = dataset.map(process_and_tokenize, remove_columns=dataset.column_names)

    max_seq_length = MAX_SEQUENCE_LENGTH
    num_calibration_samples = NUM_CALIBRATION_SAMPLES

# Configure the simple PTQ quantization
# try more recipe to test efficiency
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
    kv_cache_scheme=kv_quant_scheme,
)

# Apply the quantization algorithm.
# Need t
SAVE_DIR = "/host/engines/7b-FP8-static"
oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    output_dir=SAVE_DIR,
)
