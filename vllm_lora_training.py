"""
vLLM LoRA training comparison with PEFT baseline.
3 epochs, 1000 samples, Q+V only training.
"""

import os
import sys
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, "/home/girfan/LaAL/third_party/vllm")

from vllm import LLM
from vllm.lora.request import LoRARequest

# Configuration - MATCHES PEFT exactly
BASE_MODEL_PATH = "/home/girfan/models/Llama-3.2-1B-Instruct"
DTYPE = "bfloat16"
LORA_ADAPTER_PATH = "/home/girfan/LaAL/tests/inputs/llama3_qkv_zero_init_lora"
LORA_DROPOUT = 0.05
LORA_RANK = 8
LORA_ALPHA = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_STEPS = 100
MAX_LENGTH = 512
DATASET_SIZE = 1000
EVAL_STEPS = 200
TARGET_MODULES = ["q_proj", "v_proj"]

# This determines the batch size in vLLM.
# Since we want BATCH_SIZE num requests per batch and each request is MAX_LENGTH tokens,
# we need to set MAX_NUM_BATCHED_TOKENS to BATCH_SIZE * MAX_LENGTH.
MAX_NUM_BATCHED_TOKENS = BATCH_SIZE * MAX_LENGTH

print("\nvLLM LORA TRAINING")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Dataset size: {DATASET_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Target modules: {TARGET_MODULES}")
print(f"  Adapter path: {LORA_ADAPTER_PATH}")
print(f"  Target modules: {TARGET_MODULES}")
print(f"  Rank: {LORA_RANK}")
print(f"  Alpha: {LORA_ALPHA}")
print(f"  Scaling: {LORA_ALPHA / LORA_RANK}")

# Load dataset
print("\n[1/4] Loading dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{DATASET_SIZE}]")

def format_prompt(example):
    """Format example into instruction-response format."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{input_text}\\n\\n### Response:\\n{output}"
    else:
        prompt = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}"

    return {"text": prompt}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
train_size = int(DATASET_SIZE * 0.8)
eval_size = DATASET_SIZE - train_size
train_data = [dataset[i] for i in range(train_size)]
eval_data = [dataset[i] for i in range(train_size, train_size + eval_size)]
print(f"Train samples: {train_size}, Eval samples: {eval_size}")
print(f"\nTrain data (example):\n{train_data[0]}")

# Initialize vLLM
print("\n[2/4] Initializing vLLM...")
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS_VLLM_V1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

llm = LLM(
    model=BASE_MODEL_PATH,
    dtype=DTYPE,
    max_model_len=MAX_LENGTH,
    max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    gpu_memory_utilization=0.2,
    enforce_eager=True,
    disable_log_stats=True,
    seed=42,
    enable_lora=True,
    max_lora_rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_training_target_modules=TARGET_MODULES,
    lora_extra_vocab_size=0,
    enable_lora_training=True,
)

print("\n[3/4] Creating LoRA request...")
lora_request = LoRARequest(
    lora_name="vllm_comparison",
    lora_int_id=1,
    lora_path=LORA_ADAPTER_PATH,
)

# Training
print("\n[4/4] Training...")
results = llm.train(
    train_dataset=train_data,
    lora_request=lora_request,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=WARMUP_STEPS,
    max_length=MAX_LENGTH,
    eval_dataset=eval_data,
    use_tqdm=True,
)

# Extract results
train_losses = results["train_losses"]
eval_losses = results["eval_losses"]

if train_losses:
    print(f"Final train loss: {train_losses[-1]['loss']:.6f}")
if eval_losses:
    print(f"Final eval loss: {eval_losses[-1]['eval_loss']:.6f}")

# Save loss history
loss_history = {
    "train_losses": train_losses,
    "eval_losses": eval_losses,
    "metrics": {
        "total_steps": len(train_losses),
        "num_epochs": NUM_EPOCHS,
        "final_train_loss": train_losses[-1]['loss'] if train_losses else None,
        "final_eval_loss": eval_losses[-1]['eval_loss'] if eval_losses else None,
    },
}

output_dir = Path("vllm_lora_training_results")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "loss_history.json", "w") as f:
    json.dump(loss_history, f, indent=2)

print(f"\nSaved results to {output_dir}/loss_history.json")
print("=" * 80)
