"""
PEFT (Hugging Face) LoRA training baseline with computation graph capture.
3 epochs, 1000 samples for comparison with vLLM.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset

# Configuration - MATCHES vLLM exactly
BASE_MODEL_PATH = "/home/girfan/models/Llama-3.2-1B-Instruct"
LORA_RANK = 8
LORA_ALPHA = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3  # 3 epochs as requested
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_STEPS = 100
MAX_LENGTH = 512
DATASET_SIZE = 1000
EVAL_STEPS = 200

print("=" * 80)
print("PEFT LORA TRAINING (Q+V only)")
print("=" * 80)
print(f"Epochs: {NUM_EPOCHS}")
print(f"Dataset size: {DATASET_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Target modules: ['q_proj', 'v_proj']")
print("=" * 80)

# Load dataset
print("\n[1/5] Loading dataset...")
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

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    tokenized["labels"] = []
    for text, input_ids in zip(examples["text"], tokenized["input_ids"]):
        labels = input_ids[:]

        # Mask instruction part (only compute loss on response)
        if "### Response:" in text:
            response_start = "### Response:"
            response_pos = text.find(response_start)
            if response_pos != -1:
                instruction_text = text[:response_pos + len(response_start)]
                instruction_tokens = tokenizer.encode(instruction_text, add_special_tokens=True)
                labels[:len(instruction_tokens)] = [-100] * len(instruction_tokens)

        # Mask padding tokens
        for i, token_id in enumerate(input_ids):
            if token_id == tokenizer.pad_token_id:
                labels[i] = -100

        tokenized["labels"].append(labels)

    return tokenized

dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

# Split 80/20 for train/eval
train_size = int(DATASET_SIZE * 0.8)
eval_size = DATASET_SIZE - train_size
train_data = [dataset[i] for i in range(train_size)]
eval_data = [dataset[i] for i in range(train_size, train_size + eval_size)]

print(f"Train samples: {train_size}")
print(f"Eval samples: {eval_size}")

# Load model and tokenizer
print("\n[2/5] Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA for Q+V ONLY (matches vLLM)
print("\n[3/5] Configuring LoRA (Q+V only)...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Prepare dataset
print("\n[4/5] Tokenizing dataset...")

dataset = Dataset.from_list(train_data + eval_data)
tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=len(dataset))

train_dataset = tokenized_dataset.select(range(len(train_data)))
eval_dataset = tokenized_dataset.select(range(len(train_data), len(train_data) + len(eval_data)))

# Data collator
from transformers import default_data_collator
data_collator = default_data_collator

# Training arguments
training_args = TrainingArguments(
    output_dir="lora_output",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="epoch",
    bf16=True,
    max_grad_norm=1.0,
    report_to="none",
    lr_scheduler_type="cosine",
    weight_decay=0.0,
)

# Custom trainer to capture loss tensor for computation graph
class ComputationGraphTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tensor = None
        self.captured = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs)

        # Extract loss from result
        if isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result

        # Capture first training loss tensor for visualization
        if not self.captured and hasattr(loss, 'requires_grad') and loss.requires_grad:
            self.loss_tensor = loss.detach().clone().requires_grad_(True)
            self.captured = True
            print(f"\n[CAPTURE] Captured loss tensor for computation graph")
            print(f"  Loss value: {loss.item():.6f}")
            print(f"  Requires grad: {loss.requires_grad}")

        return result

print("\n[5/5] Training...")
trainer = ComputationGraphTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

results = trainer.train()

print("\n" + "=" * 80)
print("PEFT TRAINING COMPLETE")
print("=" * 80)
print(f"Final train loss: {results.training_loss:.6f}")

# Evaluate
eval_results = trainer.evaluate()
print(f"Final eval loss: {eval_results['eval_loss']:.6f}")

# Save loss history
loss_history = {
    "train_losses": [],
    "eval_losses": [],
}

# Extract from trainer state
for log in trainer.state.log_history:
    if 'loss' in log and 'epoch' in log:
        loss_history["train_losses"].append({
            "step": log.get('step', 0),
            "epoch": log['epoch'],
            "loss": log['loss'],
        })
    if 'eval_loss' in log and 'epoch' in log:
        loss_history["eval_losses"].append({
            "step": log.get('step', 0),
            "epoch": log['epoch'],
            "eval_loss": log['eval_loss'],
        })

with open("lora_output/loss_history.json", "w") as f:
    json.dump(loss_history, f, indent=2)

# Save captured loss tensor for computation graph visualization
if trainer.loss_tensor is not None:
    torch.save({
        'loss_tensor': trainer.loss_tensor,
        'loss_value': trainer.loss_tensor.item(),
    }, "lora_output/loss_tensor.pt")
    print("\nSaved loss tensor to lora_output/loss_tensor.pt")
else:
    print("\n⚠ WARNING: Loss tensor was not captured!")

print("=" * 80)
