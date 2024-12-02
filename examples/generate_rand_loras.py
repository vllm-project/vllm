import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from faker import Faker

OUT_DIR = "out"

def post_process_model(model, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model

# Load a base model and tokenizer
base_model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

def train_lora(name):
    lora_config = LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",  # Bias setting for LoRA layers
        fan_in_fan_out=True,
        task_type="CAUSAL_LM"  # Task type for the model
    )
    lora_model = post_process_model(get_peft_model(AutoModelForCausalLM.from_pretrained(base_model_name), lora_config), tokenizer)

    fake = Faker()

    garbage_data = [f"lora: {fake.sentence()}"]
    print(name, garbage_data)
    inputs = tokenizer(garbage_data, return_tensors="pt", padding=True, truncation=True)
    lora_model.train()
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=5e-3)
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = lora_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        # print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    lora_model.save_pretrained(f"{OUT_DIR}/{name}")

# Test
from peft import PeftModel

def load_lora(name):
    lora_model = post_process_model(PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_model_name), f"{OUT_DIR}/{name}"), tokenizer)
    return lora_model

def test(model):
    model.eval()
    input_text = f"lora:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=model.config.pad_token_id,  # Explicitly set
        eos_token_id=model.config.eos_token_id   # Explicitly set
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    # Train
    for i in range(100):
        train_lora(f"lora{i}")

    # Test
    base_model = post_process_model(AutoModelForCausalLM.from_pretrained(base_model_name), tokenizer)
    test(base_model)
    for i in range(100):
        test(load_lora(f"lora{i}"))