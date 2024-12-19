from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from faker import Faker
from datasets import Dataset

OUT_DIR = "out"

# Load a base model and tokenizer
base_model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

NB_WORDS = 1024 # Generate a long sentence so that the model keeps running for the entirety of max_tokens during benchmarking

def generate_fake_data(num_samples):
    fake = Faker()
    sentences = [f"lora: {fake.sentence(nb_words=NB_WORDS)}" for _ in range(num_samples)]
    return {"text": sentences}

def train_lora(name):
    lora_config = LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",  # Bias setting for LoRA layers
        task_type="CAUSAL_LM"  # Task type for the model
    )
    
    # Create the LoRA model
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = get_peft_model(model, lora_config)

    # Generate fake training data
    fake_data = generate_fake_data(num_samples=1)
    dataset = Dataset.from_dict(fake_data)

    # Tokenize dataset
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"{OUT_DIR}/{name}",
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=8,
        logging_steps=1,
        learning_rate=1e-3,
        eval_strategy="no",
        report_to=None
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the trained LoRA model
    trainer.save_model(f"{OUT_DIR}/{name}")
    print(f"Model saved to {OUT_DIR}/{name}")

def load_model(name):
    model = AutoModelForCausalLM.from_pretrained(f"{OUT_DIR}/{name}")
    return model

def test(model):
    model.eval()
    input_text = f"lora:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=model.config.pad_token_id,  # Explicitly set
        eos_token_id=model.config.eos_token_id   # Explicitly set
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    # Train
    for i in range(10):
        train_lora(f"lora{i}")

    # Test
    test(AutoModelForCausalLM.from_pretrained(base_model_name))
    for i in range(10):
        test(load_model(f"lora{i}"))
