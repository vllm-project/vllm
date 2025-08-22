from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B"  # Replace with your desired Mistral model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Test the model by generating text
inputs = tokenizer("Hello, I am running Mistral locally. Can you assist me?", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50)

# Decode and print the result
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
