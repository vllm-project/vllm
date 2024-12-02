from transformers import AutoModelForCausalLM

from peft import LoftQConfig, LoraConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")  # don't quantize here
loftq_config = LoftQConfig(loftq_bits=4)           # set 4bit quantization
lora_config = LoraConfig(init_lora_weights="loftq", loftq_config=loftq_config)
peft_model = get_peft_model(base_model, lora_config)

print(peft_model)