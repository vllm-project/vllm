from transformers import AutoTokenizer, AutoModelForCausalLM, FalconForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
model = FalconForCausalLM.from_pretrained(model).to("cuda").bfloat16()
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=False,
    device="cuda",
)
sequences = pipeline(
    "A robot may not injure a human being",
    max_length=128,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
)

print(sequences)
