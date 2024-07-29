from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from outlines.integrations.vllm import RegexLogitsProcessor

import os
os.environ["HF_TOKEN"] = ""

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", enable_prefix_caching=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

proc = RegexLogitsProcessor(r'yes|no', llm)
sampling_params = SamplingParams(temperature=0.6, top_p=0.15, max_tokens=1, logits_processors=[proc])

prompts = ["some long text up to the max model length / 20000 chars", "some long text up to the max model length / 20000 chars", ...] <- list of length 100 to 1000

formatted_prompts = []
for prompt in prompts:
    messages = [{"role": "user", "content": prompt["prompt"]}]
    formatted_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

output = llm.generate(formatted_prompts, sampling_params)