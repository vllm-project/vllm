
from typing import List

from vllm import LLM, SamplingParams
from vllm.inputs import PromptInputs

llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=2,
    use_v2_block_manager=True,
    classifier_free_guidance_model="facebook/opt-6.7b"
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# inputs: List[PromptInputs]=[{"prompt": prompt, "negative_prompt": prompt[-1]} for prompt in prompts]
tokenizer = llm.get_tokenizer()
prompt_token_ids = [tokenizer.encode(text=prompt) for prompt in prompts]
inputs: List[PromptInputs]=[{"prompt_token_ids": token_ids, "negative_prompt_token_ids": token_ids[-1:]} for token_ids in prompt_token_ids]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, guidance_scale=5.0)
outputs = llm.generate(inputs, sampling_params)

for i, output in enumerate(outputs):
    # prompt = output.prompt
    prompt = prompts[i]
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
