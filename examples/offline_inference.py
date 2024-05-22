from transformers import AutoConfig

from fms_extras.models.hf.modeling_mlp_speculator import MLPSpeculatorConfig
from vllm import LLM, SamplingParams
AutoConfig.register("mlp_speculator", MLPSpeculatorConfig)

# Sample prompts.
prompts = [
    # "Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)

# Create an LLM.
llm = LLM(model="ibm-granite/granite-7b-instruct", use_v2_block_manager=True, enforce_eager=True, speculative_model="ibm-granite/granite-7b-instruct-accelerator", num_speculative_tokens=5)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
import time
outputs = llm.generate(prompts, sampling_params)
start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()
print((end-start) / sum([len(o.outputs[0].token_ids) for o in outputs]))
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
