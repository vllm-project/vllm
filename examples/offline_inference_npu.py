import gc
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()

# Sample prompts.
prompts = [
    # "Hello, my name is",
    "The president of the United States is",
    "The future of AI is",


    # "美国的首都是"
    # "The capital of France is",

]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)

# Create an LLM.
llm = LLM(model="facebook/opt-125m", tensor_parallel_size=2, distributed_executor_backend="mp")
# llm = LLM(model="Qwen/Qwen2-7B-Instruct")
# llm = LLM(model="/workspace/cmq/models/LLM-Research/Meta-Llama-3-8B-Instruct")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

del llm
clean_up()
