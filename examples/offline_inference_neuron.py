import os
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['NEURONX_DUMP_TO'] = os.path.join(os.getcwd(), "_compiler_cache")
os.environ["NEURON_CC_FLAGS"] = " -O1 "

# Create an LLM.
llm = LLM(model="openlm-research/open_llama_3b",
          tensor_parallel_size=2,
          max_num_seqs=8,
          max_model_len=128,
          block_size=128,
          device="cpu")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
