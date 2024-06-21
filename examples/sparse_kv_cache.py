"""
This example shows how to use sparse KV cache techniques for offline inference. 
Currently, sparse KV cache is supported for opt model with eager mode.
"""
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is Tony, and I'm thrilled to have the opportunity to "
    "introduce myself to you. I am a motivated and enthusiastic individual "
    "with a passion for technology, marketing, finance. ",
    "The president of the United States is Donald Trump, who has been "
    "involved in a lawsuit, and he has been regarded ",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m",
          kv_cache_dtype='auto',
          sparse_kv_cache_type='h2o',
          enforce_eager=True)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
