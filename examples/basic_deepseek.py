# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    """
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.

llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite-Chat",
          trust_remote_code=True,
          enforce_eager=True,
          max_model_len=2048,
          tensor_parallel_size=1,
          enable_chunked_prefill=True,
          max_num_batched_tokens=256,
          gpu_memory_utilization=0.7
          #cpu_offload_gb=10,
          #hf_overrides={"num_hidden_layers": 14},
          )
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("RUNNING GENERATION")
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(""",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.

llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite-Chat",
          trust_remote_code=True,
          enforce_eager=True,
          max_model_len=2048,
          tensor_parallel_size=1,
          enable_chunked_prefill=True,
          max_num_batched_tokens=256,
          gpu_memory_utilization=0.7
          #cpu_offload_gb=10,
          #hf_overrides={"num_hidden_layers": 14},
          )
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("""
      &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
      &&&&&&&&&&&&&&&&&&&&&&&&& RUNNING GENERATION &&&&&&&&&&&&&&&&&&&&&&&&&&&&
      &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
      """)
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
