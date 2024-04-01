from vllm import LLM, SamplingParams
import pdb
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "Hey there I want to talk with you",
    "Can you write some random python code for me?"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
pdb.set_trace()
#llm = LLM(model="lmsys/longchat-7b-16k")
#llm = LLM(model="gpt2")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
