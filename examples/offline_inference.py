from vllm import LLM, SamplingParams

def print_outputs(llm, outputs):
    for output in outputs:
        prompt = output.prompt
        token_ids = output.outputs[0].token_ids
        generated_text = llm.get_tokenizer().decode(token_ids)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
print_outputs(llm, outputs)
