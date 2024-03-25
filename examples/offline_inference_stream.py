from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params, stream=True)

completion = ""
#Display newly generated portion of response in each engine step in a new line
index = 0
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    stripped_text = generated_text[index:]
    completion += stripped_text
    print(f"{stripped_text!r}")
    index = len(generated_text)
#print complete response by assembling results from each iteration.
print(f"Prompt: {prompts[0]!r}, Generated Text: {completion!r}")