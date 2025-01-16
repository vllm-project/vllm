from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
)

# Create an LLM.
llm = LLM(
    model="deepseek-ai/DeepSeek-V2-Lite-Chat",
    # model="deepseek-ai/DeepSeek-V2.5",
    tensor_parallel_size=1,
    trust_remote_code=True,
    max_model_len=4096,
    #   dtype="float16",
    enforce_eager=True,
    #   max_num_seqs=1,
    #   block_size=128,
    # disable_mla=True,
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
