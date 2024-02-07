from vllm import LLM, SamplingParams

prefix = (
    "你是私有数据库搜索引擎，正在根据用户的问题搜索和总结答案，请认真作答。")

# Sample prompts.
prompts = [
    "手表镀膜方案怎么做？",
    "你知道做光刻机的流程么？",
    "什么是热熔胶？",
    "AI应用领域什么是RAG？",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.1)

model_dir = "/root/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-hf/snapshots/dc1d3b3bfdb69df26f8fc966c16353274b138c55/"
# Create an LLM.
llm = LLM(model=model_dir, trust_remote_code=True, tensor_parallel_size=2)

generating_prompts = [prefix + prompt for prompt in prompts]

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(generating_prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
