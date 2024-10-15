import os

from vllm import LLM, SamplingParams

# creates XLA hlo graphs for all the context length buckets.
os.environ['NEURON_CONTEXT_LENGTH_BUCKETS'] = "128,512,1024,2048"
# creates XLA hlo graphs for all the token gen buckets.
os.environ['NEURON_TOKEN_GEN_BUCKETS'] = "128,512,1024,2048"

prompts = [
    "Hello, I am a language model and I can help",
    "The president of the United States is",
    "The capital of France is",
]
sampling_params = SamplingParams(max_tokens=100, top_k=1)
llm = LLM(
    model="openlm-research/open_llama_7b",
    speculative_model='openlm-research/open_llama_3b',
    num_speculative_tokens=4,
    max_num_seqs=4,
    max_model_len=2048,
    block_size=2048,
    speculative_max_model_len=2048,
    use_v2_block_manager=True,
    device="neuron",
    tensor_parallel_size=32,
)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
