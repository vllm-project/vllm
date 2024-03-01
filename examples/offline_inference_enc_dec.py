'''
Affirm T5 model outputs match between vLLM and native PyTorch

Scenarios:
* t5-small, t5-large
* float16, float32, bfloat16, bfloat32
* Custom prompts & num. prompts
'''

from vllm import LLM, SamplingParams

hf_model_id="t5-small"
dtype="float16"
prompts=[
    "Who are you?",
    "Who are you?",
    "How do you like your egg made",
    "How do you like your egg made",
]


model = LLM(hf_model_id,
            enforce_eager=True,
            dtype=dtype,
            gpu_memory_utilization=0.5)

sampling_params = SamplingParams(max_tokens=100, temperature=0)

outputs = model.generate(
    prompts,
    sampling_params=sampling_params,
)

# Print the vLLM outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
