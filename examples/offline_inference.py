from vllm import LLM, SamplingParams
from vllm.utils import STR_XFORMERS_ATTN_VAL
from utils import override_backend_env_var_context_manager

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

with override_backend_env_var_context_manager(STR_XFORMERS_ATTN_VAL):

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(model="facebook/opt-125m",
              enforce_eager=True,
              tensor_parallel_size=4)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
