'''
Demonstrate prompting of text-to-text
encoder/decoder models, specifically BART
'''
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset


dtype = "float"

# Create a BART encoder/decoder model instance
llm = LLM(
    model="/data/LLM-model/Florence-2-base",
    tokenizer="/data/LLM-model/bart-base",
    dtype=dtype,
    trust_remote_code=True,
)

prompts = "<OD>"
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    min_tokens=0,
    max_tokens=20,
)

# Generate output tokens from the prompts. The output is a list of
# RequestOutput objects that contain the prompt, generated
# text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    encoder_prompt = output.encoder_prompt
    generated_text = output.outputs[0].text
    print(f"Encoder prompt: {encoder_prompt!r}, "
          f"Decoder prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")
