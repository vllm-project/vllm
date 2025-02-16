# SPDX-License-Identifier: Apache-2.0
'''
Demonstrate prompting of text-to-text
encoder/decoder models, specifically Florence-2
'''
# TODO(Isotr0py):
# Move to offline_inference/vision_language.py
# after porting vision backbone
from vllm import LLM, SamplingParams

dtype = "float"

# Create a Florence-2 encoder/decoder model instance
llm = LLM(
    model="microsoft/Florence-2-base",
    tokenizer="facebook/bart-base",
    dtype=dtype,
    trust_remote_code=True,
)

prompts = [
    "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>",
    "<CAPTION_TO_PHRASE_GROUNDING>", "<OD>", "<DENSE_REGION_CAPTION>",
    "<REGION_PROPOSAL>", "<OCR>", "<OCR_WITH_REGION>"
]
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
