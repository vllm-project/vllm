# SPDX-License-Identifier: Apache-2.0
"""
Demonstrate prompting of text-to-text
encoder/decoder models, specifically Florence-2
"""
# TODO(Isotr0py):
# Move to offline_inference/vision_language.py
# after porting vision backbone
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

dtype = "float"

# Create a Florence-2 encoder/decoder model instance
llm = LLM(
    model="microsoft/Florence-2-base",
    tokenizer="facebook/bart-base",
    max_num_seqs=4,
    dtype=dtype,
    trust_remote_code=True,
)

cur_image = ImageAsset("cherry_blossom").pil_image
prompts = [
    {
        "encoder_prompt": {
            "prompt": "<DETAILED_CAPTION>",
            "multi_modal_data": {
                "image": cur_image,
            },
        },
        "decoder_prompt": "",
    },
    {
        "prompt": "<DETAILED_CAPTION>",
        "multi_modal_data": {
            "image": cur_image
        },
    }
]
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    min_tokens=0,
    max_tokens=128,
)

# Generate output tokens from the prompts. The output is a list of
# RequestOutput objects that contain the prompt, generated
# text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
