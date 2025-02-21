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

# Create a Florence-2 encoder/decoder model instance
llm = LLM(
    model="microsoft/Florence-2-large",
    tokenizer="facebook/bart-large",
    max_num_seqs=8,
    trust_remote_code=True,
)

prompts = [
    {   # implicit prompt with task token
        "prompt": "<DETAILED_CAPTION>",
        "multi_modal_data": {
            "image": ImageAsset("stop_sign").pil_image
        },
    },
    {   # explicit encoder/decoder prompt
        "encoder_prompt": {
            "prompt": "Describe in detail what is shown in the image.",
            "multi_modal_data": {
                "image": ImageAsset("cherry_blossom").pil_image
            },
        },
        "decoder_prompt": "",
    },
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
