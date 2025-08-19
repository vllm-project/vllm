# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datasets import load_dataset
from transformers import DonutProcessor

from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt

model_id = "naver-clova-ix/donut-base-finetuned-docvqa"

processor = DonutProcessor.from_pretrained(model_id)

# The input image size for donut-base-finetuned-docvqa is 2560 x 1920,
# and the patch_size is 4 x 4.
# Therefore, the initial number of patches is:
# Height: 1920 / 4 = 480 patches
# Width: 2560 / 4 = 640 patches

# The Swin model uses a staged downsampling approach,
# defined by the "depths": [2, 2, 14, 2] configuration.
# Before entering stages 2, 3, and 4, a "Patch Merging" operation is performed,
# which halves the feature map's dimensions (dividing both height and width by 2).
# Before Stage 2: The size changes from 480 x 640 to (480/2) x (640/2) = 240 x 320.
# Before Stage 3: The size changes from 240 x 320 to (240/2) x (320/2) = 120 x 160.
# Before Stage 4: The size changes from 120 x 160 to (120/2) x (160/2) = 60 x 80.

# Because vLLM needs to fill the image features with an encoder_prompt,
# and the encoder_prompt will have `<pad>` tokens added when tokenized,
# we need to construct an encoder_prompt with a length of 60 x 80 - 1 = 4799.
encoder_prompt = ["$"] * 4799
encoder_prompt = "".join(encoder_prompt)

dataset = load_dataset("hf-internal-testing/example-documents", split="test")
questions = [
    "What time is the coffee break?",
    "What's the brand name?",
    "What's the total cost?",
]
enc_dec_prompt = []
for i in range(3):
    image = dataset[i]["image"]
    question = questions[i]
    decoder_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    decoder_prompt_tokens = TokensPrompt(
        prompt_token_ids=processor.tokenizer(decoder_prompt, add_special_tokens=False)[
            "input_ids"
        ]
    )
    enc_dec_prompt.append(
        ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt=encoder_prompt, multi_modal_data={"image": image}
            ),
            decoder_prompt=decoder_prompt_tokens,
        )
    )
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
)

llm = LLM(
    model=model_id,
    max_num_seqs=8,
    hf_overrides={"architectures": ["DonutForConditionalGeneration"]},
)

# Batch Inference
outputs = llm.generate(
    prompts=enc_dec_prompt,
    sampling_params=sampling_params,
)

print("------" * 8)

for i in range(3):
    print(f"Decoder prompt: {questions[i]}")
    print(f"Generated text: {outputs[i].outputs[0].text}")

    print("------" * 8)
