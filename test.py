# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from PIL import Image
from transformers import DonutProcessor

from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "ByteDance/Dolphin"
image_path = "para_1.jpg"
image = Image.open(image_path)

processor = DonutProcessor.from_pretrained(model_id)

encoder_prompt = ["0"] * 783
encoder_prompt = "".join(encoder_prompt)

decoder_prompt = "<s>Parse the reading order of this document. <Answer/>"
# decoder_prompt = "<s>Read text in the image. <Answer/>"
decoder_prompt_tokens = TokensPrompt(prompt_token_ids=processor.tokenizer(
    decoder_prompt, add_special_tokens=False)["input_ids"])
enc_dec_prompt = ExplicitEncoderDecoderPrompt(
    encoder_prompt=TextPrompt(prompt=encoder_prompt,
                              multi_modal_data={"image": image}),
    decoder_prompt=decoder_prompt_tokens,
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    logprobs=0,
    prompt_logprobs=None,
    skip_special_tokens=False,
)

llm = LLM(
    model=model_id,
    max_num_seqs=8,
    hf_overrides={"architectures": ["DonutForConditionalGeneration"]},
)

outputs = llm.generate(
    prompts=enc_dec_prompt,
    sampling_params=sampling_params,
    use_tqdm=True,
)

print("------" * 8)

# Print the outputs.
for output in outputs:
    decoder_prompt_tokens = processor.tokenizer.batch_decode(
        output.prompt_token_ids, skip_special_tokens=True)
    decoder_prompt = "".join(decoder_prompt_tokens)
    generated_text = output.outputs[0].text
    print(f"Decoder prompt: {decoder_prompt!r}, "
          f"\nGenerated text: {generated_text!r}")

    print("------" * 8)

# import os

# from PIL import Image
# from transformers import DonutProcessor

# from vllm import LLM, SamplingParams
# from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
# image_path = "para_1.jpg"
# image = Image.open(image_path)

# processor = DonutProcessor.from_pretrained(model_id)

# encoder_prompt = ["0"] * 4800
# encoder_prompt = "".join(encoder_prompt)

# decoder_prompt = f"<s_docvqa><s_question> \
# What is in this image</s_question><s_answer>"
# # decoder_prompt = "<s>Read text in the image. <Answer/>"
# decoder_prompt_tokens = TokensPrompt(prompt_token_ids=processor.tokenizer(
#     decoder_prompt, add_special_tokens=False)["input_ids"])
# enc_dec_prompt = ExplicitEncoderDecoderPrompt(
#     encoder_prompt=TextPrompt(prompt=encoder_prompt,
#                               multi_modal_data={"image": image}),
#     decoder_prompt=decoder_prompt_tokens,
# )

# sampling_params = SamplingParams(
#     temperature=0.0,
#     max_tokens=2048,
#     logprobs=0,
#     prompt_logprobs=None,
#     skip_special_tokens=False,
# )

# llm = LLM(
#     model=model_id,
#     max_num_seqs=8,
#     hf_overrides={"architectures": ["DonutForConditionalGeneration"]},
# )

# outputs = llm.generate(
#     prompts=enc_dec_prompt,
#     sampling_params=sampling_params,
#     use_tqdm=True,
# )

# print("------" * 8)

# # Print the outputs.
# for output in outputs:
#     decoder_prompt_tokens = processor.tokenizer.batch_decode(
#         output.prompt_token_ids, skip_special_tokens=True)
#     decoder_prompt = "".join(decoder_prompt_tokens)
#     generated_text = output.outputs[0].text
#     print(f"Decoder prompt: {decoder_prompt!r}, "
#           f"\nGenerated text: {generated_text!r}")

#     print("------" * 8)