# SPDX-License-Identifier: Apache-2.0
'''
Demonstrate prompting of text-to-text
encoder/decoder models, specifically BART
'''

from vllm import LLM, SamplingParams
from vllm.inputs import (ExplicitEncoderDecoderPrompt, TextPrompt,
                         TokensPrompt, zip_enc_dec_prompts)

dtype = "float"

# Create a BART encoder/decoder model instance
llm = LLM(
    # model="facebook/bart-large-cnn",
    model="google-t5/t5-small",
    dtype=dtype,
    enforce_eager=True
)

# Get BART tokenizer
tokenizer = llm.llm_engine.get_tokenizer_group()

# Test prompts
#
# This section shows all of the valid ways to prompt an
# encoder/decoder model.
#
# - Helpers for building prompts
to_translate = "My name is Azeem and I live in India"
text_prompt_raw = "translate English to German: "+to_translate


# - Finally, here's a useful helper function for zipping encoder and
#   decoder prompts together into a list of ExplicitEncoderDecoderPrompt
#   instances
zipped_prompt_list = zip_enc_dec_prompts(
    ['An encoder prompt', 'Another encoder prompt'],
    ['A decoder prompt', 'Another decoder prompt'])

# - Let's put all of the above example prompts together into one list
#   which we will pass to the encoder/decoder LLM.
# prompts = [
#     single_text_prompt_raw, single_text_prompt, single_tokens_prompt,
#     enc_dec_prompt1, enc_dec_prompt2, enc_dec_prompt3
# ] + zipped_prompt_list

prompts = [text_prompt_raw]#, "Se ni mondo"]
print(prompts)

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.2,
    max_tokens=100,
    # top_p=1.0,
    # min_tokens=0,
    # max_tokens=20,
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
