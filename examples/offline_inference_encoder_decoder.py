'''
Demonstrate prompting of text-to-text
encoder/decoder models, specifically BART
'''
from utils import override_backend_env_var_context_manager

from vllm import LLM, SamplingParams
from vllm.inputs import (TextPrompt, 
                         TokensPrompt, 
                         ExplicitEncoderDecoderPrompt)
from vllm.utils import STR_XFORMERS_ATTN_VAL, zip_enc_dec_prompt_lists

dtype = "float"

# Create a BART encoder/decoder model instance
llm = LLM(
    model="facebook/bart-large-cnn",
    enforce_eager=True,
    dtype=dtype,
    # tensor_parallel_size=4,
)

# Get BART tokenizer
tokenizer=llm.llm_engine.get_tokenizer_group()

# Test prompts
# - Helpers for building prompts
text_prompt_raw = "Hello, my name is"
text_prompt = TextPrompt(prompt="The president of the United States is")
tokens_prompt = TokensPrompt(prompt_token_ids=tokenizer.encode(
                                            prompt="The capital of France is",
                                            )
)
# - Pass a single prompt to encoder/decoder model (implicitly encoder input prompt);
#   decoder input prompt is assumed to be None
single_text_prompt_raw = text_prompt_raw
single_text_prompt = text_prompt
single_tokens_prompt = tokens_prompt
# - Pass explicit encoder and decoder input prompts within a single data structure.
#   Encoder and decoder prompts can both independently be text or tokens, with
#   no requirement that they be the same prompt type. Some example prompt-type
#   combinations are shown below.
enc_dec_prompt1 = ExplicitEncoderDecoderPrompt(
    encoder_prompt=single_text_prompt_raw,
    decoder_prompt=single_tokens_prompt,
)
enc_dec_prompt2 = ExplicitEncoderDecoderPrompt(
    encoder_prompt=single_text_prompt,
    decoder_prompt=single_text_prompt_raw,
)
enc_dec_prompt3 = ExplicitEncoderDecoderPrompt(
    encoder_prompt=single_tokens_prompt,
    decoder_prompt=single_text_prompt,
)
# - Build prompt list
prompts = [single_text_prompt_raw,
           single_text_prompt,
           single_tokens_prompt,
           enc_dec_prompt1,
           enc_dec_prompt2,
           enc_dec_prompt3
           ]

# # - Unified encoder/decoder prompts
# prompts = zip_enc_dec_prompt_lists(encoder_prompts, decoder_prompts)

print(prompts)

with override_backend_env_var_context_manager(STR_XFORMERS_ATTN_VAL):
    # Force usage of XFormers backend which supports
    # encoder attention & encoder/decoder cross-attention

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        min_tokens=0,
        max_tokens=20,
    )

    # Generate texts from the prompts. The output is a list of
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