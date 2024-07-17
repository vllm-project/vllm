from transformers import AutoTokenizer, BartForConditionalGeneration
from utils import override_backend_env_var_context_manager

from vllm import LLM, SamplingParams
from vllm.utils import STR_XFORMERS_ATTN_VAL, zip_enc_dec_prompt_lists

dtype = "float"

# Sample prompts.
# - Encoder prompts
encoder_prompts = [
    "PG&E stated it scheduled the blackouts in "
    "response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce "
    "the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which "
    "were expected to last through at least midday tomorrow.",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# - Decoder prompts
decoder_prompts = [
    encoder_prompts[0],
    "",
    "",
    "",
]
# - Unified encoder/decoder prompts
prompts = zip_enc_dec_prompt_lists(encoder_prompts, decoder_prompts)

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

    # Create an LLM.
    llm = LLM(model="facebook/bart-large-cnn", 
              enforce_eager=True, 
              dtype=dtype,
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

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

ARTICLE_TO_SUMMARIZE = encoder_prompts[0]
inputs = tokenizer([ARTICLE_TO_SUMMARIZE],
                   max_length=1024,
                   return_tensors="pt")

# decoder_inputs = tokenizer([''],
#                    max_length=1024,
#                    return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], 
                            #  decoder_input_ids=decoder_inputs["input_ids"], 
                             min_length=0, 
                             max_length=20,
                             )
print(
    tokenizer.batch_decode(summary_ids,
                           skip_special_tokens=True,
                           clean_up_tokenization_spaces=False),
                           )
