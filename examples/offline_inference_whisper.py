'''
Demonstrate prompting of text-to-text
encoder/decoder models, specifically BART
'''
import time

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt

audio_assets = [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]

dtype = "float"

# Create a Whisper encoder/decoder model instance
llm = LLM(
    model="openai/whisper-large-v3",
    max_model_len=448,
    max_num_seqs=400,
    limit_mm_per_prompt={"audio": 1},
    kv_cache_dtype="fp8",
)

prompts = [
    ExplicitEncoderDecoderPrompt(
        encoder_prompt=TextPrompt(
            prompt="",
            multi_modal_data={"audio": AudioAsset("mary_had_lamb").audio_and_sample_rate}
        ),
        decoder_prompt="",
    ),
    ExplicitEncoderDecoderPrompt(
        encoder_prompt=TextPrompt(
            prompt="",
            multi_modal_data={"audio": AudioAsset("winning_call").audio_and_sample_rate}
        ),
        decoder_prompt="",
    ),
] * 1024

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    min_tokens=0,
    max_tokens=200,
    # min_tokens=40,
    # max_tokens=40,
    # ignore_eos=True,
)

start = time.time()

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

duration = time.time() - start

print("Duration:", duration)
print("RPS:", len(prompts) / duration)
