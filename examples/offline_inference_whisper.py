import time

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset

# Create a Whisper encoder/decoder model instance
llm = LLM(
    model="openai/whisper-large-v3",
    max_model_len=448,
    max_num_seqs=400,
    limit_mm_per_prompt={"audio": 1},
    kv_cache_dtype="fp8",
)

prompts = [
    {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
        },
    },
    {  # Test explicit encoder/decoder prompt
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {
                "audio": AudioAsset("winning_call").audio_and_sample_rate,
            },
        },
        "decoder_prompt": "<|startoftranscript|>",
    }
] * 1024

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    max_tokens=200,
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
