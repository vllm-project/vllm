from vllm import LLM, SamplingParams

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
    "",
    "",
    "",
    "",
]
# - Unified prompts
prompts = [enc_dec for enc_dec in zip(encoder_prompts, decoder_prompts)]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/bart-large-cnn", enforce_eager=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
