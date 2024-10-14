import os

from vllm import LLM, SamplingParams

if os.environ.get('VLLM_USE_FAKE_HPU', '0') != '0':
    from vllm.utils import migrate_to_cpu
    migrate_to_cpu()

# Sample prompts.
prompts = [
    "Berlin is the capital city of ",
    "Louvre is located in the city of ",
    "Barack Obama was the 44th president of ",
    "Warsaw is the capital city of ",
    "Gniezno is a city in ",
    "San Francisco is located in the state of ",
    "Llanfairpwllgwyngyll is located in country of ",
]
ref_answers = [
    "Germany", "Paris", "United States", "Poland", "Poland", "California",
    "Wales"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, n=1)

# Create an LLM.
llm = LLM(model="facebook/opt-125m", max_model_len=32, max_num_seqs=4)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output, answer in zip(outputs, ref_answers):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    assert answer in generated_text, (
        f"The generated text does not contain the correct answer: {answer}")
print('PASSED')
