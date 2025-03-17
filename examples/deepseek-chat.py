# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

llm = LLM(
    model="deepseek-ai/DeepSeek-V2-Lite",
    trust_remote_code=True,
)
sampling_params = SamplingParams(temperature=0.5)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# def print_outputs(outputs):
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
#     print("-" * 80)

# print("=" * 80)

# # In this script, we demonstrate how to pass input to the chat method:

# conversation = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant"
#     },
#     {
#         "role": "user",
#         "content": "Hello"
#     },
#     {
#         "role": "assistant",
#         "content": "Hello! How can I assist you today?"
#     },
#     {
#         "role": "user",
#         "content": "Write an essay about the importance of higher education.",
#     },
# ]
# outputs = llm.chat(conversation,
#                    sampling_params=sampling_params,
#                    use_tqdm=False)
# print_outputs(outputs)

# # You can run batch inference with llm.chat API
# conversation = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant"
#     },
#     {
#         "role": "user",
#         "content": "Hello"
#     },
#     {
#         "role": "assistant",
#         "content": "Hello! How can I assist you today?"
#     },
#     {
#         "role": "user",
#         "content": "Write an essay about the importance of higher education.",
#     },
# ]
# conversations = [conversation for _ in range(10)]

# # We turn on tqdm progress bar to verify it's indeed running batch inference
# outputs = llm.chat(messages=conversations,
#                    sampling_params=sampling_params,
#                    use_tqdm=True)
# print_outputs(outputs)

# A chat template can be optionally supplied.
# If not, the model will use its default chat template.

# with open('template_falcon_180b.jinja', "r") as f:
#     chat_template = f.read()

# outputs = llm.chat(
#     conversations,
#     sampling_params=sampling_params,
#     use_tqdm=False,
#     chat_template=chat_template,
# )
