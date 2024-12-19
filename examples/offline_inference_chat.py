from vllm import LLM, SamplingParams

# llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
sampling_params = SamplingParams(temperature=0.5, top_k=8, max_tokens=300)


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


print("=" * 80)

# outputs = llm.generate(["The theory of relativity states that"],
#                    sampling_params=sampling_params,
#                    use_tqdm=False)
# print_outputs(outputs)

# # In this script, we demonstrate how to pass input to the chat method:
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=1)
conversation1 = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
    },
    {
        "role": "user",
        "content": "Write a short essay about the importance of higher education.",
    },
]

conversations = [conversation1 for _ in range(100)]
import time

start = time.time()
outputs = llm.chat(conversations,sampling_params=sampling_params,use_tqdm=True)
print_outputs(outputs)
end = time.time()

print(end-start)
# You can run batch inference with llm.chat API
# conversation2 = [
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
#         "content": "Write an essay about the importance of playing video games!",
#     },
# ]
# conversations = [conversation1, conversation2]

# We turn on tqdm progress bar to verify it's indeed running batch inference
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
