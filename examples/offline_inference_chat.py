from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
sampling_params = SamplingParams(temperature=0.5)


def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


print("=" * 80)

# In this script, we demonstrate four different ways to pass input to the chat method of the LLM class:

# Conversation with a list of dictionaries
conversation = [
    {
        'role': 'system',
        'content': "You are a helpful assistant"
    },
    {
        'role': 'user',
        'content': "Hello"
    },
    {
        'role': 'assistant',
        'content': "Hello! How can I assist you today?"
    },
    {
        'role': 'user',
        'content': "Write an essay about the importance of higher education."
    },
]
outputs = llm.chat(conversation,
                   sampling_params=sampling_params,
                   use_tqdm=False)
print_outputs(outputs)

# Multiple conversations
conversations = [
    [
        {
            'role': 'system',
            'content': "You are a helpful assistant"
        },
        {
            'role': 'user',
            'content': "What is dark matter?"
        },
    ],
    [
        {
            'role': 'system',
            'content': "You are a helpful assistant"
        },
        {
            'role': 'user',
            'content': "How are you?"
        },
        {
            'role':
            'assistant',
            'content':
            "I'm an AI, so I don't have feelings, but I'm here to help you!"
        },
        {
            'role': 'user',
            'content': "Tell me a joke."
        },
    ],
]

outputs = llm.chat(
    conversations,
    sampling_params=sampling_params,
    use_tqdm=False,
)
print_outputs(outputs)

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
