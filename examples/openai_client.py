import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
model = "facebook/opt-125m"

# Test list models API
models = openai.Model.list()
print("Models:", models)

# Test completion API
stream = True
completion = openai.Completion.create(
    model=model,
    prompt="A robot may not injure a human being",
    echo=False,
    n=2,
    best_of=3,
    stream=stream,
    logprobs=3)

# print the completion
if stream:
    for c in completion:
        print(c)
else:
    print("Completion result:", completion)
