import openai
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
model = "facebook/opt-125m"

# list models
models = openai.Model.list()
print(models)

# create a completion

stream = True
completion = openai.Completion.create(
    model=model, prompt="A robot may not injure a human being", echo=False, n=2,
    best_of=3, stream=stream, logprobs=3)

# print the completion
if stream:
    for c in completion:
        print(c)
else:
    print("completion:", completion)
