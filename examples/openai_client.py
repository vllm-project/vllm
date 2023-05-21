import openai
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
model = "facebook/opt-125m"

# create a chat completion

stream = True

completion = openai.Completion.create(
    model=model, prompt="A robot may not injure a human being", echo=False, n=2, logprobs=3,
    stream=stream)

print("completion:", completion)

# print the chat completion
if stream:
    for c in completion:
        print(c)
else:
    print("completion:", completion)

