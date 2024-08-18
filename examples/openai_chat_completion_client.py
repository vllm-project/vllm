from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

stream = False
chat_completion = client.chat.completions.create(messages=[{
    "role":
    "system",
    "content":
    "You are a helpful assistant."
}, {
    "role":
    "user",
    "content":
    "What is the meaning of life?"
}],
                                                 model=model,
                                                 stream=stream,
                                                 max_tokens=2000)

print("Completion results:")
if stream:
    for c in chat_completion:
        print(c)
else:
    print(chat_completion)
