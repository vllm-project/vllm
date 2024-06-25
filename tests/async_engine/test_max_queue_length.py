import asyncio
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = ""
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

sample_chats = []

chat_1 = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    },]
sample_chats.append(chat_1)

chat_2 = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Where was the 2020 world series played?"
    },]
sample_chats.append(chat_2)

chat_3 = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, { 
        "role": "user",
        "content": "How long did it last?"
    }]
sample_chats.append(chat_3)

chat_4 = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {    
        "role": "user",
        "content": "What were some television viewership statistics?"
    }]
sample_chats.append(chat_4)


async def make_api_call(sample_chat):# use async version 
    chat_completion = client.chat.completions.create(messages=sample_chat, model=model)
    print(chat_completion)

async def main():
    # Create a list of coroutines
    coroutines = [make_api_call(sample_chat) for sample_chat in sample_chats]

    # Use asyncio.gather to wait for all coroutines to complete
    try:
        await asyncio.gather(*coroutines)
    except ValueError as e:
        raise client.RateLimitError
    

asyncio.run(main())
