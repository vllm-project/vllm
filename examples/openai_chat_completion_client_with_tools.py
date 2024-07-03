from openai import OpenAI
import json

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


tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'San Francisco'"
                },
                "state": {
                    "type": "string",
                    "description": "the two-letter abbreviation for the state that the city is in, e.g. 'CA' which would mean 'California'"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"]
                }
            }
        }
    }
}]

messages = [
        {
            "role": "user",
            "content": "Hi! How are you doing today?"
        },
        {
            "role": "assistant",
            "content": "I'm doing well! How can I help you?"
        },
        {
            "role": "user",
            "content": "Can you tell me what the weather will be in Dallas Texas?"
        }
    ]

chat_completion = client.chat.completions.create(
    messages=messages,
    model=model,
    tools=tools
)

print("Chat completion results:")
print(chat_completion)
messages.append({
    "role": "assistant",
    "tool_calls": chat_completion.choices[0].message.tool_calls
})

# Now, simulate a tool call
def get_current_weather(city: str, state: str, unit: 'str'):
    return "The weather in Dallas, Texas is 85 degrees fahrenheit. It is partly cloudly, with highs in the 90's."

available_tools = {
    "get_current_weather": get_current_weather
}

completion_tool_calls = chat_completion.choices[0].message.tool_calls
for call in completion_tool_calls:
    tool_to_call = available_tools[call.function.name]
    args = json.loads(call.function.arguments)
    result = tool_to_call(**args)
    print(result)
    messages.append({
        "role": "tool",
        "content": result,
        "tool_call_id": call.id,
        "name": call.function.name
    })

print("Sending new chat with messages", messages)
chat_completion_2 = client.chat.completions.create(
    messages=messages,
    model=model,
    tools=tools,
)

print(chat_completion_2)


