"""
vllm serve /data/woosuk/os-mini-weights/pytorch-rc-20b --tokenizer /data/xmo/os-mini/models/hf-converted --enforce-eager
"""
import time
from openai import OpenAI
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=False, choices=["gpt-4.1", "o4-mini"])
parser.add_argument("--port", type=int, required=False, default=8000)
args = parser.parse_args()

MODEL = args.model
if MODEL is None:
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{args.port}/v1"
else:
    openai_api_key = None
    openai_api_base = None

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)


def test_basic():
    response = client.responses.create(
        model=MODEL,
        input="What is 13 * 24?",
        # max_output_tokens=10,
    )
    print(response)


def test_basic_with_instructions():
    response = client.responses.create(
        model=MODEL,
        input="What is 13 * 24?",
        instructions="Respond in Korean.",
    )
    print(response)


def test_basic_with_reasoning_effort():
    response = client.responses.create(
        model=MODEL,
        input="What is the capital of South Korea?",
        reasoning={"effort": "low"},
    )
    print(response)


def test_chat():
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": "Respond in Korean."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "What is 13 * 24? Explain your answer."},
        ],
    )
    print(response)


def test_chat_with_input_type():
    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "What is 13*24?"}],
            },
        ],
    )
    print(response)


def test_structured_output():
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "calendar_event",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "date": {"type": "string"},
                        "participants": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "date", "participants"],
                    "additionalProperties": False,
                },
                "description": "A calendar event.",
                "strict": True,
            }
        },
    )
    print(response)


def test_structured_output_with_parse():

    from pydantic import BaseModel
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    response = client.responses.parse(
        model=MODEL,
        input="Alice and Bob are going to a science fair on Friday",
        instructions="Extract the event information",
        text_format=CalendarEvent,
    )
    print(response)


def test_store():
    for store in [True, False]:
        response = client.responses.create(
            model=MODEL,
            input="What is 13 * 24?",
            store=store,
        )
        print(response)

        response = client.responses.retrieve(response.id)
        print(response)


def test_background():
    response = client.responses.create(
        model=MODEL,
        input="What is 13 * 24?",
        background=True,
    )
    print(response)

    while True:
        response = client.responses.retrieve(response.id)
        if response.status == "completed":
            break
        time.sleep(1)
    print(response)


def test_background_cancel():
    response = client.responses.create(
        model=MODEL,
        input="Write a long story about a cat.",
        background=True,
    )
    print(response)
    time.sleep(1)
    response = client.responses.cancel(response.id)
    print(response)


def test_stateful_multi_turn():
    response1 = client.responses.create(
        model=MODEL,
        input="What is 13 * 24?",
    )
    print(response1)

    response2 = client.responses.create(
        model=MODEL,
        input="What if I increase both numbers by 1?",
        previous_response_id=response1.id,
    )
    print(response2)

    response3 = client.responses.create(
        model=MODEL,
        input="Divide the result by 2.",
        previous_response_id=response2.id,
    )
    print(response3)


def test_streaming():
    response = client.responses.create(
        model=MODEL,
        input="What is 13 * 24? Explain your answer.",
        stream=True,
    )

    for event in response:
        if "text.delta" in event.type:
            print(event.delta, end="", flush=True)
    print()


def test_web_search():
    response = client.responses.create(
        model=MODEL,
        input="Who is the president of South Korea as of now?",
        tools=[{"type": "web_search_preview"}],
    )
    print(response)


def test_code_interpreter():
    response = client.responses.create(
        model=MODEL,
        input="Multiply 643258029438.6132 * 23516705917230.84279 using Python.",
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    )
    print(response)


def test_function_calling():
    tools = [
        {
            "type": "function",
            "name": "send_email",
            "description": "Send an email to a given recipient with a subject and message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "The recipient email address."},
                    "subject": {"type": "string", "description": "Email subject line."},
                    "body": {"type": "string", "description": "Body of the email message."},
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    ]

    response = client.responses.create(
        model=MODEL,
        input="Can you send an email to alice@example.com and bob@example.com saying hi?",
        tools=tools,
    )
    print(response)
    print("Output:")
    for out in response.output:
        print(out)


if __name__ == "__main__":
    # 1. Stateless & Non-streaming tests:
    # test_basic()
    # test_basic_with_instructions()
    # test_basic_with_reasoning_effort()
    # test_chat()
    # test_chat_with_input_type()
    # test_structured_output()
    # test_structured_output_with_parse()

    # 2. Stateful & Non-streaming tests:
    # test_store()
    # test_background()
    # test_background_cancel()
    # test_stateful_multi_turn()

    # 3. Streaming tests:
    # test_streaming()

    # # 4. Tool tests:
    # test_web_search()
    test_code_interpreter()
    # test_function_calling()
