import json

from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam, Function)

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    CustomChatCompletionMessageParam, FixedChatCompletionAssistantMessageParam,
    OpenAIChatCompletionFunctionMessageParam,
    OpenAIChatCompletionSystemMessageParam,
    OpenAIChatCompletionToolMessageParam, OpenAIChatCompletionUserMessageParam,
    parse_chat_messages)
from vllm.transformers_utils.tokenizer import get_tokenizer

MODEL_NAME = "openai-community/gpt2"
MODEL_CONFIG = ModelConfig(
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    tokenizer_mode="auto",
    trust_remote_code=False,
    dtype="float32",
    seed=0,
)
TOKENIZER = get_tokenizer(tokenizer_name=MODEL_NAME)


def test_parse_system_message():
    messages = [
        OpenAIChatCompletionSystemMessageParam(
            role='system', content="you are a helpful assistant")
    ]
    conversation, _ = parse_chat_messages(messages=messages,
                                          model_config=MODEL_CONFIG,
                                          tokenizer=TOKENIZER)

    assert conversation == [{
        'role': 'system',
        'content': "you are a helpful assistant"
    }]


def test_parse_user_message():
    messages = [
        OpenAIChatCompletionUserMessageParam(role='user',
                                             content='what is 1+1?')
    ]
    conversation, _ = parse_chat_messages(messages=messages,
                                          model_config=MODEL_CONFIG,
                                          tokenizer=TOKENIZER)

    assert conversation == [{'role': 'user', 'content': 'what is 1+1?'}]


def test_parse_user_message_with_name():
    messages = [
        OpenAIChatCompletionUserMessageParam(role='user',
                                             content='what is 1+1?',
                                             name='Alice')
    ]
    conversation, _ = parse_chat_messages(messages=messages,
                                          model_config=MODEL_CONFIG,
                                          tokenizer=TOKENIZER)

    assert conversation == [{
        'role': 'user',
        'content': 'what is 1+1?',
        'name': 'Alice'
    }]


def test_parse_assistant_message():
    messages = [
        FixedChatCompletionAssistantMessageParam(role='assistant',
                                                 content='Hi there!')
    ]
    conversation, _ = parse_chat_messages(messages=messages,
                                          model_config=MODEL_CONFIG,
                                          tokenizer=TOKENIZER)

    assert conversation == [{'role': 'assistant', 'content': 'Hi there!'}]


def test_parse_assistant_message_with_tool_calls():
    tool_calls = [
        ChatCompletionMessageToolCallParam(
            id="call_1",
            type="function",
            function=Function(name="get_capital",
                              arguments=json.dumps({"country": "France"}))),
        ChatCompletionMessageToolCallParam(
            id="call_2",
            type="function",
            function=Function(name="get_capital",
                              arguments=json.dumps({"country": "Germany"}))),
        ChatCompletionMessageToolCallParam(
            id="call_3",
            type="function",
            function=Function(name="get_capital",
                              arguments=json.dumps({"country": "Italy"})))
    ]
    messages = [
        FixedChatCompletionAssistantMessageParam(role='assistant',
                                                 content='Let me check',
                                                 tool_calls=tool_calls)
    ]
    conversation, _ = parse_chat_messages(messages=messages,
                                          model_config=MODEL_CONFIG,
                                          tokenizer=TOKENIZER)

    assert conversation == [{
        'role':
        'assistant',
        'content':
        'Let me check',
        'tool_calls': [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_capital",
                "arguments": json.dumps({"country": "France"})
            }
        }, {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "get_capital",
                "arguments": json.dumps({"country": "Germany"})
            }
        }, {
            "id": "call_3",
            "type": "function",
            "function": {
                "name": "get_capital",
                "arguments": json.dumps({"country": "Italy"})
            }
        }]
    }]


def test_parse_tool_message():
    messages = [
        OpenAIChatCompletionToolMessageParam(role='tool',
                                             content='Paris',
                                             tool_call_id='call_1')
    ]
    conversation, _ = parse_chat_messages(messages=messages,
                                          model_config=MODEL_CONFIG,
                                          tokenizer=TOKENIZER)

    assert conversation == [{
        'role': 'tool',
        'content': 'Paris',
        'tool_call_id': 'call_1'
    }]


def test_parse_function_message():
    messages = [
        OpenAIChatCompletionFunctionMessageParam(role='function',
                                                 content='Paris',
                                                 name='get_capital')
    ]
    conversation, _ = parse_chat_messages(messages=messages,
                                          model_config=MODEL_CONFIG,
                                          tokenizer=TOKENIZER)

    assert conversation == [{
        'role': 'function',
        'content': 'Paris',
        'name': 'get_capital'
    }]


def test_parse_custom_completion_message():
    messages = [
        CustomChatCompletionMessageParam(role='custom_role',
                                         content='what is 1+1?',
                                         name='Alice')
    ]
    conversation, _ = parse_chat_messages(messages=messages,
                                          model_config=MODEL_CONFIG,
                                          tokenizer=TOKENIZER)

    assert conversation == [{
        'role': 'custom_role',
        'content': 'what is 1+1?',
        'name': 'Alice'
    }]


def test_parse_full_conversation():
    messages = [
        OpenAIChatCompletionSystemMessageParam(
            role='system', content="you are a helpful assistant"),
        FixedChatCompletionAssistantMessageParam(
            role='assistant', content='How can I help you?'),
        OpenAIChatCompletionUserMessageParam(
            role='user',
            content='What is the capital city of France',
            name='Alice'),
        FixedChatCompletionAssistantMessageParam(
            role='assistant',
            content='Let me check',
            tool_calls=[
                ChatCompletionMessageToolCallParam(
                    id="call_1",
                    type="function",
                    function=Function(name="get_capital",
                                      arguments=json.dumps(
                                          {"country": "France"})))
            ]),
        OpenAIChatCompletionToolMessageParam(role='tool',
                                             content='Paris',
                                             tool_call_id='call_1'),
        OpenAIChatCompletionUserMessageParam(role='user',
                                             content='Nice! How did you know?',
                                             name='Bob'),
        FixedChatCompletionAssistantMessageParam(
            role='assistant', content='I used the `get_capital` function')
    ]
    conversation, _ = parse_chat_messages(messages=messages,
                                          model_config=MODEL_CONFIG,
                                          tokenizer=TOKENIZER)

    assert conversation == [
        {
            'role': 'system',
            'content': 'you are a helpful assistant'
        },
        {
            'role': 'assistant',
            'content': 'How can I help you?'
        },
        {
            'role': 'user',
            'content': 'What is the capital city of France',
            'name': 'Alice'
        },
        {
            'role':
            'assistant',
            'content':
            'Let me check',
            'tool_calls': [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_capital",
                    "arguments": json.dumps({"country": "France"})
                }
            }]
        },
        {
            'role': 'tool',
            'content': 'Paris',
            'tool_call_id': 'call_1'
        },
        {
            'role': 'user',
            'content': 'Nice! How did you know?',
            'name': 'Bob'
        },
        {
            'role': 'assistant',
            'content': 'I used the `get_capital` function'
        },
    ]
