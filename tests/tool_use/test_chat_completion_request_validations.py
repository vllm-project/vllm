import pytest

from vllm.entrypoints.openai.protocol import ChatCompletionRequest


def test_chat_completion_request_with_no_tools():
    # tools key is not present
    request = ChatCompletionRequest.model_validate({
        'messages': [{
            'role': 'user',
            'content': 'Hello'
        }],
        'model':
        'facebook/opt-125m',
    })
    assert request.tool_choice == 'none'

    # tools key is None
    request = ChatCompletionRequest.model_validate({
        'messages': [{
            'role': 'user',
            'content': 'Hello'
        }],
        'model':
        'facebook/opt-125m',
        'tools':
        None
    })
    assert request.tool_choice == 'none'

    # tools key present but empty
    request = ChatCompletionRequest.model_validate({
        'messages': [{
            'role': 'user',
            'content': 'Hello'
        }],
        'model':
        'facebook/opt-125m',
        'tools': []
    })
    assert request.tool_choice == 'none'


def test_chat_completion_request_with_tool_choice_but_no_tools():
    with pytest.raises(ValueError,
                       match="When using `tool_choice`, `tools` must be set."):
        ChatCompletionRequest.model_validate({
            'messages': [{
                'role': 'user',
                'content': 'Hello'
            }],
            'model':
            'facebook/opt-125m',
            'tool_choice':
            'auto'
        })

    with pytest.raises(ValueError,
                       match="When using `tool_choice`, `tools` must be set."):
        ChatCompletionRequest.model_validate({
            'messages': [{
                'role': 'user',
                'content': 'Hello'
            }],
            'model':
            'facebook/opt-125m',
            'tool_choice':
            'auto',
            'tools':
            None
        })
