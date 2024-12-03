from copy import deepcopy
from typing import Any, Dict, List, Optional

from openai.types.chat import (ChatCompletionMessageParam,
                               ChatCompletionToolParam)
from typing_extensions import TypedDict

from tests.utils import VLLM_PATH


class ServerConfig(TypedDict, total=False):
    model: str
    arguments: List[str]
    system_prompt: Optional[str]
    supports_parallel: Optional[bool]
    supports_rocm: Optional[bool]


def patch_system_prompt(messages: List[Dict[str, Any]],
                        system_prompt: str) -> List[Dict[str, Any]]:
    new_messages = deepcopy(messages)
    if new_messages[0]["role"] == "system":
        new_messages[0]["content"] = system_prompt
    else:
        new_messages.insert(0, {"role": "system", "content": system_prompt})
    return new_messages


def ensure_system_prompt(messages: List[Dict[str, Any]],
                         config: ServerConfig) -> List[Dict[str, Any]]:
    prompt = config.get("system_prompt")
    if prompt:
        return patch_system_prompt(messages, prompt)
    else:
        return messages


# universal args for all models go here. also good if you need to test locally
# and change type or KV cache quantization or something.
ARGS: List[str] = ["--enable-auto-tool-choice", "--max-model-len", "1024"]

CONFIGS: Dict[str, ServerConfig] = {
    "hermes": {
        "model":
        "NousResearch/Hermes-3-Llama-3.1-8B",
        "arguments": [
            "--tool-call-parser", "hermes", "--chat-template",
            str(VLLM_PATH / "examples/tool_chat_template_hermes.jinja")
        ],
        "system_prompt":
        "You are a helpful assistant with access to tools. If a tool"
        " that you have would be helpful to answer a user query, "
        "call the tool. Otherwise, answer the user's query directly "
        "without calling a tool. DO NOT CALL A TOOL THAT IS IRRELEVANT "
        "to the user's question - just respond to it normally."
    },
    "llama": {
        "model":
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "arguments": [
            "--tool-call-parser", "llama3_json", "--chat-template",
            str(VLLM_PATH / "examples/tool_chat_template_llama3.1_json.jinja")
        ],
        "supports_parallel":
        False,
    },
    "llama3.2": {
        "model":
        "meta-llama/Llama-3.2-3B-Instruct",
        "arguments": [
            "--tool-call-parser", "llama3_json", "--chat-template",
            str(VLLM_PATH / "examples/tool_chat_template_llama3.2_json.jinja")
        ],
        "supports_parallel":
        False,
    },
    "mistral": {
        "model":
        "mistralai/Mistral-7B-Instruct-v0.3",
        "arguments": [
            "--tool-call-parser", "mistral", "--chat-template",
            str(VLLM_PATH / "examples/tool_chat_template_mistral.jinja"),
            "--ignore-patterns=\"consolidated.safetensors\""
        ],
        "system_prompt":
        "You are a helpful assistant with access to tools. If a tool"
        " that you have would be helpful to answer a user query, "
        "call the tool. Otherwise, answer the user's query directly "
        "without calling a tool. DO NOT CALL A TOOL THAT IS IRRELEVANT "
        "to the user's question - just respond to it normally."
    },
    "granite20b": {
        "model":
        "mbayser/granite-20b-functioncalling-FP8-KV",
        "arguments": [
            "--tool-call-parser", "granite-20b-fc", "--chat-template",
            str(VLLM_PATH /
                "examples/tool_chat_template_granite_20b_fc.jinja"),
            "--max_num_seqs", "1", "--enforce-eager", "--cpu-offload-gb", "20"
        ],
        "supports_parallel":
        False,
        "supports_rocm":
        False,
    },
    "granite8b": {
        "model":
        "ibm-granite/granite-3.0-8b-instruct",
        "arguments": [
            "--tool-call-parser", "granite", "--chat-template",
            str(VLLM_PATH / "examples/tool_chat_template_granite.jinja")
        ],
    },
    "internlm": {
        "model":
        "internlm/internlm2_5-7b-chat",
        "arguments": [
            "--tool-call-parser", "internlm", "--chat-template",
            str(VLLM_PATH /
                "examples/tool_chat_template_internlm2_tool.jinja"),
            "--trust_remote_code"
        ],
        "supports_parallel":
        False,
    },
    "toolACE": {
        "model":
        "Team-ACE/ToolACE-8B",
        "arguments": [
            "--tool-call-parser", "pythonic", "--chat-template",
            str(VLLM_PATH / "examples/tool_chat_template_toolace.jinja")
        ],
        "supports_parallel":
        True,
    },
}

WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type":
                    "string",
                    "description":
                    "The city to find the weather for, "
                    "e.g. 'San Francisco'"
                },
                "state": {
                    "type":
                    "string",
                    "description":
                    "must the two-letter abbreviation for the state "
                    "that the city is in, e.g. 'CA' which would "
                    "mean 'California'"
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"]
                }
            }
        }
    }
}

SEARCH_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name":
        "web_search",
        "description":
        "Search the internet and get a summary of the top "
        "10 webpages. Should only be used if you don't know "
        "the answer to a user query, and the results are likely"
        "to be able to be found with a web search",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type":
                    "string",
                    "description":
                    "The term to use in the search. This should"
                    "ideally be keywords to search for, not a"
                    "natural-language question"
                }
            },
            "required": ["search_term"]
        }
    }
}

MESSAGES_WITHOUT_TOOLS: List[ChatCompletionMessageParam] = [{
    "role":
    "user",
    "content":
    "Hi! How are you?"
}, {
    "role":
    "assistant",
    "content":
    "I'm doing great! How can I assist you?"
}, {
    "role":
    "user",
    "content":
    "Can you tell me a joke please?"
}]

MESSAGES_ASKING_FOR_TOOLS: List[ChatCompletionMessageParam] = [{
    "role":
    "user",
    "content":
    "What is the weather in Dallas, Texas in Fahrenheit?"
}]

MESSAGES_WITH_TOOL_RESPONSE: List[ChatCompletionMessageParam] = [{
    "role":
    "user",
    "content":
    "What is the weather in Dallas, Texas in Fahrenheit?"
}, {
    "role":
    "assistant",
    "tool_calls": [{
        "id": "chatcmpl-tool-03e6481b146e408e9523d9c956696295",
        "type": "function",
        "function": {
            "name":
            WEATHER_TOOL["function"]["name"],
            "arguments":
            '{"city": "Dallas", "state": "TX", '
            '"unit": "fahrenheit"}'
        }
    }]
}, {
    "role":
    "tool",
    "tool_call_id":
    "chatcmpl-tool-03e6481b146e408e9523d9c956696295",
    "content":
    "The weather in Dallas is 98 degrees fahrenheit, with partly"
    "cloudy skies and a low chance of rain."
}]

MESSAGES_ASKING_FOR_PARALLEL_TOOLS: List[ChatCompletionMessageParam] = [{
    "role":
    "user",
    "content":
    "What is the weather in Dallas, Texas and Orlando, Florida in "
    "Fahrenheit?"
}]

MESSAGES_WITH_PARALLEL_TOOL_RESPONSE: List[ChatCompletionMessageParam] = [{
    "role":
    "user",
    "content":
    "What is the weather in Dallas, Texas and Orlando, Florida in "
    "Fahrenheit?"
}, {
    "role":
    "assistant",
    "tool_calls": [{
        "id": "chatcmpl-tool-03e6481b146e408e9523d9c956696295",
        "type": "function",
        "function": {
            "name":
            WEATHER_TOOL["function"]["name"],
            "arguments":
            '{"city": "Dallas", "state": "TX", '
            '"unit": "fahrenheit"}'
        }
    }, {
        "id": "chatcmpl-tool-d027061e1bd21cda48bee7da829c1f5b",
        "type": "function",
        "function": {
            "name":
            WEATHER_TOOL["function"]["name"],
            "arguments":
            '{"city": "Orlando", "state": "Fl", '
            '"unit": "fahrenheit"}'
        }
    }]
}, {
    "role":
    "tool",
    "tool_call_id":
    "chatcmpl-tool-03e6481b146e408e9523d9c956696295",
    "content":
    "The weather in Dallas TX is 98 degrees fahrenheit with mostly "
    "cloudy skies and a chance of rain in the evening."
}, {
    "role":
    "tool",
    "tool_call_id":
    "chatcmpl-tool-d027061e1bd21cda48bee7da829c1f5b",
    "content":
    "The weather in Orlando FL is 78 degrees fahrenheit with clear"
    "skies."
}]
