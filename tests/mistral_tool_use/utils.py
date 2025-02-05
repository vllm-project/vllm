# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


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
ARGS: List[str] = ["--max-model-len", "1024"]

CONFIGS: Dict[str, ServerConfig] = {
    "mistral": {
        "model":
        "mistralai/Mistral-7B-Instruct-v0.3",
        "arguments": [
            "--tokenizer-mode", "mistral",
            "--ignore-patterns=\"consolidated.safetensors\""
        ],
        "system_prompt":
        "You are a helpful assistant with access to tools. If a tool"
        " that you have would be helpful to answer a user query, "
        "call the tool. Otherwise, answer the user's query directly "
        "without calling a tool. DO NOT CALL A TOOL THAT IS IRRELEVANT "
        "to the user's question - just respond to it normally."
    },
}
