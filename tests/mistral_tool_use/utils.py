# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from typing_extensions import TypedDict


class ServerConfig(TypedDict, total=False):
    model: str
    arguments: list[str]
    system_prompt: Optional[str]
    supports_parallel: Optional[bool]
    supports_rocm: Optional[bool]


ARGS: list[str] = ["--max-model-len", "1024"]

CONFIGS: dict[str, ServerConfig] = {
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
