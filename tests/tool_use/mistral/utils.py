# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from tests.tool_use.utils import ServerConfig

ARGS: list[str] = ["--max-model-len", "1024"]

CONFIGS: dict[str, ServerConfig] = {
    "mistral": {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "arguments": [
            "--tokenizer-mode",
            "mistral",
            "--tool-call-parser",
            "mistral",
            "--enable-auto-tool-choice",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            '--ignore-patterns="consolidated.safetensors"',
        ],
        "system_prompt": "You are a helpful assistant with access to tools. If a tool"
        " that you have would be helpful to answer a user query, "
        "call the tool. Otherwise, answer the user's query directly "
        "without calling a tool. DO NOT CALL A TOOL THAT IS IRRELEVANT "
        "to the user's question - just respond to it normally.",
    },
    "ministral-3b": {
        "model": "mistralai/Ministral-3-3B-Instruct-2512",
        "arguments": [
            "--tokenizer-mode",
            "mistral",
            "--tool-call-parser",
            "mistral",
            "--enable-auto-tool-choice",
            "--enforce-eager",
            "--no-enable-prefix-caching",
        ],
        "system_prompt": "You are a helpful assistant with access to tools. If a tool"
        " that you have would be helpful to answer a user query, "
        "call the tool. Otherwise, answer the user's query directly "
        "without calling a tool. DO NOT CALL A TOOL THAT IS IRRELEVANT "
        "to the user's question - just respond to it normally.",
        "supports_parallel": True,
    },
}
