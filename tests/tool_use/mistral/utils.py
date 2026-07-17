# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from tests.tool_use.utils import ServerConfig

# Shared across every model: cap context to 32k and limit concurrency to keep
# VRAM usage modest during CI/local runs.
ARGS: list[str] = [
    "--max-model-len",
    "32768",
    "--max-num-seqs",
    "16",
    "--gpu-memory-utilization",
    "0.5",
]

# Base args for grammar-capable Mistral models (v11+). The reasoning parser is
# only added for models that actually reason (see _MISTRAL_REASONING_ARGS).
_MISTRAL_ARGS: list[str] = [
    "--tokenizer-mode",
    "mistral",
    "--tool-call-parser",
    "mistral",
    "--enable-auto-tool-choice",
    "--enforce-eager",
    "--no-enable-prefix-caching",
]

# Reasoning-capable models additionally enable the reasoning parser so that
# reasoning_content is populated.
_MISTRAL_REASONING_ARGS: list[str] = [
    *_MISTRAL_ARGS,
    "--reasoning-parser",
    "mistral",
]

# Mistral-7B-v0.3 is pre-v11: no grammar, no reasoning. It keeps the tool
# parser but omits the reasoning parser, and ignores the consolidated weights.
_MISTRAL_7B_ARGS: list[str] = [
    *_MISTRAL_ARGS,
    '--ignore-patterns="consolidated.safetensors"',
]

# Mistral-Small-4 (119B MoE) needs tensor parallelism. Attention backend is
# left to auto-selection (FLASH_ATTN_MLA rejects this model's MLA dimensions).
_MISTRAL_SMALL_4_ARGS: list[str] = [
    *_MISTRAL_REASONING_ARGS,
    "--tensor-parallel-size",
    "2",
]

_SYSTEM_PROMPT: str = (
    "You are a helpful assistant with access to tools. If a tool"
    " that you have would be helpful to answer a user query, "
    "call the tool. Otherwise, answer the user's query directly "
    "without calling a tool. DO NOT CALL A TOOL THAT IS IRRELEVANT "
    "to the user's question - just respond to it normally."
)

# Some reasoning Mistral checkpoints emit [THINK]...[/THINK] only when the system
# prompt itself contains a real think chunk.
_REASONING_SYSTEM_PROMPT: list[dict[str, Any]] = [
    {
        "type": "text",
        "text": (
            "First draft your thinking process (inner monologue) until you "
            "arrive at a response. Write both your thoughts and the response in "
            "the same language as the input.\n\n"
            "Your thinking process must follow the template below:"
        ),
    },
    {
        "type": "thinking",
        "thinking": (
            "Your thoughts or/and draft, like working through an exercise on "
            "scratch paper. Be as casual and as long as you want until you are "
            "confident to generate the response to the user."
        ),
        "closed": True,
    },
    {
        "type": "text",
        "text": "Here, provide a self-contained response.\n\n",
    },
]

CONFIGS: dict[str, ServerConfig] = {
    "mistral": {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "arguments": _MISTRAL_7B_ARGS,
        "system_prompt": _SYSTEM_PROMPT,
        "reasoning_mode": "none",
        "supports_grammar": False,
    },
    "ministral-3b": {
        "model": "mistralai/Ministral-3-3B-Instruct-2512",
        "arguments": _MISTRAL_ARGS,
        "system_prompt": _SYSTEM_PROMPT,
        "reasoning_mode": "none",
        "supports_grammar": True,
        "supports_parallel": True,
    },
    "ministral-8b-reasoning": {
        "model": "mistralai/Ministral-3-8B-reasoning-2512",
        "arguments": _MISTRAL_REASONING_ARGS,
        "system_prompt": _REASONING_SYSTEM_PROMPT,
        "reasoning_mode": "intrinsic",
        "supports_grammar": True,
        "supports_parallel": True,
    },
    "mistral-small-4": {
        "model": "mistralai/Mistral-Small-4-119B-2603",
        "arguments": _MISTRAL_SMALL_4_ARGS,
        "system_prompt": _SYSTEM_PROMPT,
        "reasoning_mode": "effort",
        "supports_grammar": True,
        "supports_parallel": True,
        # 119B fp8 on TP=2 needs longer than the default to load/quantize.
        "startup_timeout": 1800,
    },
}
