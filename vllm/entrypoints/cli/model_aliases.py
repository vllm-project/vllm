# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Built-in short aliases for local-runtime CLI commands."""

BUILTIN_MODEL_ALIASES: dict[str, dict[str, str]] = {
    "deepseek-r1:1.5b": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "description": "DeepSeek R1 distilled Qwen 1.5B",
    },
    "deepseek-r1:7b": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "description": "DeepSeek R1 distilled Qwen 7B",
    },
    "deepseek-r1:8b": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "description": "DeepSeek R1 distilled Llama 8B",
    },
    "deepseek-r1:14b": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "description": "DeepSeek R1 distilled Qwen 14B",
    },
    "deepseek-r1:32b": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "description": "DeepSeek R1 distilled Qwen 32B",
    },
    "llama3.1:8b-instruct": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Meta Llama 3.1 8B Instruct",
    },
    "qwen2.5:7b-instruct": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 2.5 7B Instruct",
    },
    "qwen2.5:14b-instruct": {
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "description": "Qwen 2.5 14B Instruct",
    },
}
