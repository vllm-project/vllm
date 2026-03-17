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
    "deepseek-r1:70b": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "description": "DeepSeek R1 distilled Llama 70B",
    },
    "deepseek-v3": {
        "model": "deepseek-ai/DeepSeek-V3",
        "description": "DeepSeek V3",
    },
    "gemma2:2b-it": {
        "model": "google/gemma-2-2b-it",
        "description": "Gemma 2 2B Instruct",
    },
    "gemma2:9b-it": {
        "model": "google/gemma-2-9b-it",
        "description": "Gemma 2 9B Instruct",
    },
    "gemma2:27b-it": {
        "model": "google/gemma-2-27b-it",
        "description": "Gemma 2 27B Instruct",
    },
    "llama3.1:70b-instruct": {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "description": "Meta Llama 3.1 70B Instruct",
    },
    "llama3.1:8b-instruct": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Meta Llama 3.1 8B Instruct",
    },
    "llama3.2:1b-instruct": {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "description": "Meta Llama 3.2 1B Instruct",
    },
    "llama3.2:3b-instruct": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "description": "Meta Llama 3.2 3B Instruct",
    },
    "llama3.3:70b-instruct": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "description": "Meta Llama 3.3 70B Instruct",
    },
    "ministral:8b-instruct": {
        "model": "mistralai/Ministral-8B-Instruct-2410",
        "description": "Ministral 8B Instruct",
    },
    "mistral:7b-instruct": {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral 7B Instruct",
    },
    "mistral-nemo:12b-instruct": {
        "model": "mistralai/Mistral-Nemo-Instruct-2407",
        "description": "Mistral Nemo 12B Instruct",
    },
    "mixtral:8x7b-instruct": {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "description": "Mixtral 8x7B Instruct",
    },
    "phi3.5:mini-instruct": {
        "model": "microsoft/Phi-3.5-mini-instruct",
        "description": "Phi 3.5 Mini Instruct",
    },
    "phi3.5:moe-instruct": {
        "model": "microsoft/Phi-3.5-MoE-instruct",
        "description": "Phi 3.5 MoE Instruct",
    },
    "phi4": {
        "model": "microsoft/phi-4",
        "description": "Phi 4",
    },
    "qwen2.5-coder:1.5b-instruct": {
        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "description": "Qwen 2.5 Coder 1.5B Instruct",
    },
    "qwen2.5-coder:7b-instruct": {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "description": "Qwen 2.5 Coder 7B Instruct",
    },
    "qwen2.5-coder:32b-instruct": {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "description": "Qwen 2.5 Coder 32B Instruct",
    },
    "qwen2.5:0.5b-instruct": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "Qwen 2.5 0.5B Instruct",
    },
    "qwen2.5:1.5b-instruct": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Qwen 2.5 1.5B Instruct",
    },
    "qwen2.5:3b-instruct": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Qwen 2.5 3B Instruct",
    },
    "qwen2.5:7b-instruct": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 2.5 7B Instruct",
    },
    "qwen2.5:14b-instruct": {
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "description": "Qwen 2.5 14B Instruct",
    },
    "qwen2.5:32b-instruct": {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "description": "Qwen 2.5 32B Instruct",
    },
    "qwen2.5:72b-instruct": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "description": "Qwen 2.5 72B Instruct",
    },
    "smollm2:360m-instruct": {
        "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "description": "SmolLM2 360M Instruct",
    },
    "smollm2:1.7b-instruct": {
        "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "description": "SmolLM2 1.7B Instruct",
    },
}
