# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.abs_reasoning_parsers import ReasoningParser, ReasoningParserManager

__all__ = [
    "ReasoningParser",
    "ReasoningParserManager",
]
"""
Register a lazy module mapping.

Example:
    ReasoningParserManager.register_lazy_module(
        name="qwen3",
        module_path="vllm.reasoning.qwen3_reasoning_parser",
        class_name="Qwen3ReasoningParser",
    )
"""


_REASONING_PARSERS_TO_REGISTER = {
    "deepseek_r1": (  # name
        "deepseek_r1_reasoning_parser",  # filename
        "DeepSeekR1ReasoningParser",  # class_name
    ),
    "deepseek_v3": (
        "deepseek_v3_reasoning_parser",
        "DeepSeekV3ReasoningParser",
    ),
    "ernie45": (
        "ernie45_reasoning_parser",
        "Ernie45ReasoningParser",
    ),
    "glm45": (
        "glm4_moe_reasoning_parser",
        "Glm4MoeModelReasoningParser",
    ),
    "openai_gptoss": (
        "gptoss_reasoning_parser",
        "GptOssReasoningParser",
    ),
    "granite": (
        "granite_reasoning_parser",
        "GraniteReasoningParser",
    ),
    "hunyuan_a13b": (
        "hunyuan_a13b_reasoning_parser",
        "HunyuanA13BReasoningParser",
    ),
    "kimi_k2": (
        "deepseek_r1_reasoning_parser",
        "DeepSeekR1ReasoningParser",
    ),
    "minimax_m2": (
        "minimax_m2_reasoning_parser",
        "MiniMaxM2ReasoningParser",
    ),
    "minimax_m2_append_think": (
        "minimax_m2_reasoning_parser",
        "MiniMaxM2AppendThinkReasoningParser",
    ),
    "mistral": (
        "mistral_reasoning_parser",
        "MistralReasoningParser",
    ),
    "olmo3": (
        "olmo3_reasoning_parser",
        "Olmo3ReasoningParser",
    ),
    "qwen3": (
        "qwen3_reasoning_parser",
        "Qwen3ReasoningParser",
    ),
    "seed_oss": (
        "seedoss_reasoning_parser",
        "SeedOSSReasoningParser",
    ),
    "step3": (
        "step3_reasoning_parser",
        "Step3ReasoningParser",
    ),
}


def register_lazy_reasoning_parsers():
    for name, (file_name, class_name) in _REASONING_PARSERS_TO_REGISTER.items():
        module_path = f"vllm.reasoning.{file_name}"
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_reasoning_parsers()
