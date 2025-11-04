# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

__all__ = ["ToolParser", "ToolParserManager"]


"""
Register a lazy module mapping.

Example:
    ToolParserManager.register_lazy_module(
        name="kimi_k2",
        module_path="vllm.entrypoints.openai.tool_parsers.kimi_k2_parser",
        class_name="KimiK2ToolParser",
    )
"""


_TOOL_PARSERS_TO_REGISTER = {
    "deepseek_v3": (  # name
        "deepseekv3_tool_parser",  # filename
        "DeepSeekV3ToolParser",  # class_name
    ),
    "deepseek_v31": (
        "deepseekv31_tool_parser",
        "DeepSeekV31ToolParser",
    ),
    "ernie45": (
        "ernie45_tool_parser",
        "Ernie45ToolParser",
    ),
    "glm45": (
        "glm4_moe_tool_parser",
        "Glm4MoeModelToolParser",
    ),
    "granite-20b-fc": (
        "granite_20b_fc_tool_parser",
        "Granite20bFCToolParser",
    ),
    "granite": (
        "granite_tool_parser",
        "GraniteToolParser",
    ),
    "hermes": (
        "hermes_tool_parser",
        "Hermes2ProToolParser",
    ),
    "hunyuan_a13b": (
        "hunyuan_a13b_tool_parser",
        "HunyuanA13BToolParser",
    ),
    "internlm": (
        "internlm2_tool_parser",
        "Internlm2ToolParser",
    ),
    "jamba": (
        "jamba_tool_parser",
        "JambaToolParser",
    ),
    "kimi_k2": (
        "kimi_k2_tool_parser",
        "KimiK2ToolParser",
    ),
    "llama3_json": (
        "llama_tool_parser",
        "Llama3JsonToolParser",
    ),
    "llama4_json": (
        "llama_tool_parser",
        "Llama4JsonToolParser",
    ),
    "llama4_pythonic": (
        "llama4_pythonic_tool_parser",
        "Llama4PythonicToolParser",
    ),
    "longcat": (
        "longcat_tool_parser",
        "LongcatFlashToolParser",
    ),
    "minimax_m2": (
        "minimax_m2_tool_parser",
        "MinimaxM2ToolParser",
    ),
    "minimax": (
        "minimax_tool_parser",
        "MinimaxToolParser",
    ),
    "mistral": (
        "mistral_tool_parser",
        "MistralToolParser",
    ),
    "olmo3": (
        "olmo3_tool_parser",
        "Olmo3PythonicToolParser",
    ),
    "openai": (
        "openai_tool_parser",
        "OpenAIToolParser",
    ),
    "phi4_mini_json": (
        "phi4mini_tool_parser",
        "Phi4MiniJsonToolParser",
    ),
    "pythonic": (
        "pythonic_tool_parser",
        "PythonicToolParser",
    ),
    "qwen3_coder": (
        "qwen3coder_tool_parser",
        "Qwen3CoderToolParser",
    ),
    "qwen3_xml": (
        "qwen3xml_tool_parser",
        "Qwen3XmlToolParser",
    ),
    "seed_oss": (
        "seed_oss_tool_parser",
        "SeedOsSToolParser",
    ),
    "step3": (
        "step3_tool_parser",
        "Step3ToolParser",
    ),
    "xlam": (
        "xlam_tool_parser",
        "xLAMToolParser",
    ),
}


def register_lazy_tool_parsers():
    for name, (file_name, class_name) in _TOOL_PARSERS_TO_REGISTER.items():
        module_path = f"vllm.entrypoints.openai.tool_parsers.{file_name}"
        ToolParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_tool_parsers()
