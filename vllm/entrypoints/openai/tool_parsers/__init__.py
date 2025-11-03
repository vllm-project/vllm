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
        module_path=f"{_TOOL_PARSER_LIB_PATH}.kimi_k2_parser",
        class_name="KimiK2ToolParser",
    )
"""

_TOOL_PARSER_LIB_PATH = "vllm.entrypoints.openai.tool_parsers"

_TOOL_PARSERS_TO_REGISTER = {
    "deepseek_v3": (
        f"{_TOOL_PARSER_LIB_PATH}.deepseekv3_tool_parser",
        "DeepSeekV3ToolParser",
    ),
    "deepseek_v31": (
        f"{_TOOL_PARSER_LIB_PATH}.deepseekv31_tool_parser",
        "DeepSeekV31ToolParser",
    ),
    "ernie45": (
        f"{_TOOL_PARSER_LIB_PATH}.ernie45_tool_parser",
        "Ernie45ToolParser",
    ),
    "glm45": (
        f"{_TOOL_PARSER_LIB_PATH}.glm4_moe_tool_parser",
        "Glm4MoeModelToolParser",
    ),
    "granite-20b-fc": (
        f"{_TOOL_PARSER_LIB_PATH}.granite_20b_fc_tool_parser",
        "Granite20bFCToolParser",
    ),
    "granite": (
        f"{_TOOL_PARSER_LIB_PATH}.granite_tool_parser",
        "GraniteToolParser",
    ),
    "hermes": (
        f"{_TOOL_PARSER_LIB_PATH}.hermes_tool_parser",
        "Hermes2ProToolParser",
    ),
    "hunyuan_a13b": (
        f"{_TOOL_PARSER_LIB_PATH}.hunyuan_a13b_tool_parser",
        "HunyuanA13BToolParser",
    ),
    "internlm": (
        f"{_TOOL_PARSER_LIB_PATH}.internlm2_tool_parser",
        "Internlm2ToolParser",
    ),
    "jamba": (
        f"{_TOOL_PARSER_LIB_PATH}.jamba_tool_parser",
        "JambaToolParser",
    ),
    "kimi_k2": (
        f"{_TOOL_PARSER_LIB_PATH}.kimi_k2_tool_parser",
        "KimiK2ToolParser",
    ),
    "llama3_json": (
        f"{_TOOL_PARSER_LIB_PATH}.llama_tool_parser",
        "Llama3JsonToolParser",
    ),
    "llama4_json": (
        f"{_TOOL_PARSER_LIB_PATH}.llama_tool_parser",
        "Llama4JsonToolParser",
    ),
    "llama4_pythonic": (
        f"{_TOOL_PARSER_LIB_PATH}.llama4_pythonic_tool_parser",
        "Llama4PythonicToolParser",
    ),
    "longcat": (
        f"{_TOOL_PARSER_LIB_PATH}.longcat_tool_parser",
        "LongcatFlashToolParser",
    ),
    "minimax_m2": (
        f"{_TOOL_PARSER_LIB_PATH}.minimax_m2_tool_parser",
        "MinimaxM2ToolParser",
    ),
    "minimax": (
        f"{_TOOL_PARSER_LIB_PATH}.minimax_tool_parser",
        "MinimaxToolParser",
    ),
    "mistral": (
        f"{_TOOL_PARSER_LIB_PATH}.mistral_tool_parser",
        "MistralToolParser",
    ),
    "olmo3": (
        f"{_TOOL_PARSER_LIB_PATH}.olmo3_tool_parser",
        "Olmo3PythonicToolParser",
    ),
    "openai": (
        f"{_TOOL_PARSER_LIB_PATH}.openai_tool_parser",
        "OpenAIToolParser",
    ),
    "phi4_mini_json": (
        f"{_TOOL_PARSER_LIB_PATH}.phi4mini_tool_parser",
        "Phi4MiniJsonToolParser",
    ),
    "pythonic": (
        f"{_TOOL_PARSER_LIB_PATH}.pythonic_tool_parser",
        "PythonicToolParser",
    ),
    "qwen3_coder": (
        f"{_TOOL_PARSER_LIB_PATH}.qwen3coder_tool_parser",
        "Qwen3CoderToolParser",
    ),
    "qwen3_xml": (
        f"{_TOOL_PARSER_LIB_PATH}.qwen3xml_tool_parser",
        "Qwen3XmlToolParser",
    ),
    "seed_oss": (
        f"{_TOOL_PARSER_LIB_PATH}.seed_oss_tool_parser",
        "SeedOsSToolParser",
    ),
    "step3": (
        f"{_TOOL_PARSER_LIB_PATH}.step3_tool_parser",
        "Step3ToolParser",
    ),
    "xlam": (
        f"{_TOOL_PARSER_LIB_PATH}.xlam_tool_parser",
        "xLAMToolParser",
    ),
}


def register_lazy_tool_parsers():
    for name, (module_path, class_name) in _TOOL_PARSERS_TO_REGISTER.items():
        ToolParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_tool_parsers()
