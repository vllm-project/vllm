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
    "deepseek_v3": (
        "vllm.entrypoints.openai.tool_parsers.deepseek_v3_tool_parser",
        "DeepSeekV3ToolParser",
    ),
    "deepseek_v31": (
        "vllm.entrypoints.openai.tool_parsers.deepseek_v31_tool_parser",
        "DeepSeekV31ToolParser",
    ),
    "ernie45": (
        "vllm.entrypoints.openai.tool_parsers.ernie45_tool_parser",
        "Ernie45ToolParser",
    ),
    "glm45": (
        "vllm.entrypoints.openai.tool_parsers.glm4_moe_tool_parser",
        "Glm4MoeModelToolParser",
    ),
    "granite-20b-fc": (
        "vllm.entrypoints.openai.tool_parsers.granite_20b_fc_tool_parser",
        "Granite20bFCToolParser",
    ),
    "granite": (
        "vllm.entrypoints.openai.tool_parsers.granite_tool_parser",
        "GraniteToolParser",
    ),
    "hermes": (
        "vllm.entrypoints.openai.tool_parsers.hermes_tool_parser",
        "Hermes2ProToolParser",
    ),
    "hunyuan_a13b": (
        "vllm.entrypoints.openai.tool_parsers.hunyuan_a13b_tool_parser",
        "HunyuanA13BToolParser",
    ),
    "internlm": (
        "vllm.entrypoints.openai.tool_parsers.internlm2_tool_parser",
        "Internlm2ToolParser",
    ),
    "jamba": (
        "vllm.entrypoints.openai.tool_parsers.jamba_tool_parser",
        "JambaToolParser",
    ),
    "kimi_k2": (
        "vllm.entrypoints.openai.tool_parsers.kimi_k2_parser",
        "KimiK2ToolParser",
    ),
    "llama3_json": (
        "vllm.entrypoints.openai.tool_parsers.llama_tool_parser",
        "Llama3JsonToolParser",
    ),
    "llama4_json": (
        "vllm.entrypoints.openai.tool_parsers.llama_tool_parser",
        "Llama3JsonToolParser",
    ),
    "llama4_pythonic": (
        "vllm.entrypoints.openai.tool_parsers.llama4_pythonic_tool_parser",
        "Llama4PythonicToolParser",
    ),
    "longcat": (
        "vllm.entrypoints.openai.tool_parsers.longcat_tool_parser",
        "LongcatFlashToolParser",
    ),
    "minimax_m2": (
        "vllm.entrypoints.openai.tool_parsers.minimax_m2_tool_parser",
        "MinimaxM2ToolParser",
    ),
    "minimax": (
        "vllm.entrypoints.openai.tool_parsers.minimax_tool_parser",
        "MinimaxToolParser",
    ),
    "mistral": (
        "vllm.entrypoints.openai.tool_parsers.mistral_tool_parser",
        "MistralToolParser",
    ),
    "olmo3": (
        "vllm.entrypoints.openai.tool_parsers.olmo3_tool_parser",
        "Olmo3PythonicToolParser",
    ),
    "openai": (
        "vllm.entrypoints.openai.tool_parsers.openai_tool_parser",
        "OpenAIToolParser",
    ),
    "phi4_mini_json": (
        "vllm.entrypoints.openai.tool_parsers.phi4mini_tool_parser",
        "Phi4MiniJsonToolParser",
    ),
    "pythonic": (
        "vllm.entrypoints.openai.tool_parsers.pythonic_tool_parser",
        "PythonicToolParser",
    ),
    "qwen3_coder": (
        "vllm.entrypoints.openai.tool_parsers.qwen3_coder_tool_parser",
        "Qwen3CoderToolParser",
    ),
    "qwen3_xml": (
        "vllm.entrypoints.openai.tool_parsers.qwen3_xml_tool_parser",
        "Qwen3XmlToolParser",
    ),
    "seed_oss": (
        "vllm.entrypoints.openai.tool_parsers.seed_oss_tool_parser",
        "SeedOsSToolParser",
    ),
    "step3": (
        "vllm.entrypoints.openai.tool_parsers.step3_tool_parser",
        "Step3ToolParser",
    ),
    "xlam": (
        "vllm.entrypoints.openai.tool_parsers.xlam_tool_parser",
        "xLAMToolParser",
    ),
}


def register_lazy_tool_parsers():
    for name, (module_path, class_name) in _TOOL_PARSERS_TO_REGISTER.items():
        ToolParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_tool_parsers()
