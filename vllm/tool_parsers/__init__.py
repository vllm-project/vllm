# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

__all__ = ["ToolParser", "ToolParserManager", "get_auto_tool_parser"]


"""
Register a lazy module mapping.

Example:
    ToolParserManager.register_lazy_module(
        name="kimi_k2",
        module_path="vllm.tool_parsers.kimi_k2_parser",
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
    "deepseek_v32": (
        "deepseekv32_tool_parser",
        "DeepSeekV32ToolParser",
    ),
    "ernie45": (
        "ernie45_tool_parser",
        "Ernie45ToolParser",
    ),
    "glm45": (
        "glm4_moe_tool_parser",
        "Glm4MoeModelToolParser",
    ),
    "glm47": (
        "glm47_moe_tool_parser",
        "Glm47MoeModelToolParser",
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
        "Llama3JsonToolParser",
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
        "Qwen3XMLToolParser",
    ),
    "seed_oss": (
        "seed_oss_tool_parser",
        "SeedOssToolParser",
    ),
    "step3": (
        "step3_tool_parser",
        "Step3ToolParser",
    ),
    "step3p5": (
        "step3p5_tool_parser",
        "Step3p5ToolParser",
    ),
    "xlam": (
        "xlam_tool_parser",
        "xLAMToolParser",
    ),
    "gigachat3": (
        "gigachat3_tool_parser",
        "GigaChat3ToolParser",
    ),
    "functiongemma": (
        "functiongemma_tool_parser",
        "FunctionGemmaToolParser",
    ),
}


def register_lazy_tool_parsers():
    for name, (file_name, class_name) in _TOOL_PARSERS_TO_REGISTER.items():
        module_path = f"vllm.tool_parsers.{file_name}"
        ToolParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_tool_parsers()

# Mapping from HuggingFace model_type to the default tool parser name.
# Used when --tool-call-parser=auto to select the correct parser based
# on the model being served.
_MODEL_TYPE_TO_TOOL_PARSER: dict[str, str] = {
    # DeepSeek family
    "deepseek_v3": "deepseek_v3",
    "deepseek_v31": "deepseek_v31",
    "deepseek_v32": "deepseek_v32",
    # GLM family
    "glm_moe_dsa": "glm45",
    "glm4_moe": "glm45",
    "glm4_moe_lite": "glm47",
    # Granite family
    "granite": "granite",
    # InternLM family
    "internlm2": "internlm",
    # Jamba family
    "jamba": "jamba",
    # Mistral family
    "mistral": "mistral",
    # MiniMax family
    "minimax_text_01": "minimax",
    "minimax_m2": "minimax_m2",
    # Hunyuan family
    "hunyuan_moe": "hunyuan_a13b",
    # OLMo family
    "olmo3": "olmo3",
}

# Mapping from model name patterns to tool parser names.
# Used as a fallback when model_type alone is ambiguous (e.g. "llama"
# could map to several parsers). Patterns are matched as substrings
# of the full model name in lowercase. Order matters: the first match
# wins, so more specific patterns must come before generic ones.
_MODEL_NAME_TO_TOOL_PARSER: list[tuple[str, str]] = [
    # Llama 4 models prefer pythonic parser
    ("llama-4", "llama4_pythonic"),
    ("llama4", "llama4_pythonic"),
    # Llama 3.x default to JSON parser
    ("llama-3", "llama3_json"),
    ("llama3", "llama3_json"),
    # Hermes models
    ("hermes-2", "hermes"),
    ("hermes-3", "hermes"),
    ("hermes2", "hermes"),
    ("hermes3", "hermes"),
    # Qwen3-Coder
    ("qwen3-coder", "qwen3_coder"),
    ("qwen3coder", "qwen3_coder"),
    # Kimi K2
    ("kimi-k2", "kimi_k2"),
    ("kimi_k2", "kimi_k2"),
    # FunctionGemma
    ("functiongemma", "functiongemma"),
    # xLAM
    ("xlam", "xlam"),
    # GigaChat
    ("gigachat3", "gigachat3"),
    ("gigachat-3", "gigachat3"),
    # Step models
    ("step-3.5", "step3p5"),
    ("step3p5", "step3p5"),
    ("step-3", "step3"),
    ("step3-", "step3"),
    # Longcat
    ("longcat", "longcat"),
    # Ernie
    ("ernie", "ernie45"),
    # Phi-4 Mini
    ("phi-4-mini", "phi4_mini_json"),
    ("phi4-mini", "phi4_mini_json"),
    # Seed OSS
    ("seed-oss", "seed_oss"),
]


def get_auto_tool_parser(model_type: str | None, model_name: str | None) -> str | None:
    """Resolve the tool parser name automatically from model metadata.

    Tries *model_type* first (exact match), then falls back to substring
    matching on *model_name*.

    Returns:
        The parser name string, or ``None`` if no match is found.
    """
    if model_type and model_type in _MODEL_TYPE_TO_TOOL_PARSER:
        return _MODEL_TYPE_TO_TOOL_PARSER[model_type]

    if model_name:
        lower_name = model_name.lower()
        for pattern, parser_name in _MODEL_NAME_TO_TOOL_PARSER:
            if pattern in lower_name:
                return parser_name

    return None
