# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tool_parsers import get_auto_tool_parser


class TestAutoToolParserModelType:
    """Test auto-detection via model_type (exact match)."""

    @pytest.mark.parametrize(
        "model_type,expected_parser",
        [
            ("deepseek_v3", "deepseek_v3"),
            ("deepseek_v31", "deepseek_v31"),
            ("deepseek_v32", "deepseek_v32"),
            ("glm4_moe", "glm45"),
            ("glm4_moe_lite", "glm47"),
            ("glm_moe_dsa", "glm45"),
            ("granite", "granite"),
            ("internlm2", "internlm"),
            ("jamba", "jamba"),
            ("mistral", "mistral"),
            ("minimax_text_01", "minimax"),
            ("minimax_m2", "minimax_m2"),
            ("hunyuan_moe", "hunyuan_a13b"),
            ("olmo3", "olmo3"),
        ],
    )
    def test_model_type_mapping(self, model_type, expected_parser):
        assert get_auto_tool_parser(model_type, None) == expected_parser

    def test_unknown_model_type_returns_none(self):
        assert get_auto_tool_parser("unknown_model", None) is None


class TestAutoToolParserModelName:
    """Test auto-detection via model name substring matching."""

    @pytest.mark.parametrize(
        "model_name,expected_parser",
        [
            # Llama 4
            ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "llama4_pythonic"),
            ("meta-llama/Llama-4-Maverick-17B-128E", "llama4_pythonic"),
            # Llama 3
            ("meta-llama/Llama-3.1-8B-Instruct", "llama3_json"),
            ("meta-llama/Llama-3.2-1B-Instruct", "llama3_json"),
            ("meta-llama/Llama3-70B-Instruct", "llama3_json"),
            # Hermes
            ("NousResearch/Hermes-3-Llama-3.1-8B", "hermes"),
            ("NousResearch/Hermes-2-Pro-Mistral-7B", "hermes"),
            # Qwen3 Coder
            ("Qwen/Qwen3-Coder-480B-A35B-Instruct", "qwen3_coder"),
            # Kimi K2
            ("moonshotai/Kimi-K2-Instruct", "kimi_k2"),
            # FunctionGemma
            ("google/functiongemma-270m-it", "functiongemma"),
            # xLAM
            ("Salesforce/Llama-xLAM-2-8B-fc-r", "xlam"),
            # Step models
            ("step-3.5-preview", "step3p5"),
            # Longcat
            ("meituan-longcat/LongCat-Flash-Chat", "longcat"),
            # Phi-4 Mini
            ("microsoft/phi-4-mini-instruct", "phi4_mini_json"),
        ],
    )
    def test_model_name_mapping(self, model_name, expected_parser):
        # Pass unknown model_type to force name-based fallback
        assert get_auto_tool_parser("unknown", model_name) == expected_parser

    def test_unknown_model_name_returns_none(self):
        assert get_auto_tool_parser(None, "some/unknown-model") is None

    def test_none_inputs_returns_none(self):
        assert get_auto_tool_parser(None, None) is None


class TestAutoToolParserPrecedence:
    """Test that model_type takes precedence over model_name."""

    def test_model_type_preferred_over_name(self):
        # mistral model_type should win even if name contains "llama"
        result = get_auto_tool_parser("mistral", "some-llama-model")
        assert result == "mistral"

    def test_name_used_when_type_unknown(self):
        # unknown type → fall back to name matching
        result = get_auto_tool_parser("llama", "meta-llama/Llama-4-Scout")
        assert result == "llama4_pythonic"
