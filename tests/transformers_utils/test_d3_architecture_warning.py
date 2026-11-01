# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.transformers_utils import model_arch_config_convertor as convertor

GENERATION_ARCH = "Qwen3_5ForConditionalGeneration"
BACKBONE_ARCH = "Qwen3_5Model"
D3_ARCH = "D3ForSequenceClassification"
CLEF_ARCH = "ClefFlashForJointSchema"
D3_CONFIG = {"format_id": "d3-code-readout-v1"}
WARNING_MARKER = "D3 code-readout checkpoint"


def _checkpoint(tmp_path, files):
    for name, content in files.items():
        (tmp_path / name).write_text(json.dumps(content))
    return str(tmp_path)


def _convert(model_path, architecture):
    hf_config = SimpleNamespace(architectures=[architecture], name_or_path=model_path)
    convertor.Qwen3_5ModelArchConfigConvertor(
        hf_config, SimpleNamespace(), revision=None
    )
    return hf_config


def test_warns_for_d3_generation_override(tmp_path, caplog_vllm):
    model_path = _checkpoint(tmp_path, {"decision_config.json": D3_CONFIG})

    hf_config = _convert(model_path, GENERATION_ARCH)

    assert hf_config.architectures == [GENERATION_ARCH]
    assert not hasattr(hf_config, "decision_config")
    assert WARNING_MARKER in caplog_vllm.text
    assert GENERATION_ARCH in caplog_vllm.text
    assert "remove it" in caplog_vllm.text


def test_generation_without_any_config_does_not_warn(tmp_path, caplog_vllm):
    hf_config = _convert(str(tmp_path), GENERATION_ARCH)

    assert hf_config.architectures == [GENERATION_ARCH]
    assert WARNING_MARKER not in caplog_vllm.text


@pytest.mark.parametrize(
    "decision_config",
    [{}, {"format_id": "other-format"}, [], "invalid-config"],
    ids=["missing-format", "other-format", "list", "string"],
)
def test_unrecognized_decision_config_does_not_warn(
    tmp_path, caplog_vllm, decision_config
):
    model_path = _checkpoint(tmp_path, {"decision_config.json": decision_config})

    hf_config = _convert(model_path, GENERATION_ARCH)

    assert hf_config.architectures == [GENERATION_ARCH]
    assert WARNING_MARKER not in caplog_vllm.text


def test_clef_takes_precedence_over_d3(tmp_path, caplog_vllm):
    joint_config = {"head": "joint"}
    model_path = _checkpoint(
        tmp_path,
        {
            "joint_head_config.json": joint_config,
            "decision_config.json": D3_CONFIG,
        },
    )

    hf_config = _convert(model_path, GENERATION_ARCH)

    assert hf_config.architectures == [CLEF_ARCH]
    assert hf_config.joint_head_config == joint_config
    assert WARNING_MARKER not in caplog_vllm.text


def test_native_d3_detection_is_preserved(tmp_path, caplog_vllm):
    model_path = _checkpoint(tmp_path, {"decision_config.json": D3_CONFIG})

    hf_config = _convert(model_path, BACKBONE_ARCH)

    assert hf_config.architectures == [D3_ARCH]
    assert hf_config.decision_config == D3_CONFIG
    assert WARNING_MARKER not in caplog_vllm.text


def test_empty_model_path_skips_lookup(caplog_vllm):
    hf_config = _convert("", GENERATION_ARCH)

    assert hf_config.architectures == [GENERATION_ARCH]
    assert WARNING_MARKER not in caplog_vllm.text
