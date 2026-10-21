# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm.transformers_utils import model_arch_config_convertor as convertor

GENERATION_ARCH = "Qwen3_5ForConditionalGeneration"
BACKBONE_ARCH = "Qwen3_5Model"
D3_ARCH = "D3ForSequenceClassification"
CLEF_ARCH = "ClefFlashForJointSchema"
D3_CONFIG = {"format_id": "d3-code-readout-v1"}
MODEL_PATH = "test/d3-checkpoint"
REVISION = "test-revision"


@pytest.fixture
def dependencies(monkeypatch):
    exists = Mock(return_value=False)
    read_config = Mock(return_value=None)
    warning = Mock()
    monkeypatch.setattr(convertor, "file_or_path_exists", exists)
    monkeypatch.setattr(convertor, "get_hf_file_to_dict", read_config)
    monkeypatch.setattr(convertor.logger, "warning", warning)

    # Isolate architecture selection from unrelated base-class config parsing.
    with patch.object(
        convertor.ModelArchConfigConvertorBase,
        "__init__",
        autospec=True,
        return_value=None,
    ) as base_init:
        yield SimpleNamespace(
            exists=exists,
            read_config=read_config,
            warning=warning,
            base_init=base_init,
        )


def initialize(dependencies, architecture, model_path=MODEL_PATH):
    config = SimpleNamespace(
        architectures=[architecture], name_or_path=model_path
    )
    text_config = SimpleNamespace()
    instance = convertor.Qwen3_5ModelArchConfigConvertor(
        config, text_config, revision=REVISION
    )
    dependencies.base_init.assert_called_once_with(
        instance, config, text_config, REVISION
    )
    return config


def test_warns_for_d3_generation_override(dependencies):
    dependencies.exists.side_effect = (False, True)
    dependencies.read_config.return_value = D3_CONFIG.copy()

    config = initialize(dependencies, GENERATION_ARCH)

    assert config.architectures == [GENERATION_ARCH]
    assert not hasattr(config, "decision_config")
    assert dependencies.exists.call_args_list == [
        ((MODEL_PATH, "joint_head_config.json", REVISION),),
        ((MODEL_PATH, "decision_config.json", REVISION),),
    ]
    dependencies.read_config.assert_called_once_with(
        "decision_config.json", MODEL_PATH, REVISION
    )
    dependencies.warning.assert_called_once()
    message = dependencies.warning.call_args.args[0]
    assert "D3 code-readout checkpoint" in message
    assert GENERATION_ARCH in message
    assert "override" in message
    assert "remove it" in message


def test_generation_without_d3_config_does_not_warn(dependencies):
    config = initialize(dependencies, GENERATION_ARCH)

    assert config.architectures == [GENERATION_ARCH]
    dependencies.read_config.assert_not_called()
    dependencies.warning.assert_not_called()


@pytest.mark.parametrize(
    "decision_config",
    [None, {}, {"format_id": "other-format"}, [], "invalid-config"],
    ids=["unreadable", "missing-format", "other-format", "list", "string"],
)
def test_unrecognized_decision_config_does_not_warn(
    dependencies, decision_config
):
    dependencies.exists.side_effect = (False, True)
    dependencies.read_config.return_value = decision_config

    config = initialize(dependencies, GENERATION_ARCH)

    assert config.architectures == [GENERATION_ARCH]
    dependencies.read_config.assert_called_once_with(
        "decision_config.json", MODEL_PATH, REVISION
    )
    dependencies.warning.assert_not_called()


def test_native_d3_detection_is_preserved(dependencies):
    dependencies.exists.return_value = True
    dependencies.read_config.return_value = D3_CONFIG.copy()

    config = initialize(dependencies, BACKBONE_ARCH)

    assert config.architectures == [D3_ARCH]
    assert config.decision_config == D3_CONFIG
    dependencies.exists.assert_called_once_with(
        MODEL_PATH, "decision_config.json", REVISION
    )
    dependencies.read_config.assert_called_once_with(
        "decision_config.json", MODEL_PATH, REVISION
    )
    dependencies.warning.assert_not_called()


def test_clef_takes_precedence_over_d3(dependencies):
    dependencies.exists.return_value = True
    joint_config = {"test": "joint-head-config"}
    dependencies.read_config.return_value = joint_config

    config = initialize(dependencies, GENERATION_ARCH)

    assert config.architectures == [CLEF_ARCH]
    assert config.joint_head_config == joint_config
    dependencies.exists.assert_called_once_with(
        MODEL_PATH, "joint_head_config.json", REVISION
    )
    dependencies.read_config.assert_called_once_with(
        "joint_head_config.json", MODEL_PATH, REVISION
    )
    dependencies.warning.assert_not_called()


@pytest.mark.parametrize(
    "architecture", [GENERATION_ARCH, BACKBONE_ARCH]
)
def test_empty_model_path_skips_file_lookup(dependencies, architecture):
    config = initialize(dependencies, architecture, model_path="")

    assert config.architectures == [architecture]
    dependencies.exists.assert_not_called()
    dependencies.read_config.assert_not_called()
    dependencies.warning.assert_not_called()


def test_backbone_without_d3_config_is_preserved(dependencies):
    config = initialize(dependencies, BACKBONE_ARCH)

    assert config.architectures == [BACKBONE_ARCH]
    dependencies.read_config.assert_not_called()
    dependencies.warning.assert_not_called()
