# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest
import torch.nn as nn

from vllm.config import in_draft_model
from vllm.config.compilation import CompilationMode
from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
    is_mtp_completeness_check_enabled,
)
from vllm.model_executor.model_loader.utils import initialize_model


def test_disable_mtp_completeness_check_is_scoped():
    assert is_mtp_completeness_check_enabled()

    with pytest.raises(RuntimeError), disable_mtp_completeness_check():
        assert not is_mtp_completeness_check_enabled()
        raise RuntimeError

    assert is_mtp_completeness_check_enabled()


class _RecordingModel(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.built_in_draft = in_draft_model()

    def forward(self) -> bool:
        return in_draft_model()


@pytest.mark.parametrize("load", ["target", "draft", "ngram"])
def test_initialize_model_scopes_only_draft_models(load: str):
    """Draft models are built and run inside `draft_model_scope()` so opt-in
    lower-precision optimizations skip them; the target never is, including
    when ngram aliases the target config as the draft config."""
    target = types.SimpleNamespace(model="target")
    draft = target if load == "ngram" else types.SimpleNamespace(model="draft")
    vllm_config = types.SimpleNamespace(
        quant_config=None,
        model_config=target,
        speculative_config=types.SimpleNamespace(
            target_model_config=target, draft_model_config=draft
        ),
        compilation_config=types.SimpleNamespace(
            mode=CompilationMode.NONE, custom_op_log_check=lambda: None
        ),
    )

    model = initialize_model(
        vllm_config,
        model_class=_RecordingModel,
        model_config=target if load == "target" else draft,
    )

    expected = load == "draft"
    assert model.built_in_draft is expected
    assert model() is expected
    assert not in_draft_model()
