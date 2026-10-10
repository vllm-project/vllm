# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.v1.spec_decode import draft_model as draft_model_module
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

pytestmark = pytest.mark.cpu_test


def test_heterogeneous_vocab_tokenizers_honor_model_configs(monkeypatch):
    target_config = MagicMock(
        tokenizer="target-tokenizer",
        tokenizer_mode="mistral",
        tokenizer_revision="target-revision",
    )
    draft_config = MagicMock(
        model="draft-model",
        tokenizer="draft-model",
        tokenizer_revision="target-revision",
        revision="draft-revision",
    )
    speculative_config = MagicMock(
        use_heterogeneous_vocab=True,
        target_model_config=target_config,
        draft_model_config=draft_config,
    )

    monkeypatch.setattr(
        SpecDecodeBaseProposer,
        "__init__",
        lambda self, **_: setattr(self, "speculative_config", speculative_config),
    )
    monkeypatch.setattr(DraftModelProposer, "_raise_if_draft_tp_mismatch", MagicMock())
    get_tokenizer = MagicMock(side_effect=["target-tokenizer", "draft-tokenizer"])
    monkeypatch.setattr(draft_model_module, "get_tokenizer", get_tokenizer)
    monkeypatch.setattr(draft_model_module, "VocabMapping", MagicMock())
    DraftModelProposer(MagicMock(), MagicMock())

    target_call, draft_call = get_tokenizer.call_args_list
    assert target_call.kwargs["tokenizer_mode"] == "mistral"
    assert target_call.kwargs["revision"] == "target-revision"
    assert draft_call.args == ("draft-model",)
    assert draft_call.kwargs["tokenizer_mode"] == "auto"
    assert draft_call.kwargs["revision"] == "draft-revision"
