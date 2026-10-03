# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.model_executor.warmup import minimax_m3_msa_warmup as warmup

_MINIMAX_MODEL_MODULE = "vllm.models.minimax_m3.nvidia.model"


def test_warmup_module_does_not_import_minimax_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, warmup.__name__)
    monkeypatch.setitem(sys.modules, _MINIMAX_MODEL_MODULE, None)

    importlib.import_module(warmup.__name__)


@pytest.mark.parametrize(
    ("architecture", "is_blackwell", "should_run"),
    [
        ("DiffusionGemmaForBlockDiffusion", True, False),
        ("MiniMaxM3SparseForCausalLM", False, False),
        ("MiniMaxM3SparseForCausalLM", True, True),
    ],
)
def test_warmup_requires_supported_model_and_platform(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    is_blackwell: bool,
    should_run: bool,
) -> None:
    class SparseAttention:
        pass

    monkeypatch.setitem(
        sys.modules,
        _MINIMAX_MODEL_MODULE,
        SimpleNamespace(MiniMaxM3SparseAttention=SparseAttention),
    )
    monkeypatch.setattr(warmup.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        warmup.current_platform,
        "is_device_capability_family",
        lambda capability: is_blackwell and capability == 100,
    )
    get_model = Mock(return_value=SimpleNamespace(modules=lambda: [SparseAttention()]))
    dummy_run = Mock()
    worker = SimpleNamespace(
        get_model=get_model,
        model_runner=SimpleNamespace(_dummy_run=dummy_run),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(architecture=architecture)
        ),
    )

    warmup.minimax_m3_msa_warmup(worker)

    assert get_model.called is should_run
    assert dummy_run.called is should_run
