# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline pooling entry points must not modify the caller's `PoolingParams`.

`LLM.encode()` / `embed()` / `classify()` clone the caller's parameters before
the model's pooling task is filled in. `LLM.score()` used to assign the task on
the very object it was handed, so reusing a `PoolingParams()` with another
pooling model failed before inference with

    You cannot overwrite param.task='classify' with pooling_task='embed'!
"""

from types import SimpleNamespace

import pytest

from vllm import PoolingParams
from vllm.entrypoints.pooling.offline import PoolingOfflineMixin
from vllm.entrypoints.pooling.scoring.io_processor import CrossEncoderIOProcessor


def _make_score_llm(monkeypatch) -> PoolingOfflineMixin:
    """A scoring LLM with no model behind it.

    Input validation runs for real; only the rendering, engine and
    post-processing boundaries are stubbed out. The pooling task is filled in
    before any of them, so the mutation is observable without inference.
    """
    proc = CrossEncoderIOProcessor.__new__(CrossEncoderIOProcessor)
    proc.is_multimodal_model = False
    proc.architecture = "CrossEncoder"
    monkeypatch.setattr(proc, "get_request_factory_offline", lambda ctx: (None, 0))
    monkeypatch.setattr(proc, "post_process_offline", lambda ctx: [])

    llm = PoolingOfflineMixin.__new__(PoolingOfflineMixin)
    llm.runner_type = "pooling"
    llm.pooling_task = "classify"  # SCORE_TYPE_MAP -> "cross-encoder"
    llm.model_config = SimpleNamespace(hf_config=SimpleNamespace(num_labels=1))
    llm.pooling_io_processors = {"cross-encoder": proc}
    monkeypatch.setattr(llm, "_run_tiling_engine", lambda *args, **kwargs: [])
    return llm


def test_score_leaves_caller_pooling_params_untouched(monkeypatch):
    llm = _make_score_llm(monkeypatch)

    params = PoolingParams()
    llm.score("query", "doc", pooling_params=params)

    assert params.task is None, (
        "LLM.score() wrote the model's own pooling task into the caller's "
        "PoolingParams; reusing that object with another pooling model now "
        "fails with 'You cannot overwrite ...' before any inference runs."
    )


def test_score_still_rejects_conflicting_task(monkeypatch):
    llm = _make_score_llm(monkeypatch)

    with pytest.raises(ValueError, match="You cannot overwrite"):
        llm.score("query", "doc", pooling_params=PoolingParams(task="embed"))
