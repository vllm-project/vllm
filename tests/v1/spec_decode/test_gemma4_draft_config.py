# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A MoE Gemma 4 target with its dense MTP assistant must load the draft under
expert parallelism. Config only: get_model is replaced, so no weights load."""

from unittest.mock import patch

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator


class _Captured(Exception):
    def __init__(self, vllm_config, model_config):
        self.vllm_config = vllm_config
        self.model_config = model_config


def test_gemma4_mtp_draft_loads_with_expert_parallelism():
    vllm_config = EngineArgs(
        model="google/gemma-4-26B-A4B-it",
        enable_expert_parallel=True,
        max_model_len=1024,
        speculative_config={
            "method": "mtp",
            "model": "google/gemma-4-26B-A4B-it-assistant",
            "num_speculative_tokens": 1,
        },
    ).create_engine_config()
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    speculator = Gemma4Speculator.__new__(Gemma4Speculator)
    speculator.vllm_config = vllm_config
    speculator.speculative_config = speculative_config

    def fake_get_model(*, vllm_config, model_config, load_config):
        raise _Captured(vllm_config, model_config)

    with (
        patch(
            "vllm.v1.worker.gpu.spec_decode.gemma4.speculator.get_model",
            fake_get_model,
        ),
        pytest.raises(_Captured) as captured,
    ):
        speculator.load_draft_model(None, set())

    assert captured.value.vllm_config.model_config is vllm_config.model_config
    assert captured.value.model_config is speculative_config.draft_model_config
