# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.config import ModelConfig


@pytest.mark.parametrize(
    ("model_id", "runner_type"),
    [
        ("distilbert/distilgpt2", "generate"),
        ("intfloat/multilingual-e5-small", "pooling"),
        ("jason9693/Qwen2.5-1.5B-apeach", "pooling"),
    ],
)
def test_model_config(model_id, runner_type):
    config = ModelConfig(model_id, task="auto")

    if runner_type == "generate":
        assert config.head_dtype == config.dtype
    else:
        assert config.head_dtype == torch.float32


@pytest.mark.parametrize(
    ("model_id", "runner_type"),
    [
        ("distilbert/distilgpt2", "generate"),
        ("intfloat/multilingual-e5-small", "pooling"),
        ("jason9693/Qwen2.5-1.5B-apeach", "pooling"),
    ],
)
def test_VLLM_USING_FP32_HEAD_0(
    model_id,
    runner_type,
    monkeypatch,
):
    monkeypatch.setenv("VLLM_USING_FP32_HEAD", "0")
    config = ModelConfig(model_id, task="auto")
    assert config.head_dtype == config.dtype


@pytest.mark.parametrize(
    ("model_id", "runner_type"),
    [
        ("distilbert/distilgpt2", "generate"),
        ("intfloat/multilingual-e5-small", "pooling"),
        ("jason9693/Qwen2.5-1.5B-apeach", "pooling"),
    ],
)
def test_VLLM_USING_FP32_HEAD_1(model_id, runner_type, monkeypatch):
    monkeypatch.setenv("VLLM_USING_FP32_HEAD", "1")
    config = ModelConfig(model_id, task="auto")
    assert config.head_dtype == torch.float32
