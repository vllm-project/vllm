# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest

import vllm.envs as envs
from vllm import LLM
from vllm.engine.arg_utils import AsyncEngineArgs

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


def test_reject_bad_config(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "0")


def test_unsupported_configs(monkeypatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model=MODEL,
                speculative_config={
                    "model": MODEL,
                },
            ).create_engine_config()


def test_enable_by_default_fallback(monkeypatch):
    with monkeypatch.context() as m:
        if os.getenv("VLLM_USE_V1", None):
            m.delenv("VLLM_USE_V1")

        # Should default to V1 for supported config.
        _ = AsyncEngineArgs(
            model=MODEL,
            enforce_eager=True,
        ).create_engine_config()
        assert envs.VLLM_USE_V1
        m.delenv("VLLM_USE_V1")


def test_v1_llm_by_default(monkeypatch):
    with monkeypatch.context() as m:
        if os.getenv("VLLM_USE_V1", None):
            m.delenv("VLLM_USE_V1")

        # Should default to V1 for supported config.
        llm = LLM(MODEL, enforce_eager=True, enable_lora=True)
        print(llm.generate("Hello my name is"))
        assert hasattr(llm.llm_engine, "engine_core")
        m.delenv("VLLM_USE_V1")


def test_v1_attn_backend(monkeypatch):
    with monkeypatch.context() as m:
        if os.getenv("VLLM_USE_V1", None):
            m.delenv("VLLM_USE_V1")
        m.setenv("VLLM_ATTENTION_BACKEND", "XFORMERS")

        # Fall back to V0.
        _ = AsyncEngineArgs(model=MODEL).create_engine_config()
        assert not envs.VLLM_USE_V1
        m.delenv("VLLM_USE_V1")

        # Reject if V1.
        m.setenv("VLLM_USE_V1", "1")
        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(model=MODEL).create_engine_config()
        m.delenv("VLLM_USE_V1")

        m.setenv("VLLM_ATTENTION_BACKEND", "FLASHMLA")
        _ = AsyncEngineArgs(model=MODEL).create_engine_config()
        assert envs.VLLM_USE_V1
        m.delenv("VLLM_USE_V1")
