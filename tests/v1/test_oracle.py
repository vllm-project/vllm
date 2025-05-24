# SPDX-License-Identifier: Apache-2.0
import os

import pytest

import vllm.envs as envs
from vllm import LLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

UNSUPPORTED_MODELS_V1 = [
    "openai/whisper-large-v3",  # transcription
    "facebook/bart-large-cnn",  # encoder decoder
    "mistralai/Mamba-Codestral-7B-v0.1",  # mamba
    "hmellor/tiny-random-BambaForCausalLM",  # hybrid
    "BAAI/bge-m3",  # embedding
]

MODEL = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.parametrize("model", UNSUPPORTED_MODELS_V1)
def test_reject_unsupported_models(monkeypatch, model):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        args = AsyncEngineArgs(model=model)

        with pytest.raises(NotImplementedError):
            _ = args.create_engine_config()
        m.delenv("VLLM_USE_V1")


def test_reject_bad_config(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "0")


def test_unsupported_configs(monkeypatch):

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model=MODEL,
                kv_cache_dtype="fp8",
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model=MODEL,
                speculative_config={
                    "model": MODEL,
                },
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model=MODEL,
                guided_decoding_backend="lm-format-enforcer",
                guided_decoding_disable_fallback=True,
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model=MODEL,
                preemption_mode="swap",
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model=MODEL,
                disable_async_output_proc=True,
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model=MODEL,
                scheduling_policy="priority",
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model=MODEL,
                num_scheduler_steps=5,
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model=MODEL,
                scheduler_delay_factor=1.2,
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

        # Should fall back to V0 for supported model.
        _ = AsyncEngineArgs(
            model=UNSUPPORTED_MODELS_V1[0]).create_engine_config()
        assert not envs.VLLM_USE_V1
        m.delenv("VLLM_USE_V1")


def test_v1_llm_by_default(monkeypatch):
    with monkeypatch.context() as m:
        if os.getenv("VLLM_USE_V1", None):
            m.delenv("VLLM_USE_V1")

        # Should default to V1 for supported config.
        model = LLM(MODEL, enforce_eager=True, enable_lora=True)
        print(model.generate("Hello my name is"))
        assert hasattr(model.llm_engine, "engine_core")
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


def test_reject_using_constructor_directly(monkeypatch):
    with monkeypatch.context() as m:
        if os.getenv("VLLM_USE_V1", None):
            m.delenv("VLLM_USE_V1")

        # Sets VLLM_USE_V1=1.
        vllm_config = AsyncEngineArgs(model=MODEL).create_engine_config()

        # This uses the V0 constructor directly.
        with pytest.raises(ValueError):
            AsyncLLMEngine(vllm_config,
                           AsyncLLMEngine._get_executor_cls(vllm_config),
                           log_stats=True)

        m.delenv("VLLM_USE_V1")
