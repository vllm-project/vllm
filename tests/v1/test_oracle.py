# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.engine.arg_utils import AsyncEngineArgs

UNSUPPORTED_MODELS_V1 = [
    "openai/whisper-large-v3",  # transcription
    "facebook/bart-large-cnn",  # encoder decoder
    "mistralai/Mamba-Codestral-7B-v0.1",  # mamba
    "ibm-ai-platform/Bamba-9B",  # hybrid
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
                speculative_model=MODEL,
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model=MODEL,
                guided_decoding_backend="lm-format-enforcer:no-fallback",
            ).create_engine_config()


def test_enable_by_default_fallback(monkeypatch):
    with monkeypatch.context() as m:
        m.delenv("VLLM_USE_V1")
        m.setenv("VLLM_USE_V1_BY_DEFAULT", "1")

        # Should default to V1 for supported config.
        vllm_config = AsyncEngineArgs(
            model=MODEL,
            enforce_eager=True,
        ).create_engine_config()
        assert vllm_config.use_v1

        # Should fall back to V0 for experimental config.
        vllm_config = AsyncEngineArgs(
            model=MODEL,
            enable_lora=True,
        ).create_engine_config()
        assert not vllm_config.use_v1

        # Should fall back to V0 for experimental config.
        vllm_config = AsyncEngineArgs(
            model=MODEL,
            enable_lora=True,
        ).create_engine_config()
        assert not vllm_config.use_v1

        # Should fall back to V0 for unsupported config.
        vllm_config = AsyncEngineArgs(
            model=MODEL,
            enable_lora=True,
        ).create_engine_config()
        assert not vllm_config.use_v1
