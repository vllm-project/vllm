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


@pytest.mark.parametrize("model", UNSUPPORTED_MODELS_V1)
def test_unsupported_models(monkeypatch, model):
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
                model="meta-llama/Llama-3.2-3B-Instruct",
                kv_cache_dtype="fp8",
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model="meta-llama/Llama-3.2-3B-Instruct",
                speculative_model="meta-llama/Llama-3.2-1B-Instruct",
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model="meta-llama/Llama-3.2-3B-Instruct",
                guided_decoding_backend="lm-format-enforcer:no-fallback",
            ).create_engine_config()

        with pytest.raises(NotImplementedError):
            AsyncEngineArgs(
                model="meta-llama/Llama-3.2-3B-Instruct",
                guided_decoding_backend="classify",
            ).create_engine_config()


def test_enabled(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
