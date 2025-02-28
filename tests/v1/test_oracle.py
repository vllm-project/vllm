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
@pytest.mark.parametrize("use_v1", ["0", "1"])
def test_unsupported_models(
    monkeypatch,
    model,
    use_v1,
):
    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", use_v1)
        args = AsyncEngineArgs(model=model)

        if use_v1 == "1":
            with pytest.raises(NotImplementedError):
                _ = args.create_engine_config()
        else:
            config = args.create_engine_config()
            assert not config.use_v1
