# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

MODELS = ["google/gemma-2b", "google/gemma-2-2b"]


@pytest.mark.parametrize("model", MODELS)
def test_dummy_loader(vllm_runner, monkeypatch, model: str) -> None:
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(
            model,
            load_format="dummy",
        ) as llm:
            normalizers = llm.apply_model(
                lambda model: model.model.normalizer.cpu().item()
            )
            config = llm.llm.llm_engine.model_config.hf_config
            assert np.allclose(normalizers, config.hidden_size**0.5, rtol=2e-3)
