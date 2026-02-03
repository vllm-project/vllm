# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

MODELS = ["google/gemma-2b", "google/gemma-2-2b", "google/gemma-3-4b-it"]


@pytest.mark.parametrize("model", MODELS)
def test_dummy_loader(vllm_runner, monkeypatch, model: str) -> None:
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(
            model,
            load_format="dummy",
        ) as llm:
            if model == "google/gemma-3-4b-it":
                normalizers = llm.llm.collective_rpc(
                    lambda self: self.model_runner.model.language_model.model.normalizer.cpu().item()  # noqa: E501
                )
                config = llm.llm.llm_engine.model_config.hf_config.text_config
            else:
                normalizers = llm.llm.collective_rpc(
                    lambda self: self.model_runner.model.model.normalizer.cpu().item()
                )
                config = llm.llm.llm_engine.model_config.hf_config
            assert np.allclose(normalizers, config.hidden_size**0.5, rtol=2e-3)
