# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest

MODELS = ["google/gemma-2b", "google/gemma-2-2b", "google/gemma-3-4b-it"]


@pytest.mark.parametrize("model", MODELS)
def test_dummy_loader(vllm_runner, model: str) -> None:
    with vllm_runner(
            model,
            load_format="dummy",
    ) as llm:
        normalizers = llm.collective_rpc(lambda self: self.worker.model_runner.
                                         model.model.normalizer.cpu().item())
        assert np.allclose(
            normalizers,
            llm.llm_engine.model_config.hf_config.hidden_size**0.5,
            rtol=1e-3)
