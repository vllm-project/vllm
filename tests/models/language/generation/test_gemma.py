# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models import gemma

MODELS = ["google/gemma-2b", "google/gemma-2-2b", "google/gemma-3-4b-it"]


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_checkpoint_lm_head_can_override_tied_config(monkeypatch) -> None:
    """A physical LM head must load after checkpoint-driven untying."""

    class StubGemmaModel(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix):
            super().__init__()
            self.embed_tokens = VocabParallelEmbedding(4, 2)
            self.make_empty_intermediate_tensors = None

    monkeypatch.setattr(gemma, "GemmaModel", StubGemmaModel)
    config = SimpleNamespace(
        vocab_size=4,
        hidden_size=2,
        tie_word_embeddings=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
    )
    model = gemma.GemmaForCausalLM(vllm_config=cast(VllmConfig, vllm_config))
    embedding_weight = torch.full((4, 2), 1.0)
    lm_head_weight = torch.full((4, 2), 2.0)

    loaded = model.load_weights(
        [
            ("model.embed_tokens.weight", embedding_weight),
            ("lm_head.weight", lm_head_weight),
        ]
    )

    assert loaded == {"model.embed_tokens.weight", "lm_head.weight"}
    assert torch.equal(model.model.embed_tokens.weight[:4], embedding_weight)
    assert torch.equal(model.lm_head.weight[:4], lm_head_weight)


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
