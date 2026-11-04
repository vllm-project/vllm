# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A weight update calls ``load_weights`` once per bucket (the NCCL engine:
once per tensor), so tensors a model combines or derives from can land in
different calls. These models must load each tensor on its own and leave the
derived work to ``process_weights_after_loading``."""

import pytest
import torch

from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
)


def _halves() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.arange(8.0).view(4, 2), torch.full((1, 2), 9.0)


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_molmo_embedding_halves_in_one_call_go_through_the_loader():
    """Both halves in one call are concatenated for the weight loader, which
    layerwise reload wraps (a direct write would miss its meta parameter)."""
    from vllm.model_executor.models.molmo import load_split_embedding
    from vllm.model_executor.models.utils import AutoWeightsLoader

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = embed = VocabParallelEmbedding(5, 2)
    calls: list[torch.Tensor] = []

    def weight_loader(param: torch.Tensor, weight: torch.Tensor) -> None:
        calls.append(weight)
        embed.weight_loader(param, weight)

    embed.weight.weight_loader = weight_loader
    emb, new_emb = _halves()
    weights = [
        ("model.transformer.wte.new_embedding", new_emb),
        ("model.transformer.wte.embedding", emb),
    ]

    direct: set[str] = set()
    weights = load_split_embedding(embed, weights, direct)
    loaded = AutoWeightsLoader(model).load_weights(weights)

    assert loaded == {"model.embed_tokens.weight"} and not direct
    assert len(calls) == 1
    torch.testing.assert_close(embed.weight[:5], torch.cat([emb, new_emb]))


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_molmo_embedding_halves_in_separate_calls():
    """A half that arrives alone lands in its rows, with the vocab padding
    zeroed, and each call reports ``embed_tokens`` loaded."""
    from vllm.model_executor.models.molmo import load_split_embedding

    embed = VocabParallelEmbedding(5, 2)
    embed.weight.data.fill_(7.0)
    emb, new_emb = _halves()
    calls = [
        [("model.transformer.wte.embedding", emb), ("other", emb)],
        [("model.transformer.wte.new_embedding", new_emb)],
    ]

    passed, loaded = [], []
    with disable_mtp_completeness_check():  # as the weight-transfer engines
        for call in calls:
            names: set[str] = set()
            passed += [n for n, _ in load_split_embedding(embed, call, names)]
            loaded.append(names)

    assert passed == ["other"]
    assert loaded == [{"model.embed_tokens.weight"}] * 2
    torch.testing.assert_close(embed.weight[:5], torch.cat([emb, new_emb]))
    assert not embed.weight[5:].any()


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_molmo_embedding_half_alone_rejected():
    """A full load needs both halves; under layerwise reload (meta) a half sent
    alone would be dropped silently."""
    from vllm.model_executor.models.molmo import load_split_embedding

    with torch.device("meta"):
        embed = VocabParallelEmbedding(5, 2)
    weights = [("model.transformer.wte.embedding", _halves()[0])]
    with pytest.raises(ValueError, match="missing"):  # a full load
        list(load_split_embedding(embed, weights, set()))
    with (
        disable_mtp_completeness_check(),
        pytest.raises(RuntimeError, match="same load_weights call"),
    ):
        list(load_split_embedding(embed, weights, set()))


def _update_bert_with_decoder(worker) -> bool:
    """A direct update, one tensor per call, that ships an explicit MLM decoder
    weight (zeros) before the embeddings."""
    from vllm.model_executor.model_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.mtp_validation import (
        disable_mtp_completeness_check,
    )
    from vllm.model_executor.model_loader.reload import finish_reload, start_reload

    model = worker.get_model()
    loader = DefaultModelLoader(worker.vllm_config.load_config)
    weights = dict(loader.get_all_weights(worker.model_config, model))
    emb = weights["bert.embeddings.word_embeddings.weight"]
    decoder_name = "cls.predictions.decoder.weight"
    weights.pop(decoder_name, None)
    weights = {decoder_name: torch.zeros_like(emb)} | weights
    start_reload(model)
    with disable_mtp_completeness_check():
        for name, weight in weights.items():
            model.load_weights([(name, weight)])
    finish_reload(model, worker.model_config, worker.device)

    return not model.mlm_head.decoder.weight.any()


def test_bert_mlm_decoder_shipped_in_an_earlier_call(vllm_runner, monkeypatch):
    """The decoder is tied to the embeddings only when the checkpoint does not
    ship it, decided over the whole update, not per call."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        "google-bert/bert-base-uncased", runner="pooling", max_model_len=512
    ) as llm:
        assert llm.collective_rpc(_update_bert_with_decoder) == [True]
