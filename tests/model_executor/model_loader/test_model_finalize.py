# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RL weight updates call ``load_weights`` once per bucket: the model-level
finish step runs once after the last call, and a weight whose checkpoint
tensors land in different calls is written piecewise, never paired."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.reload import finalize_layerwise_reload


class _HookCounter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def process_weights_after_loading(self) -> None:
        self.calls += 1


def test_reload_finish_runs_model_hook_once():
    model = _HookCounter()
    finalize_layerwise_reload(model, None)
    assert model.calls == 1


def _molmo_embed(module_name: str) -> tuple[SimpleNamespace, Any]:
    """Stand-in for ``VocabParallelEmbedding``: 5 vocab rows, this rank holds
    rows [3, 5) plus one padding row."""
    import importlib

    mod = importlib.import_module(f"vllm.model_executor.models.{module_name}")
    embed = SimpleNamespace(
        org_vocab_size=5,
        shard_indices=SimpleNamespace(org_vocab_start_index=3, org_vocab_end_index=5),
        weight=nn.Parameter(torch.full((3, 2), 7.0), requires_grad=False),
        weight_loader=lambda param, w: param.data.copy_(w[3:5].new_zeros(3, 2)),
    )
    embed.weight.weight_loader = mod._split_embedding_loader(embed)
    return embed, mod._load_split_embedding


@pytest.mark.parametrize("module_name", ["molmo", "molmo2"], ids=["molmo", "molmo2"])
@pytest.mark.parametrize(
    "new_first", [False, True], ids=["embedding-first", "new-first"]
)
def test_molmo_embedding_halves_split_across_calls(module_name, new_first):
    """Each half goes through ``param.weight_loader`` (what layerwise reload
    wraps to buffer and count a layer's loads) and lands in this rank's vocab
    shard on arrival: rows [0, 4) come from ``wte.embedding``, row 4 from
    ``wte.new_embedding``."""
    from vllm.model_executor.model_loader.mtp_validation import (
        disable_mtp_completeness_check,
    )

    embed, load = _molmo_embed(module_name)
    inner, seen = embed.weight.weight_loader, []

    def wrapped_loader(param, loaded_weight, shard_id=None):
        seen.append(shard_id)
        inner(param, loaded_weight, shard_id)

    embed.weight.weight_loader = wrapped_loader
    emb = torch.arange(8.0).view(4, 2)
    new_emb = torch.full((1, 2), 9.0)
    calls = [
        [("model.transformer.wte.embedding", emb), ("x", emb)],
        [("model.transformer.wte.new_embedding", new_emb)],
    ]
    if new_first:
        calls.reverse()

    loaded: set[str] = set()
    with disable_mtp_completeness_check():  # a partial update, one half per call
        passed = [name for call in calls for name, _ in load(embed, call, loaded)]

    assert passed == ["x"]
    assert loaded == {"model.embed_tokens.weight"}
    assert sorted(seen) == ["embedding", "new_embedding"]
    expected = torch.tensor([[6.0, 7.0], [9.0, 9.0], [0.0, 0.0]])
    torch.testing.assert_close(embed.weight.detach(), expected)


def test_molmo_embedding_missing_half_rejected_on_full_load():
    """A checkpoint with one half is an error; both halves in one call (a cold
    load) and a whole ``embed_tokens.weight`` (the default loader) are fine."""
    embed, load = _molmo_embed("molmo")
    emb, new_emb = torch.arange(8.0).view(4, 2), torch.full((1, 2), 9.0)

    with pytest.raises(ValueError, match="wte.new_embedding"):
        list(load(embed, [("model.transformer.wte.embedding", emb)], set()))

    list(
        load(
            embed,
            [
                ("model.transformer.wte.new_embedding", new_emb),
                ("model.transformer.wte.embedding", emb),
            ],
            set(),
        )
    )
    expected = torch.tensor([[6.0, 7.0], [9.0, 9.0], [0.0, 0.0]])
    torch.testing.assert_close(embed.weight.detach(), expected)

    embed.weight.weight_loader(embed.weight, torch.ones(5, 2))
    torch.testing.assert_close(embed.weight.detach(), torch.zeros(3, 2))


@pytest.mark.parametrize("shipped", [True, False], ids=["shipped", "omitted"])
def test_bert_decoder_from_embeddings_only_when_omitted(monkeypatch, shipped):
    # the embeddings and the decoder may arrive in different calls
    import vllm.model_executor.models.bert as bert

    class Loader:
        def __init__(self, model):
            pass

        def load_weights(self, weights, mapper=None):
            return {name for name, _ in weights}

    monkeypatch.setattr(bert, "AutoWeightsLoader", Loader)
    emb, decoder = torch.ones(3, 2), torch.zeros(3, 2)
    model = SimpleNamespace(
        decoder_in_checkpoint=False,
        hf_to_vllm_mapper=None,
        bert=SimpleNamespace(
            embeddings=SimpleNamespace(word_embeddings=SimpleNamespace(weight=emb))
        ),
        mlm_head=SimpleNamespace(decoder=SimpleNamespace(weight=decoder)),
    )
    cls: Any = bert.BertForMaskedLM
    if shipped:
        cls.load_weights(model, [("mlm_head.decoder.weight", decoder)])
    cls.load_weights(model, [("bert.embeddings.word_embeddings.weight", emb)])
    cls.process_weights_after_loading(model)
    assert torch.equal(decoder, torch.zeros(3, 2) if shipped else emb)
