# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A multi-tensor weight may be split across load_weights calls (RL weight
updates send one bucket per call); its parts pair up across calls."""

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.reload import finalize_layerwise_reload
from vllm.model_executor.model_loader.weight_utils import (
    discard_weight_parts,
    finalize_model_weights,
)
from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk

PREFIX = "model.layers.0.self_attn.indexer"
FUSED = f"{PREFIX}.wk_weights_proj.weight"


class _RecordingParam:
    def __init__(self) -> None:
        self.loaded: list[torch.Tensor] = []

    def weight_loader(self, param, weight: torch.Tensor, shard_id: int) -> None:
        assert shard_id == 0
        self.loaded.append(weight.clone())


def _wk_tensors() -> list[tuple[str, torch.Tensor]]:
    weight = (torch.arange(128 * 64) % 7 - 3).float().view(128, 64)
    return [
        (f"{PREFIX}.wk.weight", weight.to(torch.float8_e4m3fn)),
        (f"{PREFIX}.wk.weight_scale_inv", torch.tensor([[0.5]])),
    ]


def _load_in_calls(module: nn.Module, calls: list[list[tuple[str, torch.Tensor]]]):
    param = _RecordingParam()
    for call in calls:
        # Fresh per-call state, as each load_weights call builds its own.
        params_dict, loaded = {FUSED: param}, set[str]()
        for name, tensor in call:
            # The transfer bucket is reused: hand over a view, then clobber it.
            bucket = tensor.clone()
            assert _try_load_fp8_indexer_wk(
                name, bucket, module, params_dict, loaded, []
            )
            bucket.zero_()
    return param.loaded


class TestSplitWeightParts:
    @pytest.mark.parametrize(
        "order", [(0, 1), (1, 0)], ids=["weight-first", "scale-first"]
    )
    def test_pair_split_across_calls(self, order):
        tensors = _wk_tensors()
        expected = _load_in_calls(nn.Module(), [tensors])
        module = nn.Module()
        got = _load_in_calls(module, [[tensors[order[0]]], [tensors[order[1]]]])

        assert len(got) == len(expected) == 1
        torch.testing.assert_close(got[0], expected[0])
        finalize_model_weights(module)

    def test_unpaired_part_raises_once_at_finish(self):
        module = nn.Module()
        _load_in_calls(module, [[_wk_tensors()[0]]])

        with pytest.raises(ValueError, match=PREFIX):
            finalize_model_weights(module)
        finalize_model_weights(module)

    def test_aborted_update_part_does_not_pair_with_next(self):
        """An update that failed midway leaves a stashed weight; the next update
        sends the scale first. The scale must wait for the new weight."""
        weight, scale = _wk_tensors()
        stale = (weight[0], (-weight[1].float()).to(weight[1].dtype))
        module = nn.Module()
        _load_in_calls(module, [[stale]])  # the update that never finished
        discard_weight_parts(module)  # the next reload starts

        assert _load_in_calls(module, [[scale]]) == []
        got = _load_in_calls(module, [[weight]])
        expected = _load_in_calls(nn.Module(), [[weight, scale]])
        assert len(got) == 1
        torch.testing.assert_close(got[0], expected[0])


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


@pytest.mark.parametrize("module_name", ["molmo", "molmo2"], ids=["molmo", "molmo2"])
def test_molmo_embedding_halves_split_across_calls(module_name):
    import importlib

    merge = importlib.import_module(
        f"vllm.model_executor.models.{module_name}"
    )._get_weights_with_merged_embedding
    module = nn.Module()
    emb, new_emb = torch.ones(4, 2), torch.zeros(1, 2)

    first = list(merge(module, [("model.transformer.wte.embedding", emb), ("x", emb)]))
    ((name, merged),) = merge(
        module, [("model.transformer.wte.new_embedding", new_emb)]
    )

    assert [n for n, _ in first] == ["x"]
    assert name == "model.embed_tokens.weight"
    torch.testing.assert_close(merged, torch.cat([emb, new_emb]))
