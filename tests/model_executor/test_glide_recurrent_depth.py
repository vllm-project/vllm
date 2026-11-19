# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLiDE's recurrent depth: blocks of middle layers run more than once.

The backbone must give the hidden states of the checkpoint's `glide/recurrent.py`:
with `P` passes, a block `G` updates its input by `x <- x + (G(x) - x) / P`,
`P` times, and the next layer reads the result.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import qwen3_next
from vllm.model_executor.models.config import GlideForSequenceClassificationConfig
from vllm.model_executor.models.qwen3_next import Qwen3NextModel

NUM_LAYERS = 6
MIDLOOP = {
    "conv": "A",
    "strategy": "euler",
    "blocks": [[0, 1], [2, 4]],
    "n_gate": 0,
    "n_lora": 0,
    "passes": 3,
}


def model_config(midloop: dict | None) -> SimpleNamespace:
    decision = {
        "format_version": 1,
        "attention_mode": "noncausal_full_attention",
        "codes": ["A", "B"],
        "token_ids": [32, 33],
    }
    if midloop is not None:
        decision["midloop"] = midloop
    text_config = SimpleNamespace(num_hidden_layers=NUM_LAYERS)
    return SimpleNamespace(
        hf_config=SimpleNamespace(decision_config=decision),
        hf_text_config=text_config,
        pooler_config=SimpleNamespace(use_activation=None),
    )


def test_midloop_sets_the_backbone_recurrent_depth():
    config = model_config(MIDLOOP)
    GlideForSequenceClassificationConfig.verify_and_update_model_config(config)
    assert config.hf_text_config.recurrent_depth == {
        "blocks": [[0, 1], [2, 4]],
        "passes": 3,
    }
    assert config.hf_text_config.is_causal is False

    plain = model_config({**MIDLOOP, "passes": 1})
    GlideForSequenceClassificationConfig.verify_and_update_model_config(plain)
    assert not hasattr(plain.hf_text_config, "recurrent_depth")


@pytest.mark.parametrize(
    "change",
    [
        {"strategy": "anderson"},
        {"n_lora": 8},
        {"blocks": [[0, 3], [3, 4]]},
        {"blocks": [[4, NUM_LAYERS]]},
        {"passes": -1},
    ],
)
def test_unsupported_midloop_is_rejected(change):
    config = model_config({**MIDLOOP, **change})
    with pytest.raises(ValueError, match="midloop"):
        GlideForSequenceClassificationConfig.verify_and_update_model_config(config)


def test_backbone_repeats_blocks_like_the_checkpoint_runtime(monkeypatch):
    torch.manual_seed(0)
    weights = [torch.randn(8, 8) / 4 for _ in range(NUM_LAYERS)]

    def reference_layer(i: int, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(x @ weights[i])

    def vllm_layer(i, positions, hidden_states, residual):
        # vLLM layers carry the stream as `hidden_states + residual`.
        x = hidden_states if residual is None else hidden_states + residual
        return torch.tanh(x @ weights[i]), x

    config = model_config(MIDLOOP)
    GlideForSequenceClassificationConfig.verify_and_update_model_config(config)
    depth = config.hf_text_config.recurrent_depth
    model = SimpleNamespace(
        embed_input_ids=lambda ids: ids,
        use_sequence_parallel=False,
        layers=[partial(vllm_layer, i) for i in range(NUM_LAYERS)],
        start_layer=0,
        end_layer=NUM_LAYERS,
        norm=lambda x, residual=None: x if residual is None else (x + residual, None),
        recurrent_blocks={last: first for first, last in depth["blocks"]},
        recurrent_passes=depth["passes"],
        _maybe_add_hidden_state=lambda aux, *args: aux,
    )
    model._repeat_block = partial(Qwen3NextModel._repeat_block, model)
    monkeypatch.setattr(
        qwen3_next,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    inputs = torch.randn(5, 8)
    expected = inputs.clone()
    layer = 0
    while layer < NUM_LAYERS:
        last = next((b for a, b in depth["blocks"] if a == layer), None)
        if last is None:
            expected = reference_layer(layer, expected)
            layer += 1
            continue
        for _ in range(depth["passes"]):
            out = expected
            for i in range(layer, last + 1):
                out = reference_layer(i, out)
            expected = expected + (out - expected) / depth["passes"]
        layer = last + 1

    hidden_states = Qwen3NextModel.forward(
        model, None, torch.arange(5), inputs_embeds=inputs.clone()
    )
    torch.testing.assert_close(hidden_states, expected)
