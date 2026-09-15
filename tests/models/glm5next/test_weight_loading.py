# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 (weight, weight_scale_inv) pairs in the GLM-5.3-Flash loader must
survive being delivered in different ``load_weights`` calls.

``Glm5NextModel.load_weights`` dequantizes the block-FP8 ``q_a_proj`` /
``kv_a_proj_with_mqa`` / ``q_b_proj`` / ``o_proj`` / ``indexer.wk`` tensors to
BF16 once both halves of a pair have been seen, parking the first half in a
pending dict. ``AutoWeightsLoader`` invokes a child's ``load_weights`` once per
contiguous run of a top-level prefix (``itertools.groupby``), so a stream such
as ``language_model…, visual…, language_model…`` produces two calls. A
streaming loader that reorders tensors (Run:ai distributed streaming deals
chunks to ranks by size and interleaves the vision-tower shard) delivers the
two halves of a pair in different calls. The pair must still complete; today
the pending dict is local to one call and the half-pair is silently dropped,
leaving ``fused_qkv_a_proj`` unwritten.

These tests run on CPU and drive the real ``load_weights`` implementations on
minimal module trees, in the style of ``tests/models/kimi_k3``.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)

pytestmark = pytest.mark.cpu_test

HIDDEN = 256
BLOCK = 128
Q_LORA_RANK = 256
KV_LORA_RANK = 128
QK_ROPE_HEAD_DIM = 64  # NoPE checkpoint: kv_a rows are padded by this much
LAYER = 7
ATTN = f"layers.{LAYER}.self_attn"
FUSED = f"{ATTN}.fused_qkv_a_proj.weight"


def _fp8(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, cols, generator=g) * 0.1).to(torch.float8_e4m3fn)


def _scale_inv(rows: int, cols: int, value: float) -> torch.Tensor:
    return torch.full(
        ((rows + BLOCK - 1) // BLOCK, (cols + BLOCK - 1) // BLOCK),
        value,
        dtype=torch.float32,
    )


def _pair_tensors() -> dict[str, torch.Tensor]:
    """The four checkpoint tensors behind layer 7's fused_qkv_a_proj."""
    return {
        f"{ATTN}.q_a_proj.weight": _fp8(Q_LORA_RANK, HIDDEN, seed=1),
        f"{ATTN}.q_a_proj.weight_scale_inv": _scale_inv(Q_LORA_RANK, HIDDEN, 0.5),
        f"{ATTN}.kv_a_proj_with_mqa.weight": _fp8(KV_LORA_RANK, HIDDEN, seed=2),
        f"{ATTN}.kv_a_proj_with_mqa.weight_scale_inv": _scale_inv(
            KV_LORA_RANK, HIDDEN, 0.25
        ),
    }


def _make_text_model() -> tuple[Glm5NextModel, list[int]]:
    """A ``Glm5NextModel`` shell with only layer 7's fused_qkv_a_proj param.

    ``load_weights`` is the real implementation; the module tree carries just
    what it touches: ``config`` and a BF16 ``fused_qkv_a_proj.weight`` whose
    ``weight_loader`` writes shard 0 (q_a) into the first rows and shard 1
    (kv_a, rope-padded) into the rest, recording the shards it received.
    """
    model = object.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        is_moe=False,
        is_linear_attn=False,
        mla_nope=True,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        num_hidden_layers=LAYER + 1,
        num_nextn_predict_layers=0,
    )
    layers = nn.ModuleList([nn.Module() for _ in range(LAYER + 1)])
    attn = nn.Module()
    fused = nn.Module()
    rows = Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM
    fused.weight = nn.Parameter(
        torch.zeros(rows, HIDDEN, dtype=torch.bfloat16), requires_grad=False
    )
    shards_seen: list[int] = []

    def weight_loader(param: torch.Tensor, loaded: torch.Tensor, shard_id: int):
        shards_seen.append(shard_id)
        if shard_id == 0:
            param.data[:Q_LORA_RANK].copy_(loaded)
        else:
            param.data[Q_LORA_RANK:].copy_(loaded)

    fused.weight.weight_loader = weight_loader
    attn.fused_qkv_a_proj = fused
    layers[LAYER].self_attn = attn
    model.layers = layers
    return model, shards_seen


def _assert_fused_written(model: Glm5NextModel, shards_seen: list[int]) -> None:
    param = dict(model.named_parameters())[FUSED]
    assert sorted(shards_seen) == [0, 1], shards_seen
    assert param[:Q_LORA_RANK].abs().sum() > 0, "q_a rows never written"
    assert param[Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK].abs().sum() > 0, (
        "kv_a rows never written"
    )
    # NoPE padding rows stay zero.
    assert param[Q_LORA_RANK + KV_LORA_RANK :].abs().sum() == 0


def test_fp8_pairs_complete_within_one_call() -> None:
    """Control: per-shard order delivers weight and scale adjacently."""
    model, shards_seen = _make_text_model()
    tensors = _pair_tensors()

    loaded = model.load_weights(iter(tensors.items()))

    assert FUSED in loaded
    _assert_fused_written(model, shards_seen)


def test_fp8_pairs_split_across_load_weights_calls() -> None:
    """The same four tensors, halves delivered in two calls.

    This is what AutoWeightsLoader does when any tensor with another
    top-level prefix (e.g. ``visual.*``) sits between the weight and its
    scale in the stream. The pair must still complete.
    """
    model, shards_seen = _make_text_model()
    tensors = _pair_tensors()
    weights_only = {k: v for k, v in tensors.items() if k.endswith(".weight")}
    scales_only = {k: v for k, v in tensors.items() if "weight_scale_inv" in k}

    loaded = set()
    loaded |= model.load_weights(iter(weights_only.items()))
    loaded |= model.load_weights(iter(scales_only.items()))

    assert FUSED in loaded, (
        "fused_qkv_a_proj was never written: the FP8 half-pairs parked in the "
        "first call were discarded when it returned"
    )
    _assert_fused_written(model, shards_seen)


def _make_multimodal_model() -> tuple[
    Glm5NextForConditionalGeneration, Glm5NextModel, list[int]
]:
    text_model, shards_seen = _make_text_model()

    language_model = object.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(language_model)
    language_model.config = SimpleNamespace(tie_word_embeddings=False)
    language_model.model = text_model

    model = object.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = language_model
    model.visual = nn.Module()
    model.visual.patch_embed = nn.Module()
    model.visual.patch_embed.proj = nn.Module()
    model.visual.patch_embed.proj.weight = nn.Parameter(
        torch.zeros(4), requires_grad=False
    )
    return model, text_model, shards_seen


def _hf_stream(order: list[str]) -> list[tuple[str, torch.Tensor]]:
    """Checkpoint-named tensors (``model.language_model.*`` / ``model.visual.*``)."""
    pair = _pair_tensors()
    visual = (
        "model.visual.patch_embed.proj.weight",
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    out = []
    for key in order:
        if key == "visual":
            out.append(visual)
        else:
            out.append((f"model.language_model.{key}", pair[key]))
    return out


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(
            [
                f"{ATTN}.q_a_proj.weight",
                f"{ATTN}.q_a_proj.weight_scale_inv",
                f"{ATTN}.kv_a_proj_with_mqa.weight",
                f"{ATTN}.kv_a_proj_with_mqa.weight_scale_inv",
                "visual",
            ],
            id="per-shard-order-vision-last",
        ),
        pytest.param(
            [
                f"{ATTN}.q_a_proj.weight",
                f"{ATTN}.kv_a_proj_with_mqa.weight",
                "visual",
                f"{ATTN}.q_a_proj.weight_scale_inv",
                f"{ATTN}.kv_a_proj_with_mqa.weight_scale_inv",
            ],
            id="vision-tensor-between-weight-and-scale",
        ),
    ],
)
def test_multimodal_stream_order_does_not_change_loaded_weights(order) -> None:
    """End to end through ``Glm5NextForConditionalGeneration.load_weights``.

    Same five tensors, two orders. In the second, the vision-tower tensor
    splits the language-model run in two, so ``AutoWeightsLoader`` calls
    ``Glm5NextModel.load_weights`` twice. The result must not depend on it.
    """
    model, text_model, shards_seen = _make_multimodal_model()

    loaded = model.load_weights(iter(_hf_stream(order)))

    assert "visual.patch_embed.proj.weight" in loaded
    assert model.visual.patch_embed.proj.weight.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert f"language_model.model.{FUSED}" in loaded, (
        "fused_qkv_a_proj was never written for this stream order"
    )
    _assert_fused_written(text_model, shards_seen)
