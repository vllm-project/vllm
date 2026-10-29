# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mistral Large 4 DSpark drafter: checkpoint mapping and trainer parity of the
recurrent Markov head and position-specific draft inputs.

The drafter has one implementation per platform (``nvidia/`` and ``amd/``), so
the tests run against the one that serves the current platform.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.models.mistral_large4.amd.dspark import (
        MistralLarge4DSparkModel,
        _to_draft_weight_name,
        draft_attention_windows,
    )
    from vllm.models.mistral_large4.amd.model import MistralLarge4Model
else:
    from vllm.models.mistral_large4.nvidia.dspark import (
        MistralLarge4DSparkModel,
        _to_draft_weight_name,
        draft_attention_windows,
    )
    from vllm.models.mistral_large4.nvidia.model import MistralLarge4Model


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name"),
    [
        (
            "mtp_dspark.backbone.layers.0.attention_norm.weight",
            "model.layers.0.input_layernorm.weight",
        ),
        (
            "mtp_dspark.backbone.layers.2.attention.wq.weight",
            "model.layers.2.self_attn.q_proj.weight",
        ),
        (
            "mtp_dspark.backbone.layers.1.attention.wkv_a_with_mqa.weight",
            "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        ),
        (
            "mtp_dspark.backbone.layers.1.feed_forward.w1.weight",
            "model.layers.1.mlp.w1.weight",
        ),
        (
            "mtp_dspark.backbone.target_projection.weight",
            "model.target_projection.weight",
        ),
        ("mtp_dspark.backbone.target_norm.weight", "model.target_norm.weight"),
        ("mtp_dspark.backbone.final_norm.weight", "model.final_norm.weight"),
        ("mtp_dspark.backbone.mask_embedding", "model.mask_embedding"),
        (
            "mtp_dspark.autoregressive_head.embedding.weight",
            "model.markov_head.markov_w1.weight",
        ),
        (
            "mtp_dspark.autoregressive_head.output.weight",
            "model.markov_head.markov_w2.weight",
        ),
        (
            "mtp_dspark.autoregressive_head.joint_proj.bias",
            "model.joint_proj.bias",
        ),
    ],
)
def test_checkpoint_weight_mapping(checkpoint_name: str, runtime_name: str):
    assert _to_draft_weight_name(checkpoint_name) == runtime_name


def _trainer_rnn_logits(
    base_logits: torch.Tensor,  # [N, D, V]
    predecessor_ids: torch.Tensor,  # [N, D]
    hidden: torch.Tensor,  # [N, D, H]
    embedding: nn.Embedding,
    joint_proj: nn.Linear,
    output: nn.Linear,
) -> torch.Tensor:
    """mistral.model.mtp.dspark.autoregressive_head.RNNHead, unrolled over D."""
    state = None
    logits = []
    for d in range(base_logits.shape[1]):
        emb = embedding(predecessor_ids[:, d])
        if state is None:
            state = torch.zeros_like(emb)
        joint = joint_proj(torch.cat([state, emb, hidden[:, d]], dim=-1))
        gate_raw, candidate_raw, output_raw = joint.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        logits.append(base_logits[:, d] + output(torch.tanh(output_raw)))
    return torch.stack(logits, dim=1)


def test_rnn_markov_head_matches_trainer():
    """Serving threads the state across the positions of one proposal, feeding
    each step the previously sampled token (teacher-forced here)."""
    torch.manual_seed(0)
    num_reqs, depth, vocab, hidden_size, rank = 3, 4, 32, 16, 8
    embedding = nn.Embedding(vocab, rank)
    joint_proj = nn.Linear(hidden_size + 2 * rank, 3 * rank)
    output = nn.Linear(rank, vocab, bias=False)
    drafter = SimpleNamespace(
        markov_head=SimpleNamespace(embed=embedding),
        joint_proj=lambda x: (joint_proj(x), None),
    )
    base_logits = torch.randn(num_reqs, depth, vocab)
    prev_ids = torch.randint(vocab, (num_reqs, depth))
    hidden = torch.randn(num_reqs, depth, hidden_size)

    state = None
    logits = []
    for d in range(depth):
        markov_embed, state = MistralLarge4DSparkModel.markov_step(
            drafter, prev_ids[:, d], hidden[:, d], state
        )
        logits.append(base_logits[:, d] + output(markov_embed))

    expected = _trainer_rnn_logits(
        base_logits, prev_ids, hidden, embedding, joint_proj, output
    )
    torch.testing.assert_close(torch.stack(logits, dim=1), expected)


def _internal_embed_dspark_inputs(
    input_ids: torch.Tensor,
    embedding: nn.Embedding,
    mask_token_id: int,
    mask_embedding: torch.Tensor,
) -> torch.Tensor:
    """mistral_vllm.models.dspark.utils.embed_dspark_inputs (#43772)."""
    if mask_embedding.ndim == 1:
        is_mask = input_ids == mask_token_id
        embeddings = embedding(input_ids.masked_fill(is_mask, 0))
        return torch.where(is_mask.unsqueeze(-1), mask_embedding, embeddings)
    block_width = mask_embedding.shape[0] + 1
    blocks = input_ids.reshape(-1, block_width)
    anchors = embedding(blocks[:, 0])
    masks = mask_embedding.expand(blocks.shape[0], block_width - 1, -1)
    return torch.cat((anchors.unsqueeze(1), masks), dim=1).reshape(
        -1, anchors.shape[-1]
    )


@pytest.mark.parametrize("position_specific", [False, True])
def test_draft_block_embeddings_match_internal(position_specific: bool):
    """Each block is [anchor, mask_1, ..., mask_{D-1}]; position-specific
    checkpoints store one mask row per slot (slot k reads row k - 1)."""
    torch.manual_seed(0)
    num_reqs, depth, vocab, hidden_size = 3, 4, 32, 8
    embed_tokens = nn.Embedding(vocab, hidden_size)
    mask_shape = (depth - 1, hidden_size) if position_specific else (hidden_size,)
    mask_embedding = torch.randn(mask_shape)
    drafter = SimpleNamespace(
        embed_tokens=embed_tokens,
        vocab_size=vocab,
        block_size=depth,
        mask_embedding=mask_embedding,
        position_specific=position_specific,
    )
    input_ids = torch.full((num_reqs, depth), vocab)  # mask sentinel = vocab
    input_ids[:, 0] = torch.randint(vocab, (num_reqs,))

    embeds = MistralLarge4DSparkModel.embed_input_ids(drafter, input_ids.flatten())

    expected = _internal_embed_dspark_inputs(
        input_ids.flatten(), embed_tokens, vocab, mask_embedding
    )
    torch.testing.assert_close(embeds, expected)
    if position_specific:
        rows = embeds.view(num_reqs, depth, -1)[:, 1:]
        torch.testing.assert_close(rows, mask_embedding.expand(num_reqs, -1, -1))


def test_sliding_window_matches_trainer_mask():
    """Serving windows each non-causal draft block by `window` keys back from
    its end; that must expose exactly the context keys of the trainer mask
    (mistral.model.mtp.dspark.attention: source - target < sliding_window,
    target <= source) plus the whole own block."""
    sliding_window, depth = 6, 4
    config = SimpleNamespace(
        layer_types=["sliding_attention"], sliding_window=sliding_window
    )
    (window,) = draft_attention_windows(config, depth)
    for source in range(12):
        # Context K/V at positions 0..source; the block (anchor at source + 1)
        # occupies source + 1 .. source + depth.
        seq_len = source + 1 + depth
        visible = set(range(max(seq_len - window, 0), seq_len))
        trainer = {t for t in range(source + 1) if source - t < sliding_window}
        trainer |= set(range(source + 1, seq_len))
        assert visible == trainer, source
    full = SimpleNamespace(layer_types=None, num_hidden_layers=2, sliding_window=None)
    assert draft_attention_windows(full, depth) == [None, None]


class _InPlaceResidualLayer(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def input_norm(self, hidden_states, residual):
        return hidden_states, hidden_states.clone() if residual is None else residual

    def residual_norm(self, norm, hidden_states, residual):
        return norm(hidden_states, residual)

    def forward(self, positions, hidden_states, residual, *args):
        residual.add_(100)
        return self.output.clone(), residual


class _InPlaceAddNorm(nn.Module):
    """Overwrites both inputs, like vLLM's fused add + RMSNorm."""

    def forward(self, hidden_states, residual):
        residual.add_(hidden_states)
        hidden_states.copy_(2 * residual)
        return hidden_states, residual


def test_final_tap_precedes_target_norm(monkeypatch):
    """The last-layer DSpark tap is the residual stream entering the target's
    final norm and survives the norm's in-place update."""
    monkeypatch.setattr(
        sys.modules[MistralLarge4Model.__module__],
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        raising=False,
    )

    model = MistralLarge4Model.__new__(MistralLarge4Model)
    nn.Module.__init__(model)
    model.rotary_emb = None
    model.layers = nn.ModuleList([_InPlaceResidualLayer(torch.tensor([[1.0, 1.0]]))])
    model.start_layer, model.end_layer = 0, 1
    model.norm = _InPlaceAddNorm()
    model.aux_hidden_state_layers = (1,)

    final_hidden, aux_hidden_states = model(
        None, torch.tensor([0]), inputs_embeds=torch.tensor([[1.0, 2.0]])
    )

    torch.testing.assert_close(aux_hidden_states[0], torch.tensor([[102.0, 103.0]]))
    torch.testing.assert_close(final_hidden, torch.tensor([[204.0, 206.0]]))
