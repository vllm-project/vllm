# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for the AMD PLE workspace token-count slicing."""

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpNGramEmbedding


class _IdentityEmbedding(nn.Module):
    params_dtype = torch.int64


def _make_layer() -> Qwen4ExpNGramEmbedding:
    layer = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(layer)
    layer.layer_name = "model.layers.1.ple"
    layer.embedding_dim = 2
    layer.ngram_size = 3
    layer.heads_per_ngram = 1
    layer.eos_token_id = 99
    layer.register_buffer("positions_buffer", torch.arange(32, dtype=torch.int64))
    layer.register_buffer(
        "padded_buffer", torch.full((4, 32), layer.eos_token_id, dtype=torch.int64)
    )
    layer.register_buffer("layer_multipliers", torch.tensor([3, 5, 7]))
    layer.register_buffer("ngram_heads_vocab_sizes", torch.tensor([101, 103]))
    layer.register_buffer("ngram_heads_offsets", torch.tensor([0, 101]))
    layer.ngram_embedding = _IdentityEmbedding()
    return layer


@pytest.fixture
def fake_ngram_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    def lookup(ngram_ids: torch.Tensor, output: torch.Tensor, _name: str) -> None:
        output.copy_(ngram_ids)

    monkeypatch.setattr(torch.ops.vllm, "qwen4_exp_amd_ple_ngram_embedding", lookup)


def _full_width_forward(
    layer: Qwen4ExpNGramEmbedding,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
) -> torch.Tensor:
    """Reference: the pre-slice implementation over the full workspace width."""
    input_ids = input_ids.reshape(-1).long()
    query_start_loc = query_start_loc.long()
    num_reqs = query_start_loc.numel() - 1
    num_tokens = input_ids.shape[0]
    positions = layer.positions_buffer[:num_tokens]
    packed = layer.padded_buffer[:num_reqs]
    packed.fill_(layer.eos_token_id)
    request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
    request_indices.clamp_(max=num_reqs - 1)
    columns = (positions - query_start_loc[request_indices]).clamp(
        0, packed.shape[1] - 1
    )
    packed[request_indices, columns] = input_ids
    ngram_context = ngram_context[:num_reqs].to(
        device=input_ids.device, dtype=torch.long
    )
    context = torch.cat([ngram_context, packed], dim=-1)
    positions_2d, position_in_segment = layer._shift_precompute(
        context, layer.eos_token_id
    )
    shifted = [context]
    for shift in range(1, layer.ngram_size):
        shifted.append(
            layer._shift_apply(
                context,
                positions_2d,
                position_in_segment,
                shift,
                layer.eos_token_id,
            )
        )
    adjusted_columns = columns + layer.ngram_size - 1
    id_blocks = []
    for ngram in range(2, layer.ngram_size + 1):
        start = (ngram - 2) * layer.heads_per_ngram
        end = start + layer.heads_per_ngram
        mixed = shifted[0] * layer.layer_multipliers[0]
        for index in range(1, ngram):
            mixed = torch.bitwise_xor(
                mixed, shifted[index] * layer.layer_multipliers[index]
            )
        sizes = layer.ngram_heads_vocab_sizes[start:end]
        offsets = layer.ngram_heads_offsets[start:end]
        ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
        id_blocks.append(ids[request_indices, adjusted_columns])
    ngram_ids = torch.cat(id_blocks, dim=-1)
    output = ngram_ids.new_empty(
        (ngram_ids.shape[0], layer.embedding_dim),
        dtype=layer.ngram_embedding.params_dtype,
    )
    torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
        ngram_ids, output, layer.layer_name
    )
    return output


# Ragged MTP chunk with an EOS inside the first segment, followed by two
# empty graph-padding requests.
RAGGED_MTP = (
    torch.tensor([10, 11, 99, 12, 20, 21]),
    torch.tensor([0, 4, 6, 6, 6], dtype=torch.int32),
)
# Full-graph padding may place every padding token into one dummy request.
DUMMY_ROW = (
    torch.tensor([10, 99, 99, 99, 99, 99, 99, 99]),
    torch.tensor([0, 1, 8, 8, 8], dtype=torch.int32),
)
SINGLE_REQUEST = (
    torch.tensor([10, 11, 99, 12, 20, 21]),
    torch.tensor([0, 6], dtype=torch.int32),
)

NGRAM_CONTEXT = torch.tensor([[7, 8], [18, 19], [99, 99], [99, 99]], dtype=torch.int32)


@pytest.mark.parametrize(
    "input_ids,query_start_loc",
    [RAGGED_MTP, DUMMY_ROW, SINGLE_REQUEST],
)
def test_slice_matches_full_width_workspace(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    fake_ngram_lookup: None,
) -> None:
    layer = _make_layer()
    expected = _full_width_forward(layer, input_ids, query_start_loc, NGRAM_CONTEXT)

    num_tokens = input_ids.shape[0]
    layer.padded_buffer.fill_(7)  # sentinel for untouched columns
    actual = layer(input_ids, query_start_loc, NGRAM_CONTEXT)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all(layer.padded_buffer[:, num_tokens:] == 7)


def test_packed_region_content(
    fake_ngram_lookup: None,
) -> None:
    layer = _make_layer()
    layer.padded_buffer.fill_(7)
    input_ids, query_start_loc = RAGGED_MTP
    layer(input_ids, query_start_loc, NGRAM_CONTEXT)

    assert layer.padded_buffer[0, :6].tolist() == [10, 11, 99, 12, 99, 99]
    assert layer.padded_buffer[1, :6].tolist() == [20, 21, 99, 99, 99, 99]
    assert torch.all(layer.padded_buffer[2:, :6] == 7)
