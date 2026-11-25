# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from vllm.models.common.ops import sequence_parallel as sp_ops
from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
from vllm.models.deepseek_v32.nvidia import mtp as deepseek_v32_mtp
from vllm.models.deepseek_v32.sequence_parallel import zigzag_local_batch
from vllm.sequence import IntermediateTensors


class _IdentityNorm(nn.Module):
    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.variance_epsilon = 1e-5

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _RecordingModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 0

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        self.num_tokens = hidden_states.shape[0]
        return hidden_states


class _RecordingProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 4), requires_grad=False)
        self.num_tokens = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.num_tokens = hidden_states.shape[0]
        return hidden_states[:, :2]


class _RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_tokens = 0

    def forward(self, positions, hidden_states, sp_shards=None):
        self.num_tokens = hidden_states.shape[0]
        if sp_shards is not None:
            # Every rank's rows, rank-major.
            return torch.cat([hidden_states, hidden_states])
        if hidden_states.shape[0] < positions.shape[0]:
            hidden_states = torch.cat([hidden_states, hidden_states])[: len(positions)]
        return hidden_states


class _SequenceParallelMTPBlock:
    use_sequence_parallel = True

    def __call__(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ):
        assert residual is None
        return hidden_states * 2, hidden_states * 3


def _mock_sequence_parallel_collectives(monkeypatch, module):
    monkeypatch.setattr(
        module,
        "sp_reduce_scatter",
        lambda tensor: tensor.chunk(2, dim=0)[0],
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "sp_shard",
        lambda tensor: torch.nn.functional.pad(tensor, (0, 0, 0, 1))[:2],
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "sp_all_gather",
        lambda tensor: torch.cat([tensor, tensor], dim=0),
    )


def test_decoder_layer_keeps_dense_states_sequence_sharded(monkeypatch):
    layer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = True
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingAttention()
    layer.mlp = _RecordingModule()

    _mock_sequence_parallel_collectives(monkeypatch, deepseek_v32_model)

    positions = torch.arange(3)
    full_hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states = deepseek_v32_model.sp_shard(full_hidden_states)
    hidden_states, residual = layer(positions, hidden_states, residual=None)

    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.self_attn.num_tokens == 2
    assert layer.mlp.num_tokens == 2

    hidden_states, residual = layer(positions, hidden_states, residual)

    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.self_attn.num_tokens == 2
    assert layer.mlp.num_tokens == 2


def test_mtp_projects_sequence_shard_and_restores_full_output(monkeypatch):
    layer = object.__new__(deepseek_v32_mtp.DeepseekV32MultiTokenPredictorLayer)
    nn.Module.__init__(layer)
    layer.enorm = _IdentityNorm()
    layer.hnorm = _IdentityNorm()
    layer.eh_proj = _RecordingProjection()
    layer._eh_plan = None
    object.__setattr__(layer, "mtp_block", _SequenceParallelMTPBlock())
    norm = Mock(
        side_effect=lambda hidden_states, residual: (hidden_states + residual, None)
    )
    object.__setattr__(layer, "shared_head", SimpleNamespace(norm=norm))

    monkeypatch.setattr(
        deepseek_v32_mtp,
        "fused_eh_norm",
        lambda positions, inputs_embeds, previous_hidden_states, *args: torch.cat(
            [inputs_embeds, previous_hidden_states], dim=-1
        ),
    )
    monkeypatch.setattr(deepseek_v32_mtp, "run_glm52_plan", lambda *args: None)
    _mock_sequence_parallel_collectives(monkeypatch, deepseek_v32_mtp)

    inputs_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states, recycled_hidden_states = layer(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        previous_hidden_states=torch.zeros_like(inputs_embeds),
        inputs_embeds=inputs_embeds,
    )

    sharded_states = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, 1))[:2]
    expected = torch.cat([sharded_states * 5, sharded_states * 5])[:3]
    assert layer.eh_proj.num_tokens == 2
    torch.testing.assert_close(hidden_states, expected)
    torch.testing.assert_close(recycled_hidden_states, expected)
    norm.assert_called_once()


class _PartialSumLayer(nn.Module):
    """Returns a per-TP-rank partial MLP output, like DeepseekV32DecoderLayer."""

    def forward(self, positions, hidden_states, residual, attn_in=None, sp_shards=None):
        assert residual is None
        return torch.ones_like(hidden_states), hidden_states + 1


def test_pp_stage_sends_tp_reduced_residual_stream(monkeypatch):
    """PP send/recv needs TP-replicated tensors, so a middle stage must reduce
    its partial hidden_states and send the full residual stream only."""
    model = object.__new__(deepseek_v32_model.DeepseekV32Model)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_PartialSumLayer()])
    model.start_layer, model.end_layer = 0, 1
    model.use_sequence_parallel = False
    model.sequence_parallel = False
    model.shardq_sequence_parallel = False
    model.aux_hidden_state_layers = ()
    model.pp_buffers = {}

    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    monkeypatch.setattr(deepseek_v32_model, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        deepseek_v32_model,
        "tensor_model_parallel_all_reduce",
        lambda tensor: tensor * 2,
    )

    received = torch.full((3, 2), 5.0)
    output = model(
        None, torch.arange(3), IntermediateTensors({"hidden_states": received})
    )

    assert set(output.tensors) == {"hidden_states"}
    # residual (5 + 1) + partial (1) all-reduced over 2 TP ranks.
    torch.testing.assert_close(output["hidden_states"], torch.full((3, 2), 8.0))


def _zigzag_shards(monkeypatch, num_tokens, world_size, rank):
    monkeypatch.setattr(
        sp_ops, "get_tensor_model_parallel_world_size", lambda: world_size
    )
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_rank", lambda: rank)
    return sp_ops.zigzag_shards(num_tokens, torch.device("cpu"))


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("num_tokens", [1, 7, 16, 37])
def test_zigzag_shards_balance_ranks_and_restore_batch_order(
    monkeypatch, world_size, num_tokens
):
    """Every rank holds one early and one late chunk; gathering them rank-major
    and applying gather_index restores the batch."""
    shards = [
        _zigzag_shards(monkeypatch, num_tokens, world_size, rank)
        for rank in range(world_size)
    ]
    chunk = sp_ops.zigzag_chunk_size(num_tokens, world_size)
    rank_major = shards[0].rank_major_rows
    for rank, rank_shards in enumerate(shards):
        local = rank_major[rank * 2 * chunk : (rank + 1) * 2 * chunk]
        assert local.tolist() == [
            *range(rank * chunk, (rank + 1) * chunk),
            *range(
                (2 * world_size - 1 - rank) * chunk, (2 * world_size - rank) * chunk
            ),
        ]
        assert (
            rank_shards.local_rows.tolist() == local.clamp(max=num_tokens - 1).tolist()
        )
    torch.testing.assert_close(
        rank_major[shards[0].gather_index], torch.arange(num_tokens)
    )

    is_padding = torch.arange(num_tokens) >= num_tokens - 1
    mask = sp_ops.zigzag_padding_mask(is_padding, num_tokens, shards[0])
    assert mask.tolist() == (rank_major >= num_tokens - 1).tolist()


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("world_size", [2, 4])
def test_zigzag_local_batch_keeps_each_token_context(per_token, world_size):
    """Each rank's local requests attend exactly the context their tokens have
    in the step: decodes, a split spec-decode request and chunked prefills."""
    query_lens = np.array([1, 1, 4, 3, 40, 9])
    seq_lens = np.array([10, 20, 30, 3, 100, 9])
    query_start_loc = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    num_tokens = int(query_lens.sum())
    num_tokens_padded = num_tokens + 5
    contexts = {}
    for req, (start, end) in enumerate(zip(query_start_loc, query_start_loc[1:])):
        for row in range(start, end):
            contexts[row] = (req, seq_lens[req] - (end - 1 - row))

    seen = []
    for rank in range(world_size):
        local = zigzag_local_batch(
            query_start_loc,
            num_tokens,
            num_tokens_padded,
            world_size,
            rank,
            per_token,
        )
        chunk = -(-num_tokens_padded // (2 * world_size))
        rows = [
            row
            for c in (rank, 2 * world_size - 1 - rank)
            for row in range(c * chunk, min((c + 1) * chunk, num_tokens))
        ]
        assert local.num_tokens == len(rows) == local.query_start_loc[-1]
        if per_token:
            assert len(local.req_indices) == len(rows)
        else:
            assert len(local.req_indices) <= len(query_lens) + 1
        for k, req in enumerate(local.req_indices):
            start, end = local.query_start_loc[k : k + 2]
            local_seq_len = seq_lens[req] - local.tails[k]
            for offset, row in enumerate(rows[start:end]):
                context = local_seq_len - (end - start - 1 - offset)
                assert contexts[row] == (req, context)
        seen += rows
    assert sorted(seen) == list(range(num_tokens))


def test_decoder_layer_runs_mlp_on_every_zigzag_row(monkeypatch):
    """Zigzag padding rows sit between the ranks' rows, so the MLP must see the
    whole rank-major gather rather than the step's first rows."""
    layer = object.__new__(deepseek_v32_model.DeepseekV32DecoderLayer)
    nn.Module.__init__(layer)
    layer.use_sequence_parallel = False
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attn = _RecordingAttention()
    layer.mlp = _RecordingModule()

    _mock_sequence_parallel_collectives(monkeypatch, deepseek_v32_model)

    positions = torch.arange(3)
    sp_shards = _zigzag_shards(monkeypatch, 3, world_size=2, rank=0)
    hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
    hidden_states, residual = layer(
        positions, hidden_states, residual=None, sp_shards=sp_shards
    )

    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.self_attn.num_tokens == 2
    assert layer.mlp.num_tokens == 4


@pytest.mark.parametrize("per_token", [False, True])
def test_model_state_builds_metadata_over_rank_tokens(monkeypatch, per_token):
    """The local metadata carries each local request's own block table row,
    context length and slots, with the rank's padding rows writing nothing."""
    from vllm.models.deepseek_v32.nvidia import model_state as sp_model_state

    world_size, rank = 2, 1
    monkeypatch.setattr(
        sp_ops, "get_tensor_model_parallel_world_size", lambda: world_size
    )
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_rank", lambda: rank)
    built = {}
    monkeypatch.setattr(
        sp_model_state, "build_attn_metadata", lambda **kwargs: built.update(kwargs)
    )

    state = object.__new__(sp_model_state.DeepseekV32SequenceParallelModelState)
    state.world_size, state.rank, state.device = world_size, rank, torch.device("cpu")
    state.max_local_tokens = 8
    state.max_local_reqs = 8
    state.query_start_loc = torch.zeros(9, dtype=torch.int32)
    state.seq_lens = torch.zeros(8, dtype=torch.int32)
    state.positions = torch.zeros(8, dtype=torch.int64)
    state.block_tables = {}
    state.slot_mappings = None

    # A decode, then a 4-token chunk of a prefill with 6 tokens of context.
    query_start_loc = np.array([0, 1, 5], dtype=np.int32)
    seq_lens = torch.tensor([7, 10], dtype=torch.int32)
    num_tokens, num_tokens_padded = 5, 6
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens_padded,
        query_start_loc_np=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        positions=torch.tensor([6, 6, 7, 8, 9, 0]),
        is_prefilling_np=np.array([False, True]),
        idx_mapping_np=np.array([3, 4]),
    )
    block_tables = (torch.tensor([[30, 31], [40, 41]], dtype=torch.int32),)
    slot_mappings = torch.tensor([[100, 101, 102, 103, 104, -1]])

    state._prepare_local_attn(
        input_batch,
        per_token,
        block_tables,
        slot_mappings,
        [[Mock()]],
        kv_cache_config=None,
        for_capture=False,
    )

    # Rank 1 owns rows 1 and 4 of the 2-row chunks [0 1 | 2 3 | 4 5 | 6 7]:
    # chunk 1 (rows 2, 3) and chunk 2 (rows 4, 5); row 5 only pads the step.
    if per_token:
        # One request per row, then a zero-length one for the padding row.
        assert built["query_start_loc_cpu"].tolist() == [0, 1, 2, 3, 3]
        assert built["seq_lens"].tolist() == [8, 9, 10, 0]
        assert built["block_tables"][0].tolist() == [[40, 41]] * 3 + [[30, 31]]
        assert built["num_tokens"] == 4
    else:
        assert built["query_start_loc_cpu"].tolist() == [0, 3]
        assert built["seq_lens"].tolist() == [10]
        assert built["block_tables"][0].tolist() == [[40, 41]]
        assert built["num_tokens"] == 3
    assert built["slot_mappings"][0].tolist() == [102, 103, 104, -1]
    assert built["positions"].tolist() == [7, 8, 9, 0]
    assert built["is_prefilling"][:3].all()
