# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 ShardQ run planning (vllm/models/deepseek_v41/shardq.py)."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.models.deepseek_v41 import shardq as dsv41_shardq
from vllm.v1.attention.ops import shardq

WORLD = 4
CPU = torch.device("cpu")


def _config(min_tokens: int = 1024, topk: int = 512, num_spec: int = 0):
    speculative_config = (
        SimpleNamespace(num_speculative_tokens=num_spec, parallel_drafting=False)
        if num_spec
        else None
    )
    return SimpleNamespace(
        speculative_config=speculative_config,
        attention_config=SimpleNamespace(sparse_mla_shard_q_min_tokens=min_tokens),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_topk=topk),
        ),
    )


@pytest.fixture
def tp_rank(monkeypatch):
    state = SimpleNamespace(rank=1)
    monkeypatch.setattr(
        shardq,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=WORLD, rank_in_group=state.rank),
    )
    return state


def _plan_on_rank(cm, config, tp_rank, rank):
    """Rebuild the step's plan as ``rank`` sees it."""
    tp_rank.rank = rank
    if hasattr(cm, "_dsv41_shardq"):
        delattr(cm, "_dsv41_shardq")
    return dsv41_shardq.build_dsv41_shardq(cm, config, CPU)


def _metadata(seq_lens, query_lens):
    return create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=16,
        device=CPU,
    )


def test_runs_follow_zigzag_chunks(tp_rank):
    """A rank's zigzag chunks become two prefill runs ending at their tokens."""
    seq_len = 8192
    cm = _metadata([seq_len], [seq_len])
    plan = dsv41_shardq.build_dsv41_shardq(cm, _config(), CPU)
    assert plan is not None
    sharded = plan.sharded_cm

    # Eight 1024-token chunks: rank 1 owns chunks 1 ([1024, 2048)) and 6
    # ([6144, 7168)).
    assert sharded.num_reqs == 2
    assert sharded.num_actual_tokens == 2048
    assert sharded.max_query_len == 1024
    assert sharded.seq_lens.tolist() == [2048, 7168]
    assert sharded.query_start_loc.tolist() == [0, 1024, 2048]

    expected = np.r_[np.arange(1024, 2048), np.arange(6144, 7168)]
    np.testing.assert_array_equal(
        sharded.slot_mapping.numpy(), cm.slot_mapping.numpy()[expected]
    )
    # The exchange metadata carries the whole step's tokens grouped by rank,
    # and the plan is reusable by every builder of the step.
    assert plan.exchange.num_decode_tokens == 0
    assert sum(plan.exchange.num_tokens_per_rank) == seq_len
    assert plan.exchange.token_ids.numel() == seq_len
    assert dsv41_shardq.build_dsv41_shardq(cm, _config(), CPU) is plan


def test_decode_prefix_stays_head_sharded(tp_rank):
    """Decode tokens stay in place; only the post-decode tokens are split."""
    cm = _metadata([1, 4096], [1, 4096])
    plan = dsv41_shardq.build_dsv41_shardq(cm, _config(), CPU)
    assert plan is not None
    assert plan.exchange.num_decode_tokens == 1
    # Rank 1 owns chunks 1 and 6 of the second request's eight chunks.
    assert plan.sharded_cm.seq_lens.tolist() == [1024, 3584]
    assert plan.sharded_cm.query_start_loc.tolist() == [0, 512, 1024]


def test_chunked_prefill_below_window_still_splits(tp_rank):
    """A main prefill step can be no wider than the sliding window without
    being the bounded-replay batch, so the split must stay enabled."""
    cm = _metadata([2048] * 8, [128] * 8)
    for rank in range(WORLD):
        assert _plan_on_rank(cm, _config(), tp_rank, rank) is not None


def test_short_batch_stays_head_sharded(tp_rank):
    cm = _metadata([512], [512])
    assert dsv41_shardq.build_dsv41_shardq(cm, _config(), CPU) is None


def test_single_token_run_stays_head_sharded(tp_rank):
    """Eight tokens over eight chunks are one-token runs, which the builders
    would read as decodes; the shared query length rejects the split."""
    cm = _metadata([8], [8])
    plan = dsv41_shardq.build_dsv41_shardq(cm, _config(min_tokens=8, topk=1), CPU)
    assert plan is None


def test_spec_decode_sized_run_stays_head_sharded(tp_rank):
    """A run no wider than a spec-decode request is read as a decode."""
    # 16 tokens over eight zigzag chunks are runs of two tokens; with one
    # speculative token anything up to two tokens is a decode.
    cm = _metadata([16], [16])
    config = _config(min_tokens=16, topk=1, num_spec=1)
    assert dsv41_shardq.build_dsv41_shardq(cm, config, CPU) is None


def test_every_rank_takes_the_same_decision(tp_rank):
    """Ranks must agree: one rank staying head-sharded would leave the others
    waiting in the exchange.  The width check reads the shared query lengths,
    not this rank's runs.

    15 tokens over eight chunks leave rank 0 a one-token chunk while ranks 1-3
    get two-token chunks, so a rank-local check would disagree; 17 tokens give
    every rank at least two tokens per chunk.
    """
    uneven = _metadata([4500, 15], [4500, 15])
    config = _config(min_tokens=15, topk=1)
    # The generic split alone would let the indexer compact here.
    assert shardq.build_shardq_split(uneven, config) is not None
    assert all(
        _plan_on_rank(uneven, config, tp_rank, rank) is None
        for rank in range(WORLD)
    )

    wide = _metadata([4096, 17], [4096, 17])
    assert all(
        _plan_on_rank(wide, _config(min_tokens=17, topk=1), tp_rank, rank)
        is not None
        for rank in range(WORLD)
    )


def test_eligibility_rejects_runs_the_builders_read_as_decodes(tp_rank):
    """The hook turns a generic split down when a request is too narrow: 15
    tokens over eight chunks leave rank 0 a one-token run."""
    config = _config(min_tokens=15, topk=1)
    uneven = _metadata([4500, 15], [4500, 15])
    split = shardq.build_shardq_split(uneven, config)
    assert split is not None
    assert not dsv41_shardq.dsv41_shardq_eligible(uneven, config, split)

    config = _config(min_tokens=17, topk=1)
    wide = _metadata([4096, 17], [4096, 17])
    split = shardq.build_shardq_split(wide, config)
    assert split is not None
    assert dsv41_shardq.dsv41_shardq_eligible(wide, config, split)


class _FakeShardQCache:
    use_shardq = True
    shardq_eligibility = staticmethod(dsv41_shardq.dsv41_shardq_eligible)


def test_indexer_reads_the_model_eligibility_hook():
    """The indexer and the attention must reject the same splits."""
    from vllm.v1.attention.backends.mla.indexer import _resolve_shardq_eligibility

    layers = {"indexer": _FakeShardQCache()}
    assert (
        _resolve_shardq_eligibility(layers, ["indexer"])
        is dsv41_shardq.dsv41_shardq_eligible
    )
    assert (
        _resolve_shardq_eligibility(
            {"indexer": SimpleNamespace(use_shardq=False)}, ["indexer"]
        )
        is None
    )


class _FakeShardQAttn:
    def __init__(self, capable: bool) -> None:
        self.shardq_capable = capable
        self.use_shardq = False
        self.workspace_allocations = 0

    def set_shardq_enabled(self, enabled: bool) -> None:
        self.use_shardq = bool(enabled and self.shardq_capable)

    def allocate_shardq_workspace(self) -> None:
        if self.use_shardq:
            self.workspace_allocations += 1


class _FakeShardQLayer:
    def __init__(self, capable: bool) -> None:
        self.attn = _FakeShardQAttn(capable)


def test_finalize_shardq_allocates_once_when_enabled():
    """The model applies its replay decision and allocates one workspace."""
    from vllm.models.deepseek_v41.nvidia import model as dsv41_model

    layers = [_FakeShardQLayer(True), _FakeShardQLayer(True), _FakeShardQLayer(False)]
    dsv41_model._finalize_shardq(layers, True)
    assert [layer.attn.use_shardq for layer in layers] == [True, True, False]
    assert sum(layer.attn.workspace_allocations for layer in layers) == 1


def test_finalize_shardq_allocates_nothing_when_disabled():
    from vllm.models.deepseek_v41.nvidia import model as dsv41_model

    layers = [_FakeShardQLayer(True), _FakeShardQLayer(True)]
    dsv41_model._finalize_shardq(layers, False)
    assert not any(layer.attn.use_shardq for layer in layers)
    assert sum(layer.attn.workspace_allocations for layer in layers) == 0


@dataclass(eq=False)
class _FakeAttentionGroup:
    layer_names: list[str]
    kernel_block_size: int | None = None
    metadata_builders: list = field(default_factory=list)

    def create_metadata_builders(self, vllm_config, device, kernel_block_size=None):
        self.metadata_builders = [SimpleNamespace(use_shardq=True), SimpleNamespace()]


def test_replay_builders_stay_head_sharded():
    """Replay is a microbatch for late layers: its builders must not build an
    unused ShardQ view when another layer of the model enables ShardQ."""
    from vllm.models.deepseek_v41.nvidia import model_state as dsv41_model_state

    groups = [
        [
            _FakeAttentionGroup(["main.attn", "late.attn"]),
            _FakeAttentionGroup(["other.attn"]),
        ]
    ]
    state = SimpleNamespace(vllm_config=None, device=None)
    copies = dsv41_model_state.DeepseekV41ModelState._copy_attn_groups(
        state, groups, {"late.attn"}
    )

    assert [group.layer_names for group in copies[0]] == [["late.attn"]]
    assert copies[0][0].metadata_builders[0].use_shardq is False
    assert not hasattr(copies[0][0].metadata_builders[1], "use_shardq")


def test_short_context_fill_follows_shardq_rows():
    """The short-context top-k fill must write the compact ShardQ rows with
    their own positions, not the batch's leading rows."""
    from vllm.models.deepseek_v41.attention import _short_context_topk_rows

    positions = torch.arange(10, 20)
    metadata = SimpleNamespace(
        num_decode_tokens=2, num_prefill_tokens=8, prefill=None
    )
    rows, num_tokens = _short_context_topk_rows(positions, metadata)
    assert num_tokens == 10
    torch.testing.assert_close(rows, positions)

    metadata.prefill = SimpleNamespace(
        shardq_gather_indices=torch.tensor([0, 1, 5, 6])
    )
    rows, num_tokens = _short_context_topk_rows(positions, metadata)
    assert num_tokens == 4
    torch.testing.assert_close(rows, torch.tensor([10, 11, 15, 16]))
