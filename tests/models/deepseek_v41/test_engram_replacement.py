# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Engram ``per_token`` lookup mode (DeepSeek V4.1)."""

from types import SimpleNamespace
from unittest.mock import patch

import pydantic
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from vllm.config.engram import EngramConfig
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.models.deepseek_v41.common.engram_replacement import (
    PerTokenTable,
    ReplacementShard,
    is_ngram_table_weight,
    per_token_comp_ids,
)

N_COLS, HEAD_DIM, VOCAB = 24, 256, 50  # n_hash_cols, head_dim, compressed vocab
LAYERS = [1, 14]


def _table(seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, N_COLS, HEAD_DIM, generator=gen).bfloat16()


def _per_token_table(tp: int = 1, rank: int = 0) -> PerTokenTable:
    return PerTokenTable(ReplacementShard(N_COLS, HEAD_DIM, tp, rank), VOCAB)


def _model_config(arch: str = "DeepseekV41ForCausalLM") -> SimpleNamespace:
    return SimpleNamespace(
        architecture=arch,
        hf_text_config=SimpleNamespace(engram_layer_ids=LAYERS),
    )


ROCM = SimpleNamespace(is_rocm=lambda: True)
NOT_ROCM = SimpleNamespace(is_rocm=lambda: False)


# --- mode parsing and config validation ------------------------------------------


def test_default_mode_is_ngram() -> None:
    assert EngramConfig().mode == "ngram"
    assert EngramConfig(**{}).mode == "ngram"  # --engram-config '{}'


def test_per_token_parses_from_a_json_style_dict() -> None:
    assert EngramConfig(**{"mode": "per_token"}).mode == "per_token"


# "zero" was a debug-only mode that has been removed.
@pytest.mark.parametrize("mode", ["bogus", "zero"])
def test_unknown_mode_is_rejected(mode: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        EngramConfig(mode=mode)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_across_dp": True},
        {"dp_shared_memory": True},
        {"use_thp": True},
    ],
)
def test_per_token_rejects_host_and_dp_sharing(kwargs) -> None:
    with pytest.raises(ValueError, match="must be off"):
        EngramConfig(mode="per_token", **kwargs)


def test_ngram_mode_still_accepts_dp_sharing() -> None:
    EngramConfig(embedding_across_dp=True)
    EngramConfig(dp_shared_memory=True)


def test_mode_enters_the_config_hash() -> None:
    assert (
        EngramConfig().compute_hash() != EngramConfig(mode="per_token").compute_hash()
    )


def test_per_token_does_not_share_host_tables_across_dp() -> None:
    parallel = SimpleNamespace(data_parallel_size=2, enable_elastic_ep=False)
    cfg = EngramConfig(mode="per_token")
    cfg.resolve_dp_shared_memory(parallel)
    assert cfg.dp_shared_memory is False
    ngram = EngramConfig()
    ngram.resolve_dp_shared_memory(parallel)
    assert ngram.dp_shared_memory is True  # unchanged default


# --- platform and model handling (consistent with the Qwen4Exp PLE modes) --------


def test_per_token_is_rocm_only() -> None:
    cfg = EngramConfig(mode="per_token")
    with patch("vllm.platforms.current_platform", ROCM):
        cfg.verify_model_config(_model_config())
    with (
        patch("vllm.platforms.current_platform", NOT_ROCM),
        pytest.raises(NotImplementedError, match="only implemented for ROCm"),
    ):
        cfg.verify_model_config(_model_config())


def test_ngram_mode_is_accepted_on_any_platform() -> None:
    with patch("vllm.platforms.current_platform", NOT_ROCM):
        EngramConfig().verify_model_config(_model_config())


def test_per_token_is_rejected_for_qwen4_exp() -> None:
    qwen = SimpleNamespace(
        architecture="Qwen4ExpForCausalLM",
        hf_text_config=SimpleNamespace(ple_layer_ids=[3]),
    )
    with patch("vllm.platforms.current_platform", ROCM):
        with pytest.raises(ValueError, match="only supported for DeepSeek V4.1"):
            EngramConfig(mode="per_token").verify_model_config(qwen)
        EngramConfig().verify_model_config(qwen)


# --- table loading through the standard weight-loading path -------------------------


@pytest.mark.parametrize("tp", [1, 2, 3, 5, 8])
def test_shards_reassemble_the_full_table(tp: int) -> None:
    full = _table()
    part = -(-N_COLS // tp)
    local = []
    for rank in range(tp):
        table = _per_token_table(tp, rank)
        assert table.weight.dtype == torch.bfloat16
        assert tuple(table.weight.shape) == (VOCAB, part, HEAD_DIM)
        table.weight.weight_loader(table.weight, full)
        local.append(table.weight.data.clone())
    # All-gather over TP and crop to n_hash_cols, as Engram.embed does; the
    # padding of the last shard is zero.
    gathered = torch.cat(local, dim=1)
    assert torch.equal(gathered[:, :N_COLS], full)
    assert not gathered[:, N_COLS:].any()


def test_loader_overwrites_uninitialized_padding() -> None:
    """The last TP shard overruns n_hash_cols; its padding must be zeroed even
    though the parameter is allocated uninitialized."""
    table = _per_token_table(tp=5, rank=4)  # part = 5, head_start = 20, used = 4
    table.weight.data.fill_(7.0)
    table.weight.weight_loader(table.weight, _table())
    assert not table.weight.data[:, 4:].any()
    assert torch.equal(table.weight.data[:, :4], _table()[:, 20:24])


def test_table_loads_through_auto_weights_loader_like_any_weight(tmp_path) -> None:
    """Checkpoint style: a bf16 safetensors tensor named ``per_token_table.weight``
    below the Engram module, loaded by the generic weights loader."""

    class _Engram(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.per_token_table = _per_token_table()
            self.other = nn.Linear(2, 2, bias=False)

    full = _table()
    path = str(tmp_path / "model.safetensors")
    save_file({"per_token_table.weight": full, "other.weight": torch.eye(2)}, path)
    engram = _Engram()
    loaded = AutoWeightsLoader(engram).load_weights(load_file(path).items())
    assert loaded == {"per_token_table.weight", "other.weight"}
    assert torch.equal(engram.per_token_table.weight.data, full)


def test_float_tables_are_rejected() -> None:
    table = _per_token_table()
    with pytest.raises(ValueError, match="must be bfloat16"):
        table.weight.weight_loader(table.weight, _table().float())


@pytest.mark.parametrize(
    "shape",
    [
        (VOCAB + 1, N_COLS, HEAD_DIM),  # compressed vocab mismatch
        (VOCAB, N_COLS - 1, HEAD_DIM),  # n_hash_cols mismatch
        (VOCAB, N_COLS, HEAD_DIM // 2),  # head_dim mismatch
        (VOCAB, N_COLS * HEAD_DIM),  # wrong rank
    ],
)
def test_wrong_table_shape_is_rejected(shape) -> None:
    table = _per_token_table()
    bad = torch.zeros(shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="expected .*compressed_vocab"):
        table.weight.weight_loader(table.weight, bad)


def test_shard_geometry_has_no_parameters_or_buffers() -> None:
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    assert not list(shard.state_dict())


def test_per_token_table_owns_only_its_weight() -> None:
    assert list(_per_token_table().state_dict()) == ["weight"]


# --- lookup semantics against the checkpoint table -----------------------------------


def test_per_token_fill_matches_indexing() -> None:
    full = _table()
    table = _per_token_table()
    table.weight.weight_loader(table.weight, full)
    ids = torch.tensor([0, 7, 7, VOCAB - 1, 3])
    rows = torch.full((ids.numel(), N_COLS, HEAD_DIM), 3.0, dtype=torch.bfloat16)
    table.fill(rows, ids)
    assert torch.equal(rows, full[ids])


def test_per_token_fill_is_per_rank_columns() -> None:
    full = _table()
    ids = torch.tensor([4, 4, 9])
    tp = 5
    for rank in range(tp):
        shard = ReplacementShard(N_COLS, HEAD_DIM, tp, rank)
        part = shard.part_n_hash_cols
        table = PerTokenTable(shard, VOCAB)
        table.weight.weight_loader(table.weight, full)
        rows = torch.empty(ids.numel(), part, HEAD_DIM, dtype=torch.bfloat16)
        table.fill(rows, ids)
        want = torch.zeros(ids.numel(), part, HEAD_DIM, dtype=torch.bfloat16)
        cols = full[ids][:, shard.head_start : shard.head_start + part]
        want[:, : cols.shape[1]] = cols
        assert torch.equal(rows, want)


def test_per_token_fill_needs_ids_for_every_row() -> None:
    table = _per_token_table()
    rows = torch.empty(4, N_COLS, HEAD_DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one compressed token id per row"):
        table.fill(rows, None)
    with pytest.raises(ValueError, match="one compressed token id per row"):
        table.fill(rows, torch.tensor([1, 2]))


def test_per_token_fill_rejects_mismatched_row_shape() -> None:
    table = _per_token_table()
    rows = torch.zeros(2, N_COLS + 1, HEAD_DIM, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        table.fill(rows, torch.tensor([0, 1]))


# --- compressed token ids ---------------------------------------------------------


def test_comp_ids_follow_the_token_map_and_dead_positions_map_to_zero() -> None:
    token_map = torch.arange(100, 0, -1, dtype=torch.int32)  # token t -> 100 - t
    ids = torch.tensor([3, 0, 7, 5])
    dead = torch.tensor([False, False, True, False])  # position 2: image span
    out = per_token_comp_ids(token_map, ids, dead)
    assert out.dtype == torch.int64
    assert out.tolist() == [97, 100, 100, 95]


def test_comp_ids_on_a_profile_run_are_all_dead() -> None:
    """Profile runs use dummy hashes (nothing is kept), yet per_token still needs
    valid ids for every row."""
    token_map = torch.arange(100, 0, -1, dtype=torch.int32)
    ids = torch.tensor([2**30, 5, -1], dtype=torch.int32)  # arbitrary dummy ids
    dead = torch.ones(3, dtype=torch.bool)
    assert per_token_comp_ids(token_map, ids, dead).tolist() == [100, 100, 100]


def test_comp_ids_clamp_ids_beyond_the_token_map() -> None:
    token_map = torch.arange(10)
    ids = torch.tensor([9, 10, 1000])
    dead = torch.zeros(3, dtype=torch.bool)
    assert per_token_comp_ids(token_map, ids, dead).tolist() == [9, 9, 9]


def test_replaced_checkpoint_tables_are_recognised() -> None:
    assert is_ngram_table_weight("model.layers.1.engram.embed_tokens.weight")
    assert not is_ngram_table_weight("model.layers.1.engram.wkv.weight")
    assert not is_ngram_table_weight("model.layers.1.self_attn.qkv_proj.weight")


def test_engram_prepare_embeddings_fills_rows_from_the_per_token_table() -> None:
    """Engram.prepare_embeddings fills the staged rows from the table without
    touching any n-gram table (no table exists in this mode)."""
    from vllm.models.deepseek_v41.common.engram import Engram

    full = _table()
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    engram = object.__new__(Engram)
    torch.nn.Module.__init__(engram)
    engram.embed_tokens = shard
    engram.staged_rows = torch.full((8, N_COLS, HEAD_DIM), 5.0, dtype=torch.bfloat16)
    engram.per_token_table = PerTokenTable(shard, VOCAB)
    engram.per_token_table.weight.weight_loader(engram.per_token_table.weight, full)
    hashes = torch.zeros(4, N_COLS, dtype=torch.int32)
    ids = torch.tensor([5, 5, 2, 0])

    engram.prepare_embeddings(hashes, ids)
    assert torch.equal(engram.staged_rows[:4], full[ids])
    assert (engram.staged_rows[4:] == 5).all()  # rows past the batch untouched
