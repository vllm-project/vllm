# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Engram ``zero`` / ``per_token`` lookup modes (DeepSeek V4.1)."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pydantic
import pytest
import torch

from vllm.config.engram import EngramConfig, per_token_table_path
from vllm.models.deepseek_v41.common.engram_replacement import (
    EngramReplacement,
    ReplacementShard,
    is_ngram_table_weight,
    load_per_token_table,
    per_token_comp_ids,
)

N_COLS, HEAD_DIM, VOCAB = 24, 256, 50  # n_hash_cols, head_dim, compressed vocab
LAYERS = [1, 14]


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bits (uint16), round to nearest even, in numpy."""
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _write_table(directory, layer: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + layer)
    bits = _bf16_bits(rng.standard_normal((VOCAB, N_COLS, HEAD_DIM)))
    np.save(per_token_table_path(str(directory), layer), bits)
    return bits


def _bits(t: torch.Tensor) -> np.ndarray:
    """bf16 tensor -> uint16 bits, to compare against numpy references."""
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def _model_config(arch: str = "DeepseekV41ForCausalLM") -> SimpleNamespace:
    return SimpleNamespace(
        architecture=arch,
        hf_text_config=SimpleNamespace(engram_layer_ids=LAYERS),
    )


ROCM = SimpleNamespace(is_rocm=lambda: True)
NOT_ROCM = SimpleNamespace(is_rocm=lambda: False)


# --- mode parsing and config validation ------------------------------------------


def test_default_mode_is_ngram() -> None:
    cfg = EngramConfig()
    assert cfg.mode == "ngram" and cfg.replacement_tables is None
    assert EngramConfig(**{}).mode == "ngram"  # --engram-config '{}'


def test_modes_parse_from_a_json_style_dict(tmp_path) -> None:
    assert EngramConfig(**{"mode": "zero"}).mode == "zero"
    cfg = EngramConfig(**{"mode": "per_token", "replacement_tables": str(tmp_path)})
    assert cfg.mode == "per_token" and cfg.replacement_tables == str(tmp_path)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        EngramConfig(mode="bogus")  # type: ignore[arg-type]


def test_per_token_requires_replacement_tables() -> None:
    with pytest.raises(ValueError, match="requires replacement_tables"):
        EngramConfig(mode="per_token")


@pytest.mark.parametrize("mode", ["ngram", "zero"])
def test_replacement_tables_only_with_per_token(mode: str, tmp_path) -> None:
    with pytest.raises(ValueError, match="only used with mode 'per_token'"):
        EngramConfig(mode=mode, replacement_tables=str(tmp_path))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_across_dp": True},
        {"dp_shared_memory": True},
        {"use_thp": True},
    ],
)
def test_replacement_modes_reject_host_and_dp_sharing(kwargs, tmp_path) -> None:
    with pytest.raises(ValueError, match="must be off"):
        EngramConfig(mode="zero", **kwargs)
    with pytest.raises(ValueError, match="must be off"):
        EngramConfig(mode="per_token", replacement_tables=str(tmp_path), **kwargs)


def test_ngram_mode_still_accepts_dp_sharing() -> None:
    EngramConfig(embedding_across_dp=True)
    EngramConfig(dp_shared_memory=True)


def test_mode_and_tables_enter_the_config_hash(tmp_path) -> None:
    hashes = {
        EngramConfig().compute_hash(),
        EngramConfig(mode="zero").compute_hash(),
        EngramConfig(
            mode="per_token", replacement_tables=str(tmp_path / "a")
        ).compute_hash(),
        EngramConfig(
            mode="per_token", replacement_tables=str(tmp_path / "b")
        ).compute_hash(),
    }
    assert len(hashes) == 4


@pytest.mark.parametrize("mode", ["zero", "per_token"])
def test_replacement_modes_do_not_share_host_tables_across_dp(
    mode: str, tmp_path
) -> None:
    parallel = SimpleNamespace(data_parallel_size=2, enable_elastic_ep=False)
    kwargs = {"replacement_tables": str(tmp_path)} if mode == "per_token" else {}
    cfg = EngramConfig(mode=mode, **kwargs)
    cfg.resolve_dp_shared_memory(parallel)
    assert cfg.dp_shared_memory is False
    ngram = EngramConfig()
    ngram.resolve_dp_shared_memory(parallel)
    assert ngram.dp_shared_memory is True  # unchanged default


# --- platform and model handling (consistent with the Qwen4Exp PLE modes) --------


@pytest.mark.parametrize("mode", ["zero", "per_token"])
def test_replacement_modes_are_rocm_only(mode: str, tmp_path) -> None:
    for layer in LAYERS:
        _write_table(tmp_path, layer)
    kwargs = {"replacement_tables": str(tmp_path)} if mode == "per_token" else {}
    cfg = EngramConfig(mode=mode, **kwargs)
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


def test_replacement_modes_are_rejected_for_qwen4_exp() -> None:
    qwen = SimpleNamespace(
        architecture="Qwen4ExpForCausalLM",
        hf_text_config=SimpleNamespace(ple_layer_ids=[3]),
    )
    with patch("vllm.platforms.current_platform", ROCM):
        with pytest.raises(ValueError, match="only supported for DeepSeek V4.1"):
            EngramConfig(mode="zero").verify_model_config(qwen)
        EngramConfig().verify_model_config(qwen)


def test_per_token_requires_every_layer_table(tmp_path) -> None:
    cfg = EngramConfig(mode="per_token", replacement_tables=str(tmp_path))
    with patch("vllm.platforms.current_platform", ROCM):
        with pytest.raises(FileNotFoundError, match="U_L1_bf16.npy"):
            cfg.verify_model_config(_model_config())
        _write_table(tmp_path, 1)
        with pytest.raises(FileNotFoundError, match="U_L14_bf16.npy") as e:
            cfg.verify_model_config(_model_config())
        assert "U_L1_bf16.npy" not in str(e.value)
        _write_table(tmp_path, 14)
        cfg.verify_model_config(_model_config())


def test_per_token_tables_must_be_a_directory(tmp_path) -> None:
    cfg = EngramConfig(mode="per_token", replacement_tables=str(tmp_path / "nope"))
    with (
        patch("vllm.platforms.current_platform", ROCM),
        pytest.raises(FileNotFoundError, match="not a directory"),
    ):
        cfg.verify_model_config(_model_config())


# --- table loading and validation -----------------------------------------------


def test_table_file_name_matches_the_documented_layout() -> None:
    assert per_token_table_path("/d", 14) == "/d/U_L14_bf16.npy"


@pytest.mark.parametrize("tp", [1, 2, 3, 5, 8])
def test_shards_reassemble_the_full_table(tp: int, tmp_path) -> None:
    bits = _write_table(tmp_path, 1)
    shards = [ReplacementShard(N_COLS, HEAD_DIM, tp, r) for r in range(tp)]
    part = -(-N_COLS // tp)
    assert all(s.part_n_hash_cols == part for s in shards)
    assert [s.head_start for s in shards] == [r * part for r in range(tp)]
    local = []
    for s in shards:
        rep = EngramReplacement("per_token", 1, s, VOCAB, str(tmp_path))
        assert rep.table is not None
        assert rep.table.dtype == torch.bfloat16
        assert tuple(rep.table.shape) == (VOCAB, part, HEAD_DIM)
        local.append(_bits(rep.table))
    # All-gather over TP and crop to n_hash_cols, as Engram.embed does; the
    # padding of the last shard is zero.
    gathered = np.concatenate(local, axis=1)
    assert np.array_equal(gathered[:, :N_COLS], bits)
    assert not gathered[:, N_COLS:].any()


def test_loader_reads_only_the_requested_columns(tmp_path) -> None:
    bits = _write_table(tmp_path, 1)
    t = load_per_token_table(
        per_token_table_path(str(tmp_path), 1),
        VOCAB,
        N_COLS,
        HEAD_DIM,
        head_start=6,
        part_n_hash_cols=3,
    )
    assert np.array_equal(_bits(t), bits[:, 6:9])


def test_missing_table_file_is_reported(tmp_path) -> None:
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    with pytest.raises(FileNotFoundError, match="U_L1_bf16.npy"):
        EngramReplacement("per_token", 1, shard, VOCAB, str(tmp_path))


def test_per_token_without_a_table_directory_is_rejected() -> None:
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    with pytest.raises(ValueError, match="requires replacement_tables"):
        EngramReplacement("per_token", 1, shard, VOCAB, None)


def test_float_tables_are_rejected(tmp_path) -> None:
    np.save(
        per_token_table_path(str(tmp_path), 1),
        np.zeros((VOCAB, N_COLS, HEAD_DIM), dtype=np.float32),
    )
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    with pytest.raises(ValueError, match="uint16"):
        EngramReplacement("per_token", 1, shard, VOCAB, str(tmp_path))


@pytest.mark.parametrize(
    "shape",
    [
        (VOCAB + 1, N_COLS, HEAD_DIM),  # compressed vocab mismatch
        (VOCAB, N_COLS - 1, HEAD_DIM),  # n_hash_cols mismatch
        (VOCAB, N_COLS, HEAD_DIM // 2),  # head_dim mismatch
        (VOCAB, N_COLS * HEAD_DIM),  # wrong rank
    ],
)
def test_wrong_table_shape_is_rejected(shape, tmp_path) -> None:
    np.save(per_token_table_path(str(tmp_path), 1), np.zeros(shape, dtype=np.uint16))
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    with pytest.raises(ValueError, match="expected .*compressed_vocab"):
        EngramReplacement("per_token", 1, shard, VOCAB, str(tmp_path))


def test_corrupt_table_file_is_rejected(tmp_path) -> None:
    (tmp_path / "U_L1_bf16.npy").write_bytes(b"not a numpy file")
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    with pytest.raises(ValueError, match="cannot read engram per-token table"):
        EngramReplacement("per_token", 1, shard, VOCAB, str(tmp_path))


def test_not_a_replacement_mode_is_rejected() -> None:
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    with pytest.raises(ValueError, match="not an engram replacement mode"):
        EngramReplacement("ngram", 1, shard, VOCAB, None)


def test_replacement_holds_no_parameters_or_buffers(tmp_path) -> None:
    _write_table(tmp_path, 1)
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    assert not list(shard.state_dict())
    for mode in ("zero", "per_token"):
        rep = EngramReplacement(mode, 1, shard, VOCAB, str(tmp_path))
        assert not list(rep.state_dict()) and not list(rep.parameters())


# --- lookup semantics against a numpy reference ------------------------------------


def test_per_token_fill_matches_numpy_indexing(tmp_path) -> None:
    bits = _write_table(tmp_path, 1)
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    rep = EngramReplacement("per_token", 1, shard, VOCAB, str(tmp_path))
    ids = torch.tensor([0, 7, 7, VOCAB - 1, 3])
    rows = torch.full((ids.numel(), N_COLS, HEAD_DIM), 3.0, dtype=torch.bfloat16)
    rep.fill(rows, ids)
    assert np.array_equal(_bits(rows), bits[ids.numpy()])


def test_per_token_fill_is_per_rank_columns(tmp_path) -> None:
    bits = _write_table(tmp_path, 1)
    ids = torch.tensor([4, 4, 9])
    tp = 5
    for rank in range(tp):
        shard = ReplacementShard(N_COLS, HEAD_DIM, tp, rank)
        part = shard.part_n_hash_cols
        rep = EngramReplacement("per_token", 1, shard, VOCAB, str(tmp_path))
        rows = torch.empty(ids.numel(), part, HEAD_DIM, dtype=torch.bfloat16)
        rep.fill(rows, ids)
        want = np.zeros((ids.numel(), part, HEAD_DIM), dtype=np.uint16)
        cols = bits[ids.numpy()][:, shard.head_start : shard.head_start + part]
        want[:, : cols.shape[1]] = cols
        assert np.array_equal(_bits(rows), want)


def test_per_token_fill_needs_ids_for_every_row(tmp_path) -> None:
    _write_table(tmp_path, 1)
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    rep = EngramReplacement("per_token", 1, shard, VOCAB, str(tmp_path))
    rows = torch.empty(4, N_COLS, HEAD_DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="one compressed token id per row"):
        rep.fill(rows, None)
    with pytest.raises(ValueError, match="one compressed token id per row"):
        rep.fill(rows, torch.tensor([1, 2]))


def test_zero_fill_overwrites_previous_rows() -> None:
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    rep = EngramReplacement("zero", 1, shard, VOCAB, None)
    rows = torch.full((5, N_COLS, HEAD_DIM), 3.0, dtype=torch.bfloat16)
    rep.fill(rows, None)
    assert not rows.any()


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


def test_per_token_fill_rejects_mismatched_row_shape(tmp_path) -> None:
    _write_table(tmp_path, 1)
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    rep = EngramReplacement("per_token", 1, shard, VOCAB, str(tmp_path))
    rows = torch.zeros(2, N_COLS + 1, HEAD_DIM, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        rep.fill(rows, torch.tensor([0, 1]))


def test_engram_prepare_embeddings_dispatches_to_the_replacement(tmp_path) -> None:
    """Engram.prepare_embeddings fills the staged rows from the replacement
    without touching any n-gram table (no table exists in these modes)."""
    from vllm.models.deepseek_v41.common.engram import Engram

    bits = _write_table(tmp_path, 1)
    shard = ReplacementShard(N_COLS, HEAD_DIM, 1, 0)
    engram = object.__new__(Engram)
    torch.nn.Module.__init__(engram)
    engram.embed_tokens = shard
    engram.staged_rows = torch.full((8, N_COLS, HEAD_DIM), 5.0, dtype=torch.bfloat16)
    hashes = torch.zeros(4, N_COLS, dtype=torch.int32)
    ids = torch.tensor([5, 5, 2, 0])

    engram.replacement = EngramReplacement("per_token", 1, shard, VOCAB, str(tmp_path))
    engram.prepare_embeddings(hashes, ids)
    assert np.array_equal(_bits(engram.staged_rows[:4]), bits[ids.numpy()])
    assert (engram.staged_rows[4:] == 5).all()  # rows past the batch untouched

    engram.replacement = EngramReplacement("zero", 1, shard, VOCAB, None)
    engram.prepare_embeddings(hashes)
    assert not engram.staged_rows[:4].any()
