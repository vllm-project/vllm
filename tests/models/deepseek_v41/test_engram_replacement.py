# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Engram `per_token` lookup mode (DeepSeek V4.1)."""

from types import SimpleNamespace
from unittest.mock import patch

import pydantic
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.engram import EngramConfig
from vllm.config.utils import hash_config
from vllm.model_executor.model_loader.weight_utils import (
    finalize_model_weights,
    initialize_single_dummy_weight,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.utils import weights_already_processed
from vllm.models.deepseek_v41.common.engram_replacement import (
    PerTokenTable,
    ReplacementShard,
    per_token_comp_ids,
    skip_engram_weight,
)
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

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


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        EngramConfig(mode="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_across_dp": True},
        {"dp_shared_memory": True},
        {"use_thp": True},
        {"peer_group_size": 2, "peer_group_store": "/tmp/peers", "cpu_offload": False},
    ],
)
def test_per_token_rejects_host_and_dp_sharing(kwargs) -> None:
    with pytest.raises(ValueError, match="must be off"):
        EngramConfig(mode="per_token", **kwargs)


def test_ngram_mode_still_accepts_dp_sharing() -> None:
    EngramConfig(embedding_across_dp=True)
    EngramConfig(dp_shared_memory=True)


def test_mode_enters_the_config_hash() -> None:
    assert hash_config(EngramConfig()) != hash_config(EngramConfig(mode="per_token"))


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


def test_loader_overwrites_stale_padding() -> None:
    """The last TP shard overruns n_hash_cols; its padding must be zeroed even
    if the parameter held other values."""
    table = _per_token_table(tp=5, rank=4)  # part = 5, head_start = 20, used = 4
    table.weight.data.fill_(7.0)
    table.weight.weight_loader(table.weight, _table())
    assert not table.weight.data[:, 4:].any()
    assert torch.equal(table.weight.data[:, :4], _table()[:, 20:24])


def test_table_loads_through_auto_weights_loader_like_any_weight(tmp_path) -> None:
    """Checkpoint style: a bf16 safetensors tensor named `per_token_table.weight`
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


def test_comp_ids_follow_the_token_map_and_dropped_positions_map_to_id_zero() -> None:
    token_map = torch.arange(100, 0, -1, dtype=torch.int32)  # token t -> 100 - t
    ids = torch.tensor([3, 0, 7, 5])
    keep = torch.tensor([True, True, False, True])  # position 2: image span
    out = per_token_comp_ids(token_map, ids, keep)
    assert out.dtype == torch.int64
    assert out.tolist() == [97, 100, 100, 95]


def test_comp_ids_on_a_profile_run_use_id_zero_everywhere() -> None:
    """Profile runs keep nothing, yet per_token still needs valid ids."""
    token_map = torch.arange(100, 0, -1, dtype=torch.int32)
    ids = torch.tensor([2**30, 5, -1], dtype=torch.int32)  # arbitrary dummy ids
    keep = torch.zeros(3, dtype=torch.bool)
    assert per_token_comp_ids(token_map, ids, keep).tolist() == [100, 100, 100]


def test_comp_ids_clamp_ids_beyond_the_token_map() -> None:
    token_map = torch.arange(10)
    ids = torch.tensor([9, 10, 1000])
    keep = torch.ones(3, dtype=torch.bool)
    assert per_token_comp_ids(token_map, ids, keep).tolist() == [9, 9, 9]


# --- per-name load decision ---------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "skipped_in_per_token", "skipped_in_ngram"),
    [
        ("model.layers.1.engram.embed_tokens.weight", True, False),
        ("model.layers.1.engram.embed_tokens.scale", True, False),
        ("model.layers.1.engram.embed_tokens.weight_scale", True, False),
        ("model.layers.1.engram.per_token_table.weight", False, True),
        ("model.layers.1.engram.wkv.weight", False, False),
        ("model.layers.1.self_attn.qkv_proj.weight", False, False),
    ],
)
def test_each_mode_skips_only_the_table_it_does_not_use(
    name: str, skipped_in_per_token: bool, skipped_in_ngram: bool
) -> None:
    """A checkpoint carrying both tables can be served in either mode."""
    assert skip_engram_weight(name, per_token=True) is skipped_in_per_token
    assert skip_engram_weight(name, per_token=False) is skipped_in_ngram


# --- never-loaded check (runs from process_weights_after_loading) ------------------


def _fill_with_weight_loader(table: PerTokenTable) -> None:
    table.weight.weight_loader(table.weight, _table())


def _fill_with_copy(table: PerTokenTable) -> None:
    """`--load-format sharded_state` copies the local shard straight in."""
    local = torch.zeros_like(table.weight)
    local[:, :4] = _table()[:, 20:24]
    table.weight.data.copy_(local)


@pytest.mark.parametrize("fill", [_fill_with_weight_loader, _fill_with_copy])
def test_hook_rejects_an_unloaded_table_until_the_shard_is_loaded(fill) -> None:
    table = _per_token_table(tp=5, rank=4)  # part = 5, head_start = 20, used = 4
    with pytest.raises(ValueError, match="never loaded"):
        table.process_weights_after_loading()
    fill(table)
    table.process_weights_after_loading()
    assert torch.equal(table.weight.data[:, :4], _table()[:, 20:24])
    assert not table.weight.data[:, 4:].any()  # padding of the last shard


def test_hook_rejects_an_unloaded_table_until_dummy_init() -> None:
    table = _per_token_table()
    with pytest.raises(ValueError, match="never loaded"):
        table.process_weights_after_loading()
    initialize_single_dummy_weight(table.weight)
    table.process_weights_after_loading()
    assert torch.isfinite(table.weight).all()


def test_hook_is_skipped_for_already_processed_weights() -> None:
    table = _per_token_table()
    with pytest.raises(ValueError, match="never loaded"):
        table.process_weights_after_loading()
    with weights_already_processed():
        table.process_weights_after_loading()


def test_hook_is_skipped_on_the_meta_device() -> None:
    table = _per_token_table()
    with pytest.raises(ValueError, match="never loaded"):
        table.process_weights_after_loading()
    table.to_empty(device="meta")
    table.process_weights_after_loading()


def test_nan_in_a_loaded_table_is_rejected() -> None:
    table = _per_token_table()
    bad = _table()
    bad[3, 2, 1] = float("nan")
    table.weight.weight_loader(table.weight, bad)
    with pytest.raises(ValueError, match="contains NaN"):
        table.process_weights_after_loading()


def test_finalize_model_weights_runs_the_hook() -> None:
    class _Engram(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.per_token_table = _per_token_table()

    engram = _Engram()
    with pytest.raises(ValueError, match="never loaded"):
        finalize_model_weights(engram)
    engram.per_token_table.weight.weight_loader(engram.per_token_table.weight, _table())
    finalize_model_weights(engram)


# --- Engram module in per_token mode, built through its real __init__ ---------------


def test_engram_per_token_mode_builds_and_looks_up_rows(dist_init) -> None:
    from vllm.models.deepseek_v41.common.engram import Engram, EngramLayout

    config = DeepseekV41Config(
        text_config=dict(
            hidden_size=64,
            hc_mult=2,
            rms_norm_eps=1e-6,
            engram_layer_ids=[1],
            engram_num_embeddings=[1000],
            engram_vocab_size=1000,
            engram_max_ngram_size=4,
            engram_n_heads=N_COLS // 3,  # n_hash_cols = 3 * 8 = 24
            engram_head_dim=HEAD_DIM,
            engram_compressed_vocab_size=VOCAB,
            engram_pad_token_id=0,
        )
    )
    layout = EngramLayout(config)
    vllm_config = VllmConfig()
    # VllmConfig rejects an engram config without a model.
    vllm_config.engram_config = EngramConfig(mode="per_token")
    with set_current_vllm_config(vllm_config):
        engram = Engram(
            config,
            None,
            layout,
            layer_hash_index=0,
            use_sequence_parallel=False,
            prefix="model.layers.1.engram",
        )

    assert isinstance(engram.embed_tokens, ReplacementShard)
    assert not list(engram.embed_tokens.parameters())
    assert engram.per_token_table is not None
    assert tuple(engram.per_token_table.weight.shape) == (VOCAB, N_COLS, HEAD_DIM)
    keys = engram.state_dict().keys()
    assert not [k for k in keys if k.startswith("embed_tokens.")]
    assert "per_token_table.weight" in keys

    full = _table()
    weight = engram.per_token_table.weight
    weight.weight_loader(weight, full)
    hashes = torch.zeros(4, N_COLS, dtype=torch.int32)
    comp_ids = torch.tensor([5, 5, 2, 0])
    engram.prepare_embeddings(hashes, comp_ids)
    assert torch.equal(engram.embed(hashes), full[comp_ids])
