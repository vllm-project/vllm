# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Engram `per_token` lookup mode (DeepSeek V4.1)."""

from types import SimpleNamespace
from unittest.mock import patch

import pydantic
import pytest
import torch

from vllm.config.engram import EngramConfig
from vllm.config.utils import hash_config
from vllm.model_executor.utils import weights_already_processed
from vllm.models.deepseek_v41.common.engram_replacement import (
    PerTokenTable,
    per_token_comp_ids,
    skip_engram_weight,
)

N_COLS, HEAD_DIM, VOCAB = 24, 256, 50


def _table() -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    return torch.randn(VOCAB, N_COLS, HEAD_DIM, generator=gen).bfloat16()


def _per_token_table(tp: int = 1, rank: int = 0) -> PerTokenTable:
    return PerTokenTable(VOCAB, N_COLS, HEAD_DIM, tp, rank)


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        architecture="DeepseekV41ForCausalLM",
        hf_text_config=SimpleNamespace(engram_layer_ids=[1, 14]),
    )


ROCM = SimpleNamespace(is_rocm=lambda: True)
NOT_ROCM = SimpleNamespace(is_rocm=lambda: False)


def test_default_mode_is_ngram() -> None:
    assert EngramConfig().mode == "ngram"


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


def test_mode_enters_the_config_hash() -> None:
    assert hash_config(EngramConfig()) != hash_config(EngramConfig(mode="per_token"))


def test_per_token_does_not_share_host_tables_across_dp() -> None:
    parallel = SimpleNamespace(data_parallel_size=2, enable_elastic_ep=False)
    cfg = EngramConfig(mode="per_token")
    cfg.resolve_dp_shared_memory(parallel)
    assert cfg.dp_shared_memory is False


def test_per_token_is_rocm_only() -> None:
    cfg = EngramConfig(mode="per_token")
    with patch("vllm.platforms.current_platform", ROCM):
        cfg.verify_model_config(_model_config())
    with (
        patch("vllm.platforms.current_platform", NOT_ROCM),
        pytest.raises(NotImplementedError, match="only implemented for ROCm"),
    ):
        cfg.verify_model_config(_model_config())


def test_per_token_is_rejected_for_qwen4_exp() -> None:
    qwen = SimpleNamespace(
        architecture="Qwen4ExpForCausalLM",
        hf_text_config=SimpleNamespace(ple_layer_ids=[3]),
    )
    with patch("vllm.platforms.current_platform", ROCM):
        with pytest.raises(ValueError, match="only supported for DeepSeek V4.1"):
            EngramConfig(mode="per_token").verify_model_config(qwen)
        EngramConfig().verify_model_config(qwen)


@pytest.mark.parametrize("tp", [2, 5])
def test_shards_reassemble_the_full_table(tp: int) -> None:
    full = _table()
    part = -(-N_COLS // tp)
    local = []
    for rank in range(tp):
        table = _per_token_table(tp, rank)
        assert table.weight.dtype == torch.bfloat16
        assert tuple(table.weight.shape) == (VOCAB, part, HEAD_DIM)
        table.weight.weight_loader(table.weight, full)
        local.append(table.weight.detach())
    gathered = torch.cat(local, dim=1)
    assert torch.equal(gathered[:, :N_COLS], full)
    assert not gathered[:, N_COLS:].any()


def test_float_tables_are_rejected() -> None:
    table = _per_token_table()
    with pytest.raises(ValueError, match="must be bfloat16"):
        table.weight.weight_loader(table.weight, _table().float())


@pytest.mark.parametrize(
    "shape",
    [
        (VOCAB + 1, N_COLS, HEAD_DIM),
        (VOCAB, N_COLS - 1, HEAD_DIM),
        (VOCAB, N_COLS, HEAD_DIM // 2),
        (VOCAB, N_COLS * HEAD_DIM),
    ],
)
def test_wrong_table_shape_is_rejected(shape) -> None:
    table = _per_token_table()
    bad = torch.zeros(shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="expected .*compressed_vocab"):
        table.weight.weight_loader(table.weight, bad)


def test_comp_ids_follow_the_token_map_and_dropped_positions_map_to_id_zero() -> None:
    token_map = torch.arange(100, 0, -1, dtype=torch.int32)
    ids = torch.tensor([3, 0, 7, 5])
    keep = torch.tensor([True, True, False, True])
    out = per_token_comp_ids(token_map, ids, keep)
    assert out.dtype == torch.int64
    assert out.tolist() == [97, 100, 100, 95]


def test_comp_ids_clamp_ids_beyond_the_token_map() -> None:
    token_map = torch.arange(10)
    ids = torch.tensor([9, 10, 1000])
    keep = torch.ones(3, dtype=torch.bool)
    assert per_token_comp_ids(token_map, ids, keep).tolist() == [9, 9, 9]


@pytest.mark.parametrize(
    ("name", "skipped_in_per_token", "skipped_in_ngram"),
    [
        ("model.layers.1.engram.embed_tokens.weight", True, False),
        ("model.layers.1.engram.embed_tokens.weight_scale_inv", True, False),
        ("model.layers.1.engram.per_token_table.weight", False, True),
        ("model.layers.1.engram.wkv.weight", False, False),
        ("model.layers.1.self_attn.qkv_proj.weight", False, False),
    ],
)
def test_each_mode_skips_only_the_table_it_does_not_use(
    name: str, skipped_in_per_token: bool, skipped_in_ngram: bool
) -> None:
    assert skip_engram_weight(name, per_token=True) is skipped_in_per_token
    assert skip_engram_weight(name, per_token=False) is skipped_in_ngram


def test_sharded_state_copy_passes_post_load_check() -> None:
    table = _per_token_table()
    with pytest.raises(ValueError, match="per_token_table.weight"):
        table.process_weights_after_loading()
    table.state_dict()["weight"].copy_(_table())
    table.process_weights_after_loading()


def test_hook_is_skipped_for_already_processed_weights() -> None:
    table = _per_token_table()
    with weights_already_processed():
        table.process_weights_after_loading()


def test_hook_is_skipped_on_the_meta_device() -> None:
    with torch.device("meta"):
        table = _per_token_table()
    table.process_weights_after_loading()


def test_nan_in_a_loaded_table_is_rejected() -> None:
    table = _per_token_table()
    bad = _table()
    bad[3, 2, 1] = float("nan")
    table.weight.weight_loader(table.weight, bad)
    with pytest.raises(ValueError, match="contains NaN"):
        table.process_weights_after_loading()
