# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek V4 IndexCache layer selection."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.models.deepseek_v4.attention as dsv4_attn_mod
import vllm.models.deepseek_v4.nvidia.model as dsv4_nvidia_model
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Attention,
    compute_dsv4_index_cache_skip_flags,
)

pytestmark = pytest.mark.skip_global_cleanup

COMPRESS_RATIOS = [128, 128, 4, 128, 4, 128, 4, 4, 128, 4]
NUM_HIDDEN_LAYERS = len(COMPRESS_RATIOS)


def _skipped_layer_ids(flags: tuple[bool, ...]) -> list[int]:
    return [idx for idx, skip in enumerate(flags) if skip]


def test_index_cache_default_freq_skips_no_layers():
    flags = compute_dsv4_index_cache_skip_flags(
        COMPRESS_RATIOS,
        NUM_HIDDEN_LAYERS,
    )
    assert _skipped_layer_ids(flags) == []


def test_index_cache_freq_one_skips_no_layers():
    flags = compute_dsv4_index_cache_skip_flags(
        COMPRESS_RATIOS,
        NUM_HIDDEN_LAYERS,
        index_topk_freq=1,
    )
    assert _skipped_layer_ids(flags) == []


def test_index_cache_freq_four_uses_default_offset_two():
    flags = compute_dsv4_index_cache_skip_flags(
        COMPRESS_RATIOS,
        NUM_HIDDEN_LAYERS,
        index_topk_freq=4,
    )
    assert _skipped_layer_ids(flags) == [6, 7, 9]


def test_index_cache_offset_one_matches_fsss_pattern():
    flags = compute_dsv4_index_cache_skip_flags(
        COMPRESS_RATIOS,
        NUM_HIDDEN_LAYERS,
        index_topk_freq=4,
        index_skip_topk_offset=1,
    )
    assert _skipped_layer_ids(flags) == [4, 6, 7]


def test_index_cache_custom_pattern_maps_to_c4_layers_only():
    flags = compute_dsv4_index_cache_skip_flags(
        COMPRESS_RATIOS,
        NUM_HIDDEN_LAYERS,
        index_topk_pattern="FSFSS",
    )
    assert _skipped_layer_ids(flags) == [4, 7, 9]


def test_non_c4_layers_are_never_skipped():
    flags = compute_dsv4_index_cache_skip_flags(
        COMPRESS_RATIOS,
        NUM_HIDDEN_LAYERS,
        index_topk_pattern="FSSSS",
    )
    assert all(
        not flags[idx] for idx, ratio in enumerate(COMPRESS_RATIOS) if ratio != 4
    )


def test_extra_compress_ratios_do_not_affect_backbone_mapping():
    flags = compute_dsv4_index_cache_skip_flags(
        COMPRESS_RATIOS + [4, 4, 4],
        NUM_HIDDEN_LAYERS,
        index_topk_pattern="FSFSS",
    )
    assert len(flags) == NUM_HIDDEN_LAYERS
    assert _skipped_layer_ids(flags) == [4, 7, 9]


def test_each_pp_stage_first_local_c4_layer_is_forced_full():
    flags = compute_dsv4_index_cache_skip_flags(
        COMPRESS_RATIOS,
        NUM_HIDDEN_LAYERS,
        index_topk_pattern="FSSSS",
        local_start_layer=3,
        local_end_layer=8,
    )
    assert _skipped_layer_ids(flags) == [6, 7, 9]
    assert not flags[4]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"index_topk_freq": 0}, "index_topk_freq must be positive"),
        ({"index_skip_topk_offset": 0}, "index_skip_topk_offset must be at least 1"),
        ({"index_topk_pattern": "FXFSS"}, "only supports 'F'.*'S'"),
        ({"index_topk_pattern": "FSS"}, "length must match"),
        ({"index_topk_pattern": "SFFFF"}, "must start with 'F'"),
    ],
)
def test_invalid_index_cache_config_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        compute_dsv4_index_cache_skip_flags(
            COMPRESS_RATIOS,
            NUM_HIDDEN_LAYERS,
            **kwargs,
        )


def test_short_compress_ratios_raises():
    with pytest.raises(ValueError, match="at least num_hidden_layers"):
        compute_dsv4_index_cache_skip_flags(
            [4, 128],
            3,
        )


def test_invalid_local_range_raises():
    with pytest.raises(ValueError, match="local layer range"):
        compute_dsv4_index_cache_skip_flags(
            COMPRESS_RATIOS,
            NUM_HIDDEN_LAYERS,
            local_start_layer=8,
            local_end_layer=7,
        )


class _TestDeepseekV4Attention(DeepseekV4Attention):
    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        pass

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return o


def _make_attention(skip_topk: bool) -> _TestDeepseekV4Attention:
    attn = object.__new__(_TestDeepseekV4Attention)
    attn.indexer = Mock(name="indexer")
    attn.skip_topk = skip_topk
    attn.compressor = Mock(name="compressor")
    attn.aux_stream_list = None
    attn.ln_events = [None, None, None, None]
    attn.n_local_heads = 1
    attn.head_dim = 2
    attn.wq_b = Mock(return_value=torch.zeros(1, 2))
    attn._fused_qnorm_rope_kv_insert = Mock(side_effect=lambda q, *_args: q)
    attn.forward_mqa = Mock()
    attn.indexer_rotary_emb = object()
    attn.rotary_emb = object()
    return attn


@pytest.fixture()
def _patch_attention_helpers(monkeypatch):
    monkeypatch.setattr(
        dsv4_attn_mod,
        "get_forward_context",
        lambda: type("Context", (), {"attn_metadata": {}})(),
    )
    monkeypatch.setattr(
        dsv4_attn_mod,
        "execute_in_parallel",
        lambda main_fn, fns, *_args, **_kwargs: (
            main_fn(),
            [fn() if fn is not None else None for fn in fns],
        ),
    )
    monkeypatch.setattr(
        dsv4_attn_mod,
        "maybe_execute_in_parallel",
        lambda first_fn, second_fn, *_args, **_kwargs: (first_fn(), second_fn()),
    )


def test_skip_topk_bypasses_indexer_but_keeps_main_compressor(_patch_attention_helpers):
    attn = _make_attention(skip_topk=True)

    attn.attention_impl(
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        None,
        None,
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 1, 2),
    )

    attn.indexer.assert_not_called()
    attn.compressor.assert_called_once()
    attn.forward_mqa.assert_called_once()


def test_full_topk_path_calls_indexer_and_compressor(_patch_attention_helpers):
    attn = _make_attention(skip_topk=False)

    attn.attention_impl(
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 1, 2),
    )

    attn.indexer.assert_called_once()
    attn.compressor.assert_called_once()
    attn.forward_mqa.assert_called_once()


def test_index_cache_platform_guard_ignores_no_skipped_layers(monkeypatch):
    monkeypatch.setattr(
        dsv4_nvidia_model.current_platform,
        "get_device_capability",
        lambda: None,
    )

    dsv4_nvidia_model._validate_dsv4_index_cache_platform(False)


def test_index_cache_platform_guard_allows_hopper(monkeypatch):
    monkeypatch.setattr(
        dsv4_nvidia_model.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=9),
    )

    dsv4_nvidia_model._validate_dsv4_index_cache_platform(True)


def test_index_cache_platform_guard_rejects_non_hopper(monkeypatch):
    monkeypatch.setattr(
        dsv4_nvidia_model.current_platform,
        "get_device_capability",
        lambda: SimpleNamespace(major=12),
    )

    with pytest.raises(NotImplementedError, match="only on Hopper"):
        dsv4_nvidia_model._validate_dsv4_index_cache_platform(True)


def test_index_cache_ubatching_guard_ignores_no_skipped_layers():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=True),
    )

    dsv4_nvidia_model._validate_dsv4_index_cache_ubatching(False, vllm_config)


def test_index_cache_ubatching_guard_rejects_enabled_ubatching():
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_ubatching=True),
    )

    with pytest.raises(NotImplementedError, match="does not support ubatching"):
        dsv4_nvidia_model._validate_dsv4_index_cache_ubatching(True, vllm_config)


def test_skipped_indexer_weight_filter_matches_only_omitted_indexers():
    skipped_layer_ids = frozenset({6, 7})

    assert dsv4_nvidia_model._is_dsv4_skipped_indexer_weight(
        "model.layers.6.attn.indexer.wk_weights_proj.weight",
        skipped_layer_ids,
    )
    assert not dsv4_nvidia_model._is_dsv4_skipped_indexer_weight(
        "model.layers.6.attn.compressor.fused_wkv_wgate.weight",
        skipped_layer_ids,
    )
    assert not dsv4_nvidia_model._is_dsv4_skipped_indexer_weight(
        "model.layers.16.attn.indexer.wk_weights_proj.weight",
        skipped_layer_ids,
    )
