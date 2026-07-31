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
    DeepseekV4Indexer,
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


def test_skip_topk_attention_still_constructs_indexer(monkeypatch):
    class FakeLayer:
        def __init__(self, *_args, **kwargs):
            self.skip_topk = kwargs.get("skip_topk")

    monkeypatch.setattr(
        dsv4_attn_mod,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    for layer_cls in (
        "MergedColumnParallelLinear",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "RMSNorm",
        "DeepseekV4Indexer",
        "DeepseekV4SWACache",
        "DeepseekCompressor",
    ):
        monkeypatch.setattr(dsv4_attn_mod, layer_cls, FakeLayer)
    monkeypatch.setattr(dsv4_attn_mod, "build_deepseek_v4_rope", FakeLayer)
    monkeypatch.setattr(torch.cuda, "Event", Mock)

    config = SimpleNamespace(
        hidden_size=256,
        num_attention_heads=4,
        q_lora_rank=64,
        o_lora_rank=64,
        head_dim=128,
        qk_rope_head_dim=64,
        o_groups=1,
        sliding_window=128,
        num_hidden_layers=2,
        compress_ratios=[128, 4],
        rms_norm_eps=1e-6,
        max_position_embeddings=1024,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, max_model_len=1024),
        quant_config=None,
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    attn = _TestDeepseekV4Attention(
        vllm_config,
        prefix="model.layers.1.attn",
        topk_indices_buffer=torch.empty(8, dtype=torch.int32),
        skip_topk=True,
    )

    assert attn.indexer is not None
    assert attn.indexer.skip_topk


def _make_attention(skip_topk: bool) -> tuple[_TestDeepseekV4Attention, Mock]:
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
    forward_mqa = Mock()
    object.__setattr__(attn, "forward_mqa", forward_mqa)
    attn.indexer_rotary_emb = object()
    attn.rotary_emb = object()
    return attn, forward_mqa


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
    attn, forward_mqa = _make_attention(skip_topk=True)

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
    forward_mqa.assert_called_once()


def test_full_topk_path_calls_indexer_and_compressor(_patch_attention_helpers):
    attn, forward_mqa = _make_attention(skip_topk=False)

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
    forward_mqa.assert_called_once()


def test_skip_topk_indexer_keeps_cache_with_meta_weights(monkeypatch):
    init_devices = []
    quant_configs = []

    class FakeLinear:
        def __init__(self, *_args, quant_config=None, **_kwargs):
            init_devices.append(torch.empty(0).device.type)
            quant_configs.append(quant_config)

    class FakeIndexerCache:
        def __init__(self, *_args, prefix, **_kwargs):
            self.prefix = prefix

    class FakeCompressor:
        def __init__(self, *_args, **_kwargs):
            init_devices.append(torch.empty(0).device.type)

    monkeypatch.setattr(dsv4_attn_mod, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(dsv4_attn_mod, "DeepseekV4IndexerCache", FakeIndexerCache)
    monkeypatch.setattr(dsv4_attn_mod, "DeepseekCompressor", FakeCompressor)
    monkeypatch.setattr(
        dsv4_attn_mod,
        "get_max_prefill_buffer_size",
        lambda _vllm_config: 1024,
    )
    monkeypatch.setattr(torch.cuda, "Event", Mock)

    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(use_fp4_indexer_cache=False),
        model_config=SimpleNamespace(max_model_len=1024),
    )
    config = SimpleNamespace(
        index_topk=8,
        index_n_heads=4,
        index_head_dim=128,
        qk_rope_head_dim=64,
    )

    indexer = DeepseekV4Indexer(
        vllm_config,
        config=config,
        hidden_size=256,
        q_lora_rank=64,
        quant_config=object(),
        cache_config=object(),
        topk_indices_buffer=torch.empty(8, dtype=torch.int32),
        compress_ratio=4,
        prefix="model.layers.1.attn.indexer",
        skip_topk=True,
    )

    assert indexer.k_cache.prefix.endswith(".k_cache")
    assert indexer.indexer_op is None
    assert indexer.aux_stream is None
    assert init_devices == ["meta", "meta", "meta"]
    assert quant_configs == [None, None]


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
