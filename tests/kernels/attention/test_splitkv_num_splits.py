# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.rocm_attn import _splitkv_workspace_support
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    _MAX_SPLITS,
    _choose_compute_block_size,
    _choose_fallback_block_size,
    _get_num_splits,
    _paged_attention_2d_splitkv_decode,
    _splitkv_workspace_shapes,
)
from vllm.v1.attention.ops.rdna4_splitkv import _can_use_rdna4_splitkv_decode
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

GFX1201_WGPS = 32


def _can_route(**overrides) -> bool:
    args = {
        "query_dtype": torch.bfloat16,
        "key_cache_dtype": torch.bfloat16,
        "value_cache_dtype": torch.bfloat16,
        "kv_quant_mode": KVQuantMode.NONE,
        "is_e4m3_kv_cache": False,
        "head_size": 256,
        "num_query_heads": 12,
        "num_kv_heads": 2,
        "use_alibi_slopes": False,
        "sliding_window": 0,
        "has_sinks": False,
        "has_output_scale": False,
        "is_gfx1x": True,
        "is_gfx12x": True,
    }
    args.update(overrides)
    return _can_use_rdna4_splitkv_decode(**args)


@pytest.mark.parametrize(
    "physical_block_size,expected",
    [(16, 16), (32, 32), (528, 32), (784, 32), (1056, 32), (1568, 32)],
)
def test_choose_compute_block_size(physical_block_size: int, expected: int) -> None:
    assert _choose_compute_block_size(physical_block_size) == expected


@pytest.mark.parametrize(
    "physical_block_size,expected",
    [(16, 16), (32, 32), (64, 64), (128, 128), (528, 32), (1568, 32)],
)
def test_choose_fallback_block_size(physical_block_size: int, expected: int) -> None:
    assert _choose_fallback_block_size(physical_block_size) == expected


def test_fp8_short_context_uses_full_wave_split_policy() -> None:
    args = {
        "batch_size": 1,
        "num_kv_heads": 2,
        "head_size": 256,
        "physical_block_size": 1568,
        "max_seq_len": 1374,
        "num_sms": GFX1201_WGPS,
    }
    assert _get_num_splits(**args, allow_short_context=False) == 1
    assert _get_num_splits(**args, allow_short_context=True) == 16


@pytest.mark.parametrize("allow_short_context", [False, True])
@pytest.mark.parametrize("num_sms", [1, 16, 32, 42, 64, 304])
@pytest.mark.parametrize("max_num_splits", [1, 8, _MAX_SPLITS])
def test_num_splits_stays_within_bounds(
    allow_short_context: bool, num_sms: int, max_num_splits: int
) -> None:
    for physical_block_size in (16, 32, 528, 784, 1056, 1568):
        for batch_size in (1, 4, 16):
            for max_seq_len in (1, 512, 4096, 32768, 131072):
                splits = _get_num_splits(
                    batch_size,
                    num_kv_heads=2,
                    head_size=256,
                    physical_block_size=physical_block_size,
                    max_seq_len=max_seq_len,
                    max_num_splits=max_num_splits,
                    num_sms=num_sms,
                    allow_short_context=allow_short_context,
                )
                assert 1 <= splits <= max_num_splits
                assert splits & (splits - 1) == 0


@pytest.mark.parametrize("allow_short_context", [False, True])
@pytest.mark.parametrize("physical_block_size", [16, 32, 528, 1568])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 8])
def test_workspace_envelope_covers_every_splitting_batch(
    allow_short_context: bool,
    physical_block_size: int,
    num_kv_heads: int,
) -> None:
    max_batch_size = 64
    max_seq_len = 8192
    shapes = _splitkv_workspace_shapes(
        max_batch_size,
        num_query_heads=16,
        num_kv_heads=num_kv_heads,
        head_size=256,
        physical_block_size=physical_block_size,
        max_seq_len=max_seq_len,
        allow_short_context=allow_short_context,
        num_sms=GFX1201_WGPS,
    )
    assert shapes is not None
    mid_out_shape, mid_lse_shape = shapes
    assert mid_out_shape[:3] == mid_lse_shape

    for batch_size in range(1, max_batch_size + 1):
        splits = _get_num_splits(
            batch_size,
            num_kv_heads,
            head_size=256,
            physical_block_size=physical_block_size,
            max_seq_len=max_seq_len,
            num_sms=GFX1201_WGPS,
            allow_short_context=allow_short_context,
        )
        if splits > 1:
            assert batch_size <= mid_out_shape[0]
            assert splits <= mid_out_shape[2]


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "query_dtype": torch.float16,
            "key_cache_dtype": torch.float16,
            "value_cache_dtype": torch.float16,
            "is_gfx12x": False,
        },
        {
            "key_cache_dtype": torch.float8_e4m3fn,
            "value_cache_dtype": torch.float8_e4m3fn,
            "kv_quant_mode": KVQuantMode.FP8_PER_TENSOR,
            "is_e4m3_kv_cache": True,
        },
    ],
)
def test_splitkv_route_accepts_validated_configs(overrides) -> None:
    assert _can_route(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "key_cache_dtype": torch.float8_e4m3fn,
            "value_cache_dtype": torch.float8_e4m3fn,
            "kv_quant_mode": KVQuantMode.FP8_PER_TENSOR,
            "is_e4m3_kv_cache": True,
            "is_gfx12x": False,
        },
        {
            "key_cache_dtype": torch.float8_e5m2,
            "value_cache_dtype": torch.float8_e5m2,
            "kv_quant_mode": KVQuantMode.FP8_PER_TENSOR,
        },
        {
            "kv_quant_mode": KVQuantMode.FP8_PER_TENSOR,
            "is_e4m3_kv_cache": True,
        },
        {"kv_quant_mode": KVQuantMode.FP8_PER_TOKEN_HEAD},
        {"query_dtype": torch.float32},
        {"head_size": 64},
        {"num_query_heads": 13},
        {"num_query_heads": 17, "num_kv_heads": 1},
        {"use_alibi_slopes": True},
        {"sliding_window": 128},
        {"has_sinks": True},
        {"has_output_scale": True},
        {"key_cache_dtype": torch.float16, "value_cache_dtype": torch.float16},
        {"is_gfx1x": False, "is_gfx12x": False},
    ],
)
def test_splitkv_route_rejects_unsupported_configs(overrides) -> None:
    assert not _can_route(**overrides)


@pytest.mark.parametrize(
    "bad_scale",
    [
        1.0,
        torch.ones(2, dtype=torch.float32),
        torch.ones((), dtype=torch.bfloat16),
        torch.ones((), dtype=torch.float32, device="meta"),
    ],
)
def test_fp8_launcher_requires_valid_scale_tensors(bad_scale) -> None:
    query = torch.zeros(1, 12, 256, dtype=torch.bfloat16)
    key_cache = torch.zeros(1, 2, 16, 32, 16, dtype=torch.float8_e4m3fn)
    value_cache = torch.zeros(1, 2, 256, 32, dtype=torch.float8_e4m3fn)
    block_tables = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)

    good_scale = torch.ones((), dtype=torch.float32)
    with pytest.raises(TypeError, match="scalar float32 tensor"):
        _paged_attention_2d_splitkv_decode(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale=256**-0.5,
            k_scale=bad_scale,
            v_scale=good_scale,
            actual_max_splits=2,
        )


@pytest.mark.parametrize("kv_dtype", [torch.uint8, torch.float8_e5m2])
def test_splitkv_launcher_rejects_unsupported_byte_cache(kv_dtype) -> None:
    query = torch.zeros(1, 12, 256, dtype=torch.bfloat16)
    key_cache = torch.zeros(1, 2, 16, 32, 16, dtype=kv_dtype)
    value_cache = torch.zeros(1, 2, 256, 32, dtype=kv_dtype)
    block_tables = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)

    with pytest.raises(TypeError, match="E4M3 FP8"):
        _paged_attention_2d_splitkv_decode(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            scale=256**-0.5,
            k_scale=torch.ones((), dtype=torch.float32),
            v_scale=torch.ones((), dtype=torch.float32),
            actual_max_splits=2,
        )


@pytest.mark.parametrize(
    "dtype,quant_mode,is_e4m3,is_gfx12x,expected",
    [
        (torch.bfloat16, KVQuantMode.NONE, False, False, (True, False)),
        (torch.uint8, KVQuantMode.FP8_PER_TENSOR, True, True, (True, True)),
        (torch.uint8, KVQuantMode.FP8_PER_TENSOR, True, False, (False, False)),
        (torch.uint8, KVQuantMode.FP8_PER_TENSOR, False, True, (False, False)),
        (torch.uint8, KVQuantMode.NONE, False, True, (False, False)),
    ],
)
def test_splitkv_workspace_supports_byte_storage_fp8_specs(
    dtype: torch.dtype,
    quant_mode: KVQuantMode,
    is_e4m3: bool,
    is_gfx12x: bool,
    expected: tuple[bool, bool],
) -> None:
    spec = AttentionSpec(
        block_size=1568,
        num_kv_heads=2,
        head_size=256,
        dtype=dtype,
        kv_quant_mode=quant_mode,
    )
    assert (
        _splitkv_workspace_support(
            spec,
            torch.bfloat16,
            is_e4m3_kv_cache=is_e4m3,
            is_gfx1x=True,
            is_gfx12x=is_gfx12x,
        )
        == expected
    )
