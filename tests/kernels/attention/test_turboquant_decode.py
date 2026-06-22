# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def test_turboquant_kernels_do_not_specialize_runtime_strides() -> None:
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _tq_decode_stage1,
        _tq_full_dequant_kv,
    )

    assert set(_tq_decode_stage1.do_not_specialize) == {
        "stride_qb",
        "stride_qh",
        "stride_bt_b",
        "stride_cache_block",
        "stride_cache_pos",
        "stride_cache_head",
        "stride_mid_b",
        "stride_mid_h",
        "stride_mid_s",
    }
    assert set(_tq_full_dequant_kv.do_not_specialize) == {
        "stride_bt_b",
        "stride_cache_block",
        "stride_cache_pos",
        "stride_cache_head",
        "stride_ko_b",
        "stride_ko_h",
        "stride_ko_s",
        "stride_vo_b",
        "stride_vo_h",
        "stride_vo_s",
    }
