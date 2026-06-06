# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import sentinel

import torch

import vllm.utils.deep_gemm as deep_gemm
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.model_executor.warmup import deep_gemm_warmup


def test_theoretical_alignment_uses_new_two_arg_binding(monkeypatch):
    calls = []

    def fake_impl(expected_m, num_groups):
        calls.append((expected_m, num_groups))
        return 64

    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        deep_gemm,
        "_get_theoretical_mk_alignment_for_contiguous_layout_impl",
        fake_impl,
    )

    assert (
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(
            expected_m=256,
            num_groups=128,
        )
        == 64
    )
    assert calls == [(256, 128)]


def test_theoretical_alignment_legacy_binding_receives_per_group_m(monkeypatch):
    calls = []

    def fake_impl(expected_m):
        calls.append(expected_m)
        return 16

    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        deep_gemm,
        "_get_theoretical_mk_alignment_for_contiguous_layout_impl",
        fake_impl,
    )

    assert (
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(
            expected_m=257,
            num_groups=128,
        )
        == 16
    )
    assert calls == [3]


def test_k_grouped_packed_ue8m0_wrapper_forwards_deepgemm_signature(monkeypatch):
    sf = torch.empty((8, 4), dtype=torch.float32)
    ks_tensor = torch.empty((2,), dtype=torch.int32)
    ks = [64, 192]
    calls = []

    def fake_impl(*args):
        calls.append(args)
        return sentinel.packed_scales

    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        deep_gemm,
        "_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl",
        fake_impl,
    )

    assert (
        deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
            sf, ks_tensor, ks, 32
        )
        is sentinel.packed_scales
    )
    assert calls == [(sf, ks_tensor, ks, 32)]


def test_weight_scale_post_process_accepts_uint8_e8m0(monkeypatch):
    calls = []

    def fake_transform(**kwargs):
        calls.append(kwargs)
        return sentinel.dg_scale

    monkeypatch.setattr(fp8_utils, "transform_sf_into_required_layout", fake_transform)

    scale = torch.full((2, 4), 127, dtype=torch.uint8)
    assert (
        fp8_utils.deepgemm_post_process_weight_scale_block(
            ws=scale,
            mn=2,
            k=128,
            quant_block_shape=(1, 32),
            num_groups=1,
        )
        is sentinel.dg_scale
    )

    assert calls[0]["sf"].dtype == torch.float32
    assert calls[0]["recipe"] == (1, 1, 32)


def test_grouped_deepgemm_warmup_cases_use_returned_alignment(monkeypatch):
    monkeypatch.setattr(
        deep_gemm_warmup, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(
        deep_gemm_warmup, "get_mk_alignment_for_contiguous_layout", lambda: [128, 128]
    )
    monkeypatch.setattr(
        deep_gemm_warmup,
        "_generate_optimal_warmup_m_values",
        lambda max_tokens, n, device: [1, 32],
    )

    def fake_compute_aligned_M_and_alignment(
        M, num_topk, local_num_experts, alignment, expert_tokens_meta
    ):
        assert num_topk == 8
        assert local_num_experts == 4
        assert alignment == 128
        assert expert_tokens_meta is None
        if M == 1:
            return 64, 16
        return 256, 64

    monkeypatch.setattr(
        deep_gemm_warmup,
        "compute_aligned_M_and_alignment",
        fake_compute_aligned_M_and_alignment,
    )

    w = torch.empty((4, 128, 128))

    max_m, block_m, warmup_cases = deep_gemm_warmup._get_grouped_gemm_params(
        w, w, num_topk=8, max_tokens=32
    )

    assert max_m == 256
    assert block_m == 128
    assert [
        (m, align, expert_ids.numel()) for m, align, expert_ids in warmup_cases
    ] == [
        (64, 16, 64),
        (256, 64, 256),
    ]
