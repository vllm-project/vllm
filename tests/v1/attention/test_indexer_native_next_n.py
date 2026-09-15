# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which next_n the DSA indexer decode path may hand to DeepGEMM unflattened.

Getting this wrong is not a slow path but a crash: `fp8_fp4_paged_mqa_logits`
asserts both that the architecture implements the requested `next_n` and that
the schedule metadata was sized for the matching slot count.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _paged_mqa_logits_schedule_slots
from vllm.v1.attention.backends.mla import indexer
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.block_table import get_block_table_width

NUM_SMS = 114  # H100 PCIe


def _set_arch(monkeypatch, family: int, *, cuda: bool = True, deep_gemm: bool = True):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability // 10 == family,
    )
    monkeypatch.setattr(indexer, "has_deep_gemm", lambda: deep_gemm)


@pytest.mark.parametrize(
    "family,expected_native",
    [
        # SM90 gained next_n=4 (MTP=3) via 2-CTA multicast, but never 3.
        (9, {1, 2, 4}),
        # SM100 schedules any next_n with multi-atom tiles.
        (10, {1, 2, 3, 4, 5, 8}),
        # SM120 advertises multi-atom too but is unvalidated on hardware, so
        # it stays on the conservative gate. Loosen it only with measurements.
        (12, {1, 2}),
    ],
)
def test_native_decode_gate_per_architecture(monkeypatch, family, expected_native):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4, 5, 8):
        assert indexer._supports_native_decode(next_n) == (next_n in expected_native), (
            f"family={family} next_n={next_n}"
        )


@pytest.mark.parametrize(
    "cuda,deep_gemm", [(False, True), (True, False), (False, False)]
)
def test_native_decode_gate_without_deepgemm(monkeypatch, cuda, deep_gemm):
    """Without the DeepGEMM kernels only the shapes every backend handles."""
    _set_arch(monkeypatch, 9, cuda=cuda, deep_gemm=deep_gemm)
    assert [indexer._supports_native_decode(n) for n in (1, 2, 3, 4)] == [
        True,
        True,
        False,
        False,
    ]


def test_sm90_next_n_4_halves_the_schedule_slots(monkeypatch):
    """SM90 next_n=4 runs one scheduler task per 2-CTA cluster, not per SM."""
    _set_arch(monkeypatch, 9)
    assert _paged_mqa_logits_schedule_slots(NUM_SMS, 4) == NUM_SMS // 2
    for next_n in (1, 2, 3):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


@pytest.mark.parametrize("family", [10, 12])
def test_multicast_is_sm90_only(monkeypatch, family):
    _set_arch(monkeypatch, family)
    for next_n in (1, 2, 3, 4):
        assert _paged_mqa_logits_schedule_slots(NUM_SMS, next_n) == NUM_SMS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_flattening", [False, True])
def test_padded_spec_decode_request_gets_zero_context_length(use_flattening):
    """A FULL-CUDA-graph replay pads the batch with requests whose seq_len is
    0. With next_n > 1 the per-token context length of such a request's first
    row is seq_len - next_n + 1 and would be negative without the clamp from
    #51538; the top-k kernels consume it as uint32 (#51593). Cover the native
    (B, next_n) path and the flattened uniform-decode kernel path."""
    device = torch.device("cuda")
    next_n, block_size, ctx = 2, 64, 300
    vllm_config = create_vllm_config(
        model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
        max_model_len=1024,
        block_size=block_size,
    )
    vllm_config.speculative_config = SimpleNamespace(
        num_speculative_tokens=next_n - 1, enable_adaptive_verification=False
    )
    kv_cache_spec = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    block_table_width = get_block_table_width(
        kv_cache_spec.max_num_blocks_per_req(vllm_config, 1024), block_size
    )
    builder = indexer.DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
    )
    builder.use_flattening = use_flattening
    builder.supports_varlen = False

    # Three live requests plus one CUDA-graph padding request (seq_len 0).
    batch = BatchSpec(seq_lens=[ctx, ctx, ctx, 0], query_lens=[next_n] * 4)
    common = create_common_attn_metadata(batch, block_size, device)
    common = replace(
        common,
        block_table_tensor=torch.zeros(
            (4, block_table_width), dtype=torch.int32, device=device
        ),
    )

    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert md.num_decodes == 4 and md.decode is not None
    lens = md.decode.seq_lens.reshape(-1).tolist()
    assert lens == [ctx - 1, ctx] * 3 + [0, 0], lens
