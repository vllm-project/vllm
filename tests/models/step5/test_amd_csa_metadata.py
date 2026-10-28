# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 CSA per-step metadata on ROCm."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.models.step5.common.csa_cache import Step5CSAMetadataBuilder
from vllm.platforms import current_platform

REGION = 8

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Step5 CSA metadata is the ROCm path"
)


def _builder(num_speculative_tokens: int | None = None) -> Step5CSAMetadataBuilder:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                sparse_config=SimpleNamespace(region_block_size=REGION, topk=512)
            )
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        speculative_config=(
            None
            if num_speculative_tokens is None
            else SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
        ),
    )
    return Step5CSAMetadataBuilder(SimpleNamespace(), [], config, torch.device("cuda"))


def test_builder_positions_fallback():
    """Spec-decode metadata carries no positions; the builder must derive them."""
    spec = BatchSpec(seq_lens=[40, 9, 130], query_lens=[4, 9, 1])
    m = create_common_attn_metadata(spec, block_size=16, device=torch.device("cuda"))
    assert m.positions is None
    builder = _builder()
    md = builder.build(0, m)
    assert md.positions is not None and md.token_req is not None

    want_pos = [
        p for s, q in zip(spec.seq_lens, spec.query_lens) for p in range(s - q, s)
    ]
    want_req = [r for r, q in enumerate(spec.query_lens) for _ in range(q)]
    assert md.positions.tolist() == want_pos
    assert md.token_req.tolist() == want_req


def test_builder_plans_prefill_only_for_decode_or_mtp_verify():
    """Equal-length prefills are not mistaken for decode or MTP verification."""
    prefill = BatchSpec(seq_lens=[9, 9], query_lens=[9, 9])
    md = _builder().build(
        0, create_common_attn_metadata(prefill, 16, device=torch.device("cuda"))
    )
    assert md.prefill_groups is not None and md.prefill_groups.shape == (18, 10)

    decode = BatchSpec(seq_lens=[40, 9, 130], query_lens=[1, 1, 1])
    md = _builder().build(
        0, create_common_attn_metadata(decode, 16, device=torch.device("cuda"))
    )
    assert md.prefill_groups is None

    mtp = BatchSpec(seq_lens=[40, 9, 130], query_lens=[4, 4, 4])
    md = _builder(num_speculative_tokens=3).build(
        0, create_common_attn_metadata(mtp, 16, device=torch.device("cuda"))
    )
    assert md.prefill_groups is None

    mixed = BatchSpec(seq_lens=[40, 9, 130], query_lens=[1, 9, 1])
    md = _builder().build(
        0, create_common_attn_metadata(mixed, 16, device=torch.device("cuda"))
    )
    assert md.prefill_groups is not None and md.prefill_groups.shape == (11, 10)
    assert md.prefill_indexer is None  # 17 regions fit in top-k
