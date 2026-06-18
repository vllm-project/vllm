# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDNAttentionMetadataBuilder.build() — specifically the
reclassification of non-spec decodes as prefills when spec decodes exist.
Covers the fix for https://github.com/vllm-project/vllm/issues/34845.
"""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import CUDAGraphMode, SpeculativeConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    mamba_cache_mode: str = "none",
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", block_size=BLOCK_SIZE)
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    # Disable CUDA graphs for unit tests — test block tables are sized
    # for the actual sequences, not for max_model_len as in production.
    vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


# =====================================================================
# Tests for mamba_cache_mode="all"
# =====================================================================


def test_gdn_build_all_mode_block_indices():
    """In 'all' mode with a mixed prefill+decode batch, verify block index
    fields and full block table state indices are computed correctly."""
    builder = _create_gdn_builder(mamba_cache_mode="all")
    # 2 decode requests (query_len=1) + 1 prefill request (query_len=50)
    seq_lens = [32, 64, 100]
    query_lens = [1, 1, 50]
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE, arange_block_indices=True
    )
    meta = builder.build(common_prefix_len=0, common_attn_metadata=common)

    # Recompute expected values from the formulas
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    num_computed_tokens = seq_lens_t - torch.tensor(
        query_lens, dtype=torch.int32
    )

    expected_last_scheduled = (
        cdiv(seq_lens_t, BLOCK_SIZE) - 1
    ).clamp(min=0)
    expected_first_scheduled = cdiv(num_computed_tokens + 1, BLOCK_SIZE) - 1
    expected_last_computed = (
        cdiv(num_computed_tokens, BLOCK_SIZE) - 1
    ).clamp(min=0)

    # Verify block_idx_last_scheduled_token (covers all requests)
    assert meta.block_idx_last_scheduled_token is not None
    torch.testing.assert_close(
        meta.block_idx_last_scheduled_token,
        expected_last_scheduled,
    )

    # Verify block_idx_last_computed_token (covers all requests)
    assert meta.block_idx_last_computed_token is not None
    torch.testing.assert_close(
        meta.block_idx_last_computed_token,
        expected_last_computed,
    )

    # In non-spec mode: num_decodes=2, num_prefills=1
    assert meta.num_decodes == 2
    assert meta.num_prefills == 1

    # block_idx_first_scheduled_token_p is sliced to prefill requests only
    assert meta.block_idx_first_scheduled_token_p is not None
    torch.testing.assert_close(
        meta.block_idx_first_scheduled_token_p,
        expected_first_scheduled[2:],  # only the prefill request
    )

    # state_indices_all_d should be the full block table for decode requests
    assert meta.state_indices_all_d is not None
    assert meta.state_indices_all_d.shape[0] == 2  # 2 decode requests
    # Full block table columns (not just column 0)
    assert meta.state_indices_all_d.shape[1] > 1
    # Verify it matches the first 2 rows of the block table
    expected_bt = common.block_table_tensor[:2]
    torch.testing.assert_close(meta.state_indices_all_d, expected_bt)

    # state_indices_all_p should be the full block table for prefill requests
    assert meta.state_indices_all_p is not None
    assert meta.state_indices_all_p.shape[0] == 1  # 1 prefill request
    expected_bt_p = common.block_table_tensor[2:]
    torch.testing.assert_close(meta.state_indices_all_p, expected_bt_p)


def test_gdn_build_all_mode_spec_decode():
    """In 'all' mode with spec decode, verify that
    block_idx_last_scheduled_token_prev_step is correctly computed
    from prev_last_scheduled_idx."""
    builder = _create_gdn_builder(
        num_speculative_tokens=2, mamba_cache_mode="all"
    )
    # Both are spec decode requests
    seq_lens = [50, 30]
    query_lens = [3, 3]
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, DEVICE, arange_block_indices=True
    )

    # Simulate prev_last_scheduled_idx from the previous scheduler step
    prev_last_scheduled_idx = torch.tensor([1, 0], dtype=torch.int32)

    num_decode_draft_tokens = torch.tensor([2, 2], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32, device=DEVICE)

    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens,
        num_accepted_tokens=num_accepted_tokens,
        prev_last_scheduled_idx=prev_last_scheduled_idx,
    )

    # With valid (>=0) prev_last_scheduled_idx, the prev_step tensor should
    # use those values directly (torch.where branch).
    assert meta.block_idx_last_scheduled_token_prev_step is not None
    torch.testing.assert_close(
        meta.block_idx_last_scheduled_token_prev_step,
        prev_last_scheduled_idx,
    )

    # Also verify that block_idx_last_scheduled_token is set
    assert meta.block_idx_last_scheduled_token is not None
    expected_last_scheduled = (
        cdiv(torch.tensor(seq_lens, dtype=torch.int32), BLOCK_SIZE) - 1
    ).clamp(min=0)
    torch.testing.assert_close(
        meta.block_idx_last_scheduled_token,
        expected_last_scheduled,
    )


def test_gdn_build_all_mode_align_unchanged():
    """Regression test: with mamba_cache_mode='align', all 'all'-mode
    fields must be None, and normal classify behavior is unchanged."""
    builder = _create_gdn_builder(mamba_cache_mode="align")
    # Mixed batch: 2 decodes + 1 prefill
    seq_lens = [32, 64, 100]
    query_lens = [1, 1, 50]
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    meta = _build(builder, batch)

    # All "all"-mode fields must be None
    assert meta.block_idx_last_scheduled_token is None
    assert meta.block_idx_first_scheduled_token_p is None
    assert meta.block_idx_last_computed_token is None
    assert meta.block_idx_last_scheduled_token_prev_step is None
    assert meta.state_indices_all_p is None
    assert meta.state_indices_all_d is None

    # Normal classification still works
    assert meta.num_decodes == 2
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == 50
