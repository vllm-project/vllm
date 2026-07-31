# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the EAGLE speculator's draft attention metadata builder.

These tests guard the regression where ``_build_draft_attn_metadata`` did
not populate ``seq_lens_cpu_upper_bound`` on the per-step
``CommonAttentionMetadata``. Several downstream attention backends and
helpers (``split_decodes_prefills_and_extends``, the MLA indexer,
flex-attention, cross-attention) assert this field is non-None, so
omitting it caused crashes at the start of draft decode for certain
backends (e.g. ``ROCM_AITER_FA`` with eagle/eagle3 spec decode):

    AssertionError: assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.v1.worker.gpu.spec_decode import speculator as base_speculator
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator


def _make_fake_speculator(
    *,
    max_num_reqs: int = 8,
    max_num_tokens: int = 16,
    max_model_len: int = 1024,
    draft_max_seq_len: int = 1024,
) -> SimpleNamespace:
    """Build a fake EagleSpeculator with just the attributes used by
    ``_build_draft_attn_metadata``. We deliberately avoid constructing a
    real ``EagleSpeculator`` because that requires a full ``VllmConfig``
    and a draft model.
    """
    fake_input_buffers = SimpleNamespace(
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32),
    )
    fake_block_tables = SimpleNamespace(
        input_block_tables=[torch.zeros(max_num_reqs, 4, dtype=torch.int32)],
        slot_mappings=torch.zeros(1, max_num_tokens, dtype=torch.int64),
    )
    return SimpleNamespace(
        arange=torch.arange(max_num_reqs + 1, dtype=torch.int32, device="cpu"),
        block_tables=fake_block_tables,
        input_buffers=fake_input_buffers,
        attn_groups=[],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        max_model_len=max_model_len,
        draft_max_seq_len=draft_max_seq_len,
    )


def _run_build(fake, *, num_reqs, num_reqs_padded, num_tokens_padded, base, step):
    captured: dict[str, object] = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return {}

    with patch.object(base_speculator, "build_attn_metadata", fake_build_attn_metadata):
        EagleSpeculator._build_draft_attn_metadata(
            fake,  # type: ignore[arg-type]
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            seq_lens_cpu_upper_bound=base,
            step=step,
        )
    return captured


def test_build_draft_attn_metadata_sets_seq_lens_cpu_upper_bound():
    """The fix: every per-step ``CommonAttentionMetadata`` carries a non-None
    ``seq_lens_cpu_upper_bound`` derived from the target-side upper bound plus
    the current draft-step offset. Padded entries are zeroed (matching the
    main model runner's convention)."""
    fake = _make_fake_speculator()
    base = torch.tensor([100, 200, 300, 0], dtype=torch.int32)

    captured = _run_build(
        fake, num_reqs=3, num_reqs_padded=4, num_tokens_padded=4, base=base, step=2
    )

    bound = captured["seq_lens_cpu_upper_bound"]
    assert isinstance(bound, torch.Tensor), (
        "seq_lens_cpu_upper_bound must be a tensor, not None"
    )
    assert bound.shape == (4,), (
        f"expected shape (num_reqs_padded=4,), got {bound.shape}"
    )
    assert bound.device.type == "cpu"
    assert bound.dtype == torch.int32
    # base[:num_reqs] + step, padded tail zeroed.
    assert torch.equal(bound, torch.tensor([102, 202, 302, 0], dtype=torch.int32))


def test_build_draft_attn_metadata_handles_zero_unpadded_reqs():
    """Edge case: when ``num_reqs == 0`` the upper-bound tensor must
    still be a valid all-zero tensor of length ``num_reqs_padded``."""
    fake = _make_fake_speculator()
    base = torch.zeros(2, dtype=torch.int32)

    captured = _run_build(
        fake, num_reqs=0, num_reqs_padded=2, num_tokens_padded=2, base=base, step=1
    )

    bound = captured["seq_lens_cpu_upper_bound"]
    assert isinstance(bound, torch.Tensor)
    assert bound.shape == (2,)
    assert torch.equal(bound, torch.zeros(2, dtype=torch.int32))


def test_build_draft_attn_metadata_clamps_to_max_model_len():
    """The per-request upper bound (target bound + step) is clamped to the
    model length so it never exceeds the allocated KV range."""
    fake = _make_fake_speculator(max_model_len=1024)
    base = torch.tensor([1023, 500], dtype=torch.int32)

    captured = _run_build(
        fake, num_reqs=2, num_reqs_padded=2, num_tokens_padded=2, base=base, step=3
    )

    bound = captured["seq_lens_cpu_upper_bound"]
    # 1023 + 3 = 1026 -> clamped to 1024; 500 + 3 = 503 unaffected.
    assert torch.equal(bound, torch.tensor([1024, 503], dtype=torch.int32))
