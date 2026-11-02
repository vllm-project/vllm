# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA prefill through AITER FlashKDA on ROCm. Nothing here
needs a GPU: the platform and the AITER probe are faked."""

import pytest
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.models.glm5next.common import kda
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def test_rocm_flashkda_resolves_to_aiter(monkeypatch):
    """``--kda-prefill-backend flashkda`` selects AITER FlashKDA on ROCm
    instead of being rejected."""
    monkeypatch.setattr(kda.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        rocm_aiter_ops, "is_kda_flashkda_prefill_available", lambda: True
    )
    backend = kda._resolve_kda_prefill_backend(
        "flashkda",
        head_dim=128,
        dtype=torch.bfloat16,
        lower_bound=-5.0,
        state_dtype=torch.float32,
    )
    assert backend == "aiter_flashkda"


@pytest.mark.parametrize(
    ("num_spec_decodes", "expected"), [(0, True), (1, False)], ids=["no-spec", "spec"]
)
def test_aiter_flashkda_skips_steps_with_spec_tokens(num_spec_decodes, expected):
    """Steps that carry speculative-decode tokens keep the Triton chunk path."""
    spec_tokens = 4 * num_spec_decodes
    metadata = GDNAttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=74,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=spec_tokens,
        num_actual_tokens=74 + spec_tokens,
        spec_sequence_masks=(
            torch.tensor([True] * num_spec_decodes + [False] * 2)
            if num_spec_decodes
            else None
        ),
    )
    assert kda._aiter_flashkda_runs_prefill("aiter_flashkda", metadata) is expected
