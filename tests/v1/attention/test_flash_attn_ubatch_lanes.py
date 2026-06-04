# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-microbatch state isolation for unified FlashAttention (RFC #42449).

The unified FA backend is a single DBO-aware instance that indexes its per-step
state by microbatch id (``self._step[ubatch_id]``). The correctness-critical
claim is that dual-batch-overlap microbatches, which run concurrently sharing
the one backend, never clobber each other's per-step state. This pins that:
building two microbatch slots with different batch shapes leaves each slot
carrying its own state.
"""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="FlashAttention requires CUDA"
)
def test_ubatch_slots_hold_independent_per_step_state():
    device = torch.device("cuda")
    vllm_config = create_vllm_config(
        model_name="ai21labs/Jamba-tiny-dev", max_model_len=512, block_size=16
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    backend = FlashAttentionBackend(
        kv_cache_spec,
        ["layer0"],
        vllm_config,
        device,
        kv_cache_group_ids=[0],
        num_ubatches=2,
    )

    # Single instance with one per-microbatch state slot each.
    assert backend._num_ubatches == 2
    assert backend._step == [None, None]

    def _cm(seq_len, q_len):
        return create_common_attn_metadata(
            BatchSpec(seq_lens=[seq_len], query_lens=[q_len]),
            block_size=16,
            device=device,
            arange_block_indices=True,
        )

    cm0 = _cm(seq_len=8, q_len=8)  # prefill-shaped microbatch
    cm1 = _cm(seq_len=33, q_len=1)  # decode-shaped microbatch

    backend.prep_forward(cm0, ubatch_id=0)
    # The (single) backend is the routed per-step object.
    assert backend.attn_metadata is backend
    assert backend._step[0].num_actual_tokens == 8

    backend.prep_forward(cm1, ubatch_id=1)
    # Building slot 1 must not have disturbed slot 0's state.
    assert backend._step[1].num_actual_tokens == 1
    assert backend._step[0].num_actual_tokens == 8

    # Per-step tensors are each slot's own, not shared.
    assert backend._step[0].seq_lens is not backend._step[1].seq_lens
    torch.testing.assert_close(backend._step[0].seq_lens, cm0.seq_lens)
    torch.testing.assert_close(backend._step[1].seq_lens, cm1.seq_lens)
