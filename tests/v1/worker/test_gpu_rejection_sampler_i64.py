# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""int64 indexing in the block-verification rejection sampler kernels.

With a GLM-scale vocab (~155k), logit_idx * vocab_stride exceeds int32 once
logit_idx >= ~13.8k (and req_state_idx * draft_stride_0 similarly), so the
block-verification kernels must promote indices to int64 before the stride
multiply. These tests place one request at a high logit/request-state index
and check its outputs match the same request run alone at index 0.

Each case allocates ~5 GiB of GPU memory.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

VOCAB_SIZE = 155264
NUM_SPECULATIVE_STEPS = 2
LOGITS_PER_REQ = NUM_SPECULATIVE_STEPS + 1


def _run(
    target_logits: torch.Tensor,  # [LOGITS_PER_REQ, V], the request under test
    draft_logits: torch.Tensor,  # [NUM_SPECULATIVE_STEPS, V]
    draft_tokens: list[int],
    num_reqs: int,
    max_num_reqs: int,
    roi_req: int,
    req_state_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run rejection_sample on a batch of uniform k=2 requests, with the
    request under test at request index `roi_req` and draft-logits row
    `req_state_idx`. Returns that request's (sampled, num_sampled)."""
    num_logits = num_reqs * LOGITS_PER_REQ
    roi_row = roi_req * LOGITS_PER_REQ

    full_target = torch.randn(
        num_logits, VOCAB_SIZE, dtype=torch.bfloat16, device=device
    )
    full_target[roi_row : roi_row + LOGITS_PER_REQ] = target_logits
    full_draft = torch.randn(
        max_num_reqs,
        NUM_SPECULATIVE_STEPS,
        VOCAB_SIZE,
        dtype=torch.bfloat16,
        device=device,
    )
    full_draft[req_state_idx] = draft_logits

    # Draft token for step i of request r lives at row r*LOGITS_PER_REQ + i + 1.
    draft_sampled = torch.zeros(num_logits, dtype=torch.int64, device=device)
    for i, tok in enumerate(draft_tokens):
        draft_sampled[roi_row + i + 1] = tok

    cu_num_logits = torch.arange(
        0, num_logits + 1, LOGITS_PER_REQ, dtype=torch.int32, device=device
    )
    # Filler requests all map to draft-logits row 0.
    idx_mapping = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    idx_mapping[roi_req] = req_state_idx
    expanded_idx_mapping = idx_mapping.repeat_interleave(LOGITS_PER_REQ)
    expanded_local_pos = (
        torch.arange(LOGITS_PER_REQ, dtype=torch.int32, device=device)
        .repeat(num_reqs)
        .contiguous()
    )

    # Positions only need to match between the reference and big-batch runs
    # for the request under test (they key the Gumbel noise and uniforms).
    pos = torch.arange(num_logits, dtype=torch.int64, device=device)
    pos[roi_row : roi_row + LOGITS_PER_REQ] = torch.arange(
        1000, 1000 + LOGITS_PER_REQ, dtype=torch.int64, device=device
    )
    temperature = torch.ones(max_num_reqs, dtype=torch.float32, device=device)
    seeds = torch.full((max_num_reqs,), 42, dtype=torch.int64, device=device)

    sampled, num_sampled = rejection_sample(
        target_logits=full_target,
        draft_logits=full_draft,
        draft_sampled=draft_sampled,
        cu_num_logits=cu_num_logits,
        pos=pos,
        idx_mapping=idx_mapping,
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        temperature=temperature,
        seed=seeds,
        num_speculative_steps=NUM_SPECULATIVE_STEPS,
        use_block_verification=True,
    )
    return sampled[roi_req].clone(), num_sampled[roi_req].clone()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    # Overflow triggers when roi_req * LOGITS_PER_REQ * VOCAB_SIZE (target
    # logits) or req_state_idx * NUM_SPECULATIVE_STEPS * VOCAB_SIZE (draft
    # logits) exceeds 2**31.
    ("num_reqs", "max_num_reqs", "roi_req", "req_state_idx"),
    [
        (5500, 4, 5000, 2),  # target-side: logit_idx 15000 * V > 2**31
        (22, 8192, 11, 8000),  # draft-side: req_state_idx 8000 * 2V > 2**31
    ],
)
def test_block_verification_i64_indexing(
    num_reqs: int, max_num_reqs: int, roi_req: int, req_state_idx: int
):
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    target_logits = torch.randn(
        LOGITS_PER_REQ, VOCAB_SIZE, dtype=torch.bfloat16, device=device
    )
    draft_logits = torch.randn(
        NUM_SPECULATIVE_STEPS, VOCAB_SIZE, dtype=torch.bfloat16, device=device
    )
    draft_tokens = [123, 45678]

    sampled_ref, num_sampled_ref = _run(
        target_logits,
        draft_logits,
        draft_tokens,
        num_reqs=1,
        max_num_reqs=1,
        roi_req=0,
        req_state_idx=0,
        device=device,
    )
    sampled, num_sampled = _run(
        target_logits,
        draft_logits,
        draft_tokens,
        num_reqs=num_reqs,
        max_num_reqs=max_num_reqs,
        roi_req=roi_req,
        req_state_idx=req_state_idx,
        device=device,
    )
    assert num_sampled.item() == num_sampled_ref.item()
    n = num_sampled_ref.item()
    assert torch.equal(sampled[:n], sampled_ref[:n])
