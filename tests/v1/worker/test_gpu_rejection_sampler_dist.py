# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output distribution of rejection sampling with probabilistic drafts.

Rejection sampling only preserves the target distribution if the draft
proposal, the acceptance uniform, and the recovery Gumbel noise are
independent. The rejection sampler keys the acceptance uniform and the
recovery noise for position P by Philox offset P, so the draft's Gumbel
stream (draft_gumbel_pos) must live in a disjoint offset range: keying the
draft by offset P as well makes the recovery draw reuse the exact noise
vector that selected the rejected draft token, inflating draft-favored
tokens (TV distance ~0.0125 vs a ~0.003 noise floor in this setup).
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.spec_decode.speculator import draft_gumbel_pos

VOCAB_SIZE = 32768
NUM_REQS = 4096
ITERS = 16
NUM_SPEC = 1
POS = 1000


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_probabilistic_draft_output_distribution():
    device = torch.device("cuda:0")
    # Target p and draft q with mass on the first 8 tokens, deliberately
    # ranked in opposite order so draft-noise reuse would visibly skew the
    # output marginal toward q's favorites.
    p_probs = torch.tensor([0.30, 0.20, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05])
    q_probs = torch.tensor([0.05, 0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.30])
    p_logits = torch.full((VOCAB_SIZE,), -30.0)
    p_logits[:8] = p_probs.log()
    q_logits = torch.full((VOCAB_SIZE,), -30.0)
    q_logits[:8] = q_probs.log()
    true_p = torch.softmax(p_logits, dim=0).double()

    num_logits = NUM_REQS * (NUM_SPEC + 1)
    target_logits = torch.empty(
        num_logits, VOCAB_SIZE, dtype=torch.float32, device=device
    )
    target_logits[0::2] = p_logits.to(device)
    target_logits[1::2] = p_logits.to(device)  # bonus row; not measured

    cu_num_logits = torch.arange(NUM_REQS + 1, device=device, dtype=torch.int32) * 2
    idx_mapping = torch.arange(NUM_REQS, device=device, dtype=torch.int32)
    expanded_idx_mapping = idx_mapping.repeat_interleave(2)
    expanded_local_pos = torch.tensor([0, 1], device=device, dtype=torch.int32).repeat(
        NUM_REQS
    )
    pos = torch.tensor([POS, POS + 1], device=device, dtype=torch.int64).repeat(
        NUM_REQS
    )
    temperature = torch.ones(NUM_REQS, dtype=torch.float32, device=device)
    q_batch = q_logits.to(device).expand(NUM_REQS, VOCAB_SIZE).contiguous()
    draft_step = torch.zeros((), dtype=torch.int64, device=device)
    # Draft rows sit at the position preceding the proposed token, exactly
    # as in DraftModelSpeculator.sample_draft.
    draft_row_pos = torch.full((NUM_REQS,), POS - 1, dtype=torch.int64, device=device)

    counts = torch.zeros(VOCAB_SIZE, dtype=torch.int64)
    accepted = 0
    gen = torch.Generator(device="cpu").manual_seed(999)
    for _ in range(ITERS):
        seeds = torch.randint(
            -(2**62), 2**62, (NUM_REQS,), generator=gen, dtype=torch.int64
        ).to(device)
        draft_logits = torch.zeros(
            NUM_REQS, NUM_SPEC, VOCAB_SIZE, dtype=torch.float32, device=device
        )
        draft_tokens = gumbel_sample(
            q_batch.clone(),
            idx_mapping,
            temperature,
            seeds,
            draft_gumbel_pos(draft_row_pos),
            apply_temperature=True,
            output_processed_logits=draft_logits.view(NUM_REQS, -1),
            output_processed_logits_col=draft_step,
        )
        # Row layout per request: [prev sampled token, draft token].
        draft_sampled = torch.stack(
            [torch.zeros_like(draft_tokens), draft_tokens], dim=1
        ).flatten()
        sampled, num_sampled = rejection_sample(
            target_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            pos,
            idx_mapping,
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            seeds,
            NUM_SPEC,
        )
        counts += torch.bincount(sampled[:, 0].cpu(), minlength=VOCAB_SIZE)
        accepted += (num_sampled == 2).sum().item()

    total = NUM_REQS * ITERS
    empirical = counts.double() / total
    tv = 0.5 * (empirical - true_p).abs().sum().item()
    # Noise floor at this sample count is ~0.0055; the coupled-stream bug
    # measures ~0.0125. Deterministic given the fixed seeds.
    assert tv < 0.008, f"output marginal deviates from target: TV={tv:.5f}"
    # Acceptance rate must stay ~sum(min(p, q)) = 0.5; decoupling the draft
    # stream must not cost acceptance.
    accept_rate = accepted / total
    assert 0.47 < accept_rate < 0.53, f"unexpected acceptance rate {accept_rate:.4f}"
