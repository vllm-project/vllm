# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for block-verify rejection sampling.

Covers the PyTorch reference helpers and Triton-vs-PyTorch parity. The
dispatcher gate in `rejection_sample` ensures these paths only run when
`draft_probs is not None` and `max_spec_len >= 3`, so tests exercise only
that regime.
"""

import pytest
import torch

from vllm.v1.sample.rejection_sampler import (
    _BLOCK_VERIFY_VOCAB_BLOCK,
    PLACEHOLDER_TOKEN_ID,
    rejection_random_sample_block_verify_kernel,
    rejection_random_sample_block_verify_pytorch,
    sample_recovered_tokens_block_verify_kernel,
    sample_recovered_tokens_blockwise_pytorch,
)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_block_verify_pytorch_accepts_and_appends_bonus():
    device = _device()
    batch_size = 2
    max_spec_len = 3
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID, device=device
    )

    cu_num_draft_tokens = torch.tensor([2, 1], device=device)
    draft_token_ids = torch.tensor([1, 0, 2], device=device)
    draft_probs = torch.tensor(
        [
            [0.0, 0.6, 0.0, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.5, 0.0, 0.0],
        ],
        device=device,
    )
    target_probs = torch.tensor(
        [
            [0.0, 0.8, 0.0, 0.2],
            [0.2, 0.1, 0.3, 0.4],
            [0.9, 0.1, 0.0, 0.0],
        ],
        device=device,
    )
    bonus_token_ids = torch.tensor([[100], [200]], device=device)
    recovered_token_ids = torch.tensor([1, 2, 3], device=device)
    uniform_probs = torch.tensor([0.7, 0.6, 0.5], device=device)
    is_greedy = torch.tensor([False, False], device=device)
    vocab_size = 4

    rejection_random_sample_block_verify_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
    )

    # Request 0: both drafts accepted, bonus written at position 2.
    assert output_token_ids[0, 0].item() == 1
    assert output_token_ids[0, 1].item() == 0
    assert output_token_ids[0, 2].item() == 100


def test_sample_recovered_blockwise_pytorch_selects_argmax():
    device = _device()
    output_token_ids = torch.empty(2, dtype=torch.int32, device=device)
    cu_num_draft_tokens = torch.tensor([1, 2], device=device)
    draft_token_ids = torch.tensor([0, 1], device=device)
    draft_probs = torch.tensor(
        [
            [0.6, 0.1, 0.3],
            [0.2, 0.7, 0.1],
        ],
        device=device,
    )
    target_probs = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.3, 0.6, 0.1],
        ],
        device=device,
    )
    q = torch.tensor(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.8, 0.1],
        ],
        device=device,
    )

    sample_recovered_tokens_blockwise_pytorch(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size=3,
    )
    assert output_token_ids[0].item() == 0
    assert output_token_ids[1].item() == 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton parity tests require CUDA"
)
def test_triton_parity_with_pytorch():
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size = 4
    max_spec_len = 5
    vocab_size = 128

    num_draft_per_req = torch.tensor([5, 3, 5, 4], device=device)
    cu_num_draft_tokens = torch.cumsum(num_draft_per_req, dim=0).to(torch.int32)
    num_tokens = int(cu_num_draft_tokens[-1].item())

    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device)
    draft_logits = torch.randn(num_tokens, vocab_size, device=device)
    target_logits = torch.randn(num_tokens, vocab_size, device=device)
    draft_probs = draft_logits.softmax(dim=-1).contiguous()
    target_probs = target_logits.softmax(dim=-1).contiguous()
    bonus_token_ids = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    uniform_probs = torch.rand(num_tokens, device=device)
    is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)
    # Fix q so both paths sample the same recovered distribution.
    q = torch.full((batch_size, vocab_size), 1.0, device=device)

    # PyTorch reference.
    rec_pt = torch.empty(num_tokens, dtype=torch.int32, device=device)
    sample_recovered_tokens_blockwise_pytorch(
        rec_pt,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
    )
    out_pt = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    rejection_random_sample_block_verify_pytorch(
        out_pt,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        rec_pt,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
    )

    # Triton.
    rec_tri = torch.empty(num_tokens, dtype=torch.int32, device=device)
    sample_recovered_tokens_block_verify_kernel[(batch_size, max_spec_len)](
        rec_tri,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        q,
        vocab_size,
        BLOCK_SIZE=_BLOCK_VERIFY_VOCAB_BLOCK,
    )
    out_tri = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    rejection_random_sample_block_verify_kernel[(batch_size,)](
        out_tri,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        rec_tri,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        BLOCK_SIZE=_BLOCK_VERIFY_VOCAB_BLOCK,
    )

    assert torch.equal(rec_pt, rec_tri), (
        f"recovered mismatch: pt={rec_pt.tolist()} tri={rec_tri.tolist()}"
    )
    assert torch.equal(out_pt, out_tri), (
        f"accept mismatch:\npt={out_pt.tolist()}\ntri={out_tri.tolist()}"
    )


def test_speculative_config_has_verify_method_default_standard():
    from vllm.config.speculative import SpeculativeConfig

    field_names = {f.name for f in SpeculativeConfig.__dataclass_fields__.values()}
    assert "verify_method" in field_names
    assert SpeculativeConfig.__dataclass_fields__["verify_method"].default == "standard"
