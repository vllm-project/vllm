# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DiffusionGemma sampler helpers."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import (
    DiffusionSampler,
    _compiled_sample_step,
)


def test_compiled_sample_step_self_conditioning_matmul():
    """``_compiled_sample_step`` should not crash on the self-conditioning
    matmul that previously failed under ``torch.compile`` when
    ``embed_weight`` and ``probs`` had mismatched vocab dimensions.

    Regression test for https://github.com/vllm-project/vllm/issues/47129.
    """
    num_decode = 2
    num_reqs = 3
    max_num_reqs = 4
    CL = 8
    vocab = 16
    hidden = 8
    ST = 2
    device = "cpu"

    logits = torch.randn(num_decode * CL, vocab, device=device)
    decode_slots = torch.arange(num_decode, device=device, dtype=torch.int64)
    decode_idx = torch.arange(num_decode, device=device, dtype=torch.int64)
    all_slots = torch.arange(num_reqs, device=device, dtype=torch.int64)
    valid_canvas_len = torch.full(
        (num_decode,), CL, device=device, dtype=torch.int64
    )
    canvas = torch.zeros(max_num_reqs, CL, device=device, dtype=torch.int64)
    argmax_canvas = torch.zeros(
        max_num_reqs, CL, device=device, dtype=torch.int64
    )
    step_tensor = torch.zeros(max_num_reqs, device=device, dtype=torch.int32)
    is_encoder_phase = torch.zeros(
        max_num_reqs, device=device, dtype=torch.bool
    )
    confident_tensor = torch.zeros(
        max_num_reqs, device=device, dtype=torch.bool
    )
    sc_embeds = torch.zeros(
        max_num_reqs, CL, hidden, device=device, dtype=torch.float32
    )
    local_embed_weight = torch.randn(vocab, hidden, device=device)
    normalizer = torch.tensor(1.0, device=device)
    history = torch.zeros(
        max_num_reqs, ST, CL, device=device, dtype=torch.int64
    )
    history_len_tensor = torch.zeros(
        max_num_reqs, device=device, dtype=torch.int32
    )
    sampled = torch.zeros(num_reqs, CL, device=device, dtype=torch.int32)
    num_sampled = torch.zeros(num_reqs, device=device, dtype=torch.int32)
    draft_tokens = torch.zeros(
        max_num_reqs, CL, device=device, dtype=torch.int64
    )

    scaled = _compiled_sample_step(
        logits,
        decode_slots,
        decode_idx,
        all_slots,
        valid_canvas_len,
        canvas,
        argmax_canvas,
        step_tensor,
        is_encoder_phase,
        confident_tensor,
        sc_embeds,
        local_embed_weight,
        normalizer,
        history,
        history_len_tensor,
        sampled,
        num_sampled,
        draft_tokens,
        max_denoising_steps=16.0,
        t_min=0.0,
        t_max=1.0,
        confidence_threshold=10.0,
        vocab_size=vocab,
        CL=CL,
        ST=ST,
        entropy_bound=0.1,
        sc_vocab_start=0,
        sc_vocab_end=vocab,
        tp_size=1,
        tp_group_name="",
    )

    assert scaled.shape == (num_decode, CL, vocab)
    # Self-conditioning embeddings should have been written for the decode
    # slots (all of them, because we left is_encoder_phase as False and the
    # function treated every slot as a denoise step).
    assert sc_embeds[decode_slots].abs().sum() > 0


def test_diffusion_sampler_pre_slices_embed_weight():
    """``DiffusionSampler`` should pre-slice ``embed_weight`` once at init
    and reject an embedding that is smaller than the expected shard size.
    """
    mock_sampler = MagicMock()
    mock_sampler.sampling_states = MagicMock()
    mock_sampler.req_states = MagicMock()
    mock_diffusion_states = MagicMock()
    mock_diffusion_states.max_num_reqs = 4
    mock_diffusion_states.device = torch.device("cpu")

    vocab = 16
    hidden = 8
    embed_weight = torch.randn(vocab, hidden)

    sampler = DiffusionSampler(
        sampler=mock_sampler,
        diffusion_config=None,
        vocab_size=vocab,
        diffusion_states=mock_diffusion_states,
        confidence_threshold=10.0,
        t_min=0.0,
        t_max=1.0,
        entropy_bound=0.1,
        embed_weight=embed_weight,
        normalizer=torch.tensor(1.0),
        sc_vocab_start=0,
        sc_vocab_end=vocab,
        tp_size=1,
        tp_group_name="",
    )

    assert sampler.local_embed_weight.shape == (vocab, hidden)
    assert sampler.local_embed_weight.data_ptr() == embed_weight.data_ptr()

    # A shard size larger than the weight should fail at init time rather
    # than inside the compiled step.
    with pytest.raises(ValueError, match="self-conditioning shard size"):
        DiffusionSampler(
            sampler=mock_sampler,
            diffusion_config=None,
            vocab_size=vocab,
            diffusion_states=mock_diffusion_states,
            confidence_threshold=10.0,
            t_min=0.0,
            t_max=1.0,
            entropy_bound=0.1,
            embed_weight=embed_weight,
            normalizer=torch.tensor(1.0),
            sc_vocab_start=0,
            sc_vocab_end=vocab + 4,
            tp_size=1,
            tp_group_name="",
        )
