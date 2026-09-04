# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    _iter_request_chunks,
)


def test_iter_request_chunks_preserves_request_boundaries():
    cu_num_logits = np.array([0, 3, 4, 11, 13], dtype=np.int32)

    assert list(_iter_request_chunks(cu_num_logits, max_chunk_logits=5)) == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
def test_chunked_scores_match_full_batch(logprobs_mode: str):
    device = torch.device("cuda")
    cu_num_logits_np = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    num_logits_per_req = np.diff(cu_num_logits_np)
    idx_mapping_np = np.array([7, 2, 9, 1], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=4,
        cu_num_logits_np=cu_num_logits_np,
        cu_num_logits=torch.from_numpy(cu_num_logits_np).to(device),
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.from_numpy(idx_mapping_np).to(device),
        expanded_idx_mapping=torch.from_numpy(
            np.repeat(idx_mapping_np, num_logits_per_req)
        ).to(device),
        expanded_local_pos=torch.from_numpy(
            np.concatenate(
                [np.arange(count, dtype=np.int32) for count in num_logits_per_req]
            )
        ).to(device),
    )
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = SimpleNamespace(
        logprobs_mode=logprobs_mode, return_sampling_mask=False
    )
    rejection_sampler.num_speculative_steps = 3
    rejection_sampler.enable_adaptive_verification = False

    def fake_verify(
        self,
        logits,
        _draft_logits,
        _draft_sampled,
        _pos,
        cu_num_logits,
        idx_mapping,
        *_mappings,
    ):
        num_sampled = torch.diff(cu_num_logits).to(torch.int32)
        sampled = (
            idx_mapping.to(torch.int64).unsqueeze(1) + torch.arange(4, device=device)
        ) % logits.shape[1]
        return logits.float() + 1, sampled, num_sampled

    rejection_sampler._verify = MethodType(fake_verify, rejection_sampler)
    logits = torch.arange(170, dtype=torch.float32, device=device).view(10, 17)

    sampled, num_sampled, chunked_logprobs, sampling_mask_tensors = (
        rejection_sampler._verify_in_chunks(
            logits,
            input_batch,
            draft_logits=None,
            draft_sampled=torch.arange(10, device=device),
            pos=torch.arange(10, device=device),
            max_chunk_logits=5,
            max_num_logprobs=2,
        )
    )
    score_logits = logits + 1 if logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
    full_logprobs = rejection_sampler._get_logprobs_tensors(
        sampled,
        num_sampled,
        score_logits,
        input_batch.cu_num_logits,
        input_batch.cu_num_logits_np,
        max_num_logprobs=2,
    )

    assert sampled[:, 0].tolist() == idx_mapping_np.tolist()
    assert num_sampled.tolist() == num_logits_per_req.tolist()
    assert chunked_logprobs is not None
    assert full_logprobs is not None
    assert torch.equal(
        chunked_logprobs.logprob_token_ids,
        full_logprobs.logprob_token_ids,
    )
    assert torch.equal(chunked_logprobs.logprobs, full_logprobs.logprobs)
    assert torch.equal(
        chunked_logprobs.selected_token_ranks,
        full_logprobs.selected_token_ranks,
    )
    assert (
        chunked_logprobs.cu_num_generated_tokens
        == full_logprobs.cu_num_generated_tokens
    )
    assert sampling_mask_tensors is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_replay_on_off_preserves_rejection_sampling_and_rng(monkeypatch):
    device = torch.device("cuda")
    num_reqs = 3
    num_speculative_steps = 3
    rows_per_request = num_speculative_steps + 1
    vocab_size = 8
    cu_num_logits_np = np.arange(0, 13, rows_per_request, dtype=np.int32)
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        cu_num_logits_np=cu_num_logits_np,
        cu_num_logits=torch.from_numpy(cu_num_logits_np).to(device),
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.from_numpy(idx_mapping_np).to(device),
        expanded_idx_mapping=torch.repeat_interleave(
            torch.arange(num_reqs, device=device), rows_per_request
        ),
        expanded_local_pos=torch.arange(rows_per_request, device=device).repeat(
            num_reqs
        ),
    )
    processed_logits = torch.zeros((12, vocab_size), device=device)
    processed_logits[0, 0] = -float("inf")
    processed_logits[rows_per_request + 1, 1] = -float("inf")
    draft_logits = torch.zeros(
        (num_reqs, num_speculative_steps, vocab_size), device=device
    )
    draft_sampled = torch.zeros(
        (num_reqs, rows_per_request), dtype=torch.int64, device=device
    )
    draft_sampled[:, 1:] = torch.tensor([0, 1, 2], device=device)
    draft_sampled = draft_sampled.flatten()
    sampler = SimpleNamespace(
        logprobs_mode="processed_logprobs",
        return_sampling_mask=False,
        sampling_states=SimpleNamespace(
            top_k=SimpleNamespace(np=np.full(num_reqs, vocab_size)),
            temperature=SimpleNamespace(gpu=torch.ones(num_reqs, device=device)),
            seeds=SimpleNamespace(
                gpu=torch.arange(num_reqs, dtype=torch.int64, device=device)
            ),
        ),
        use_fp64_gumbel=False,
        apply_sampling_params=lambda logits, *_: logits,
    )
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = sampler
    rejection_sampler.num_speculative_steps = num_speculative_steps
    rejection_sampler.enable_adaptive_verification = False
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    pos = torch.arange(12, dtype=torch.int32, device=device)

    pack_calls: list[tuple[list[int], int, int]] = []
    pack = SamplingMaskTensors.from_logits.__func__

    def track_pack(
        cls,
        logits,
        cu_num_logits,
        num_sampled_tokens,
        max_num_kept,
        rows_per_request=1,
    ):
        pack_calls.append(
            (cu_num_logits.cpu().tolist(), max_num_kept, rows_per_request)
        )
        return pack(
            cls,
            logits,
            cu_num_logits,
            num_sampled_tokens,
            max_num_kept,
            rows_per_request,
        )

    monkeypatch.setattr(
        SamplingMaskTensors,
        "from_logits",
        classmethod(track_pack),
    )

    rng_before = torch.cuda.get_rng_state()
    off_sampled, off_counts, off_logprobs, off_masks = (
        rejection_sampler._verify_in_chunks(
            processed_logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            max_chunk_logits=8,
            max_num_logprobs=0,
        )
    )
    rng_after_off = torch.cuda.get_rng_state()
    assert pack_calls == []

    sampler.return_sampling_mask = True
    on_sampled, on_counts, on_logprobs, on_masks = rejection_sampler._verify_in_chunks(
        processed_logits,
        input_batch,
        draft_logits,
        draft_sampled,
        pos,
        max_chunk_logits=8,
        max_num_logprobs=0,
    )
    rng_after_on = torch.cuda.get_rng_state()

    assert pack_calls == [
        ([0, 4, 8], vocab_size, rows_per_request),
        ([0, 4], vocab_size, rows_per_request),
    ]
    assert off_masks is None and on_masks is not None
    assert on_masks.rows_per_request == rows_per_request
    assert on_masks.token_ids.shape == (num_reqs * rows_per_request, vocab_size)
    assert on_masks.packed_mask.shape[0] == num_reqs * rows_per_request
    assert off_counts.tolist() == [1, 2, 4]
    assert torch.equal(off_sampled, on_sampled)
    assert torch.equal(off_counts, on_counts)
    assert off_logprobs is not None and on_logprobs is not None
    assert torch.equal(off_logprobs.logprob_token_ids, on_logprobs.logprob_token_ids)
    assert torch.equal(off_logprobs.logprobs, on_logprobs.logprobs)
    assert torch.equal(
        off_logprobs.selected_token_ranks, on_logprobs.selected_token_ranks
    )
    assert torch.equal(rng_before, rng_after_off)
    assert torch.equal(rng_after_off, rng_after_on)

    masks = (
        on_masks.to_cpu_nonblocking().tolists(on_counts.cpu().numpy()).to_nested_list()
    )
    mapped_rows = [0, 4, 5, 8, 9, 10, 11]
    expected_masks = [
        torch.isfinite(processed_logits[row]).nonzero().flatten().tolist()
        for row in mapped_rows
    ]
    emitted = [
        int(on_sampled[req_idx, slot_idx])
        for req_idx, count in enumerate(on_counts.cpu().tolist())
        for slot_idx in range(count)
    ]
    expected_logprobs = torch.stack(
        [
            torch.log_softmax(processed_logits[row], dim=-1)[token_id]
            for row, token_id in zip(mapped_rows, emitted)
        ]
    )

    assert masks == expected_masks
    assert all(token_id in mask for token_id, mask in zip(emitted, masks))
    assert torch.equal(on_logprobs.logprobs[mapped_rows, 0], expected_logprobs)
