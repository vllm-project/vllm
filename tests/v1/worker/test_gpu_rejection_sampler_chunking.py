# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import MethodType, SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    _iter_request_chunks,
)
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker


def _make_input_batch(
    cu_num_logits_np: np.ndarray,
    idx_mapping_np: np.ndarray,
    device: torch.device,
) -> SimpleNamespace:
    """Minimal InputBatch fields read by RejectionSampler._verify_in_chunks."""
    num_reqs = len(idx_mapping_np)
    num_logits_per_req = np.diff(cu_num_logits_np)
    return SimpleNamespace(
        num_reqs=num_reqs,
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
        seq_lens_cpu_upper_bound=torch.full(
            (num_reqs,), 64, dtype=torch.int32, device="cpu"
        ),
    )


def _make_rejection_sampler(
    sampler: SimpleNamespace, num_speculative_steps: int
) -> RejectionSampler:
    """RejectionSampler for standard, fixed-boundary, unwatermarked verification."""
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = sampler
    rejection_sampler.num_speculative_steps = num_speculative_steps
    rejection_sampler.enable_adaptive_verification = False
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    rejection_sampler.fly_window_size = 0
    rejection_sampler.fly_entropy_threshold = 0.3
    rejection_sampler.fly_entropy_top_k = 3
    rejection_sampler.watermark_key = None
    return rejection_sampler


def test_iter_request_chunks_preserves_request_boundaries():
    cu_num_logits = np.array([0, 3, 4, 11, 13], dtype=np.int32)

    assert list(_iter_request_chunks(cu_num_logits, max_chunk_logits=5)) == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Requires CUDA or XPU",
)
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_chunked_grammar_preserves_raw_scores(temperature):
    device = torch.device(current_platform.device_type)
    vocab_size = 32
    counts = np.array([3, 1, 2], dtype=np.int32)
    cu = np.array([0, 3, 4, 6], dtype=np.int32)
    batch = InputBatch.make_dummy(3, 6, InputBuffers(3, 6, device))
    batch = replace(
        batch,
        cu_num_logits_np=cu,
        cu_num_logits=torch.from_numpy(cu).to(device),
        expanded_idx_mapping=torch.from_numpy(
            np.repeat(batch.idx_mapping_np, counts)
        ).to(device),
        expanded_local_pos=torch.tensor([0, 1, 2, 0, 0, 1], device=device),
    )
    req_states = RequestState(3, 16, 6, 2, vocab_size, device)
    sampler = Sampler(VllmConfig(), 3, vocab_size, device, req_states)
    for i in range(3):
        sampler.add_request(
            i, SamplingParams(temperature=temperature, logprobs=2, seed=7)
        )
    sampler.apply_staged_writes()
    rejection_sampler = RejectionSampler(
        sampler,
        SpeculativeConfig(method="ngram", num_speculative_tokens=2),
        device,
    )
    worker = StructuredOutputsWorker(9, vocab_size, device, mask_stride=3)
    # Reverse grammar order, leave req 1 unconstrained, and pad req 2's last row.
    allowed = [4, 5, 6, 1, 2, 3]
    bitmask = np.array([[1 << token] for token in allowed], dtype=np.int32)
    apply_grammar = worker.prepare_grammar_bitmask(
        batch, [batch.req_ids[2], batch.req_ids[0]], bitmask
    )
    logits = (
        torch.arange(6 * vocab_size, device=device, dtype=torch.float32).view(
            6, vocab_size
        )
        / 100
    )
    original = logits.clone()
    draft = torch.tensor([0, 1, 2, 0, 0, 4], dtype=torch.int32, device=device)
    args = (logits, batch, None, draft, torch.arange(6, device=device))
    chunked = rejection_sampler._verify_in_chunks(
        *args, max_chunk_logits=3, max_num_logprobs=2, apply_grammar=apply_grammar
    )
    full = rejection_sampler._verify_in_chunks(
        *args, max_chunk_logits=6, max_num_logprobs=2, apply_grammar=apply_grammar
    )
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    for result in (chunked, full):
        sampled, num_sampled, scores, _ = result
        assert sampled[0, :3].tolist() == [1, 2, 3]
        assert sampled[2, :2].tolist() == [4, 5]
        assert num_sampled.tolist() == [3, 1, 2]
        assert scores is not None
        expected = torch.log_softmax(original, dim=-1)
        torch.testing.assert_close(
            scores.logprobs[:3, 0], expected[:3, [1, 2, 3]].diag()
        )
    assert chunked[2] is not None and full[2] is not None
    torch.testing.assert_close(chunked[2].logprobs, full[2].logprobs)


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Requires CUDA or XPU",
)
@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
def test_chunked_scores_match_full_batch(
    monkeypatch: pytest.MonkeyPatch, logprobs_mode: str
):
    device = torch.device(current_platform.device_type)
    cu_num_logits_np = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    num_logits_per_req = np.diff(cu_num_logits_np)
    idx_mapping_np = np.array([7, 2, 9, 1], dtype=np.int32)
    input_batch = _make_input_batch(cu_num_logits_np, idx_mapping_np, device)
    rejection_sampler = _make_rejection_sampler(
        SimpleNamespace(logprobs_mode=logprobs_mode, return_sampling_mask=False),
        num_speculative_steps=3,
    )

    def fake_verify(
        self,
        logits,
        _draft_logits,
        _draft_sampled,
        _pos,
        cu_num_logits,
        idx_mapping,
        *_mappings,
        **_grammar,
    ):
        num_sampled = torch.diff(cu_num_logits).to(torch.int32)
        sampled = (
            idx_mapping.to(torch.int64).unsqueeze(1) + torch.arange(4, device=device)
        ) % logits.shape[1]
        return logits.float() + 1, sampled, num_sampled

    monkeypatch.setattr(
        rejection_sampler, "_verify", MethodType(fake_verify, rejection_sampler)
    )
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


@pytest.mark.skipif(
    not (current_platform.is_cuda() or current_platform.is_xpu()),
    reason="Requires CUDA or XPU",
)
def test_replay_on_off_preserves_rejection_sampling_and_rng(monkeypatch):
    device = torch.device(current_platform.device_type)
    num_reqs = 3
    num_speculative_steps = 3
    rows_per_request = num_speculative_steps + 1
    vocab_size = 8
    cu_num_logits_np = np.arange(0, 13, rows_per_request, dtype=np.int32)
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    input_batch = _make_input_batch(cu_num_logits_np, idx_mapping_np, device)
    processed_logits = torch.zeros((12, vocab_size), device=device)
    processed_logits[0, 0] = -float("inf")
    processed_logits[rows_per_request + 1, 1] = -float("inf")
    draft_logits = DraftLogitsCache(
        num_reqs,
        num_speculative_steps,
        vocab_size,
        dtype=torch.float32,
        fill=0.0,
        device=device,
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
        apply_sampling_params=lambda logits, *_, **_grammar: logits,
        trace_replay_state=None,
    )
    rejection_sampler = _make_rejection_sampler(sampler, num_speculative_steps)
    pos = torch.arange(12, dtype=torch.int32, device=device)

    pack_calls: list[tuple[list[int], int, int]] = []
    # A classmethod looked up on its class is a bound method; keep the plain
    # function so the patched classmethod can forward its own `cls`.
    bound_from_logits = SamplingMaskTensors.from_logits
    assert isinstance(bound_from_logits, MethodType)
    pack = bound_from_logits.__func__

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

    rng_before = torch.get_device_module(device).get_rng_state()
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
    rng_after_off = torch.get_device_module(device).get_rng_state()
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
    rng_after_on = torch.get_device_module(device).get_rng_state()

    assert pack_calls == [
        ([0, 4, 8], vocab_size, rows_per_request),
        ([0, 4], vocab_size, rows_per_request),
    ]
    assert off_masks is None and on_masks is not None
    assert on_masks.rows_per_request == rows_per_request
    assert on_masks.token_ids.shape == (num_reqs * rows_per_request, vocab_size)
    assert on_masks.packed_mask.shape[0] == num_reqs * rows_per_request
    assert off_counts.tolist() == [1, 2, 4]
    assert torch.equal(off_counts, on_counts)
    counts = off_counts.cpu().tolist()
    for req_idx, count in enumerate(counts):
        assert torch.equal(off_sampled[req_idx, :count], on_sampled[req_idx, :count])

    mapped_rows = [
        int(input_batch.cu_num_logits_np[req_idx]) + slot_idx
        for req_idx, count in enumerate(counts)
        for slot_idx in range(count)
    ]
    assert off_logprobs is not None and on_logprobs is not None
    assert torch.equal(
        off_logprobs.logprob_token_ids[mapped_rows],
        on_logprobs.logprob_token_ids[mapped_rows],
    )
    assert torch.equal(
        off_logprobs.logprobs[mapped_rows], on_logprobs.logprobs[mapped_rows]
    )
    assert torch.equal(
        off_logprobs.selected_token_ranks[mapped_rows],
        on_logprobs.selected_token_ranks[mapped_rows],
    )
    assert torch.equal(rng_before, rng_after_off)
    assert torch.equal(rng_after_off, rng_after_on)

    masks = (
        on_masks.to_cpu_nonblocking().tolists(on_counts.cpu().numpy()).to_nested_list()
    )
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
