# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    _iter_request_chunks,
)


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
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), 64, dtype=torch.int32),
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
        SimpleNamespace(
            logprobs_mode=logprobs_mode,
            return_sampling_mask=False,
            logprob_token_ids_state=None,
        ),
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
            logprobs_dims=(2, 0),
        )
    )
    score_logits = logits + 1 if logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
    full_logprobs = rejection_sampler._get_logprobs_tensors(
        sampled,
        num_sampled,
        score_logits,
        input_batch.cu_num_logits,
        input_batch.cu_num_logits_np,
        input_batch.expanded_idx_mapping,
        logprobs_dims=(2, 0),
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
@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "raw_logits"])
def test_logprob_token_ids_without_topk_logprobs(logprobs_mode: str):
    """Requests that only set `logprob_token_ids` (no `logprobs`) still get the
    scores of those tokens when their batch is verified with speculative drafts."""
    device = torch.device("cuda")
    vocab_size = 17
    cu_num_logits_np = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    idx_mapping_np = np.array([3, 0, 2, 1], dtype=np.int32)
    input_batch = _make_input_batch(cu_num_logits_np, idx_mapping_np, device)
    wanted = {3: [5, 11], 0: [2], 2: [], 1: [16, 0, 7]}  # request state index -> ids
    token_ids_state = LogprobTokenIdsState(4, device)
    for req_idx, ids in wanted.items():
        token_ids_state.add_request(req_idx, SamplingParams(logprob_token_ids=ids))
    token_ids_state.apply_staged_writes()
    max_token_ids = token_ids_state.max_num_token_ids(idx_mapping_np)
    assert max_token_ids == 3
    rejection_sampler = _make_rejection_sampler(
        SimpleNamespace(
            logprobs_mode=logprobs_mode,
            return_sampling_mask=False,
            logprob_token_ids_state=token_ids_state,
        ),
        num_speculative_steps=3,
    )

    def fake_verify(
        self, logits, _draft_logits, _draft_sampled, _pos, cu_num_logits, *_
    ):
        num_sampled = torch.diff(cu_num_logits).to(torch.int32)
        sampled = (
            torch.arange(4, device=device).repeat(len(num_sampled), 1) % vocab_size
        )
        return logits, sampled, num_sampled

    rejection_sampler._verify = MethodType(fake_verify, rejection_sampler)
    torch.manual_seed(0)
    logits = torch.randn(10, vocab_size, device=device)

    _, _, chunked, _ = rejection_sampler._verify_in_chunks(
        logits,
        input_batch,
        draft_logits=None,
        draft_sampled=torch.arange(10, device=device),
        pos=torch.arange(10, device=device),
        max_chunk_logits=5,
        logprobs_dims=(0, max_token_ids),
    )
    assert chunked is not None
    assert chunked.logprobs.shape == (10, 1 + max_token_ids)

    expected = logits if logprobs_mode == "raw_logits" else logits.log_softmax(-1)
    for req, req_idx in enumerate(idx_mapping_np):
        ids = wanted[int(req_idx)]
        for row in range(cu_num_logits_np[req], cu_num_logits_np[req + 1]):
            assert chunked.logprob_token_ids[row, 1 : 1 + len(ids)].tolist() == ids
            torch.testing.assert_close(
                chunked.logprobs[row, 1 : 1 + len(ids)], expected[row, ids]
            )
            assert torch.isinf(chunked.logprobs[row, 1 + len(ids) :]).all()

    # Without any requested token ids nothing is computed.
    assert (
        rejection_sampler._get_logprobs_tensors(
            torch.zeros(4, 4, dtype=torch.int64, device=device),
            torch.ones(4, dtype=torch.int32, device=device),
            logits[:4],
            input_batch.cu_num_logits[:5],
            input_batch.cu_num_logits_np[:5],
            input_batch.expanded_idx_mapping[:4],
            None,
        )
        is None
    )


def _make_sampler_with_token_ids(
    logprobs_mode: str,
    wanted: dict[int, list[int]],
    num_logprobs: int,
    device: torch.device,
) -> SimpleNamespace:
    """Sampler stand-in with the real `Sampler.get_logprobs_dims`: every request
    state index in `wanted` asks for `logprob_token_ids`, and the batch-wide top-k
    `logprobs` is `num_logprobs` (NO_LOGPROBS when unset)."""
    token_ids_state = LogprobTokenIdsState(4, device)
    for req_idx, ids in wanted.items():
        token_ids_state.add_request(req_idx, SamplingParams(logprob_token_ids=ids))
    token_ids_state.apply_staged_writes()
    sampler = SimpleNamespace(
        logprobs_mode=logprobs_mode,
        return_sampling_mask=False,
        logprob_token_ids_state=token_ids_state,
        sampling_states=SimpleNamespace(max_num_logprobs=lambda _idx: num_logprobs),
    )
    sampler.get_logprobs_dims = MethodType(Sampler.get_logprobs_dims, sampler)
    return sampler


def _fake_verify(device: torch.device, vocab_size: int):
    def fake_verify(
        self, logits, _draft_logits, _draft_sampled, _pos, cu_num_logits, *_
    ):
        num_sampled = torch.diff(cu_num_logits).to(torch.int32)
        sampled = (
            torch.arange(4, device=device).repeat(len(num_sampled), 1) % vocab_size
        )
        return logits, sampled, num_sampled

    return fake_verify


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_get_logprobs_dims_with_only_token_ids():
    """`logprob_token_ids` alone is enough to ask for logprobs, whatever the
    batch's draft tokens (the sharded gather sizes its buffers from these dims)."""
    device = torch.device("cuda")
    idx_mapping_np = np.array([3, 0, 2, 1], dtype=np.int32)
    wanted = {3: [5, 11], 0: [2], 2: [], 1: [16, 0, 7]}
    sampler = _make_sampler_with_token_ids("raw_logprobs", wanted, NO_LOGPROBS, device)
    assert sampler.get_logprobs_dims(idx_mapping_np) == (0, 3)
    sampler = _make_sampler_with_token_ids("raw_logprobs", wanted, 1, device)
    assert sampler.get_logprobs_dims(idx_mapping_np) == (1, 3)
    sampler = _make_sampler_with_token_ids("raw_logprobs", {}, NO_LOGPROBS, device)
    assert sampler.get_logprobs_dims(idx_mapping_np) is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "raw_logits"])
def test_logprob_token_ids_with_topk_logprobs(logprobs_mode: str):
    """Top-k `logprobs` narrower than the widest `logprob_token_ids`: the rows are
    as wide as the token ids, requests with token ids get exactly those tokens and
    the others get the top-k."""
    device = torch.device("cuda")
    vocab_size = 17
    cu_num_logits_np = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    idx_mapping_np = np.array([3, 0, 2, 1], dtype=np.int32)
    input_batch = _make_input_batch(cu_num_logits_np, idx_mapping_np, device)
    wanted = {3: [5, 11], 0: [], 2: [], 1: [16, 0, 7, 9]}
    sampler = _make_sampler_with_token_ids(logprobs_mode, wanted, 1, device)
    dims = sampler.get_logprobs_dims(idx_mapping_np)
    assert dims == (1, 4)
    rejection_sampler = _make_rejection_sampler(sampler, num_speculative_steps=3)
    rejection_sampler._verify = MethodType(
        _fake_verify(device, vocab_size), rejection_sampler
    )
    torch.manual_seed(0)
    logits = torch.randn(10, vocab_size, device=device)

    _, _, lp, _ = rejection_sampler._verify_in_chunks(
        logits,
        input_batch,
        draft_logits=None,
        draft_sampled=torch.arange(10, device=device),
        pos=torch.arange(10, device=device),
        max_chunk_logits=5,
        logprobs_dims=dims,
    )
    assert lp is not None
    assert lp.logprobs.shape == (10, 1 + 4)

    expected = logits if logprobs_mode == "raw_logits" else logits.log_softmax(-1)
    top1 = logits.argmax(-1)
    for req, req_idx in enumerate(idx_mapping_np):
        ids = wanted[int(req_idx)] or None
        for row in range(cu_num_logits_np[req], cu_num_logits_np[req + 1]):
            if ids is None:
                assert lp.logprob_token_ids[row, 1] == top1[row]
                torch.testing.assert_close(
                    lp.logprobs[row, 1], expected[row, top1[row]]
                )
                assert torch.isinf(lp.logprobs[row, 2:]).all()
            else:
                assert lp.logprob_token_ids[row, 1 : 1 + len(ids)].tolist() == ids
                torch.testing.assert_close(
                    lp.logprobs[row, 1 : 1 + len(ids)], expected[row, ids]
                )
                assert torch.isinf(lp.logprobs[row, 1 + len(ids) :]).all()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_logprob_token_ids_with_adaptive_verification():
    """Adaptive verification verifies one chunk over compacted logits, with
    `cu_num_logits` on device and `cu_num_logits_np` an upper bound: the scores of
    `logprob_token_ids` must match the fixed-boundary layout."""
    device = torch.device("cuda")
    vocab_size = 17
    true_cu = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    idx_mapping_np = np.array([3, 0, 2, 1], dtype=np.int32)
    wanted = {3: [5, 11], 0: [2], 2: [], 1: [16, 0, 7]}
    sampler = _make_sampler_with_token_ids("raw_logprobs", wanted, -1, device)
    dims = sampler.get_logprobs_dims(idx_mapping_np)
    torch.manual_seed(0)
    logits = torch.randn(10, vocab_size, device=device)

    def run(adaptive: bool):
        input_batch = _make_input_batch(true_cu, idx_mapping_np, device)
        if adaptive:
            # Pre-compaction layout on the host (4 rows per request); the device
            # tensors hold the compacted one.
            input_batch.cu_num_logits_np = np.arange(0, 17, 4, dtype=np.int32)
            input_batch.expanded_idx_mapping = torch.cat(
                [
                    input_batch.expanded_idx_mapping,
                    torch.zeros(6, dtype=torch.int64, device=device),
                ]
            )
        rejection_sampler = _make_rejection_sampler(sampler, num_speculative_steps=3)
        rejection_sampler.enable_adaptive_verification = adaptive
        rejection_sampler._verify = MethodType(
            _fake_verify(device, vocab_size), rejection_sampler
        )
        return rejection_sampler._verify_in_chunks(
            logits,
            input_batch,
            draft_logits=None,
            draft_sampled=torch.arange(10, device=device),
            pos=torch.arange(10, device=device),
            max_chunk_logits=16,
            logprobs_dims=dims,
        )[2]

    fixed, adaptive = run(False), run(True)
    assert fixed is not None and adaptive is not None
    assert torch.equal(fixed.logprob_token_ids, adaptive.logprob_token_ids)
    assert torch.equal(fixed.logprobs, adaptive.logprobs)
    assert fixed.logprobs.shape == (10, 1 + 3)


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
        logprob_token_ids_state=None,
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
            logprobs_dims=(0, 0),
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
        logprobs_dims=(0, 0),
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
