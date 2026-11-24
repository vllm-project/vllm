# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft early exit: the stopping decision of the draft WHILE node, and how
verification and the acceptance estimator treat the draft slots it skipped."""

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    _draft_early_exit_kernel,
    rejections_of_drafted,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled


def keep_drafting(
    confidences: np.ndarray,
    is_real: np.ndarray,
    step: int,
    threshold: float,
    budget: int,
    num_steps: int,
) -> tuple[bool, float]:
    """Whether to draft position `step`, with positions 0..step-1 drafted: one is
    left, the budget is above 0, and the sum over real rows of max(0, predicted
    survival at `step` - edge) reaches the threshold. The edge is the budget-th
    highest survival among the drafted positions, 0 while there are fewer."""
    survival = np.cumprod(confidences[is_real, :step].astype(np.float64), axis=1)
    predicted = survival[:, step - 1] * confidences[is_real, step - 1]
    drafted = np.sort(survival.ravel())[::-1]
    edge = drafted[budget - 1] if 0 < budget <= len(drafted) else 0.0
    gain = float(np.maximum(predicted - edge, 0.0).sum())
    return step < num_steps and budget > 0 and gain >= threshold, gain


def decide(
    confidences: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_reqs: int,
    current_step: torch.Tensor,
    threshold: float,
    budget: int,
    advance: bool,
) -> bool:
    keep = torch.ones((), dtype=torch.bool, device=confidences.device)
    params = torch.tensor(
        [threshold, budget], dtype=torch.float32, device=confidences.device
    )
    num_steps = confidences.shape[1]
    _draft_early_exit_kernel[(1,)](
        keep,
        current_step,
        params,
        confidences,
        confidences.stride(0),
        idx_mapping,
        num_reqs,
        num_steps,
        ADVANCE=advance,
        BLOCK_R=1 << max(num_reqs - 1, 0).bit_length(),
        BLOCK_K=1 << (num_steps - 1).bit_length(),
    )
    return bool(keep)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_steps", [3, 7])
@pytest.mark.parametrize("seed", range(4))
def test_draft_early_exit_kernel_matches_reference(num_steps, seed):
    """Single decisions and whole loops, as the WHILE node runs them. Padding rows
    (idx_mapping < 0) hold NaN and must not count; some confidences tie."""
    rng = np.random.default_rng(seed)
    num_reqs = int(rng.integers(1, 40))
    confidences = rng.random((num_reqs, num_steps)) ** rng.choice([0.25, 1.0, 3.0])
    if seed % 2:
        confidences = np.round(confidences * 4) / 4
    is_real = rng.random(num_reqs) < 0.8
    confidences[~is_real] = np.nan
    confidences = confidences.astype(np.float32)
    conf_gpu = torch.from_numpy(confidences).cuda()
    idx_mapping = torch.from_numpy(np.where(is_real, np.arange(num_reqs), -1)).cuda()
    num_real = int(is_real.sum())

    def near(gain: float, threshold: float) -> bool:
        # float32 kernel vs float64 reference
        return abs(gain - threshold) <= 1e-4 * max(1.0, threshold)

    for step in range(1, num_steps + 1):
        for budget in {0, 1, num_real * step, num_real * step + 1, 1 << 20}:
            _, gain = keep_drafting(confidences, is_real, step, 0.0, budget, num_steps)
            for threshold in (0.0, 0.5 * gain, 2.0 * gain, gain + 1.0):
                if near(gain, threshold) and threshold > 0:
                    continue
                expected, _ = keep_drafting(
                    confidences, is_real, step, threshold, budget, num_steps
                )
                current = torch.tensor(step, device="cuda")
                assert (
                    decide(
                        conf_gpu,
                        idx_mapping,
                        num_reqs,
                        current,
                        threshold,
                        budget,
                        False,
                    )
                    == expected
                )

    for budget in (0, 1, num_real, 1 << 20):
        for threshold in (0.0, 0.05, 0.3, 1.0):
            gains = [
                keep_drafting(confidences, is_real, s, 0.0, budget, num_steps)[1]
                for s in range(1, num_steps)
            ]
            if any(near(gain, threshold) for gain in gains):
                continue
            expected = 1
            while keep_drafting(
                confidences, is_real, expected, threshold, budget, num_steps
            )[0]:
                expected += 1
            current = torch.tensor(1, device="cuda")
            keep = decide(
                conf_gpu, idx_mapping, num_reqs, current, threshold, budget, False
            )
            while keep:
                keep = decide(
                    conf_gpu, idx_mapping, num_reqs, current, threshold, budget, True
                )
            assert int(current) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gather_draft_sampled_marks_slots_past_draft_counts():
    """Draft i sits at local position i + 1; past a request's draft count, its
    draft slots become -1 placeholders, which verification rejects."""
    rows_per_req = [4, 4, 2]
    num_logits = sum(rows_per_req)
    input_ids = torch.arange(100, 100 + num_logits, device="cuda")
    positions = torch.arange(50, 50 + num_logits, device="cuda")
    expanded_idx_mapping = torch.tensor(
        [r for r, n in enumerate(rows_per_req) for _ in range(n)], device="cuda"
    )
    expanded_local_pos = torch.tensor(
        [j for n in rows_per_req for j in range(n)],
        dtype=torch.int32,
        device="cuda",
    )
    prefill_len = torch.ones(3, dtype=torch.int32, device="cuda")
    num_drafts = torch.tensor([3, 1, 0], dtype=torch.int32, device="cuda")

    draft_sampled, pos = gather_draft_sampled(
        input_ids,
        positions,
        torch.arange(num_logits, device="cuda"),
        expanded_idx_mapping,
        expanded_local_pos,
        prefill_len,
        num_drafts,
    )
    assert draft_sampled.tolist() == [100, 101, 102, 103, 104, 105, -1, -1, 108, -1]
    assert torch.equal(pos, positions)
    unmarked, _ = gather_draft_sampled(
        input_ids,
        positions,
        torch.arange(num_logits, device="cuda"),
        expanded_idx_mapping,
        expanded_local_pos,
        prefill_len,
    )
    assert torch.equal(unmarked, input_ids)


def test_rejections_of_drafted_grades_only_drafted_positions():
    """Verification accepts no placeholder, so it accepts at most min(verified,
    drafted) drafts; the estimator sees the rest of the drafted ones as rejected."""
    verified = torch.tensor([3, 3, 3, 0, 2])
    drafted = torch.tensor([3, 1, 0, 2, 3], dtype=torch.int32)
    accepted = torch.tensor([1, 1, 0, 0, 2])
    num_bonus_tokens = 1
    cu_num_logits = torch.cat(
        [torch.zeros(1, dtype=torch.long), (verified + num_bonus_tokens).cumsum(0)]
    )
    num_rejected = rejections_of_drafted(
        verified - accepted, cu_num_logits, num_bonus_tokens, drafted
    )
    assert num_rejected.tolist() == [2, 0, 0, 0, 0]
