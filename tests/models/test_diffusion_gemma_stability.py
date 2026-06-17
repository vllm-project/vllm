# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DiffusionGemma stability/convergence convention.

The checkpoint's ``stability_threshold`` is the HF convention: ``k`` means stop
once the argmax canvas repeats ``k`` times (``k + 1`` identical canvases).
``_compiled_sample_step`` uses a sliding window of ``ST`` canvases, so
``DiffusionGemmaRequestStates`` derives ``stability_window = stability_threshold
+ 1``; without it, ``stability_threshold == 1`` commits one denoise step early
and drops the first answer token on short, high-confidence prompts.
"""

import torch

from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaRequestStates,
    _compiled_sample_step,
)

VOCAB = 16
CL = 4
HIDDEN = 2


def test_stability_window_matches_hf_convention():
    """``stability_window`` is ``stability_threshold + 1``, sizes the buffer."""
    states = DiffusionGemmaRequestStates(
        max_num_reqs=2,
        canvas_length=CL,
        vocab_size=VOCAB,
        max_denoising_steps=48,
        device=torch.device("cpu"),
        hidden_size=HIDDEN,
        stability_threshold=1,
    )
    assert states.stability_window == 2
    # The history buffer is sized by the window, not the raw threshold.
    assert states.accepted_canvas_history.shape[1] == 2


def _first_commit_step(argmax_per_step, st):
    """Run denoise steps for one request; return the step it commits on.

    Drives ``_compiled_sample_step`` with greedy, always-confident logits so
    convergence is governed purely by the stability window ``st``.
    ``argmax_per_step[i]`` is the desired argmax canvas at denoise step ``i``.
    """
    canvas = torch.zeros(1, CL, dtype=torch.int64)
    argmax_canvas = torch.zeros(1, CL, dtype=torch.int64)
    step_tensor = torch.zeros(1, dtype=torch.int32)
    is_encoder_phase = torch.zeros(1, dtype=torch.bool)  # False -> denoise
    confident = torch.zeros(1, dtype=torch.bool)
    sc_embeds = torch.zeros(1, CL, HIDDEN)
    embed_weight = torch.zeros(VOCAB, HIDDEN)
    normalizer = torch.tensor(1.0)
    history = torch.zeros(1, st, CL, dtype=torch.int64)
    history_len = torch.zeros(1, dtype=torch.int32)
    sampled = torch.zeros(1, CL, dtype=torch.int64)
    num_sampled = torch.zeros(1, dtype=torch.int64)
    draft = torch.zeros(1, CL, dtype=torch.int64)
    decode_slots = torch.tensor([0])
    decode_idx = torch.tensor([0])
    all_slots = torch.tensor([0])
    valid_canvas_len = torch.tensor([CL])

    for step, desired in enumerate(argmax_per_step, start=1):
        is_encoder_phase[0] = False  # force a denoise step
        logits = torch.full((CL, VOCAB), -30.0)
        for pos, tok in enumerate(desired):
            logits[pos, tok] = 30.0
        _compiled_sample_step(
            logits,
            decode_slots,
            decode_idx,
            all_slots,
            valid_canvas_len,
            canvas,
            argmax_canvas,
            step_tensor,
            is_encoder_phase,
            confident,
            sc_embeds,
            embed_weight,
            normalizer,
            history,
            history_len,
            sampled,
            num_sampled,
            draft,
            max_denoising_steps=100.0,
            t_min=0.0,
            t_max=0.0,
            confidence_threshold=1e9,  # always "confident"
            vocab_size=VOCAB,
            CL=CL,
            ST=st,
            entropy_bound=1e9,  # accept every position
        )
        # A converged denoise step flips ``is_encoder_phase`` to True (commit next).
        if bool(is_encoder_phase[0]):
            return step
    return None


def test_does_not_commit_on_first_noisy_step():
    """A still-changing canvas must not converge on the first denoise step.

    Position 0 is noisy at step 1 (token 1) then settles to token 8. With the
    HF-equivalent window (``stability_threshold + 1 == 2``) the request must wait
    for the argmax to repeat (step 3), matching the reference; the buggy window
    of 1 would commit the noisy step-1 canvas.
    """
    seq = [[1, 5, 5, 5], [8, 5, 5, 5], [8, 5, 5, 5], [8, 5, 5, 5]]
    states = DiffusionGemmaRequestStates(
        max_num_reqs=1,
        canvas_length=CL,
        vocab_size=VOCAB,
        max_denoising_steps=100,
        device=torch.device("cpu"),
        hidden_size=HIDDEN,
        stability_threshold=1,
    )
    # The fixed window commits on the first repeated argmax (step 3), not step 1.
    assert _first_commit_step(seq, states.stability_window) == 3
    # Sanity: the raw (off-by-one) window would have committed immediately.
    assert _first_commit_step(seq, states.stability_threshold) == 1


def test_already_stable_canvas_needs_one_confirmation():
    """An already-stable canvas commits once the argmax has repeated (step 2)."""
    seq = [[8, 5, 5, 5]] * 4
    assert _first_commit_step(seq, 2) == 2  # window = stability_threshold + 1
