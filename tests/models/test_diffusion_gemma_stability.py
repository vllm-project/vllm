# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DiffusionGemma stability-window convention.

The checkpoint's ``stability_threshold`` follows the HF convention: ``k`` means
the denoiser's argmax canvas must repeat ``k`` times, i.e. ``k + 1`` consecutive
identical canvases. ``_compiled_sample_step`` instead checks a sliding window of
``ST`` canvases, so ``DiffusionGemmaRequestStates`` derives ``stability_window =
stability_threshold + 1``. Without it, ``stability_threshold == 1`` makes the
stability gate vacuous and commits one denoise step early.
"""

import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import DiffusionGemmaRequestStates


@pytest.mark.cpu_test
@pytest.mark.parametrize("stability_threshold", [0, 1, 2])
def test_stability_window_matches_hf_convention(stability_threshold):
    states = DiffusionGemmaRequestStates(
        max_num_reqs=2,
        canvas_length=4,
        vocab_size=16,
        max_denoising_steps=48,
        device=torch.device("cpu"),
        hidden_size=2,
        stability_threshold=stability_threshold,
    )
    # The compiled stability check requires ``stability_threshold + 1`` identical
    # canvases, and the history buffer is sized to match.
    assert states.stability_window == stability_threshold + 1
    assert states.accepted_canvas_history.shape[1] == stability_threshold + 1
