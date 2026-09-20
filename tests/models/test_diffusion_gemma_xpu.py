# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def test_xpu_diffusion_compile_is_identity(monkeypatch):
    from vllm.model_executor.models import diffusion_gemma

    monkeypatch.setattr(diffusion_gemma.current_platform, "is_xpu", lambda: True)
    fn = lambda value: value  # noqa: E731

    assert diffusion_gemma._compile_diffusion_fn(fn) is fn


def test_self_conditioning_buffer_uses_model_dtype():
    from vllm.model_executor.models.diffusion_gemma import (
        DiffusionGemmaRequestStates,
    )

    states = DiffusionGemmaRequestStates(
        max_num_reqs=2,
        canvas_length=4,
        vocab_size=8,
        max_denoising_steps=3,
        device=torch.device("cpu"),
        hidden_size=16,
        hidden_dtype=torch.bfloat16,
        stability_threshold=2,
    )

    assert states.self_conditioning_embeds.dtype == torch.bfloat16
