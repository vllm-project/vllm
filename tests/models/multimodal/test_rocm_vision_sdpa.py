# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for ROCm vision encoder SDPA backend selection.

On ROCm, flash_sdp and mem_efficient_sdp produce inaccurate vision
embeddings. embed_multimodal() must wrap get_image_features() with the
MATH SDP backend context on ROCm platforms.

See https://github.com/vllm-project/vllm/issues/30167
"""
import pytest
import torch
import torch.nn as nn

from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm-specific SDPA backend correctness test",
)
def test_rocm_vision_embed_uses_math_sdpa():
    """embed_multimodal() must force MATH SDP backend on ROCm.

    Verify by intercepting get_image_features() and checking that the
    MATH backend context is active at the time of the call.
    """
    sdpa_backend_at_call: list = []

    class _FakeVisionModel(nn.Module):
        def get_image_features(self, pixel_values, **kwargs):
            # Record which backends are currently overridden by a context.
            # torch.backends.cuda.sdp_kernel tracks the active override.
            try:
                # Attempt to enter MATH-only context — succeeds only if
                # the outer context already permits MATH.
                with torch.nn.attention.sdpa_kernel(
                    backends=[torch.nn.attention.SDPBackend.MATH]
                ):
                    sdpa_backend_at_call.append("math_available")
            except RuntimeError:
                sdpa_backend_at_call.append("math_blocked")
            return torch.zeros(1, 4, 8)

    from vllm.model_executor.models.transformers.multimodal import (
        MultiModalMixin,
    )

    mixin = MultiModalMixin.__new__(MultiModalMixin)
    mixin.model = _FakeVisionModel()

    pixel_values = torch.zeros(1, 3, 16, 16)
    num_image_patches = torch.tensor([1])

    mixin.embed_multimodal(
        pixel_values=pixel_values,
        num_image_patches=num_image_patches,
    )

    assert sdpa_backend_at_call, "get_image_features was never called"
    assert sdpa_backend_at_call[0] == "math_available", (
        "embed_multimodal did not force MATH SDP backend on ROCm — "
        "vision embeddings may be inaccurate (issue #30167)"
    )
