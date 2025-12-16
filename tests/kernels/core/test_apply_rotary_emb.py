# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for ApplyRotaryEmb CustomOp dispatch behavior.

This test ensures that RotaryEmbedding classes correctly call the appropriate
ApplyRotaryEmb methods based on the calling context:

1. RotaryEmbedding.forward_native() -> ApplyRotaryEmb.forward_native()
2. RotaryEmbedding.forward_cuda() -> ApplyRotaryEmb.forward() (auto-dispatch)
3. RotaryEmbedding.forward_hip() -> ApplyRotaryEmb.forward() (auto-dispatch)
"""

from dataclasses import dataclass

import pytest
import torch

from vllm.config import (
    CompilationConfig,
    VllmConfig,
    get_cached_compilation_config,
    set_current_vllm_config,
)
from vllm.platforms import current_platform

CUDA_DEVICES = ["cuda:0"]


@dataclass
class RotaryEmbeddingTestCase:
    """Test case configuration for RotaryEmbedding dispatch tests."""

    name: str
    rope_class: type
    rope_kwargs: dict
    method_name: str  # forward_native, forward_cuda, forward
    positions_shape: tuple  # (num_tokens,) or (3, num_tokens) or (4, num_tokens)
    expect_forward_native: bool  # Should call ApplyRotaryEmb.forward_native()
    expect_forward: bool  # Should call ApplyRotaryEmb.forward()


def get_test_cases() -> list[RotaryEmbeddingTestCase]:
    """Generate test cases for all RotaryEmbedding classes."""
    from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
        Ernie4_5_VLRotaryEmbedding,
    )
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    from vllm.model_executor.layers.rotary_embedding.xdrope import XDRotaryEmbedding

    common_kwargs = {
        "head_size": 128,
        "rotary_dim": 128,
        "max_position_embeddings": 4096,
        "base": 10000,
        "is_neox_style": True,
        "dtype": torch.bfloat16,
    }

    return [
        # MRotaryEmbedding tests
        RotaryEmbeddingTestCase(
            name="MRotaryEmbedding.forward_native",
            rope_class=MRotaryEmbedding,
            rope_kwargs={**common_kwargs, "mrope_section": [16, 24, 24]},
            method_name="forward_native",
            positions_shape=(3, 32),  # 2D for multimodal
            expect_forward_native=True,
            expect_forward=False,
        ),
        RotaryEmbeddingTestCase(
            name="MRotaryEmbedding.forward_cuda_1d",
            rope_class=MRotaryEmbedding,
            rope_kwargs={**common_kwargs, "mrope_section": [16, 24, 24]},
            method_name="forward_cuda",
            positions_shape=(32,),  # 1D triggers apply_rotary_emb path
            expect_forward_native=False,
            expect_forward=True,
        ),
        # XDRotaryEmbedding tests
        RotaryEmbeddingTestCase(
            name="XDRotaryEmbedding.forward",
            rope_class=XDRotaryEmbedding,
            rope_kwargs={
                **common_kwargs,
                "scaling_alpha": 1.0,
                "xdrope_section": [16, 16, 16, 16],
            },
            method_name="forward",
            positions_shape=(4, 32),  # 4D for P/W/H/T
            expect_forward_native=False,
            expect_forward=True,
        ),
        # Ernie4_5_VLRotaryEmbedding tests
        RotaryEmbeddingTestCase(
            name="Ernie4_5_VLRotaryEmbedding.forward_native",
            rope_class=Ernie4_5_VLRotaryEmbedding,
            rope_kwargs={**common_kwargs, "mrope_section": [22, 22, 20]},
            method_name="forward_native",
            positions_shape=(3, 32),  # 2D for multimodal
            expect_forward_native=True,
            expect_forward=False,
        ),
    ]


def run_dispatch_test(
    test_case: RotaryEmbeddingTestCase,
    device: str,
):
    """Run a dispatch test for a RotaryEmbedding class."""
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all", "+apply_rotary_emb"])
    )
    get_cached_compilation_config.cache_clear()

    with set_current_vllm_config(vllm_config):
        rope = test_case.rope_class(**test_case.rope_kwargs).to(device=device)

        apply_rotary_emb = rope.apply_rotary_emb

        # Verify custom op is enabled
        if test_case.expect_forward_native:
            assert (
                apply_rotary_emb._forward_method != apply_rotary_emb.forward_native
            ), "Test setup error: ApplyRotaryEmb custom op should be enabled"

        # Setup call tracking
        call_tracker = {"forward_native_called": False, "forward_called": False}
        original_forward_native = apply_rotary_emb.forward_native
        original_forward = apply_rotary_emb.forward

        def tracked_forward_native(*args, **kwargs):
            call_tracker["forward_native_called"] = True
            return original_forward_native(*args, **kwargs)

        def tracked_forward(*args, **kwargs):
            call_tracker["forward_called"] = True
            return original_forward(*args, **kwargs)

        apply_rotary_emb.forward_native = tracked_forward_native
        apply_rotary_emb.forward = tracked_forward

        try:
            num_tokens = test_case.positions_shape[-1]
            num_q_heads = 8
            num_kv_heads = 2
            head_size = test_case.rope_kwargs["head_size"]
            max_position = test_case.rope_kwargs["max_position_embeddings"]

            positions = torch.randint(
                0, max_position // 4, test_case.positions_shape, device=device
            )
            query = torch.randn(
                num_tokens, num_q_heads * head_size, dtype=torch.bfloat16, device=device
            )
            key = torch.randn(
                num_tokens,
                num_kv_heads * head_size,
                dtype=torch.bfloat16,
                device=device,
            )

            # Call the method under test
            method = getattr(rope, test_case.method_name)
            method(positions, query.clone(), key.clone())

            # Verify expectations
            if test_case.expect_forward_native:
                assert call_tracker["forward_native_called"], (
                    f"{test_case.name} should call ApplyRotaryEmb.forward_native()"
                )
            if not test_case.expect_forward:
                assert not call_tracker["forward_called"], (
                    f"{test_case.name} should NOT call ApplyRotaryEmb.forward(). "
                    "Bug: when +apply_rotary_emb is enabled, forward_native() "
                    "incorrectly dispatches to CUDA/HIP kernels."
                )
            if test_case.expect_forward:
                assert call_tracker["forward_called"], (
                    f"{test_case.name} should call ApplyRotaryEmb.forward()"
                )
        finally:
            apply_rotary_emb.forward_native = original_forward_native
            apply_rotary_emb.forward = original_forward


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skipping CUDA/ROCm only tests."
)
@pytest.mark.parametrize("test_case", get_test_cases(), ids=lambda tc: tc.name)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_rotary_embedding_dispatch(
    test_case: RotaryEmbeddingTestCase,
    device: str,
):
    """
    Test that RotaryEmbedding classes dispatch to the correct ApplyRotaryEmb method.

    - forward_native methods should call ApplyRotaryEmb.forward_native()
    - forward_cuda/forward methods should call ApplyRotaryEmb.forward()
    """
    run_dispatch_test(test_case, device)
