# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for FusedMMInputNorm and the fused affine transform kernel."""

import pytest
import torch

from vllm.model_executor.layers.fusion.mm_input_norm import (
    FusedMMInputNorm,
    IdentityInputNorm,
    NormParams,
    fused_mm_input_norm_triton,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import set_random_seed

# Module-level tests run on any platform: the CPU platform exercises
# forward_native, accelerators exercise the fused kernel. Only the raw
# kernel tests require an accelerator.
_DEVICE = torch.device(current_platform.device_type)

requires_accelerator = pytest.mark.skipif(
    current_platform.is_cpu(),
    reason="fused Triton kernel requires a CUDA/XPU accelerator",
)
requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="requires Triton")
requires_vllm_config = pytest.mark.usefixtures("default_vllm_config")


def _reference_input_norm(
    pixel_values: torch.Tensor,
    image_mean: list[float],
    image_std: list[float],
    rescale_factor: float,
    channel: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Straightforward per-channel affine: (x * rescale - mean) / std."""
    patches, size = pixel_values.shape
    patch_size = size // channel
    mean = torch.tensor(
        image_mean, dtype=torch.float32, device=pixel_values.device
    ).view(1, channel, 1)
    std = torch.tensor(image_std, dtype=torch.float32, device=pixel_values.device).view(
        1, channel, 1
    )
    x = pixel_values.to(torch.float32).reshape(patches, channel, patch_size)
    x = (x * rescale_factor - mean) / std
    return x.view(patches, size).to(out_dtype)


# CLIP-style RGB normalization constants reused across tests.
_RGB_MEAN = [0.48145466, 0.4578275, 0.40821073]
_RGB_STD = [0.26862954, 0.26130258, 0.27577711]
_RGB_RESCALE = 1.0 / 255.0


def _check_against_reference(
    pixel_values: torch.Tensor,
    out_dtype: torch.dtype,
    channel: int = 3,
    image_mean: list[float] | None = None,
    image_std: list[float] | None = None,
    rescale_factor: float = _RGB_RESCALE,
    **close_kwargs,
):
    """Run ``FusedMMInputNorm`` on ``pixel_values`` and compare against the
    eager per-channel affine reference. Uses the RGB constants by default."""
    if image_mean is None:
        image_mean = list(_RGB_MEAN)
    if image_std is None:
        image_std = list(_RGB_STD)
    norm = FusedMMInputNorm(
        image_mean=image_mean,
        image_std=image_std,
        rescale_factor=rescale_factor,
        channel=channel,
    ).to(_DEVICE)
    expected = _reference_input_norm(
        pixel_values, image_mean, image_std, rescale_factor, channel, out_dtype
    )
    torch.testing.assert_close(
        norm(pixel_values, visual_dtype=out_dtype), expected, **close_kwargs
    )


# ===========================================================================
# Module-level behavior
# ===========================================================================
@requires_vllm_config
class TestFusedMMInputNormModule:
    def test_matches_reference_above_cudnn_grid_limit(self):
        """num_patches above the old cuDNN batch-norm grid limit (~65535)
        used to raise CUDNN_STATUS_INTERNAL_ERROR."""
        set_random_seed(0)
        pixel_values = torch.randint(
            0, 256, (70000, 3 * 196), dtype=torch.float32, device=_DEVICE
        )
        _check_against_reference(pixel_values, torch.float32)

    def test_identity_passthrough(self):
        norm = IdentityInputNorm()

        pixel_values = torch.randn(8, 3 * 196, dtype=torch.float32, device=_DEVICE)
        out = norm(pixel_values, visual_dtype=torch.bfloat16)
        torch.testing.assert_close(out, pixel_values.to(torch.bfloat16))


# ===========================================================================
# dtype coverage
# ===========================================================================
@requires_vllm_config
class TestFusedMMInputNormDtypes:
    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            (torch.float32, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.float32, torch.float16),
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
            (torch.float16, torch.float16),
            (torch.uint8, torch.bfloat16),
            (torch.uint8, torch.float32),
        ],
    )
    def test_dtype_combinations(self, in_dtype: torch.dtype, out_dtype: torch.dtype):
        set_random_seed(0)
        if in_dtype == torch.uint8:
            pixel_values = torch.randint(
                0, 256, (32, 3 * 16), dtype=torch.uint8, device=_DEVICE
            )
        else:
            pixel_values = torch.rand(
                (32, 3 * 16), dtype=torch.float32, device=_DEVICE
            ).to(in_dtype)

        _check_against_reference(
            pixel_values,
            out_dtype,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            atol=1e-2,
            rtol=1e-2,
        )


# ===========================================================================
# Shape / channel coverage
# ===========================================================================
@requires_vllm_config
class TestFusedMMInputNormShapes:
    @pytest.mark.parametrize("channel", [1, 3, 4])
    def test_channel_variants(self, channel: int):
        set_random_seed(0)
        pixel_values = torch.randint(
            0, 256, (16, channel * 64), dtype=torch.float32, device=_DEVICE
        )
        _check_against_reference(
            pixel_values,
            torch.float32,
            channel=channel,
            image_mean=[0.5] * channel,
            image_std=[0.25] * channel,
        )

    @pytest.mark.parametrize("patch_size", [1, 2048])
    def test_block_boundaries(self, patch_size: int):
        """Masked-tail (12 elements) and exact-multiple (24 blocks) paths."""
        set_random_seed(0)
        pixel_values = torch.randint(
            0, 256, (4, 3 * patch_size), dtype=torch.float32, device=_DEVICE
        )
        _check_against_reference(pixel_values, torch.float32)


# ===========================================================================
# Input handling: non-contiguous inputs
# ===========================================================================
@requires_vllm_config
class TestFusedMMInputNormInputHandling:
    def test_non_contiguous_input_matches_reference(self):
        set_random_seed(0)
        base = torch.randint(
            0, 256, (8, 3 * 32, 2), dtype=torch.float32, device=_DEVICE
        )
        non_contig = base[..., 0]
        assert not non_contig.is_contiguous()

        _check_against_reference(non_contig, torch.float32)


# ===========================================================================
# Raw kernel entry point
# ===========================================================================
@requires_accelerator
@requires_triton
class TestFusedMMInputNormKernel:
    @pytest.mark.parametrize(
        "N, C, L",
        [
            (1, 1, 1),
            (1, 3, 1),
            (3, 3, 7),  # tile straddles channel boundaries
            (2, 8, 63),
            (4, 3, 1000),
        ],
    )
    def test_channel_boundary_crossing(self, N: int, C: int, L: int):
        """The 1D kernel recovers ``c = (offs // L) % C`` per lane, so a tile
        can straddle a channel boundary and gather weight/bias per lane.
        These shapes force that path by keeping ``L`` well below the block."""
        set_random_seed(0)
        x = torch.randn(N, C, L, dtype=torch.float32, device=_DEVICE)
        w = torch.randn(C, dtype=torch.float32, device=_DEVICE)
        b = torch.randn(C, dtype=torch.float32, device=_DEVICE)

        out = torch.empty_like(x)
        fused_mm_input_norm_triton(x, out, w, b)

        expected = x * w.view(1, C, 1) + b.view(1, C, 1)
        torch.testing.assert_close(out, expected)

    def test_non_contiguous_inputs_materialized(self):
        """Read-only tensors are made contiguous internally."""
        N, C, L = 3, 3, 100
        set_random_seed(0)
        base = torch.randn(N, C, L, 2, dtype=torch.float32, device=_DEVICE)
        x = base[..., 0]
        assert not x.is_contiguous()
        w = torch.randn(C, dtype=torch.float32, device=_DEVICE)
        b = torch.randn(C, dtype=torch.float32, device=_DEVICE)

        out = torch.empty(N, C, L, dtype=torch.float32, device=_DEVICE)
        fused_mm_input_norm_triton(x, out, w, b)

        expected = x * w.view(1, C, 1) + b.view(1, C, 1)
        torch.testing.assert_close(out, expected)

    def test_rejects_mismatched_output_shape(self):
        """``outputs`` must be shaped exactly like ``inputs``."""
        N, C, L = 3, 3, 100
        x = torch.randn(N, C, L, dtype=torch.float32, device=_DEVICE)
        w = torch.randn(C, dtype=torch.float32, device=_DEVICE)
        b = torch.randn(C, dtype=torch.float32, device=_DEVICE)

        for out_shape in ((N, C + 1, L), (N, C, L + 1), (N + 2, C, L)):
            with pytest.raises(AssertionError):
                fused_mm_input_norm_triton(
                    x,
                    torch.empty(*out_shape, dtype=torch.float32, device=_DEVICE),
                    w,
                    b,
                )


# ===========================================================================
# Construction / configuration
# ===========================================================================
@requires_vllm_config
class TestFusedMMInputNormConstruction:
    """Weight/bias buffer semantics at init time."""

    def test_identity_config_buffers(self):
        """Numerically identity parameters still build plain weight/bias
        buffers; the identity shortcut lives in ``from_model_config``."""
        norm = FusedMMInputNorm(
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            rescale_factor=1.0,
        )
        torch.testing.assert_close(norm.weight, torch.ones(3))
        torch.testing.assert_close(norm.bias, torch.zeros(3))

    def test_non_identity_buffers(self):
        channel = 3
        norm = FusedMMInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        )
        assert norm.weight.shape == (channel,)
        assert norm.bias.shape == (channel,)

        mean = torch.tensor(_RGB_MEAN, dtype=torch.float32)
        std = torch.tensor(_RGB_STD, dtype=torch.float32)
        torch.testing.assert_close(norm.weight, _RGB_RESCALE / std)
        torch.testing.assert_close(norm.bias, -mean / std)

    def test_is_identity_params(self):
        """The numeric-identity check mirrors the old allclose detection."""
        assert NormParams([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0).is_identity
        assert not NormParams(_RGB_MEAN, _RGB_STD, _RGB_RESCALE).is_identity

    @pytest.mark.parametrize(
        ("image_mean", "image_std", "rescale_factor"),
        [
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0),
            ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25], 1 / 255),
        ],
    )
    def test_buffers_initialized_on_default_device(
        self,
        image_mean: list[float],
        image_std: list[float],
        rescale_factor: float,
    ):
        """Buffers are built on CPU first, then moved to the caller's
        default device, so init never touches the accelerator."""
        # The meta device gives the CPU-only test shard the same non-CPU
        # default-device semantics without requiring a CUDA build.
        default_device = "cuda" if torch.cuda.is_available() else "meta"
        with torch.device(default_device):
            norm = FusedMMInputNorm(image_mean, image_std, rescale_factor)

        assert norm.weight.device.type == default_device
        assert norm.bias.device.type == default_device
