# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for FusedMMInputNorm and the fused affine transform kernel."""

import pytest
import torch

from vllm.model_executor.layers.fusion.mm_input_norm import (
    FusedMMInputNorm,
    fused_mm_input_norm_triton,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import set_random_seed

# The fused Triton kernel cannot run on CPU tensors, so derive the test device
# from the active vLLM platform and skip accelerator-only tests on CPU.
_DEVICE_TYPE = current_platform.device_type
_DEVICE = torch.device(_DEVICE_TYPE)

requires_accelerator = pytest.mark.skipif(
    _DEVICE_TYPE == "cpu",
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
    x = pixel_values.to(torch.float32).view(patches, channel, patch_size)
    x = (x * rescale_factor - mean) / std
    return x.view(patches, size).to(out_dtype)


# CLIP-style RGB normalization constants reused across tests.
_RGB_MEAN = [0.48145466, 0.4578275, 0.40821073]
_RGB_STD = [0.26862954, 0.26130258, 0.27577711]
_RGB_RESCALE = 1.0 / 255.0


# ===========================================================================
# Module-level behavior
# ===========================================================================
@requires_vllm_config
@requires_accelerator
class TestFusedMMInputNormModule:
    @pytest.mark.parametrize("num_patches", [1, 37, 70000])
    @pytest.mark.parametrize(
        "in_dtype",
        [torch.bfloat16, torch.uint8],
        ids=["bfloat16", "uint8"],
    )
    def test_matches_reference(self, num_patches: int, in_dtype: torch.dtype):
        """Including num_patches above the old cuDNN batch-norm grid limit
        (~65535), which used to raise CUDNN_STATUS_INTERNAL_ERROR.
        """
        channel = 3
        patch_size = 14 * 14

        set_random_seed(0)
        if in_dtype == torch.uint8:
            pixel_values = torch.randint(
                0,
                256,
                (num_patches, channel * patch_size),
                dtype=torch.uint8,
                device=_DEVICE,
            )
        else:
            pixel_values = torch.randint(
                0,
                256,
                (num_patches, channel * patch_size),
                dtype=torch.float32,
                device=_DEVICE,
            )

        norm = FusedMMInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        ).to(_DEVICE)
        assert not norm.is_identity

        out = norm(pixel_values, visual_dtype=torch.float32)
        expected = _reference_input_norm(
            pixel_values,
            _RGB_MEAN,
            _RGB_STD,
            _RGB_RESCALE,
            channel,
            torch.float32,
        )
        torch.testing.assert_close(out, expected)

    def test_identity_passthrough(self):
        norm = FusedMMInputNorm.identity().to(_DEVICE)
        assert norm.is_identity

        pixel_values = torch.randn(8, 3 * 196, dtype=torch.float32, device=_DEVICE)
        out = norm(pixel_values, visual_dtype=torch.bfloat16)
        torch.testing.assert_close(out, pixel_values.to(torch.bfloat16))


# ===========================================================================
# dtype coverage
# ===========================================================================
@requires_vllm_config
@requires_accelerator
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
        channel = 3
        patch_size = 16
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
        rescale_factor = 1.0 / 255.0

        set_random_seed(0)
        if in_dtype == torch.uint8:
            pixel_values = torch.randint(
                0,
                256,
                (32, channel * patch_size),
                dtype=torch.uint8,
                device=_DEVICE,
            )
        else:
            pixel_values = torch.rand(
                (32, channel * patch_size),
                dtype=torch.float32,
                device=_DEVICE,
            ).to(in_dtype)

        norm = FusedMMInputNorm(
            image_mean=image_mean,
            image_std=image_std,
            rescale_factor=rescale_factor,
            channel=channel,
        ).to(_DEVICE)
        out = norm(pixel_values, visual_dtype=out_dtype)
        expected = _reference_input_norm(
            pixel_values,
            image_mean,
            image_std,
            rescale_factor,
            channel,
            out_dtype,
        )
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)


# ===========================================================================
# Shape / channel coverage
# ===========================================================================
@requires_vllm_config
@requires_accelerator
class TestFusedMMInputNormShapes:
    @pytest.mark.parametrize("channel", [1, 3, 4])
    def test_channel_variants(self, channel: int):
        patch_size = 64
        image_mean = [0.5] * channel
        image_std = [0.25] * channel
        rescale_factor = 1.0 / 255.0

        set_random_seed(0)
        pixel_values = torch.randint(
            0,
            256,
            (16, channel * patch_size),
            dtype=torch.float32,
            device=_DEVICE,
        )

        norm = FusedMMInputNorm(
            image_mean=image_mean,
            image_std=image_std,
            rescale_factor=rescale_factor,
            channel=channel,
        ).to(_DEVICE)
        out = norm(pixel_values, visual_dtype=torch.float32)
        expected = _reference_input_norm(
            pixel_values,
            image_mean,
            image_std,
            rescale_factor,
            channel,
            torch.float32,
        )
        torch.testing.assert_close(out, expected)

    @pytest.mark.parametrize("patch_size", [1, 255, 256, 2047, 2048, 2049, 8191, 8192])
    def test_block_boundaries(self, patch_size: int):
        """Cover both masked-tail and exact-multiple paths of the 1D kernel."""
        channel = 3
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        rescale_factor = 1.0 / 255.0

        set_random_seed(0)
        pixel_values = torch.randint(
            0,
            256,
            (4, channel * patch_size),
            dtype=torch.float32,
            device=_DEVICE,
        )

        norm = FusedMMInputNorm(
            image_mean=image_mean,
            image_std=image_std,
            rescale_factor=rescale_factor,
            channel=channel,
        ).to(_DEVICE)
        out = norm(pixel_values, visual_dtype=torch.float32)
        expected = _reference_input_norm(
            pixel_values,
            image_mean,
            image_std,
            rescale_factor,
            channel,
            torch.float32,
        )
        torch.testing.assert_close(out, expected)


# ===========================================================================
# Input handling: non-contiguous inputs
# ===========================================================================
@requires_vllm_config
@requires_accelerator
class TestFusedMMInputNormInputHandling:
    def test_non_contiguous_input_matches_reference(self):
        channel = 3
        patch_size = 32
        patches = 8

        set_random_seed(0)
        base = torch.randint(
            0,
            256,
            (patches, channel * patch_size, 2),
            dtype=torch.float32,
            device=_DEVICE,
        )
        non_contig = base[..., 0]
        assert not non_contig.is_contiguous()

        norm = FusedMMInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        ).to(_DEVICE)

        out = norm(non_contig, visual_dtype=torch.float32)
        expected = _reference_input_norm(
            non_contig.contiguous(),
            _RGB_MEAN,
            _RGB_STD,
            _RGB_RESCALE,
            channel,
            torch.float32,
        )
        torch.testing.assert_close(out, expected)

    def test_non_contiguous_input_with_out_buffer(self):
        channel = 3
        patch_size = 16
        patches = 4
        extra_rows = 3

        set_random_seed(0)
        base = torch.randint(
            0,
            256,
            (patches, channel * patch_size, 2),
            dtype=torch.float32,
            device=_DEVICE,
        )
        non_contig = base[..., 0]
        assert not non_contig.is_contiguous()

        norm = FusedMMInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        ).to(_DEVICE)

        out = torch.full(
            (patches + extra_rows, channel * patch_size),
            123.0,
            dtype=torch.float32,
            device=_DEVICE,
        )
        returned = norm(non_contig, visual_dtype=torch.float32, out=out)

        assert returned.data_ptr() == out.data_ptr()
        assert returned.shape == (patches, channel * patch_size)

        expected = _reference_input_norm(
            non_contig.contiguous(),
            _RGB_MEAN,
            _RGB_STD,
            _RGB_RESCALE,
            channel,
            torch.float32,
        )
        torch.testing.assert_close(returned, expected)
        assert torch.all(out[patches:] == 123.0)


# ===========================================================================
# Preallocated out= buffer
# ===========================================================================
@requires_vllm_config
@requires_accelerator
class TestFusedMMInputNormOutBuffer:
    def test_reuse(self):
        channel = 3
        patch_size = 32

        set_random_seed(0)
        pixel_values = torch.randint(
            0,
            256,
            (8, channel * patch_size),
            dtype=torch.float32,
            device=_DEVICE,
        )

        norm = FusedMMInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        ).to(_DEVICE)

        fresh = norm(pixel_values, visual_dtype=torch.float32)

        out = torch.empty_like(pixel_values)
        sentinel = out.data_ptr()
        returned = norm(pixel_values, visual_dtype=torch.float32, out=out)

        assert returned.data_ptr() == sentinel, "out must be written in place"
        assert returned.shape == pixel_values.shape
        torch.testing.assert_close(out, fresh)
        torch.testing.assert_close(returned, fresh)

    def test_oversized_out_buffer(self):
        """Only the leading ``patches`` rows of an oversized buffer are
        written; the returned tensor is a view into that region."""
        channel = 3
        patch_size = 32
        patches = 8
        extra_rows = 5

        set_random_seed(0)
        pixel_values = torch.randint(
            0,
            256,
            (patches, channel * patch_size),
            dtype=torch.float32,
            device=_DEVICE,
        )

        norm = FusedMMInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        ).to(_DEVICE)

        out = torch.full(
            (patches + extra_rows, channel * patch_size),
            123.0,
            dtype=torch.float32,
            device=_DEVICE,
        )
        returned = norm(pixel_values, visual_dtype=torch.float32, out=out)

        assert returned.data_ptr() == out.data_ptr()
        assert returned.shape == (patches, channel * patch_size)

        expected = _reference_input_norm(
            pixel_values,
            _RGB_MEAN,
            _RGB_STD,
            _RGB_RESCALE,
            channel,
            torch.float32,
        )
        torch.testing.assert_close(returned, expected)
        assert torch.all(out[patches:] == 123.0)

    def test_identity_oversized_out_buffer(self):
        norm = FusedMMInputNorm.identity().to(_DEVICE)
        x = torch.randn(4, 3 * 8, dtype=torch.float32, device=_DEVICE)

        out = torch.full((10, 3 * 8), 7.0, dtype=torch.bfloat16, device=_DEVICE)
        returned = norm(x, visual_dtype=torch.bfloat16, out=out)

        assert returned.data_ptr() == out.data_ptr()
        assert returned.shape == (4, 3 * 8)
        torch.testing.assert_close(returned, x.to(torch.bfloat16))
        assert torch.all(out[4:] == 7.0)

    def test_validation(self):
        """Wrong shape / dtype / device must be rejected."""
        channel = 3
        norm = FusedMMInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        ).to(_DEVICE)
        pixel_values = torch.randn(4, channel * 16, dtype=torch.float32, device=_DEVICE)

        with pytest.raises(AssertionError):
            norm(
                pixel_values,
                visual_dtype=torch.float32,
                out=torch.empty(4, channel * 16 + 1, device=_DEVICE),
            )
        with pytest.raises(AssertionError):
            norm(
                pixel_values,
                visual_dtype=torch.float32,
                out=torch.empty_like(pixel_values, dtype=torch.bfloat16),
            )
        with pytest.raises(AssertionError):
            norm(
                pixel_values,
                visual_dtype=torch.float32,
                out=torch.empty_like(pixel_values, device="cpu"),
            )

    def test_identity_out_buffer(self):
        norm = FusedMMInputNorm.identity().to(_DEVICE)
        x = torch.randn(4, 3 * 8, dtype=torch.float32, device=_DEVICE)
        out = torch.empty_like(x, dtype=torch.bfloat16)
        sentinel = out.data_ptr()

        returned = norm(x, visual_dtype=torch.bfloat16, out=out)

        assert returned.data_ptr() == sentinel
        assert returned.shape == x.shape
        torch.testing.assert_close(returned, x.to(torch.bfloat16))
        torch.testing.assert_close(out, x.to(torch.bfloat16))


# ===========================================================================
# Raw kernel entry point
# ===========================================================================
@requires_accelerator
@requires_triton
class TestFusedMMInputNormKernel:
    @pytest.mark.parametrize("block", [128, 256, 1024, 2048])
    def test_block_sizes(self, block: int):
        """``N*C*L`` is never a multiple of the tested blocks, so the masked
        tail path is exercised for every parameterisation."""
        N, C, L = 5, 3, 1000
        set_random_seed(0)
        x = torch.randn(N, C, L, dtype=torch.float32, device=_DEVICE)
        w = torch.randn(C, dtype=torch.float32, device=_DEVICE)
        b = torch.randn(C, dtype=torch.float32, device=_DEVICE)

        out = torch.empty_like(x)
        fused_mm_input_norm_triton(x, out, w, b, block=block)

        expected = x * w.view(1, C, 1) + b.view(1, C, 1)
        torch.testing.assert_close(out, expected)

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

    def test_larger_output_buffer(self):
        """Only the leading ``N`` rows of an oversized ``out`` are written."""
        N, C, L = 3, 3, 100
        set_random_seed(0)
        x = torch.randn(N, C, L, dtype=torch.float32, device=_DEVICE)
        w = torch.randn(C, dtype=torch.float32, device=_DEVICE)
        b = torch.randn(C, dtype=torch.float32, device=_DEVICE)

        # Pad only along dim 0. Padding C or L would break the flat index
        # mapping and is explicitly disallowed.
        out = torch.full(
            (N + 2, C, L),
            123.0,
            dtype=torch.float32,
            device=_DEVICE,
        )
        fused_mm_input_norm_triton(x, out, w, b)

        expected = x * w.view(1, C, 1) + b.view(1, C, 1)
        torch.testing.assert_close(out[:N], expected)
        assert torch.all(out[N:] == 123.0)

    def test_rejects_channel_or_width_padded_output(self):
        """The flat 1D kernel cannot address a buffer padded along C or L."""
        N, C, L = 3, 3, 100
        x = torch.randn(N, C, L, dtype=torch.float32, device=_DEVICE)
        w = torch.randn(C, dtype=torch.float32, device=_DEVICE)
        b = torch.randn(C, dtype=torch.float32, device=_DEVICE)

        with pytest.raises(AssertionError):
            fused_mm_input_norm_triton(
                x,
                torch.empty(N, C + 1, L, dtype=torch.float32, device=_DEVICE),
                w,
                b,
            )
        with pytest.raises(AssertionError):
            fused_mm_input_norm_triton(
                x,
                torch.empty(N, C, L + 1, dtype=torch.float32, device=_DEVICE),
                w,
                b,
            )


# ===========================================================================
# Construction / configuration
# ===========================================================================
@requires_vllm_config
class TestFusedMMInputNormConstruction:
    """Identity detection and weight/bias buffer semantics at init time."""

    def test_identity_from_identity_config(self):
        norm = FusedMMInputNorm(
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            rescale_factor=1.0,
        )
        assert norm.is_identity
        assert norm.weight is None
        assert norm.bias is None

    def test_non_identity_buffers(self):
        channel = 3
        norm = FusedMMInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        )
        assert not norm.is_identity
        assert norm.weight.shape == (channel,)
        assert norm.bias.shape == (channel,)

        mean = torch.tensor(_RGB_MEAN, dtype=torch.float32)
        std = torch.tensor(_RGB_STD, dtype=torch.float32)
        torch.testing.assert_close(norm.weight, _RGB_RESCALE / std)
        torch.testing.assert_close(norm.bias, -mean / std)

    @pytest.mark.parametrize(
        ("image_mean", "image_std", "rescale_factor", "is_identity"),
        [
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0, True),
            ([0.5, 0.5, 0.5], [0.25, 0.25, 0.25], 1 / 255, False),
        ],
    )
    def test_fused_input_norm_initialization_on_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
        image_mean: list[float],
        image_std: list[float],
        rescale_factor: float,
        is_identity: bool,
    ):
        """Identity detection must not synchronize the default device."""
        original_allclose = torch.allclose

        def cpu_allclose(input: torch.Tensor, other: torch.Tensor, *args, **kwargs):
            assert input.device.type == "cpu"
            assert other.device.type == "cpu"
            return original_allclose(input, other, *args, **kwargs)

        monkeypatch.setattr(torch, "allclose", cpu_allclose)
        # The meta device gives the CPU-only test shard the same non-CPU
        # default-device semantics without requiring a CUDA build.
        default_device = "cuda" if torch.cuda.is_available() else "meta"
        with torch.device(default_device):
            input_norm = FusedMMInputNorm(image_mean, image_std, rescale_factor)

        assert input_norm.is_identity is is_identity
        if is_identity:
            assert input_norm.weight is None
            assert input_norm.bias is None
        else:
            assert input_norm.weight.device.type == default_device
            assert input_norm.bias.device.type == default_device
