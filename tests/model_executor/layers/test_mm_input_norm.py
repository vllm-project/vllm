# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for FusedInputNorm and the fused affine transform kernel."""

import pytest
import torch

from vllm.model_executor.layers.fusion.mm_input_norm import (
    FusedInputNorm,
    fused_input_norm_triton,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import set_random_seed

# ---------------------------------------------------------------------------
# Device / accelerator gating
# ---------------------------------------------------------------------------
# The fused Triton kernel writes/reads device pointers directly; it cannot run
# on a CPU tensor. Derive the test device from the active vLLM platform and
# skip the accelerator-only tests when none is available.
_DEVICE_TYPE = current_platform.device_type  # e.g. "cuda", "xpu", "cpu"
_DEVICE = torch.device(_DEVICE_TYPE)

requires_accelerator = pytest.mark.skipif(
    _DEVICE_TYPE == "cpu",
    reason="fused Triton kernel requires a CUDA/XPU accelerator",
)
requires_triton = pytest.mark.skipif(
    not HAS_TRITON,
    reason="requires Triton",
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
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


# Common RGB normalization constants (CLIP-style) reused across tests.
_RGB_MEAN = [0.48145466, 0.4578275, 0.40821073]
_RGB_STD = [0.26862954, 0.26130258, 0.27577711]
_RGB_RESCALE = 1.0 / 255.0


# ===========================================================================
# Module-level behavior
# ===========================================================================
@requires_accelerator
class TestFusedInputNormModule:
    """End-to-end behavior of the nn.Module wrapper."""

    @pytest.mark.parametrize("num_patches", [1, 37, 70000])
    def test_matches_reference(self, num_patches: int):
        """FusedInputNorm must equal the plain affine, including for
        num_patches above the cuDNN batch-norm grid limit (~65535) that
        previously raised CUDNN_STATUS_INTERNAL_ERROR (issue #51717)."""
        channel = 3
        patch_size = 14 * 14

        set_random_seed(0)
        pixel_values = torch.randint(
            0,
            256,
            (num_patches, channel * patch_size),
            dtype=torch.float32,
            device=_DEVICE,
        )

        norm = FusedInputNorm(
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
        """The identity configuration returns the input unchanged (cast
        only)."""
        norm = FusedInputNorm.identity().to(_DEVICE)
        assert norm.is_identity

        pixel_values = torch.randn(8, 3 * 196, dtype=torch.float32, device=_DEVICE)
        out = norm(pixel_values, visual_dtype=torch.bfloat16)
        torch.testing.assert_close(out, pixel_values.to(torch.bfloat16))


# ===========================================================================
# dtype coverage
# ===========================================================================
@requires_accelerator
class TestFusedInputNormDtypes:
    """Input/output dtype combinations supported by the Triton fast path."""

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
        """All supported input/output dtype combinations should agree with
        the float32 reference within the output dtype's tolerance."""
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

        norm = FusedInputNorm(
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
@requires_accelerator
class TestFusedInputNormShapes:
    """Channel variants and block-size boundaries."""

    @pytest.mark.parametrize("channel", [1, 3, 4])
    def test_channel_variants(self, channel: int):
        """The kernel folds all C channels into one program; a single tuned
        block_l should still be correct for a range of C."""
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

        norm = FusedInputNorm(
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

    @pytest.mark.parametrize(
        "patch_size",
        [
            1,  # tiny L, single masked block
            255,  # just under a power of two
            256,  # aligned to 256
            2047,  # just under default block_l=2048
            2048,  # exactly one block, no mask
            2049,  # one masked tail
            8191,  # several full blocks + mask
            8192,  # exact multiple of 2048
        ],
    )
    def test_block_boundaries(self, patch_size: int):
        """Exercise both the HAS_MASK=True and HAS_MASK=False kernel branches
        by picking L values that straddle the default block size."""
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

        norm = FusedInputNorm(
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
@requires_accelerator
class TestFusedInputNormInputHandling:
    """The wrapper must transparently handle non-contiguous inputs.

    ``forward`` materializes a contiguous copy internally when the input is
    not contiguous, because the Triton kernel and the trailing ``.view(...)``
    both require contiguous storage.
    """

    def test_non_contiguous_input_matches_reference(self):
        """A non-contiguous input (e.g. a strided slice) must produce the
        same result as its contiguous copy."""
        channel = 3
        patch_size = 32
        patches = 8

        set_random_seed(0)
        # Build a 3D tensor and take a strided 2D slice along the last dim so
        # the resulting 2D view has non-trivial strides on both axes.
        base = torch.randint(
            0,
            256,
            (patches, channel * patch_size, 2),
            dtype=torch.float32,
            device=_DEVICE,
        )
        non_contig = base[..., 0]
        assert not non_contig.is_contiguous()

        norm = FusedInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        ).to(_DEVICE)

        out = norm(non_contig, visual_dtype=torch.float32)

        # Reference is computed on the contiguous copy of the same values.
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
        """A non-contiguous input combined with an ``out=`` buffer (also
        possibly oversized) must still write only the leading rows."""
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

        norm = FusedInputNorm(
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

        # The untouched tail must still be the sentinel value.
        assert torch.all(out[patches:] == 123.0)


# ===========================================================================
# Preallocated out= buffer
# ===========================================================================
@requires_accelerator
class TestFusedInputNormOutBuffer:
    """The optional ``out=`` argument: buffer reuse and validation."""

    def test_reuse(self):
        """Passing a preallocated `out` buffer must write in-place and return
        a tensor aliasing the same storage (the leading ``patches`` rows),
        with results identical to the allocating path."""
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

        norm = FusedInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        ).to(_DEVICE)

        fresh = norm(pixel_values, visual_dtype=torch.float32)

        out = torch.empty_like(pixel_values)
        sentinel = out.data_ptr()
        returned = norm(pixel_values, visual_dtype=torch.float32, out=out)

        # ``forward`` returns ``out[:patches]``, a view that aliases the same
        # storage. It is *not* the same Python object as ``out``, so compare
        # by data pointer + shape rather than identity.
        assert returned.data_ptr() == sentinel, "out must be written in place"
        assert returned.shape == pixel_values.shape
        torch.testing.assert_close(out, fresh)
        torch.testing.assert_close(returned, fresh)

    def test_oversized_out_buffer(self):
        """``out`` may be larger than the input along dim 0 (``N``). Only the
        leading ``patches`` rows are written; the returned tensor is a view
        into that region and the trailing rows are untouched.

        This is the core contract introduced by allowing the caller to reuse
        a buffer sized for the maximum batch across calls.
        """
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

        norm = FusedInputNorm(
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

        # The returned tensor aliases the leading rows of ``out`` and has the
        # shape of the input, not of the buffer.
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

        # Only the leading ``patches`` rows were written; the tail keeps the
        # sentinel value.
        assert torch.all(out[patches:] == 123.0)

    def test_identity_oversized_out_buffer(self):
        """The identity fast path must honor an oversized ``out`` buffer as
        well, writing only the leading rows and returning a view into them."""
        norm = FusedInputNorm.identity().to(_DEVICE)
        x = torch.randn(4, 3 * 8, dtype=torch.float32, device=_DEVICE)

        out = torch.full((10, 3 * 8), 7.0, dtype=torch.bfloat16, device=_DEVICE)
        returned = norm(x, visual_dtype=torch.bfloat16, out=out)

        assert returned.data_ptr() == out.data_ptr()
        assert returned.shape == (4, 3 * 8)
        torch.testing.assert_close(returned, x.to(torch.bfloat16))
        assert torch.all(out[4:] == 7.0)

    def test_validation(self):
        """`out` with a wrong shape / dtype / device must be rejected."""
        channel = 3
        norm = FusedInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        ).to(_DEVICE)
        pixel_values = torch.randn(4, channel * 16, dtype=torch.float32, device=_DEVICE)

        # Wrong shape.
        with pytest.raises(AssertionError):
            norm(
                pixel_values,
                visual_dtype=torch.float32,
                out=torch.empty(4, channel * 16 + 1, device=_DEVICE),
            )
        # Wrong dtype.
        with pytest.raises(AssertionError):
            norm(
                pixel_values,
                visual_dtype=torch.float32,
                out=torch.empty_like(pixel_values, dtype=torch.bfloat16),
            )
        # Wrong device.
        with pytest.raises(AssertionError):
            norm(
                pixel_values,
                visual_dtype=torch.float32,
                out=torch.empty_like(pixel_values, device="cpu"),
            )

    def test_identity_out_buffer(self):
        """`out=` must also be honored on the identity fast path."""
        norm = FusedInputNorm.identity().to(_DEVICE)
        x = torch.randn(4, 3 * 8, dtype=torch.float32, device=_DEVICE)
        out = torch.empty_like(x, dtype=torch.bfloat16)
        sentinel = out.data_ptr()

        returned = norm(x, visual_dtype=torch.bfloat16, out=out)

        # Returned tensor aliases ``out`` (view), not the same object.
        assert returned.data_ptr() == sentinel
        assert returned.shape == x.shape
        torch.testing.assert_close(returned, x.to(torch.bfloat16))
        torch.testing.assert_close(out, x.to(torch.bfloat16))


# ===========================================================================
# Raw kernel entry point
# ===========================================================================
@requires_accelerator
@requires_triton
class TestFusedInputNormKernel:
    """Direct tests of the ``fused_input_norm_triton`` entry point."""

    @pytest.mark.parametrize("block_l", [128, 256, 1024, 2048])
    def test_block_sizes(self, block_l: int):
        """The raw kernel should produce the reference output for a range of
        block_l values, including ones that force a masked tail."""
        N, C, L = 5, 3, 1000  # L not divisible by any of the tested blocks
        set_random_seed(0)
        x = torch.randn(N, C, L, dtype=torch.float32, device=_DEVICE)
        w = torch.randn(C, dtype=torch.float32, device=_DEVICE)
        b = torch.randn(C, dtype=torch.float32, device=_DEVICE)

        out = torch.empty_like(x)
        fused_input_norm_triton(
            x, out, w, b, compute_dtype=torch.float32, block_l=block_l
        )

        expected = x * w.view(1, C, 1) + b.view(1, C, 1)
        torch.testing.assert_close(out, expected)

    def test_larger_output_buffer(self):
        """The kernel writes only ``[:N, :C, :L]``; the rest of an oversized
        output buffer must be left untouched."""
        N, C, L = 3, 3, 100
        set_random_seed(0)
        x = torch.randn(N, C, L, dtype=torch.float32, device=_DEVICE)
        w = torch.randn(C, dtype=torch.float32, device=_DEVICE)
        b = torch.randn(C, dtype=torch.float32, device=_DEVICE)

        out = torch.full(
            (N + 2, C + 1, L + 5),
            123.0,
            dtype=torch.float32,
            device=_DEVICE,
        )
        fused_input_norm_triton(x, out, w, b, compute_dtype=torch.float32)

        expected = x * w.view(1, C, 1) + b.view(1, C, 1)
        torch.testing.assert_close(out[:N, :C, :L], expected)

        # The untouched tail must still be the sentinel value.
        assert torch.all(out[N:] == 123.0)
        assert torch.all(out[:N, C:, :] == 123.0)
        assert torch.all(out[:N, :C, L:] == 123.0)

    def test_rejects_unsupported_compute_dtype(self):
        N, C, L = 2, 3, 64
        x = torch.randn(N, C, L, dtype=torch.float32, device=_DEVICE)
        out = torch.empty_like(x)
        w = torch.randn(C, dtype=torch.float32, device=_DEVICE)
        b = torch.randn(C, dtype=torch.float32, device=_DEVICE)

        with pytest.raises(AssertionError):
            fused_input_norm_triton(x, out, w, b, compute_dtype=torch.float16)


# ===========================================================================
# Construction / configuration
# ===========================================================================
class TestFusedInputNormConstruction:
    """Identity detection and weight/bias buffer semantics at init time.

    These do not touch the device-side kernel and run on CPU.
    """

    def test_identity_from_identity_config(self):
        """An identity config (rescale=1, mean=0, std=1) should collapse to
        the identity module."""
        norm = FusedInputNorm(
            image_mean=[0.0, 0.0, 0.0],
            image_std=[1.0, 1.0, 1.0],
            rescale_factor=1.0,
        )
        assert norm.is_identity
        assert norm.weight is None
        assert norm.bias is None

    def test_non_identity_buffers(self):
        """A non-identity module must expose fp32 weight/bias of the right
        shape and matching the closed-form affine coefficients."""
        channel = 3
        norm = FusedInputNorm(
            image_mean=_RGB_MEAN,
            image_std=_RGB_STD,
            rescale_factor=_RGB_RESCALE,
            channel=channel,
        )
        assert not norm.is_identity
        assert norm.weight.shape == (channel,)
        assert norm.bias.shape == (channel,)

        inv_rescale = 1.0 / _RGB_RESCALE
        mean = torch.tensor(_RGB_MEAN, dtype=torch.float32) * inv_rescale
        std = torch.tensor(_RGB_STD, dtype=torch.float32) * inv_rescale
        torch.testing.assert_close(norm.weight, 1.0 / std)
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
        # Exercise the real accelerator when available. The meta device gives
        # the CPU-only test shard the same non-CPU default-device semantics
        # without requiring a CUDA-enabled PyTorch build.
        default_device = "cuda" if torch.cuda.is_available() else "meta"
        with torch.device(default_device):
            input_norm = FusedInputNorm(image_mean, image_std, rescale_factor)

        assert input_norm.is_identity is is_identity
        if is_identity:
            assert input_norm.weight is None
            assert input_norm.bias is None
        else:
            assert input_norm.weight.device.type == default_device
            assert input_norm.bias.device.type == default_device
