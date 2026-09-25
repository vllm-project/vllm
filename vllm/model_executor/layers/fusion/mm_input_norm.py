# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused Normalisation on the Device.

Equivalent to::

    output = (input * rescale_factor - image_mean) / image_std

This is implemented as a single per-channel affine transform::

    output = input * weight[c] + bias[c]

where::

    weight = rescale_factor / image_std
    bias = -image_mean / image_std
"""

import math
from typing import Any, NamedTuple

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.transformers_utils.processor import cached_get_processor, get_processor_config
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_mm_input_norm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    numel,
    L,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 1D grid over the flattened (N, C, L) tensor. Each program processes
    # BLOCK contiguous elements; the channel index is recovered from the
    # flat offset via `(offs // L) % C`. int32 offsets: int64 division is
    # ~2x slower here (measured on GB200); the wrapper asserts
    # numel < 2**31 so the flat index never overflows.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    c = (offs // L) % C

    x = tl.load(
        x_ptr + offs,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    w = tl.load(w_ptr + c, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=mask, other=0).to(tl.float32)
    tl.store(
        y_ptr + offs,
        x * w + b,
        mask=mask,
        eviction_policy="evict_first",
    )


def fused_mm_input_norm_triton(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
):
    """Fused per-channel affine transform for normalisation.

    Args:
        inputs: Input tensor, shape ``(N, C, L)``. A contiguous copy is
            materialized internally if needed.
        outputs: Output tensor, contiguous and shaped exactly ``(N, C, L)``.
        weight: Per-channel scale, shape ``(C,)``.
        bias: Per-channel shift, shape ``(C,)``.

    Returns:
        ``outputs``, for chaining.

    """
    N, C, L = inputs.shape
    assert outputs.shape == (N, C, L), (
        f"outputs shape {tuple(outputs.shape)} != inputs shape {(N, C, L)}"
    )
    assert weight.numel() == C and bias.numel() == C, (
        f"weight/bias must have {C} elements, got {weight.numel()} / {bias.numel()}"
    )
    # The flat 1D kernel writes outputs by raw offset; a non-contiguous
    # buffer would be written at the wrong locations and cannot be
    # auto-materialized without breaking the caller's aliasing.
    assert outputs.is_contiguous(), "outputs must be contiguous"

    # Tile size along the flattened element axis. 1024 with 4 warps gives
    # 8 elements per lane and benchmarks fastest across realistic image
    # sizes; larger tiles starve the SMs on small inputs.
    BLOCK_SIZE = 1024

    numel = N * C * L
    # The kernel indexes with int32 offsets (int64 division costs ~2x
    # throughput); guard the flat index range here.
    assert numel < 2**31, f"numel={numel} exceeds the int32 index range"

    grid = (triton.cdiv(numel, BLOCK_SIZE),)

    _fused_mm_input_norm_kernel[grid](
        inputs.contiguous(),
        outputs,
        weight.contiguous(),
        bias.contiguous(),
        numel,
        L,
        C=C,
        BLOCK=BLOCK_SIZE,
        num_warps=4,  # 8 elements per lane at BLOCK_SIZE=1024
    )
    return outputs


class NormParams(NamedTuple):
    """Resolved per-channel affine parameters (flags already folded in)."""

    image_mean: list[float]
    image_std: list[float]
    rescale_factor: float

    @property
    def is_identity(self) -> bool:
        """Whether ``weight = rescale/std`` and ``bias = -mean/std`` are
        numerically the identity transform (mirrors ``torch.allclose``'s
        default fp32 tolerances)."""
        return all(
            math.isclose(self.rescale_factor / s, 1.0, rel_tol=1e-5, abs_tol=1e-8)
            and math.isclose(m / s, 0.0, abs_tol=1e-8)
            for m, s in zip(self.image_mean, self.image_std)
        )


def _load_norm_params(model_config: ModelConfig) -> NormParams:
    """Resolve the per-channel affine parameters ``(image_mean, image_std,
    rescale_factor)`` from the processor config, falling back to the image
    processor object."""
    model = model_config.model
    revision = model_config.revision

    config = get_processor_config(model, revision=revision)
    # NOTE: do not use cached_image_processor_from_config here — it merges
    # mm_processor_kwargs, which mm_device_do_normalize poisons with
    # do_normalize=False.
    image_processor = cached_get_processor(
        model, revision=revision, trust_remote_code=model_config.trust_remote_code
    ).image_processor

    def resolve(key: str) -> Any:
        """Processor config value, falling back to the image_processor."""
        if (value := config.get(key)) is not None:
            return value
        return getattr(image_processor, key, None)

    do_rescale = bool(resolve("do_rescale"))
    do_normalize = bool(resolve("do_normalize"))

    # Parameters whose flag is off are unused; default them to no-ops
    # without resolving them.
    rescale_factor = resolve("rescale_factor") if do_rescale else 1.0
    image_mean = resolve("image_mean") if do_normalize else [0.0] * 3
    image_std = resolve("image_std") if do_normalize else [1.0] * 3

    assert rescale_factor is not None, "rescale_factor is still None after resolution."
    assert image_mean is not None, "image_mean is still None after resolution."
    assert image_std is not None, "image_std is still None after resolution."
    assert len(image_mean) == len(image_std), (
        f"image_mean and image_std have different lengths: "
        f"{len(image_mean)} vs {len(image_std)}"
    )

    return NormParams(
        image_mean=[float(v) for v in image_mean],
        image_std=[float(v) for v in image_std],
        rescale_factor=float(rescale_factor),
    )


class IdentityInputNorm(nn.Module):
    """Stand-in used when the processor requires no rescale/normalise.

    Not a no-op: with ``mm_device_do_normalize`` enabled, raw ``uint8``
    pixels travel to the device unprocessed and must be cast to
    ``visual_dtype`` here.
    """

    def forward(
        self, pixel_values: torch.Tensor, visual_dtype: torch.dtype
    ) -> torch.Tensor:
        return pixel_values.to(visual_dtype, copy=False)


@CustomOp.register("fused_mm_input_norm")
class FusedMMInputNorm(CustomOp):
    """Module that applies rescaling and normalisation to input images.
    Equivalent to: output = (input * rescale_factor - mean) / std

    Dtype semantics:

    * Input dtype — the dtype of the ``pixel_values`` argument to ``forward_*``.
      It is ``uint8`` when ``mm_device_do_normalize`` is enabled (raw bytes
      travel to the device unprocessed) and equals ``visual_dtype``
      otherwise.
    * Output dtype — the ``visual_dtype`` argument of ``forward_*``. The
      computation itself is always fp32, independent of the output dtype
      (e.g. compute fp32, emit bf16).

    Platform dispatch:

    * ``forward_native`` — pure PyTorch eager path, used as the semantic
      reference and default fallback on all platforms.
    * ``forward_cuda`` — Triton kernel path for CUDA devices.
    * ``forward_xpu`` — custom XPU kernel path.
    * ``forward_oot`` — out-of-tree platform override entry point; falls back
      to ``forward_native`` unless a plugin overrides it.
    """

    def __init__(
        self,
        image_mean: list[float],
        image_std: list[float],
        rescale_factor: float,
        channel: int = 3,
    ):
        super().__init__()

        assert len(image_mean) == len(image_std) == channel, (
            f"image_mean/image_std must have {channel} entries, "
            f"got {len(image_mean)} / {len(image_std)}"
        )
        assert rescale_factor != 0.0, "rescale_factor must be non-zero"

        self.channel = channel

        # Model construction can set the accelerator as PyTorch's default
        # device; build the buffers on CPU first, then move them over.
        mean = torch.tensor(image_mean, dtype=torch.float32, device="cpu")
        std = torch.tensor(image_std, dtype=torch.float32, device="cpu")
        device = torch.get_default_device()
        self.register_buffer("weight", (rescale_factor / std).to(device))
        self.register_buffer("bias", (-mean / std).to(device))

    # ------------------------------------------------------------------
    # Internal helpers shared by the platform-specific forward_* methods
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_2d(pixel_values: torch.Tensor) -> tuple[int, int]:
        assert pixel_values.ndim == 2, (
            f"pixel_values must be 2D (patches, size), got {pixel_values.dim()}D "
            f"with shape {tuple(pixel_values.shape)}"
        )
        patches, size = pixel_values.shape
        return patches, size

    def _patch_size(self, size: int) -> int:
        assert size % self.channel == 0, (
            f"size={size} is not divisible by channel={self.channel}"
        )
        return size // self.channel

    # ------------------------------------------------------------------
    # Platform-specific implementations
    # ------------------------------------------------------------------

    def forward_native(
        self, pixel_values: torch.Tensor, visual_dtype: torch.dtype
    ) -> torch.Tensor:
        """Pure PyTorch eager implementation.

        This is the semantic reference implementation and the fallback used
        on any platform without a specialised kernel.
        """
        patches, size = self._unpack_2d(pixel_values)
        patch_size = self._patch_size(size)

        # weight/bias are fp32, so type promotion makes the arithmetic fp32
        x = pixel_values.reshape(patches, self.channel, patch_size)
        x = x * self.weight.view(1, self.channel, 1) + self.bias.view(
            1, self.channel, 1
        )
        return x.view(patches, size).to(visual_dtype)

    def forward_cuda(
        self, pixel_values: torch.Tensor, visual_dtype: torch.dtype
    ) -> torch.Tensor:
        """Triton kernel path for CUDA devices."""
        patches, size = self._unpack_2d(pixel_values)
        patch_size = self._patch_size(size)

        x3 = pixel_values.reshape(patches, self.channel, patch_size)
        y = torch.empty((patches, size), dtype=visual_dtype, device=pixel_values.device)
        y3 = y.view(patches, self.channel, patch_size)

        fused_mm_input_norm_triton(x3, y3, self.weight, self.bias)
        return y

    def forward_xpu(
        self, pixel_values: torch.Tensor, visual_dtype: torch.dtype
    ) -> torch.Tensor:
        """XPU fused custom kernel path.

        On XPU, fuse the whole rescale + normalise into a single custom
        kernel. The eager path materializes an fp32 intermediate and then
        casts back, which adds device-side compute that cancels the
        bandwidth saving of transferring uint8 pixel_values. The fused
        kernel reads uint8 directly and writes ``visual_dtype`` in one pass.
        """
        # The out-of-tree XPU kernel only supports the uint8 input that
        # device-side normalisation guarantees; fail loudly otherwise.
        assert pixel_values.dtype == torch.uint8, (
            f"xpu_fused_input_norm requires uint8 input, got {pixel_values.dtype}"
        )
        return torch.ops.vllm.xpu_fused_input_norm(
            pixel_values, self.weight, self.bias, visual_dtype
        )

    def forward_oot(
        self, pixel_values: torch.Tensor, visual_dtype: torch.dtype
    ) -> torch.Tensor:
        """Out-of-tree platform override entrypoint."""
        return self.forward_native(pixel_values, visual_dtype)


def build_mm_input_norm(model_config: ModelConfig) -> nn.Module:
    """Build the input normalisation module for a model.

    Returns an ``IdentityInputNorm`` when device-side normalisation is
    disabled or the processor's rescale/normalise is numerically the identity
    transform; otherwise a ``FusedMMInputNorm`` built from the processor's
    parameters.
    """
    mm_config = getattr(model_config, "multimodal_config", None)
    if not getattr(mm_config, "mm_device_do_normalize", False):
        return IdentityInputNorm()

    params = _load_norm_params(model_config)

    # Flags off are already defaulted to identity values by
    # ``_load_norm_params``; this also catches configs that request
    # rescale/normalise but are numerically the identity transform.
    if params.is_identity:
        return IdentityInputNorm()

    return FusedMMInputNorm(
        image_mean=params.image_mean,
        image_std=params.image_std,
        rescale_factor=params.rescale_factor,
        channel=len(params.image_mean),
    )
