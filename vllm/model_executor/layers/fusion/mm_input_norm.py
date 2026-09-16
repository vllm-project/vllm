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

from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.transformers_utils.processor import get_processor, get_processor_config
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)

# torch dtype -> triton dtype mapping used for COMPUTE_DTYPE constexpr.
_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}

_SUPPORTED_INPUTS = (
    torch.uint8,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)
_SUPPORTED_OUTPUTS = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_COMPUTE = (torch.float32,)

# Default tile size along the flattened element axis. 4096 keeps each
# program's payload large enough to amortise launch overhead while
# staying small enough that the grid saturates the SMs for typical
# image tensors.
_DEFAULT_BLOCK = 4096

# Target elements per lane for the num_warps heuristic.
_ELEMS_PER_THREAD = 8


@triton.jit
def _fused_input_norm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    b_ptr,
    numel,
    L,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    # 1D grid over the flattened (N, C, L) tensor. Each program processes
    # BLOCK contiguous elements; the channel index is recovered from the
    # flat offset via `(offs // L) % C`.
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    c = (offs // L) % C

    x = tl.load(
        x_ptr + offs,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    ).to(COMPUTE_DTYPE)
    w = tl.load(w_ptr + c, mask=mask, other=0).to(COMPUTE_DTYPE)
    b = tl.load(b_ptr + c, mask=mask, other=0).to(COMPUTE_DTYPE)
    tl.store(
        y_ptr + offs,
        x * w + b,
        mask=mask,
        eviction_policy="evict_first",
    )


def fused_input_norm_triton(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    compute_dtype: torch.dtype,
    block: int | None = None,
    num_warps: int | None = None,
):
    """Fused per-channel affine transform for normalisation.

    Computes ``y = (x * weight[c] + bias[c]).to(y.dtype)`` in a single pass.
    Equivalent to::

        outputs[:N] = (inputs * weight.view(1, C, 1) + bias.view(1, C, 1)).to(
            outputs.dtype
        )

    Args:
        inputs: Input tensor, shape ``(N, C, L)``. Must be contiguous; the
            caller is expected to materialize a contiguous copy beforehand.
        outputs: Output tensor. Must be contiguous and shaped exactly
            ``(N_out, C, L)`` with ``N_out >= N``; only the leading ``N``
            rows are written.
        weight: Per-channel scale, shape ``(C,)``, contiguous.
        bias: Per-channel shift, shape ``(C,)``, contiguous.
        compute_dtype: Compute dtype used inside the kernel. Only
            ``torch.float32`` is currently supported.
        block: Block size along the flattened element axis. Defaults to
            ``_DEFAULT_BLOCK``.
        num_warps: Number of warps per program. If ``None``, derived from
            ``block`` targeting ~8 elements per lane.

    Returns:
        ``outputs``, for chaining.
    """
    # --- dtype validation ---------------------------------------------
    assert inputs.dtype in _SUPPORTED_INPUTS, f"unsupported input dtype: {inputs.dtype}"
    assert outputs.dtype in _SUPPORTED_OUTPUTS, (
        f"unsupported output dtype: {outputs.dtype}"
    )
    assert compute_dtype in _SUPPORTED_COMPUTE, (
        f"unsupported compute dtype: {compute_dtype}"
    )

    # --- shape validation ---------------------------------------------
    assert inputs.dim() == 3, (
        f"expected inputs to be 3D (N, C, L), got {tuple(inputs.shape)}"
    )
    assert outputs.dim() == 3, (
        f"expected outputs to be 3D (N, C, L), got {tuple(outputs.shape)}"
    )
    N, C, L = inputs.shape
    assert outputs.shape[0] >= N, (
        f"outputs.shape[0]={outputs.shape[0]} < inputs.shape[0]={N}"
    )
    # The flat 1D kernel addresses the output buffer as a contiguous
    # ``N * C * L`` block (``y_ptr + offs``), so the buffer's physical
    # layout must match the input exactly on the C and L axes. Only the
    # batch dim (dim 0) may be padded.
    assert outputs.shape[1:] == (C, L), (
        f"outputs.shape[1:]={tuple(outputs.shape[1:])} != (C, L)={(C, L)}; "
        "the flat 1D kernel addresses the output as a contiguous "
        "N * C * L block and cannot handle a channel- or width-padded "
        "output buffer"
    )
    assert weight.numel() == C and bias.numel() == C, (
        f"weight/bias must have {C} elements, got {weight.numel()} / {bias.numel()}"
    )
    assert weight.is_contiguous() and bias.is_contiguous(), (
        "weight and bias must be contiguous"
    )
    assert inputs.is_contiguous(), (
        "inputs must be contiguous; materialize a copy before calling the kernel"
    )
    assert outputs.is_contiguous(), (
        "outputs must be contiguous; only inputs is auto-materialized"
    )

    # --- derive launch config -----------------------------------------
    if block is None:
        block = _DEFAULT_BLOCK

    # The kernel only ever writes the leading ``N`` rows
    numel = N * C * L
    grid = (triton.cdiv(numel, block),)

    if num_warps is None:
        # Target ~_ELEMS_PER_THREAD elements per lane. Triton requires
        # num_warps to be a power of two; round up and clamp to [1, 16].
        target = block // (32 * _ELEMS_PER_THREAD)
        num_warps = 1
        while num_warps < target and num_warps < 16:
            num_warps *= 2

    # --- dispatch ------------------------------------------------------
    _fused_input_norm_kernel[grid](
        inputs,
        outputs,
        weight,
        bias,
        numel,
        L,
        C=C,
        BLOCK=block,
        COMPUTE_DTYPE=_TL_DTYPE[compute_dtype],
        num_warps=num_warps,
    )
    return outputs


@CustomOp.register("fused_mm_input_norm")
class FusedMMInputNorm(CustomOp):
    """
    Module that applies rescaling and normalisation to input images.
    Equivalent to: output = (input * rescale_factor - mean) / std

    Note on dtype semantics:

    * ``dtype`` controls the *internal compute precision* — the dtype in which
      the per-channel ``weight`` / ``bias`` are stored and applied. It should
      normally be ``torch.float32``; other compute dtypes can introduce
      precision loss during rescale + normalise.
    * ``visual_dtype`` (passed to :meth:`forward`) controls the *output tensor
      dtype* only. It is completely independent of the compute dtype and can
      legitimately differ from it (e.g. compute in fp32, emit bf16 for the
      vision tower).

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
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        assert len(image_mean) == channel, (
            f"image_mean has {len(image_mean)} entries but channel={channel}"
        )
        assert len(image_std) == channel, (
            f"image_std has {len(image_std)} entries but channel={channel}"
        )
        assert rescale_factor != 0.0, "rescale_factor must be non-zero"

        self.channel = channel
        self._compute_dtype = dtype

        # Model construction can set the accelerator as PyTorch's default
        # device. Determine whether the normalisation is an identity on CPU
        # so torch.allclose does not introduce a device synchronization while
        # the model is being initialized. The buffers registered below are then
        # moved to the caller's default device.
        inv_rescale = 1.0 / rescale_factor
        image_mean_cpu = (
            torch.tensor(image_mean, dtype=dtype, device="cpu") * inv_rescale
        )
        image_std_cpu = torch.tensor(image_std, dtype=dtype, device="cpu") * inv_rescale
        weight_cpu = 1.0 / image_std_cpu
        bias_cpu = -image_mean_cpu / image_std_cpu
        self.is_identity = bool(
            torch.allclose(weight_cpu, torch.ones_like(weight_cpu))
            and torch.allclose(bias_cpu, torch.zeros_like(bias_cpu))
        )

        if not self.is_identity:
            image_mean_tensor = torch.tensor(image_mean, dtype=dtype) * (
                1.0 / rescale_factor
            )
            image_std_tensor = torch.tensor(image_std, dtype=dtype) * (
                1.0 / rescale_factor
            )
            weight = 1.0 / image_std_tensor
            bias = -image_mean_tensor / image_std_tensor
            self.register_buffer("weight", weight)
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("weight", None)
            self.register_buffer("bias", None)

        if not self.is_identity and dtype != torch.float32:
            logger.warning_once(
                "FusedMMInputNorm is initialized with compute dtype=%s, which "
                "is not torch.float32. The per-channel weight/bias are stored "
                "and applied at this reduced precision, which can cause "
                "precision loss during rescale + normalise. Recommend "
                "dtype=torch.float32 for computation; use visual_dtype in "
                "forward() to select the output tensor dtype.",
                dtype,
            )

    @property
    def compute_dtype(self) -> torch.dtype:
        """The dtype used for internal computation (may differ from output)."""
        return self._compute_dtype

    @classmethod
    def identity(
        cls, channel: int = 3, dtype: torch.dtype = torch.float32
    ) -> "FusedMMInputNorm":
        return cls(
            image_mean=[0.0] * channel,
            image_std=[1.0] * channel,
            rescale_factor=1.0,
            channel=channel,
            dtype=dtype,
        )

    @staticmethod
    def _load_norm_params(
        model_config: "ModelConfig",
    ) -> tuple[bool, bool, list[float], list[float], float]:
        """Load ``(do_rescale, do_normalize, image_mean, image_std,
        rescale_factor)`` from the processor config, falling back to the image
        processor object.

        Returns concrete, non-``None`` values: ``image_mean`` / ``image_std``
        are ``list[float]`` and ``rescale_factor`` is ``float``. Explicit
        per-variable narrowing is used rather than ``assert None not in [...]``
        because mypy cannot narrow individual variables from a container-level
        ``in`` check.
        """
        model = model_config.model
        revision = model_config.revision

        # Try to read parameters from the processor config.
        config = get_processor_config(model, revision=revision)
        do_rescale: Any = config.get("do_rescale", None)
        do_normalize: Any = config.get("do_normalize", None)
        image_mean: Any = config.get("image_mean", None)
        image_std: Any = config.get("image_std", None)
        rescale_factor: Any = config.get("rescale_factor", None)

        # Fallback to the image_processor object if any parameter is missing.
        if None in [
            do_rescale,
            do_normalize,
            image_mean,
            image_std,
            rescale_factor,
        ]:
            image_processor = get_processor(model, revision=revision).image_processor

            if do_rescale is None:
                do_rescale = getattr(image_processor, "do_rescale", None)
            if do_normalize is None:
                do_normalize = getattr(image_processor, "do_normalize", None)
            if image_mean is None:
                image_mean = getattr(image_processor, "image_mean", None)
            if image_std is None:
                image_std = getattr(image_processor, "image_std", None)
            if rescale_factor is None:
                rescale_factor = getattr(image_processor, "rescale_factor", None)

        # Apply defaults based on flags.
        if not do_rescale:
            rescale_factor = 1.0
        if not do_normalize:
            image_mean = [0.0, 0.0, 0.0]
            image_std = [1.0, 1.0, 1.0]

        # Explicit per-variable narrowing: mypy cannot narrow ``Any | None``
        # via ``assert None not in [...]``, but it *can* narrow each variable
        # through a direct ``is not None`` assertion.
        assert rescale_factor is not None, (
            "rescale_factor is still None after resolution."
        )
        assert image_mean is not None, "image_mean is still None after resolution."
        assert image_std is not None, "image_std is still None after resolution."

        # Normalize to concrete types so the return type is exactly as
        # declared (``Any`` is not assignable to ``list[float]`` / ``float``
        # without a cast or a value-level construction).
        return (
            bool(do_rescale),
            bool(do_normalize),
            [float(v) for v in image_mean],
            [float(v) for v in image_std],
            float(rescale_factor),
        )

    @classmethod
    def from_model_config(cls, model_config: "ModelConfig") -> nn.Module:
        mm_config = getattr(model_config, "multimodal_config", None)
        if not getattr(mm_config, "mm_device_do_normalize", False):
            return cls.identity()

        do_rescale, do_normalize, image_mean, image_std, rescale_factor = (
            cls._load_norm_params(model_config)
        )

        # If no processing is needed, return an identity module.
        if not do_rescale and not do_normalize:
            return cls.identity()

        channel = len(image_mean)
        assert len(image_std) == channel, (
            f"image_mean and image_std have different lengths: "
            f"{len(image_mean)} vs {len(image_std)}"
        )

        return cls(
            image_mean=image_mean,
            image_std=image_std,
            rescale_factor=rescale_factor,
            channel=channel,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Internal helpers shared by the platform-specific forward_* methods
    # ------------------------------------------------------------------

    def _prepare_output(
        self,
        grid_thw: torch.Tensor,
        visual_dtype: torch.dtype,
        out: torch.Tensor | None,
    ) -> tuple[int, int, torch.Tensor | None]:
        """Validate ``out`` (if provided) and return ``(patches, size, out_view)``."""
        assert grid_thw.ndim == 2
        patches, size = grid_thw.shape

        out_view: torch.Tensor | None = None
        if out is not None:
            assert out.dim() == 2, f"out must be 2D, got {out.dim()}D"
            assert out.shape[0] >= patches, (
                f"out.shape[0]={out.shape[0]} < grid_thw.shape[0]={patches}"
            )
            assert out.shape[1] == size, (
                f"out.shape[1]={out.shape[1]} != grid_thw.shape[1]={size}"
            )
            assert out.dtype == visual_dtype, (
                f"out.dtype={out.dtype} != visual_dtype={visual_dtype}"
            )
            assert out.is_contiguous(), "out must be contiguous"
            assert out.device == grid_thw.device, (
                f"out.device={out.device} != grid_thw.device={grid_thw.device}"
            )
            out_view = out[:patches]

        return patches, size, out_view

    def _identity_forward(
        self,
        grid_thw: torch.Tensor,
        visual_dtype: torch.dtype,
        out_view: torch.Tensor | None,
    ) -> torch.Tensor:
        if out_view is not None:
            out_view.copy_(grid_thw)
            return out_view
        return grid_thw.to(visual_dtype, copy=False)

    # ------------------------------------------------------------------
    # Platform-specific implementations
    # ------------------------------------------------------------------

    def forward_native(
        self,
        grid_thw: torch.Tensor,
        visual_dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pure PyTorch eager implementation.

        This is the semantic reference implementation and the fallback used
        on any platform without a specialised kernel.
        """
        patches, size, out_view = self._prepare_output(grid_thw, visual_dtype, out)

        if self.is_identity:
            return self._identity_forward(grid_thw, visual_dtype, out_view)

        assert size % self.channel == 0, (
            f"size={size} is not divisible by channel={self.channel}"
        )
        patch_size = size // self.channel

        x = grid_thw.to(self._compute_dtype).view(patches, self.channel, patch_size)
        x = x * self.weight.view(1, self.channel, 1) + self.bias.view(
            1, self.channel, 1
        )
        y = x.view(patches, size)
        if out_view is None:
            if y.dtype != visual_dtype:
                y = y.to(visual_dtype)
            return y
        out_view.copy_(y)
        return out_view

    def forward_cuda(
        self,
        grid_thw: torch.Tensor,
        visual_dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Triton kernel path for CUDA devices."""
        patches, size, out_view = self._prepare_output(grid_thw, visual_dtype, out)

        if self.is_identity:
            return self._identity_forward(grid_thw, visual_dtype, out_view)

        # Fall back to native if the Triton kernel cannot service this call.
        if (
            not HAS_TRITON
            or grid_thw.dtype not in _SUPPORTED_INPUTS
            or visual_dtype not in _SUPPORTED_OUTPUTS
            or self.weight.dtype not in _SUPPORTED_COMPUTE
            or not self.weight.is_contiguous()
            or not self.bias.is_contiguous()
        ):
            return self.forward_native(grid_thw, visual_dtype, out)

        assert size % self.channel == 0, (
            f"size={size} is not divisible by channel={self.channel}"
        )
        patch_size = size // self.channel

        # Materialize a contiguous copy once if needed; the kernel
        # requires contiguous inputs.
        x = grid_thw if grid_thw.is_contiguous() else grid_thw.contiguous()
        x3 = x.view(patches, self.channel, patch_size)

        # The Triton kernel writes in-place into a destination buffer, so
        # this is the one path that genuinely needs ``out_view`` to exist
        # before dispatch.
        if out_view is None:
            out_view = torch.empty(
                (patches, size), dtype=visual_dtype, device=grid_thw.device
            )
        y3 = out_view.view(patches, self.channel, patch_size)

        fused_input_norm_triton(
            x3,
            y3,
            self.weight,
            self.bias,
            compute_dtype=self._compute_dtype,
        )
        return out_view

    def forward_xpu(
        self,
        grid_thw: torch.Tensor,
        visual_dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """XPU fused custom kernel path.

        On XPU, fuse the whole rescale + normalise into a single custom
        kernel. The eager path materializes an fp32 intermediate and then
        casts back, which adds device-side compute that cancels the
        bandwidth saving of transferring uint8 pixel_values. The fused
        kernel reads uint8 directly and writes ``visual_dtype`` in one pass.
        """
        patches, size, out_view = self._prepare_output(grid_thw, visual_dtype, out)

        if self.is_identity:
            return self._identity_forward(grid_thw, visual_dtype, out_view)

        if grid_thw.dtype == torch.uint8 and self.weight.dtype == torch.float32:
            y = torch.ops.vllm.xpu_fused_input_norm(
                grid_thw, self.weight, self.bias, visual_dtype
            )
            if out_view is None:
                return y
            out_view.copy_(y)
            return out_view

        # Fall back to native for unsupported dtypes on XPU.
        return self.forward_native(grid_thw, visual_dtype, out)

    def forward_oot(
        self,
        grid_thw: torch.Tensor,
        visual_dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Out-of-tree platform override entrypoint."""
        return self.forward_native(grid_thw, visual_dtype, out)
