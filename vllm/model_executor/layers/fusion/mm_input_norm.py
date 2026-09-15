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

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, triton

if HAS_TRITON:
    import triton.language as tl

from vllm.platforms import current_platform
from vllm.transformers_utils.processor import get_processor, get_processor_config

logger = init_logger(__name__)


if HAS_TRITON:
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

    # Default tile size along L. Tuned for the common case C=3 (RGB): the
    # kernel folds all C channels into one program, so a 2048-wide tile
    # keeps each lane busy with ~8 elements without excessive register
    # pressure. Adjust if a model ever uses a much larger C.
    _DEFAULT_BLOCK_L = 2048

    # Target elements per lane for the num_warps heuristic.
    _ELEMS_PER_THREAD = 8

    @triton.jit
    def _fused_input_norm_kernel(
        x_ptr,
        y_ptr,
        w_ptr,
        b_ptr,
        L,
        stride_xn,
        stride_xc,
        stride_yn,
        stride_yc,
        C: tl.constexpr,
        HAS_MASK: tl.constexpr,
        BLOCK_L: tl.constexpr,
        COMPUTE_DTYPE: tl.constexpr,
    ):
        # 2D grid: (N, cdiv(L, BLOCK_L)). Each program processes all C channels
        # for a single (n, L-block) tile, so the C loop is fully unrolled and
        # no integer div/mod is needed to recover (n, c) from a flat pid.
        n = tl.program_id(0)
        lb = tl.program_id(1)

        offs = lb * BLOCK_L + tl.arange(0, BLOCK_L)
        # Tell the compiler that `offs` forms a contiguous BLOCK_L tile whose
        # base is a multiple of BLOCK_L. This enables wide vectorized
        # ld.global.v2/v4 and st.global.v2/v4 on the L axis.
        offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_L), BLOCK_L)

        # Hoist the per-program base pointers out of the channel loop. The
        # per-channel offset `c * stride` is then just a scalar add inside
        # the unrolled loop, which the compiler can fold cheaply.
        x_base = x_ptr + n * stride_xn + offs
        y_base = y_ptr + n * stride_yn + offs

        if HAS_MASK:
            mask = offs < L
            for c in tl.static_range(C):
                # Per-channel scalars: read by every program in the grid and
                # reused C times within each program, so keep them hot in L2/L1.
                w = tl.load(w_ptr + c, eviction_policy="evict_last").to(COMPUTE_DTYPE)
                b = tl.load(b_ptr + c, eviction_policy="evict_last").to(COMPUTE_DTYPE)
                # Streaming load: read once, evict early to protect L2 from
                # being thrashed by large one-shot image tensors.
                x = tl.load(
                    x_base + c * stride_xc,
                    mask=mask,
                    other=0,
                    eviction_policy="evict_first",
                ).to(COMPUTE_DTYPE)
                tl.store(
                    y_base + c * stride_yc,
                    x * w + b,
                    mask=mask,
                    eviction_policy="evict_first",
                )
        else:
            for c in tl.static_range(C):
                w = tl.load(w_ptr + c, eviction_policy="evict_last").to(COMPUTE_DTYPE)
                b = tl.load(b_ptr + c, eviction_policy="evict_last").to(COMPUTE_DTYPE)
                x = tl.load(
                    x_base + c * stride_xc,
                    eviction_policy="evict_first",
                ).to(COMPUTE_DTYPE)
                tl.store(
                    y_base + c * stride_yc,
                    x * w + b,
                    eviction_policy="evict_first",
                )

    def fused_input_norm_kernel(
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        compute_dtype: torch.dtype,
        block_l: int | None = None,
        num_warps: int | None = None,
    ):
        """Fused per-channel affine transform for normalisation with broadcast along the L axis.

        Computes ``y = (x * weight[c] + bias[c]).to(y.dtype)`` in a single pass.
        Equivalent to::

            outputs[:N, :C, :L] = (
                inputs * weight.view(1, C, 1) + bias.view(1, C, 1)
            ).to(outputs.dtype)

        Args:
            inputs: Input tensor, shape ``(N, C, L)``. Must be contiguous; the
                caller is expected to materialize a contiguous copy beforehand.
            outputs: Output tensor. Must be contiguous, with ``outputs.shape[i]
                >= inputs.shape[i]`` for every dim; only the ``[:N, :C, :L]``
                region is written. This allows the caller to reuse a larger
                preallocated buffer without a per-call allocation.
            weight: Per-channel scale, shape ``(C,)``, contiguous.
            bias: Per-channel shift, shape ``(C,)``, contiguous.
            compute_dtype: Compute dtype used inside the kernel. Only
                ``torch.float32`` is currently supported.
            block_l: Block size along the L axis. Defaults to
                ``_DEFAULT_BLOCK_L`` (tuned for C=3). Only override if a
                model uses an unusually large ``C``.
            num_warps: Number of warps per program. If ``None``, derived from
                ``block_l`` and ``C`` targeting ~8 elements per lane.

        Returns:
            ``outputs``, for chaining.
        """
        # --- dtype validation ---------------------------------------------
        assert inputs.dtype in _SUPPORTED_INPUTS, (
            f"unsupported input dtype: {inputs.dtype}"
        )
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
        assert outputs.shape[1] >= C, (
            f"outputs.shape[1]={outputs.shape[1]} < inputs.shape[1]={C}"
        )
        assert outputs.shape[2] >= L, (
            f"outputs.shape[2]={outputs.shape[2]} < inputs.shape[2]={L}"
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
        # C is effectively always 3 (RGB), so a single tuned default is
        # enough. Callers that know better can still override block_l.
        if block_l is None:
            block_l = _DEFAULT_BLOCK_L

        has_mask = (L % block_l) != 0
        grid = (N, triton.cdiv(L, block_l))

        if num_warps is None:
            # Target ~_ELEMS_PER_THREAD elements per lane: enough ILP to
            # hide memory latency without excessive register pressure. The
            # effective per-program payload is C * block_l. Triton requires
            # num_warps to be a power of two, so round the target up to the
            # next power of two and clamp to a sane range.
            target = (C * block_l) // (32 * _ELEMS_PER_THREAD)
            num_warps = 1
            while num_warps < target and num_warps < 16:
                num_warps *= 2

        # --- dispatch ------------------------------------------------------
        _fused_input_norm_kernel[grid](
            inputs,
            outputs,
            weight,
            bias,
            L,
            inputs.stride(0),
            inputs.stride(1),
            outputs.stride(0),
            outputs.stride(1),
            C=C,
            HAS_MASK=has_mask,
            BLOCK_L=block_l,
            COMPUTE_DTYPE=_TL_DTYPE[compute_dtype],
            num_warps=num_warps,
        )
        return outputs


class FusedInputNorm(nn.Module):
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
        self._dtype = dtype

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
            # Reuse the CPU-side tensors computed above; the default device is
            # picked up implicitly by the subsequent ``to(dtype)`` (no-op for
            # dtype) followed by buffer registration, which keeps the tensors
            # on the default device when the module is later moved by the
            # caller.
            self.register_buffer("weight", weight_cpu.to(dtype=dtype))
            self.register_buffer("bias", bias_cpu.to(dtype=dtype))
        else:
            self.register_buffer("weight", None)
            self.register_buffer("bias", None)

        # Non-fp32 compute dtypes lose precision because weight/bias are
        # stored and applied at the reduced precision. Warn once per process
        # so callers see a clear recommendation without log spam. Skip the
        # warning when the transform degenerates to identity (no compute
        # happens on that path anyway).
        if not self.is_identity and dtype != torch.float32:
            logger.warning_once(
                "FusedInputNorm is initialized with compute dtype=%s, which "
                "is not torch.float32. The per-channel weight/bias are stored "
                "and applied at this reduced precision, which can cause "
                "precision loss during rescale + normalise. Recommend "
                "dtype=torch.float32 for computation; use visual_dtype in "
                "forward() to select the output tensor dtype.",
                dtype,
            )

    @property
    def dtype(self) -> torch.dtype:
        # ``weight`` is None on the identity path, so fall back to the dtype
        # captured at construction time.
        return self.weight.dtype if self.weight is not None else self._dtype

    @property
    def compute_dtype(self) -> torch.dtype:
        """The dtype used for internal computation (may differ from output)."""
        return self._dtype

    @classmethod
    def identity(
        cls, channel: int = 3, dtype: torch.dtype = torch.float32
    ) -> "FusedInputNorm":
        return cls(
            image_mean=[0.0] * channel,
            image_std=[1.0] * channel,
            rescale_factor=1.0,
            channel=channel,
            dtype=dtype,
        )

    @classmethod
    def from_model_config(cls, model_config: "ModelConfig") -> nn.Module:
        mm_config = getattr(model_config, "multimodal_config", None)
        if not getattr(mm_config, "mm_device_do_normalize", False):
            return cls.identity()

        model = model_config.model
        revision = model_config.revision

        # Try to read parameters from the processor config
        config = get_processor_config(model, revision=revision)
        do_rescale = config.get("do_rescale", None)
        do_normalize = config.get("do_normalize", None)
        image_mean = config.get("image_mean", None)
        image_std = config.get("image_std", None)
        rescale_factor = config.get("rescale_factor", None)

        # Fallback to the image_processor object if any parameter is missing
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

        # Apply defaults based on flags
        if not do_rescale:
            rescale_factor = 1.0
        if not do_normalize:
            image_mean = [0.0, 0.0, 0.0]
            image_std = [1.0, 1.0, 1.0]

        # Ensure all required parameters are resolved
        assert None not in [
            do_rescale,
            do_normalize,
            image_mean,
            image_std,
            rescale_factor,
        ], "Some normalisation parameters are still None after resolution."

        # If no processing is needed, return an identity module
        if not do_rescale and not do_normalize:
            return cls.identity()

        # Infer the channel dimension from the resolved parameters so that
        # non-RGB inputs are handled correctly.
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

    def forward(
        self,
        grid_thw: torch.Tensor,
        visual_dtype: torch.dtype,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply rescale + normalise.

        Args:
            grid_thw: Input tensor of shape ``(patches, size)`` where
                ``size == channel * patch_size``.
            visual_dtype: Desired output dtype.
            out: Optional preallocated output buffer of shape ``(patches,
                size)`` and dtype ``visual_dtype``. When provided, it is
                written in-place and returned (no allocation). Useful for
                steady-state inference loops where the caller can reuse a
                buffer across calls.

        Returns:
            The transformed tensor (either ``out`` when supplied, or a freshly
            allocated tensor of dtype ``visual_dtype``).
        """
        # ---- identity shortcut -------------------------------------------
        if self.is_identity:
            if out is not None:
                assert out.shape == grid_thw.shape, (
                    f"out.shape={tuple(out.shape)} != grid_thw.shape="
                    f"{tuple(grid_thw.shape)}"
                )
                assert out.dtype == visual_dtype, (
                    f"out.dtype={out.dtype} != visual_dtype={visual_dtype}"
                )
                out.copy_(grid_thw)
                return out
            return grid_thw.to(visual_dtype)

        assert grid_thw.ndim == 2
        patches, size = grid_thw.shape
        assert size % self.channel == 0, (
            f"size={size} is not divisible by channel={self.channel}"
        )
        patch_size = size // self.channel

        # ---- output buffer validation ------------------------------------
        if out is not None:
            assert out.shape == grid_thw.shape, (
                f"out.shape={tuple(out.shape)} != grid_thw.shape="
                f"{tuple(grid_thw.shape)}"
            )
            assert out.dtype == visual_dtype, (
                f"out.dtype={out.dtype} != visual_dtype={visual_dtype}"
            )
            assert out.is_contiguous(), "out must be contiguous"
            assert out.device == grid_thw.device, (
                f"out.device={out.device} != grid_thw.device={grid_thw.device}"
            )

        # ---- XPU fused custom kernel -------------------------------------
        # On XPU, fuse the whole rescale + normalise into a single custom
        # kernel. The eager path below materializes an fp32 intermediate and
        # then casts back, which adds device-side compute that cancels the
        # bandwidth saving of transferring uint8 pixel_values. The fused
        # kernel reads uint8 directly and writes ``visual_dtype`` in one pass.
        if (
            current_platform.is_xpu()
            and grid_thw.dtype == torch.uint8
            and self.weight.dtype == torch.float32
        ):
            result = torch.ops.vllm.xpu_fused_input_norm(
                grid_thw, self.weight, self.bias, visual_dtype
            )
            if out is not None:
                out.copy_(result)
                return out
            return result

        # ---- Triton fast path --------------------------------------------
        # Fuse the per-channel affine transform into a single pass over the
        # input, avoiding the intermediate fp32 materialization of the eager
        # path below.
        if (
            HAS_TRITON
            and grid_thw.dtype in _SUPPORTED_INPUTS
            and visual_dtype in _SUPPORTED_OUTPUTS
            and self.weight.dtype in _SUPPORTED_COMPUTE
            and self.weight.is_contiguous()
            and self.bias.is_contiguous()
        ):
            # Materialize a contiguous copy once if needed; the kernel
            # requires contiguous inputs.
            x = grid_thw if grid_thw.is_contiguous() else grid_thw.contiguous()
            x3 = x.view(patches, self.channel, patch_size)

            if out is not None:
                y3 = out.view(patches, self.channel, patch_size)
            else:
                y3 = torch.empty_like(x3, dtype=visual_dtype)

            fused_input_norm_kernel(
                x3,
                y3,
                self.weight,
                self.bias,
                compute_dtype=self._dtype,
            )
            # `out` (2D) and `y3.view(...)` are the same storage; return the
            # caller-supplied tensor when we were given one so the aliasing
            # contract is explicit.
            return out if out is not None else y3.view(patches, size)

        # ---- Fallback eager path -----------------------------------------
        # Applies the per-channel affine transform directly instead of via
        # F.batch_norm. batch_norm dispatches to cuDNN, whose batch-norm
        # kernels cap the batch dimension near the CUDA grid limit (~65535);
        # here that dimension is the number of patches, which grows unbounded
        # with image resolution and batch size and overflows the cap on large
        # image-heavy requests (CUDNN_STATUS_INTERNAL_ERROR). The plain
        # broadcasted multiply-add is numerically identical and has no such
        # limit.
        x = grid_thw.to(self.dtype).view(patches, self.channel, patch_size)
        x = x * self.weight.view(1, self.channel, 1) + self.bias.view(
            1, self.channel, 1
        )
        result = x.view(patches, size)
        if result.dtype != visual_dtype:
            result = result.to(visual_dtype)

        if out is not None:
            out.copy_(result)
            return out
        return result
