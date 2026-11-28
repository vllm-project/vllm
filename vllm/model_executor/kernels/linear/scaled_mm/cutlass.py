# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Sequence

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
    convert_to_channelwise,
    cutlass_fp8_supported,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_gemm_fp8_nt_groupwise,
    has_flashinfer_trtllm_fp8_groupwise_gemm,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)


class CutlassInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        config = self.config
        # WEIGHT
        # Cutlass kernels need transposed weight.
        weight = getattr(layer, w_q_name)
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(weight.t().data, requires_grad=False),
        )

        # WEIGHT SCALE
        # Cutlass kernels support only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, w_s_name)
        if is_fused_module and not config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        if config.is_static_input_scheme:
            input_scale = getattr(layer, i_s_name)

            if config.input_symmetric:
                replace_parameter(
                    layer,
                    i_s_name,
                    torch.nn.Parameter(input_scale.max(), requires_grad=False),
                )
            else:
                input_zero_point = getattr(layer, i_zp_name)

                # reconstruct the ranges
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
                replace_parameter(
                    layer, i_s_name, torch.nn.Parameter(scale, requires_grad=False)
                )

                # AZP loaded as int8 but used as int32
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
                replace_parameter(
                    layer, i_zp_name, torch.nn.Parameter(azp, requires_grad=False)
                )

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # For more details, see
        # csrc/quantization/w8a8/cutlass/Epilogues.md
        # https://github.com/vllm-project/vllm/blob/main/csrc/quantization/w8a8/cutlass/Epilogues.md
        if not config.input_symmetric:
            weight = getattr(layer, w_q_name)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if config.is_static_input_scheme:
                # cutlass_w8a8 requires azp to be folded into azp_adj
                # in the per-tensor case
                azp_adj = getattr(layer, i_zp_name) * azp_adj
            replace_parameter(layer, azp_adj_name, azp_adj)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=symmetric
        )

        if x_zp is not None:
            # Currently, static is always per-tensor and dynamic is per-token
            static = i_zp is not None
            azp = None if static else x_zp
            return ops.cutlass_scaled_mm_azp(
                x_q,
                w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=x.dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )
        return ops.cutlass_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )


class CutlassFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        self.logical_output_size = c.weight_shape[0]
        super().__init__(c, layer_param_names)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not cutlass_fp8_supported():
            return False, "CUTLASS FP8 kernels not available"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def input_quant_key(self) -> QuantKey | None:
        """Static per-tensor and dynamic per-token activations can be quantized
        outside the kernel: neither is padded, and cutlass_scaled_mm takes the
        `(M, 1)` token scales as is."""
        key = self.config.activation_quant_key
        if key in (kFp8StaticTensorSym, kFp8DynamicTokenSym):
            return key
        return None

    @staticmethod
    def _pad_to_alignment(
        x: torch.Tensor, dim: int, alignment: int, value: float = 0.0
    ) -> torch.Tensor:
        """Pad tensor `x` along `dim` to the next multiple of
        `alignment`."""
        remainder = x.shape[dim] % alignment
        if remainder == 0:
            return x
        pad_size = alignment - remainder
        pad_spec = [0] * (2 * x.dim())
        pad_spec[-(2 * dim + 1)] = pad_size
        return torch.nn.functional.pad(x, pad_spec, value=value)

    @staticmethod
    def padded_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if loaded_weight.shape != param.shape:
            slices = tuple(slice(0, s) for s in loaded_weight.shape)
            param.data[slices].copy_(loaded_weight)
        else:
            param.data.copy_(loaded_weight.view(param.shape))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_name, weight_scale_name, _, _ = self.layer_param_names
        weight = getattr(layer, weight_name)

        # keep the logical output width so runtime can slice away static padding.
        self.logical_output_size = weight.shape[1]

        pad_k = (16 - weight.shape[0] % 16) % 16
        pad_n = (16 - weight.shape[1] % 16) % 16
        if pad_k == 0 and pad_n == 0:
            return

        # B is column-major [K, N]
        padded_weight = torch.nn.functional.pad(
            weight.t().contiguous(),
            (0, pad_k, 0, pad_n),
        ).t()
        replace_parameter(layer, weight_name, padded_weight.data)
        # The kept param already has a loader; reloads must use the padded one.
        getattr(layer, weight_name).weight_loader = self.padded_weight_loader

        weight_scale = getattr(layer, weight_scale_name, None)
        if weight_scale is not None and pad_n > 0 and weight_scale.numel() > 1:
            flat_scale = weight_scale.reshape(-1)
            padded_scale = self._pad_to_alignment(
                flat_scale, dim=0, alignment=16, value=1.0
            ).view(-1, *weight_scale.shape[1:])
            replace_parameter(layer, weight_scale_name, padded_scale.data)
            getattr(layer, weight_name).weight_loader = self.padded_weight_loader

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        padded_k, padded_n = B.shape
        output_size = self.logical_output_size
        assert output_size is not None
        pad_k = padded_k - A.shape[1]
        pad_n = padded_n - output_size

        if pad_k > 0:
            A = self._pad_to_alignment(A, dim=1, alignment=16)
        if pad_n > 0 and bias is not None:
            bias = self._pad_to_alignment(bias, dim=0, alignment=16)

        output = ops.cutlass_scaled_mm(
            A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
        )

        if pad_n > 0:
            output = output[..., :output_size].contiguous()

        return output.view(*output_shape[:-1], output_size)


class CutlassFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        act_scale_descriptor = config.activation_quant_key.scale
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
            use_ue8m0=False,
            column_major_scales=True,
        )
        # Small-M problems run FlashInfer's TRT-LLM groupwise FP8 GEMM, which
        # gives the same output as the CUTLASS kernel (see
        # _use_flashinfer_trtllm_fp8_block_gemm).
        self.use_flashinfer_small_m = (
            config.out_dtype == torch.bfloat16
            and has_flashinfer_trtllm_fp8_groupwise_gemm()
        )

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not CUTLASS_BLOCK_FP8_SUPPORTED:
            return (
                False,
                (
                    "The device compute capability of "
                    f"{compute_capability} is not supported."
                ),
            )
        return True, None

    def input_quant_key(self) -> QuantKey | None:
        """Dynamic group-128 activations can be quantized outside the kernel
        when no token padding is needed. The producer must write the scales
        column-major, `(M, K // 128)` with strides `(1, M)`, as `self.quant_fp8`
        does."""
        key = self.config.activation_quant_key
        if key == kFp8Dynamic128Sym and self.get_output_padding() is None:
            return key
        return None

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                (
                    "Supports only dynamic per token group activation "
                    "quantization with group_shape=(1,128)."
                ),
            )
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        if self.use_flashinfer_small_m:
            return torch.ops.vllm.fp8_block_scaled_mm_small_m(A, B, As, Bs)
        return ops.cutlass_scaled_mm(
            A,
            B.T,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs.T,
        )


def _use_flashinfer_trtllm_fp8_block_gemm(m: int, n: int, k: int) -> bool:
    """Whether FlashInfer's TRT-LLM groupwise FP8 GEMM beats the CUTLASS block
    kernel for an (m, n, k) problem on SM100 and SM103. Both give bit-identical
    output; on SM100 the TRT-LLM kernel is 1.03-1.54x faster when
    m * n <= 2^18 (decode batch sizes) and slower for larger m. It is 1-4%
    slower at k = 7168 with 6144 <= n <= 8192, and it needs k >= 256. The
    same rule was checked on SM103, where it is never slower."""
    if k < 256 or m * n > 1 << 18:
        return False
    return not (k >= 7168 and 6144 <= n <= 8192)


def _fp8_block_scaled_mm_small_m(
    A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor
) -> torch.Tensor:
    # Decided per call so that every CUDA graph size takes the faster kernel.
    m, k = A.shape
    if (
        m > 0
        and (m == 1 or As.stride(0) == 1)
        and Bs.is_contiguous()
        and _use_flashinfer_trtllm_fp8_block_gemm(m, B.shape[0], k)
    ):
        # As is (m, k // 128) with stride (1, m) and Bs.T is (k // 128,
        # n // 128) with stride (1, k // 128): the layouts the kernel expects.
        return flashinfer_gemm_fp8_nt_groupwise(
            A, B, As, Bs.T, out_dtype=torch.bfloat16, backend="trtllm"
        )
    return ops.cutlass_scaled_mm(
        A, B.T, out_dtype=torch.bfloat16, scale_a=As, scale_b=Bs.T
    )


def _fp8_block_scaled_mm_small_m_fake(
    A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor
) -> torch.Tensor:
    return A.new_empty((A.shape[0], B.shape[0]), dtype=torch.bfloat16)


direct_register_custom_op(
    "fp8_block_scaled_mm_small_m",
    _fp8_block_scaled_mm_small_m,
    fake_impl=_fp8_block_scaled_mm_small_m_fake,
)


def cutlass_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return ops.cutlass_scaled_mm(
        A,
        B.T,
        out_dtype=output_dtype,
        scale_a=As,
        scale_b=Bs.T,
    )
