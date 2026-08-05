# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import (
    CUDAGraphMode,
    get_current_vllm_config,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    should_auto_disable_deep_gemm,
    should_use_deepgemm_for_fp8_linear,
)
from vllm.utils.import_utils import (
    has_deep_gemm,
    has_helion,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from .cutlass import CutlassFP8ScaledMMLinearKernel, CutlassInt8ScaledMMLinearKernel
from .deep_gemm import DeepGemmFp8BlockScaledMMKernel, fp8_gemm_nt
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

# TODO(xiaohongchen1991):
# Currently, Helion linear backend is only supported on Hopper.
# The same threshold may not apply for other hardware types.
# Need to generalize this threshold later on when supporting more hardwares.
HELION_SCALED_MM_MAX_NUM_TOKENS = 32
HELION_BLOCK_SCALED_MM_MAX_NUM_TOKENS = 24


def _hybrid_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    helion_max_num_tokens: int,
) -> torch.Tensor:
    """Compute a 2D scaled_mm ``[M, K] @ [K, N] -> [M, N]``.

    Dispatches to the Helion / CUTLASS hybrid kernel when ``B`` is 16-aligned,
    otherwise falls back to ``triton_scaled_mm``.
    """
    cutlass_compatible_b = B.shape[0] % 16 == 0 and B.shape[1] % 16 == 0
    if not cutlass_compatible_b:
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
            triton_scaled_mm,
        )

        return triton_scaled_mm(A, B, As, Bs, out_dtype, bias)

    output = torch.empty((A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device)
    torch.ops._C.helion_cutlass_hybrid_scaled_mm(
        output, A, B, As, Bs, bias, helion_max_num_tokens
    )
    return output


class HelionFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """
    Hybrid Helion / Cutlass FP8 scaled_mm kernel

    Dispatches between Helion and CUTLASS based on the input batch size (M):
    - Small batches: Use Helion.
    - Large batches: use CUTLASS.

    Restricting Helion to small batches:
    - reduces autotuning time and config space
    - avoids requiring large max_cudagraph_capture_size to cover all batch sizes
    - focuses Helion on the batch sizes where it provides the most benefit.
    """

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)
        self.fallback: CutlassFP8ScaledMMLinearKernel = CutlassFP8ScaledMMLinearKernel(
            c, layer_param_names
        )
        vllm_config = get_current_vllm_config().compilation_config
        self.helion_max_num_tokens = min(
            vllm_config.max_cudagraph_capture_size,
            HELION_SCALED_MM_MAX_NUM_TOKENS,
        )

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not has_helion():
            return False, "requires helion to be installed"

        if not current_platform.is_cuda():
            return False, "requires CUDA"

        # TODO(xiaohongchen1991): support blackwell hardwares when CuteDSL
        # backend supported by Helion
        if not current_platform.is_device_capability(90):
            return (
                False,
                "is only supported on SM90 (Hopper) architecture",
            )

        if envs.VLLM_BATCH_INVARIANT:
            return False, "is not supported for batch invariant execution."

        # Require CUDA graph capture and reply for Helion kernel
        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if (
            compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            or compilation_config.max_cudagraph_capture_size == 0
        ):
            return False, "requires enabling CUDA Graph mode"

        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        # can_implement is called after is_supported.
        # Place Helion kernel checks here to avoid unnecessary kernel registration
        from vllm.kernels.helion.ops.scaled_mm import scaled_mm

        # Helion kernel is disabled if no config exists for the hardware used.
        if scaled_mm._disabled:
            return False, f"op is disabled. {scaled_mm._disabled_reason}"

        # Enable Helion only if pre-tuned configs cover all input shapes.
        # (K, N) come from the weight; M ranges over the small-batch sizes
        # Helion is dispatched for: cudagraph_capture_sizes capped by the
        # Helion max (larger M falls back to CUTLASS).
        N, K = c.weight_shape
        in_dtype = str(current_platform.fp8_dtype())
        compilation_config = get_current_vllm_config().compilation_config
        helion_max_num_tokens = min(
            compilation_config.max_cudagraph_capture_size,
            HELION_SCALED_MM_MAX_NUM_TOKENS,
        )
        m_sizes = [
            m
            for m in compilation_config.cudagraph_capture_sizes
            if m <= helion_max_num_tokens
        ]

        # Allow forcing Helion even without exact config coverage; pick_config
        # falls back to the closest available config.
        if envs.VLLM_HELION_LINEAR_SKIP_CONFIG_CHECK:
            return True, None

        configs = scaled_mm.get_configured_op().configs
        tuned = {
            (key["K"], key["N"], key["M"])
            for key in configs
            if not key.is_default()
            and {"K", "N", "M", "in_dtype"} <= key.keys()
            and key["in_dtype"] == in_dtype
        }
        missing = [M for M in m_sizes if (K, N, M) not in tuned]
        # TODO(xiaohongchen1991): update the message to include the
        # instruction to run autotune for the missing shapes.
        if missing:
            return (
                False,
                f"has no pre-tuned config for weight shape (K={K}, N={N}) "
                f"with in_dtype={in_dtype} at M={missing}. Set "
                f"VLLM_HELION_LINEAR_SKIP_CONFIG_CHECK=1 to run anyway with "
                f"the closest available config",
            )

        return True, None

    def input_quant_key(self) -> QuantKey | None:
        """Only static per-tensor activation quantization is supported for external
        quantization."""
        if self.config.activation_quant_key == kFp8StaticTensorSym:
            return kFp8StaticTensorSym
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.fallback.process_weights_after_loading(layer)

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
        output_size = self.fallback.logical_output_size
        assert output_size is not None
        pad_k = padded_k - A.shape[1]
        pad_n = padded_n - output_size

        if pad_k > 0:
            A = self.fallback._pad_to_alignment(A, dim=1, alignment=16)
        if pad_n > 0 and bias is not None:
            bias = self.fallback._pad_to_alignment(bias, dim=0, alignment=16)

        output = _hybrid_scaled_mm(
            A, B, As, Bs, bias, out_dtype, self.helion_max_num_tokens
        )

        if pad_n > 0:
            output = output[..., :output_size].contiguous()

        return output.view(*output_shape[:-1], output_size)


class HelionINT8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    """
    Hybrid Helion / Cutlass INT8 scaled_mm kernel

    Dispatches between Helion and CUTLASS based on the input batch size (M) for
    the symmetric case:
    - Small batches: use Helion.
    - Large batches: use CUTLASS.

    The asymmetric (azp) case always uses CUTLASS.
    """

    def __init__(
        self, c: Int8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)
        self.fallback: CutlassInt8ScaledMMLinearKernel = (
            CutlassInt8ScaledMMLinearKernel(c, layer_param_names)
        )
        compilation_config = get_current_vllm_config().compilation_config
        self.helion_max_num_tokens = min(
            compilation_config.max_cudagraph_capture_size,
            HELION_SCALED_MM_MAX_NUM_TOKENS,
        )

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not has_helion():
            return False, "requires helion to be installed"

        if not current_platform.is_cuda():
            return False, "requires CUDA"

        # TODO(xiaohongchen1991): support blackwell hardwares when CuteDSL
        # backend supported by Helion
        if not current_platform.is_device_capability(90):
            return (
                False,
                "is only supported on SM90 (Hopper) architecture",
            )

        if envs.VLLM_BATCH_INVARIANT:
            return False, "is not supported for batch invariant execution."

        # Require CUDA graph capture and replay for Helion kernel
        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if (
            compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            or compilation_config.max_cudagraph_capture_size == 0
        ):
            return False, "requires enabling CUDA Graph mode"

        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        # can_implement is called after is_supported.
        # Place Helion kernel checks here to avoid unnecessary kernel registration
        from vllm.kernels.helion.ops.scaled_mm import scaled_mm

        # Helion kernel is disabled if no config exists for the hardware used.
        if scaled_mm._disabled:
            return False, f"op is disabled. {scaled_mm._disabled_reason}"

        # Allow forcing Helion even without exact config coverage; pick_config
        # falls back to the closest available config.
        if envs.VLLM_HELION_LINEAR_SKIP_CONFIG_CHECK:
            return True, None

        # Enable Helion only if pre-tuned configs cover all input shapes.
        # (K, N) come from the weight; M ranges over the small-batch sizes
        # Helion is dispatched for: cudagraph_capture_sizes capped by the
        # Helion max (larger M falls back to CUTLASS).
        N, K = c.weight_shape
        in_dtype = str(torch.int8)
        compilation_config = get_current_vllm_config().compilation_config
        helion_max_num_tokens = min(
            compilation_config.max_cudagraph_capture_size,
            HELION_SCALED_MM_MAX_NUM_TOKENS,
        )
        m_sizes = [
            m
            for m in compilation_config.cudagraph_capture_sizes
            if m <= helion_max_num_tokens
        ]

        configs = scaled_mm.get_configured_op().configs
        tuned = {
            (key["K"], key["N"], key["M"])
            for key in configs
            if not key.is_default()
            and {"K", "N", "M", "in_dtype"} <= key.keys()
            and key["in_dtype"] == in_dtype
        }
        missing = [M for M in m_sizes if (K, N, M) not in tuned]
        # TODO(xiaohongchen1991): update the message to include the
        # instruction to run autotune for the missing shapes.
        if missing:
            return (
                False,
                f"has no pre-tuned config for weight shape (K={K}, N={N}) "
                f"with in_dtype={in_dtype} at M={missing}. Set "
                f"VLLM_HELION_LINEAR_SKIP_CONFIG_CHECK=1 to run anyway with "
                f"the closest available config",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.fallback.process_weights_after_loading(layer)

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

        target_shape = (*x_q.shape[:-1], w_q.shape[1])
        output = _hybrid_scaled_mm(
            x_q.view(-1, x_q.shape[-1]),
            w_q,
            x_s,
            w_s,
            bias,
            x.dtype,
            self.helion_max_num_tokens,
        )
        return output.view(*target_shape)


class HelionFP8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """
    Conditional Helion / DeepGEMM FP8 block-scaled GEMM.

    Dispatches between two kernels based on input batch size:
    - Small batches (M < 32): use Helion.
    - Large batches (M >= 32): use DeepGEMM.
    """

    apply_input_quant: ClassVar[bool] = False

    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        self.fallback: DeepGemmFp8BlockScaledMMKernel = DeepGemmFp8BlockScaledMMKernel(
            config
        )
        vllm_config = get_current_vllm_config().compilation_config
        self.helion_max_num_tokens = min(
            vllm_config.max_cudagraph_capture_size,
            HELION_BLOCK_SCALED_MM_MAX_NUM_TOKENS,
        )

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not has_helion():
            return False, "requires helion to be installed"

        if not current_platform.is_cuda():
            return False, "requires CUDA"

        if not (envs.VLLM_USE_DEEP_GEMM and has_deep_gemm()):
            return False, "requires DeepGEMM"

        if not current_platform.is_device_capability(90):
            return (
                False,
                "is only supported on SM90 (Hopper) architecture",
            )

        if envs.VLLM_BATCH_INVARIANT:
            return False, "is not supported for batch invariant execution."

        # Require CUDA graph capture and reply for Helion kernel
        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if (
            compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            or compilation_config.max_cudagraph_capture_size == 0
        ):
            return False, "requires enabling CUDA Graph mode"

        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        if config.out_dtype != torch.bfloat16:
            return (False, "Supports only output dtype of bfloat16")

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )
        model_config = get_current_vllm_config().model_config

        if model_config is None:
            return False, "Model configuration is required."

        model_type = getattr(model_config.hf_text_config, "model_type", None)
        if should_auto_disable_deep_gemm(model_type):
            return False, f"DeepGEMM is not supported for model {model_type}"

        if not should_use_deepgemm_for_fp8_linear(
            config.out_dtype, config.weight_shape
        ):
            return False, "The provided metadata is not supported."

        # can_implement is called after is_supported.
        # Place Helion kernel checks here to avoid unnecessary kernel registration
        from vllm.kernels.helion.ops.block_scaled_mm import block_scaled_mm

        # Helion kernel is disabled if no config exists for the hardware used.
        if block_scaled_mm._disabled:
            return False, f"GEMM op is disabled. {block_scaled_mm._disabled_reason}"

        # Allow forcing Helion even without exact config coverage; pick_config
        # falls back to the closest available config.
        if envs.VLLM_HELION_LINEAR_SKIP_CONFIG_CHECK:
            return True, None

        # Enable Helion only if pre-tuned configs cover all input shapes.
        # (K, N) come from the weight; M ranges over the small-batch sizes
        # Helion is dispatched for: cudagraph_capture_sizes capped by the
        # Helion max (larger M falls back to DeepGEMM).
        N, K = config.weight_shape
        in_dtype = str(current_platform.fp8_dtype())
        compilation_config = get_current_vllm_config().compilation_config
        helion_max_num_tokens = min(
            compilation_config.max_cudagraph_capture_size,
            HELION_BLOCK_SCALED_MM_MAX_NUM_TOKENS,
        )
        m_sizes = [
            m
            for m in compilation_config.cudagraph_capture_sizes
            if m <= helion_max_num_tokens
        ]

        configs = block_scaled_mm.get_configured_op().configs
        tuned = {
            (key["K"], key["N"], key["M"])
            for key in configs
            if not key.is_default()
            and {"K", "N", "M", "in_dtype"} <= key.keys()
            and key["in_dtype"] == in_dtype
        }
        missing = [M for M in m_sizes if (K, N, M) not in tuned]
        # TODO(xiaohongchen1991): update the message to include the
        # instruction to run autotune for the missing shapes.
        if missing:
            return (
                False,
                f"has no pre-tuned config for weight shape (K={K}, N={N}) "
                f"with in_dtype={in_dtype} at M={missing}. Set "
                f"VLLM_HELION_LINEAR_SKIP_CONFIG_CHECK=1 to run anyway with "
                f"the closest available config",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # DeepGEMM need post-processing; both kernels share the same
        # parameter tensor layout so processing once is sufficient.
        self.fallback.process_weights_after_loading(layer)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        group_size = self.weight_group_shape.col
        use_deep_gemm_e8m0 = self.fallback.use_deep_gemm_e8m0
        A, As = per_token_group_quant_fp8(
            A,
            group_size=group_size,
            column_major_scales=True,
            use_ue8m0=use_deep_gemm_e8m0,
        )

        out = torch.empty(
            (A.shape[0], B.shape[0]),
            dtype=torch.bfloat16,
            device=A.device,
        )

        torch.ops.vllm.hybrid_block_scaled_mm(
            out, A, B, As, Bs, use_deep_gemm_e8m0, self.helion_max_num_tokens
        )

        return out


def _hybrid_block_scaled_mm_impl(
    out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    use_deep_gemm_e8m0: bool,
    helion_max_num_tokens: int,
) -> None:
    if A.shape[0] > helion_max_num_tokens:
        fp8_gemm_nt(
            (A, As),
            (B, Bs),
            out,
            is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
        )
    else:
        from vllm.kernels.helion.ops.block_scaled_mm import block_scaled_mm

        block_scaled_mm(out, A, B.T, As, Bs.T)


def _hybrid_block_scaled_mm_fake(
    out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    use_deep_gemm_e8m0: bool,
    helion_max_num_tokens: int,
) -> None:
    return


direct_register_custom_op(
    "hybrid_block_scaled_mm",
    _hybrid_block_scaled_mm_impl,
    fake_impl=_hybrid_block_scaled_mm_fake,
    mutates_args=["out"],
)
