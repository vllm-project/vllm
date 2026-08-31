# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.nn.parameter import Parameter

import vllm._custom_ops as ops
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.fused_moe.router import bf16x3_router_gemm_rocm
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    """MoE gate linear layer with multi-tier GEMM dispatch:

    1. cuteDSL ll_bf16_gemm (SM90+, M<=16, bf16 in, fp32 out,
       K divisible by 8)
    2. fp32 specialized kernel  (SM90+, bf16/fp32 in, fp32 out, M<=32,
       (H, E) in {(3072, 256), (6144, 128)})
    3. experimental bf16x3 CuteDSL kernel (opt-in, SM100, bf16 in, fp32 weight)
    4. ROCm bf16x3 router GEMM (gfx950, bf16 in, fp32 weight, fp32 out,
       M>=2048; smaller batches fall through to tier 6)
    5. cuBLAS bf16×bf16→fp32 (SM90+ + bf16 weight + fp32 out_dtype)
    6. F.linear via ReplicatedLinear (ultimate fallback)

    The ``out_dtype`` attribute is mutable and can be set after init
    (e.g. when the required dtype depends on the expert quantization
    method which is only known later).
    """

    # (hidden_size, num_experts) pairs with an instantiated fp32 kernel:
    #   (3072, 256) -> MiniMax-M2/M2.5,  (6144, 128) -> MiniMax-M3
    FP32_SUPPORTED_SHAPES = {(3072, 256), (6144, 128)}
    FP32_MAX_TOKENS = 32

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        is_hopper = current_platform.is_device_capability((9, 0))
        is_blackwell = current_platform.is_device_capability_family(100)
        can_use_specialized_kernels = (
            current_platform.is_cuda() and (is_hopper or is_blackwell) and not bias
        )

        # If fp32 compute is required and no specialized kernel is available,
        # store weights in fp32 so the fallback linear path computes in fp32.
        if force_fp32_compute and not can_use_specialized_kernels:
            params_dtype = torch.float32

        super().__init__(
            input_size,
            output_size,
            bias=bias,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        self.out_dtype = out_dtype

        self.allow_specialized_router_gemm = can_use_specialized_kernels

        # fp32 specialized kernel eligibility (SM90+, exact dims, fp32 weight)
        vllm_config = get_current_vllm_config_or_none()
        enable_bf16x3_router_gemm = (
            vllm_config is not None
            and vllm_config.kernel_config.enable_bf16x3_router_gemm
        )
        self.allow_fp32_router_gemm = (
            not bias
            and self.weight.dtype == torch.float32
            and current_platform.is_cuda()
            and (is_hopper or is_blackwell)
            and (input_size, output_size) in self.FP32_SUPPORTED_SHAPES
        )
        self.allow_bf16x3_router_gemm = (
            not bias
            and self.weight.dtype == torch.float32
            and current_platform.is_cuda()
            and is_blackwell
            and input_size % 8 == 0
            and enable_bf16x3_router_gemm
        )
        if self.allow_bf16x3_router_gemm:
            logger.info_once("Enabled experimental SM100 BF16x3 router GEMM.")

        # Fused bf16 x bf16 -> fp32 GEMM eligibility. torch.mm's out_dtype
        # epilogue folds the fp32 cast into the GEMM, removing the standalone
        # bf16->fp32 copy kernel that otherwise runs before grouped_topk. This is
        # the plain cuBLAS (CUDA) / hipBLASLt (ROCm) out_dtype epilogue, so it
        # applies on any CUDA-alike device (no bias, since torch.mm has no bias
        # term). The specialized-kernel gate above excludes family-120 Blackwell
        # (GB10 / DGX Spark), which this tier still covers. See #49921.
        self._router_gemm_no_bias = not bias
        self._router_gemm_cublas_capable = (
            current_platform.is_cuda() or current_platform.is_rocm()
        ) and self._router_gemm_no_bias
        self.allow_cublas_router_gemm = (
            self._router_gemm_cublas_capable
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

        # ROCm bf16x3 router GEMM eligibility. out_dtype may still be None here
        # and is folded in by set_out_dtype; the split itself is only built
        # once the tier is known to be on (see _GateLinearMethod).
        self._rocm_bf16x3_weight_eligible = (
            self._router_gemm_no_bias
            and self.weight.dtype == torch.float32
            and bf16x3_router_gemm_rocm.platform_supported()
        )
        self.allow_rocm_bf16x3_router_gemm = (
            self._rocm_bf16x3_weight_eligible and self.out_dtype == torch.float32
        )
        # Filled in by process_weights_after_loading. Not persistent: it is
        # derived from `weight` rather than loaded from a checkpoint.
        self.register_buffer("_bf16x3_weight", None, persistent=False)
        if self._rocm_bf16x3_weight_eligible:
            # Safe to swap after create_weights() because
            # UnquantizedLinearMethod holds no per-layer state. Asserted
            # because the swap would silently drop quantization if a
            # quant_config were ever plumbed into this layer.
            assert isinstance(self.quant_method, UnquantizedLinearMethod)
            self.quant_method = _GateLinearMethod()
            logger.info_once("Enabled ROCm BF16x3 router GEMM.")

        # cuteDSL ll_bf16_gemm eligibility. Any dims supported, but SM90+ required bc:
        # 1. PDL support. Both dot-product and split-K kernels.
        # 2. Thread Block Clusters. Split-K kernel for cross-CTA reduction.
        self.allow_ll_bf16_gemm = False
        if can_use_specialized_kernels:
            from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
                is_available,
            )

            self.allow_ll_bf16_gemm = (
                self.weight.dtype == torch.bfloat16
                and self.out_dtype == torch.float32
                and is_available()
            )

    def set_out_dtype(self, out_dtype: torch.dtype) -> None:
        """Set output dtype for the router logits after init.

        Useful when the required dtype depends on the expert quantization
        method which is only known after the gate is constructed.
        """
        if self.out_dtype is not None:
            raise ValueError("out_dtype has already been set")
        self.out_dtype = out_dtype

        if (
            not self.allow_cublas_router_gemm
            and self._router_gemm_cublas_capable
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16

        # out_dtype may start as None -> recompute eligibility here
        self.allow_rocm_bf16x3_router_gemm = (
            self._rocm_bf16x3_weight_eligible and out_dtype == torch.float32
        )

        if self.allow_specialized_router_gemm:
            from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
                is_available,
            )

            self.allow_ll_bf16_gemm = (
                self.weight.dtype == torch.bfloat16
                and out_dtype == torch.float32
                and is_available()
            )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # Tier 1: cuteDSL ll_bf16_gemm (SM90+, any dims)
        if self.allow_ll_bf16_gemm and x.shape[0] <= 16 and x.dtype == torch.bfloat16:
            from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
                ll_bf16_gemm,
            )

            output = ll_bf16_gemm(x, self.weight)
            return output, None

        # Tier 2: fp32 specialized kernel (H=3072, E=256, M<=32)
        # Dispatch is wrapped in a custom op so that torch.compile/CUDA-graph
        # capture does not freeze the runtime num_tokens branch.
        if self.allow_fp32_router_gemm and x.dtype in (
            torch.float32,
            torch.bfloat16,
        ):
            output = torch.ops.vllm.fp32_router_gemm_dispatch(
                x, self.weight, self.allow_bf16x3_router_gemm
            )
            return output, None

        # Tier 3: experimental bf16x3 CuteDSL kernel for fp32 router weights
        if self.allow_bf16x3_router_gemm and x.dtype == torch.bfloat16:
            from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (  # noqa: E501
                bf16x3_router_gemm,
            )

            output = bf16x3_router_gemm(x, self.weight)
            return output, None

        # Tier 4: ROCm bf16x3 router GEMM. As in tier 2, the runtime num_tokens
        # dispatch lives inside a custom op.
        if self.allow_rocm_bf16x3_router_gemm and x.dtype == torch.bfloat16:
            if self._bf16x3_weight is None:
                logger.warning_once(
                    "ROCm BF16x3 router GEMM is enabled for %s but its weight "
                    "split is missing; falling back to fp32.",
                    self.prefix,
                )
            else:
                output = torch.ops.vllm.rocm_bf16x3_router_gemm_dispatch(
                    x, self.weight, self._bf16x3_weight
                )
                return output, None

        # Tier 5: cuBLAS bf16→fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
            output = torch.mm(x, self.weight.T, out_dtype=torch.float32)
            return output, None

        # Tier 6: F.linear (ReplicatedLinear)
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias


class _GateLinearMethod(UnquantizedLinearMethod):
    """UnquantizedLinearMethod plus the ROCm bf16x3 weight split.

    Building the split here rather than lazily in forward() charges it to the
    weights memory pool ahead of KV cache profiling, and rebuilds it on weight
    reload. The latter matters because an in-place ``param.data.copy_()``
    leaves both ``data_ptr()`` and ``_version`` untouched, so a stale split
    cannot be detected from inside forward().

    ``allow_rocm_bf16x3_router_gemm`` is final by the time this runs: the only
    caller of ``set_out_dtype`` does so during model construction.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The base CPU branch may delete layer.weight, so it must run first;
        # nothing breaks today only because platform_supported() is False on
        # CPU.
        super().process_weights_after_loading(layer)
        if not layer.allow_rocm_bf16x3_router_gemm:
            return
        try:
            layer._bf16x3_weight = bf16x3_router_gemm_rocm.split_bf16x3(layer.weight)
        except ValueError as exc:
            layer.allow_rocm_bf16x3_router_gemm = False
            logger.warning_once(
                "Disabling ROCm BF16x3 router GEMM for %s: %s", layer.prefix, exc
            )


_FP32_ROUTER_GEMM_MAX_TOKENS = GateLinear.FP32_MAX_TOKENS


def rocm_bf16x3_router_gemm_dispatch_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_split: torch.Tensor,
) -> torch.Tensor:
    """Run the ROCm bf16x3 router GEMM, or fall back for small batches.

    This must be a custom op because our torch.compile integration does not
    support runtime dispatching on num_tokens: models carrying fp32 router
    weights (MiniMax-M2, HunYuan-V3) call the gate from
    ``@support_torch_compile`` model code and vLLM drops all Dynamo guards, so
    a plain Python branch on ``x.shape[0]`` would be frozen at first trace.
    """
    if bf16x3_router_gemm_rocm.is_supported(x, weight):
        return bf16x3_router_gemm_rocm.bf16x3_router_gemm(x, weight_split)
    return torch.nn.functional.linear(x.float(), weight)


def rocm_bf16x3_router_gemm_dispatch_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_split: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="rocm_bf16x3_router_gemm_dispatch",
    op_func=rocm_bf16x3_router_gemm_dispatch_impl,
    fake_impl=rocm_bf16x3_router_gemm_dispatch_fake,
)


def fp32_router_gemm_dispatch_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    allow_bf16x3_router_gemm: bool,
) -> torch.Tensor:
    """
    Dynamically run fp32 specialized gemm if num_tokens <= FP32_MAX_TOKENS,
    otherwise optionally run the experimental BF16x3 kernel for medium/large
    SM100 router batches, then fall back to F.linear.
    This must be wrapped in a custom op because our torch.compile integration
    does not support runtime dispatching on num_tokens.
    """
    if x.shape[0] <= _FP32_ROUTER_GEMM_MAX_TOKENS:
        return ops.fp32_router_gemm(x, weight)

    if allow_bf16x3_router_gemm and x.dtype == torch.bfloat16:
        from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (  # noqa: E501
            bf16x3_router_gemm,
        )

        return bf16x3_router_gemm(x, weight)

    return torch.nn.functional.linear(x.float(), weight)


def fp32_router_gemm_dispatch_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    allow_bf16x3_router_gemm: bool,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="fp32_router_gemm_dispatch",
    op_func=fp32_router_gemm_dispatch_impl,
    fake_impl=fp32_router_gemm_dispatch_fake,
)
