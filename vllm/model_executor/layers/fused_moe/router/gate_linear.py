# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.nn.parameter import Parameter

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms import current_platform
from vllm.utils.hpc import (
    has_hpc_bf16xfp32_gemm,
    hpc_gemm_bf16xfp32,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    """MoE gate linear layer with multi-tier GEMM dispatch:

    1. cuteDSL ll_bf16_gemm (SM90+, M<=16, bf16 in, fp32 out,
       K divisible by 8)
    2. DSV3 specialized kernel (SM90+, M<=16, H=7168 E=256/384, H=6144 E=256)
    3. fp32 specialized kernel  (SM90+, bf16/fp32 in, fp32 out, M<=32,
       (H, E) in {(3072, 256), (6144, 128), (6144, 256)})
    4. experimental bf16x3 CuteDSL kernel (opt-in, SM100, bf16 in, fp32 weight)
    5. cuBLAS bf16×bf16→fp32 (SM90+ + bf16 weight + fp32 out_dtype)
    6. F.linear via ReplicatedLinear (ultimate fallback)

    The ``out_dtype`` attribute is mutable and can be set after init
    (e.g. when the required dtype depends on the expert quantization
    method which is only known later).
    """

    # Dimensions supported by the DSV3 specialized kernel.
    # Valid (hidden_size, num_experts) combinations:
    #   (7168, 256) -> DeepSeek-V3,  (7168, 384) -> Kimi-K2,
    #   (6144, 256) -> GLM-5
    DSV3_SUPPORTED_NUM_EXPERTS = [256, 384]
    DSV3_SUPPORTED_HIDDEN_SIZES = [7168, 6144]
    # num_experts=384 is only instantiated for hidden_size=7168.
    DSV3_UNSUPPORTED_SHAPES = {(6144, 384)}

    # (hidden_size, num_experts) pairs with an instantiated fp32 kernel:
    #   (3072, 256) -> MiniMax-M2/M2.5,  (6144, 128) -> MiniMax-M3
    FP32_SUPPORTED_SHAPES = {(3072, 256), (6144, 128), (6144, 256)}
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

        # DSV3 specialized kernel eligibility (SM90+, exact dims)
        self.allow_specialized_router_gemm = can_use_specialized_kernels
        self.allow_dsv3_router_gemm = (
            self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and output_size in self.DSV3_SUPPORTED_NUM_EXPERTS
            and input_size in self.DSV3_SUPPORTED_HIDDEN_SIZES
            and (input_size, output_size) not in self.DSV3_UNSUPPORTED_SHAPES
        )
        # See https://github.com/vllm-project/vllm/pull/44217
        # for more details.
        self._dsv3_max_batch = 16 if is_hopper else 8

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

        # cuBLAS bf16→fp32 eligibility
        self.allow_cublas_router_gemm = (
            self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

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

        self.allow_hpc_router_gemm = (
            envs.VLLM_ENABLE_HPC_ROUTER_GEMM
            and not bias
            and self.weight.dtype == torch.float32
            and current_platform.is_cuda()
            and is_hopper
            and input_size % 8 == 0
            and output_size % 64 == 0
            and has_hpc_bf16xfp32_gemm()
            and out_dtype == torch.float32
        )

        if self.allow_hpc_router_gemm:
            logger.info_once("Enabled HPC BF16xFP32 router GEMM.")

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
            and self.allow_specialized_router_gemm
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16

        # out_dtype may start as None -> recompute eligibility here
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

        # Tier 2: DSV3 specialized kernel (fallback for when cuteDSL unavailable)
        if self.allow_dsv3_router_gemm and x.shape[0] <= self._dsv3_max_batch:
            output = ops.dsv3_router_gemm(
                hidden_states=x,
                router_weight=self.weight,
                output_dtype=self.out_dtype,
            )
            return output, None

        # HPC BF16 activation x FP32 router weight.
        if (
            self.allow_hpc_router_gemm
            and x.ndim == 2
            and x.dtype == torch.bfloat16
            and x.shape[0] > _FP32_ROUTER_GEMM_MAX_TOKENS
        ):
            output = torch.ops.vllm.hpc_gemm_bf16xfp32_dispatch(
                x,
                self.weight,
            )
            return output, None

        # Tier 3: fp32 specialized kernel (H=3072, E=256, M<=32)
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

        # Tier 4: experimental bf16x3 CuteDSL kernel for fp32 router weights
        if self.allow_bf16x3_router_gemm and x.dtype == torch.bfloat16:
            from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (  # noqa: E501
                bf16x3_router_gemm,
            )

            output = bf16x3_router_gemm(x, self.weight)
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


_FP32_ROUTER_GEMM_MAX_TOKENS = GateLinear.FP32_MAX_TOKENS


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


_HPC_GEMM_WEIGHT_CACHE_ATTR = "_vllm_bf16xfp32_weight_cache"
# The HPC-Ops bf16xfp32 GEMM consumes the fp32 weight decomposed into two
# bf16 halves: w_high = w.bf16 and w_low = ((w - w_high) / scale).bf16 with
# scale = 1/256, so that w ~= w_high + scale * w_low.
_HPC_GEMM_WEIGHT_SCALE = 1.0 / 256.0


def _get_bf16xfp32_weight_split(
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the fp32 weight for the HPC-Ops kernel and cache the result
    (plus the split-K flag workspace, which the kernel leaves zeroed) on the
    weight tensor."""
    import hpc

    cache_key = (
        y.data_ptr(),
        y._version,
        tuple(y.shape),
        tuple(y.stride()),
        y.device.index,
        y.dtype,
    )
    cache = getattr(y, _HPC_GEMM_WEIGHT_CACHE_ATTR, None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2], cache[3]

    with torch.no_grad():
        w_high = y.to(torch.bfloat16)
        w_low = ((y - w_high.float()) / _HPC_GEMM_WEIGHT_SCALE).to(torch.bfloat16)
    split_flag = hpc.get_gemm_bf16xfp32_workspace(y.shape[0])
    setattr(y, _HPC_GEMM_WEIGHT_CACHE_ATTR, (cache_key, w_high, w_low, split_flag))
    return w_high, w_low, split_flag


def _can_use_hpc_gemm_bf16xfp32(
    x: torch.Tensor, weight: torch.Tensor, *, min_m: int = 32
) -> bool:
    if x.dim() != 2 or weight.dim() != 2 or x.shape[1] != weight.shape[1]:
        return False
    if x.shape[0] < min_m:
        return False
    if not (x.is_cuda and weight.is_cuda):
        return False
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        return False
    if not (x.is_contiguous() and weight.is_contiguous()):
        return False
    if weight.shape[0] % 64 != 0:
        return False
    return has_hpc_bf16xfp32_gemm()


def hpc_gemm_bf16xfp32_dispatch_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if not _can_use_hpc_gemm_bf16xfp32(x, weight):
        return torch.nn.functional.linear(x.float(), weight)

    weight_high, weight_low, split_flag = _get_bf16xfp32_weight_split(weight)
    return hpc_gemm_bf16xfp32(
        x=x,
        weight_high=weight_high,
        weight_low=weight_low,
        scale=_HPC_GEMM_WEIGHT_SCALE,
        split_flag=split_flag,
    )


def hpc_gemm_bf16xfp32_dispatch_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty(
        (x.shape[0], weight.shape[0]),
        dtype=torch.float32,
    )


direct_register_custom_op(
    op_name="hpc_gemm_bf16xfp32_dispatch",
    op_func=hpc_gemm_bf16xfp32_dispatch_impl,
    fake_impl=hpc_gemm_bf16xfp32_dispatch_fake,
)
