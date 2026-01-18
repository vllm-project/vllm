# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gptq_triton import (
    GPTQ_TRITON_SUPPORTED_GROUP_SIZES,
    gptq_gemm_triton,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import HAS_TRITON


class TritonLinearKernel(MPLinearKernel):
    """Triton kernel for GPTQ W4A16 on compute capability >= 7.0 (Volta and above)."""

    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]

    # Type annotations for inherited attributes (helps mypy with --follow-imports skip)
    config: MPLinearLayerConfig
    w_q_name: str
    w_s_name: str
    w_zp_name: str | None
    w_gidx_name: str | None

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return (
                False,
                "GPTQ Triton is only supported on CUDA and ROCm",
            )
        capability_tuple = current_platform.get_device_capability()
        if capability_tuple is None or capability_tuple.to_int() < 70:
            return False, "GPTQ Triton requires compute capability >= 7.0"
        if not HAS_TRITON:
            return False, "Triton is not available"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                "GPTQ Triton only supports GPTQ W4A16 "
                f"(weights {cls.SUPPORTED_QUANT_TYPES})",
            )
        if c.zero_points:
            return False, "GPTQ Triton only supports symmetric GPTQ W4A16"
        if c.act_type != torch.float16:
            return False, "GPTQ Triton only supports float16 activations (W4A16)"
        if c.out_type is not None and c.out_type != torch.float16:
            return False, "GPTQ Triton only supports float16 outputs (W4A16)"
        if c.has_g_idx and c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return (
                False,
                "Act reordering not supported when input features "
                "are partitioned across devices",
            )
        if c.partition_weight_shape[0] % 8 != 0:
            return (
                False,
                "Input features must be divisible by 8 for GPTQ 4-bit packing",
            )
        if c.group_size != -1 and c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide "
                f"the number of input features ({c.full_weight_shape[0]})",
            )
        if c.group_size not in GPTQ_TRITON_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Unsupported group_size {c.group_size} for GPTQ Triton",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        if c.has_g_idx:

            def transform_w_g_idx(x):
                # Convert group indices to permutation array
                return torch.argsort(x).to(torch.int)

            self._transform_param(layer, self.w_gidx_name, transform_w_g_idx)
        else:
            self.w_gidx_name = "weight_g_idx"
            device = getattr(layer, self.w_q_name).device
            empty_g_idx = torch.nn.Parameter(
                torch.empty((0,), dtype=torch.int, device=device), requires_grad=False
            )
            setattr(layer, self.w_gidx_name, empty_g_idx)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            assert self.w_gidx_name is not None
            g_idx = getattr(layer, self.w_gidx_name)

            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            if g_idx.numel() > 0:
                ops.gptq_shuffle(x_cont, g_idx, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        c = self.config
        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, _, w_g_idx = self._get_weight_params(layer)

        if w_g_idx is not None and w_g_idx.numel() > 0:
            x_2d = x_2d[:, w_g_idx.to(torch.long)]

        output = gptq_gemm_triton(x_2d, w_q, w_s, c.group_size, split_k_iters=1)

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
