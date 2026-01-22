# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.kernels.input_quant import (
    select_quant_kernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)


# --8<-- [start:quant_fp8]
@CustomOp.register("quant_fp8")
class QuantFP8(CustomOp):
    """
    Quantize input tensor to FP8 (per-tensor, per-token, per-channel, or per-group).
    This CustomOp supports both static and dynamic quantization.
    """

    # --8<-- [end:quant_fp8]

    def __init__(
        self,
        static: bool,
        group_shape: GroupShape,
        num_token_padding: int | None = None,
        column_major_scales: bool = False,
        use_ue8m0: bool | None = None,  # for Torch compile
    ):
        """
        :param static: static or dynamic quantization
        :param group_shape: quantization group shape (PER_TOKEN, PER_TENSOR,
            or arbitrary block size)
        :param num_token_padding: Pad the token dimension of output to this
            size
        :param column_major_scales: For group quantization, output scales in
            column major format
        """
        super().__init__()

        use_ue8m0 = False if use_ue8m0 is None else use_ue8m0

        self.input_quant_kernel = select_quant_kernel(
            static, group_shape, column_major_scales, use_ue8m0, num_token_padding
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, scale, scale_ub)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_quant_kernel.apply(x, scale, scale_ub)
