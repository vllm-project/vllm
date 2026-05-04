# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def apply_mxfp4_flashinfer_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    size_n: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.utils.flashinfer import flashinfer_mm_fp4, flashinfer_mxfp4_quantize

    x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    x_fp4, x_scale = flashinfer_mxfp4_quantize(x)

    dummy_alpha = torch.ones(1, dtype=torch.float32, device=x.device)
    out = flashinfer_mm_fp4(
        x_fp4,
        weight.t(),  # [N, K//2] -> [K//2, N]
        x_scale,
        weight_scale.t(),  # [N, K//32] -> [K//32, N]
        dummy_alpha,
        input.dtype,
        use_8x4_sf_layout=False,
        backend="cute-dsl",
        block_size=32,
        use_nvfp4=False,
    )

    if bias is not None:
        out = out + bias
    return out.view(out_shape)
