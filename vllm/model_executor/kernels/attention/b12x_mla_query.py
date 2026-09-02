# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X BF16 MLA query projection and assembly."""

from collections.abc import Iterable

import torch

from vllm.utils.b12x import get_b12x_mla_query_projection
from vllm.utils.torch_utils import direct_register_custom_op


def can_implement_bf16_mla_query(
    *,
    num_heads: int,
    max_m: int,
    nope_dim: int,
    latent_dim: int,
    output_dtype: torch.dtype,
    device: torch.device,
) -> bool:
    module = get_b12x_mla_query_projection()
    return bool(
        module is not None
        and module.can_implement(
            num_heads=num_heads,
            max_m=max_m,
            nope_dim=nope_dim,
            latent_dim=latent_dim,
            output_dtype=output_dtype,
            weight_format="bf16",
            device=device,
        )
    )


def _b12x_bf16_mla_query_impl(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    output: torch.Tensor,
) -> None:
    module = get_b12x_mla_query_projection()
    if module is None:
        raise ImportError("b12x.gemm.mla_query_projection is not available")
    module.run(q_nope, weight, q_pe, output)


def _b12x_bf16_mla_query_fake(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    output: torch.Tensor,
) -> None:
    del q_nope, weight, q_pe, output


direct_register_custom_op(
    op_name="b12x_bf16_mla_query",
    op_func=_b12x_bf16_mla_query_impl,
    mutates_args=["output"],
    fake_impl=_b12x_bf16_mla_query_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def run_bf16_mla_query(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    q_pe: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    torch.ops.vllm.b12x_bf16_mla_query(q_nope, weight, q_pe, output)
    return output


def prewarm_bf16_mla_query(
    weight: torch.Tensor,
    m_values: Iterable[int],
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> int:
    module = get_b12x_mla_query_projection()
    if module is None:
        return 0
    return int(module.prewarm(weight, m_values, output_dtype=output_dtype))


__all__ = [
    "can_implement_bf16_mla_query",
    "prewarm_bf16_mla_query",
    "run_bf16_mla_query",
]
