# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import torch
from torch import nn

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

# Max number of tokens supported by the Lamport fused allreduce+RMSNorm kernel.
# Larger batches fall back to the eager allreduce + RMSNorm path.
MINIMAX_QK_NORM_MAX_TOKEN_NUM = 2048

_MINIMAX_FUSED_AR_RMS_QK = getattr(torch.ops._C, "minimax_allreduce_rms_qk", None)


@torch.compile(backend=current_platform.simple_compile_backend, dynamic=True)
def _minimax_qk_norm_fallback(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
    tp_rank: int,
    tp_world: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    q_var = q.pow(2).mean(dim=-1, keepdim=True)
    k_var = k.pow(2).mean(dim=-1, keepdim=True)
    if tp_world > 1:
        qk_var = torch.cat([q_var, k_var], dim=-1)
        qk_var = tensor_model_parallel_all_reduce(qk_var) / tp_world
        q_var, k_var = qk_var.chunk(2, dim=-1)
    q = q * torch.rsqrt(q_var + eps) * q_weight
    k = k * torch.rsqrt(k_var + eps) * k_weight
    return q.to(orig_dtype), k.to(orig_dtype)


def _minimax_qk_norm_fusion(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
    tp_rank: int,
    tp_world: int,
    eps: float,
    workspace: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert qkv.ndim == 2
    num_tokens = qkv.shape[0]
    if (
        workspace is not None
        and tp_world > 1
        and num_tokens <= MINIMAX_QK_NORM_MAX_TOKEN_NUM
        and _MINIMAX_FUSED_AR_RMS_QK is not None
    ):
        return _MINIMAX_FUSED_AR_RMS_QK(
            qkv,
            q_weight,
            k_weight,
            workspace,
            q_size,
            kv_size,
            tp_rank,
            tp_world,
            eps,
        )
    return _minimax_qk_norm_fallback(
        qkv, q_weight, k_weight, q_size, kv_size, tp_rank, tp_world, eps
    )


def _minimax_qk_norm_fusion_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
    tp_rank: int,
    tp_world: int,
    eps: float,
    workspace: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert qkv.ndim == 2
    num_tokens = qkv.shape[0]
    return (
        torch.empty([num_tokens, q_size], dtype=qkv.dtype, device=qkv.device),
        torch.empty([num_tokens, kv_size], dtype=qkv.dtype, device=qkv.device),
    )


direct_register_custom_op(
    op_name="minimax_qk_norm_fusion",
    op_func=_minimax_qk_norm_fusion,
    fake_impl=_minimax_qk_norm_fusion_fake,
    mutates_args=[],
)


@CustomOp.register("minimax_text01_rmsnorm_tp")
class MiniMaxText01RMSNormTP(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        weight_shard_world_size: int | None = None,
        weight_shard_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight_shard_world = weight_shard_world_size or self.tp_world
        self.weight_shard_rank = (
            self.tp_rank if weight_shard_rank is None else weight_shard_rank
        )

        self.weight = nn.Parameter(torch.ones(hidden_size // self.weight_shard_world))
        self.weight.weight_loader = partial(
            self.weight_loader,
            shard_world_size=self.weight_shard_world,
            shard_rank=self.weight_shard_rank,
        )
        self.variance_epsilon = eps

        self.workspace = None
        if _MINIMAX_FUSED_AR_RMS_QK is not None and self.tp_world > 1:
            from .lamport_workspace import (
                get_allreduce_workspace,
            )

            self.workspace = get_allreduce_workspace(
                rank=self.tp_rank,
                world_size=self.tp_world,
                max_tokens=MINIMAX_QK_NORM_MAX_TOKEN_NUM,
                process_group=get_tp_group().cpu_group,
            )

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_world_size: int | None = None,
        shard_rank: int | None = None,
    ) -> None:
        if shard_world_size is None:
            shard_world_size = get_tensor_model_parallel_world_size()
        if shard_rank is None:
            shard_rank = get_tensor_model_parallel_rank()

        shard_size = loaded_weight.shape[0] // shard_world_size
        shard = slice(shard_rank * shard_size, (shard_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard])

    def _forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
        if self.tp_world > 1:
            variance = tensor_model_parallel_all_reduce(variance) / self.tp_world
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        return x

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert residual is None, "RMSNorm does not support residual connection."
        return self._forward(x)

    @staticmethod
    def forward_qk(
        q_norm: "MiniMaxText01RMSNormTP",
        k_norm: "MiniMaxText01RMSNormTP",
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = q.dtype
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        q_var = q.pow(2).mean(dim=-1, keepdim=True)
        k_var = k.pow(2).mean(dim=-1, keepdim=True)
        if q_norm.tp_world > 1:
            qk_var = torch.cat([q_var, k_var], dim=-1)
            qk_var = tensor_model_parallel_all_reduce(qk_var) / q_norm.tp_world
            q_var, k_var = qk_var.chunk(2, dim=-1)
        q = q * torch.rsqrt(q_var + q_norm.variance_epsilon) * q_norm.weight
        k = k * torch.rsqrt(k_var + k_norm.variance_epsilon) * k_norm.weight
        q = q.to(orig_dtype)
        k = k.to(orig_dtype)
        return q, k

    @staticmethod
    def forward_qkv(
        q_norm: "MiniMaxText01RMSNormTP",
        k_norm: "MiniMaxText01RMSNormTP",
        qkv: torch.Tensor,
        q_size: int,
        kv_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert qkv.ndim == 2
        assert q_norm.variance_epsilon == k_norm.variance_epsilon
        q, k = torch.ops.vllm.minimax_qk_norm_fusion(
            qkv,
            q_norm.weight,
            k_norm.weight,
            q_size,
            kv_size,
            q_norm.tp_rank,
            q_norm.tp_world,
            q_norm.variance_epsilon,
            q_norm.workspace,
        )
        _, _, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        return q, k, v
