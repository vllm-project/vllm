# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from torch import nn

import vllm._custom_ops  # noqa: F401 — registers fake-tensor impls for torch.compile
from vllm.config import get_current_vllm_config
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


class MiniMaxText01RMSNormTP(CustomOp):
    name = "MiniMaxText01RMSNormTP"

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight = nn.Parameter(torch.ones(int(hidden_size / self.tp_world)))
        self.weight.weight_loader = self.weight_loader
        self.variance_epsilon = eps

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        tp_world = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = loaded_weight.shape[0] // tp_world
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
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


class MiniMaxQKNormWrapper(nn.Module):
    def __init__(
        self,
        q_norm: MiniMaxText01RMSNormTP,
        k_norm: MiniMaxText01RMSNormTP,
        q_size: int,
        kv_size: int,
        max_tokens: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.q_size = q_size
        self.kv_size = kv_size
        self.max_token = max_tokens
        self.layer_name = prefix
        self.tp_world = self.q_norm.tp_world
        self.tp_rank = self.q_norm.tp_rank
        self._ar_workspace: torch.Tensor | None = None
        self._lamport_max_token: int = 0

        if current_platform.is_cuda() and self.tp_world > 1:
            self._lamport_max_token = min(2048, max_tokens)
            from .lamport_workspace import get_allreduce_workspace

            self._ar_workspace = get_allreduce_workspace(
                self.tp_rank,
                self.tp_world,
                max_tokens=self._lamport_max_token,
                process_group=get_tp_group().cpu_group,
            )

        compilation_config = get_current_vllm_config().compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

    def forward(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.minimax_rms_norm(
            qkv, self.q_size, self.kv_size, self.layer_name
        )

    def _fused_kernel(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.q_norm.variance_epsilon == self.k_norm.variance_epsilon
        torch.ops._C.minimax_allreduce_rms_qk(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self._ar_workspace,
            self.q_size,
            self.kv_size,
            self.tp_rank,
            self.tp_world,
            self.q_norm.variance_epsilon,
        )
        return qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    def _native_ops(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.contiguous()
        k = k.contiguous()
        orig_dtype = q.dtype
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        q_var = q.pow(2).mean(dim=-1, keepdim=True)
        k_var = k.pow(2).mean(dim=-1, keepdim=True)
        if self.tp_world > 1:
            qk_var = torch.cat([q_var, k_var], dim=-1)
            qk_var = tensor_model_parallel_all_reduce(qk_var) / self.tp_world
            q_var, k_var = qk_var.chunk(2, dim=-1)
        q = q * torch.rsqrt(q_var + self.q_norm.variance_epsilon) * self.q_norm.weight
        k = k * torch.rsqrt(k_var + self.k_norm.variance_epsilon) * self.k_norm.weight
        q = q.to(orig_dtype)
        k = k.to(orig_dtype)
        return q, k, v


def minimax_rms_norm(
    qkv: torch.Tensor,
    q_size: int,
    kv_size: int,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    token_num = qkv.size(0)
    if (
        current_platform.is_cuda()
        and self.tp_world > 1
        and self._ar_workspace is not None
        and token_num <= self._lamport_max_token
    ):
        return self._fused_kernel(qkv)
    else:
        return self._native_ops(qkv)


def minimax_rms_norm_fake(
    qkv: torch.Tensor,
    q_size: int,
    kv_size: int,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return qkv.split([q_size, kv_size, kv_size], dim=-1)


direct_register_custom_op(
    op_name="minimax_rms_norm",
    op_func=minimax_rms_norm,
    fake_impl=minimax_rms_norm_fake,
)
