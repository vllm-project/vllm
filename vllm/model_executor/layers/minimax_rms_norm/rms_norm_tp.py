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
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Max number of tokens supported by the Lamport fused allreduce+RMSNorm kernel.
# Larger batches fall back to the eager allreduce + RMSNorm path.
MINIMAX_QK_NORM_MAX_TOKEN_NUM = 2048

_MINIMAX_FUSED_AR_RMS_QK = getattr(torch.ops._C, "minimax_allreduce_rms_qk", None)


def _all_reduce_variance(var: torch.Tensor) -> torch.Tensor:
    """All-reduce a per-token variance tensor across the TP group.

    Variance is accumulated in fp32 for numerical stability. The FlashInfer
    fused all-reduce caches a single global workspace keyed to the model's
    16-bit activation dtype (``use_fp32_lamport=False``); routing an fp32
    reduction through it would read against a mismatched workspace and corrupt
    the result. FlashInfer's fast-path only triggers for 2D inputs, so reducing
    a flattened (1D) view keeps these fp32 reductions on custom all-reduce /
    pynccl, both of which handle fp32 correctly.
    """
    return tensor_model_parallel_all_reduce(var.flatten()).view_as(var)


@triton.jit
def _minimax_qk_var_kernel(
    qkv_ptr,  # [num_tokens, hidden], 16-bit activations
    var_ptr,  # [num_tokens, 2], fp32
    row_stride,  # element stride between tokens in qkv
    q_size: tl.constexpr,  # constant per deployment -> loops unroll, mask elides
    kv_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """TP-pre stage: per-token mean-of-squares for the q and k segments.

    Accumulates in fp32 while reading the 16-bit qkv in place, so no fp32
    copy of q/k is materialized. ``var[:, 0]`` is the q variance and
    ``var[:, 1]`` the k variance; both are the local-shard means, ready for
    the all-reduce that follows.
    """
    token = tl.program_id(0)
    base = qkv_ptr + token * row_stride

    q_acc = 0.0
    for off in range(0, q_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < q_size
        x = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
        q_acc += tl.sum(x * x, axis=0)

    k_acc = 0.0
    for off in range(0, kv_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < kv_size
        x = tl.load(base + q_size + idx, mask=mask, other=0.0).to(tl.float32)
        k_acc += tl.sum(x * x, axis=0)

    tl.store(var_ptr + token * 2 + 0, q_acc / q_size)
    tl.store(var_ptr + token * 2 + 1, k_acc / kv_size)


@triton.jit
def _minimax_rms_apply_kernel(
    qkv_ptr,  # [num_tokens, hidden]
    var_ptr,  # [num_tokens, 2], fp32, all-reduced sum of per-shard means
    q_w_ptr,  # [q_size], q per-channel weight
    k_w_ptr,  # [kv_size], k per-channel weight
    q_out_ptr,  # [num_tokens, q_size], contiguous
    k_out_ptr,  # [num_tokens, kv_size], contiguous
    row_stride,  # element stride between tokens in qkv
    q_size: tl.constexpr,  # constant per deployment -> loops unroll, mask elides
    kv_size: tl.constexpr,
    tp_world: tl.constexpr,  # folds the post-all-reduce /tp_world into rsqrt
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """TP-post stage: ``x * rsqrt(var / tp_world + eps) * weight``.

    A single program normalizes both the q and k segments of one token, so q
    and k share one launch instead of two. The all-reduce yields the sum of
    per-shard means, so the ``/ tp_world`` that recovers the global
    mean-of-squares is folded into the ``rsqrt`` here rather than run as a
    separate elementwise pass over the ``[num_tokens, 2]`` variance tensor.
    """
    token = tl.program_id(0)
    base = qkv_ptr + token * row_stride

    q_inv = tl.rsqrt(tl.load(var_ptr + token * 2 + 0) / tp_world + eps)
    q_out_row = q_out_ptr + token * q_size
    for off in range(0, q_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < q_size
        x = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(q_w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = x * q_inv * w
        tl.store(q_out_row + idx, y.to(q_out_ptr.dtype.element_ty), mask=mask)

    k_inv = tl.rsqrt(tl.load(var_ptr + token * 2 + 1) / tp_world + eps)
    k_out_row = k_out_ptr + token * kv_size
    for off in range(0, kv_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < kv_size
        x = tl.load(base + q_size + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(k_w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = x * k_inv * w
        tl.store(k_out_row + idx, y.to(k_out_ptr.dtype.element_ty), mask=mask)


def _minimax_qk_norm_tp_eager(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
    tp_world: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch reference path used when Triton is unavailable."""
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    q_var = q.pow(2).mean(dim=-1, keepdim=True)
    k_var = k.pow(2).mean(dim=-1, keepdim=True)

    qk_var = torch.cat([q_var, k_var], dim=-1)
    qk_var = _all_reduce_variance(qk_var) / tp_world
    q_var, k_var = qk_var.chunk(2, dim=-1)
    q = q * torch.rsqrt(q_var + eps) * q_weight
    k = k * torch.rsqrt(k_var + eps) * k_weight
    return q.to(orig_dtype), k.to(orig_dtype)


def _minimax_qk_norm_tp_fallback(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_size: int,
    kv_size: int,
    tp_rank: int,
    tp_world: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce + QK RMSNorm without the Lamport fused kernel.

    The all-reduce is a TP communication barrier and cannot live inside a
    single kernel, so the eager-torch path is split into two Triton kernels
    around it: a variance reduction before the all-reduce and a normalize
    after. Compared to the ``torch.compile`` path this avoids materializing
    fp32 copies of q/k and the ``cat``/``chunk`` temporaries.
    """
    if not HAS_TRITON:
        return _minimax_qk_norm_tp_eager(
            qkv, q_weight, k_weight, q_size, kv_size, tp_world, eps
        )

    num_tokens = qkv.shape[0]
    row_stride = qkv.stride(0)
    BLOCK = 1024
    grid = (num_tokens,)

    qk_var = torch.empty(num_tokens, 2, dtype=torch.float32, device=qkv.device)
    _minimax_qk_var_kernel[grid](
        qkv, qk_var, row_stride, q_size=q_size, kv_size=kv_size, BLOCK=BLOCK
    )

    # All-reduce sums the per-shard means; the /tp_world that turns this back
    # into the global mean is folded into the apply kernel's rsqrt below.
    qk_var = _all_reduce_variance(qk_var)

    q_out = torch.empty(num_tokens, q_size, dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty(num_tokens, kv_size, dtype=qkv.dtype, device=qkv.device)
    _minimax_rms_apply_kernel[grid](
        qkv,
        qk_var,
        q_weight,
        k_weight,
        q_out,
        k_out,
        row_stride,
        q_size=q_size,
        kv_size=kv_size,
        tp_world=tp_world,
        eps=eps,
        BLOCK=BLOCK,
    )
    return q_out, k_out


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
    return _minimax_qk_norm_tp_fallback(
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
            from vllm.distributed.device_communicators.custom_all_reduce import (
                _can_p2p,
            )

            from .lamport_workspace import (
                get_allreduce_workspace,
            )

            # The Lamport workspace exchanges CUDA IPC handles and enables peer
            # access between GPUs. This requires P2P (IPC peer access) to be
            # available; on topologies where it is not (e.g. consumer PCIe cards
            # with P2P disabled in the driver), allocation raises. Fall back to
            # the eager allreduce + RMSNorm path instead of failing model load.
            #
            # Note that the driver may report P2P as available
            # (can_device_access_peer() returns True and the IPC handle
            # exchange succeeds) while actual peer writes silently never
            # arrive. On such topologies the Lamport spin-wait loops forever
            # waiting for flags that are never delivered, hanging startup with
            # all ranks at 100% GPU. Guard with the same functional P2P write
            # check used by the custom allreduce instead of trusting the
            # driver's report.
            try:
                if not _can_p2p(self.tp_rank, self.tp_world):
                    logger.warning_once(
                        "MiniMax fused allreduce+RMSNorm disabled: functional "
                        "P2P access check failed (the driver may report P2P "
                        "as available even though peer writes do not work, "
                        "e.g. on consumer PCIe multi-GPU boards). Falling "
                        "back to the eager allreduce + RMSNorm path."
                    )
                else:
                    self.workspace = get_allreduce_workspace(
                        rank=self.tp_rank,
                        world_size=self.tp_world,
                        max_tokens=MINIMAX_QK_NORM_MAX_TOKEN_NUM,
                        process_group=get_tp_group().cpu_group,
                    )
            except Exception as e:
                logger.warning_once(
                    "Failed to initialize MiniMax fused allreduce+RMSNorm "
                    "Lamport workspace: %s. This is expected on GPUs without "
                    "P2P (IPC peer access) support. Falling back to the eager "
                    "allreduce + RMSNorm path.",
                    e,
                )
                self.workspace = None

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
            variance = _all_reduce_variance(variance) / self.tp_world
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
        # Case 0： tp_size=1
        if get_tensor_model_parallel_world_size() == 1:
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            q, k = MiniMaxText01RMSNormTP.forward_qk(q_norm, k_norm, q, k)
            return q, k, v
        # Case : tp_size>1
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
