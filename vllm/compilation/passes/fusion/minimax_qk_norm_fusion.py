# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Fusion pass: replace MiniMax QK allreduce + RMS norm with the Lamport
fused kernel (minimax_allreduce_rms_qk) for decode-size batches.

Pattern (inlined forward_qk in compiled graph):
    q, k, v = qkv.split([q_size, kv_size, kv_size], -1)
    q_fp32 = q.to(float32); k_fp32 = k.to(float32)
    q_var = q_fp32.pow(2).mean(-1, keepdim=True)
    k_var = k_fp32.pow(2).mean(-1, keepdim=True)
    qk_var = cat([q_var, k_var], -1)
    qk_var = allreduce(qk_var) / tp_world
    q_var, k_var = qk_var.chunk(2, -1)
    q_out = (q_fp32 * rsqrt(q_var + eps) * q_weight).to(orig_dtype)
    k_out = (k_fp32 * rsqrt(k_var + eps) * k_weight).to(orig_dtype)
    return q_out, k_out, v

Replacement (pure, no in-place on qkv/q/k):
    q_out, k_out = minimax_qk_norm_fused(qkv, q_weight, k_weight, workspace, ...)
    v = qkv.split([q_size, kv_size, kv_size], -1)[2]
    return q_out, k_out, v

is_applicable_for_range: only fires for compile_range.end <= max_decode_tokens
so that large prefill batches fall through to the original forward_qk (= main).
"""

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

MAX_TOKEN_NUM = 2048

_MINIMAX_QK_NORM_FUSED_OP = None
if hasattr(torch.ops._C, "minimax_allreduce_rms_qk"):

    def _minimax_qk_norm_fused(
        qkv: torch.Tensor,
        norm_weight_q: torch.Tensor,
        norm_weight_k: torch.Tensor,
        q_size: int,
        kv_size: int,
        rank: int,
        nranks: int,
        eps: float,
        max_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.distributed.parallel_state import get_tp_group
        from vllm.model_executor.layers.mamba.lamport_workspace import (
            get_allreduce_workspace,
        )

        workspace = get_allreduce_workspace(
            rank=rank,
            world_size=nranks,
            max_tokens=max_tokens,
            process_group=get_tp_group().cpu_group,
        )
        return torch.ops._C.minimax_allreduce_rms_qk(
            qkv,
            norm_weight_q,
            norm_weight_k,
            workspace,
            q_size,
            kv_size,
            rank,
            nranks,
            eps,
        )

    def _minimax_qk_norm_fused_fake(
        qkv: torch.Tensor,
        norm_weight_q: torch.Tensor,
        norm_weight_k: torch.Tensor,
        q_size: int,
        kv_size: int,
        rank: int,
        nranks: int,
        eps: float,
        max_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = qkv.shape[0]
        return (
            torch.empty([T, q_size], dtype=qkv.dtype, device=qkv.device),
            torch.empty([T, kv_size], dtype=qkv.dtype, device=qkv.device),
        )

    direct_register_custom_op(
        op_name="minimax_qk_norm_fused",
        op_func=_minimax_qk_norm_fused,
        fake_impl=_minimax_qk_norm_fused_fake,
        mutates_args=[],
    )
    _MINIMAX_QK_NORM_FUSED_OP = torch.ops.vllm.minimax_qk_norm_fused.default


class MiniMaxQKNormPattern:
    """
    Match the forward_qk allreduce+rms pattern and replace with Lamport kernel.
    """

    def __init__(
        self,
        q_size: int,
        kv_size: int,
        eps: float,
        tp_world: int,
        tp_rank: int,
        max_tokens: int,
        dtype: torch.dtype,
        device: str | None,
    ) -> None:
        self.q_size = q_size
        self.kv_size = kv_size
        self.eps = eps
        self.tp_world = tp_world
        self.tp_rank = tp_rank
        self.max_tokens = max_tokens
        self.dtype = dtype
        self.device = device

    def get_inputs(self) -> list[torch.Tensor]:
        T = 4
        qkv = torch.empty(
            [T, self.q_size + 2 * self.kv_size],
            device=self.device,
            dtype=self.dtype,
        )
        q_weight = torch.empty([self.q_size], device=self.device, dtype=self.dtype)
        k_weight = torch.empty([self.kv_size], device=self.device, dtype=self.dtype)
        return [qkv, q_weight, k_weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        q_size = self.q_size
        kv_size = self.kv_size
        eps = self.eps
        tp_world = self.tp_world
        max_tokens = self.max_tokens
        tp_rank = self.tp_rank
        dtype = self.dtype

        def pattern(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            q_fp32 = q.to(torch.float32)
            k_fp32 = k.to(torch.float32)
            q_var = q_fp32.pow(2).mean(dim=-1, keepdim=True)
            k_var = k_fp32.pow(2).mean(dim=-1, keepdim=True)
            qk_var = torch.cat([q_var, k_var], dim=-1)
            qk_var = tensor_model_parallel_all_reduce(qk_var) / tp_world
            q_var, k_var = qk_var.chunk(2, dim=-1)
            q_out = (q_fp32 * torch.rsqrt(q_var + eps) * q_weight).to(dtype)
            k_out = (k_fp32 * torch.rsqrt(k_var + eps) * k_weight).to(dtype)
            return q_out, k_out, v

        def replacement(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert _MINIMAX_QK_NORM_FUSED_OP is not None
            q_out, k_out = torch.ops.vllm.minimax_qk_norm_fused(
                qkv,
                q_weight,
                k_weight,
                q_size,
                kv_size,
                tp_rank,
                tp_world,
                eps,
                max_tokens,
            )
            _, _, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            return q_out, k_out, v

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

        # Second pattern: three separate split_with_sizes nodes (one per output),
        # each with _users=1. This occurs when the QKV projection uses a
        # functional GEMM kernel (e.g. cutlass_scaled_mm via auto_functionalized),
        # which causes inductor to generate one split per consumer.
        def pattern_split3(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q = qkv.split([q_size, kv_size, kv_size], dim=-1)[0]
            k = qkv.split([q_size, kv_size, kv_size], dim=-1)[1]
            v = qkv.split([q_size, kv_size, kv_size], dim=-1)[2]
            q_fp32 = q.to(torch.float32)
            k_fp32 = k.to(torch.float32)
            q_var = q_fp32.pow(2).mean(dim=-1, keepdim=True)
            k_var = k_fp32.pow(2).mean(dim=-1, keepdim=True)
            qk_var = torch.cat([q_var, k_var], dim=-1)
            qk_var = tensor_model_parallel_all_reduce(qk_var) / tp_world
            q_var, k_var = qk_var.chunk(2, dim=-1)
            q_out = (q_fp32 * torch.rsqrt(q_var + eps) * q_weight).to(dtype)
            k_out = (k_fp32 * torch.rsqrt(k_var + eps) * k_weight).to(dtype)
            return q_out, k_out, v

        pm.register_replacement(
            pattern_split3, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiniMaxQKNormPass(VllmPatternMatcherPass):
    """
    Replace forward_qk allreduce+norm with the Lamport fused kernel.
    Only applied for decode-size compile ranges (small token counts).
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = True

        if _MINIMAX_QK_NORM_FUSED_OP is None:
            logger.warning_once(
                "minimax_allreduce_rms_qk op not found, MiniMaxQKNormPass disabled."
            )
            return

        tp_world = get_tensor_model_parallel_world_size()
        if tp_world <= 1:
            logger.warning_once("MiniMaxQKNormPass disabled: tp_size <= 1.")
            return

        if config.model_config is None:
            logger.warning_once("MiniMaxQKNormPass disabled: no model_config.")
            return

        hf_cfg = config.model_config.hf_config

        model_name = getattr(hf_cfg, "architectures", "")[0]
        if model_name != "MiniMaxM2ForCausalLM":
            return

        num_attention_heads = getattr(hf_cfg, "num_attention_heads", 0)
        num_key_value_heads = getattr(hf_cfg, "num_key_value_heads", 0)
        hidden_size = getattr(hf_cfg, "hidden_size", 0)
        head_dim = getattr(hf_cfg, "head_dim", 0)
        eps: float = getattr(hf_cfg, "rms_norm_eps", 1e-6)

        if (
            num_attention_heads != 48
            or num_key_value_heads != 8
            or hidden_size != 3072
            or head_dim != 128
        ):
            logger.warning_once(
                "MiniMaxQKNormPass disabled: cannot infer model info from hf_config."
            )
            return

        num_heads_per_rank = num_attention_heads // tp_world
        num_kv_heads_per_rank = max(1, num_key_value_heads // tp_world)
        q_size = num_heads_per_rank * head_dim
        kv_size = num_kv_heads_per_rank * head_dim

        self.max_token_num = min(
            MAX_TOKEN_NUM, config.scheduler_config.max_num_batched_tokens
        )

        tp_rank = get_tensor_model_parallel_rank()
        # Allocate Lamport workspace first.
        from vllm.distributed.parallel_state import get_tp_group
        from vllm.model_executor.layers.mamba.lamport_workspace import (
            get_allreduce_workspace,
        )

        get_allreduce_workspace(
            rank=tp_rank,
            world_size=tp_world,
            max_tokens=self.max_token_num,
            process_group=get_tp_group().cpu_group,
        )

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="minimax_qk_norm_pass"
        )
        self._register_patterns(q_size, kv_size, eps, tp_world, tp_rank)
        self.dump_patterns(config, self.patterns)
        self.disabled = False

    @enable_fake_mode
    def _register_patterns(
        self,
        q_size: int,
        kv_size: int,
        eps: float,
        tp_world: int,
        tp_rank: int,
    ) -> None:
        MiniMaxQKNormPattern(
            q_size=q_size,
            kv_size=kv_size,
            eps=eps,
            tp_world=tp_world,
            tp_rank=tp_rank,
            max_tokens=self.max_token_num,
            dtype=self.model_dtype,
            device=self.device,
        ).register(self.patterns)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if self.disabled:
            return False

        return bool(compile_range.end <= self.max_token_num)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            return
        self.matched_count = self.patterns.apply(graph)
        logger.debug("MiniMaxQKNormPass replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(self, MiniMaxQKNormPattern)
