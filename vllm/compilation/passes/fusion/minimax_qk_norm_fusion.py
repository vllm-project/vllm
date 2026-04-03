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

Replacement:
    minimax_allreduce_rms_qk(qkv, q_weight, k_weight, workspace, ...)  # in-place
    return qkv.split([q_size, kv_size, kv_size], -1)

is_applicable_for_range: only fires for compile_range.end <= max_decode_tokens
so that large prefill batches fall through to the original forward_qk (= main).
"""

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)

_MINIMAX_AR_RMS_QK_OP = None
if hasattr(torch.ops._C, "minimax_allreduce_rms_qk"):
    _MINIMAX_AR_RMS_QK_OP = torch.ops._C.minimax_allreduce_rms_qk.default


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
        workspace: torch.Tensor,
        dtype: torch.dtype,
        device: str | None,
    ) -> None:
        self.q_size = q_size
        self.kv_size = kv_size
        self.eps = eps
        self.tp_world = tp_world
        self.tp_rank = tp_rank
        self.workspace = workspace
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
        workspace = self.workspace
        tp_rank = self.tp_rank
        dtype = self.dtype

        def pattern(
            qkv: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            q = q.contiguous()
            k = k.contiguous()
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
            assert _MINIMAX_AR_RMS_QK_OP is not None
            result = auto_functionalized(
                _MINIMAX_AR_RMS_QK_OP,
                qkv=qkv,
                norm_weight_q=q_weight,
                norm_weight_k=k_weight,
                workspace=workspace,
                q_size=q_size,
                kv_size=kv_size,
                rank=tp_rank,
                nranks=tp_world,
                eps=eps,
            )
            qkv_out = result[1]  # first (and only) mutated arg
            return qkv_out.split([q_size, kv_size, kv_size], dim=-1)  # type: ignore

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class MiniMaxQKNormPass(VllmPatternMatcherPass):
    """
    Replace forward_qk allreduce+norm with the Lamport fused kernel.
    Only applied for decode-size compile ranges (small token counts).
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = True

        if _MINIMAX_AR_RMS_QK_OP is None:
            logger.warning_once(
                "minimax_allreduce_rms_qk op not found, "
                "MiniMaxQKNormPass disabled."
            )
            return

        tp_world = get_tensor_model_parallel_world_size()
        if tp_world <= 1:
            logger.warning_once(
                "MiniMaxQKNormPass disabled: tp_size <= 1."
            )
            return

        if config.model_config is None:
            logger.warning_once(
                "MiniMaxQKNormPass disabled: no model_config."
            )
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

        # Max tokens for which Lamport is beneficial
        self.max_token_num = 2048

        tp_rank = get_tensor_model_parallel_rank()

        # Allocate (or reuse cached) Lamport workspace.
        from vllm.distributed.parallel_state import get_tp_group
        from vllm.model_executor.layers.minimax_rms_norm.lamport_workspace import (
            get_allreduce_workspace,
        )

        workspace = get_allreduce_workspace(
            rank=tp_rank,
            world_size=tp_world,
            max_tokens=self.max_token_num,
            process_group=get_tp_group().cpu_group,
        )

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="minimax_qk_norm_pass"
        )
        self._register_patterns(q_size, kv_size, eps, tp_world, tp_rank, workspace)
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
        workspace: torch.Tensor,
    ) -> None:
        MiniMaxQKNormPattern(
            q_size=q_size,
            kv_size=kv_size,
            eps=eps,
            tp_world=tp_world,
            tp_rank=tp_rank,
            workspace=workspace,
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
