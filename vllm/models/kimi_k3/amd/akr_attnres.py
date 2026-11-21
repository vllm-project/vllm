# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The amd_kernels `norm` route: the TP all-reduce + AttnRes mix + RMSNorm in one
launch, entry `norm/attnres_rmsnorm_common` (communicator `attnres_comm`, op its
`allreduce_attnres_rmsnorm` method).

In a K3 decoder layer the AttnRes ops (linear.py `_apply_attn_res`) consume a TP-reduced
delta: the attention o_proj output (RowParallelLinear all-reduce) and the previous
layer's MoE / MLP output (runner final all-reduce). With the route on, both stay TP
partials (o_proj `reduce_results=False`, the runner's `_akr_defer_final`) and
`linear._apply_attn_res_partial` sums them inside the kernel; calls it cannot take
(`allreduce_attnres_rmsnorm_unsupported_reason`: rows, dtypes, layouts; fp8-quantised
consumers) all-reduce first and run the stock `_apply_attn_res` -- the same math, split
in two ops.

Co-residency: the kernel spins on peer mailboxes with one workgroup per row. Every
producer of its partial (the o_proj GEMM, the MoE tail with the aux-stream shared
experts joined before the latent kernel, the DSA aux-stream join) completes before it
is launched (data dependencies), so no other kernel is resident while it runs; it is
only ever launched on the compute stream.
"""

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_config, akr_runtime, akr_switch

logger = init_logger(__name__)


def unsupported_reason(vllm_config) -> str | None:
    pc = vllm_config.parallel_config
    if pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1:
        return "needs PP 1 and DP 1 (the final reduce runs in the next layer's norm)"
    hf = vllm_config.model_config.hf_text_config
    if getattr(hf, "attn_res_block_size", None) is None:
        return "needs attention residuals"
    if vllm_config.model_config.dtype != torch.bfloat16:
        return "needs bf16"
    return None


def enabled(vllm_config) -> bool:
    if not akr_switch.on("ATTNRES_AR"):
        return False
    return not akr_config.unsupported_reason("norm", unsupported_reason(vllm_config))


def _args(
    partial, prefix, bank, proj, norm, num_valid_blocks, output_norm, block_write_idx
) -> tuple:
    return (
        partial,
        prefix,
        bank,
        norm.weight,
        proj.weight.squeeze(0),
        num_valid_blocks,
        block_write_idx,
        None if output_norm is None else output_norm.weight,
        float(norm.variance_epsilon),
        0.0 if output_norm is None else float(output_norm.variance_epsilon),
    )


def fits(
    partial: torch.Tensor,
    prefix: torch.Tensor | None,
    bank: torch.Tensor,
    proj,
    norm,
    num_valid_blocks: int,
    *,
    output_norm=None,
    block_write_idx: int = -1,
) -> bool:
    """Whether the fused kernel takes this call (the entry's reason)."""
    comm = akr_runtime.attnres_comm
    if comm is None:
        return False
    from amd_kernels.norm.attnres_rmsnorm_common import (
        allreduce_attnres_rmsnorm_unsupported_reason,
    )

    args = _args(
        partial,
        prefix,
        bank,
        proj,
        norm,
        num_valid_blocks,
        output_norm,
        block_write_idx,
    )
    return not akr_config.call_unsupported(
        "norm", allreduce_attnres_rmsnorm_unsupported_reason(comm, *args)
    )


def fused(
    partial: torch.Tensor,
    prefix: torch.Tensor | None,
    bank: torch.Tensor,
    proj,
    norm,
    num_valid_blocks: int,
    *,
    output_norm=None,
    block_write_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(hidden, prefix) = AttnRes(prefix + all-reduce(partial)).

    `prefix` is updated in place; None: the new prefix is bf16(all-reduce(partial)) (a
    fresh tensor, the block-write layers' attention output)."""
    return akr_runtime.attnres_comm.allreduce_attnres_rmsnorm(
        *_args(
            partial,
            prefix,
            bank,
            proj,
            norm,
            num_valid_blocks,
            output_norm,
            block_write_idx,
        )
    )


def reduce(partial: torch.Tensor) -> torch.Tensor:
    """The TP all-reduce a deferred partial still needs (stock fallback)."""
    return tensor_model_parallel_all_reduce(partial)


def warm_up(layer) -> None:
    """Dummy calls of the three kernel variants the decoder uses (prefix + output norm,
    no prefix + output norm, prefix without output norm): compiles them and launches
    once on zeros (collective: every rank, same order). `layer` is any KimiDecoderLayer.
    """
    dev = layer.mlp_res_norm.weight.device
    m, h = 16, layer.mlp_res_norm.weight.shape[0]
    z = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)  # noqa: E731
    for has_prefix, out_norm in ((True, True), (False, True), (True, False)):
        fused(
            z(m, h),
            z(m, h) if has_prefix else None,
            z(m, 8, h),
            layer.mlp_res_proj,
            layer.mlp_res_norm,
            4,
            output_norm=layer.post_attention_layernorm if out_norm else None,
        )
    torch.accelerator.synchronize()
