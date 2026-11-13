# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MiniMax-M3.1 decoder-layer norms on amd_kernels: the qkv input norm (``gemm``
route) and the TP4 all-reduce + add + Gemma RMSNorm at both sites (``norm`` route)."""

import contextlib
import dataclasses
import time
import types
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8 import rocm_akr
from vllm.models.minimax_m3.amd.mono.selection import DECODE, enabled_monokernels

logger = init_logger(__name__)


_comms: Any = None  # amd_kernels DecoderArNorm, set by init()


@dataclasses.dataclass
class DeferredFfn:
    """A layer's FFN TP partial (amd_kernels' ``FfnPartial``) for the next norm;
    ``taps`` (the DSpark drafter's aux hidden states) gets the residual it makes."""

    ffn: Any
    taps: list | None = None


def _lib():
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import allreduce_add_rmsnorm

    return allreduce_add_rmsnorm


def unsupported_reason(vllm_config) -> str | None:
    """None when the decoder layers can run the route, else why not."""
    pc = vllm_config.parallel_config
    # the FFN all-reduce runs in the next layer's norm: one stage holds every layer
    if pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1:
        return "needs PP 1 and DP 1"
    if vllm_config.kernel_config.monokernel and DECODE in enabled_monokernels(
        vllm_config
    ):
        return "the mono decode path (--monokernel) owns the layers"
    return _lib().unsupported_reason(
        pc.tensor_parallel_size, vllm_config.model_config.hf_text_config.hidden_size
    )


def init(model: torch.nn.Module) -> None:
    """Build and warm up the communicators (a TP collective) once the weights are
    processed, and enter their ``capture()`` around every TP graph capture."""
    global _comms
    L = _lib()
    layers = model.layers[model.start_layer : model.end_layer]
    for layer in layers:
        # every input norm but the first writes the qkv GEMM's codes
        if not rocm_akr.is_akr_layer(layer.self_attn.qkv_proj):
            raise ValueError(
                f"amd_kernels norm: layer {layer.layer_id} qkv is not on amd_kernels"
            )
        if layer.is_moe_layer:
            layer.akr_ffn = layer.block_sparse_moe._akr_weights
        else:
            g, d, mlp = layer.mlp.gate_up_proj, layer.mlp.down_proj, layer.mlp
            layer.akr_ffn = L.DenseMlp(
                g.weight,
                g.weight_scale,
                d.weight,
                d.weight_scale,
                mlp.swiglu_alpha,
                mlp.swiglu_beta,
                mlp.swiglu_limit,
            )
    if _comms is not None:
        return
    from vllm.distributed.parallel_state import get_tp_group

    moe = next((x.akr_ffn for x in layers if x.is_moe_layer), None)
    tp = get_tp_group()
    norm = layers[0].post_attention_layernorm
    dev = torch.device("cuda", torch.accelerator.current_device_index())
    t0 = time.time()
    eps = float(norm.variance_epsilon)
    comms = L.DecoderArNorm(
        tp.rank_in_group, tp.cpu_group, dev, norm.weight.data, eps, moe=moe
    )
    original = tp.graph_capture

    @contextlib.contextmanager
    def graph_capture(self, graph_capture_context=None):
        with comms.capture(), original(graph_capture_context) as ctx:
            yield ctx

    bound = types.MethodType(graph_capture, tp)
    tp.graph_capture = bound  # type: ignore[method-assign]
    _comms = comms
    logger.info(
        "amd_kernels norm: TP4 all-reduce + add + Gemma RMSNorm on "
        "amd_kernels at both decoder-layer sites (start-up %.1f s)",
        time.time() - t0,
    )


def _active() -> bool:
    from vllm.v1.worker.ubatching import dbo_enabled

    return _comms is not None and not dbo_enabled()


def layer_tail(layer: torch.nn.Module, hidden_states, residual):
    """all-reduce(o_proj partial) + residual -> post-attention norm -> the FFN of
    ``layer`` (a ``MiniMaxM3DecoderLayer``), whose TP partial the next norm reduces
    (a ``DeferredFfn``); None outside the kernels' rows and DBO microbatches."""
    if not _active():
        return None
    norm = layer.post_attention_layernorm
    out = _comms.post_attention(
        hidden_states,
        residual,
        norm.weight,
        norm.variance_epsilon,
        layer.akr_ffn,
        final=layer.akr_defer_ffn == "final",
    )
    if out is None:
        return None
    x, residual = out
    if isinstance(x, tuple):  # above the library MLP's rows: gate_up's codes
        return layer.mlp(rocm_akr.akr_codes(*x, *residual.shape)), residual
    return DeferredFfn(x), residual


def deferred_norm(deferred: DeferredFfn, residual, norm, gemm=None):
    """The previous FFN's all-reduce + residual -> ``norm``, one launch: (codes for
    ``gemm``, or bf16 without one, residual)."""
    out, residual = _comms.pre_attention(
        deferred.ffn,
        residual,
        norm.weight,
        norm.variance_epsilon,
        gemm_weight=None if gemm is None else gemm.weight,
    )
    if gemm is not None:
        out = rocm_akr.akr_codes(*out, *residual.shape)
    if deferred.taps is not None:
        deferred.taps.append(residual)
    return out, residual


def input_norm(x, residual, norm, qkv_proj):
    """The input norm of a layer whose qkv runs amd_kernels gemm, writing that GEMM's
    codes; ``x`` is the previous FFN's output or its ``DeferredFfn``."""
    if isinstance(x, DeferredFfn):
        return deferred_norm(x, residual, norm, qkv_proj)
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import (
        gemma_add_rmsnorm_mxfp8_for,
    )

    q, s, residual = gemma_add_rmsnorm_mxfp8_for(
        x, residual, norm.weight, norm.variance_epsilon, qkv_proj.weight
    )
    return rocm_akr.akr_codes(q, s, *x.shape), residual
