# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram WKV ahead of the decoder layers.

``Engram`` looks up its rows before the decoder layers: inline for GPU-resident
tables, on a prefetch stream for host-offloaded ones. The rest of the WKV chain
(head all-gather, MXFP8 quantize, column-parallel GEMM, output all-gather) reads
only those rows, so for decode steps ``EngramPrefetch`` runs it on a side stream
right after each layer's lookup, and the layer only waits for it before the
fused post kernel. The kernels and inputs are those of ``Engram.forward``,
except that the two all-gathers run on a Lamport all-gather of the chain's own.

The chain issues TP collectives on the side stream, so whether a step starts it
depends only on the step's rows, which every TP rank shares.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.models.deepseek_v41.common.engram import (
    Engram,
    _fused_engram_post_wkv_kernel,
)
from vllm.triton_utils import triton

logger = init_logger(__name__)


def prefetchable(engram: Engram) -> bool:
    """Whether ``engram``'s WKV can run ahead: TP-only tables and a CuTe DSL
    MXFP8 WKV (FlashInfer's CUTLASS MXFP8 GEMM shares one process-wide
    workspace with GEMMs on other streams)."""
    kernel = getattr(getattr(engram.wkv, "quant_method", None), "kernel", None)
    return (
        engram._batch is None
        and engram.embed_tokens.dp_size == 1
        and not engram.use_sequence_parallel
        and isinstance(kernel, FlashInferCutedslMxfp8LinearKernel)
    )


class EngramPrefetch:
    """The WKV chains of a decode step's Engram layers, on one side stream."""

    def __init__(self, engrams: list[Engram], max_rows: int) -> None:
        self.stream = torch.cuda.Stream()
        self._buffers: dict[Engram, tuple[torch.Tensor, torch.cuda.Event]] = {
            engram: (
                torch.empty(
                    max_rows,
                    (engram.hc_mult + 1) * engram.dim,
                    dtype=torch.bfloat16,
                    device=engram.q_weight.device,
                ),
                torch.cuda.Event(),
            )
            for engram in engrams
            if prefetchable(engram)
        }
        # Started chains no layer has waited for yet.
        self._pending: set[Engram] = set()
        self.all_gather = None
        if any(engram.embed_tokens.tp_size > 1 for engram in self._buffers):
            from ...ops.cute_dsl import LamportAllGather

            # Its own mailbox: the chain runs beside the executor's gathers.
            # A rank sends its head rows, then its WKV output columns.
            row_bytes = max(
                max(
                    engram.staged_rows[0].nbytes,
                    out[0].nbytes // engram.embed_tokens.tp_size,
                )
                for engram, (out, _) in self._buffers.items()
            )
            self.all_gather = LamportAllGather(
                rank_capacity_bytes=max_rows * row_bytes,
                device=torch.device("cuda", torch.accelerator.current_device_index()),
            )
        if self._buffers:
            logger.info_once(
                "Engram WKV runs ahead of the decoder layers for decode steps of "
                "up to %d rows.",
                max_rows,
            )

    def start(self, num_tokens: int) -> None:
        """Start every layer's WKV chain once its lookup is done."""
        stream = self.stream
        for engram, (out, done) in self._buffers.items():
            if engram._prefetch_done is None:
                # GPU-resident rows were looked up inline on this stream.
                stream.wait_stream(torch.cuda.current_stream())
            else:
                stream.wait_event(engram._prefetch_done)
            tp = engram.embed_tokens.tp_size
            with torch.cuda.stream(stream):
                heads = engram.staged_rows[:num_tokens]
                if self.all_gather is not None:
                    # [tp * rows, heads, d] -> [rows, tp * heads, d]
                    heads = (
                        self.all_gather(heads)
                        .unflatten(0, (tp, num_tokens))
                        .transpose(0, 1)
                        .flatten(1, 2)
                    )
                heads = heads[:, : engram.embed_tokens.n_hash_cols].flatten(-2)
                # ``engram.wkv`` without its output gather; it has no bias.
                kv = engram.wkv.quant_method.apply(engram.wkv, heads, None)
                # Temporaries stay on this stream; only the persistent output
                # reaches the consumer, after the event below.
                if self.all_gather is None:
                    out[:num_tokens].copy_(kv)
                else:
                    # [tp * rows, cols] -> [rows, tp * cols]
                    out[:num_tokens].view(num_tokens, tp, -1).copy_(
                        self.all_gather(kv)
                        .unflatten(0, (tp, num_tokens))
                        .transpose(0, 1)
                    )
            done.record(stream)
            self._pending.add(engram)

    def join(self) -> None:
        """Make the current stream wait for every chain no layer waited for."""
        if self._pending:
            torch.cuda.current_stream().wait_stream(self.stream)
            self._pending.clear()

    def forward(
        self,
        engram: Engram,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """``engram(hidden_states, hash_ids, token_mask)``, on the WKV started
        ahead when there is one."""
        if engram not in self._pending:
            return engram(hidden_states, hash_ids, token_mask)
        self._pending.remove(engram)
        out, done = self._buffers[engram]
        torch.cuda.current_stream().wait_event(done)

        # Engram.forward launches the post kernel inline after its own WKV,
        # with no wrapper to call, so this repeats that launch (without
        # sequence parallelism) on the WKV output.
        num_tokens, hc_mult, dim = hidden_states.shape
        assert hash_ids.shape[0] == num_tokens and hc_mult == engram.hc_mult
        assert token_mask is None or token_mask.shape == (num_tokens,)
        kv = out[:num_tokens]
        output = torch.empty_like(hidden_states)
        block_size = triton.next_power_of_2(dim)
        _fused_engram_post_wkv_kernel[(num_tokens * hc_mult,)](
            hidden_states,
            kv,
            engram.q_weight,
            engram.k_weight,
            token_mask if token_mask is not None else hidden_states,
            output,
            num_tokens,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            kv.stride(0),
            kv.stride(1),
            engram.q_weight.stride(0),
            engram.q_weight.stride(1),
            engram.k_weight.stride(0),
            engram.k_weight.stride(1),
            token_mask.stride(0) if token_mask is not None else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            engram.eps,
            engram.clamp_value,
            DIM=dim,
            HC_MULT=hc_mult,
            BLOCK_SIZE=block_size,
            HAS_MASK=token_mask is not None,
            num_warps=8 if block_size >= 2048 else 4,
        )
        return output
