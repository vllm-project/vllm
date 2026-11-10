# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel choices of a Hopper executor step, one function per kernel.

Each function runs a Hopper low-batch kernel for the rows it serves and the
stock kernel of the stock layer otherwise. Every Hopper kernel reads the
tensors the stock layer reads, in place (Humming W8A16 dense weights, Humming
W4A8 experts, the FlashMLA KV and index caches), so every step outside the
executor and every prefill keeps running the stock forward on a single copy of
the weights; the small-batch MLA chain's `wo_a`, which the stock SM90 layer
keeps only as BF16, is the one exception (`dense.fp8_wo_a`).

Each layer's low-batch dense kernels are `dense.DenseFp8Linear` calls, one
per layer and seam, built with the executor, up to `dense.MAX_ROWS` rows.
Every launch here is captured into a FULL CUDA graph of the executor only, and
the Hopper kernels chain with programmatic dependent launch. The sparse-MLA
decode is `sparse_mla.FlashMlaDecode.run`.

Outside the executor's own kernels, this module relies on these stock MoE
internals (`hopper/AGENTS.md`): `SharedMlp.installed` replaces the shared
expert MLP's `forward` on the instance; `partial_output` sets the runner's
`moe_config.skip_final_all_reduce` for the call; `moe` calls the runner's
`gate`.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
from vllm.models.deepseek_v41.attention import DeepseekV4Attention, DeepseekV4Indexer

from ...ops.o_proj import dsv41_o_proj
from .dense import DenseFp8Linear, fp8_wo_a
from .moe import HopperLowBatchMoE

if TYPE_CHECKING:
    from ...model import DeepseekV4MoE


@dataclass
class AttentionLinears:
    """The low-batch providers of one attention layer's dense linears."""

    wqa_wkv: DenseFp8Linear | None
    wq_b: DenseFp8Linear | None
    index_wq_b: DenseFp8Linear | None
    wo_a: DenseFp8Linear | None
    wo_b: DenseFp8Linear | None

    @classmethod
    def of(cls, attn: DeepseekV4Attention) -> "AttentionLinears":
        indexer = attn.indexer
        return cls(
            wqa_wkv=DenseFp8Linear.of("wqa_wkv", attn.fused_wqa_wkv),
            wq_b=DenseFp8Linear.of("wq_b", attn.wq_b),
            index_wq_b=(
                None
                if indexer is None
                else DenseFp8Linear.of("index_wq_b", indexer.wq_b)
            ),
            wo_a=fp8_wo_a(attn.wo_a, attn.n_local_groups),
            wo_b=DenseFp8Linear.of("wo_b", attn.wo_b),
        )

    def all(self) -> list[DenseFp8Linear]:
        linears = (self.wqa_wkv, self.wq_b, self.index_wq_b, self.wo_a, self.wo_b)
        return [linear for linear in linears if linear is not None]


def _serves(linear: DenseFp8Linear | None, rows: int) -> bool:
    return linear is not None and linear.serves(rows)


def wqa_wkv(
    attn: DeepseekV4Attention, linears: AttentionLinears, x: torch.Tensor
) -> torch.Tensor:
    """`fused_wqa_wkv` of the BF16 layer input: replicated, N=1792, K=5120."""
    if _serves(linears.wqa_wkv, x.shape[0]):
        assert linears.wqa_wkv is not None
        return linears.wqa_wkv(x)
    return attn._fused_wqa_wkv_gemm(x)


def wq_b(
    attn: DeepseekV4Attention, linears: AttentionLinears, qr: torch.Tensor
) -> torch.Tensor:
    """`wq_b` of the normalized query: N=8192, K=1280."""
    if _serves(linears.wq_b, qr.shape[0]):
        assert linears.wq_b is not None
        return linears.wq_b(qr)
    return attn._wq_b_proj(qr)


def index_wq_b(
    indexer: DeepseekV4Indexer, linears: AttentionLinears, qr: torch.Tensor
) -> torch.Tensor:
    """The indexer's `wq_b` of the normalized query, on aux stream 2:
    replicated, N=4096, K=1280."""
    if _serves(linears.index_wq_b, qr.shape[0]):
        assert linears.index_wq_b is not None
        return linears.index_wq_b(qr)
    return indexer._wq_b_proj(qr, None)


def wo_b(
    attn: DeepseekV4Attention,
    linears: AttentionLinears,
    z: torch.Tensor,
) -> torch.Tensor:
    """`wo_b` of the `wo_a` output: N=5120, K=2048. Its TP-partial output
    goes to the next boundary's fused all-reduce."""
    if _serves(linears.wo_b, z.shape[0]):
        assert linears.wo_b is not None
        return linears.wo_b(z)
    return _wo_b_partial(attn, z)


def _wo_b_partial(attn: DeepseekV4Attention, z: torch.Tensor) -> torch.Tensor:
    """The stock `wo_b`'s TP-partial output, without the all-reduce of a
    `wo_b` that reduces its own output."""
    wo_b = attn.wo_b
    if not wo_b.reduce_results:
        return attn._wo_b_proj(z)
    assert wo_b.quant_method is not None
    return wo_b.quant_method.apply(wo_b, z, None)


def o_proj(
    attn: DeepseekV4Attention,
    linears: AttentionLinears,
    out: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """The stock output projection of the attention output `out` (padded
    heads), with `wo_b` on its low-batch call."""
    return dsv41_o_proj(
        attn, out, positions, wo_b_proj=lambda z: wo_b(attn, linears, z)
    )


@dataclass
class SharedMlp:
    """The low-batch providers of a MoE layer's shared expert MLP."""

    mlp: nn.Module
    gate_up: DenseFp8Linear
    down: DenseFp8Linear

    @classmethod
    def of(cls, ffn: "DeepseekV4MoE") -> "SharedMlp | None":
        mlp = ffn.shared_experts
        if mlp is None:
            return None
        gate_up = DenseFp8Linear.of("shared_gate_up", mlp.gate_up_proj)
        down = DenseFp8Linear.of("shared_down", mlp.down_proj)
        if gate_up is None or down is None:
            return None
        return cls(mlp, gate_up, down)

    def serves(self, rows: int) -> bool:
        return self.gate_up.serves(rows) and self.down.serves(rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`DeepseekV4MLP.forward`: gate/up, clamped SiLU-and-mul, down."""
        return self.down(self.mlp.act_fn(self.gate_up(x)))

    @contextmanager
    def installed(self) -> Iterator[None]:
        """Within the block, the MLP's forward (which the MoE runner calls,
        on its shared-expert stream) runs `forward`."""
        self.mlp.forward = self.forward
        try:
            yield
        finally:
            del self.mlp.forward


def shared_installed(
    shared: SharedMlp | None, rows: int
) -> AbstractContextManager[None]:
    """`shared.installed()` where its calls serve `rows` rows."""
    if shared is None or not shared.serves(rows):
        return nullcontext()
    return shared.installed()


@contextmanager
def partial_output(ffn: "DeepseekV4MoE") -> Iterator[None]:
    """Within the block, `ffn` leaves its output TP-partial for the next
    boundary to reduce; the stock forward outside the executor reduces it
    itself."""
    config = ffn.experts.moe_config
    saved = config.skip_final_all_reduce
    config.skip_final_all_reduce = True
    try:
        yield
    finally:
        config.skip_final_all_reduce = saved


def moe(
    ffn: "DeepseekV4MoE",
    shared: SharedMlp | None,
    routed: HopperLowBatchMoE | None,
    x: torch.Tensor,
    input_ids: torch.Tensor | None,
) -> torch.Tensor | MoEOutput:
    """The MoE layer's TP-partial output, or its routed output left
    unfinalized with the shared output.

    Up to `moe.MAX_ROWS` rows the routed and shared experts run the low-batch
    kernel (`routed`), from the stock router's logits, and stop before the
    top-k reduction: the next boundary's fused all-reduce + mHC folds the
    reduction and the shared-expert add into its publish
    (`AllReduceMHC.finalize`). Larger calls run the stock MoE, without its
    final all-reduce, with the shared expert on the low-batch dense calls where
    they serve.
    """
    rows = x.shape[0]
    if routed is None or not routed.serves(rows):
        with shared_installed(shared, rows), partial_output(ffn):
            return ffn(x, input_ids, None)
    router_logits, _ = ffn.experts.gate(x)
    output, shared_output = routed(x, router_logits, input_ids)
    return MoEOutput(routed=output, shared_output=shared_output)
