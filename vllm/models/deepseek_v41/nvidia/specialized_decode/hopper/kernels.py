# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernel choices of a Hopper executor step, one function per kernel.

Each function runs a Hopper low-batch kernel for the rows it serves and the
stock kernel of the stock layer otherwise. Every Hopper kernel reads the
tensors the stock layer reads, in place (Humming W8A16 dense weights, Humming
W4A8 experts, the FlashMLA KV and index caches), so every step outside the
executor and every prefill keeps running the stock forward on a single copy of
the weights; ``wo_a``, which the stock SM90 layer keeps only as BF16, is the one
exception (``dense.fp8_wo_a``).

Each layer's low-batch kernels are providers (``DenseProvider``), one per layer
and seam, built with the executor: today ``dense.DenseFp8Linear`` up to
``dense.MAX_ROWS`` rows; a weight-preloading or fused kernel replaces one by
implementing the same interface. Every launch here is captured into a FULL CUDA
graph of the executor only, and the Hopper kernels chain with programmatic
dependent launch. The sparse-MLA decode is ``sparse_mla.FlashMlaDecode.run``.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.fused_moe.moe_output import MoEOutput
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v41.attention import DeepseekV4Attention, DeepseekV4Indexer
from vllm.models.deepseek_v41.common.engram import Engram

from ...ops.o_proj import dsv41_o_proj
from .dense import DenseFp8Linear, fp8_wo_a
from .moe import HopperLowBatchMoE

if TYPE_CHECKING:
    from ...model import DeepseekV4MoE
    from ..boundary.all_reduce import BoundaryAllReduce


class DenseProvider(Protocol):
    """One layer's low-batch kernel of one dense seam."""

    seam: str
    n: int
    k: int

    def serves(self, rows: int) -> bool:
        """Whether a call of ``rows`` rows runs here (host state only)."""
        ...

    def publishes(self, rows: int) -> bool:
        """Whether such a call can also publish into the fused all-reduce."""
        ...

    def __call__(
        self,
        x: torch.Tensor,
        all_reduce: "BoundaryAllReduce | None" = None,
    ) -> torch.Tensor:
        """The product, published into ``all_reduce`` when given."""
        ...

    def warm_up(self, rows: int, publish: bool = False) -> None:
        """Compile what a ``rows``-row call launches, before graph capture."""
        ...


@dataclass
class AttentionLinears:
    """The low-batch providers of one attention layer's dense linears."""

    wqa_wkv: DenseProvider | None
    wq_b: DenseProvider | None
    index_wq_b: DenseProvider | None
    wo_a: DenseProvider | None
    wo_b: DenseProvider | None

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

    def all(self) -> list[DenseProvider]:
        linears = (self.wqa_wkv, self.wq_b, self.index_wq_b, self.wo_a, self.wo_b)
        return [linear for linear in linears if linear is not None]


def _serves(linear: DenseProvider | None, rows: int) -> bool:
    return linear is not None and linear.serves(rows)


def wqa_wkv(
    attn: DeepseekV4Attention, linears: AttentionLinears, x: torch.Tensor
) -> torch.Tensor:
    """``fused_wqa_wkv`` of the BF16 layer input: replicated, N=1792, K=5120."""
    if _serves(linears.wqa_wkv, x.shape[0]):
        assert linears.wqa_wkv is not None
        return linears.wqa_wkv(x)
    return attn._fused_wqa_wkv_gemm(x)


def wq_b(
    attn: DeepseekV4Attention, linears: AttentionLinears, qr: torch.Tensor
) -> torch.Tensor:
    """``wq_b`` of the normalized query: N=8192, K=1280."""
    if _serves(linears.wq_b, qr.shape[0]):
        assert linears.wq_b is not None
        return linears.wq_b(qr)
    return attn._wq_b_proj(qr)


def index_wq_b(
    indexer: DeepseekV4Indexer, linears: AttentionLinears, qr: torch.Tensor
) -> torch.Tensor:
    """The indexer's ``wq_b`` of the normalized query, on aux stream 2:
    replicated, N=4096, K=1280."""
    if _serves(linears.index_wq_b, qr.shape[0]):
        assert linears.index_wq_b is not None
        return linears.index_wq_b(qr)
    return indexer._wq_b_proj(qr, None)


def wo_b(
    attn: DeepseekV4Attention,
    linears: AttentionLinears,
    z: torch.Tensor,
    all_reduce: "BoundaryAllReduce | None",
) -> torch.Tensor:
    """``wo_b`` of the ``wo_a`` output: N=5120, K=2048. Its TP-partial output
    goes to the next boundary's fused all-reduce; where the CuTe kernel runs,
    it also publishes into that all-reduce's mailbox."""
    linear = linears.wo_b
    rows = z.shape[0]
    if linear is None or not linear.serves(rows):
        return _wo_b_partial(attn, z)
    return linear(z, all_reduce if linear.publishes(rows) else None)


def _wo_b_partial(attn: DeepseekV4Attention, z: torch.Tensor) -> torch.Tensor:
    """The stock ``wo_b`` without the all-reduce of a ``wo_b`` that reduces its
    output (the DSpark draft's stock layers)."""
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
    all_reduce: "BoundaryAllReduce | None",
) -> torch.Tensor:
    """The output projection of the attention output ``out`` (padded heads):
    inverse RoPE, the grouped ``wo_a`` (FP8, or the stock BF16 bmm on the
    weight dequantized at load), then ``wo_b``."""

    def project(z: torch.Tensor) -> torch.Tensor:
        return wo_b(attn, linears, z, all_reduce)

    if not _serves(linears.wo_a, out.shape[0]):
        return dsv41_o_proj(attn, out, positions, wo_b_proj=project)
    assert linears.wo_a is not None
    # [tokens, groups, heads_per_group * head_dim], group-major in memory.
    rotated, _ = fused_inv_rope_fp8_quant(
        out[:, : attn.n_local_heads],
        positions,
        attn.rotary_emb.cos_sin_cache,
        n_groups=attn.n_local_groups,
        heads_per_group=attn.n_local_heads // attn.n_local_groups,
        nope_dim=attn.nope_head_dim,
        rope_dim=attn.rope_head_dim,
        quantize=False,
    )
    return project(linears.wo_a(rotated))


@dataclass
class SharedMlp:
    """The low-batch providers of a MoE layer's shared expert MLP."""

    mlp: nn.Module
    gate_up: DenseProvider
    down: DenseProvider

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
        """``DeepseekV4MLP.forward``: gate/up, clamped SiLU-and-mul, down."""
        return self.down(self.mlp.act_fn(self.gate_up(x)))

    @contextmanager
    def installed(self) -> Iterator[None]:
        """Within the block, the MLP's forward (which the MoE runner calls,
        on its shared-expert stream) runs ``forward``."""
        self.mlp.forward = self.forward
        try:
            yield
        finally:
            del self.mlp.forward


def shared_installed(
    shared: SharedMlp | None, rows: int
) -> AbstractContextManager[None]:
    """``shared.installed()`` where its calls serve ``rows`` rows."""
    if shared is None or not shared.serves(rows):
        return nullcontext()
    return shared.installed()


def moe(
    ffn: "DeepseekV4MoE",
    shared: SharedMlp | None,
    routed: HopperLowBatchMoE | None,
    x: torch.Tensor,
    input_ids: torch.Tensor | None,
) -> torch.Tensor | MoEOutput:
    """The MoE layer's TP-partial output, or its routed output left
    unfinalized with the shared output.

    Up to ``moe.MAX_ROWS`` rows the routed experts run the low-batch kernel
    (``routed``), from the stock router's logits (on hash-routed layers, its
    top-k), and stop before the top-k reduction: the next boundary's fused
    all-reduce + mHC folds the reduction and the shared-expert add into its
    publish (``AllReduceMHC.finalize``). The kernel also runs the shared expert
    up to ``moe.SHARED_MAX_ROWS`` rows; above that the stock shared expert runs
    on its side stream, on the low-batch dense calls where they serve. Larger
    calls run the stock MoE.
    """
    rows = x.shape[0]
    if routed is None or not routed.serves(rows):
        with shared_installed(shared, rows):
            return ffn(x, input_ids, None)
    runner = ffn.experts
    runner.routed_experts._ensure_moe_quant_config_init()
    side = runner._shared_experts
    fused = routed.fuses_shared(rows)
    overlapping = False
    with shared_installed(shared, rows):
        if not fused:
            assert side is not None
            overlapping = side.maybe_forward_async(x)
        router_logits, _ = runner.gate(x)
        if not fused:
            runner._maybe_apply_shared_experts(x, SharedExpertsOrder.NO_OVERLAP)
    if routed.hash_routed:
        topk_weights, topk_ids = runner.router.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            topk_indices_dtype=runner._quant_method.topk_indices_dtype,
            input_ids=input_ids,
        )
        output, shared_output = routed(x, topk_ids=topk_ids, topk_weights=topk_weights)
    else:
        output, shared_output = routed(
            x, router_logits=router_logits, input_ids=input_ids
        )
    if not fused:
        if overlapping:
            side.wait()
        # Read once: the property hands the output over and clears it.
        shared_output = side.output
    assert shared_output is not None
    return MoEOutput(routed=output, shared_output=shared_output)


def engram_wkv(
    linears: dict[Engram, DenseProvider],
) -> Callable[[Engram, torch.Tensor], torch.Tensor]:
    """The Engram WKV projection (column-parallel N=6400, K=6144, gathered over
    the TP group) of the prefetch side stream."""

    def wkv(engram: Engram, x: torch.Tensor) -> torch.Tensor:
        linear = linears.get(engram)
        if linear is None or not linear.serves(x.shape[0]):
            return engram.wkv(x)
        out = linear(x)
        if engram.wkv.gather_output and engram.wkv.tp_size > 1:
            out = tensor_model_parallel_all_gather(out)
        return out

    return wkv
