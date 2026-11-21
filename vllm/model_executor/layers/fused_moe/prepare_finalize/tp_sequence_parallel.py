# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from contextlib import contextmanager

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)

# Below this many gathered rows the bf16 gather is latency-bound, and AITER's
# tuned small-M kernels take bf16 input (quantizing in-kernel), so
# pre-quantized input would select different kernels and change numerics.
MXFP4_DISPATCH_MIN_TOKENS = 1024


def _gather_packed(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """All-gather row-aligned tensors over TP as a single bf16-typed buffer.

    One collective instead of one per tensor; the bf16 view keeps the gather
    on the custom all-gather path, which does not take 1-byte dtypes.
    """
    rows = tensors[0].shape[0]
    byte_views = [t.contiguous().view(torch.uint8).view(rows, -1) for t in tensors]
    widths = [v.shape[1] for v in byte_views]
    pad = sum(widths) % 2
    if pad:
        byte_views.append(byte_views[0].new_empty(rows, pad))
    packed = torch.cat(byte_views, dim=1).view(torch.bfloat16)
    gathered = get_tp_group().all_gather(packed, dim=0).view(torch.uint8)
    split_widths = widths + ([pad] if pad else [])
    parts = [gathered.new_empty(gathered.shape[0], w) for w in split_widths]
    # One copy kernel for all parts instead of one per tensor.
    torch.split_with_sizes_copy(gathered, split_widths, dim=1, out=parts)
    return [
        part.view(t.dtype).view(-1, *t.shape[1:]) for t, part in zip(tensors, parts)
    ]


class MoEPrepareAndFinalizeTPSequenceParallel(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize for TP-sharded experts fed sequence-parallel tokens.

    Each TP rank holds an equal share of the tokens and has routed them
    locally. prepare() all-gathers the activations together with the top-k
    weights and ids over the TP group; finalize() reduce-scatters the partial
    expert outputs back to the local rows, so the output is fully reduced.

    When the experts accept pre-quantized input (`defer_input_quant=False`)
    and the weights are MXFP4 W4A4, large batches are quantized to MXFP4
    before the gather (about 3.5x fewer bytes). Smaller batches stay bf16 and
    are quantized by the experts, so those experts must accept both. Below
    `gather_before_router`, the caller gathers the bf16 tokens before routing
    and runs prepare() under `input_gathered()`; finalize() is unchanged.
    """

    def __init__(self, tp_size: int) -> None:
        super().__init__()
        self.tp_size = tp_size
        self._input_gathered = False

    def gather_before_router(self, num_local_tokens: int) -> bool:
        """Whether the caller should all-gather bf16 tokens before routing.

        Small batches are launch-bound: gathering the activations alone and
        routing every token on each rank is cheaper than packing the top-k
        results into the gather.
        """
        return num_local_tokens * self.tp_size < MXFP4_DISPATCH_MIN_TOKENS

    @contextmanager
    def input_gathered(self) -> Iterator[None]:
        """Treat prepare() input as already gathered over the TP group."""
        self._input_gathered = True
        try:
            yield
        finally:
            self._input_gathered = False

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self.tp_size

    def output_is_reduced(self) -> bool:
        return True

    def _quantize_for_dispatch(
        self,
        a1: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> bool:
        return (
            not defer_input_quant
            and quant_config.use_mxfp4_w4a4
            and a1.shape[0] * self.tp_size >= MXFP4_DISPATCH_MIN_TOKENS
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        if self._input_gathered:
            return a1, None, None, topk_ids, topk_weights

        if self._quantize_for_dispatch(a1, quant_config, defer_input_quant):
            from aiter import QuantType, get_hip_quant

            # Packed MXFP4 [M, K // 2] with unshuffled e8m0 scales; AITER
            # fused_moe sorts the scales itself for pre-quantized input.
            a1q, a1q_scale = get_hip_quant(QuantType.per_1x32)(a1, shuffle=False)
            a1q_scale = a1q_scale[: a1.shape[0]]
            a1q, a1q_scale, topk_weights, topk_ids = _gather_packed(
                [a1q, a1q_scale, topk_weights, topk_ids]
            )
        else:
            a1q, topk_weights, topk_ids = _gather_packed([a1, topk_weights, topk_ids])
            a1q_scale = None

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        if self._input_gathered:
            # The caller reduce-scatters, mirroring its gather.
            weight_and_reduce_impl.apply(
                output=output,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            return
        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        output.copy_(get_tp_group().reduce_scatter(out, dim=0))
