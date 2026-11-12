# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA for Inkling2.

Adapters use the checkpoint's names (`model.llm.layers.N...`):

- attention `wq_du` / `wr_du` / `gate_attn` / `attn_sink_proj` /
  `wk_dv` / `wv_dv`, packed into `qkvr` (vLLM's variable-slice merged
  column LoRA layer), and `wo_ud`;
- routed experts `mlp.experts.w13_weight` / `mlp.experts.w2_weight`, one 3D
  tensor per projection: `lora_A` `[E, r, in]`, `lora_B` `[E, out, r]`
  (`w13` rows interleave gate / up like the base weight), packed into
  `mlp`.

The experts are not a vLLM `FusedMoE`, so `Inkling2MoEWithLoRA` wraps
the whole `mlp` (vLLM's manager special-cases modules named `*.experts`)
and adds the deltas between the expert GEMMs through `moe.ExpertsLoRA`,
which needs BF16 experts (the NVFP4 / MXFP8 ones raise).

The attention layers run their GEMMs on `qkvr.weight` / `wo_ud.weight`
directly (`ops.linear`), past the LoRA wrappers' forward, and add the deltas
with `add_linear_lora`.
vLLM only wraps `LinearBase` / `MoERunner` modules; `register` adds
`Inkling2MoE` to the modules it considers and this class to its LoRA layers.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import (
    BaseLayerWithLoRA,
    MergedColumnParallelLinearVariableSliceWithLoRA,
)
from vllm.lora.punica_wrapper import PunicaWrapperBase
from vllm.model_executor.layers.linear import MergedColumnParallelLinear

from .moe import Inkling2MoE
from .ops.grouped_gemm import grouped_gemm
from .ops.seq_parallel import SeqParallel

# Packed modules: the runtime module -> the adapter modules it holds (siblings).
PACKED_MODULES_MAPPING = {
    "qkvr": ["wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv"],
    "mlp": ["mlp_w13", "mlp_w2"],
}
# Adapter name -> module name (``Inkling2ForConditionalGeneration.hf_to_vllm_mapper``).
# The expert projections become ``mlp_w13`` / ``mlp_w2``, siblings of ``mlp``
# that vLLM packs into it like q/k/v into qkv.
LORA_NAME_PREFIXES = {"model.llm.": "model."}
LORA_NAME_SUBSTRINGS = {
    ".mlp.experts.w13_weight.": ".mlp_w13.",
    ".mlp.experts.w2_weight.": ".mlp_w2.",
}


class Inkling2QKVRWithLoRA(MergedColumnParallelLinearVariableSliceWithLoRA):
    """`qkvr` LoRA; its trailing alignment slice (`Inkling2Attention`, at
    some TP sizes) has no adapter."""

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor | None],
        lora_b: torch.Tensor | list[torch.Tensor | None],
    ) -> None:
        if isinstance(lora_a, list) and len(lora_a) < self.n_slices:
            pad = [None] * (self.n_slices - len(lora_a))
            lora_a, lora_b = [*lora_a, *pad], [*lora_b, *pad]
        super().set_lora(index, lora_a, lora_b)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config=None,
        decorate: bool = True,
    ) -> bool:
        return (
            type(source_layer) is MergedColumnParallelLinear
            and packed_modules_list == PACKED_MODULES_MAPPING["qkvr"]
            and not lora_config.fully_sharded_loras
        )


class Inkling2MoEWithLoRA(BaseLayerWithLoRA):
    """`Inkling2MoE` plus per-expert LoRA on `w13` and `w2`."""

    def __init__(self, base_layer: Inkling2MoE) -> None:
        super().__init__()
        self.base_layer = base_layer
        experts = base_layer.experts
        self.num_local_experts = experts.num_experts
        self.expert_start = experts.ep_rank * experts.num_experts
        self.hidden_size = experts.hidden_size
        self.intermediate_size = experts.intermediate_size
        self.n_slices = 2
        self._row_masks: torch.Tensor | None = None

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config=None,
    ) -> None:
        self.max_loras = max_loras
        rank = lora_config.max_lora_rank
        hidden, inter = self.hidden_size, self.intermediate_size
        device = self.base_layer.gate.weight.device

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(
                max_loras,
                self.num_local_experts,
                *shape,
                dtype=lora_config.lora_dtype,
                device=device,
            )

        # (w13, w2); dims [max_loras, local experts, r, in] / [..., out, r].
        self.lora_a_stacked = (zeros(rank, hidden), zeros(rank, inter))
        self.lora_b_stacked = (zeros(2 * inter, rank), zeros(hidden, rank))

    def reset_lora(self, index: int) -> None:
        for stacked in (*self.lora_a_stacked, *self.lora_b_stacked):
            stacked[index].zero_()

    def _local(self, tensor: torch.Tensor) -> torch.Tensor:
        """This rank's experts of an `[E, ...]` adapter tensor; a 2D tensor
        (e.g. vLLM's warmup dummy) applies to every expert."""
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).expand(self.num_local_experts, *tensor.shape)
        return tensor[self.expert_start : self.expert_start + self.num_local_experts]

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor | None],
        lora_b: torch.Tensor | list[torch.Tensor | None],
    ) -> None:
        self.reset_lora(index)
        assert isinstance(lora_a, list) and isinstance(lora_b, list)
        for a, b, a_stacked, b_stacked in zip(
            lora_a, lora_b, self.lora_a_stacked, self.lora_b_stacked
        ):
            if a is None or b is None:
                continue
            a, b = self._local(a), self._local(b)
            rank = a.shape[-2]
            a_stacked[index, :, :rank].copy_(a, non_blocking=True)
            b_stacked[index, :, :, :rank].copy_(b, non_blocking=True)

    def set_mapping(self, punica_wrapper: PunicaWrapperBase) -> None:
        self.punica_wrapper = punica_wrapper

    def forward(
        self, x: torch.Tensor, reduce: bool = True, push: SeqParallel | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        return self.base_layer(x, reduce, push, lora=self)

    def add_w13(
        self,
        gate_up: torch.Tensor,
        x: torch.Tensor,
        perm: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
    ) -> None:
        assert self.punica_wrapper is not None
        # Each row's adapter slot (-1: none), from its token's.
        token_lora = self.punica_wrapper._token_lora_indices[: x.shape[0]]
        row_lora = token_lora[perm // topk]
        slots = torch.arange(self.max_loras, device=x.device)
        self._row_masks = (row_lora[None, :] == slots[:, None]).to(x.dtype)
        self._add(gate_up, x, 0, perm, offsets, topk, gather=True)

    def add_w2(
        self,
        down: torch.Tensor,
        activated: torch.Tensor,
        perm: torch.Tensor,
        offsets: torch.Tensor,
        topk: int,
    ) -> None:
        self._add(down, activated, 1, perm, offsets, topk, gather=False)
        self._row_masks = None

    def _add(self, out, x, proj, perm, offsets, topk, gather) -> None:
        assert self._row_masks is not None
        num_rows = out.shape[0]
        for slot in range(self.max_loras):
            a = self.lora_a_stacked[proj][slot]
            b = self.lora_b_stacked[proj][slot]
            shrunk = grouped_gemm(x, a, perm, offsets, num_rows, topk, gather)
            shrunk.mul_(self._row_masks[slot, :, None])
            grouped_gemm(shrunk, b, perm, offsets, num_rows, topk, False, out=out)

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config=None,
    ) -> bool:
        return isinstance(source_layer, Inkling2MoE)


def add_linear_lora(
    layer: nn.Module,
    out: torch.Tensor,
    x: torch.Tensor,
    slices: Iterable[int] | None = None,
) -> None:
    """Add `layer`'s LoRA deltas for `x` to `out` in place, for callers
    that run the base GEMM on `layer.weight` themselves
    (`Inkling2Attention`). `slices` picks the output slices of a merged
    layer (default: all), whose deltas land in `out`'s leading columns, in
    order. No-op when `layer` has no LoRA. For a row-parallel layer the
    delta is, like the base output, this rank's partial sum.
    """
    if not isinstance(layer, BaseLayerWithLoRA):
        return
    if slices is None:
        slices = range(len(layer.output_slices))
    slices = list(slices)
    sizes = tuple(layer.output_slices[i] for i in slices)
    delta = out.new_zeros(out.shape[0], sum(sizes))
    layer.punica_wrapper.add_lora_linear(
        delta,
        x,
        tuple(layer.lora_a_stacked[i] for i in slices),
        tuple(layer.lora_b_stacked[i] for i in slices),
        1.0,
        sizes,
    )
    out[:, : delta.shape[1]] += delta


_registered = False


def register() -> None:
    """Let vLLM's LoRA manager wrap `Inkling2MoE` with
    `Inkling2MoEWithLoRA`, and `qkvr` with `Inkling2QKVRWithLoRA`
    (ahead of the generic variable-slice class, which would also match it).
    Idempotent."""
    global _registered
    if _registered:
        return
    import vllm.lora.model_manager as model_manager
    import vllm.lora.utils as lora_utils

    lora_utils._all_lora_classes = (
        Inkling2QKVRWithLoRA,
        *lora_utils._all_lora_classes,
        Inkling2MoEWithLoRA,
    )
    get_supported = lora_utils.get_supported_lora_modules
    is_supported = lora_utils.is_supported_lora_module

    def get_supported_lora_modules(model: nn.Module) -> list[str]:
        names = set(get_supported(model))
        for name, module in model.named_modules():
            if isinstance(module, Inkling2MoE):
                names.add(name.split(".")[-1])
        return list(names)

    def is_supported_lora_module(module_name, module, supported_lora_modules):
        if isinstance(module, Inkling2MoE):
            return module_name.rsplit(".", 1)[-1] in supported_lora_modules
        return is_supported(module_name, module, supported_lora_modules)

    lora_utils.get_supported_lora_modules = get_supported_lora_modules
    lora_utils.is_supported_lora_module = is_supported_lora_module
    # model_manager binds both names at import.
    model_manager.get_supported_lora_modules = get_supported_lora_modules  # type: ignore[attr-defined]
    model_manager.is_supported_lora_module = is_supported_lora_module  # type: ignore[attr-defined]
    _registered = True
