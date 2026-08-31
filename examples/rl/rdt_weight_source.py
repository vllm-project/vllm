# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``WeightSource`` for ``rlhf_sharded_rdt_small_ep.py``.

A trainer's producer job is to present its weights under the names the inference
side expects, which for vLLM means HF CHECKPOINT names. Transformers does not
store MoE experts that way -- it fuses each layer's into ``[E, ...]`` tensors --
so that is the one conversion this file does. Kept beside the example rather than
inside it so the example stays about the sync itself, alongside
``rdt_vllm_serve.py``.
"""

import regex as re
import torch

from vllm.distributed.weight_transfer import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.base import materialize_full_tensor

# Transformers fuses a layer's experts into ``[E, ...]`` tensors; the checkpoint
# stores them per expert. This matches the fused parameter names.
_FUSED_EXPERT_RE = re.compile(r"^(.*\.experts)\.(gate_up_proj|down_proj)$")


class CheckpointNameSource(WeightSource):
    """`WeightSource` over an FSDP2 model that publishes HF CHECKPOINT names.

    Transformers keeps each layer's experts fused as ``[E, ...]`` tensors, but
    vLLM's MoE loaders are written against the per-expert checkpoint entries --
    only ``qwen{2,3}_moe`` also accept the fused names, so publishing them
    straight from ``named_parameters()`` limits the example to one family and
    fails the consumer's bake on ``experts.gate_up_proj`` everywhere else.
    Splitting them back is also what a real trainer does, since converting an
    internal layout to checkpoint names is the normal producer job.

    Every rank publishes the whole model: fused params are all-gathered and then
    sliced per expert. Ownership stays uniform (no ``held_names``) -- serving
    only the experts a rank holds is a real trainer's concern, not an example's.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    @staticmethod
    def _expand(name: str, param: torch.Tensor) -> list[tuple[str, tuple]]:
        """The checkpoint entries ``name`` contributes, as (name, shape).

        ``gate_up_proj`` is ``[E, 2I, H]`` with gate in rows ``:I`` and up in
        ``I:``; ``down_proj`` is ``[E, H, I]``. Anything else is already a
        checkpoint name and passes through.
        """
        m = _FUSED_EXPERT_RE.match(name)
        if m is None:
            return [(name, tuple(param.shape))]
        prefix, kind = m.group(1), m.group(2)
        experts, rows, cols = param.shape
        if kind == "down_proj":
            return [
                (f"{prefix}.{e}.down_proj.weight", (rows, cols)) for e in range(experts)
            ]
        half = rows // 2
        return [
            (f"{prefix}.{e}.{proj}_proj.weight", (half, cols))
            for e in range(experts)
            for proj in ("gate", "up")
        ]

    def metadata(self) -> list[ParamMeta]:
        return [
            ParamMeta(entry_name, param.dtype, shape)
            for name, param in self._module.named_parameters()
            for entry_name, shape in self._expand(name, param)
        ]

    def __iter__(self):
        for name, param in self._module.named_parameters():
            m = _FUSED_EXPERT_RE.match(name)
            if m is None:
                yield name, materialize_full_tensor(param)
                continue
            # One gather per FUSED param, not per expert: the views below are
            # into that one tensor, and every expert of a layer rides in the
            # same gather group, so it stays resident exactly as long as needed.
            prefix, kind = m.group(1), m.group(2)
            full = materialize_full_tensor(param)
            if kind == "down_proj":
                for e in range(full.shape[0]):
                    yield f"{prefix}.{e}.down_proj.weight", full[e]
            else:
                half = full.shape[1] // 2
                for e in range(full.shape[0]):
                    yield f"{prefix}.{e}.gate_proj.weight", full[e, :half, :]
                    yield f"{prefix}.{e}.up_proj.weight", full[e, half:, :]
