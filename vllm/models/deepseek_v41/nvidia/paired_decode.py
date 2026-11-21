# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward-scoped query unions for the opt-in SM90 sparse backend."""

from dataclasses import dataclass, field

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import ForwardContext
from vllm.models.deepseek_v41.nvidia.ops import small_head_paired_decode as paired
from vllm.models.deepseek_v41.nvidia.ops import small_head_query_union as union
from vllm.triton_utils import triton
from vllm.v1.attention.backends.mla.compressor_utils import get_dspark_swa_index_width
from vllm.v1.worker.workspace import current_workspace_manager


@dataclass
class DecodeStep:
    tokens: int
    last_layer: int = -1
    pairs_ready: bool = False
    groups: dict[tuple, tuple] = field(default_factory=dict)


def decode_step(context: ForwardContext, key: str, layer: int, tokens: int):
    if context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
        return DecodeStep(tokens, layer)
    state = context.additional_kwargs.get(key)
    if state is None or state.tokens != tokens or layer <= state.last_layer:
        state = DecodeStep(tokens)
        context.additional_kwargs[key] = state
    state.last_layer = layer
    return state


def paired_config(tokens: int) -> tuple[int, int, int]:
    return (16 if tokens <= 32 else 8 if tokens <= 160 else 4), 32, 4


class PairedDecode:
    def __init__(self, layer, config: VllmConfig):
        spec = config.speculative_config
        drafts = spec.num_speculative_tokens if spec is not None else 0
        self.capacity = min(
            config.scheduler_config.max_num_batched_tokens,
            config.scheduler_config.max_num_seqs * (drafts + 1),
        )
        self.layer = layer.layer_id
        self.enabled = (
            layer.n_local_heads == 8
            and layer.compress_ratio > 0
            and self.layer < config.model_config.hf_text_config.num_hidden_layers
        )
        self.key = f"dsv41_paired_decode_{id(config)}"
        self.group = (
            layer.kv_source_layer_id,
            layer.index_source_layer_id,
            layer.compress_ratio,
        )
        self.extra_width = config.model_config.hf_text_config.index_topk
        self.swa_width = layer.window_size
        if spec is not None and spec.use_dspark():
            self.swa_width = get_dspark_swa_index_width(layer.window_size, drafts)

    def _persistent(self, tokens: int):
        manager = current_workspace_manager()
        key = self.key
        rows = manager.get_persistent((key, "pairs"), (self.capacity, 2), torch.int32)
        count = manager.get_persistent((key, "count"), (), torch.int32)
        width = 2 * triton.next_power_of_2(self.extra_width)
        outputs = tuple(
            manager.get_persistent((key, self.group, name), shape, torch.int32)[:tokens]
            for name, shape in (
                ("indices", (self.capacity, width)),
                ("multiplicity", (self.capacity, width)),
                ("lengths", (self.capacity,)),
            )
        )
        return rows[:tokens], count, outputs

    @staticmethod
    def _scratch(tokens: int, splits: int, swa_width: int):
        width = 2 * triton.next_power_of_2(swa_width)
        return current_workspace_manager().get_simultaneous(
            ((tokens, splits, 16, 512), torch.float32),
            ((tokens, splits, 16), torch.float32),
            ((tokens, width), torch.int32),
            ((tokens, width), torch.int32),
            ((tokens,), torch.int32),
        )

    def reserve(self):
        if self.enabled:
            self._persistent(self.capacity)
            self._scratch(self.capacity, 16, self.swa_width)

    def run(
        self, context, q, sc, si, sl, ec, ei, el, sink, scale, out, req, valid
    ) -> bool:
        tokens = q.shape[0]
        if (
            not self.enabled
            or not 16 < tokens <= self.capacity
            or ec is None
            or ei is None
            or el is None
            or valid is None
            or sc.shape[0] * sc.shape[1] >= 2**30
            or ec.shape[0] * ec.shape[1] >= 2**30
        ):
            return False
        si, ei = si[:tokens].reshape(tokens, -1), ei[:tokens].reshape(tokens, -1)
        if si.shape[1] > self.swa_width or ei.shape[1] != self.extra_width:
            return False
        rows, count, extra = self._persistent(tokens)
        config = paired_config(tokens)
        scratch = self._scratch(tokens, config[0], si.shape[1])
        partials, swa = scratch[:2], scratch[2:]
        state = decode_step(context, self.key, self.layer, tokens)
        if not state.pairs_ready:
            union._pairs[(1,)](
                req,
                valid,
                rows,
                count,
                tokens,
                triton.next_power_of_2(tokens),
                num_warps=4,
            )
            state.pairs_ready = True
        # Pointer/layout checks only distinguish groups within this forward.
        layout = (ec.data_ptr(), tuple(ec.shape), tuple(ec.stride()))
        if state.groups.get(self.group) != layout:
            union._both[(tokens, 2)](
                rows,
                count,
                si,
                sl,
                *swa,
                ei,
                el,
                *extra,
                si.stride(0),
                si.shape[1],
                triton.next_power_of_2(si.shape[1]),
                ei.stride(0),
                ei.shape[1],
                triton.next_power_of_2(ei.shape[1]),
                num_warps=8,
            )
            state.groups[self.group] = layout
        else:
            union._union[(tokens,)](
                rows,
                count,
                si,
                sl,
                *swa,
                si.stride(0),
                si.shape[1],
                triton.next_power_of_2(si.shape[1]),
                True,
                num_warps=4,
            )
        out.zero_()
        paired.run(
            q, sc, ec, sink, scale, out, (rows, count, [swa, extra]), partials, config
        )
        return True
