# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import prod

import torch

from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.utils.math_utils import round_up


class DeepseekV4EagerScratchPool:
    """Model-wide outputs and scratch used inside the attention eager break."""

    _ALIGNMENT = 256

    def __init__(
        self,
        max_num_tokens: int,
        padded_q_heads: int,
        q_head_dim: int,
        index_q_heads: int,
        index_q_head_dim: int,
        index_topk: int,
        device: torch.device | str,
    ) -> None:
        self.max_num_tokens = max_num_tokens
        self.index_topk = index_topk
        self._q = torch.empty(
            (max_num_tokens, padded_q_heads, q_head_dim),
            dtype=torch.bfloat16,
            device=device,
        )

        fp4_specs = (
            ((max_num_tokens, index_q_heads, index_q_head_dim // 2), torch.uint8),
            (
                (
                    max_num_tokens,
                    index_q_heads,
                    index_q_head_dim // MXFP4_BLOCK_SIZE,
                ),
                torch.uint8,
            ),
            ((max_num_tokens, index_q_heads), torch.float32),
        )
        global_specs = (
            ((max_num_tokens, index_topk), torch.int32),
            ((max_num_tokens,), torch.int32),
        )
        compressor_specs = (((max_num_tokens, q_head_dim), torch.float32),)
        # FP4 indexer is C4 only, global mapping after FP4 indexer
        # compressor scratch is C128 only
        # so here we use max instead of sum
        aux_bytes = max(
            self._packed_size(specs)
            for specs in (fp4_specs, global_specs, compressor_specs)
        )
        storage = torch.empty(aux_bytes, dtype=torch.uint8, device=device)

        self._q_outputs: dict[int, torch.Tensor] = {}
        fp4_values, fp4_scales, fp4_weights = self._views(storage, fp4_specs)
        self._fp4_template = (fp4_values, fp4_scales, fp4_weights)
        self._fp4_outputs: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        global_indices, global_lens = self._views(storage, global_specs)
        self._global_template = (global_indices, global_lens)
        self._global_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._compressor_template = self._views(storage, compressor_specs)[0]
        self._compressor_outputs: dict[int, torch.Tensor] = {}
        self._storage = storage

    @classmethod
    def _packed_size(
        cls, specs: tuple[tuple[tuple[int, ...], torch.dtype], ...]
    ) -> int:
        offset = 0
        for shape, dtype in specs:
            offset = round_up(offset, cls._ALIGNMENT) + prod(shape) * dtype.itemsize
        return round_up(offset, cls._ALIGNMENT)

    @classmethod
    def _views(
        cls,
        storage: torch.Tensor,
        specs: tuple[tuple[tuple[int, ...], torch.dtype], ...],
    ) -> list[torch.Tensor]:
        offset = 0
        views = []
        for shape, dtype in specs:
            offset = round_up(offset, cls._ALIGNMENT)
            num_bytes = prod(shape) * dtype.itemsize
            views.append(storage[offset : offset + num_bytes].view(dtype).view(shape))
            offset += num_bytes
        return views

    def q_out(self, num_tokens: int) -> torch.Tensor:
        output = self._q_outputs.get(num_tokens)
        if output is None:
            output = self._q[:num_tokens]
            self._q_outputs[num_tokens] = output
        return output

    def compressor_scratch(self, num_tokens: int) -> torch.Tensor:
        output = self._compressor_outputs.get(num_tokens)
        if output is None:
            output = self._compressor_template[:num_tokens]
            self._compressor_outputs[num_tokens] = output
        return output

    def indexer_q_outputs(
        self,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self._fp4_outputs.get(num_tokens)
        if output is None:
            values, scales, weights = self._fp4_template
            output = (
                values[:num_tokens],
                scales[:num_tokens],
                weights[:num_tokens],
            )
            self._fp4_outputs[num_tokens] = output
        return output

    def global_topk_outputs(
        self, topk_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens, topk = topk_indices.shape
        assert topk == self.index_topk
        output = self._global_outputs.get(num_tokens)
        if output is None:
            indices, lens = self._global_template
            output = (indices[:num_tokens], lens[:num_tokens])
            self._global_outputs[num_tokens] = output
        return output
