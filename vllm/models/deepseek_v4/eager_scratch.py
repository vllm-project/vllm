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
        *,
        allocate_q: bool = True,
    ) -> None:
        self.max_num_tokens = max_num_tokens
        self.index_topk = index_topk
        # DeepseekV4Attention writes Q in place whenever padded_q_heads equals
        # the per-rank head count, and only falls back to this buffer otherwise.
        # padded_q_heads is model-wide (get_padded_num_q_heads is a pure
        # classmethod of the single attention class the process selects), so the
        # caller can decide once at construction whether the buffer can ever be
        # read. On the default SM12x config it cannot, and it is by far the
        # largest thing in the pool -- 256 MiB at TP=2 with the standard
        # max_num_batched_tokens=8192, against 36 MiB for all the aux views
        # combined -- charged against KV-cache headroom for its whole lifetime.
        self._q = (
            torch.empty(
                (max_num_tokens, padded_q_heads, q_head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            if allocate_q
            else None
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
        # These three families each get their OWN bytes -- sum, not max.
        #
        # This used to be max(), on the reasoning that the FP4 indexer is C4-only,
        # the compressor is C128-only, and the global mapping runs after the FP4
        # indexer, so the three are temporally disjoint and can alias from offset
        # 0. That is a claim about ordering, and it does not survive the attention
        # eager break running the indexer/compressor on parallel aux streams:
        # under concurrent mixed prefill+decode one consumer clobbers another
        # mid-read, producing garbled top-k indices -> attention over the wrong
        # KV -> corrupted output for whichever request happens to be co-batched.
        #
        # Reported with a clean bisect on vllm-project/vllm#41834 (tobymao,
        # TP=4 SM12x, 1M context): pool active 7/7 rounds corrupt, the same build
        # with the pool disabled 0/2, a pre-pool build 0/2. The symptom is leaked
        # BOS tokens mid-sentence, multilingual token salad, terminal single-token
        # repetition -- and it reproduces with speculative decoding both on and
        # off, so it is not a drafter problem.
        #
        # Separating them costs the aux views' whole footprint, ~36 MiB against
        # the 256 MiB Q buffer this pool exists to manage, and it removes the
        # ordering assumption entirely rather than trying to prove it holds.
        # Proving a temporal-disjointness invariant across parallel streams needs
        # events; buying it outright is cheaper and cannot regress.
        fp4_bytes = self._packed_size(fp4_specs)
        global_bytes = self._packed_size(global_specs)
        compressor_bytes = self._packed_size(compressor_specs)
        storage = torch.empty(
            fp4_bytes + global_bytes + compressor_bytes,
            dtype=torch.uint8,
            device=device,
        )

        self._q_outputs: dict[int, torch.Tensor] = {}
        fp4_values, fp4_scales, fp4_weights = self._views(storage, fp4_specs, 0)
        self._fp4_template = (fp4_values, fp4_scales, fp4_weights)
        self._fp4_outputs: dict[
            int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        global_indices, global_lens = self._views(storage, global_specs, fp4_bytes)
        self._global_template = (global_indices, global_lens)
        self._global_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._compressor_template = self._views(
            storage, compressor_specs, fp4_bytes + global_bytes
        )[0]
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
        base: int,
    ) -> list[torch.Tensor]:
        """Carve views for one template family starting at ``base``.

        ``base`` is required rather than defaulting to 0: every caller used to
        take the default, which is exactly how all three families came to alias
        the same bytes.
        """
        offset = base
        views = []
        for shape, dtype in specs:
            offset = round_up(offset, cls._ALIGNMENT)
            num_bytes = prod(shape) * dtype.itemsize
            views.append(storage[offset : offset + num_bytes].view(dtype).view(shape))
            offset += num_bytes
        return views

    def q_out(self, num_tokens: int) -> torch.Tensor:
        if self._q is None:
            # Reaching here means padded_q_heads != the per-rank head count
            # after all, so the construction-time decision was wrong. Fail
            # loudly: silently re-allocating would hide a real config drift.
            raise RuntimeError(
                "DeepseekV4EagerScratchPool was built with allocate_q=False, so "
                "attention was expected to write Q in place. Rebuild the pool "
                "with allocate_q=True."
            )
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
