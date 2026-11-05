# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference versions of the tree parts vLLM leaves to a speculator's plugin,
in PyTorch, for correctness only: the KV compaction (``layout:paged_token``
commit)."""

from collections.abc import Mapping, Sequence

import torch

from vllm.v1.spec_decode.resources import PLAN_KINDS, PlanKind
from vllm.v1.worker.gpu.spec_decode.kernels import (
    PAGED_KV_COMMIT,
    CommitStep,
    KernelContext,
    KernelKey,
    PagedKVLayout,
    SpecKernel,
    SpecLayerSpec,
)

# Caches copy as integers of their element size (fp8 has no index_put on CPU).
_AS_INT = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _kernel_slot(
    layout: PagedKVLayout, req: torch.Tensor, position: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (block, slot) of each request's token at ``position``."""
    table_block = layout.table_block_size
    block = layout.block_table[req, position // table_block].long()
    token = block * table_block + position % table_block
    return token // layout.block_size, token % layout.block_size


class ReferenceKVCompactor(SpecKernel):
    """Copies the KV of accepted path row ``path[b, j]`` to chain slot ``j``."""

    def __init__(self) -> None:
        self.layouts: tuple[PagedKVLayout, ...] = ()

    def supports(self, layer: SpecLayerSpec) -> frozenset[PlanKind]:
        return frozenset(PLAN_KINDS)

    def bind(self, ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None:
        self.layouts = ctx.kv_layouts

    def commit(self, step: CommitStep) -> None:
        result, n = step.result, step.num_reqs
        path = result.path[:n].long()
        cols = torch.arange(path.shape[1], device=path.device)
        moved = (cols >= 1) & (cols < result.path_len[:n, None]) & (path != cols)
        # The verify rows are the query's last rows, so row r sits at context + r.
        context = (step.seq_lens[:n] - step.rows[:n]).long()[:, None]
        req = torch.arange(n, device=path.device)[:, None].expand_as(path)[moved]
        src, dst = (context + path)[moved], (context + cols).expand_as(path)[moved]
        # path[j] >= j: no source is an earlier destination, so read all first.
        for layout in self.layouts:
            src_at, dst_at = (
                _kernel_slot(layout, req, src),
                _kernel_slot(layout, req, dst),
            )
            for cache in layout.caches:
                view = cache.view(_AS_INT[cache.element_size()]).as_strided(
                    layout.shape, layout.strides
                )
                view[dst_at] = view[src_at]


class ReferenceKernels:
    """``spec_kernel_providers`` factory for the reference kernels."""

    name = "reference"

    def __init__(self, vllm_config: object = None) -> None:
        self._kernels: dict[KernelKey, SpecKernel] = {
            PAGED_KV_COMMIT: ReferenceKVCompactor()
        }

    def kernels(self) -> Mapping[KernelKey, SpecKernel]:
        return self._kernels
