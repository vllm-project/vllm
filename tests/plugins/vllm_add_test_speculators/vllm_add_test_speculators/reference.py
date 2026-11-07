# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference versions of the tree parts vLLM leaves to a speculator's plugin,
in PyTorch, for correctness only: the tree walk (``tree_accept``), the KV
compaction (``layout:paged_token`` commit) and the tree attention mask."""

from collections.abc import Sequence

import torch

from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.worker.gpu.spec_decode.kernels import (
    PAGED_KV_COMMIT,
    CommitStep,
    KernelContext,
    KernelKey,
    PagedKVLayout,
    SpecKernel,
    SpecLayerSpec,
)
from vllm.v1.worker.gpu.spec_decode.round import (
    AcceptResult,
    TargetTokens,
    VerifyPlan,
)


def walk_tree(plan: VerifyPlan, target: TargetTokens, width: int) -> AcceptResult:
    """From the root, step to the child whose token is the target's token at
    the current row; stop when no child matches. Rows past a request's rows and
    rows drafted -1 are never accepted."""
    num_reqs, num_cols = plan.parent.shape
    device = target.tokens.device
    cols = torch.arange(num_cols, device=device)
    rows = (target.row_start[1:] - target.row_start[:-1]).long()[:, None]
    flat = (target.row_start[:-1].long()[:, None] + cols).clamp(
        max=max(target.tokens.numel() - 1, 0)
    )
    live = (cols > 0) & (cols < rows)
    tokens, drafted = target.tokens[flat], target.drafted[flat]
    parent = plan.parent.long()
    sampled = torch.zeros(num_reqs, width, dtype=torch.int64, device=device)
    path = cols.int().repeat(num_reqs, 1)
    path_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    node = torch.zeros(num_reqs, dtype=torch.int64, device=device)
    alive = rows[:, 0] > 0
    for step in range(num_cols):
        token = tokens.gather(1, node[:, None])[:, 0]
        sampled[:, step] = torch.where(alive, token, sampled[:, step])
        path[:, step] = torch.where(alive, node.int(), path[:, step])
        path_len += alive.int()
        match = live & (parent == node[:, None]) & (drafted == token[:, None])
        child = torch.where(match, cols, num_cols).min(dim=1).values
        alive &= child < num_cols
        node = torch.where(alive, child, node)
    return AcceptResult(path=path, path_len=path_len, sampled=sampled)


def tree_window(
    parents: torch.Tensor, query_start_loc: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The last ``width`` query rows of each request: ``[R, W]`` token indices,
    ``[R, W]`` bool whether the column is a row of the request's query, and
    ``[R, W, W]`` bool: column ``i`` sees column ``j`` when ``j`` is ``i``, an
    ancestor along ``parents`` (rows of the query, -1 for the first) or before
    the query. Static shapes and no host reads, so a CUDA graph can capture it
    and still read each batch's parents."""
    num_reqs = query_start_loc.shape[0] - 1
    cols = torch.arange(width, device=parents.device)
    end = query_start_loc[1:].long()
    # Column of the query's first row: columns before it are context.
    first = (width - (end - query_start_loc[:-1].long()))[:, None]
    tokens = end[:, None] - width + cols
    valid = cols >= first
    # The parent's column; negative when it is before the window, so visible.
    parent = parents[tokens.clamp(min=0)].long() + first
    parent = torch.where(valid, parent, -1)
    sees = torch.eye(width, dtype=torch.bool, device=parents.device)
    sees = sees.repeat(num_reqs, 1, 1) | (cols < first)[:, None, :]
    node = parent
    for _ in range(width - 1):
        sees |= cols == node[..., None]
        node = torch.where(node >= 0, parent.gather(1, node.clamp(min=0)), -1)
    return tokens, valid, sees


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

    def supports(self, layer: SpecLayerSpec) -> bool:
        return True

    def bind(self, ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None:
        self.layouts = ctx.kv_layouts

    def commit(self, step: CommitStep) -> None:
        result, n = step.result, step.num_reqs
        path = result.path[:n].long()
        cols = torch.arange(path.shape[1], device=path.device)
        moved = (cols >= 1) & (cols < result.path_len[:n, None]) & (path != cols)
        # The verify rows are the query's last rows, so row r sits at context + r.
        context = (step.seq_lens[:n] - step.rows[:n]).long()[:, None]
        # Boolean-mask indexing reads the count of moved rows on the host.
        with gpu_sync_allowed():
            req = torch.arange(n, device=path.device)[:, None].expand_as(path)[moved]
            src = (context + path)[moved]
            dst = (context + cols).expand_as(path)[moved]
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


def reference_kernels(vllm_config: object = None) -> dict[KernelKey, SpecKernel]:
    """``spec_kernels`` factory: the reference kernels."""
    return {PAGED_KV_COMMIT: ReferenceKVCompactor()}
