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


def tree_ancestors(
    parents: torch.Tensor, query_start_loc: torch.Tensor
) -> list[torch.Tensor | None]:
    """Per request, ``[n, n]`` bool over its ``n`` query rows: row ``i`` sees
    row ``j`` when ``j`` is ``i`` or an ancestor along ``parents`` (rows of the
    query, -1 for the first). None for a request whose rows form a chain, which
    attends causally."""
    # A readable reference, not a fast path: it walks the tree on the host.
    with gpu_sync_allowed():
        bounds, host_parents = query_start_loc.tolist(), parents.tolist()
    masks: list[torch.Tensor | None] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        local = host_parents[start:end]
        if local == list(range(-1, end - start - 1)):
            masks.append(None)
            continue
        sees = torch.eye(end - start, dtype=torch.bool, device=parents.device)
        for row, parent in enumerate(local):
            if parent >= 0:
                sees[row] |= sees[parent]
        masks.append(sees)
    return masks


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
