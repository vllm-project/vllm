# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonicalize tensor partitions to a single ``split_with_sizes`` node.

Functionalization may replay the individual outputs of a multi-output split as
contiguous ``slice`` nodes. This pass converts complete slice partitions back
to ``split_with_sizes`` and coalesces equivalent split nodes so downstream
pattern-matching passes see one canonical split with all users attached.

See Also:
  - vLLM  #33295  (original duplicate-split issue)
  - PyTorch #174472 (upstream CSE gap)
  - PyTorch #194037 (multi-output views replayed as individual slices)

"""

import operator
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import fx

from vllm.logger import init_logger

from ..fx_utils import is_func
from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


@dataclass(frozen=True)
class _SliceInfo:
    node: fx.Node
    dim: int
    start: int
    end: int


class SplitCoalescingPass(VllmInductorPass):
    """Canonicalize tensor partitions to one shared ``split_with_sizes``.

    A complete contiguous slice partition is rewritten from::

        q = slice(qkv, 0, q_size)
        k = slice(qkv, q_size, q_size + kv_size)
        v = slice(qkv, q_size + kv_size, q_size + 2 * kv_size)

    to::

        qkv_split = split_with_sizes(qkv, [q_size, kv_size, kv_size])
        q = qkv_split[0]
        k = qkv_split[1]
        v = qkv_split[2]

    Equivalent splits are also coalesced from::

        q = split_with_sizes(qkv, sizes)[0]
        k = split_with_sizes(qkv, sizes)[1]

    to::

        qkv_split = split_with_sizes(qkv, sizes)
        q = qkv_split[0]
        k = qkv_split[1]
    """

    @staticmethod
    def _get_arg(node: fx.Node, index: int, name: str, default: Any) -> Any:
        """Read an FX argument by position, keyword, or schema default."""
        if len(node.args) > index:
            return node.args[index]
        return node.kwargs.get(name, default)

    @staticmethod
    def _normalize_dim(dim: int, rank: int) -> int:
        return dim + rank if dim < 0 else dim

    @staticmethod
    def _tensor_val(node: fx.Node) -> torch.Tensor | None:
        value = node.meta.get("val")
        return value if isinstance(value, torch.Tensor) else None

    def _slice_info(self, node: fx.Node) -> _SliceInfo | None:
        source = node.args[0]
        source_val = self._tensor_val(source)
        slice_val = self._tensor_val(node)
        # in case someone forgot to set meta !
        if source_val is None or slice_val is None:
            return None

        dim = self._normalize_dim(self._get_arg(node, 1, "dim", 0), source_val.dim())
        extent = source_val.shape[dim]

        # do not support dynamic slices.
        if not isinstance(extent, int):
            return None

        start = self._get_arg(node, 2, "start", None)
        end = self._get_arg(node, 3, "end", None)
        step = self._get_arg(node, 4, "step", 1)
        start = 0 if start is None else start
        end = extent if end is None else end
        if not isinstance(start, int) or not isinstance(end, int) or step != 1:
            return None

        if start < 0 or end < 0:
            return None

        # aten.slice clamps an oversized end to the selected dimension.
        end = min(end, extent)
        if start >= end:
            return None

        return _SliceInfo(node=node, dim=dim, start=start, end=end)

    def _canonicalize_slices(self, graph: fx.Graph) -> int:
        groups: dict[tuple[fx.Node, int], list[_SliceInfo]] = defaultdict(list)

        for node in graph.find_nodes(
            op="call_function", target=torch.ops.aten.slice.Tensor
        ):
            slice_info = self._slice_info(node)
            if slice_info is None:
                continue
            groups[(node.args[0], slice_info.dim)].append(slice_info)

        count = 0
        for slice_group in groups.values():
            if len(slice_group) < 2:
                continue

            source = slice_group[0].node.args[0]
            dim = slice_group[0].dim
            source_val = self._tensor_val(source)
            assert source_val is not None
            extent = source_val.shape[dim]
            assert isinstance(extent, int)

            ordered = sorted(slice_group, key=lambda info: info.start)
            # Ensure they are contiguous partitions.
            if (
                ordered[0].start != 0
                or ordered[-1].end != extent
                or any(
                    left.end != right.start for left, right in zip(ordered, ordered[1:])
                )
            ):
                continue

            split_sizes = [info.end - info.start for info in ordered]
            # Use -1 for the last dimension; otherwise use the positive dimension index.
            split_dim = -1 if dim == source_val.dim() - 1 else dim
            first_slice = slice_group[0].node

            with graph.inserting_before(first_slice):
                split = graph.call_function(
                    torch.ops.aten.split_with_sizes.default,
                    args=(source, split_sizes, split_dim),
                )
                split.meta["val"] = [info.node.meta["val"] for info in ordered]

                replacements: list[tuple[fx.Node, fx.Node]] = []
                for index, info in enumerate(ordered):
                    getitem = graph.call_function(
                        operator.getitem,
                        args=(split, index),
                        type_expr=info.node.type,
                    )
                    getitem.meta = copy(info.node.meta)
                    replacements.append((info.node, getitem))

            for slice_node, getitem in replacements:
                slice_node.replace_all_uses_with(getitem)
                graph.erase_node(slice_node)

            count += 1

        return count

    def _coalesce_splits(self, graph: fx.Graph) -> int:
        count = 0
        split_nodes: dict[tuple[fx.Node, int], list[fx.Node]] = defaultdict(list)

        for node in graph.find_nodes(
            op="call_function", target=torch.ops.aten.split_with_sizes.default
        ):
            if not all(is_func(user, operator.getitem) for user in node.users):
                continue

            source, split_sizes = node.args[:2]
            source_val = self._tensor_val(source)
            if source_val is None:
                continue
            raw_dim = self._get_arg(node, 2, "dim", 0)
            dim = self._normalize_dim(raw_dim, source_val.dim())
            candidates = split_nodes[(source, dim)]
            canonical = next(
                (
                    candidate
                    for candidate in candidates
                    if list(candidate.args[1]) == list(split_sizes)
                ),
                None,
            )
            if canonical is not None:
                node.replace_all_uses_with(canonical)
                graph.erase_node(node)
                count += 1
            else:
                candidates.append(node)

        return count

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        canonicalized = self._canonicalize_slices(graph)
        coalesced = self._coalesce_splits(graph)
        logger.debug(
            "Canonicalized %d slice partitions and coalesced %d split nodes",
            canonicalized,
            coalesced,
        )
