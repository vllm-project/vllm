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
from torch.fx.experimental.symbolic_shapes import statically_known_true

from vllm.logger import init_logger

from ..fx_utils import is_func
from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


@dataclass(frozen=True)
class _SliceInfo:
    node: fx.Node
    source: fx.Node
    dim: int
    start: int
    end: int


class SplitCoalescingPass(VllmInductorPass):
    """Canonicalize contiguous slices and coalesce duplicate split nodes."""

    @staticmethod
    def _get_arg(node: fx.Node, index: int, name: str, default: Any) -> Any:
        """Read an FX argument by position, keyword, or schema default."""
        if len(node.args) > index:
            return node.args[index]
        return node.kwargs.get(name, default)

    @staticmethod
    def _normalize_dim(dim: Any, rank: int) -> int | None:
        if not isinstance(dim, int):
            return None
        normalized = dim + rank if dim < 0 else dim
        if normalized < 0 or normalized >= rank:
            return None
        return normalized

    @staticmethod
    def _same_sizes(lhs: Any, rhs: Any) -> bool:
        try:
            lhs_sizes = list(lhs)
            rhs_sizes = list(rhs)
        except TypeError:
            return False
        return len(lhs_sizes) == len(rhs_sizes) and all(
            statically_known_true(lhs_size == rhs_size)
            for lhs_size, rhs_size in zip(lhs_sizes, rhs_sizes)
        )

    @staticmethod
    def _tensor_val(node: fx.Node) -> torch.Tensor | None:
        value = node.meta.get("val")
        return value if isinstance(value, torch.Tensor) else None

    def _slice_info(self, node: fx.Node) -> _SliceInfo | None:
        if not is_func(node, torch.ops.aten.slice.Tensor):
            return None

        source = node.args[0]
        if not isinstance(source, fx.Node):
            return None

        source_val = self._tensor_val(source)
        slice_val = self._tensor_val(node)
        if source_val is None or slice_val is None:
            return None

        raw_dim = self._get_arg(node, 1, "dim", 0)
        dim = self._normalize_dim(raw_dim, source_val.dim())
        if dim is None:
            return None

        extent = source_val.shape[dim]
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

        if source_val.dim() != slice_val.dim():
            return None
        if source_val.dtype != slice_val.dtype or source_val.device != slice_val.device:
            return None

        expected_shape = list(source_val.shape)
        expected_shape[dim] = end - start
        if len(expected_shape) != len(slice_val.shape) or not all(
            statically_known_true(expected == actual)
            for expected, actual in zip(expected_shape, slice_val.shape)
        ):
            return None

        return _SliceInfo(node=node, source=source, dim=dim, start=start, end=end)

    def _canonicalize_slices(self, graph: fx.Graph) -> int:
        groups: dict[tuple[fx.Node, int], list[_SliceInfo]] = defaultdict(list)
        node_order = {node: index for index, node in enumerate(graph.nodes)}

        for node in graph.nodes:
            slice_info = self._slice_info(node)
            if slice_info is None:
                continue
            groups[(slice_info.source, slice_info.dim)].append(slice_info)

        count = 0
        # Process nested partitions before their parent slices. Replacing a parent
        # slice then redirects the nested split's source without invalidating it.
        ordered_groups = sorted(
            groups.items(), key=lambda item: node_order[item[0][0]], reverse=True
        )
        for (source, dim), slices in ordered_groups:
            if len(slices) < 2:
                continue

            source_val = self._tensor_val(source)
            assert source_val is not None
            extent = source_val.shape[dim]
            assert isinstance(extent, int)

            ordered = sorted(slices, key=lambda info: info.start)
            cursor = 0
            for info in ordered:
                if info.start != cursor:
                    break
                cursor = info.end
            else:
                if cursor != extent:
                    continue

                split_sizes = [info.end - info.start for info in ordered]
                # Preserve -1 for final-dimension splits because downstream QKV
                # patterns are traced with dim=-1.
                split_dim = -1 if dim == source_val.dim() - 1 else dim
                first_slice = min(
                    (info.node for info in ordered), key=node_order.__getitem__
                )

                with graph.inserting_before(first_slice):
                    split = graph.call_function(
                        torch.ops.aten.split_with_sizes.default,
                        args=(source, split_sizes, split_dim),
                    )
                    split.meta["val"] = [info.node.meta["val"] for info in ordered]

                    replacements = []
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

        for node in graph.nodes:
            if not is_func(node, torch.ops.aten.split_with_sizes.default):
                continue
            if not all(is_func(user, operator.getitem) for user in node.users):
                continue

            source, split_sizes = node.args[:2]
            if not isinstance(source, fx.Node):
                continue
            source_val = self._tensor_val(source)
            if source_val is None:
                continue
            raw_dim = self._get_arg(node, 2, "dim", 0)
            dim = self._normalize_dim(raw_dim, source_val.dim())
            if dim is None:
                continue

            candidates = split_nodes[(source, dim)]
            canonical = next(
                (
                    candidate
                    for candidate in candidates
                    if self._same_sizes(candidate.args[1], split_sizes)
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
