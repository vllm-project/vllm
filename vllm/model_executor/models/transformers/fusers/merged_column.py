# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuser for parallel linear projections."""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.models.transformers.fusers.base import (
    StackedFuser,
    local_output_sizes,
)
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    innermost_block,
    is_linear,
    recover_forward,
    replace_expr,
    single_self_call,
)
from vllm.model_executor.models.utils import ShardId, maybe_prefix

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class MergedColumnParallelFuser(StackedFuser):
    """Fuse any number of same-input linears into one merged projection.

    Deliberately absent from `fuser.FUSERS`: which sibling linears may share a
    column-parallel GEMM is a semantic question, so subclasses opt in (fusing
    every group blindly would replace head-aware QKV sharding, for one).
    """

    linear_names: tuple[str, ...]
    merged_name: ClassVar[str] = "merged_proj"
    merged_cls_name: ClassVar[str] = "MergedColumnParallelLinear"

    @property
    def shards(self) -> list[tuple[str, ShardId]]:
        return [(name, index) for index, name in enumerate(self.linear_names)]

    @classmethod
    def sibling_groups(cls, graph: fx.Graph, module: nn.Module) -> list[list[fx.Node]]:
        """Groups of sibling linears reading the same input, each fusable.

        A group whose members are not distinct direct children is dropped: the
        source rewrite addresses each projection as `self.<name>` exactly once.
        """
        if hasattr(module, cls.merged_name):
            return []
        by_input: dict[fx.Node, list[fx.Node]] = {}
        for node in graph.nodes:
            if (
                is_linear(node, module)
                and len(node.args) == 1
                and not node.kwargs
                and isinstance(node.args[0], fx.Node)
            ):
                by_input.setdefault(node.args[0], []).append(node)
        groups = [nodes for nodes in by_input.values() if len(nodes) >= 2]
        return [group for group in groups if cls._names(group) is not None]

    @staticmethod
    def _names(group: list[fx.Node]) -> tuple[str, ...] | None:
        names = tuple(str(node.target) for node in group)
        if len(set(names)) != len(names) or any("." in name for name in names):
            return None
        return names

    @classmethod
    def match(
        cls, graph: fx.Graph, module: nn.Module
    ) -> "MergedColumnParallelFuser | None":
        """Fuse the module's sibling linears when there is only one such group."""
        groups = cls.sibling_groups(graph, module)
        if len(groups) != 1:
            return None
        if (names := cls._names(groups[0])) is None:
            return None
        # Semantic subclasses add their own fields after reusing this match.
        return MergedColumnParallelFuser(
            source_cls=type(module).__name__, linear_names=names
        )

    def update_forward(self, module: nn.Module) -> None:
        """Replace the parallel calls with one merged call and split."""
        funcdef, fn = recover_forward(type(module))
        calls = [single_self_call(funcdef, name) for name in self.linear_names]
        if len({ast.dump(call.args[0]) for call in calls}) != 1:
            raise ValueError("parallel linears read different inputs")
        blocks = [innermost_block(funcdef.body, call) for call in calls]
        if any(found is None for found in blocks):
            raise ValueError("parallel linear calls not found in the function body")
        if len({id(block) for block, _ in blocks}) != 1:
            raise ValueError("parallel linear calls are in different blocks")

        block = blocks[0][0]
        index = min(index for _, index in blocks)
        # Moving projections must not cross operations that can change their input.
        for statement in block[index : max(index for _, index in blocks) + 1]:
            if not isinstance(statement, (ast.Assign, ast.Return)):
                raise ValueError("parallel projections cross other operations")
            if isinstance(statement, ast.Assign) and any(
                not isinstance(target, ast.Name) for target in statement.targets
            ):
                raise ValueError("parallel projection assignment has side effects")
            value = statement.value
            values = value.elts if isinstance(value, ast.Tuple) else [value]
            if any(value not in calls for value in values) or any(
                not isinstance(call.args[0], ast.Name) for call in calls
            ):
                raise ValueError("parallel projections cross other operations")

        names = {node.id for node in ast.walk(funcdef) if isinstance(node, ast.Name)}
        temps = [f"_vllm_merged_{index}" for index in range(len(calls))]
        if names & set(temps):
            raise ValueError("fused temporaries would shadow existing names")
        targets = ", ".join(temps)
        sections = local_output_sizes(self.merged_name)
        source = f"{targets} = self.{self.merged_name}(__arg__).split({sections}, -1)"
        assign = ast.parse(source).body[0]
        arg = next(
            node
            for node in ast.walk(assign)
            if isinstance(node, ast.Name) and node.id == "__arg__"
        )
        replace_expr(assign, arg, calls[0].args[0])
        ast.copy_location(assign, block[index])
        block.insert(index, assign)
        for call, temp in zip(calls, temps):
            replace_expr(funcdef, call, ast.Name(id=temp, ctx=ast.Load()))
        self.fused_forward = compile_forward(funcdef, fn)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        """Check that the projections can share one column-parallel layer."""
        linears = [module.get_submodule(name) for name in self.linear_names]
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        return (
            len(linears) >= 2
            and len({linear.in_features for linear in linears}) == 1
            and len({linear.bias is None for linear in linears}) == 1
            and all(linear.out_features % tp_size == 0 for linear in linears)
        )

    def update_attrs(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> None:
        """Replace the source projections with their merged equivalent."""
        linears = [module.get_submodule(name) for name in self.linear_names]
        merged = MergedColumnParallelLinear(
            input_size=linears[0].in_features,
            output_sizes=[linear.out_features for linear in linears],
            bias=linears[0].bias is not None,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, self.merged_name),
            return_bias=False,
        )
        setattr(module, self.merged_name, merged)
        for name in self.linear_names:
            delattr(module, name)
