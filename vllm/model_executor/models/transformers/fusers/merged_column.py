# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuser for parallel linear projections."""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.models.transformers.fusers.base import StackedFuser
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    innermost_block,
    is_linear,
    recover_forward,
    single_self_call,
)
from vllm.model_executor.models.utils import ShardId, maybe_prefix

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class MergedColumnParallelFuser(StackedFuser):
    """Fuser for merging column-parallel linear projections."""

    linear_names: tuple[str, ...]
    merged_cls_name: ClassVar[str] = "MergedColumnParallelLinear"

    @property
    def merged_name(self) -> str:
        """Programmatic name for the merged projection, based on the original names."""
        # len is used to disambiguate names like `a_b` + `c` vs `a` + `b_c`
        parts = "_".join(f"{len(name)}_{name}" for name in self.linear_names)
        return f"merged_proj_{parts}"

    @property
    def shards(self) -> list[tuple[str, ShardId]]:
        return [(name, index) for index, name in enumerate(self.linear_names)]

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
        by_input: dict[fx.Node, list[fx.Node]] = {}
        for node in graph.nodes:
            if (
                is_linear(node, module)
                and len(node.args) == 1
                and not node.kwargs
                and isinstance(node.args[0], fx.Node)
                # Like QKVFuser/PackedQKVFuser: this fuser has no head or
                # TP-replication awareness, so it must not absorb a QKV-shaped
                # pattern (e.g. behind a norm) a more specific fuser declined.
                and node.args[0].op == "placeholder"
            ):
                by_input.setdefault(node.args[0], []).append(node)
        # A group whose members are not distinct direct children is dropped:
        # the source rewrite addresses each projection as `self.<name>` once.
        groups = [
            (group, names)
            for group in by_input.values()
            if len(group) >= 2 and (names := cls._names(group)) is not None
        ]
        if len(groups) > 1:
            logger.debug(
                "%s has %d fusable sibling-linear groups; skipping fusion "
                "since which one to merge is ambiguous",
                type(module).__name__,
                len(groups),
            )
        if len(groups) != 1:
            return None
        _, names = groups[0]
        fuser = cls(source_cls=type(module).__name__, linear_names=names)
        return None if hasattr(module, fuser.merged_name) else fuser

    def update_forward(self, module: nn.Module) -> None:
        """Replace the parallel calls with one merged call and split."""
        funcdef, fn = recover_forward(type(module))
        calls = [single_self_call(funcdef, name) for name in self.linear_names]
        if len(set(ast.dump(call.args[0]) for call in calls)) != 1:
            raise ValueError("parallel linears read different inputs")
        blocks = [innermost_block(funcdef.body, call) for call in calls]
        if any(found is None for found in blocks):
            raise ValueError("parallel linear calls not found in the function body")
        if len(set(id(block) for block, _ in blocks)) != 1:
            raise ValueError("parallel linear calls are in different blocks")

        block = blocks[0][0]
        index = min(index for _, index in blocks)
        calls_have_name_arg = all(isinstance(call.args[0], ast.Name) for call in calls)
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
            if any(value not in calls for value in values) or not calls_have_name_arg:
                raise ValueError("parallel projections cross other operations")

        # l1(x), l2(x), ... -> merged(x).split(merged.output_sizes / merged.tp_size, -1)
        self._splice_merged_split(funcdef, calls, block, index)
        self.fused_forward = compile_forward(funcdef, fn)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        """Check that the parallel linears are compatible for merging."""
        linear_layers = [module.get_submodule(name) for name in self.linear_names]
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        return (
            len(linear_layers) >= 2
            and len(set(linear.in_features for linear in linear_layers)) == 1
            and len(set(linear.bias is None for linear in linear_layers)) == 1
            and all(linear.out_features % tp_size == 0 for linear in linear_layers)
        )

    def update_attrs(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> None:
        """Replace the module's parallel linears with one merged projection."""
        linear_modules = [module.get_submodule(name) for name in self.linear_names]
        merged = MergedColumnParallelLinear(
            input_size=linear_modules[0].in_features,
            output_sizes=[linear.out_features for linear in linear_modules],
            bias=linear_modules[0].bias is not None,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, self.merged_name),
            return_bias=False,
        )
        setattr(module, self.merged_name, merged)
        for name in self.linear_names:
            delattr(module, name)
        logger.debug(
            "%s -> %s: %s",
            ", ".join(f"{n}: {m}" for n, m in zip(self.linear_names, linear_modules)),
            self.merged_name,
            merged,
        )
