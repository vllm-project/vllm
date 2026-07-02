# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QKV projection fuser: `q(x), k(x), v(x)` -> a fused qkv linear + split."""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.transformers.fusers.base import StackedFuser
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    innermost_block,
    is_linear,
    recover_forward,
    replace_expr,
    single_self_call,
)
from vllm.model_executor.models.transformers.utils import (
    log_replacement,
    replace_linear_class,
)
from vllm.model_executor.models.utils import ShardId, maybe_prefix

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)


@dataclass
class QKVFuser(StackedFuser):
    """Fuser for the attention QKV pattern `q(x), k(x), v(x)`."""

    q_name: str
    k_name: str
    v_name: str
    o_name: str | None
    merged_name: ClassVar[str] = "qkv_proj"
    merged_cls: ClassVar[str] = "QKVParallelLinear"

    @property
    def shards(self) -> list[tuple[str, ShardId]]:
        return [(self.q_name, "q"), (self.k_name, "k"), (self.v_name, "v")]

    @classmethod
    def _get_qkv_nodes(
        cls, graph: fx.Graph, module: nn.Module
    ) -> tuple[fx.Node, fx.Node, fx.Node] | None:
        """Search `graph` for the QKV pattern `q(x), k(x), v(x)`."""
        by_input: dict[fx.Node, list[fx.Node]] = {}
        for node in graph.nodes:
            if (
                is_linear(node, module)
                and len(node.args) == 1
                and not node.kwargs
                and isinstance(node.args[0], fx.Node)
                and node.args[0].op == "placeholder"
            ):
                by_input.setdefault(node.args[0], []).append(node)
        triples = [nodes for nodes in by_input.values() if len(nodes) == 3]
        if len(triples) != 1:
            return None

        q_node, k_node, v_node = nodes = triples[0]
        outs = [module.get_submodule(node.target).out_features for node in nodes]
        if len(set(outs)) == 2:
            # q is identified as the larger projection (GQA)
            (q_node,) = (n for n, out in zip(nodes, outs) if outs.count(out) == 1)
            k_node, v_node = (n for n, out in zip(nodes, outs) if outs.count(out) == 2)
            if module.get_submodule(q_node.target).out_features != max(outs):
                return None
        elif len(set(outs)) != 1:
            return None
        return q_node, k_node, v_node

    @classmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "QKVFuser | None":
        if (qkv_nodes := cls._get_qkv_nodes(graph, module)) is None:
            return None
        q, k, v = qkv_nodes
        names = dict(q_name=q.target, k_name=k.target, v_name=v.target)
        predicate = lambda n, c: isinstance(c, nn.Linear) and n not in names.values()
        others = [n for n, c in module.named_children() if predicate(n, c)]
        if len(others) == 1:
            names["o_name"] = others[0]
        return cls(source_cls=type(module).__name__, **names)

    def update_forward(self, module: nn.Module) -> None:
        """Replace `q(x), k(x), v(x)` with `qkv(x).split(sizes, -1)` in source."""
        funcdef, fn = recover_forward(type(module))
        calls = [
            single_self_call(funcdef, name)
            for name in (self.q_name, self.k_name, self.v_name)
        ]
        arg_dumps = {ast.dump(call.args[0]) for call in calls}
        if len(arg_dumps) != 1:
            raise ValueError("projection inputs are written differently")
        # The trace may be partial, so prove projection exclusivity in source:
        # no other linear child may consume the same input (else the matched
        # three may not be q, k and v)
        other_linears = {
            name
            for name, child in module.named_children()
            if isinstance(child, nn.Linear)
        } - {self.q_name, self.k_name, self.v_name}
        for node in ast.walk(funcdef):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in other_linears
                and any(ast.dump(arg) in arg_dumps for arg in node.args)
            ):
                raise ValueError("another linear consumes the same input")
        blocks = [innermost_block(funcdef.body, call) for call in calls]
        if any(found is None for found in blocks):
            raise ValueError("projection calls not found in the function body")
        if len({id(block) for block, _ in blocks}) != 1:
            raise ValueError("projection calls are in different blocks")

        # q(x), k(x), v(x) -> q, k, v = qkv(x).split(self.qkv.split_sizes, -1)
        names = {node.id for node in ast.walk(funcdef) if isinstance(node, ast.Name)}
        temps = [f"{name}_fused" for name in (self.q_name, self.k_name, self.v_name)]
        if names & set(temps):
            raise ValueError("fused temporaries would shadow existing names")
        merged = f"self.{self.merged_name}"
        template = (
            f"{', '.join(temps)} = {merged}(__arg__).split({merged}.split_sizes, -1)"
        )
        assign = ast.parse(template).body[0]
        arg = next(
            node
            for node in ast.walk(assign)
            if isinstance(node, ast.Name) and node.id == "__arg__"
        )
        replace_expr(assign, arg, calls[0].args[0])
        block, index = blocks[0]
        ast.copy_location(assign, block[index])
        block.insert(min(index for _, index in blocks), assign)
        for call, temp in zip(calls, temps):
            replace_expr(funcdef, call, ast.Name(id=temp, ctx=ast.Load()))
        self.fused_forward = compile_forward(funcdef, fn)

    def validate(self, module: nn.Module, model_config: "ModelConfig") -> bool:
        """Shapes must be compatible for a single merged, head-sharded GEMM."""
        q = module.get_submodule(self.q_name)
        k = module.get_submodule(self.k_name)
        v = module.get_submodule(self.v_name)
        head_size = model_config.get_head_size()
        compatible = (
            q.in_features == k.in_features == v.in_features
            and len({proj.bias is None for proj in (q, k, v)}) == 1
            and k.out_features == v.out_features
            and q.out_features % head_size == 0
            and k.out_features % head_size == 0
        )
        if not compatible:
            logger.debug("%s is not compatible with QKV fusion", type(module))
        return compatible

    def update_attrs(
        self,
        module: nn.Module,
        prefix: str,
        model_config: "ModelConfig",
        quant_config: "QuantizationConfig",
    ) -> None:
        head_size = model_config.get_head_size()
        q = module.get_submodule(self.q_name)
        k = module.get_submodule(self.k_name)
        merged = QKVParallelLinear(
            hidden_size=q.in_features,
            head_size=head_size,
            total_num_heads=q.out_features // head_size,
            total_num_kv_heads=k.out_features // head_size,
            bias=q.bias is not None,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, self.merged_name),
            return_bias=False,
        )
        logger.debug(
            "%s: %s, %s: %s, %s: %s -> %s: %s",
            self.q_name,
            q,
            self.k_name,
            k,
            self.v_name,
            module.get_submodule(self.v_name),
            self.merged_name,
            merged,
        )
        # The rewritten forward splits the merged projection into this rank's
        # shard sizes (see `update_forward`)
        merged.split_sizes = [
            merged.num_heads * merged.head_size,
            merged.num_kv_heads * merged.head_size,
            merged.num_kv_heads * merged.v_head_size,
        ]
        setattr(module, self.merged_name, merged)
        # Drop the consumed submodules so their (meta) params are not expected.
        for name in (self.q_name, self.k_name, self.v_name):
            delattr(module, name)
        # If there is an output projection, we know it must be rowwise.
        if self.o_name is not None:
            o_proj_prefix = maybe_prefix(prefix, self.o_name)
            o_proj = module.get_submodule(self.o_name)
            new_o = replace_linear_class(
                o_proj, "rowwise", quant_config, prefix=o_proj_prefix
            )
            setattr(module, self.o_name, new_o)
            log_replacement(o_proj_prefix, o_proj, new_o)
