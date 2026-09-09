# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QKV projection fuser: `q(x), k(x), v(x)` -> a fused qkv linear + split."""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.transformers.fusers.base import (
    StackedFuser,
    fused_head_size,
)
from vllm.model_executor.models.transformers.fx_utils import (
    block_chain,
    compile_forward,
    is_linear,
    recover_forward,
    returned_linear,
)
from vllm.model_executor.models.transformers.utils import (
    log_replacement,
    replace_linear_class,
)
from vllm.model_executor.models.utils import ShardId, maybe_prefix

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class QKVFuser(StackedFuser):
    """Fuser for the attention QKV pattern `q(x), k(x), v(x)`."""

    q_name: str
    k_name: str
    v_name: str
    o_name: str | None
    merged_name: ClassVar[str] = "qkv_proj"
    merged_cls_name: ClassVar[str] = "QKVParallelLinear"

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
        # o_proj produces the module's output.
        o_name = returned_linear(graph, module)
        # o_proj must be compatible with the q/k/v projections.
        if o_name in names.values() or (
            o_name is not None
            and module.get_submodule(o_name).in_features
            != module.get_submodule(q.target).out_features
        ):
            o_name = None
        names["o_name"] = o_name
        return cls(source_cls=type(module).__name__, **names)

    def update_forward(self, module: nn.Module) -> None:
        """Replace `q(x), k(x), v(x)` with `qkv(x).split(sizes, -1)` in source.

        A projection may be guarded by an existence check -- a
        `self.<proj> is (not) None` comparison (e.g. Gemma 4's `v_proj`) or a bare
        truthiness test -- which is folded to its constant value (see
        `bypass_existence_guard`). The calls may sit in different branches, so the
        fused GEMM is inserted before the earliest of them, in the innermost block
        that dominates all three.
        """
        funcdef, fn = recover_forward(type(module))
        proj_names = (self.q_name, self.k_name, self.v_name)
        calls = self._unguarded_calls(funcdef, proj_names)
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
        } - set(proj_names)
        for node in ast.walk(funcdef):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in other_linears
                and any(ast.dump(arg) in arg_dumps for arg in node.args)
            ):
                raise ValueError("another linear consumes the same input")

        # Insert the fused GEMM before the earliest call, in the innermost block
        # common to all three (the calls may be split across branches).
        chains = [block_chain(funcdef.body, call) for call in calls]
        if any(not chain for chain in chains):
            raise ValueError("projection calls not found in the function body")
        depth = 0
        for level in zip(*chains):
            if len({id(block) for block, _ in level}) != 1:
                break
            depth += 1
        block = chains[0][depth - 1][0]
        indices = [chain[depth - 1][1] for chain in chains]
        insert_index = min(indices)
        self._check_input_stable(funcdef, module, calls, block, indices)

        # q(x), k(x), v(x) -> q, k, v = qkv(x).split(qkv.output_sizes / qkv.tp_size, -1)
        self._splice_merged_split(funcdef, calls, block, insert_index)
        self.fused_forward = compile_forward(funcdef, fn)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        """Shapes must be compatible for a single merged, head-sharded GEMM."""
        q = module.get_submodule(self.q_name)
        k = module.get_submodule(self.k_name)
        v = module.get_submodule(self.v_name)
        head_size = fused_head_size(module, vllm_config)
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
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> None:
        quant_config = vllm_config.quant_config
        head_size = fused_head_size(module, vllm_config)
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
