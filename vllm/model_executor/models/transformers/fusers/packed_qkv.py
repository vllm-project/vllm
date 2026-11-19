# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-QKV fuser: `c_attn(x).split((q, kv, kv))` -> a `QKVParallelLinear`.

Also matches a packed projection sliced into q, k and v along the last dim.
"""

import ast
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from torch import fx, nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.transformers.fusers.base import (
    RewriteFuser,
    fused_head_size,
    local_output_sizes,
)
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    is_fn,
    is_linear,
    is_method,
    recover_forward,
    replace_expr,
    returned_linear,
    single_self_call,
    upstream_linear,
)
from vllm.model_executor.models.transformers.utils import (
    log_replacement,
    replace_linear_class,
)
from vllm.model_executor.models.utils import maybe_prefix

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class PackedQKVFuser(RewriteFuser):
    """Fuser for attention with q, k and v packed into one projection."""

    qkv_name: str
    o_name: str | None
    q_size: int
    kv_size: int
    form: Literal["split", "slice"] = "split"
    """How the module separates q, k and v: `.split(...)` or three slices."""

    def info(self, name: str) -> str:
        return (
            f"Fused: {self.qkv_name} ({name}: {self.source_cls}) -> QKVParallelLinear"
        )

    @staticmethod
    def _split_sizes(node: fx.Node) -> tuple[int, int] | None:
        """`(q, kv)` from a `split((q, kv, kv), ...)` call, if it is one."""
        if not is_method(node, "split") or len(node.args) < 2:
            return None
        sizes = node.args[1]
        if not isinstance(sizes, (tuple, list)) or len(sizes) != 3:
            return None
        if not all(isinstance(size, int) for size in sizes):
            return None
        q_size, k_size, v_size = sizes
        if k_size != v_size or q_size < k_size:
            return None
        return q_size, k_size

    @staticmethod
    def _sliced_sizes(node: fx.Node) -> tuple[int, int] | None:
        """`(q, kv)` if the output of `node` is only sliced into q, k and v.

        The three users must be `x[..., :q]`, `x[..., q:q + kv]` and
        `x[..., q + kv:]` on `node` itself, so slices of anything derived from
        it (e.g. rotary) never match.
        """
        if len(node.users) != 3:
            return None
        bounds = []
        for user in node.users:
            if not is_fn(user, operator.getitem) or user.args[0] is not node:
                return None
            index = user.args[1]
            if not (
                isinstance(index, tuple)
                and len(index) == 2
                and index[0] is Ellipsis
                and isinstance(index[1], slice)
                and index[1].step is None
            ):
                return None
            start, stop = index[1].start, index[1].stop
            # Bounds derived from a traced tensor are fx nodes, not ints
            if not all(b is None or isinstance(b, int) for b in (start, stop)):
                return None
            bounds.append((start, stop))
        bounds.sort(key=lambda b: -1 if b[0] is None else b[0])
        (q_lo, q_hi), (k_lo, k_hi), (v_lo, v_hi) = bounds
        if q_lo is not None or v_hi is not None or None in (q_hi, k_lo, k_hi, v_lo):
            return None
        q_size, kv_size = q_hi, k_hi - q_hi
        if k_lo != q_size or v_lo != k_hi:
            return None
        if kv_size <= 0 or q_size < kv_size:
            return None
        return q_size, kv_size

    @classmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "PackedQKVFuser | None":
        for node in graph.nodes:
            form: Literal["split", "slice"]
            if (sizes := cls._split_sizes(node)) is not None:
                form, qkv_node = "split", upstream_linear(node.args[0], module)
            elif (sizes := cls._sliced_sizes(node)) is not None:
                form, qkv_node = "slice", node if is_linear(node, module) else None
            else:
                continue
            if qkv_node is None:
                continue
            q_size, kv_size = sizes
            qkv_name = str(qkv_node.target)
            # q, k and v must consume the whole projection.
            if module.get_submodule(qkv_name).out_features != q_size + 2 * kv_size:
                continue
            # o_proj produces the module's output and consumes the query width.
            o_name = returned_linear(graph, module)
            if o_name == qkv_name or (
                o_name is not None
                and module.get_submodule(o_name).in_features != q_size
            ):
                o_name = None
            return cls(
                source_cls=type(module).__name__,
                qkv_name=qkv_name,
                o_name=o_name,
                q_size=q_size,
                kv_size=kv_size,
                form=form,
            )
        return None

    def _split_call(self, funcdef: ast.FunctionDef) -> ast.Call:
        """The unique `self.<qkv_name>(...)....split((a, b, c), ...)` call."""
        calls = [
            node
            for node in ast.walk(funcdef)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "split"
            and node.args
            and isinstance(node.args[0], (ast.Tuple, ast.List))
            and len(node.args[0].elts) == 3
            and any(
                isinstance(inner, ast.Attribute) and inner.attr == self.qkv_name
                for inner in ast.walk(node.func.value)
            )
        ]
        if len(calls) != 1:
            raise ValueError(f"{self.qkv_name} has {len(calls)} three-way splits")
        return calls[0]

    def _slices_to_split(self, funcdef: ast.FunctionDef) -> None:
        """Rewrite the three `qkv[..., a:b]` slices as a `split((q, kv, kv))`."""
        call = single_self_call(funcdef, self.qkv_name)
        assigns = [
            node
            for node in ast.walk(funcdef)
            if isinstance(node, ast.Assign) and node.value is call
        ]
        if len(assigns) != 1 or not (
            len(assigns[0].targets) == 1 and isinstance(assigns[0].targets[0], ast.Name)
        ):
            raise ValueError(f"{self.qkv_name} output is not bound to a single name")
        name = assigns[0].targets[0].id
        refs = [
            node
            for node in ast.walk(funcdef)
            if isinstance(node, ast.Name) and node.id == name
        ]
        loads = [ref for ref in refs if isinstance(ref.ctx, ast.Load)]
        subscripts = [
            node
            for node in ast.walk(funcdef)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == name
        ]
        if len(refs) - len(loads) != 1 or len(loads) != 3 or len(subscripts) != 3:
            raise ValueError(f"{name} is rebound or not only sliced three ways")
        position: dict[int, ast.Subscript] = {}
        for subscript in subscripts:
            index = subscript.slice
            if not (
                isinstance(index, ast.Tuple)
                and len(index.elts) == 2
                and isinstance(index.elts[0], ast.Constant)
                and index.elts[0].value is Ellipsis
                and isinstance(index.elts[1], ast.Slice)
                and index.elts[1].step is None
            ):
                raise ValueError(f"{name} has a slice that is not [..., a:b]")
            part = index.elts[1]
            # q has no lower bound, k has both, v has no upper bound
            pos = {(False, True): 0, (True, True): 1, (True, False): 2}.get(
                (part.lower is not None, part.upper is not None)
            )
            if pos is None or pos in position:
                raise ValueError(f"{name} slices are not a q, k, v partition")
            position[pos] = subscript

        # name = self.qkv(x)  ->  name = self.qkv(x).split((q, kv, kv), dim=-1)
        # name[..., a:b]      ->  name[0], name[1], name[2]
        sizes = f"({self.q_size}, {self.kv_size}, {self.kv_size})"
        assigns[0].value = ast.Call(
            func=ast.Attribute(value=call, attr="split", ctx=ast.Load()),
            args=[ast.parse(sizes, mode="eval").body],
            keywords=[ast.keyword(arg="dim", value=ast.Constant(-1))],
        )
        for pos, subscript in position.items():
            replace_expr(
                funcdef, subscript, ast.parse(f"{name}[{pos}]", mode="eval").body
            )

    def update_forward(self, module: nn.Module) -> None:
        """Rewrite the split sizes to the sharded projection's per-rank widths."""
        funcdef, fn = recover_forward(type(module))
        if self.form == "slice":
            self._slices_to_split(funcdef)
        split = self._split_call(funcdef)
        # (q, kv, kv) -> [s // qkv.tp_size for s in qkv.output_sizes]
        sections = local_output_sizes(self.qkv_name)
        split.args[0] = ast.parse(sections, mode="eval").body
        self.fused_forward = compile_forward(funcdef, fn)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        """Shapes must be compatible with a head-sharded packed GEMM."""
        head_size = fused_head_size(module, vllm_config)
        qkv = module.get_submodule(self.qkv_name)
        compatible = (
            self.q_size % head_size == 0
            and self.kv_size % head_size == 0
            and qkv.out_features == self.q_size + 2 * self.kv_size
        )
        if not compatible:
            logger.debug("%s is not compatible with packed QKV fusion", type(module))
        return compatible

    def update_attrs(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> None:
        quant_config = vllm_config.quant_config
        head_size = fused_head_size(module, vllm_config)
        qkv_prefix = maybe_prefix(prefix, self.qkv_name)
        qkv = module.get_submodule(self.qkv_name)
        merged = QKVParallelLinear(
            hidden_size=qkv.in_features,
            head_size=head_size,
            total_num_heads=self.q_size // head_size,
            total_num_kv_heads=self.kv_size // head_size,
            bias=qkv.bias is not None,
            quant_config=quant_config,
            prefix=qkv_prefix,
            return_bias=False,
        )
        setattr(module, self.qkv_name, merged)
        log_replacement(qkv_prefix, qkv, merged)
        # If there is an output projection, we know it must be rowwise.
        if self.o_name is not None:
            o_prefix = maybe_prefix(prefix, self.o_name)
            o_proj = module.get_submodule(self.o_name)
            new_o = replace_linear_class(
                o_proj, "rowwise", quant_config, prefix=o_prefix
            )
            setattr(module, self.o_name, new_o)
            log_replacement(o_prefix, o_proj, new_o)
