# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fusion patterns for the Transformers modeling backend.

Each module class is traced once with `torch.fx` to detect a pattern from its
dataflow, then its forward *source* is rewritten (AST) to replace only the
matched calls; the rest stays live Python. The rewritten forward is compiled
once per class and bound to each fused instance in place.
"""

import ast
import inspect
import operator
import textwrap
import types
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from cachetools import cached
from torch import fx, nn
from transformers.activations import ACT2CLS

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import (
    _ACTIVATION_AND_MUL_REGISTRY,
    get_act_and_mul_fn,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.models.utils import ShardId, maybe_prefix

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)

CLS2ACT: dict[type, list[str]] = {}
for _act_name, _act_cls in ACT2CLS.items():
    if isinstance(_act_cls, tuple):
        _act_cls = _act_cls[0]
    CLS2ACT.setdefault(_act_cls, []).append(_act_name)

ACT_AND_MUL_NAMES = frozenset(_ACTIVATION_AND_MUL_REGISTRY.keys())


def _infer_len(node: fx.Node) -> int | None:
    """Concrete length of a proxy's value, inferred from its node chain.

    Lets tracing pass through the shape unpacks and `*`-splats (e.g.
    `(*input_shape, -1, head_dim)`) that precede the patterns in HF attention.
    """
    # `x.shape` has the rank of `x`, when known
    if (
        node.op == "call_function"
        and node.target is getattr
        and node.args[1] == "shape"
        and (rank := _rank(node.args[0])) is not None
    ):
        return rank
    # Slices of known-length values
    if node.op == "call_function" and node.target is operator.getitem:
        src_len = _infer_len(node.args[0])
        index = node.args[1]
        if src_len is not None and isinstance(index, slice):
            return len(range(*index.indices(src_len)))
    return None


def _rank(node: fx.Node) -> int | None:
    """The tensor rank of `node`'s value, if known."""
    # vLLM always feeds the model [1, seq_len, hidden_size] hidden states
    if node.op == "placeholder" and node.target == "hidden_states":
        return 3
    return None


class _SizedProxy(fx.Proxy):
    """Proxy whose `len` is inferred from the graph (see `_infer_len`)."""

    def __len__(self) -> int:
        length = _infer_len(self.node)
        if length is None:
            return super().__len__()
        return length


class _AllLeafTracer(fx.Tracer):
    """Tracer that treats every submodule as a leaf.

    Each child stays one `call_module` node, so matching sees the module's own
    forward structure (activations aren't decomposed into e.g. `sigmoid * x`).
    `iter` traces through the leading shape unpacks (see `_infer_len`); anything
    else untraceable ends the trace early and the partial graph is matched.
    """

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        return True

    def proxy(self, node: fx.Node) -> fx.Proxy:
        return _SizedProxy(node, self)

    def iter(self, obj: fx.Proxy):
        length = _infer_len(obj.node)
        if length is None:
            return super().iter(obj)
        return iter([obj[i] for i in range(length)])


def _recover_forward(cls: type[nn.Module]) -> tuple[ast.FunctionDef, Callable]:
    """Parse the source of `cls.forward`, ready for rewriting."""
    fn = inspect.unwrap(cls.forward)
    if fn.__code__.co_freevars:
        raise ValueError("forward is a closure")
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    funcdef = tree.body[0]
    if not isinstance(funcdef, ast.FunctionDef):
        raise ValueError("source is not a plain function definition")
    # `fn` is already unwrapped; don't re-apply its decorators
    funcdef.decorator_list.clear()
    # Annotations may not evaluate outside the defining module (e.g. with
    # postponed evaluation); they're not needed at runtime
    funcdef.returns = None
    args = funcdef.args
    for arg in (
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        *filter(None, (args.vararg, args.kwarg)),
    ):
        arg.annotation = None
    # Recompiling outside the class body would break name mangling
    for node in ast.walk(funcdef):
        name = getattr(node, "attr", None) or getattr(node, "id", None)
        if name and name.startswith("__") and not name.endswith("__"):
            raise ValueError(f"{name} would be name mangled")
    return funcdef, fn


def _compile_forward(funcdef: ast.FunctionDef, fn: Callable) -> Callable:
    """Compile `funcdef` in `fn`'s module so tracebacks point at the source."""
    module = ast.Module(body=[funcdef], type_ignores=[])
    ast.fix_missing_locations(module)
    ast.increment_lineno(module, fn.__code__.co_firstlineno - 1)
    code = compile(module, fn.__code__.co_filename, "exec")
    namespace: dict = {}
    exec(code, fn.__globals__, namespace)
    return namespace[funcdef.name]


def _single_self_call(funcdef: ast.FunctionDef, name: str) -> ast.Call:
    """The unique `self.<name>(arg)` call in `funcdef`.

    Raises unless `name` appears exactly once, as such a call, so the source
    rewrite agrees with the fx match.
    """
    uses = [
        node
        for node in ast.walk(funcdef)
        if isinstance(node, ast.Attribute) and node.attr == name
    ]
    if len(uses) != 1:
        raise ValueError(f"{name} is referenced {len(uses)} times")
    calls = [
        node
        for node in ast.walk(funcdef)
        if isinstance(node, ast.Call)
        and node.func is uses[0]
        and len(node.args) == 1
        and not isinstance(node.args[0], ast.Starred)
        and not node.keywords
    ]
    if (
        len(calls) != 1
        or not isinstance(uses[0].value, ast.Name)
        or uses[0].value.id != "self"
    ):
        raise ValueError(f"{name} is not a single-argument call on self")
    return calls[0]


def _innermost_block(
    block: list[ast.stmt], node: ast.AST
) -> tuple[list[ast.stmt], int] | None:
    """The innermost statement list containing `node`, and the index within."""
    for index, stmt in enumerate(block):
        if not any(child is node for child in ast.walk(stmt)):
            continue
        child_blocks = [
            getattr(stmt, fld, None) for fld in ("body", "orelse", "finalbody")
        ]
        child_blocks += [h.body for h in getattr(stmt, "handlers", [])]
        child_blocks += [c.body for c in getattr(stmt, "cases", [])]
        for child_block in child_blocks:
            if (
                isinstance(child_block, list)
                and child_block
                and (found := _innermost_block(child_block, node)) is not None
            ):
                return found
        return block, index
    return None


def _replace_expr(module: ast.AST, old: ast.expr, new: ast.expr) -> None:
    """Replace the expression `old` (by identity) with `new` within `module`."""

    class _Replacer(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            if node is old:
                return new
            return super().generic_visit(node)

    _Replacer().visit(module)


def _is_linear(node: fx.Node, module: nn.Module) -> bool:
    """Is node `nn.Linear.__call__()`."""
    return node.op == "call_module" and isinstance(
        module.get_submodule(node.target), nn.Linear
    )


@dataclass
class BaseFuser(ABC):
    """A per-class fusion plan and the steps to apply it.

    `match` and `update_forward` analyse the module *class* once (cached, see
    `get_fuser`); `fuse` then builds the merged submodule and binds the compiled
    forward on an instance in place, so it keeps its class and any attribute the
    fusion does not consume.
    """

    merged_name: ClassVar[str]
    """Attribute name of the merged module created by `update_attrs`."""

    fused_forward: Callable = field(init=False, repr=False)
    """The compiled rewritten forward, set by `update_forward`."""

    @property
    @abstractmethod
    def shards(self) -> list[tuple[str, ShardId]]:
        """Each projection's original name and its shard id in the merged module.

        Source for both `orig_to_new_stacked` and `packed_modules_mapping`."""

    def orig_to_new_stacked(self, prefix: str) -> dict[str, tuple[str, ShardId]]:
        """`WeightsMapper.orig_to_new_stacked` entries for one fused instance.

        Maps each checkpoint name to `(merged_name, shard_id)`, keyed by qualname
        so only this exact layer is remapped, never a same-named projection
        elsewhere (e.g. an unfused MoE expert's `gate_proj`)."""
        merged = maybe_prefix(prefix, self.merged_name)
        return {
            maybe_prefix(prefix, name): (merged, shard) for name, shard in self.shards
        }

    @property
    def packed_modules_mapping(self) -> dict[str, list[str]]:
        """`{merged_name: [projection names]}` so quantization can unpack the
        fused layer into its per-shard configs."""
        return {self.merged_name: [name for name, _ in self.shards]}

    @classmethod
    @abstractmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "BaseFuser | None":
        """Match the pattern in `graph`, returning a fuser if found."""

    @abstractmethod
    def update_forward(self, module: nn.Module) -> None:
        """Rewrite and compile `type(module)`'s forward source.

        Raises if the source does not admit the rewrite (fusion is then skipped).
        """

    @abstractmethod
    def validate(self, module: nn.Module, model_config: "ModelConfig") -> bool:
        """Whether this fuser can be applied to this `module` instance."""

    @abstractmethod
    def update_attrs(
        self,
        module: nn.Module,
        prefix: str,
        model_config: "ModelConfig",
        quant_config: "QuantizationConfig",
    ) -> None:
        """Replace `module`'s submodules with the merged module."""

    def fuse(
        self,
        module: nn.Module,
        prefix: str,
        model_config: "ModelConfig",
        quant_config: "QuantizationConfig",
    ) -> nn.Module:
        """Fuse an already-validated `module` in place (see `Fusers.__getitem__`).

        Builds the merged submodule and binds the compiled forward."""
        self.update_attrs(module, prefix, model_config, quant_config)
        module.forward = types.MethodType(self.fused_forward, module)
        return module


@dataclass
class GLUFuser(BaseFuser):
    """Fuser for the GLU pattern `act(gate(x)) * up(x)`."""

    act_name: str
    gate_name: str
    up_name: str
    merged_name: ClassVar[str] = "gate_up_proj"

    @property
    def shards(self) -> list[tuple[str, ShardId]]:
        return [(self.gate_name, 0), (self.up_name, 1)]

    @classmethod
    def _is_act_of_gate(cls, node: fx.Node, module: nn.Module) -> bool:
        """Is node `act(gate(x))` where `gate` is linear and `act` is not linear."""
        return (
            node.op == "call_module"
            and not _is_linear(node, module)
            and len(node.args) == 1
            and isinstance(node.args[0], fx.Node)
            and _is_linear(node.args[0], module)
        )

    @classmethod
    def _get_glu_nodes(
        cls, graph: fx.Graph, module: nn.Module
    ) -> tuple[fx.Node, fx.Node, fx.Node] | None:
        """Search graph for the GLU pattern `act(gate(x)) * up(x)`."""
        for node in graph.nodes:
            if (
                node.op == "call_function"
                and node.target == operator.mul
                and len(node.args) == 2
                and all(isinstance(arg, fx.Node) for arg in node.args)
            ):
                a, b = node.args
                if cls._is_act_of_gate(a, module) and _is_linear(b, module):
                    act, gate, up = a, a.args[0], b
                elif cls._is_act_of_gate(b, module) and _is_linear(a, module):
                    act, gate, up = b, b.args[0], a
                else:
                    continue
                if (
                    all(len(args) == 1 for args in (gate.args, up.args))
                    and isinstance(x := gate.args[0], fx.Node)
                    and x is up.args[0]
                ):
                    return act, gate, up
        return None

    @staticmethod
    def _get_act_and_mul_name(act: nn.Module) -> str | None:
        """Get the name of `act` if it has an `...AndMul` equivalent."""
        for name in CLS2ACT.get(type(act), []):
            if name in ACT_AND_MUL_NAMES:
                return name
        # nn.GELU is not in ACT2CLS, but could be in model code
        if type(act) is nn.GELU:
            return "gelu_pytorch_tanh" if act.approximate == "tanh" else "gelu"
        return None

    @classmethod
    def _get_act_and_mul(cls, act: nn.Module) -> nn.Module:
        """Get the `...AndMul` equivalent of a Transformers activation module."""
        if name := cls._get_act_and_mul_name(act):
            return get_act_and_mul_fn(name)
        raise ValueError(f"No AndMul equivalent for {type(act)}")

    @classmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "GLUFuser | None":
        if (glu_nodes := cls._get_glu_nodes(graph, module)) is None:
            return None
        act_node, gate_node, up_node = glu_nodes

        gate = module.get_submodule(gate_node.target)
        up = module.get_submodule(up_node.target)
        # Shapes must be compatible for a single merged GEMM.
        if gate.in_features == up.in_features and (gate.bias is None) == (
            up.bias is None
        ):
            return cls(act_node.target, gate_node.target, up_node.target)
        return None

    def update_forward(self, module: nn.Module) -> None:
        """Replace `act(gate(x)) * up(x)` with `act(gate_up(x))` in source."""
        funcdef, fn = _recover_forward(type(module))
        act_call = _single_self_call(funcdef, self.act_name)
        gate_call = _single_self_call(funcdef, self.gate_name)
        up_call = _single_self_call(funcdef, self.up_name)
        if act_call.args[0] is not gate_call:
            raise ValueError("activation does not directly wrap the gate")
        if ast.dump(gate_call.args[0]) != ast.dump(up_call.args[0]):
            raise ValueError("gate and up inputs are written differently")
        muls = [
            node
            for node in ast.walk(funcdef)
            if isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and {id(node.left), id(node.right)} == {id(act_call), id(up_call)}
        ]
        if len(muls) != 1:
            raise ValueError("no multiply of the activation and up projection")

        # act(gate(x)) * up(x) -> act(gate_up(x))
        assert isinstance(gate_call.func, ast.Attribute)
        gate_call.func.attr = self.merged_name
        _replace_expr(funcdef, muls[0], act_call)
        self.fused_forward = _compile_forward(funcdef, fn)

    def validate(self, module: nn.Module, model_config: "ModelConfig") -> bool:
        act = module.get_submodule(self.act_name)
        if self._get_act_and_mul_name(act) is None:
            logger.debug("No AndMul equivalent for %s; skipping fusion", type(act))
            return False
        return True

    def update_attrs(
        self,
        module: nn.Module,
        prefix: str,
        model_config: "ModelConfig",
        quant_config: "QuantizationConfig",
    ) -> None:
        act_fn = self._get_act_and_mul(module.get_submodule(self.act_name))
        gate = module.get_submodule(self.gate_name)
        up = module.get_submodule(self.up_name)
        merged = MergedColumnParallelLinear(
            input_size=gate.in_features,
            output_sizes=[gate.out_features, up.out_features],
            bias=gate.bias is not None,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, self.merged_name),
            return_bias=False,
        )
        logger.debug(
            "%s: %s, %s: %s -> %s: %s",
            self.gate_name,
            gate,
            self.up_name,
            up,
            self.merged_name,
            merged,
        )
        setattr(module, self.merged_name, merged)
        setattr(module, self.act_name, act_fn)
        # Drop the consumed submodules so their (meta) params are not expected.
        delattr(module, self.gate_name)
        delattr(module, self.up_name)


@dataclass
class QKVFuser(BaseFuser):
    """Fuser for the attention QKV pattern `q(x), k(x), v(x)`."""

    q_name: str
    k_name: str
    v_name: str
    merged_name: ClassVar[str] = "qkv_proj"

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
                _is_linear(node, module)
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
        return cls(q_name=q.target, k_name=k.target, v_name=v.target)

    def update_forward(self, module: nn.Module) -> None:
        """Replace `q(x), k(x), v(x)` with `qkv(x).split(sizes, -1)` in source."""
        funcdef, fn = _recover_forward(type(module))
        calls = [
            _single_self_call(funcdef, name)
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
        blocks = [_innermost_block(funcdef.body, call) for call in calls]
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
        _replace_expr(assign, arg, calls[0].args[0])
        block, index = blocks[0]
        ast.copy_location(assign, block[index])
        block.insert(min(index for _, index in blocks), assign)
        for call, temp in zip(calls, temps):
            _replace_expr(funcdef, call, ast.Name(id=temp, ctx=ast.Load()))
        self.fused_forward = _compile_forward(funcdef, fn)

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


@cached(cache={}, key=type)
def get_fuser(module: nn.Module) -> BaseFuser | None:
    """The fuser for `type(module)` (cached per class), or `None` if no match."""
    # Currently, all patterns require at least 2 linears, this is a cheap early skip
    if sum(isinstance(c, nn.Linear) for c in module.children()) < 2:
        return None
    tracer = _AllLeafTracer()
    try:
        graph = tracer.trace(module)
    except Exception as exc:
        # The graph is only evidence for matching, and the patterns appear at
        # the top of their forwards, so the partial graph traced before the
        # failure can still be matched.
        logger.debug("Could not fully trace %s for fusion: %s", type(module), exc)
        if (graph := getattr(tracer, "graph", None)) is None:
            return None
    for fuser_cls in (GLUFuser, QKVFuser):
        if (fuser := fuser_cls.match(graph, module)) is not None:
            try:
                fuser.update_forward(module)
            except Exception as exc:
                # An unrecognised source just means we cannot fuse here.
                logger.debug("Could not rewrite %s for fusion: %s", type(module), exc)
                return None
            return fuser
    return None


class Fusers(UserDict):
    """Mapping from module class to fuser, for all fusable classes in a model."""

    def __init__(self, model: nn.Module, model_config: "ModelConfig"):
        self.model_config = model_config
        super().__init__({type(m): get_fuser(m) for m in model.modules()})

    def __getitem__(self, m: nn.Module) -> BaseFuser | None:
        fuser = self.data.get(type(m))
        if fuser is not None and fuser.validate(m, self.model_config):
            return fuser
        return None
