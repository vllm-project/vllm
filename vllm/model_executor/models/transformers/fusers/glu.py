# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLU projection fuser: `act(gate(x)) * up(x)` -> a fused gate/up linear."""

import ast
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn
from transformers.activations import ACT2CLS

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_AND_MUL_REGISTRY
from vllm.model_executor.models.transformers.fusers.merged_column import (
    MergedColumnParallelFuser,
)
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    find_node,
    is_linear,
    peel,
    recover_forward,
    replace_expr,
    single_self_call,
)
from vllm.model_executor.models.transformers.layers import get_act_and_mul_fn
from vllm.model_executor.models.transformers.utils import (
    log_replacement,
    replace_linear_class,
)
from vllm.model_executor.models.utils import maybe_prefix

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


CLS2ACT: dict[type, list[str]] = {}
for _act_name, _act_cls in ACT2CLS.items():
    if isinstance(_act_cls, tuple):
        _act_cls = _act_cls[0]
    CLS2ACT.setdefault(_act_cls, []).append(_act_name)

ACT_AND_MUL_NAMES = frozenset(_ACTIVATION_AND_MUL_REGISTRY.keys())


@dataclass
class GLUFuser(MergedColumnParallelFuser):
    """Fuser for the GLU pattern `act(gate(x)) * up(x)`."""

    act_name: str
    down_name: str | None
    merged_name: ClassVar[str] = "gate_up_proj"

    @property
    def gate_name(self) -> str:
        return self.linear_names[0]

    @property
    def up_name(self) -> str:
        return self.linear_names[1]

    @classmethod
    def _is_act_of_gate(cls, node: fx.Node, module: nn.Module) -> bool:
        """Is node `act(gate(x))` where `gate` is linear and `act` is not linear."""
        return (
            node.op == "call_module"
            and not is_linear(node, module)
            and len(node.args) == 1
            and isinstance(node.args[0], fx.Node)
            and is_linear(node.args[0], module)
        )

    @classmethod
    def _get_glu_nodes(
        cls, graph: fx.Graph, module: nn.Module
    ) -> tuple[fx.Node, fx.Node, fx.Node, fx.Node] | None:
        """Search graph for the GLU pattern `act(gate(x)) * up(x)`."""
        for mul in graph.nodes:
            if (
                mul.op == "call_function"
                and mul.target == operator.mul
                and len(mul.args) == 2
                and all(isinstance(arg, fx.Node) for arg in mul.args)
            ):
                a, b = mul.args
                if cls._is_act_of_gate(a, module) and is_linear(b, module):
                    act, gate, up = a, a.args[0], b
                elif cls._is_act_of_gate(b, module) and is_linear(a, module):
                    act, gate, up = b, b.args[0], a
                else:
                    continue
                if (
                    all(len(args) == 1 for args in (gate.args, up.args))
                    and isinstance(x := gate.args[0], fx.Node)
                    and x is up.args[0]
                ):
                    return act, gate, up, mul
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
        act_node, gate_node, up_node, mul_node = glu_nodes

        predicate = lambda n: is_linear(n, module) and peel(n.args[0]) is mul_node
        down_node = find_node(graph, predicate)
        return cls(
            source_cls=type(module).__name__,
            act_name=act_node.target,
            linear_names=(gate_node.target, up_node.target),
            down_name=down_node.target if down_node is not None else None,
        )

    def update_forward(self, module: nn.Module) -> None:
        """Replace `act(gate(x)) * up(x)` with `act(gate_up(x))` in source."""
        funcdef, fn = recover_forward(type(module))
        act_call = single_self_call(funcdef, self.act_name)
        gate_call = single_self_call(funcdef, self.gate_name)
        up_call = single_self_call(funcdef, self.up_name)
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
        replace_expr(funcdef, muls[0], act_call)
        self.fused_forward = compile_forward(funcdef, fn)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        if not super().validate(module, vllm_config):
            return False
        # An AndMul kernel splits its input in half, so gate and up must match.
        gate = module.get_submodule(self.gate_name)
        up = module.get_submodule(self.up_name)
        if gate.out_features != up.out_features:
            logger.debug("%s and %s differ in size; skipping fusion", gate, up)
            return False
        act = module.get_submodule(self.act_name)
        if self._get_act_and_mul_name(act) is None:
            logger.debug("%s has no AndMul equivalent; skipping fusion", act)
            return False
        return True

    def update_attrs(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> None:
        super().update_attrs(module, prefix, vllm_config)
        act_fn = self._get_act_and_mul(module.get_submodule(self.act_name))
        setattr(module, self.act_name, act_fn)
        # If there is a down projection, we know it must be rowwise.
        if self.down_name is not None:
            down_prefix = maybe_prefix(prefix, self.down_name)
            down = module.get_submodule(self.down_name)
            new_down = replace_linear_class(
                down, "rowwise", vllm_config.quant_config, prefix=down_prefix
            )
            setattr(module, self.down_name, new_down)
            log_replacement(down_prefix, down, new_down)
