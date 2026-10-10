# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence parallelism and async TP for the Transformers modeling backend.

The residual stream is sharded across TP ranks along the token dimension between
the first and last decoder layer of each pipeline stage. Each child of a decoder
layer that reduces across TP (attention, MLP, MoE, etc.) is a region that
all-gathers its input and reduce-scatters its output instead of all-reducing it.
Everything inside a region sees every token, and everything outside (norms,
residual adds) only sees this rank's tokens.

With async TP, a region whose input feeds unquantized column parallel linears, or
whose output is an unquantized row parallel linear, fuses the collective into
those GEMMs using symmetric memory.
"""

from collections import Counter
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import fx, nn

from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.models.transformers.fx_utils import (
    output_value,
    peel,
    trace,
)
from vllm.models.common.ops.sequence_parallel import sp_shard

logger = init_logger(__name__)


def pad_tokens(x: torch.Tensor, dim: int, multiple: int) -> torch.Tensor:
    """Pad `x` along `dim` so that its size is a multiple of `multiple`."""
    if (pad := -x.shape[dim] % multiple) == 0:
        return x
    dim = dim % x.ndim
    return F.pad(x, (0, 0) * (x.ndim - 1 - dim) + (0, pad))


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    # Transformers modules use [1, num_tokens, hidden_size]
    return x.reshape(-1, x.shape[-1])


def _unflatten_tokens(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return x.view(*like.shape[:-2], -1, x.shape[-1])


def shard(x: torch.Tensor) -> torch.Tensor:
    return _unflatten_tokens(sp_shard(_flatten_tokens(x)), x)


def all_gather(x: torch.Tensor) -> torch.Tensor:
    return _unflatten_tokens(tensor_model_parallel_all_gather(_flatten_tokens(x), 0), x)


def _reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    output = tensor_model_parallel_reduce_scatter(_flatten_tokens(x), 0)
    return _unflatten_tokens(output, x)


def _map_input(args: tuple, kwargs: dict, fn: Callable) -> tuple[tuple, dict]:
    """Apply `fn` to a module's first input, the hidden states."""
    if args:
        return (fn(args[0]), *args[1:]), kwargs
    name = next(iter(kwargs))
    return args, {**kwargs, name: fn(kwargs[name])}


def _map_output(output: Any, fn: Callable) -> Any:
    """Apply `fn` to a module's first output, the hidden states."""
    if isinstance(output, tuple):
        return (fn(output[0]), *output[1:])
    return fn(output)


class AsyncTPAllGatherMatmul(UnquantizedLinearMethod):
    """Returns the output its region computed in `fused_all_gather_matmul`."""

    pending: torch.Tensor | None = None
    """The output computed from the region's input, which is this linear's input."""

    def apply(self, layer, x, bias=None):
        output, self.pending = self.pending, None
        if output is None:
            return super().apply(layer, x, bias)
        output = _unflatten_tokens(output, x)
        return output if bias is None else output + bias


class AsyncTPMatmulReduceScatter(UnquantizedLinearMethod):
    """GEMM and reduce-scatter the tokens in one overlapped op."""

    def __init__(self, group_name: str):
        super().__init__()
        self.group_name = group_name

    def apply(self, layer, x, bias=None):
        output = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            _flatten_tokens(x),
            layer.weight.t(),
            "sum",
            scatter_dim=0,
            group_name=self.group_name,
        )
        # `RowParallelLinear` only passes the bias to rank 0, add it after scattering
        if layer.bias is not None and not layer.skip_bias_add:
            output = output + layer.bias
        return _unflatten_tokens(output, x)


def _is_tp(module: nn.Module) -> bool:
    return getattr(module, "tp_size", 1) > 1


def _can_fuse(linear: LinearBase) -> bool:
    return type(linear.quant_method) is UnquantizedLinearMethod


def _returned_module(graph: fx.Graph, module: nn.Module) -> nn.Module | None:
    """The submodule whose output `module` returns, if returned unmodified."""
    value = output_value(graph)
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    while isinstance(value := peel(value), fx.Node) and value.op == "call_module":
        submodule = module.get_submodule(str(value.target))
        if not isinstance(submodule, (nn.Dropout, nn.Identity)):
            return submodule
        value = value.args[0]
    return None


_PASSTHROUGH_OPS = frozenset(
    {
        "reshape",
        "view",
        "contiguous",
        "transpose",
        "permute",
        "flatten",
        "squeeze",
        "unsqueeze",
        "expand",
        "clone",
        "neg",
        "getitem",
        "split",
        "chunk",
        "dropout",
        "to",
        "float",
        "half",
        "bfloat16",
        "type_as",
    }
)
"""Ops whose output is a partial sum if their first input is."""
_ADDITIVE_OPS = frozenset(
    {"add", "add_", "iadd", "sub", "sub_", "isub", "cat", "stack"}
)
"""Ops whose output is a partial sum if all their tensor inputs are."""
_SCALING_OPS = frozenset({"mul", "mul_", "imul", "truediv", "div", "div_"})
"""Ops whose output is a partial sum if exactly one input is (the numerator)."""


class _Unverifiable(Exception):
    pass


def _nodes(arg: object) -> list[fx.Node]:
    found: list[fx.Node] = []
    fx.node.map_aggregate(
        arg, lambda a: found.append(a) if isinstance(a, fx.Node) else a
    )
    return found


def _op_name(node: fx.Node) -> str:
    target = node.target
    return target if isinstance(target, str) else getattr(target, "__name__", "")


def _check_partial_output(module: nn.Module, reducers: set[nn.Module]) -> None:
    """Check that `module` returns the sum of its `reducers`' partial outputs.

    Reducers skip their TP all-reduce under sequence parallelism, so only ops that
    commute with the sum across ranks may act on their outputs before the region's
    reduce-scatter.

    Raises:
        ValueError: If an op that does not commute with the sum is applied.
        _Unverifiable: If `module` could not be traced completely.

    """
    graph = trace(module)
    if graph is None or (value := output_value(graph)) is None:
        raise _Unverifiable
    partial: dict[fx.Node, bool] = {}
    for node in graph.nodes:
        if node.op == "output":
            break
        inputs = _nodes((node.args, node.kwargs))
        partial_inputs = [n for n in inputs if partial[n]]
        if node.op == "call_module":
            child = module.get_submodule(str(node.target))
            if any(m in reducers for m in child.modules()):
                if partial_inputs:
                    raise ValueError(f"`{node.target}` reduces a partial sum again")
                if child not in reducers and not isinstance(child, MoERunnerInterface):
                    _check_partial_output(child, reducers)
                partial[node] = True
                continue
            if partial_inputs and not isinstance(child, (nn.Dropout, nn.Identity)):
                raise ValueError(f"`{node.target}` is applied to a partial sum")
        elif partial_inputs:
            name = _op_name(node)
            args = node.args
            if name in _PASSTHROUGH_OPS:
                safe = bool(args) and args[0] in partial_inputs
            elif name in _ADDITIVE_OPS:
                constants = [a for a in args if isinstance(a, (int, float)) and a]
                safe = len(partial_inputs) == len(inputs) and not (
                    constants and name != "cat" and name != "stack"
                )
            elif name in _SCALING_OPS:
                safe = len(partial_inputs) == 1 and (
                    "div" not in name or args[0] in partial_inputs
                )
            else:
                safe = False
            if not safe:
                raise ValueError(f"`{name}` is applied to a partial sum")
        partial[node] = bool(partial_inputs)
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    if not (isinstance(value, fx.Node) and partial[value]):
        raise ValueError("its output is not a partial sum of its reductions")


class Region:
    """A child of a decoder layer that reduces across TP, so sees every token."""

    def __init__(self, module: nn.Module, async_tp: bool, group_name: str):
        self.group_name = group_name
        self.gather_linears: list[ColumnParallelLinear] = []
        """Column parallel linears fused with the input all-gather."""
        self.gather_methods: list[AsyncTPAllGatherMatmul] = []
        self.scatter_linear: RowParallelLinear | None = None
        """Row parallel linear fused with the output reduce-scatter."""

        moes = [m for m in module.modules() if isinstance(m, MoERunnerInterface)]
        rows = [
            m
            for m in module.modules()
            if isinstance(m, RowParallelLinear)
            and _is_tp(m)
            and m.input_is_parallel
            and m.reduce_results
        ]
        # Some MoE backends (e.g. all2all) always reduce their output, in which
        # case this region's output is replicated, not partial
        self.partial = all(m.moe_config.skip_final_all_reduce for m in moes)
        if self.partial:
            try:
                _check_partial_output(module, {*rows, *moes})
            except _Unverifiable:
                logger.warning_once(
                    "Could not verify that %s is compatible with sequence parallelism.",
                    type(module).__name__,
                )
            except ValueError as e:
                raise NotImplementedError(
                    f"Sequence parallelism does not support {type(module).__name__}: "
                    f"{e}."
                ) from None
            for row in rows:
                row.reduce_results = False
            if async_tp and not moes:
                self._fuse(module, rows)

        module.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        module.register_forward_hook(self._post_hook, with_kwargs=True)

    @staticmethod
    def contains_reduction(module: nn.Module) -> bool:
        return any(
            isinstance(m, MoERunnerInterface)
            or (
                isinstance(m, RowParallelLinear)
                and _is_tp(m)
                and m.input_is_parallel
                and m.reduce_results
            )
            for m in module.modules()
        )

    def _fuse(self, module: nn.Module, rows: list[RowParallelLinear]):
        # Only fuse if the whole forward traced, so no call is missed
        if (graph := trace(module)) is None or output_value(graph) is None:
            return
        placeholder = next((n for n in graph.nodes if n.op == "placeholder"), None)
        if placeholder is None:
            return
        calls = Counter(str(n.target) for n in graph.nodes if n.op == "call_module")
        for user in placeholder.users:
            if user.op != "call_module" or user.args[:1] != (placeholder,):
                continue
            linear = module.get_submodule(str(user.target))
            if (
                isinstance(linear, ColumnParallelLinear)
                and _is_tp(linear)
                and not linear.gather_output
                and _can_fuse(linear)
                # Called exactly once, on the region's input
                and calls[str(user.target)] == 1
            ):
                linear.quant_method = AsyncTPAllGatherMatmul()
                self.gather_linears.append(linear)
                self.gather_methods.append(linear.quant_method)
        if (
            len(rows) == 1
            and _can_fuse(rows[0])
            and _returned_module(graph, module) is rows[0]
        ):
            rows[0].quant_method = AsyncTPMatmulReduceScatter(self.group_name)
            self.scatter_linear = rows[0]

    def _all_gather_matmul(self, x: torch.Tensor) -> torch.Tensor:
        gathered, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            _flatten_tokens(x),
            [linear.weight.t() for linear in self.gather_linears],
            gather_dim=0,
            group_name=self.group_name,
        )
        for method, output in zip(self.gather_methods, outputs):
            method.pending = output
        return _unflatten_tokens(gathered, x)

    def _pre_hook(self, module: nn.Module, args: tuple, kwargs: dict):
        fn = self._all_gather_matmul if self.gather_linears else all_gather
        return _map_input(args, kwargs, fn)

    def _post_hook(self, module: nn.Module, args: tuple, kwargs: dict, output: Any):
        if self.scatter_linear is not None:
            return output
        return _map_output(output, _reduce_scatter if self.partial else shard)


def shard_input_hook(module: nn.Module, args: tuple, kwargs: dict):
    return _map_input(args, kwargs, shard)


def all_gather_output_hook(module: nn.Module, args: tuple, kwargs: dict, output):
    return _map_output(output, all_gather)


_TOKEN_MIXING_OPS = frozenset(
    {"cumsum", "cumprod", "roll", "flip", "sort", "argsort", "conv1d", "conv2d"}
)


def warn_if_mixing_tokens(layer: nn.Module, is_region: Callable[[nn.Module], bool]):
    """Warn if `layer` may mix tokens outside its regions, where they are sharded."""
    if (graph := trace(layer)) is None:
        return
    for node in graph.nodes:
        if node.op == "call_module":
            child = layer.get_submodule(str(node.target))
            token_local = (
                is_region(child)
                or isinstance(child, (nn.Dropout, nn.Identity, nn.Linear, LinearBase))
                or "norm" in type(child).__name__.lower()
                or next(child.parameters(), None) is None
            )
            suspect = None if token_local else f"`{node.target}`"
        else:
            name = _op_name(node)
            suspect = f"`{name}`" if name in _TOKEN_MIXING_OPS else None
        if suspect is not None:
            logger.warning_once(
                "%s in %s may mix tokens, which sequence parallelism shards outside "
                "of attention and MLP-like modules. Outputs may be incorrect.",
                suspect,
                type(layer).__name__,
            )
