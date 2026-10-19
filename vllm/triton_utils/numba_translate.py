# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run Triton kernels on CPU by translating their source to numba.

The kernel's AST is rewritten and compiled with numba: a launch runs the body
once per program id, blocks are numpy arrays, and pointers are `Ptr(mem, off)`,
a flat view of a tensor plus element offsets, so the kernel's pointer
arithmetic is kept (`+`/`-` go through `tl_add`). Constexpr and `None`
arguments are specialized and folded, as Triton does.

Unsupported `tl` ops raise `UnsupportedKernel`. Integers load as int64, so
kernels relying on int32 wraparound differ. Block ops allocate temporaries,
so vocab-sized kernels are slow.
"""

import ast
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import os
import pathlib
import sys
import textwrap
from collections import namedtuple
from collections.abc import Callable
from typing import Any

import numba
import numpy as np
import torch
from numba import types
from numba.extending import intrinsic, overload

Ptr = namedtuple("Ptr", ["mem", "off"])


class UnsupportedKernel(Exception):
    pass


# --- numba runtime -------------------------------------------------------------


def _jit_op(typer: Callable[..., Any]) -> Callable[..., Any]:
    """A jitted op whose implementation `typer` picks from the argument types."""

    def op(*args):
        raise NotImplementedError("only callable from numba")

    op.__name__ = typer.__name__
    # Not inline="always": numba then copies the arrays in returned Ptrs, and
    # stores through them are lost. LLVM inlines these anyway.
    overload(op)(typer)
    return op


def _is_ptr(ty: Any) -> bool:
    return isinstance(ty, types.BaseNamedTuple) and ty.instance_class is Ptr


def _is_block(ty: Any) -> bool:
    return isinstance(ty, types.Array)


@_jit_op
def tl_add(a, b):
    if _is_ptr(a):
        return lambda a, b: Ptr(a.mem, a.off + b)
    if _is_ptr(b):
        return lambda a, b: Ptr(b.mem, a + b.off)
    return lambda a, b: a + b


@_jit_op
def tl_load(ptr, mask, other):
    # Integers widen to int64: numba won't unify the int32/int64 values that
    # Triton promotes across branches. Stores narrow them back.
    elem = ptr.types[0].dtype
    wide = isinstance(elem, types.Integer) and elem.bitwidth < 64
    dt = np.int64 if wide else numba.np.numpy_support.as_dtype(elem).type
    if not _is_block(ptr.types[1]):
        return lambda ptr, mask, other: dt(ptr.mem[ptr.off]) if mask else dt(other)

    def load(ptr, mask, other):
        off = ptr.off.ravel()
        keep = (np.zeros(ptr.off.shape, np.bool_) | mask).ravel()
        out = np.empty(off.size, dt)
        for i in range(off.size):
            out[i] = ptr.mem[off[i]] if keep[i] else other
        return out.reshape(ptr.off.shape)

    return load


@_jit_op
def tl_store(ptr, value, mask):
    if not _is_block(ptr.types[1]):

        def store_scalar(ptr, value, mask):
            if mask:
                ptr.mem[ptr.off] = value

        return store_scalar

    def store(ptr, value, mask):
        off = ptr.off.ravel()
        keep = (np.zeros(ptr.off.shape, np.bool_) | mask).ravel()
        values = (np.zeros(ptr.off.shape, ptr.mem.dtype) + value).ravel()
        for i in range(off.size):
            if keep[i]:
                ptr.mem[off[i]] = values[i]

    return store


@numba.njit
def tl_atomic_add(ptr, value, mask):
    # Programs run one at a time, so this is atomic.
    old = tl_load(ptr, mask, 0)
    tl_store(ptr, old + value, mask)
    return old


@_jit_op
def tl_cast(x, dtype):
    if _is_block(x):
        return lambda x, dtype: x.astype(dtype)
    return lambda x, dtype: dtype(x)


@_jit_op
def tl_where(cond, x, y):
    if _is_block(cond) or _is_block(x) or _is_block(y):
        return lambda cond, x, y: np.where(cond, x, y)
    return lambda cond, x, y: x if cond else y


def _reduction(name: str, reduce: Callable[..., Any]) -> Callable[..., Any]:
    # numba's np.max/np.min take no axis: support 1D blocks only.
    def typer(x, axis=0):
        if _is_block(x) and x.ndim > 1:
            raise UnsupportedKernel(f"tl.{name} over a {x.ndim}D block")
        return lambda x, axis=0: reduce(x)

    typer.__name__ = f"tl_{name}"
    return _jit_op(typer)


tl_max = _reduction("max", np.max)
tl_min = _reduction("min", np.min)
tl_cdiv = numba.njit(lambda a, b: (a + b - 1) // b)


@intrinsic
def _inttoptr(typingctx, addr, dtype):
    sig = types.CPointer(dtype.dtype)(addr, dtype)

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], context.get_value_type(sig.return_type))

    return sig, codegen


@numba.njit
def tl_as_ptr(addr, dtype):
    """A pointer read from a pointer array: memory from `addr` on."""
    return Ptr(numba.carray(_inttoptr(np.int64(addr), dtype), (1 << 40,)), 0)


# --- translation ---------------------------------------------------------------

_DTYPES = {
    "int1": "np.bool_",
    **{f"{k}{b}": f"np.{k}{b}" for k in ("int", "uint") for b in (8, 16, 32, 64)},
    **{f"float{b}": f"np.float{b}" for b in (16, 32, 64)},
}
# tl.<op> -> replacement.
_OPS = {
    **{op: f"tl_{op}" for op in ("load", "store", "atomic_add", "where", "cast")},
    **{op: f"tl_{op}" for op in ("as_ptr", "max", "min", "cdiv")},
    **{op: f"np.{op}" for op in ("sum", "argmax", "argmin", "arange", "full")},
    **{op: f"np.{op}" for op in ("zeros", "maximum", "minimum", "cumsum", "abs")},
    **{op: f"np.{op}" for op in ("exp", "log", "sqrt")},
    "range": "range",
    "static_range": "range",
}
_HINTS = ("multiple_of", "max_contiguous")  # Replaced by their first argument.
_DROPPED = ("debug_barrier", "static_assert")
# Arguments made positional, with their defaults.
_SIGNATURES = {
    "load": {"pointer": "", "mask": "True", "other": "0"},
    "store": {"pointer": "", "value": "", "mask": "True"},
    "atomic_add": {"pointer": "", "val": "", "mask": "True"},
}


def _expr(src: str) -> ast.expr:
    return ast.parse(src, mode="eval").body


def _is_tl(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in ("tl", "triton")


class _Rewriter(ast.NodeTransformer):
    def __init__(self, consts: dict[str, Any]):
        self.consts = consts

    def _fold(self, test: ast.expr) -> bool | None:
        """The value of `test` if it only uses constants."""
        names = {n.id for n in ast.walk(test) if isinstance(n, ast.Name)}
        if names <= self.consts.keys():
            with contextlib.suppress(Exception):
                return bool(eval(ast.unparse(test), dict(self.consts)))
        return None

    def visit_If(self, node: ast.If) -> Any:
        test = self._fold(node.test)
        if test is not None:
            return [self.visit(s) for s in (node.body if test else node.orelse)]
        return self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        test = self._fold(node.test)
        if test is not None:
            return self.visit(node.body if test else node.orelse)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if not _is_tl(node.value):
            return self.generic_visit(node)
        if node.attr not in _DTYPES:
            raise UnsupportedKernel(f"tl.{node.attr}")
        return _expr(_DTYPES[node.attr])

    def visit_List(self, node: ast.List) -> Any:  # shapes, as in tl.full([N], ...)
        return self.visit(ast.Tuple(node.elts, ast.Load()))

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        self.generic_visit(node)
        if isinstance(node.op, ast.Sub):
            node.right = ast.UnaryOp(ast.USub(), node.right)
        elif not isinstance(node.op, ast.Add):
            return node
        return ast.Call(ast.Name("tl_add"), [node.left, node.right], [])

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        if not isinstance(node.op, (ast.Add, ast.Sub)):
            return self.generic_visit(node)
        current = _expr(ast.unparse(node.target))
        return ast.Assign(
            [node.target], self.visit(ast.BinOp(current, node.op, node.value))
        )

    def visit_Expr(self, node: ast.Expr) -> Any:
        value = self.visit(node.value)
        return None if value is None else ast.Expr(value)

    def visit_Call(self, node: ast.Call) -> Any:
        func: Any = node.func
        args: list[Any] = node.args
        if isinstance(func, ast.Attribute) and func.attr == "to":
            func, args = _expr("tl.cast"), [func.value, *args]
        if not (isinstance(func, ast.Attribute) and _is_tl(func.value)):
            return self.generic_visit(node)
        op = func.attr
        if op == "cast" and "pointer_type" in ast.unparse(args[1]):
            op, args = "as_ptr", [args[0], args[1].args[0]]
        if op in ("program_id", "num_programs"):
            axis = ast.literal_eval(args[0] if args else node.keywords[0].value)
            return ast.Name(f"__{'pid' if op == 'program_id' else 'grid'}{axis}")
        if op in _HINTS:
            return self.visit(args[0])
        if op in _DROPPED:
            return None
        if op not in _OPS:
            raise UnsupportedKernel(f"tl.{op}")
        keywords = [k for k in node.keywords if k.arg in ("axis", "dtype")]
        if op in _SIGNATURES:
            given = {k.arg: k.value for k in node.keywords}
            missing = list(_SIGNATURES[op].items())[len(args) :]
            args = args + [given.get(name) or _expr(dflt) for name, dflt in missing]
            keywords = []
        return self.generic_visit(ast.Call(_expr(_OPS[op]), args, keywords))


def _source(fn: Callable[..., Any]) -> ast.FunctionDef:
    fdef = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    assert isinstance(fdef, ast.FunctionDef)
    return fdef


def _is_triton_jit(obj: Any) -> bool:
    with contextlib.suppress(OSError, TypeError):
        return (
            inspect.isfunction(obj)
            and "triton.jit" in (inspect.getsource(obj).split("def ", 1)[0])
        )
    return False


def _repr(value: Any) -> str:
    if isinstance(value, type) and issubclass(value, np.generic):
        return f"np.{value.__name__}"
    return repr(value)


def _translate(fn: Callable[..., Any], consts: dict[str, Any], kernel: bool):
    """Returns (numba source for `fn`, its runtime params, jit fns it calls)."""
    fdef = _source(fn)
    used = {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name)}
    globals_ = {n: fn.__globals__[n] for n in used if n in fn.__globals__}
    module_consts = {
        n: v for n, v in globals_.items() if isinstance(v, (int, float, bool))
    }
    callees = {n: v for n, v in globals_.items() if _is_triton_jit(v)}
    for arg in fdef.args.args:
        arg.annotation = None
    fdef.decorator_list, fdef.returns = [], None
    fdef = _Rewriter({**module_consts, **consts}).visit(fdef)
    params = [a.arg for a in fdef.args.args if a.arg not in consts]
    pids = [f"__{p}{i}" for p in ("pid", "grid") for i in range(3)] if kernel else []
    fdef.args.args, fdef.args.defaults = [ast.arg(p) for p in pids + params], []
    src = "\n".join(
        [
            *(f"{n} = {v!r}" for n, v in module_consts.items()),
            "@njit(cache=True, nogil=True)",
            ast.unparse(ast.fix_missing_locations(fdef)),
        ]
    )
    return src, params, callees


def _generate(fn: Callable[..., Any], consts: dict[str, Any]) -> tuple[str, list[str]]:
    """Numba source for `fn` specialized on `consts`, and its launch params."""
    body, params, pending = _translate(fn, consts, kernel=True)
    helpers: dict[str, str] = {}
    while pending:
        name, helper = pending.popitem()
        if name not in helpers:
            helpers[name], _, callees = _translate(helper, {}, kernel=False)
            pending.update(callees)
    args = ", ".join(params)
    grid = "__grid0, __grid1, __grid2"
    launch = textwrap.dedent(f"""
        @njit(cache=True, nogil=True)
        def __launch({grid}, {args}):
            for __pid0 in range(__grid0):
                for __pid1 in range(__grid1):
                    for __pid2 in range(__grid2):
                        {fn.__name__}(__pid0, __pid1, __pid2, {grid}, {args})
    """)
    header = [
        "import numpy as np",
        "from numba import njit",
        "from vllm.triton_utils.numba_translate import *  # noqa",
        *(f"{k} = {_repr(v)}" for k, v in consts.items()),
    ]
    return "\n".join([*header, *reversed(helpers.values()), body, launch]), params


_RUNTIME_HASH = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
_CACHE_DIR = pathlib.Path(os.getenv("VLLM_CACHE_ROOT", "~/.cache/vllm"))


def _compile(src: str) -> Callable[..., None]:
    # numba's disk cache needs a source file and doesn't see changes to the
    # runtime above, so key generated modules on both.
    digest = hashlib.sha256((src + _RUNTIME_HASH).encode()).hexdigest()[:16]
    path = _CACHE_DIR.expanduser() / "triton_numba" / f"k_{digest}.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}")
        tmp.write_text(src)
        tmp.replace(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__launch


def _to_ptr(t: torch.Tensor) -> Ptr:
    """Every element `t` can reach through its strides, as a flat array."""
    extent = t.numel() and 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
    return Ptr(t.as_strided((extent,), (1,)).numpy(), 0)


class NumbaKernel:
    """A Triton kernel translated to numba, launched like the original."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.__name__ = fn.__name__
        self.signature = inspect.signature(fn)
        self.constexprs = {
            a.arg
            for a in _source(fn).args.args
            if a.annotation and ast.unparse(a.annotation).endswith("constexpr")
        }
        _generate(fn, dict.fromkeys(self.constexprs, 0))  # Fail early.
        self._compiled: dict[str, tuple[Callable[..., None], list[str]]] = {}
        # The kernel's parameters plus the launch grid, for the dispatcher.
        grid = inspect.Parameter("grid", inspect.Parameter.KEYWORD_ONLY)
        self.__signature__ = self.signature.replace(
            parameters=[*self.signature.parameters.values(), grid]
        )

    def __getitem__(self, grid: Any) -> Callable[..., None]:
        return lambda *args, **kwargs: self(*args, grid=grid, **kwargs)

    def __call__(self, *args: Any, grid: Any, **kwargs: Any) -> None:
        for option in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
            kwargs.pop(option, None)
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        consts = {k: v for k, v in values.items() if k in self.constexprs or v is None}
        key = repr(sorted((k, _repr(v)) for k, v in consts.items()))
        if key not in self._compiled:
            src, params = _generate(self.fn, consts)
            self._compiled[key] = _compile(src), params
        launch, params = self._compiled[key]
        grid = tuple(grid(values) if callable(grid) else grid)
        # numpy has no bfloat16: run on float32 copies, then write them back.
        bf16 = {
            p: values[p].float()
            for p in params
            if isinstance(values[p], torch.Tensor) and values[p].dtype == torch.bfloat16
        }
        launch(
            *grid + (1,) * (3 - len(grid)),
            *(
                _to_ptr(v) if isinstance(v, torch.Tensor) else v
                for v in (bf16.get(p, values[p]) for p in params)
            ),
        )
        for p, copy in bf16.items():
            values[p].copy_(copy)


def translate_kernel(qualname: str) -> NumbaKernel:
    """Translates the Triton kernel at `qualname`, e.g. `pkg.module._kernel`."""
    module, name = qualname.rsplit(".", 1)
    return NumbaKernel(getattr(importlib.import_module(module), name))


def translate_kernels(root: str) -> dict[str, NumbaKernel]:
    """Translates the `*_kernel` Triton functions under `root`, a path in the
    vllm source tree, keyed by qualified name for `register_kernels`."""
    base = pathlib.Path(__file__).parents[2]
    paths = [base / root] if (base / root).is_file() else (base / root).rglob("*.py")
    kernels: dict[str, NumbaKernel] = {}
    for path in sorted(paths):
        if "triton.jit" not in path.read_text():
            continue
        module_name = ".".join(path.relative_to(base).with_suffix("").parts)
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name, fn in list(vars(module).items()):
            if name.endswith("_kernel") and _is_triton_jit(fn):
                with contextlib.suppress(UnsupportedKernel):
                    kernels[f"{module_name}.{name}"] = NumbaKernel(fn)
    return kernels
