# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse ``cute.compile`` results without tracing (VLLM_CUTE_DSL_AOT_CACHE).

The compile cache (VLLM_CUTE_DSL_COMPILE_CACHE) skips the MLIR -> cubin
pipeline, but CuTe DSL still traces every kernel to compute its key, and on a
warm start tracing is most of what is left.

This cache computes its key before tracing, from what tracing and the compiled
binary depend on:

- the content of the installed vllm, flashinfer and quack Python sources and
  of the cutlass and tvm_ffi sources and binaries, by path relative to the
  package, plus the CuTe DSL runtime libraries, so moving the installation or
  touching mtimes keeps the key and rebuilding a binary changes it;
- the host (CPU architecture, OS, libc, Python), the GPU (name, compute
  capability, SM count), the CUDA runtime and driver versions and the torch,
  CuTe DSL, FlashInfer, QuACK and TVM FFI distribution versions;
- vLLM's compile factors and every ``CUTE_*``, ``CUTLASS_*``, ``FLASHINFER_*``
  and ``NVIDIA_TF32*`` variable except the cache and dump locations in
  ``_LOCATION_ENVS``;
- the compile options, the callable (function, closure, defaults and, for
  methods and kernel objects, the instance state; every function, class and
  enum by its full module name, and for code outside the hashed packages, the
  content of its whole top-level package) and a structural description of
  every argument.

A miss compiles as usual and exports the result as an object file; a hit loads
it with ``cute.runtime.load_module``. A hit behaves like the compiled function
when called; using any other attribute (``export_to_c``, ...) compiles the
kernel for real first, reported to the JIT monitor like a ``cute.compile``
call, so callers that export or inspect the result keep working. An entry that
passes its checksum but fails to load is replaced by the recompiled result,
unless another process has already replaced it.

A compile runs uncached when its key cannot be computed exactly (torch tensors
or opaque objects among the arguments, unreadable code, no CUDA device), when
it asks for dumps, device functions, extra link libraries or no caching, or
when its compiled function cannot be exported. Tracing must not read other
mutable state (module globals, files). Attributes that tracing sets on the
kernel object are restored on a hit when they pickle; the others are replaced
by placeholders that raise on use.

Each entry is one file holding a format version, the key, the metadata, the
object file and their SHA-256, published under a unique temporary name with a
link that never replaces a valid entry. Corrupt, truncated, older-format and
untrusted entries are misses. Entries contain executable code and pickled
metadata, so only processes trusted to run code in the deployment may write
the directory; a file is used only if its owner is this user, root or the
directory's owner and it is not world-writable.
"""

import contextlib
import enum
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import os
import pickle
import platform
import stat
import struct
import sys
import sysconfig
import tempfile
import threading
import time
import types
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_installed: bool = False
_cache_dir: str | None = None
_writable: bool = False

_FORMAT_VERSION = 1
_MAGIC = b"vLLM-CuTe-AOT\x00"
# magic, format version, key, metadata length, object length, SHA-256 of both.
_HEADER = struct.Struct("<14sI32sQQ32s")
_SUFFIX = ".cuteaot"

_PLAIN = (bool, int, float, str, bytes, type(None))
# Runtime pointers without ``__cache_key__``: codegen sees their dtype, address
# space and alignment, never the address.
_POINTER_TYPES = {"flashinfer.cute_dsl.utils._Pointer"}
# Packages whose Python code tracing runs, including QuACK, whose helpers vLLM
# kernels call. Only the DSL compiler and runtime binaries affect the result;
# vLLM and FlashInfer extensions are not traced.
_SOURCE_PACKAGES = ("vllm", "flashinfer", "quack")
_BINARY_PACKAGES = ("cutlass", "tvm_ffi")
# Identified by their distribution version rather than their content.
_VERSIONED_MODULES = frozenset({"torch"})
_KEY_DISTRIBUTIONS = (
    "torch",
    "nvidia-cutlass-dsl",
    "flashinfer-python",
    "quack-kernels",
    "apache-tvm-ffi",
)
_KEY_ENV_PREFIXES = ("CUTE_", "CUTLASS_", "FLASHINFER_", "NVIDIA_TF32")
# Where CuTe DSL keeps its file cache and dumps (both declared not to affect
# compilation) and where it finds its runtime libraries (hashed by content).
_LOCATION_ENVS = frozenset({"CUTE_DSL_CACHE_DIR", "CUTE_DSL_DUMP_DIR", "CUTE_DSL_LIBS"})
_NO_CACHE_ENVS = (
    "CUTE_DSL_NO_CACHE",
    "CUTE_DSL_KEEP",
    "CUTE_DSL_KEEP_PTX",
    "CUTE_DSL_KEEP_CUBIN",
    "CUTE_DSL_KEEP_SASS",
    "CUTE_DSL_KEEP_IR",
    "CUTE_DSL_DRYRUN",
    "CUTE_DSL_PRINT_IR",
)
# Compile options whose effects (dumped files, device functions, linked
# libraries read from disk) a cache hit would not reproduce.
_NO_CACHE_OPTIONS = frozenset(
    {
        "KeepPTX",
        "KeepCUBIN",
        "KeepSASS",
        "DumpDir",
        "DeviceTarget",
        "LinkLibraries",
        "FlattenLocsOutputJsonPath",
    }
)
_NO_CACHE_KWARGS = ("trace_finalize_hooks", "compile_to_precompiled_mlir")


class _Uncacheable(Exception):
    pass


class _InvalidEntry(Exception):
    pass


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

_stats_lock = threading.Lock()
_stats: dict[str, float] = {}
_uncacheable_reasons: Counter[str] = Counter()


def _record(name: str, value: float = 1) -> None:
    with _stats_lock:
        _stats[name] = _stats.get(name, 0) + value


def get_aot_cache_stats() -> dict[str, float]:
    """Counters and timings of this process's AOT cache.

    ``hits`` loaded a kernel instead of tracing it; ``misses`` traced and
    compiled a cacheable kernel; ``uncacheable`` compiled a kernel that cannot
    be keyed exactly. ``invalid_entries`` and ``load_failures`` are entries that
    were found but not used. ``exports`` wrote a new entry,
    ``export_existing`` found one written concurrently, ``export_unsupported``
    and ``export_failures`` wrote none. ``deferred_compiles`` compiled a hit
    for a caller that needed more than calling it. ``*_s`` are wall times.
    """
    with _stats_lock:
        return dict(_stats)


def log_cute_dsl_cache_stats() -> None:
    """Log this process's CuTe DSL cache counters, if either cache is on."""
    from vllm.utils import cute_dsl_compile_cache

    if not (_installed or cute_dsl_compile_cache._installed):
        return
    with _stats_lock:
        reasons = _uncacheable_reasons.most_common(5)
    logger.info(
        "CuTe DSL caches: AOT %s (top uncacheable: %s), compile %s",
        _format(get_aot_cache_stats()),
        reasons,
        _format(cute_dsl_compile_cache.get_compile_cache_stats()),
    )


def _format(stats: Mapping[str, float]) -> str:
    return ", ".join(
        f"{k}={v:.3f}" if k.endswith("_s") else f"{k}={int(v)}"
        for k, v in sorted(stats.items())
    )


# ---------------------------------------------------------------------------
# Identity: code, host, GPU and environment
# ---------------------------------------------------------------------------


def _is_binary(filename: str) -> bool:
    return filename.endswith(".so") or ".so." in filename


def _package_files(
    package: str, roots: Iterable[str], binaries: bool
) -> list[tuple[str, str]]:
    """``(label, path)`` of the package's sources (and binaries), labelled by
    their path relative to the package root."""
    files = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            rel_dir = os.path.relpath(dirpath, root)
            for filename in filenames:
                if filename.endswith(".py") or (binaries and _is_binary(filename)):
                    rel = os.path.normpath(os.path.join(rel_dir, filename))
                    label = f"{package}/{rel.replace(os.sep, '/')}"
                    files.append((label, os.path.join(dirpath, filename)))
    return files


def _file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files_digest(files: Iterable[tuple[str, str]]) -> str:
    """Digest of the files' labels and contents, independent of their paths."""
    files = sorted(files)
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        digests = list(pool.map(lambda item: _file_digest(item[1]), files))
    h = hashlib.sha256()
    for (label, _), digest in zip(files, digests):
        h.update(f"{label}\0{digest}\n".encode())
    return h.hexdigest()


def _runtime_libraries() -> list[tuple[str, str]]:
    """The CuTe DSL and TVM FFI libraries ``load_module`` links against."""
    from cutlass.base_dsl.env_manager import get_prefix_dsl_libs

    paths = (get_prefix_dsl_libs("CUTE_DSL") or "").split(os.pathsep)
    import tvm_ffi

    paths.append(tvm_ffi.libinfo.find_libtvm_ffi())
    return [(f"lib/{os.path.basename(p)}", p) for p in paths if p and os.path.isfile(p)]


@functools.cache
def _hashed_roots(package: str) -> list[str]:
    spec = importlib.util.find_spec(package)
    return sorted(spec.submodule_search_locations or ()) if spec else []


def _code_identity() -> str:
    files = []
    for package in _SOURCE_PACKAGES + _BINARY_PACKAGES:
        roots = _hashed_roots(package)
        files += _package_files(package, roots, package in _BINARY_PACKAGES)
    files += _runtime_libraries()
    versions = []
    for dist in _KEY_DISTRIBUTIONS:
        try:
            versions.append(f"{dist}=={importlib.metadata.version(dist)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{dist} absent")
    return f"{_files_digest(files)}|{','.join(versions)}"


def _host_identity() -> str:
    abi = sysconfig.get_config_var("SOABI")
    if not abi:
        raise _Uncacheable("Python build ABI is unavailable")
    return "|".join(
        (
            sys.platform,
            platform.machine(),
            "-".join(platform.libc_ver()),
            sys.implementation.cache_tag or "",
            abi,
            platform.python_version(),
        )
    )


def _gpu_identity(device: int) -> str:
    import torch
    from cuda.bindings import driver

    props = torch.cuda.get_device_properties(device)

    err, driver_version = driver.cuDriverGetVersion()
    if err != driver.CUresult.CUDA_SUCCESS:
        raise _Uncacheable(f"unknown CUDA driver version ({err})")
    return (
        f"{props.name}|{props.major}.{props.minor}|{props.multi_processor_count}"
        f"|cuda {torch.version.cuda}|driver {driver_version}"
    )


def _env_identity(environ: Mapping[str, str], factors: Mapping[str, Any]) -> str:
    """The compile factors plus the CuTe DSL / CUTLASS / FlashInfer
    variables, without the ones that only say where files live."""
    env = sorted((k, repr(v)) for k, v in factors.items())
    env += sorted(
        (k, v)
        for k, v in environ.items()
        if k.startswith(_KEY_ENV_PREFIXES) and k not in _LOCATION_ENVS
    )
    return hashlib.sha256(repr(env).encode()).hexdigest()


_identity_lock = threading.Lock()
_identity: dict[str, str | _Uncacheable] = {}


def _memoized(name: str, compute: Callable[[], str]) -> str:
    """``compute()`` once per process; a failure is remembered as well."""
    with _identity_lock:
        if name not in _identity:
            start = time.perf_counter()
            try:
                _identity[name] = compute()
            except _Uncacheable as e:
                _identity[name] = e
            except Exception as e:
                _identity[name] = _Uncacheable(f"{name} identity unavailable: {e!r}")
            _record("identity_s", time.perf_counter() - start)
        value = _identity[name]
    if isinstance(value, _Uncacheable):
        raise value
    return value


def _process_identity() -> str:
    import torch

    if not torch.cuda.is_available():
        raise _Uncacheable("no CUDA device")
    device = torch.accelerator.current_device_index()
    return hashlib.sha256(
        "\n".join(
            (
                f"format {_FORMAT_VERSION}",
                _memoized("code", _code_identity),
                _memoized("host", _host_identity),
                _memoized(f"gpu{device}", lambda: _gpu_identity(device)),
            )
        ).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Describing the callable and its arguments
# ---------------------------------------------------------------------------


def _is_under(path: str, roots: Iterable[str]) -> bool:
    real = os.path.realpath(path)
    return any(real.startswith(os.path.join(os.path.realpath(r), "")) for r in roots)


def _source_digest(module_name: str, path: str | None) -> str:
    if not path or not path.endswith(".py"):
        raise _Uncacheable(f"code in {module_name} that cannot be fingerprinted")
    try:
        return _file_digest(path)
    except OSError as e:
        raise _Uncacheable(f"unreadable source of {module_name}: {e}") from e


@functools.cache
def _package_digest(package: str) -> str:
    """Content of an imported top-level package (or single-file module) that
    the code identity does not cover, independent of where it is installed."""
    module = sys.modules.get(package)
    roots = sorted(getattr(module, "__path__", None) or ())
    if not roots:
        return _source_digest(package, getattr(module, "__file__", None))
    try:
        return _files_digest(_package_files(package, roots, binaries=True))
    except OSError as e:
        raise _Uncacheable(f"unreadable files in {package}: {e}") from e


@functools.cache
def _module_identity(module_name: str) -> str:
    """Full name of ``module_name`` plus whatever identifies its code.

    Hashed packages are covered by the code identity, the standard library by
    the Python version and versioned modules by their distribution version.
    Other code is identified by the content of its whole top-level package,
    since a module behaves according to the helpers it imports from there. A
    module whose file lies outside its package's tree adds its own content.
    """
    if module_name == "__main__":
        raise _Uncacheable("code in __main__")
    top = module_name.partition(".")[0]
    if top in sys.stdlib_module_names or top in _VERSIONED_MODULES:
        return module_name
    path = getattr(sys.modules.get(module_name), "__file__", None)
    if top in _SOURCE_PACKAGES + _BINARY_PACKAGES:
        # Names without a file of their own come from the package's hashed
        # extension binaries.
        if path is None or _is_under(path, _hashed_roots(top)):
            return module_name
        return f"{module_name}@{_source_digest(module_name, path)}"
    identity = f"{module_name}@{top}:{_package_digest(top)}"
    roots = getattr(sys.modules.get(top), "__path__", None) or ()
    if roots and (path is None or not _is_under(path, roots)):
        identity += f"@{_source_digest(module_name, path)}"
    return identity


class _DynamicScalar:
    """A Python scalar passed for a dynamic (non-Constexpr) numeric parameter.

    It becomes a runtime argument of the compiled function, so only its type
    reaches the traced IR.
    """

    def __init__(self, annotation: type, value: object) -> None:
        self.annotation = annotation
        self.value_type = type(value)


class _TraceOnly:
    """Stands in for a kernel attribute that only tracing sets (layouts, local
    classes); any use after a cache hit raises instead of reading a wrong value.
    """

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def _fail(self, *args, **kwargs):
        raise RuntimeError(
            f"{self._name!r} is set while tracing a CuTe DSL kernel and is not "
            "available after a VLLM_CUTE_DSL_AOT_CACHE hit."
        )

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        self._fail()

    __call__ = __bool__ = __int__ = __index__ = __float__ = __len__ = _fail
    __iter__ = __getitem__ = __eq__ = __add__ = __mul__ = _fail


class _Describer:
    """Builds a canonical description of compile inputs.

    Values are described by content; objects reached twice (e.g. one SymInt
    shared by two fake tensors) by their first-visit index, so a description
    captures sharing without depending on ``id()``.
    """

    def __init__(self) -> None:
        self._visited: dict[int, int] = {}

    def __call__(self, v: Any, depth: int = 0) -> str:
        if depth > 32:
            raise _Uncacheable("argument nesting is too deep")
        if isinstance(v, _PLAIN):
            return f"{type(v).__name__}:{v!r}"
        if isinstance(v, enum.Enum):
            return f"enum:{_code(type(v))}.{v.name}"
        if isinstance(v, type):
            return f"type:{_code(v)}"
        if isinstance(v, (tuple, list)):
            return f"{type(v).__name__}[{','.join(self(x, depth + 1) for x in v)}]"
        if isinstance(v, (set, frozenset)):
            items = sorted(self(x, depth + 1) for x in v)
            return f"{type(v).__name__}{{{','.join(items)}}}"
        if isinstance(v, dict):
            items = sorted(
                f"{self(k, depth + 1)}={self(x, depth + 1)}" for k, x in v.items()
            )
            return f"{type(v).__name__}{{{','.join(items)}}}"
        if isinstance(v, _DynamicScalar):
            return f"dynamic:{_qualname(v.annotation)}:{v.value_type.__name__}"
        torch = sys.modules.get("torch")
        if torch is not None:
            if isinstance(v, (torch.dtype, torch.device)):
                return f"torch:{v}"
            if isinstance(v, torch.Tensor):
                raise _Uncacheable("a torch.Tensor argument")
        if isinstance(v, types.MethodType):
            return (
                f"method[{self(v.__func__, depth + 1)}|{self(v.__self__, depth + 1)}]"
            )
        if isinstance(v, types.FunctionType):
            v = inspect.unwrap(v)
            parts = [f"function:{_module_identity(v.__module__)}:{v.__qualname__}"]
            if v.__closure__:
                try:
                    cells = [c.cell_contents for c in v.__closure__]
                except ValueError as e:
                    raise _Uncacheable("an unbound closure variable") from e
                parts.append(self(cells, depth + 1))
            parts.append(self(v.__defaults__, depth + 1))
            parts.append(self(v.__kwdefaults__, depth + 1))
            return "|".join(parts)
        if isinstance(v, (types.BuiltinFunctionType, types.ModuleType)):
            return f"{type(v).__name__}:{getattr(v, '__module__', '')}.{v.__name__}"
        if isinstance(v, functools.partial):
            return (
                f"partial[{self(v.func, depth + 1)}|{self(v.args, depth + 1)}"
                f"|{self(v.keywords, depth + 1)}]"
            )
        cls = type(v)
        if _qualname(cls) in _POINTER_TYPES:
            return (
                f"pointer:{self(v._dtype, depth + 1)}|{v._addr_space}"
                f"|{v._assumed_align}"
            )
        if cls.__module__.startswith(("cuda.bindings.driver", "cuda.bindings.runtime")):
            # Streams and contexts are runtime arguments.
            return f"handle:{_qualname(cls)}"
        if hasattr(cls, "__cache_key__") and not any(
            k.__name__.startswith("_Fake") for k in cls.__mro__
        ):
            # from_dlpack tensors and cute pointers: CuTe DSL's own
            # specialization key. Fake tensors key SymInts by id(), so they are
            # described by their state instead.
            return f"cache_key:{_code(cls)}[{self(v.__cache_key__, depth + 1)}]"
        if id(v) in self._visited:
            return f"ref:{self._visited[id(v)]}"
        self._visited[id(v)] = len(self._visited)
        return f"object:{_code(cls)}{{{self._state(v, depth)}}}"

    def _state(self, v: Any, depth: int) -> str:
        state = dict(getattr(v, "__dict__", None) or {})
        for klass in type(v).__mro__:
            slots = getattr(klass, "__slots__", ())
            for name in (slots,) if isinstance(slots, str) else slots:
                if name not in ("__dict__", "__weakref__") and hasattr(v, name):
                    state[name] = getattr(v, name)
        if not state and not hasattr(v, "__dict__"):
            raise _Uncacheable(f"an opaque {_qualname(type(v))} argument")
        return ",".join(f"{k}={self(x, depth + 1)}" for k, x in sorted(state.items()))


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _code(cls: type) -> str:
    return f"{_module_identity(cls.__module__)}:{cls.__qualname__}"


def _owner(func: Any) -> Any:
    """The object whose state tracing reads, if any."""
    if isinstance(func, types.MethodType):
        return func.__self__
    if isinstance(func, types.FunctionType):
        return None
    return func


def _normalize_args(
    func: Any, args: tuple, kwargs: dict
) -> tuple[list[Any], dict[str, Any]]:
    """Replace scalars bound to dynamic numeric parameters with their type."""
    from cutlass.base_dsl.runtime.jit_arg_adapters import is_arg_annotation_constexpr
    from cutlass.base_dsl.typing import Numeric

    if isinstance(func, types.MethodType):
        target, offset = func.__func__, 1
    elif isinstance(func, types.FunctionType):
        target, offset = func, 0
    else:
        target, offset = type(func).__call__, 1
    target = inspect.unwrap(target)
    try:
        params = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        return list(args), dict(kwargs)

    def normalize(index: int, param: inspect.Parameter, value: Any) -> Any:
        ann = param.annotation
        if (
            type(value) in (bool, int, float)
            and isinstance(ann, type)
            and issubclass(ann, Numeric)
            and not is_arg_annotation_constexpr(ann, param.name, index, target)
        ):
            return _DynamicScalar(ann, value)
        return value

    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    out_args = []
    for i, value in enumerate(args, start=offset):
        if i < len(params) and params[i].kind in positional:
            value = normalize(i, params[i], value)
        out_args.append(value)
    by_name = {p.name: (i, p) for i, p in enumerate(params)}
    out_kwargs = {
        k: normalize(*by_name[k], v) if k in by_name else v for k, v in kwargs.items()
    }
    return out_args, out_kwargs


def _check_options(compile_options: Any, options: Any) -> None:
    from cutlass.base_dsl.compiler import _parse_compile_options_from_str

    parsed = [compile_options]
    if isinstance(options, str) and options:
        parsed.append(_parse_compile_options_from_str(options))
    elif options is not None:
        raise _Uncacheable(f"options of type {type(options).__name__}")
    for opts in parsed:
        for cls, option in getattr(opts, "options", {}).items():
            if cls.__name__ in _NO_CACHE_OPTIONS and option.value:
                raise _Uncacheable(f"the compile option {cls.__name__}")


def _cache_key(compile_callable: Any, func: Any, args: tuple, kwargs: dict) -> str:
    if any(os.environ.get(v, "0") not in ("", "0") for v in _NO_CACHE_ENVS):
        raise _Uncacheable("CuTe DSL is asked for dumps or no caching")
    if any(kwargs.get(k) for k in _NO_CACHE_KWARGS):
        raise _Uncacheable("a trace hook or a precompiled-MLIR compile")
    compile_options = getattr(compile_callable, "_compile_options", None)
    _check_options(compile_options, kwargs.get("options"))
    process = _process_identity()
    environment = _env_identity(os.environ, envs.compile_factors())
    key_args, key_kwargs = _normalize_args(func, args, kwargs)
    describe = _Describer()
    parts = [
        process,
        environment,
        describe(compile_options),
        describe(func),
        describe(key_args),
        describe(key_kwargs),
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def _entry_path(directory: str, key: str) -> str:
    return os.path.join(directory, key + _SUFFIX)


def _pack_entry(key: str, metadata: bytes, obj: bytes) -> bytes:
    digest = hashlib.sha256(metadata + obj).digest()
    header = _HEADER.pack(
        _MAGIC, _FORMAT_VERSION, bytes.fromhex(key), len(metadata), len(obj), digest
    )
    return header + metadata + obj


def _unpack_entry(key: str, data: bytes) -> tuple[bytes, bytes]:
    if len(data) < _HEADER.size:
        raise _InvalidEntry("truncated header")
    magic, version, entry_key, meta_len, obj_len, digest = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != _FORMAT_VERSION:
        raise _InvalidEntry("unknown format")
    if entry_key != bytes.fromhex(key):
        raise _InvalidEntry("entry for another key")
    if len(data) != _HEADER.size + meta_len + obj_len:
        raise _InvalidEntry("truncated or oversized entry")
    body = memoryview(data)[_HEADER.size :]
    if hashlib.sha256(body).digest() != digest:
        raise _InvalidEntry("checksum mismatch")
    return bytes(body[:meta_len]), bytes(body[meta_len:])


def _check_trusted(directory: str, st: os.stat_result) -> None:
    dir_st = os.stat(directory)
    if dir_st.st_mode & stat.S_IWOTH and not dir_st.st_mode & stat.S_ISVTX:
        raise _InvalidEntry("world-writable cache directory")
    if st.st_mode & stat.S_IWOTH:
        raise _InvalidEntry("world-writable entry")
    if st.st_uid not in (os.geteuid(), 0, dir_st.st_uid):
        raise _InvalidEntry(f"entry owned by untrusted uid {st.st_uid}")


def _read_entry(directory: str, key: str) -> tuple[bytes, bytes] | None:
    """The entry's metadata and object file; None if there is no entry.

    Raises:
        _InvalidEntry: The entry is unreadable, damaged, from another format
            version or written by an untrusted user.

    """
    try:
        with open(_entry_path(directory, key), "rb") as f:
            _check_trusted(directory, os.fstat(f.fileno()))
            data = f.read()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as e:
        raise _InvalidEntry(f"unreadable entry: {e}") from e
    return _unpack_entry(key, data)


def _write_entry(
    directory: str,
    key: str,
    metadata: bytes,
    obj: bytes,
    replace: tuple[bytes, bytes] | None = None,
) -> bool:
    """Publish an entry; False if a valid one already exists.

    The entry appears complete or not at all: it is written to a uniquely named
    temporary file and linked into place, which fails rather than replace an
    entry another process published first. A damaged entry is replaced, and so
    is a valid one whose content is ``replace`` (an entry that failed to load);
    an observed different entry, such as another writer's repair, is kept.
    This check is not a compare-and-swap: concurrent repairs may replace each
    other's complete, same-key entries. Readers still see one coherent entry.
    """
    final = _entry_path(directory, key)
    fd, tmp = tempfile.mkstemp(prefix=f".{key[:16]}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_pack_entry(key, metadata, obj))
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        try:
            os.link(tmp, final)
            return True
        except FileExistsError:
            pass
        except OSError:
            # Filesystems without hard links: rename, which may replace an
            # equivalent entry published concurrently.
            pass
        try:
            current = _read_entry(directory, key)
        except _InvalidEntry:
            current = None
        if current is not None and current != replace:
            return False
        os.replace(tmp, final)
        return True
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def _prepare_cache_dir(directory: str) -> bool | None:
    """Create the directory if needed.

    Returns:
        Whether new entries can be written, or None if the directory does not
        exist and cannot be created.

    """
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        if not os.path.isdir(directory):
            logger.warning("Cannot create %s: %s", directory, e)
            return None
    if not os.access(directory, os.W_OK | os.X_OK):
        logger.info("%s is read-only; using existing CuTe DSL AOT entries.", directory)
        return False
    return True


# ---------------------------------------------------------------------------
# Export and load
# ---------------------------------------------------------------------------


def _export(
    compiled: Any,
    key: str,
    owner: Any,
    before: dict,
    replace: tuple[bytes, bytes] | None,
) -> bool:
    """Write the compiled function as an entry, replacing ``replace`` (an
    entry that failed to load); False if it cannot be written."""
    from cutlass.base_dsl.jit_executor import JitCompiledFunction
    from cutlass.cutlass_dsl.tvm_ffi_provider import (
        TVMFFIJitCompiledFunction,
        TVMFFIJitCompiledFunctionWithKwargs,
    )

    restore, trace_only = {}, []
    for name, value in (getattr(owner, "__dict__", None) or {}).items():
        if name in before and before[name] is value:
            continue
        try:
            pickle.dumps(value)
        except Exception:
            if hasattr(type(owner), name):
                # A hit would expose the class attribute instead.
                return False
            trace_only.append(name)
        else:
            restore[name] = value

    kwargs_spec = None
    if type(compiled) in (
        TVMFFIJitCompiledFunction,
        TVMFFIJitCompiledFunctionWithKwargs,
    ):
        if compiled.execution_args.has_pointer_address_arg_specs:
            return False
        if type(compiled) is TVMFFIJitCompiledFunctionWithKwargs:
            spec, dataclass_args = compiled._vllm_aot_kwargs_spec
            kwargs_spec = {**spec._asdict(), "map_dataclass_to_tuple": dataclass_args}
        tvm_ffi = True
    elif isinstance(compiled, JitCompiledFunction) and not hasattr(
        compiled, "_kwargs_wrapper"
    ):
        tvm_ffi = False
    else:
        return False

    prefix = f"vllm_aot_{key[:24]}"
    metadata = pickle.dumps(
        {
            "prefix": prefix,
            "tvm_ffi": tvm_ffi,
            "kwargs_spec": kwargs_spec,
            "restore": restore,
            "trace_only": trace_only,
        }
    )
    if tvm_ffi:
        with tempfile.TemporaryDirectory(prefix="vllm-cute-aot-") as tmp:
            path = os.path.join(tmp, "kernel.o")
            compiled.export_to_c(path, prefix)
            with open(path, "rb") as f:
                obj = f.read()
    else:
        obj = compiled.dump_to_object(prefix)
    assert _cache_dir is not None
    if _write_entry(_cache_dir, key, metadata, obj, replace):
        _record("exports")
    else:
        _record("export_existing")
    return True


class _LoadedKernel:
    """A kernel loaded from the cache.

    Calls go straight to the loaded function. Any other attribute (for
    example ``export_to_c``) comes from the kernel compiled as usual, on first
    use, so code written against a compiled function keeps working on a hit.
    That compile does not pass through ``cute.compile``, so it is reported to
    the JIT monitor itself.
    """

    __slots__ = ("_name", "_fn", "_module", "_compile_fn", "_compiled", "_lock")

    def __init__(
        self, name: str, fn: Callable, module: Any, compile_fn: Callable
    ) -> None:
        self._name = name
        self._fn = fn
        self._module = module
        self._compile_fn = compile_fn
        self._compiled: Any = None
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def _compile(self) -> Any:
        from vllm.utils import jit_monitor

        with self._lock:
            if self._compiled is None:
                jit_monitor.report_cutedsl_compile(self._name)
                logger.debug("Compiling AOT cache hit %s for export.", self._name)
                self._compiled = self._compile_fn()
                _record("deferred_compiles")
            return self._compiled

    def __getattr__(self, name: str):
        if name in _LoadedKernel.__slots__ or (
            name.startswith("__") and name.endswith("__")
        ):
            raise AttributeError(name)
        return getattr(self._compile(), name)


def _load(
    metadata: bytes, obj: bytes, owner: Any, name: str, compile_fn: Callable
) -> _LoadedKernel:
    import cutlass.cute as cute

    meta = pickle.loads(metadata)
    with tempfile.NamedTemporaryFile(prefix="vllm-cute-aot-", suffix=".o") as f:
        f.write(obj)
        f.flush()
        # load_module reads the object file before returning.
        module = cute.runtime.load_module(f.name, enable_tvm_ffi=meta["tvm_ffi"])
    fn = module[meta["prefix"]]
    if meta["kwargs_spec"] is not None:
        from tvm_ffi.utils import kwargs_wrapper

        fn = kwargs_wrapper.make_kwargs_wrapper(fn, **meta["kwargs_spec"])
    for name, value in meta["restore"].items():
        setattr(owner, name, value)
    for name in meta["trace_only"]:
        setattr(owner, name, _TraceOnly(name))
    return _LoadedKernel(name, fn, module, compile_fn)


def _name(func: Any) -> str:
    return getattr(func, "__qualname__", None) or type(func).__qualname__


def _timed(name: str, fn: Callable[[], Any]) -> Any:
    start = time.perf_counter()
    try:
        return fn()
    finally:
        _record(name, time.perf_counter() - start)


def maybe_enable_cute_dsl_aot_cache() -> bool:
    """Enable the cache in this process if VLLM_CUTE_DSL_AOT_CACHE is set.

    Must run before the first ``cute.compile`` of the process. Returns whether
    the cache is enabled.
    """
    global _installed, _cache_dir, _writable
    if _installed:
        return True
    if not envs.VLLM_CUTE_DSL_AOT_CACHE:
        return False

    try:
        from cutlass.base_dsl.compiler import (
            CompileCallable,
            _parse_compile_options_from_str,  # noqa: F401
        )
        from cutlass.cutlass_dsl.tvm_ffi_provider import (
            TVMFFIJitCompiledFunctionWithKwargs,
        )
        from tvm_ffi.utils import kwargs_wrapper  # noqa: F401
    except Exception:
        logger.debug("CuTe DSL is not available; not persisting cute.compile results.")
        return False
    compile_ = getattr(CompileCallable, "_compile", None)
    if compile_ is None:
        logger.warning(
            "VLLM_CUTE_DSL_AOT_CACHE is set, but this CuTe DSL version has no "
            "CompileCallable._compile hook; cute.compile results are not persisted."
        )
        return False

    cache_dir = os.path.join(envs.VLLM_CACHE_ROOT, "cute_dsl_aot")
    writable = _prepare_cache_dir(cache_dir)
    if writable is None:
        return False
    _cache_dir, _writable = cache_dir, writable

    kwargs_init = TVMFFIJitCompiledFunctionWithKwargs.__init__

    @functools.wraps(kwargs_init)
    def _init_keeping_kwargs_spec(self, *args, **kwargs):
        spec = kwargs.get("kwargs_wrapper_spec")
        dataclass_args = list(kwargs.get("map_dataclass_to_tuple") or ())
        kwargs_init(self, *args, **kwargs)
        # A hit rebuilds the same kwargs wrapper around the loaded function.
        self._vllm_aot_kwargs_spec = (spec, dataclass_args)

    @functools.wraps(compile_)
    def _compile(self, func, *args, **kwargs):
        def compile_now():
            return compile_(self, func, *args, **kwargs)

        start = time.perf_counter()
        try:
            key = _cache_key(self, func, args, kwargs)
        except Exception as e:
            reason = str(e) if isinstance(e, _Uncacheable) else repr(e)
            logger.debug("Not persisting cute.compile of %s: %s", _name(func), reason)
            _record("uncacheable")
            with _stats_lock:
                _uncacheable_reasons[reason] += 1
            return _timed("compile_s", compile_now)
        _record("key_s", time.perf_counter() - start)

        assert _cache_dir is not None
        owner = _owner(func)
        name = _name(func)
        failed = None
        try:
            entry = _read_entry(_cache_dir, key)
        except _InvalidEntry as e:
            logger.warning(
                "Ignoring CuTe DSL AOT entry %s: %s", _entry_path(_cache_dir, key), e
            )
            _record("invalid_entries")
            entry = None
        if entry is not None:
            try:
                kernel = _timed(
                    "load_s", lambda: _load(*entry, owner, name, compile_now)
                )
            except Exception as e:
                logger.warning(
                    "Failed to load %s from %s (%s); compiling it.", name, _cache_dir, e
                )
                _record("load_failures")
                # The new export replaces exactly this entry, so it does not
                # fail every later start, but keeps another writer's repair.
                failed = entry
            else:
                _record("hits")
                return kernel

        _record("misses")
        before = dict(getattr(owner, "__dict__", None) or {})
        compiled = _timed("compile_s", compile_now)
        if _writable:
            try:
                if not _timed(
                    "export_s", lambda: _export(compiled, key, owner, before, failed)
                ):
                    logger.debug("cute.compile of %s cannot be persisted.", _name(func))
                    _record("export_unsupported")
            except Exception as e:
                logger.debug("Failed to persist cute.compile of %s: %s", _name(func), e)
                _record("export_failures")
        return compiled

    TVMFFIJitCompiledFunctionWithKwargs.__init__ = _init_keeping_kwargs_spec
    CompileCallable._compile = _compile
    _installed = True
    logger.info_once(
        "Persisting cute.compile results in %s (VLLM_CUTE_DSL_AOT_CACHE).",
        _cache_dir,
    )
    return True
