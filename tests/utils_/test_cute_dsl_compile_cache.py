# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm.utils.cute_dsl_compile_cache as compile_cache


@pytest.fixture
def dsl_stubs(monkeypatch, tmp_path):
    """The external CuTe DSL hooks, without requiring its CUDA libraries."""

    @dataclass
    class Setup:
        no_cache: bool = True
        compile_only: bool = True

    compiler = ModuleType("cutlass.base_dsl.compiler")
    options = {}
    for name in ("KeepPTX", "KeepCUBIN", "KeepSASS"):
        option = type(name, (), {})
        setattr(compiler, name, option)
        options[option] = SimpleNamespace(value=False)
    monkeypatch.setattr(
        compiler,
        "CompileOptions",
        lambda: SimpleNamespace(options=options),
        raising=False,
    )

    class BaseDSL:
        _CompilationSetup = Setup

        def __init__(self):
            self.envar = SimpleNamespace()
            self.compile_options = SimpleNamespace(options=options)

        def _prepare_compilation(self, funcBody):
            return Setup()

    class Context:
        current = None

        def __enter__(self):
            Context.current = self
            self.threading = True
            return self

        def enable_multithreading(self, enabled):
            self.threading = enabled

        def __exit__(self, *args):
            Context.current = None

    def parse(content):
        assert Context.current is not None
        assert Context.current.threading is False
        return content

    def load_ir(file, asBytecode=False, bytecode_reader=None):
        raise AssertionError("the original multithreaded loader must be replaced")

    helpers = ModuleType("cutlass.base_dsl.cache_helpers")
    monkeypatch.setattr(helpers, "load_ir", load_ir, raising=False)
    monkeypatch.setattr(
        helpers,
        "ir",
        SimpleNamespace(Context=Context, Module=SimpleNamespace(parse=parse)),
        raising=False,
    )
    dsl = ModuleType("cutlass.base_dsl.dsl")
    monkeypatch.setattr(dsl, "BaseDSL", BaseDSL, raising=False)
    for module in (dsl, compiler, helpers):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(compile_cache, "_installed", False)
    monkeypatch.setenv("VLLM_CUTE_DSL_COMPILE_CACHE", "1")
    monkeypatch.setenv("CUTE_DSL_CACHE_DIR", str(tmp_path / "cute_dsl"))
    return BaseDSL, helpers, compiler


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_CUTE_DSL_COMPILE_CACHE", raising=False)
    monkeypatch.delenv("CUTE_DSL_CACHE_DIR", raising=False)
    assert not compile_cache.maybe_enable_cute_dsl_compile_cache()
    assert "CUTE_DSL_CACHE_DIR" not in os.environ


def test_missing_cute_dsl_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CUTE_DSL_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.delenv("CUTE_DSL_CACHE_DIR", raising=False)
    # A None entry makes the import fail, as when CuTe DSL is not installed.
    monkeypatch.setitem(sys.modules, "cutlass.base_dsl.dsl", None)
    assert not compile_cache.maybe_enable_cute_dsl_compile_cache()
    assert "CUTE_DSL_CACHE_DIR" not in os.environ


def test_cache_directory_that_cannot_exist_is_rejected(tmp_path):
    """CuTe DSL would fail every compile creating it, so the cache stays off."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    assert not compile_cache._ensure_dir(str(blocker / "cute_dsl"))
    assert compile_cache._ensure_dir(str(tmp_path / "cute_dsl"))
    os.chmod(tmp_path, 0o555)
    try:
        # An existing read-only directory still serves hits.
        assert compile_cache._ensure_dir(str(tmp_path / "cute_dsl"))
    finally:
        os.chmod(tmp_path, 0o755)


@pytest.mark.parametrize(
    "opt_out", [None, "no_cache", "keep_ptx", "keep_cubin", "keep_sass"]
)
@pytest.mark.parametrize("compile_only", [False, True])
def test_compile_only_cache_respects_dsl_opt_out(dsl_stubs, opt_out, compile_only):
    """Only compile-only builds without an explicit DSL opt-out become cached."""
    BaseDSL, _, _ = dsl_stubs
    # Set the setup at the dependency boundary, as prepare() would return it.
    BaseDSL._prepare_compilation = lambda self, funcBody: BaseDSL._CompilationSetup(
        compile_only=compile_only
    )
    assert compile_cache.maybe_enable_cute_dsl_compile_cache()
    dsl = BaseDSL()
    dsl.envar = SimpleNamespace(**({opt_out: True} if opt_out else {}))
    setup = dsl._prepare_compilation(None)
    assert setup.no_cache == (not compile_only or opt_out is not None)


@pytest.mark.parametrize("keep", [None, "KeepPTX", "KeepCUBIN", "KeepSASS"])
def test_compile_options_requesting_artifacts_stay_uncached(dsl_stubs, keep):
    BaseDSL, _, compiler = dsl_stubs
    prepare = BaseDSL._prepare_compilation

    def prepare_with_options(self, funcBody):
        setup = prepare(self, funcBody)
        if keep:
            self.compile_options.options[getattr(compiler, keep)].value = True
        return setup

    BaseDSL._prepare_compilation = prepare_with_options
    assert compile_cache.maybe_enable_cute_dsl_compile_cache()
    assert BaseDSL()._prepare_compilation(None).no_cache == (keep is not None)


@pytest.mark.parametrize("missing", ["KeepPTX", "KeepCUBIN", "KeepSASS"])
def test_missing_keep_options_leave_compiles_uncached(dsl_stubs, missing, caplog):
    BaseDSL, helpers, compiler = dsl_stubs
    del compiler.CompileOptions().options[getattr(compiler, missing)]
    prepare, load = BaseDSL._prepare_compilation, helpers.load_ir
    assert not compile_cache.maybe_enable_cute_dsl_compile_cache()
    assert not compile_cache._installed
    assert BaseDSL._prepare_compilation is prepare
    assert helpers.load_ir is load
    assert "not cached" in caplog.text


@pytest.mark.parametrize("mode", ["text", "bytecode", "reader"])
def test_file_loads_disable_threading(dsl_stubs, tmp_path, mode):
    _, helpers, _ = dsl_stubs
    original = helpers.load_ir
    assert compile_cache.maybe_enable_cute_dsl_compile_cache()
    installed = helpers.load_ir
    assert installed is not original
    assert compile_cache.maybe_enable_cute_dsl_compile_cache()
    assert helpers.load_ir is installed
    path = tmp_path / "cute_dsl_kernel.mlir"
    path.write_bytes(b"module")

    def reader(f):
        assert f.mode == "rb"
        return helpers.ir.Module.parse(f.read())

    name, module = helpers.load_ir(
        str(path),
        asBytecode=mode != "text",
        bytecode_reader=reader if mode == "reader" else None,
    )
    assert name == "kernel"
    assert module == ("module" if mode == "text" else b"module")


@pytest.mark.parametrize("hook", ["missing_load", "changed_load", "missing_prepare"])
def test_unsupported_hooks_leave_compiles_uncached(
    dsl_stubs, monkeypatch, hook, caplog
):
    BaseDSL, helpers, _ = dsl_stubs
    if hook == "missing_load":
        monkeypatch.delattr(helpers, "load_ir")
    elif hook == "changed_load":
        monkeypatch.setattr(helpers, "load_ir", lambda file: None)
    else:
        monkeypatch.delattr(BaseDSL, "_prepare_compilation")
    prepare = getattr(BaseDSL, "_prepare_compilation", None)
    load = getattr(helpers, "load_ir", None)
    assert not compile_cache.maybe_enable_cute_dsl_compile_cache()
    assert not compile_cache._installed
    assert getattr(BaseDSL, "_prepare_compilation", None) is prepare
    assert getattr(helpers, "load_ir", None) is load
    assert "not cached" in caplog.text


try:
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_driver
except ImportError:
    cute = None

if cute is not None:

    @cute.kernel
    def _double_kernel(x: cute.Tensor, y: cute.Tensor, n: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        if i < n:
            y[i] = x[i] + x[i]

    class _Double:
        def __init__(self, block: int) -> None:
            self.block = block

        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            y: cute.Tensor,
            n: cutlass.Int32,
            stream: cuda_driver.CUstream,
        ):
            _double_kernel(x, y, n).launch(
                grid=[(n + self.block - 1) // self.block, 1, 1],
                block=[self.block, 1, 1],
                stream=stream,
            )


def _install_compile_counters() -> dict[str, int]:
    """Observe tracing, file loads and full pipeline runs in a test child."""
    from cutlass.base_dsl.compile_backend import CompilerBackend
    from cutlass.base_dsl.dsl import BaseDSL

    stats = {"traces": 0, "file_hits": 0, "misses": 0, "backend_compiles": 0}
    prepare = BaseDSL._prepare_compilation
    resolve = CompilerBackend._resolve_compile_cache
    compile_module = BaseDSL._compile_or_jit_module

    def prepare_counted(self, funcBody, *args, **kwargs):
        setup = prepare(self, funcBody, *args, **kwargs)
        if setup.compile_only:
            stats["traces"] += 1
        return setup

    def resolve_counted(self, ctx):
        state = resolve(self, ctx)
        if state.cache_hit and state.load_from_file_cache:
            stats["file_hits"] += 1
        elif not state.cache_hit:
            stats["misses"] += 1
        return state

    def compile_module_counted(self, *args, **kwargs):
        # File-cache loads also enter CompilerBackend.compile; only this
        # miss path runs the full MLIR compilation pipeline.
        stats["backend_compiles"] += 1
        return compile_module(self, *args, **kwargs)

    # Private hooks belong to this test subprocess, not the production cache.
    BaseDSL._prepare_compilation = prepare_counted
    CompilerBackend._resolve_compile_cache = resolve_counted
    BaseDSL._compile_or_jit_module = compile_module_counted
    return stats


def _compile_case() -> None:
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

    compile_cache.maybe_enable_cute_dsl_compile_cache()
    stats = _install_compile_counters()
    n = cute.sym_int()
    x, y = (make_fake_compact_tensor(cutlass.Float32, (n,)) for _ in "xy")
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = _Double(128)
    compiled = cute.compile(kernel, x, y, 1, stream, options="--enable-tvm-ffi")
    x = torch.arange(1000, dtype=torch.float32, device="cuda")
    y = torch.zeros_like(x)
    compiled(x, y, x.numel())
    torch.accelerator.synchronize()
    tasks = Path("/proc/self/task")
    llvm_workers = None
    if tasks.is_dir():
        llvm_workers = 0
        for comm in tasks.glob("*/comm"):
            # A thread may exit while we enumerate it.
            with contextlib.suppress(FileNotFoundError):
                llvm_workers += comm.read_text().startswith("llvm-worker")
    print(
        json.dumps(
            {
                "correct": bool(torch.equal(y, x * 2)),
                "llvm_workers": llvm_workers,
                **stats,
            }
        )
    )


def _run_compile(cache_root: Path) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tests.utils_.test_cute_dsl_compile_cache import _compile_case; "
                "_compile_case()"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "VLLM_CUTE_DSL_COMPILE_CACHE": "1",
            "VLLM_CACHE_ROOT": str(cache_root),
            "CUTE_DSL_CACHE_DIR": str(cache_root / "cute_dsl"),
        },
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(
    cute is None or not torch.cuda.is_available(),
    reason="needs CuTe DSL and a CUDA GPU",
)
def test_compile_cache_alone_skips_compilation_not_tracing(tmp_path):
    first = _run_compile(tmp_path)
    assert first["correct"] and first["misses"] >= 1
    assert first["backend_compiles"] >= 1
    second = _run_compile(tmp_path)
    assert second["correct"] and second["traces"] == first["traces"]
    assert second["traces"] >= 1
    assert second["file_hits"] >= 1 and second["misses"] == 0
    assert second["backend_compiles"] == 0
    if second["llvm_workers"] is not None:
        assert second["llvm_workers"] == 0
