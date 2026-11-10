# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import sysconfig
import threading
from contextlib import suppress
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm.envs as envs
import vllm.utils.cute_dsl_aot_cache as aot
import vllm.utils.cute_dsl_compile_cache as compile_cache
from tests.utils_.test_cute_dsl_compile_cache import dsl_stubs as dsl_stubs
from vllm.platforms import current_platform

_KEY = hashlib.sha256(b"kernel").hexdigest()


# ---------------------------------------------------------------------------
# Identity: what the key depends on
# ---------------------------------------------------------------------------


def _set_soabi(monkeypatch, abi: str | None) -> None:
    get_config_var = sysconfig.get_config_var
    monkeypatch.setattr(
        sysconfig,
        "get_config_var",
        lambda name: abi if name == "SOABI" else get_config_var(name),
    )


def test_host_identity_distinguishes_python_build_abi(monkeypatch):
    """Equal Python versions with different binary ABIs must not share objects."""
    _set_soabi(monkeypatch, "abi-a")
    first = aot._host_identity()
    _set_soabi(monkeypatch, "abi-b")
    assert aot._host_identity() != first


def test_unknown_python_build_abi_is_uncacheable(monkeypatch):
    _set_soabi(monkeypatch, None)
    with pytest.raises(aot._Uncacheable, match="ABI"):
        aot._host_identity()


def _make_package(root: Path, mtime: int) -> Path:
    """A package with real vLLM sources and a compiled-extension-like binary."""
    package = root / "site-packages" / "pkg"
    shutil.copytree(Path(aot.__file__).parent, package / "utils")
    (package / "__init__.py").write_text("")
    (package / "_ir.cpython-312-x86_64-linux-gnu.so").write_bytes(bytes(range(256)))
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "x.cpython-312.pyc").write_bytes(str(root).encode())
    for path in package.rglob("*"):
        os.utime(path, (mtime, mtime))
    return package


def _code_digest(package: Path) -> str:
    return aot._files_digest(aot._package_files("pkg", [str(package)], True))


def test_code_identity_ignores_install_location_and_mtimes(tmp_path):
    """The same installation copied to another mount after a rebuild is a hit."""
    a = _make_package(tmp_path / "mount-a", mtime=1_000_000)
    b = _make_package(tmp_path / "other" / "mount-b", mtime=2_000_000)
    assert _code_digest(a) == _code_digest(b)


@pytest.mark.parametrize(
    "path, content",
    [
        ("_ir.cpython-312-x86_64-linux-gnu.so", bytes(range(255, -1, -1))),
        ("utils/cute_dsl_aot_cache.py", b"# edited\n"),
        ("utils/extra.py", b"X = 1\n"),
    ],
    ids=["rebuilt-binary", "edited-source", "added-source"],
)
def test_code_identity_follows_file_contents(tmp_path, path, content):
    """A same-version rebuild of a binary or a source edit is a miss."""
    package = _make_package(tmp_path, mtime=1_000_000)
    before = _code_digest(package)
    (package / path).write_bytes(content)
    assert _code_digest(package) != before


def test_env_identity_ignores_only_cache_locations():
    environ = {
        "CUTE_DSL_ARCH": "sm_100a",
        "CUTE_DSL_CACHE_DIR": "/mount-a/cute_dsl",
        "CUTE_DSL_DUMP_DIR": "/mount-a/dump",
        "CUTE_DSL_LIBS": "/mount-a/lib/libcute_dsl_runtime.so",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "/mount-a/fa",
        "QUACK_CACHE_DIR": "/mount-a/quack",
        "HOME": "/home/a",
    }
    base = aot._env_identity(environ)

    moved = {
        **environ,
        "CUTE_DSL_CACHE_DIR": "/mount-b/cache",
        "CUTE_DSL_DUMP_DIR": "/mount-b/dump",
        "CUTE_DSL_LIBS": "/mount-b/lib/libcute_dsl_runtime.so",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": "/mount-b/fa",
        "QUACK_CACHE_DIR": "/mount-b/quack",
        "HOME": "/home/b",
    }
    assert aot._env_identity(moved) == base

    for name, value in [
        ("CUTE_DSL_ARCH", "sm_103a"),
        ("CUTE_DSL_COMPILER_OPT", "opt-level=2"),
        ("CUTLASS_ANY_SETTING", "1"),
        ("FA_DISABLE_2CTA", "1"),
        ("FLASH_ATTENTION_NUM_SMS", "148"),
        ("FLASHINFER_ANY_SETTING", "1"),
        # FlashInfer's MegaMoE selects its replay protocol at trace time.
        ("MEGA_USE_NCU", "1"),
        ("NVIDIA_TF32_OVERRIDE", "0"),
        ("QUACK_ARCH", "100"),
    ]:
        assert aot._env_identity({**environ, name: value}) != base, name


@pytest.mark.parametrize(
    "name, value",
    [
        ("VLLM_MOE_EXPERT_LOAD_CONFIG", '{"directory": "/scratch/other/moe-load"}'),
        ("VLLM_HY4_SPARSE_MLA_DECODE_BACKEND", "flashinfer"),
    ],
)
def test_deployment_settings_keep_the_env_identity(name, value):
    """One image serves several deployments: entries must hit in all of them."""
    environ = {"CUTE_DSL_ARCH": "sm_100a"}
    assert aot._env_identity({**environ, name: value}) == aot._env_identity(environ)


def _imports_cutlass(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(name.split(".")[0] == "cutlass" for name in names):
            return True
    return False


def _env_reads(tree: ast.Module) -> set[str]:
    """The variables a module reads by name."""
    names = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "envs"
        ):
            names.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and node.args
            and ast.unparse(node.func) in ("os.getenv", "os.environ.get")
        ):
            names.add(str(getattr(node.args[0], "value", "")))
        elif (
            isinstance(node, ast.Subscript) and ast.unparse(node.value) == "os.environ"
        ):
            names.add(str(getattr(node.slice, "value", "")))
    return names


_HOST_ENV_READS = {
    # Read per forward and passed to the kernel as a runtime argument.
    "models/deepseek_v41/nvidia/specialized_decode/moe/routed_expert_gemm.py": {
        "VLLM_MOE_SKIP_PADDING"
    },
}


def test_modules_with_cute_kernels_do_not_read_vllm_variables():
    """The key does not cover vLLM's variables, so tracing must not read them.

    A setting that changes a kernel has to reach it as an argument or through
    the kernel object, where the key sees it.
    """
    root = Path(aot.__file__).parents[1]
    offenders = {}
    for path in root.rglob("*.py"):
        source = path.read_text(errors="replace")
        if "cutlass" not in source:
            continue
        tree = ast.parse(source)
        if _imports_cutlass(tree):
            reads = {name for name in _env_reads(tree) if name.startswith("VLLM_")}
            relative_path = str(path.relative_to(root))
            reads -= _HOST_ENV_READS.get(relative_path, set())
            if reads:
                offenders[relative_path] = sorted(reads)
    assert not offenders, offenders


def test_cache_settings_are_not_compile_factors(monkeypatch, tmp_path):
    """Turning a cache on or moving it must not change compiled-artifact keys."""
    baseline = envs.compile_factors()
    monkeypatch.setenv("VLLM_CUTE_DSL_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_CUTE_DSL_AOT_CACHE", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    assert envs.compile_factors() == baseline


@pytest.mark.parametrize(
    "options, reason",
    [
        (None, None),
        ("", None),
        ("KeepCUBIN", "the compile option KeepCUBIN"),
        (123, "options of type int"),
    ],
)
def test_empty_options_are_cacheable_but_unsafe_options_still_fall_back(
    monkeypatch, options, reason
):
    class KeepCUBIN:
        pass

    def parse(value):
        assert value == "KeepCUBIN"
        return SimpleNamespace(options={KeepCUBIN: SimpleNamespace(value=True)})

    compiler = ModuleType("cutlass.base_dsl.compiler")
    monkeypatch.setattr(
        compiler, "_parse_compile_options_from_str", parse, raising=False
    )
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    if reason is None:
        aot._check_options(None, options)
    else:
        with pytest.raises(aot._Uncacheable, match=reason):
            aot._check_options(None, options)


def test_whitespace_options_still_go_to_the_dsl_parser(monkeypatch):
    def parse(value):
        assert value == " "
        raise ValueError("parser received whitespace")

    compiler = ModuleType("cutlass.base_dsl.compiler")
    monkeypatch.setattr(
        compiler, "_parse_compile_options_from_str", parse, raising=False
    )
    monkeypatch.setitem(sys.modules, compiler.__name__, compiler)
    with pytest.raises(ValueError, match="parser received whitespace"):
        aot._check_options(None, " ")


def _import_module(monkeypatch, path: Path, source: str, name: str):
    """Import `source` from the file `path` as module `name`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


_KERNEL_SOURCE = """
class Kernel:
    def __init__(self, tile):
        self.tile = tile

    def __call__(self, a, n):
        pass


def make_kernel(tile):
    def kernel(a, n):
        return tile

    return kernel
"""


def _clear_identity_caches() -> None:
    """Forget code identities, as a new process would."""
    aot._module_identity.cache_clear()
    aot._package_digest.cache_clear()


@pytest.fixture
def user_kernels(monkeypatch, tmp_path):
    """Import the out-of-tree module `user_kernels` from `tmp_path / root`,
    forgetting code identities as a new process would."""

    def import_from(root: str, source: str = _KERNEL_SOURCE):
        _clear_identity_caches()
        path = tmp_path / root / "user_kernels.py"
        return _import_module(monkeypatch, path, source, "user_kernels")

    yield import_from
    _clear_identity_caches()


def _describe(value) -> str:
    return aot._Describer()(value)


def test_kernels_outside_vllm_are_keyed_by_source(user_kernels):
    """Out-of-tree kernels: same source anywhere hits, changed source misses."""
    a = _describe(user_kernels("a").Kernel(128))
    b = _describe(user_kernels("b").Kernel(128))
    edited = _KERNEL_SOURCE.replace("pass", "return None")
    c = _describe(user_kernels("c", edited).Kernel(128))
    assert a == b
    assert a != c


def test_closure_and_owner_state_are_part_of_the_key(user_kernels):
    module = user_kernels("src")
    assert _describe(module.Kernel(128)) == _describe(module.Kernel(128))
    assert _describe(module.Kernel(128)) != _describe(module.Kernel(256))
    assert _describe(module.Kernel(128).__call__) != _describe(
        module.Kernel(256).__call__
    )
    assert _describe(module.make_kernel(128)) == _describe(module.make_kernel(128))
    assert _describe(module.make_kernel(128)) != _describe(module.make_kernel(256))


@pytest.mark.parametrize(
    "bodies",
    [
        ("x + 1", "x - 1"),
        ("x + 1", "x + 2"),
        ("left(x)", "right(x)"),
        ("(lambda y: y + 1)(x)", "(lambda y: y + 2)(x)"),
    ],
    ids=["bytecode", "constants", "names", "nested-code"],
)
def test_same_qualname_functions_are_keyed_by_code(user_kernels, bodies):
    source = f"""
def choose(first):
    if first:
        def kernel(x):
            return {bodies[0]}
    else:
        def kernel(x):
            return {bodies[1]}
    return kernel
"""
    module = user_kernels("src", source)
    one, two = module.choose(True), module.choose(False)
    assert one.__qualname__ == two.__qualname__
    assert _describe(one) != _describe(two)
    if bodies[0].startswith("(lambda"):
        script = (
            "import json, sys; sys.path.insert(0, sys.argv[1]); "
            "from user_kernels import choose; "
            "from vllm.utils.cute_dsl_aot_cache import _Describer; "
            "print(json.dumps(_Describer()(choose(True))))"
        )
        for seed in ("1", "2"):
            proc = subprocess.run(
                [sys.executable, "-c", script, str(Path(module.__file__).parent)],
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert proc.returncode == 0, proc.stderr
            assert json.loads(proc.stdout.strip().splitlines()[-1]) == _describe(one)


def test_cache_key_also_describes_stride_width(user_kernels):
    module = user_kernels(
        "src",
        """
class Tensor:
    __cache_key__ = ("same-layout",)
    def __init__(self, stride32):
        self._use_32bit_stride = stride32
""",
    )
    assert _describe(module.Tensor(True)) != _describe(module.Tensor(False))


@pytest.mark.parametrize("base, value", [("IntEnum", "1"), ("StrEnum", "'a'")])
def test_plain_enums_keep_module_identity(tmp_path, monkeypatch, base, value):
    descriptions = []
    for suffix in ("one", "two"):
        module = _import_module(
            monkeypatch,
            tmp_path / f"{suffix}.py",
            f"import enum\nclass Mode(enum.{base}):\n    A = {value}\n",
            f"vllm.cute_aot_enum_test_{suffix}",
        )
        descriptions.append(_describe(module.Mode.A))
    assert descriptions[0] != descriptions[1]


@pytest.mark.parametrize(
    "value",
    [torch.empty(4), object(), [None, {"x": torch.zeros(1)}]],
    ids=["torch-tensor", "opaque-object", "nested-tensor"],
)
def test_values_that_cannot_be_described_exactly_are_uncacheable(value):
    with pytest.raises(aot._Uncacheable):
        _describe(value)


def test_code_without_a_source_file_is_uncacheable():
    with pytest.raises(aot._Uncacheable):
        aot._module_identity("__main__")


_NAMESPACE_SOURCE = """
import enum


def kernel(x):
    return x + {addend}


class Kernel:
    def __call__(self, x):
        return x + {addend}


class Mode(enum.Enum):
    A = {addend}
"""


def test_module_namespace_is_part_of_the_key(tmp_path, monkeypatch):
    """Same qualnames in different modules of one hashed package differ."""
    described = []
    for suffix, addend in (("one", 1), ("two", 1)):
        module = _import_module(
            monkeypatch,
            tmp_path / f"{suffix}.py",
            _NAMESPACE_SOURCE.format(addend=addend),
            f"vllm.cute_aot_namespace_test_{suffix}",
        )
        described.append(
            [_describe(v) for v in (module.kernel, module.Kernel(), module.Mode.A)]
        )
    for one, two in zip(*described):
        assert one != two


_PACKAGE_PROBE = """
import json, sys
sys.path.insert(0, sys.argv[1])
from vllm.utils.cute_dsl_aot_cache import _Describer
from example_kernels.kernel import kernel
print(json.dumps(_Describer()(kernel)))
"""


def _describe_installed_kernel(prefix: Path, addend: int) -> str:
    """Install a package whose kernel calls a helper; describe it in a fresh
    process, as a restarted engine would."""
    package = prefix / "example_kernels"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "kernel.py").write_text(
        "from .helper import add\n\n\ndef kernel(x):\n    return add(x)\n"
    )
    (package / "helper.py").write_text(f"def add(x):\n    return x + {addend}\n")
    proc = subprocess.run(
        [sys.executable, "-c", _PACKAGE_PROBE, str(prefix)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


def test_external_kernels_are_keyed_by_their_whole_package(tmp_path):
    """A relocated identical package hits; an edited helper module misses."""
    original = _describe_installed_kernel(tmp_path / "a", addend=1)
    relocated = _describe_installed_kernel(tmp_path / "elsewhere" / "b", addend=1)
    edited_helper = _describe_installed_kernel(tmp_path / "c", addend=2)
    assert original == relocated
    assert original != edited_helper


# ---------------------------------------------------------------------------
# Entries: publication, validation and storage failures
# ---------------------------------------------------------------------------


def test_entry_round_trip(tmp_path):
    assert aot._read_entry(str(tmp_path), _KEY) is None
    assert aot._write_entry(str(tmp_path), _KEY, b"meta", b"object")
    assert aot._read_entry(str(tmp_path), _KEY) == (b"meta", b"object")
    assert [p.name for p in tmp_path.iterdir()] == [_KEY + aot._SUFFIX]


def _entry(tmp_path: Path) -> Path:
    aot._write_entry(str(tmp_path), _KEY, b"meta", b"object" * 100)
    return tmp_path / (_KEY + aot._SUFFIX)


def _flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))


def _truncate(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-7])


def _other_key(path: Path) -> None:
    other = hashlib.sha256(b"other kernel").hexdigest()
    path.write_bytes(aot._pack_entry(other, b"meta", b"object" * 100))


def _old_format(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(aot._MAGIC) : len(aot._MAGIC) + 4] = (0).to_bytes(4, "little")
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    "damage",
    [_flip_last_byte, _truncate, _other_key, _old_format],
    ids=["corrupt", "truncated", "other-key", "old-format"],
)
def test_damaged_entries_are_rejected_and_replaced(tmp_path, damage):
    damage(_entry(tmp_path))
    with pytest.raises(aot._InvalidEntry):
        aot._read_entry(str(tmp_path), _KEY)
    assert aot._write_entry(str(tmp_path), _KEY, b"meta", b"fresh")
    assert aot._read_entry(str(tmp_path), _KEY) == (b"meta", b"fresh")


def test_invalid_lengths_are_rejected_before_reading_body(tmp_path, monkeypatch):
    path = tmp_path / (_KEY + aot._SUFFIX)
    path.write_bytes(
        aot._HEADER.pack(
            aot._MAGIC,
            aot._FORMAT_VERSION,
            bytes.fromhex(_KEY),
            1 << 60,
            1 << 60,
            b"0" * 32,
        )
        + b"body"
    )

    class HeaderOnly(io.BufferedReader):
        def read(self, size=-1):
            assert size == aot._HEADER.size, "body read before validating the header"
            return super().read(size)

    monkeypatch.setattr(
        aot, "open", lambda *args: HeaderOnly(io.FileIO(path, "r")), raising=False
    )
    with pytest.raises(aot._InvalidEntry, match="truncated or oversized"):
        aot._read_entry(str(tmp_path), _KEY)


def test_entry_memory_error_is_an_invalid_entry(tmp_path, monkeypatch):
    path = _entry(tmp_path)

    class OutOfMemory(io.BufferedReader):
        def read(self, size=-1):
            if size == aot._HEADER.size:
                return super().read(size)
            raise MemoryError("entry allocation")

    monkeypatch.setattr(
        aot, "open", lambda *args: OutOfMemory(io.FileIO(path, "r")), raising=False
    )
    with pytest.raises(aot._InvalidEntry, match="entry allocation"):
        aot._read_entry(str(tmp_path), _KEY)


def test_previous_layout_and_interrupted_writes_are_misses(tmp_path):
    """Files from the old .o/.meta layout and a writer that died mid-write."""
    (tmp_path / f"{_KEY}.o").write_bytes(b"object")
    (tmp_path / f"{_KEY}.meta").write_bytes(b"meta")
    (tmp_path / f".{_KEY[:16]}.dead.tmp").write_bytes(b"partial")
    assert aot._read_entry(str(tmp_path), _KEY) is None
    assert aot._write_entry(str(tmp_path), _KEY, b"meta", b"object")
    assert aot._read_entry(str(tmp_path), _KEY) == (b"meta", b"object")


def test_world_writable_entries_are_not_trusted(tmp_path):
    os.chmod(_entry(tmp_path), 0o666)
    with pytest.raises(aot._InvalidEntry, match="world-writable"):
        aot._read_entry(str(tmp_path), _KEY)


def test_entry_that_failed_to_load_is_repaired_once(tmp_path):
    """A checksum-valid entry that cannot be loaded must not poison every later
    start, and its repair must not overwrite another writer's newer entry."""
    directory = str(tmp_path)
    aot._write_entry(directory, _KEY, b"meta", b"unloadable")
    failed = aot._read_entry(directory, _KEY)
    # Without naming the failed entry, a valid one is kept.
    assert not aot._write_entry(directory, _KEY, b"meta", b"fresh")
    assert aot._read_entry(directory, _KEY) == failed
    assert aot._write_entry(directory, _KEY, b"meta", b"fresh", replace=failed)
    assert aot._read_entry(directory, _KEY) == (b"meta", b"fresh")
    # Another process that saw the same failure keeps the repaired entry.
    assert not aot._write_entry(directory, _KEY, b"meta", b"late", replace=failed)
    assert aot._read_entry(directory, _KEY) == (b"meta", b"fresh")
    assert list(tmp_path.glob(".*.tmp")) == []


def test_concurrent_writers_with_one_pid_publish_one_complete_entry(tmp_path):
    """Threads share a PID, as processes in separate containers can."""
    directory = str(tmp_path)
    payloads = [bytes([i]) * (1 << 20) for i in range(8)]
    start = threading.Barrier(len(payloads) + 1)
    writers_done = threading.Event()
    published: dict[int, bool] = {}
    seen: list[bytes] = []
    errors: list[BaseException] = []

    def write(i: int) -> None:
        start.wait()
        published[i] = aot._write_entry(directory, _KEY, b"meta", payloads[i])

    def read() -> None:
        start.wait()
        try:
            while not writers_done.is_set():
                entry = aot._read_entry(directory, _KEY)
                if entry is not None:
                    seen.append(entry[1])
        except BaseException as e:
            errors.append(e)

    writers = [threading.Thread(target=write, args=(i,)) for i in range(len(payloads))]
    reader = threading.Thread(target=read)
    for thread in [reader, *writers]:
        thread.start()
    for thread in writers:
        thread.join(timeout=60)
        assert not thread.is_alive()
    writers_done.set()
    reader.join(timeout=60)

    assert errors == []
    winners = [i for i, won in published.items() if won]
    assert len(published) == len(payloads) and len(winners) == 1
    winner = payloads[winners[0]]
    assert aot._read_entry(directory, _KEY) == (b"meta", winner)
    assert all(payload == winner for payload in seen)
    assert list(tmp_path.glob(".*.tmp")) == []


def test_read_only_directory_serves_hits(tmp_path):
    _entry(tmp_path)
    os.chmod(tmp_path, 0o555)
    try:
        assert aot._read_entry(str(tmp_path), _KEY) == (b"meta", b"object" * 100)
        assert aot._read_entry(str(tmp_path), hashlib.sha256(b"x").hexdigest()) is None
    finally:
        os.chmod(tmp_path, 0o755)


def test_uncreatable_directory_is_a_miss_not_an_error(tmp_path):
    """A cache root that cannot exist (here: below a file) must not fail."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    directory = str(blocker / "cute_dsl_aot")
    assert aot._prepare_cache_dir(directory) is None
    assert aot._read_entry(directory, _KEY) is None
    with pytest.raises(OSError):
        # The compile hook catches this and returns the compiled kernel.
        aot._write_entry(directory, _KEY, b"meta", b"object")


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_CUTE_DSL_AOT_CACHE", raising=False)
    assert not aot.maybe_enable_cute_dsl_aot_cache()


def test_missing_cute_dsl_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CUTE_DSL_AOT_CACHE", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    # A None entry makes the import fail, as when CuTe DSL is not installed.
    monkeypatch.setitem(sys.modules, "cutlass.base_dsl.compiler", None)
    assert not aot.maybe_enable_cute_dsl_aot_cache()
    assert not (tmp_path / "cute_dsl_aot").exists()


# ---------------------------------------------------------------------------
@pytest.fixture
def aot_runtime(dsl_stubs, monkeypatch, tmp_path):
    """CPU substitutes for the external compiler, exporter and object loader."""
    _, _, compiler = dsl_stubs

    class Compiled:
        def dump_to_object(self, prefix):
            return b"object"

    class TVMCompiled(Compiled):
        pass

    class TVMCompiledWithKwargs(TVMCompiled):
        pass

    class CompileCallable:
        def _compile(self, func, *args, **kwargs):
            return func(*args, **kwargs)

    modules = {
        name: ModuleType(name)
        for name in (
            "cutlass",
            "cutlass.cute",
            "cutlass.base_dsl.jit_executor",
            "cutlass.cutlass_dsl.tvm_ffi_provider",
            "tvm_ffi",
            "tvm_ffi.utils",
        )
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        modules["cutlass"], "cute", modules["cutlass.cute"], raising=False
    )
    monkeypatch.setattr(
        modules["cutlass.cute"],
        "runtime",
        SimpleNamespace(
            load_module=lambda *args, **kwargs: {
                f"vllm_aot_{_KEY[:24]}": lambda: "loaded"
            }
        ),
        raising=False,
    )
    monkeypatch.setattr(
        modules["cutlass.base_dsl.jit_executor"],
        "JitCompiledFunction",
        Compiled,
        raising=False,
    )
    provider = modules["cutlass.cutlass_dsl.tvm_ffi_provider"]
    monkeypatch.setattr(
        provider, "TVMFFIJitCompiledFunction", TVMCompiled, raising=False
    )
    monkeypatch.setattr(
        provider,
        "TVMFFIJitCompiledFunctionWithKwargs",
        TVMCompiledWithKwargs,
        raising=False,
    )
    monkeypatch.setattr(modules["tvm_ffi.utils"], "kwargs_wrapper", None, raising=False)
    monkeypatch.setattr(compiler, "CompileCallable", CompileCallable, raising=False)
    monkeypatch.setattr(
        compiler, "_parse_compile_options_from_str", lambda s: None, raising=False
    )
    monkeypatch.setattr(aot, "_installed", False)
    monkeypatch.setattr(aot, "_cache_dir", None)
    monkeypatch.setattr(aot, "_writable", False)
    # Isolate export/load behavior from GPU-specific key inputs.
    monkeypatch.setattr(aot, "_cache_key", lambda *args, **kwargs: _KEY)
    monkeypatch.setenv("VLLM_CUTE_DSL_AOT_CACHE", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    return CompileCallable, Compiled


@pytest.mark.parametrize("mutation", ["inplace", "replaced", "deleted"])
def test_hits_restore_owner_changes_from_tracing(aot_runtime, mutation):
    CompileCallable, Compiled = aot_runtime

    class Kernel:
        def __init__(self):
            self.items = [1]
            self.config = {"tile": 128}
            self.obsolete = True

        def __call__(self):
            if mutation == "inplace":
                self.items.append(2)
                self.config["tile"] = 256
            elif mutation == "replaced":
                self.items = [1, 2]
                self.config = {"tile": 256}
            else:
                del self.obsolete
            return Compiled()

    assert aot.maybe_enable_cute_dsl_aot_cache()
    cold, warm = Kernel(), Kernel()
    CompileCallable()._compile(cold)
    loaded = CompileCallable()._compile(warm)
    assert isinstance(loaded, aot._LoadedKernel)
    assert warm.__dict__ == cold.__dict__


def test_restored_attributes_preserve_kernel_name(aot_runtime):
    CompileCallable, Compiled = aot_runtime

    class Kernel:
        def __call__(self):
            self.tile = 128
            self.layout = lambda: None
            return Compiled()

    assert aot.maybe_enable_cute_dsl_aot_cache()
    CompileCallable()._compile(Kernel())
    warm = Kernel()
    loaded = CompileCallable()._compile(warm)
    assert loaded._name == aot._name(warm)
    assert warm.tile == 128
    assert isinstance(warm.layout, aot._TraceOnly)


@pytest.fixture
def loaded_functions(aot_runtime, monkeypatch):
    """Compiled and loaded functions whose calls, `to()` and `library` share
    one CUDA library, as CuTe DSL's do."""
    CompileCallable, _ = aot_runtime
    compiles: list[None] = []

    class Library:
        carveout = None

    class Function:
        def __init__(self, origin):
            self.origin, self._library = origin, Library()

        def __call__(self):
            return self.origin, self._library.carveout

        def dump_to_object(self, prefix):
            return b"object"

        def to(self, device=None):
            return self

        @property
        def library(self):
            return self._library

    class Kernel:
        def __call__(self):
            compiles.append(None)
            return Function("compiled")

    monkeypatch.setattr(
        sys.modules["cutlass.base_dsl.jit_executor"], "JitCompiledFunction", Function
    )
    monkeypatch.setattr(
        sys.modules["cutlass.cute"],
        "runtime",
        SimpleNamespace(
            load_module=lambda *args, **kwargs: {
                f"vllm_aot_{_KEY[:24]}": Function("loaded")
            }
        ),
    )
    assert aot.maybe_enable_cute_dsl_aot_cache()
    CompileCallable()._compile(Kernel())
    loaded = CompileCallable()._compile(Kernel())
    assert isinstance(loaded, aot._LoadedKernel)
    compiles.clear()
    return loaded, compiles


def test_a_hit_configures_the_kernel_it_runs(loaded_functions):
    """Kimi K3's prepare_shared_carveout sets a driver attribute on the kernel
    in `compiled.library`, then launches through `compiled(...)`."""
    loaded, compiles = loaded_functions
    deferred = aot.get_aot_cache_stats().get("deferred_compiles", 0)
    executor = loaded.to(0)
    loaded.library.carveout = 100
    assert loaded() == executor() == ("loaded", 100)
    assert compiles == []
    assert aot.get_aot_cache_stats().get("deferred_compiles", 0) == deferred


def test_after_a_deferred_compile_calls_run_the_compiled_kernel(loaded_functions):
    loaded, compiles = loaded_functions
    loaded.dump_to_object("export")
    assert len(compiles) == 1
    loaded.library.carveout = 100
    assert loaded() == loaded.to(0)() == ("compiled", 100)
    assert len(compiles) == 1


def test_aot_export_compiles_bypass_only_the_compile_cache_miss(
    aot_runtime, dsl_stubs, monkeypatch
):
    CompileCallable, Compiled = aot_runtime
    BaseDSL, _, _ = dsl_stubs
    uncached = []

    class Kernel:
        def __call__(self):
            uncached.append(BaseDSL()._prepare_compilation(None).no_cache)
            other_thread = []
            thread = threading.Thread(
                target=lambda: other_thread.append(
                    BaseDSL()._prepare_compilation(None).no_cache
                )
            )
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive()
            assert other_thread == [False]
            return Compiled()

    assert compile_cache.maybe_enable_cute_dsl_compile_cache()
    assert aot.maybe_enable_cute_dsl_aot_cache()
    CompileCallable()._compile(Kernel())
    assert uncached == [True], "an AOT export must compile its own FFI wrapper"
    loaded = CompileCallable()._compile(Kernel())
    loaded.dump_to_object("export")
    assert uncached == [True, False], "deferred compiles may use the compile cache"
    assert not BaseDSL()._prepare_compilation(None).no_cache

    def uncacheable(*args, **kwargs):
        raise aot._Uncacheable("test argument")

    monkeypatch.setattr(aot, "_cache_key", uncacheable)
    CompileCallable()._compile(Kernel())
    assert uncached == [True, False, False]

    class BrokenKernel:
        def __call__(self):
            raise RuntimeError("compile failed")

    monkeypatch.setattr(aot, "_cache_key", lambda *args, **kwargs: "0" * 64)
    with pytest.raises(RuntimeError, match="compile failed"):
        CompileCallable()._compile(BrokenKernel())
    assert not BaseDSL()._prepare_compilation(None).no_cache


def test_serving_checkpoint_uses_project_namespace():
    assert _MODEL == "vllm-project/tiny-random-Gemma2ForCausalLM"
    assert _REVISION == "7551e53abee9dd16896a86847cd137cbfe41764f"


# ---------------------------------------------------------------------------
# Real CuTe DSL compile, export and load (needs CuTe DSL and a CUDA GPU). Each
# run is a fresh process, so the hooks do not leak into other tests and a hit
# comes from disk.
# ---------------------------------------------------------------------------

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
        """A kernel object with a keyword-only parameter whose __call__ sets
        state while tracing."""

        def __init__(self, block: int) -> None:
            self.block = block

        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            y: cute.Tensor,
            n: cutlass.Int32,
            stream: cuda_driver.CUstream,
            *,
            unroll: cutlass.Constexpr = 1,
        ):
            self.traced_block = self.block
            _double_kernel(x, y, n).launch(
                grid=[(n + self.block - 1) // self.block, 1, 1],
                block=[self.block, 1, 1],
                stream=stream,
            )


def _run(case: str, cache_root: Path, aot_cache: bool = True) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from tests.utils_.test_cute_dsl_aot_cache import _case; _case({case!r})",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "VLLM_CACHE_ROOT": str(cache_root),
            "CUTE_DSL_CACHE_DIR": str(cache_root / "cute_dsl"),
            "VLLM_CUTE_DSL_COMPILE_CACHE": "1",
            "VLLM_CUTE_DSL_AOT_CACHE": "1" if aot_cache else "0",
        },
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _case(case: str) -> None:
    import tempfile

    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

    from tests.utils_.test_cute_dsl_compile_cache import _install_compile_counters
    from vllm.utils import cute_dsl_compile_cache, jit_monitor

    cute_dsl_compile_cache.maybe_enable_cute_dsl_compile_cache()
    aot.maybe_enable_cute_dsl_aot_cache()
    dsl_stats = _install_compile_counters()

    def compile_double(kernel):
        n = cute.sym_int()
        x, y = (make_fake_compact_tensor(cutlass.Float32, (n,)) for _ in "xy")
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel, x, y, 1, stream, unroll=1, options="--enable-tvm-ffi"
        )

    def doubles(fn) -> bool:
        x = torch.arange(1000, dtype=torch.float32, device="cuda")
        y = torch.zeros_like(x)
        fn(x, y, x.numel())
        torch.accelerator.synchronize()
        return bool(torch.equal(y, x * 2))

    kernel = _Double(128)
    compiled = compile_double(kernel)
    result = {"correct": doubles(compiled), "traced_block": kernel.traced_block}
    if case == "export":
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "exported.o")
            compiled.export_to_c(path, function_name="exported")
            module = cute.runtime.load_module(path, enable_tvm_ffi=True)
        result["export_correct"] = doubles(module["exported"])
    if case == "monitor":
        jit_monitor.activate(mode="error")
        before = aot.get_aot_cache_stats()
        try:
            compile_double(kernel)
        except RuntimeError:
            result["monitor_raised"] = True
        result["stats_unchanged"] = aot.get_aot_cache_stats() == before
    result.update(aot.get_aot_cache_stats())
    result.update({f"dsl_{name}": value for name, value in dsl_stats.items()})
    print(json.dumps(result))


_needs_cute_dsl = pytest.mark.skipif(
    cute is None or not torch.cuda.is_available(),
    reason="needs CuTe DSL and a CUDA GPU",
)


@_needs_cute_dsl
def test_a_moved_cache_loads_instead_of_tracing(tmp_path):
    first = _run("plain", tmp_path / "a")
    assert first["correct"] and first["misses"] == 1 and first["exports"] == 1
    assert first["dsl_traces"] >= 1

    # A copy at another location, as on another mount or node.
    shutil.copytree(tmp_path / "a", tmp_path / "elsewhere" / "b")
    second = _run("plain", tmp_path / "elsewhere" / "b")
    assert second["correct"] and second["hits"] == 1
    assert second.get("misses", 0) == 0 and second.get("dsl_traces", 0) == 0
    # State that tracing set on the kernel object is restored on a hit.
    assert second["traced_block"] == 128


@_needs_cute_dsl
def test_a_hit_can_still_be_exported(tmp_path):
    """Callers such as FlashInfer's kernel cache export what they compile."""
    _run("plain", tmp_path)
    result = _run("export", tmp_path)
    assert result["hits"] == 1 and result["deferred_compiles"] == 1
    assert result["correct"] and result["export_correct"]


@_needs_cute_dsl
def test_strict_jit_monitor_rejects_compiles_after_warmup(tmp_path):
    """The monitor sees cute.compile calls, not what the cache would serve."""
    _run("plain", tmp_path)
    result = _run("monitor", tmp_path)
    assert result["hits"] == 1
    assert result["monitor_raised"] and result["stats_unchanged"]


# ---------------------------------------------------------------------------
# Serving: the GPU worker installs the caches from the opt-in flags, and a
# restarted engine serves the same requests from the saved entries. FLASH_ATTN
# runs the CuTe DSL FA4 kernels on SM10x, the hardware this is qualified on.
# ---------------------------------------------------------------------------

_MODEL = "vllm-project/tiny-random-Gemma2ForCausalLM"
_REVISION = "7551e53abee9dd16896a86847cd137cbfe41764f"
# As in test_basic_correctness.py: past Gemma2's 4096-token sliding window, so
# both the sliding and the global attention kernels run.
_PROMPT = (
    "The following numbers of the sequence "
    + ", ".join(str(i) for i in range(1024))
    + " are:"
)
_FAILURE_COUNTERS = (
    "uncacheable",
    "invalid_entries",
    "load_failures",
    "export_failures",
    "export_unsupported",
    "deferred_compiles",
)


def _serve(model: str, out: str) -> None:
    from tests.conftest import VllmRunner
    from tests.utils_.test_cute_dsl_compile_cache import _install_compile_counters
    from vllm.utils import cute_dsl_compile_cache

    # Nothing here enables the caches: only the worker can.
    assert not aot._installed and not cute_dsl_compile_cache._installed
    dsl_stats = _install_compile_counters()
    with VllmRunner(
        model,
        trust_remote_code=False,
        max_model_len=8192,
        max_num_seqs=4,
        kv_cache_memory_bytes=1 << 30,
        enforce_eager=False,
        enable_prompt_embeds=False,
        async_scheduling=False,
        distributed_executor_backend="uni",
        attention_config={"backend": "FLASH_ATTN"},
    ) as llm:
        outputs = llm.generate_greedy([_PROMPT], 5)
    result = {
        "installed": [aot._installed, cute_dsl_compile_cache._installed],
        "outputs": outputs,
        "aot": aot.get_aot_cache_stats(),
        "dsl": dsl_stats,
    }
    Path(out).write_text(json.dumps(result))


def _serve_fresh(
    model: str, root: Path, phase: str, *, cache_enabled: bool = True
) -> dict:
    """Serve with private cache paths in a fresh, bounded process group."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in ("CUTE_DSL_CACHE_DIR", "VLLM_FLASH_ATTN_VERSION", "VLLM_BATCH_INVARIANT")
    }
    env.update(
        {
            "VLLM_CUTE_DSL_AOT_CACHE": "1" if cache_enabled else "0",
            "VLLM_CUTE_DSL_COMPILE_CACHE": "1" if cache_enabled else "0",
            # FA4's own disk cache would serve its kernels before ours is asked.
            "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "0",
            # The qualified route: EngineCore and the worker in this process.
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_NO_USAGE_STATS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HOME": str(root / "hf"),
            "HF_HUB_CACHE": str(root / "hf" / "hub"),
            "VLLM_CACHE_ROOT": str(root / "vllm"),
            "VLLM_CONFIG_ROOT": str(root / "config"),
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_CACHE_HOME": str(root / "xdg"),
            "FLASHINFER_WORKSPACE_BASE": str(root / "flashinfer"),
            "FLASHINFER_NO_DOWNLOAD": "1",
            "CUDA_CACHE_PATH": str(root / "cuda"),
            "CUTE_DSL_CACHE_DIR": str(root / "vllm" / "cute_dsl"),
            "TRITON_CACHE_DIR": str(root / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
            "TORCH_EXTENSIONS_DIR": str(root / "torch_extensions"),
            "PYTORCH_KERNEL_CACHE_PATH": str(root / "torch_kernels"),
            "TMPDIR": str(root / "tmp"),
        }
    )
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    out = root / f"{phase}.json"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys; from tests.utils_.test_cute_dsl_aot_cache import _serve; "
                "_serve(*sys.argv[1:])"
            ),
            model,
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=600)
    except BaseException:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        raise
    assert proc.returncode == 0, f"{phase} failed:\n{stderr[-6000:]}"
    return json.loads(out.read_text())


def _entries(root: Path) -> dict[str, str]:
    directory = root / "vllm" / "cute_dsl_aot"
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob(f"*{aot._SUFFIX}"))
    }


def _check_no_failures(phase: str, result: dict) -> None:
    assert result["installed"] == [True, True], f"{phase}: worker hooks not installed"
    failures = {k: result["aot"][k] for k in _FAILURE_COUNTERS if result["aot"].get(k)}
    assert not failures, f"{phase}: {failures}"


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the serving path is qualified on SM10x, where FLASH_ATTN runs FA4",
)
def test_restarted_engine_serves_from_saved_entries(tmp_path):
    """Opt-in flags alone make the worker save CuTe DSL kernels; a fresh engine
    then serves the same requests without tracing or compiling any of them."""
    from tests.models.utils import check_outputs_equal
    from vllm.transformers_utils.repo_utils import hf_api

    # One snapshot pins weights, config and tokenizer for both processes.
    model = hf_api().snapshot_download(
        _MODEL,
        revision=_REVISION,
        cache_dir=os.environ.get("HF_HUB_CACHE", str(tmp_path / "hf" / "hub")),
    )

    reference = _serve_fresh(model, tmp_path, "flags_off", cache_enabled=False)
    assert reference["installed"] == [False, False], reference
    assert not _entries(tmp_path)

    cold = _serve_fresh(model, tmp_path, "cold")
    _check_no_failures("cold", cold)
    saved = _entries(tmp_path)
    # Cold may hit an entry it exported itself; every miss is exported.
    assert cold["aot"].get("misses", 0) > 0, cold
    assert cold["aot"].get("exports", 0) == cold["aot"]["misses"], cold
    assert len(saved) == cold["aot"]["exports"], (len(saved), cold["aot"])
    assert cold["dsl"].get("backend_compiles", 0) > 0, cold

    warm = _serve_fresh(model, tmp_path, "warm")
    _check_no_failures("warm", warm)
    assert warm["aot"].get("hits", 0) > 0, warm
    for counter in ("misses", "exports", "export_existing"):
        assert warm["aot"].get(counter, 0) == 0, (counter, warm["aot"])
    for counter in ("traces", "backend_compiles"):
        assert warm["dsl"].get(counter, 0) == 0, (counter, warm["dsl"])
    assert _entries(tmp_path) == saved

    check_outputs_equal(
        outputs_0_lst=reference["outputs"],
        outputs_1_lst=cold["outputs"],
        name_0="flags_off",
        name_1="cold",
    )
    check_outputs_equal(
        outputs_0_lst=cold["outputs"],
        outputs_1_lst=warm["outputs"],
        name_0="cold",
        name_1="warm",
    )
