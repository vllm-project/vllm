# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative native ROCm source attribution, without requiring a GPU."""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "ci"
spec = importlib.util.spec_from_file_location(
    "rocm_kernel_symbol_map", TOOLS / "rocm_kernel_symbol_map.py"
)
assert spec is not None and spec.loader is not None
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)

SYMBOL = "_Z14selector_probeIiEvPT_"
ELF_HEADER = "ELF Header:\n  Type: REL (Relocatable file)\nSection Headers:\n"
ELF = f"""
Symbol table '.dynsym' contains 3 entries:
  Num: Value Size Type Bind Vis Ndx Name
  1: 00001900 32 FUNC GLOBAL PROTECTED 7 {SYMBOL}
  2: 000007c0 64 OBJECT GLOBAL PROTECTED 6 {SYMBOL}.kd
AMDGPU Metadata:
amdhsa.kernels:
  - .name: {SYMBOL}
    .symbol: {SYMBOL}.kd
"""


def test_descriptor_suffix_matches_mangled_dispatch_name():
    assert producer.parse_kernel_symbols(ELF) == [SYMBOL]


@pytest.mark.parametrize(
    "text",
    [
        ELF.replace(f"7 {SYMBOL}", f"UND {SYMBOL}"),
        ELF.replace(f".symbol: {SYMBOL}.kd", ".symbol: missing.kd"),
        "Symbol table '.dynsym' contains 0 entries:",
        "AMDGPU Metadata:\namdhsa.kernels:\n  - .unknown: value",
    ],
)
def test_unreadable_device_symbols_are_unknown_not_empty(text):
    with pytest.raises(ValueError):
        producer.parse_kernel_symbols(text)


@pytest.mark.parametrize(
    "text",
    [
        "probe.o: #deps 1, deps mtime 12 (STALE)\n    source.hip\n",
        "probe.o: #deps 2, deps mtime 12 (VALID)\n    source.hip\n",
        "unrecognized output\n",
    ],
)
def test_incomplete_or_stale_dependencies_cannot_publish(text):
    with pytest.raises(ValueError):
        producer.parse_deps(text)


@pytest.fixture
def compilation(tmp_path, monkeypatch):
    source = tmp_path / "source"
    build = tmp_path / "build"
    (source / "csrc").mkdir(parents=True)
    (build / "csrc").mkdir(parents=True)
    original = source / "csrc" / "cuda_kernel.cu"
    original.write_text("__global__ void probe() {}\n")
    header = source / "csrc" / "shared.h"
    header.write_text("constexpr int kValue = 1;\n")
    host = source / "csrc" / "host.cpp"
    host.write_text('#include "shared.h"\n')
    generated = build / "csrc" / "hip_kernel.hip"
    generated.write_text(original.read_text())
    copied_header = build / "csrc" / "shared.h"
    copied_header.write_text(header.read_text())
    commands = []
    dependency_records = []
    objects = []
    for name, src, deps in (
        ("device", generated, [generated, copied_header]),
        ("host", host, [host, header]),
    ):
        obj = build / "CMakeFiles" / f"{name}.dir" / f"{name}.o"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b"object")
        objects.append(obj)
        commands.append({"directory": str(build), "file": str(src), "output": str(obj)})
        dependency_records.append(
            f"{obj.relative_to(build)}: #deps {len(deps)}, deps mtime 1 (VALID)\n"
            + "".join(f"    {dep}\n" for dep in deps)
            + "\n"
        )
    (build / "compile_commands.json").write_text(json.dumps(commands))
    (build / ".ninja_deps").touch()
    manifest = tmp_path / "hipify.json"
    manifest.write_text(
        json.dumps({str(generated): str(original), str(copied_header): str(header)})
    )
    deps_output = "".join(dependency_records)
    monkeypatch.setattr(
        producer,
        "run",
        lambda args: "tool version" if "--version" in args else deps_output,
    )
    monkeypatch.setattr(
        producer,
        "device_symbols",
        lambda obj, tools: (
            ([SYMBOL], ["hipv4-amdgcn-amd-amdhsa--gfx942"])
            if obj == objects[0]
            else ([], [])
        ),
    )
    monkeypatch.setattr(producer, "device_dependencies", lambda *args: set())
    return {
        "build_root": build,
        "source_root": source,
        "tools": {"ninja": "ninja"},
        "commit": "same-build",
        "hipify_map": manifest,
    }, objects


def test_hipified_source_and_header_keep_original_paths_and_host_use(compilation):
    options, _ = compilation
    result = producer.build_map(**options)
    assert result["version"] == 2 and result["backend"] == "rocm"
    assert not result["incomplete"], result.get("reason")
    assert result["stats"]["objects"] == 2
    objects = {obj["source"]: obj for obj in result["objects"]}
    native = objects["csrc/cuda_kernel.cu"]
    assert native["device"] and native["symbols"] == [SYMBOL]
    assert native["deps"] == ["csrc/cuda_kernel.cu", "csrc/shared.h"]
    host = objects["csrc/host.cpp"]
    assert not host["device"] and host["symbols"] == []
    assert host["deps"] == ["csrc/host.cpp", "csrc/shared.h"]


def test_missing_hipify_header_provenance_invalidates_every_object(compilation):
    options, _ = compilation
    manifest = options["hipify_map"]
    manifest.write_text(
        json.dumps(
            {
                k: v
                for k, v in json.loads(manifest.read_text()).items()
                if not k.endswith(".h")
            }
        )
    )
    result = producer.build_map(**options)
    assert result["objects"] == []
    assert result["incomplete"]
    assert "has no provenance" in result["reason"]


def test_missing_host_object_cannot_hide_its_header_dependencies(compilation):
    options, objects = compilation
    objects[1].unlink()
    result = producer.build_map(**options)
    assert result["objects"] == []
    assert result["incomplete"]
    assert "missing compiled object" in result["reason"]


def test_host_object_omitted_by_compilation_database_invalidates_map(compilation):
    options, _ = compilation
    database = options["build_root"] / "compile_commands.json"
    database.write_text(json.dumps(json.loads(database.read_text())[:1]))
    result = producer.build_map(**options)
    assert result["objects"] == []
    assert "absent from compilation databases" in result["reason"]


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_object_records_must_agree_on_provenance(compilation, conflicting):
    options, _ = compilation
    database = options["build_root"] / "compile_commands.json"
    commands = json.loads(database.read_text())
    duplicate = dict(commands[0])
    if conflicting:
        duplicate["file"] = commands[1]["file"]
    database.write_text(json.dumps([*commands, duplicate]))
    result = producer.build_map(**options)
    if conflicting:
        assert result["objects"] == []
        assert "conflicting compilation records" in result["reason"]
    else:
        assert not result["incomplete"], result.get("reason")
        assert result["stats"]["objects"] == 2


def test_postbuild_source_edit_invalidates_old_provenance(compilation):
    options, objects = compilation
    source = options["source_root"] / "csrc" / "cuda_kernel.cu"
    later = max(obj.stat().st_mtime_ns for obj in objects) + 1_000_000_000
    os.utime(source, ns=(later, later))
    result = producer.build_map(**options)
    assert result["objects"] == []
    assert "newer than object" in result["reason"]


def test_extraction_failure_invalidates_the_whole_map(compilation, monkeypatch):
    options, _ = compilation

    def unreadable(obj, tools):
        raise ValueError("unrecognized fatbin")

    monkeypatch.setattr(producer, "device_symbols", unreadable)
    result = producer.build_map(**options)
    assert result["objects"] == []
    assert result["incomplete"]
    assert result["reason"] == "unrecognized fatbin"


def test_objcopy_never_rewrites_the_compiled_object(tmp_path, monkeypatch):
    commands = []

    def fake_run(args):
        commands.append(args)
        if "--sections" in args:
            return ELF_HEADER + ".hip_fatbin"
        if "--list" in args:
            return "hipv4-amdgcn-amd-amdhsa--gfx942\nhost-x86_64-unknown-linux-gnu-"
        if "--symbols" in args:
            return ELF
        return ""

    monkeypatch.setattr(producer, "run", fake_run)
    tools = {
        name: name for name in ("llvm-readelf", "llvm-objcopy", "clang-offload-bundler")
    }
    obj = tmp_path / "probe.o"
    obj.write_bytes(b"\x7fELF")
    assert producer.device_symbols(obj, tools)[0] == [SYMBOL]
    objcopy = next(args for args in commands if args[0] == "llvm-objcopy")
    assert objcopy[-1] == os.devnull, "objcopy defaults to modifying its input in place"


@pytest.mark.parametrize("magic", [b"BC\xc0\xde", b"\xde\xc0\x17\x0b"])
def test_bitcode_cannot_be_mistaken_for_host_only_elf(tmp_path, magic):
    obj = tmp_path / "lto.hip.o"
    obj.write_bytes(magic + bytes(64))
    with pytest.raises(ValueError, match="not an ELF object"):
        producer.device_symbols(obj, {})


@pytest.mark.parametrize(
    "sections",
    ["", ELF_HEADER + "__CLANG_OFFLOAD_BUNDLE__hip-amdgcn-amd-amdhsa--gfx942"],
)
def test_unsupported_native_formats_cannot_be_mistaken_for_host_only(
    tmp_path, monkeypatch, sections
):
    obj = tmp_path / "rdc.hip.o"
    obj.write_bytes(b"\x7fELF")
    monkeypatch.setattr(producer, "run", lambda args: sections)
    with pytest.raises(ValueError, match="ELF header/sections|linked-code provenance"):
        producer.device_symbols(obj, {"llvm-readelf": "llvm-readelf"})


def test_automatic_build_manifest_is_used_without_manual_flag(compilation):
    options, _ = compilation
    options["hipify_map"].rename(options["build_root"] / "hipify-source-map.json")
    del options["hipify_map"]
    result = producer.build_map(**options)
    assert not result["incomplete"], result.get("reason")
    assert result["objects"][0]["source"] == "csrc/cuda_kernel.cu"


def test_installed_torch_hipified_headers_remain_external_dependencies(tmp_path):
    header = tmp_path / "torch" / "HIPGuardImpl.h"
    header.parent.mkdir()
    header.write_text("// This file was automatically generated by hipify\n")
    assert producer.source_path(
        header, tmp_path / "vllm", [tmp_path / "build"], {}
    ) == str(header)


def test_patched_vendor_header_keeps_both_original_and_patch_dependencies(compilation):
    options, objects = compilation
    build = options["build_root"]
    source = options["source_root"]
    patched = build / "csrc/shared.h"
    manifest = options["hipify_map"]
    provenance = json.loads(manifest.read_text())
    del provenance[str(patched)]
    manifest.write_text(json.dumps(provenance))
    vendor = build.parent / "torch-header.h"
    patch = source / "torch-fix.patch"
    for path in (vendor, patch):
        path.write_text("input to patched header\n")
        earlier = objects[0].stat().st_mtime_ns - 1000
        os.utime(path, ns=(earlier, earlier))
    (build / "generated-source-deps.json").write_text(
        json.dumps({str(patched): [str(vendor), str(patch)]})
    )
    result = producer.build_map(**options)
    assert not result["incomplete"], result.get("reason")
    native = next(obj for obj in result["objects"] if obj["device"])
    assert str(vendor) in native["deps"]
    assert "torch-fix.patch" in native["deps"]


def test_cyclic_generated_provenance_is_rejected(tmp_path):
    first, second = tmp_path / "first.h", tmp_path / "second.h"
    with pytest.raises(ValueError, match="cyclic"):
        producer.expand_dependencies(first, {first: {second}, second: {first}})


def test_device_depfile_preserves_escaped_file_names():
    assert producer.parse_depfile(
        "kernrec_device_deps: source.hip \\\n"
        "  path\\ with\\ spaces.h hash\\#dollar$$.h\n"
    ) == ["source.hip", "path with spaces.h", "hash#dollar$.h"]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "other: file.h",
        "kernrec_device_deps:",
        "kernrec_device_deps: file.h\nother: more.h",
    ],
)
def test_missing_or_ambiguous_device_depfile_is_rejected(text):
    with pytest.raises(ValueError):
        producer.parse_depfile(text)


def test_compiler_response_files_expand_relative_to_working_directory(tmp_path):
    (tmp_path / "flags.rsp").write_text('@nested.rsp -o "object file.o"')
    (tmp_path / "nested.rsp").write_text('-I"./include dir" -DDEVICE_HEADER=1')
    command = {"directory": str(tmp_path), "arguments": ["clang++", "@flags.rsp"]}
    assert producer.compilation_output(command) == tmp_path / "object file.o"
    assert producer.compilation_arguments(command) == [
        "clang++",
        "-I./include dir",
        "-DDEVICE_HEADER=1",
        "-o",
        "object file.o",
    ]
    (tmp_path / "nested.rsp").write_text("@flags.rsp")
    with pytest.raises(ValueError, match="cyclic"):
        producer.compilation_arguments(command)


def test_response_quotes_follow_compiler_backslash_rules():
    assert producer.response_arguments(r'''"-DNAME=foo\q" '-DOTHER=bar\x' ""''') == [
        "-DNAME=fooq",
        "-DOTHER=barx",
        "",
    ]


def test_compiler_frontend_diagnostics_unescape_shell_metacharacters():
    assert producer.response_arguments(
        r'''"clang" "-D" "HEADER=\"dollar\$name.h\""'''
    ) == ["clang", "-D", 'HEADER="dollar$name.h"']


@pytest.mark.parametrize(
    "error",
    [ValueError("missing device target"), subprocess.TimeoutExpired("clang++", 180)],
)
def test_device_dependency_failure_invalidates_every_object(
    compilation, monkeypatch, error
):
    options, _ = compilation

    def fail(*args):
        raise error

    monkeypatch.setattr(producer, "device_dependencies", fail)
    result = producer.build_map(**options)
    assert result["incomplete"] and result["objects"] == []


@pytest.mark.parametrize("response_files", [False, True])
def test_real_hip_device_headers_reach_all_dependent_kernels(tmp_path, response_files):
    """Host depfiles omit device-only includes even when another TU includes them."""
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    compiler = rocm / "llvm/bin/clang++"
    tools = {
        name: str(rocm / "llvm/bin" / name)
        for name in ("llvm-readelf", "llvm-objcopy", "clang-offload-bundler")
    }
    ninja, cmake = shutil.which("ninja"), shutil.which("cmake")
    if (
        not compiler.is_file()
        or not all(Path(p).is_file() for p in tools.values())
        or not ninja
        or not cmake
    ):
        pytest.skip("requires ROCm's HIP compiler and CMake/Ninja")
    assert ninja is not None and cmake is not None
    tools["ninja"] = ninja
    source, build = tmp_path / "source with spaces", tmp_path / "build"
    source.mkdir()
    (source / "device only.h").write_text("constexpr int device_bias = 2;\n")
    for arch in ("gfx942", "gfx950"):
        (source / f"{arch}.h").write_text(f"constexpr int arch_bias = {arch[3:]};\n")
    (source / "first.hip").write_text("""#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__) && ENABLE_DEVICE_HEADER
#include "device only.h"
#if defined(__gfx942__)
#include "gfx942.h"
#elif defined(__gfx950__)
#include "gfx950.h"
#endif
#endif
__global__ void first(int* output) {
#if defined(__HIP_DEVICE_COMPILE__)
  output[0] = device_bias + arch_bias;
#endif
}
""")
    (source / "second.hip").write_text("""#include <hip/hip_runtime.h>
#include "device only.h"
__global__ void second(int* output) { output[0] = device_bias; }
""")
    (source / "host.cpp").write_text("int host_only() { return 0; }\n")
    (source / "CMakeLists.txt").write_text("""cmake_minimum_required(VERSION 3.26)
project(device_dependencies LANGUAGES CXX HIP)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_library(probe OBJECT first.hip second.hip host.cpp)
target_compile_definitions(probe PRIVATE ENABLE_DEVICE_HEADER=1)
""")
    if response_files:
        cmake_file = source / "CMakeLists.txt"
        cmake_file.write_text(
            cmake_file.read_text().replace(
                "target_compile_definitions(probe PRIVATE ENABLE_DEVICE_HEADER=1)",
                """file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/nested.rsp
  "-DENABLE_DEVICE_HEADER=1")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/flags.rsp "@nested.rsp")
target_compile_options(probe PRIVATE @flags.rsp)""",
            )
        )
    subprocess.run(
        [
            cmake,
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_HIP_COMPILER={compiler}",
            "-DCMAKE_HIP_ARCHITECTURES=gfx942:sramecc+:xnack-;gfx950:sramecc+:xnack-",
            *(["-DCMAKE_NINJA_FORCE_RESPONSE_FILE=ON"] if response_files else []),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [cmake, "--build", str(build)], check=True, capture_output=True, text=True
    )
    objects = {p: p.read_bytes() for p in build.rglob("*.o")}
    result = producer.build_map(build, source, tools)
    assert not result["incomplete"], result.get("reason")
    by_source = {obj["source"]: obj for obj in result["objects"]}
    assert {"device only.h", "gfx942.h", "gfx950.h"} <= set(
        by_source["first.hip"]["deps"]
    )
    assert "device only.h" in by_source["second.hip"]["deps"]
    assert by_source["first.hip"]["symbols"] == ["_Z5firstPi"]
    assert by_source["second.hip"]["symbols"] == ["_Z6secondPi"]
    assert not by_source["host.cpp"]["device"]
    assert all(p.read_bytes() == before for p, before in objects.items())

    later = max(p.stat().st_mtime_ns for p in objects) + 1_000_000_000
    os.utime(source / "gfx942.h", ns=(later, later))
    stale = producer.build_map(build, source, tools)
    assert stale["incomplete"] and stale["objects"] == []
    assert "newer than object" in stale["reason"]


@pytest.mark.parametrize("header", [r"a\b.h", "dollar$name.h"])
def test_real_device_scan_preserves_outputs_and_response_header_choice(
    tmp_path, header
):
    compiler = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin/clang++"
    if not compiler.is_file():
        pytest.skip("requires ROCm's HIP compiler")
    source = tmp_path / "probe.hip"
    source.write_text("""#include <hip/hip_runtime.h>
#ifdef __HIP_DEVICE_COMPILE__
#include HEADER
#endif
__global__ void probe(int* output) {
#ifdef __HIP_DEVICE_COMPILE__
  output[0] = selected_value;
#endif
}
""")
    (tmp_path / "ab.h").write_text("constexpr int selected_value = 7;\n")
    (tmp_path / "dollar$name.h").write_text("constexpr int selected_value = 13;\n")
    (tmp_path / "a").mkdir()
    (tmp_path / "a/b.h").write_text("constexpr int selected_value = 11;\n")
    (tmp_path / "flags.rsp").write_text(f'"-DHEADER=\\"{header}\\""')
    args = [
        str(compiler),
        "--offload-arch=gfx950",
        "-c",
        str(source),
        "@flags.rsp",
        "-o",
        "probe.o",
        "-MD",
        "-MF",
        "probe.d",
        "-MT",
        "probe.o",
        "-MJ",
        "fragment.json",
        "-serialize-diagnostics",
        "diagnostics.dia",
        "-Xclang",
        "-dependency-dot",
        "-Xclang",
        "dependencies.dot",
        "-Xclang",
        "-header-include-file",
        "-Xclang",
        "includes.txt",
        "-Xclang",
        "-diagnostic-log-file",
        "-Xclang",
        "diagnostics.log",
        "-Xclang",
        "-stats-file=stats.json",
    ]
    subprocess.run(args, cwd=tmp_path, check=True, capture_output=True, text=True)
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    expanded = producer.compilation_arguments(
        {"directory": str(tmp_path), "arguments": args}
    )
    dependencies = producer.device_dependencies(
        expanded, tmp_path, source, ["hipv4-amdgcn-amd-amdhsa--gfx950"]
    )
    assert tmp_path / header.replace("\\", "") in dependencies
    assert tmp_path / "a/b.h" not in dependencies
    after = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert after == before
