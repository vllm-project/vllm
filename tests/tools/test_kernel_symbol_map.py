# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conservative native ROCm source attribution, without requiring a GPU."""

import importlib.util
import json
import os
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
        lambda obj, tools: ([SYMBOL], ["hipv4-amdgcn-amd-amdhsa--gfx942"])
        if obj == objects[0]
        else ([], []),
    )
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
