#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a ROCm kernel/source map from a completed CMake/Ninja build.

Requires CMAKE_EXPORT_COMPILE_COMMANDS=ON, retained objects and .ninja_deps,
and ROCm's llvm-readelf, llvm-objcopy and clang-offload-bundler. HIPified
sources and copied headers use the generated-to-original manifests written
by the build, or an explicit --hipify-map. Relative paths use --source-root.
Provenance is recorded during generation, never guessed from path names.

Device dependencies are collected by preprocessing every GPU target with
the original compiler's frontend arguments; Ninja only records host includes.
All compilation units are inspected, including host-only units. Missing
objects, dependency information, provenance, or unreadable device code
invalidate the entire map. This is an ahead-of-time map; an AITER JIT cache
without complete source and dependency provenance is not an input.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import regex as re

DEPS_HEADER = re.compile(r"^(.+): #deps (\d+), deps mtime \d+ \((VALID|STALE)\)$")
TARGET = re.compile(r"(?:^|/)CMakeFiles/([^/]+)\.dir/")


def run(command: list[str], *, cwd: Path | None = None, diagnostics=False) -> str:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, timeout=180
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise ValueError(
            f"{Path(command[0]).name} exited {result.returncode}: {detail}"
        )
    return result.stdout + result.stderr if diagnostics else result.stdout


def parse_deps(text: str) -> dict[str, list[str]]:
    """Read only complete, valid Ninja dependency records."""
    records: dict[str, list[str]] = {}
    current: str | None = None
    expected = 0

    def finish():
        if current is not None and len(records[current]) != expected:
            raise ValueError(f"incomplete dependency record: {current}")

    for line in text.splitlines():
        match = DEPS_HEADER.match(line)
        if match:
            finish()
            current, count, status = match.groups()
            if status != "VALID":
                raise ValueError(f"stale dependency record: {current}")
            if current in records:
                raise ValueError(f"duplicate dependency record: {current}")
            expected = int(count)
            records[current] = []
        elif line.startswith("    ") and current is not None:
            records[current].append(line.strip())
        elif not line.strip():
            finish()
            current = None
        else:
            raise ValueError(f"unrecognized Ninja dependency output: {line}")
    finish()
    return records


def parse_kernel_symbols(text: str) -> list[str]:
    """Join AMDGPU metadata descriptors to their defined kernel entry symbols."""
    if "AMDGPU Metadata:" not in text or "amdhsa.kernels:" not in text:
        raise ValueError("no supported AMDGPU kernel metadata")
    functions: set[str] = set()
    descriptors: set[str] = set()
    metadata: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0].rstrip(":").isdigit():
            _, _, _, kind, _, _, section, name = parts[:8]
            if section == "UND":
                continue
            if kind == "FUNC":
                functions.add(name)
            elif kind == "OBJECT" and name.endswith(".kd"):
                descriptors.add(name.removesuffix(".kd"))
        match = re.match(r"^\s+\.symbol:\s+['\"]?([^\s'\"]+)['\"]?\s*$", line)
        if match:
            symbol = match.group(1)
            if not symbol.endswith(".kd"):
                raise ValueError(f"unsupported kernel descriptor: {symbol}")
            metadata.add(symbol.removesuffix(".kd"))
    if descriptors != metadata or not descriptors <= functions:
        raise ValueError("kernel metadata and ELF entry symbols disagree")
    if not metadata and not re.search(r"amdhsa\.kernels:\s*\[\s*\]", text):
        raise ValueError("unrecognized AMDGPU kernel metadata")
    return sorted(metadata)


def device_symbols(obj: Path, tools: dict[str, str]) -> tuple[list[str], list[str]]:
    """Read HIP fatbins in a host object; retain names across every architecture."""
    with obj.open("rb") as stream:
        magic = stream.read(4)
    if magic != b"\x7fELF":
        raise ValueError(f"not an ELF object (bitcode/LTO is unsupported): {obj}")
    sections = run([tools["llvm-readelf"], "--sections", "--file-header", str(obj)])
    if (
        "ELF Header:" not in sections
        or "Section Headers:" not in sections
        or not re.search(r"^\s+Type:\s+REL\b", sections, re.MULTILINE)
    ):
        raise ValueError(f"cannot identify relocatable ELF header/sections: {obj}")
    if any(
        marker in sections
        for marker in (".llvm.offloading", ".llvmbc", "__CLANG_OFFLOAD_BUNDLE__")
    ):
        raise ValueError(
            "LLVM offloading/RDC/LTO objects require linked-code provenance"
        )
    if ".hip_fatbin" not in sections:
        if "AMDGPU" in sections:
            raise ValueError("device-only/RDC objects require linked-code provenance")
        return [], []
    with tempfile.TemporaryDirectory(prefix="rocm-symbol-map-") as temporary:
        fatbin = Path(temporary) / "fatbin"
        run(
            [
                tools["llvm-objcopy"],
                f"--dump-section=.hip_fatbin={fatbin}",
                str(obj),
                os.devnull,
            ]
        )
        targets = run(
            [
                tools["clang-offload-bundler"],
                "--type=o",
                f"--input={fatbin}",
                "--list",
            ]
        ).splitlines()
        device_targets = [t for t in targets if t.startswith("hipv4-amdgcn-")]
        unexpected = [
            t for t in targets if not t.startswith(("host-", "hipv4-amdgcn-"))
        ]
        if not device_targets or unexpected:
            raise ValueError(f"unsupported offload bundles: {targets}")
        symbols: set[str] = set()
        for index, target in enumerate(device_targets):
            code = Path(temporary) / f"device-{index}.hsaco"
            run(
                [
                    tools["clang-offload-bundler"],
                    "--type=o",
                    f"--input={fatbin}",
                    "--unbundle",
                    f"--targets={target}",
                    f"--output={code}",
                ]
            )
            symbols.update(
                parse_kernel_symbols(
                    run([tools["llvm-readelf"], "--symbols", "--notes", str(code)])
                )
            )
        return sorted(symbols), sorted(device_targets)


def absolute(path: str | Path, base: Path) -> Path:
    return (base / path).resolve()


def relative(path: Path, source_root: Path) -> str:
    try:
        return path.relative_to(source_root).as_posix()
    except ValueError:
        return path.as_posix()


def load_provenance(path: Path | None, source_root: Path) -> dict[Path, Path]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError("HIPify manifest must map generated paths to original paths")
    result: dict[Path, Path] = {}
    for generated, original in data.items():
        key, value = absolute(generated, source_root), absolute(original, source_root)
        if key in result and result[key] != value:
            raise ValueError(f"ambiguous HIPify provenance: {key}")
        if not key.is_file() or not value.is_file():
            raise ValueError(
                f"missing file in HIPify provenance: {generated}: {original}"
            )
        if not value.is_relative_to(source_root):
            raise ValueError(f"HIPify original is outside source root: {original}")
        result[key] = value
    return result


def source_path(
    path: Path, source_root: Path, build_dirs: list[Path], provenance: dict[Path, Path]
) -> str:
    if path in provenance:
        return relative(provenance[path], source_root)
    if any(path.is_relative_to(build) for build in build_dirs):
        raise ValueError(f"generated source/header has no provenance: {path}")
    if not path.is_relative_to(source_root):
        return path.as_posix()
    # HIPify can also write alongside its input, outside the build tree.
    if path.suffix in (".hip", ".cu", ".cpp", ".cc", ".c", ".h", ".hpp", ".cuh"):
        with path.open("rb") as stream:
            if b"automatically generated by hipify" in stream.read(256):
                raise ValueError(f"HIPified source/header has no provenance: {path}")
    return relative(path, source_root)


def generated_dependencies(
    build_root: Path, source_root: Path
) -> dict[Path, set[Path]]:
    """Read explicit inputs for build-generated headers, including vendor patches."""
    result: dict[Path, set[Path]] = {}
    for manifest in build_root.rglob("generated-source-deps.json"):
        data = json.loads(manifest.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"invalid generated dependency manifest: {manifest}")
        for output, inputs in data.items():
            if (
                not isinstance(output, str)
                or not isinstance(inputs, list)
                or not inputs
            ):
                raise ValueError(f"invalid generated dependencies: {manifest}")
            if not all(isinstance(path, str) for path in inputs):
                raise ValueError(f"invalid generated dependency paths: {manifest}")
            output = absolute(output, source_root)
            paths = {absolute(path, source_root) for path in inputs}
            if output in result and result[output] != paths:
                raise ValueError(f"conflicting generated dependencies: {output}")
            result[output] = paths
    return result


def expand_dependencies(
    path: Path, generated: dict[Path, set[Path]], active=()
) -> set[Path]:
    if path not in generated:
        return {path}
    if path in active:
        raise ValueError(f"cyclic generated dependencies: {path}")
    return {
        original
        for dependency in generated[path]
        for original in expand_dependencies(dependency, generated, (*active, path))
    }


def response_arguments(text: str) -> list[str]:
    """Clang's GNU response syntax escapes the next character in either quote."""
    arguments, word = [], []
    quote = None
    started = False
    chars = iter(text)
    for char in chars:
        if char == "\\":
            escaped = next(chars, None)
            if escaped is None:
                raise ValueError("unfinished compiler response escape")
            word.append(escaped)
            started = True
        elif quote:
            if char == quote:
                quote = None
            else:
                word.append(char)
        elif char in ("'", '"'):
            quote = char
            started = True
        elif char in " \t\r\n\v\f":
            if started:
                arguments.append("".join(word))
                word, started = [], False
        else:
            word.append(char)
            started = True
    if quote:
        raise ValueError("unfinished compiler response quote")
    if started:
        arguments.append("".join(word))
    return arguments


def compilation_arguments(
    command: dict, response_files: set[Path] | None = None
) -> list[str]:
    args = command.get("arguments") or shlex.split(command.get("command", ""))
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError("invalid compiler arguments")
    directory = Path(command["directory"]).resolve()

    def expand(arguments, active=()):
        expanded = []
        for arg in arguments:
            if not arg.startswith("@"):
                expanded.append(arg)
                continue
            response = absolute(arg[1:], directory)
            if response in active or len(active) >= 16:
                raise ValueError(f"cyclic or deeply nested response file: {response}")
            if response_files is not None:
                response_files.add(response)
            expanded.extend(
                expand(response_arguments(response.read_text()), (*active, response))
            )
        return expanded

    expanded = expand(args)
    if any(arg.startswith("--rsp-quoting") for arg in expanded):
        raise ValueError("explicit compiler response quoting is unsupported")
    return expanded


def compilation_output(command: dict) -> Path:
    directory = Path(command["directory"]).resolve()
    output = command.get("output")
    args = compilation_arguments(command)
    if "-fgpu-rdc" in args or any(a.startswith("-flto") for a in args):
        raise ValueError("RDC/LTO needs linked-code provenance")
    if not output:
        for index, arg in enumerate(args[:-1]):
            if arg == "-o":
                output = args[index + 1]
                break
    if not output:
        raise ValueError(f"compilation has no object output: {command.get('file')}")
    return absolute(output, directory)


def parse_depfile(text: str) -> list[str]:
    """Read Clang's single-rule Make depfile, including escaped file names."""
    text = text.replace("\\\r\n", "").replace("\\\n", "")
    prefix = "kernrec_device_deps:"
    if not text.startswith(prefix):
        raise ValueError("unexpected device dependency target")
    paths, word = [], []
    chars = iter(text[len(prefix) :])
    for char in chars:
        if char == "\\":
            escaped = next(chars, "")
            if escaped not in (" ", "\t", "#", ":", "\\"):
                raise ValueError("unsupported device dependency escape")
            word.append(escaped)
        elif char == "$":
            if next(chars, "") != "$":
                raise ValueError("unsupported device dependency variable")
            word.append("$")
        elif char in "#:;\0":
            raise ValueError("unexpected device dependency rule")
        elif char.isspace():
            if word:
                paths.append("".join(word))
                word = []
        else:
            word.append(char)
    if word:
        paths.append("".join(word))
    if not paths:
        raise ValueError("empty device dependencies")
    return paths


def device_dependencies(
    args: list[str], directory: Path, source: Path, targets: list[str]
) -> set[Path]:
    """Preprocess the actual AMDGPU frontends without writing build outputs."""
    if not targets:
        return set()
    if not args:
        raise ValueError(f"missing compiler command for device source: {source}")
    expected = set()
    for target in targets:
        match = re.fullmatch(
            r"hipv4-amdgcn-amd-amdhsa--(gfx[0-9a-z]+)((?::(?:sramecc|xnack)[+-])*)",
            target,
        )
        if not match:
            raise ValueError(f"unsupported device dependency target: {target}")
        expected.add((match[1], tuple(sorted(filter(None, match[2].split(":"))))))

    def option(frontend, name):
        if name not in frontend or frontend.index(name) + 1 == len(frontend):
            raise ValueError(f"missing device compiler argument: {name}")
        return frontend[frontend.index(name) + 1]

    # Clang creates/removes driver output files even with -###.
    driver = []
    arguments = iter(args)
    for arg in arguments:
        if arg in ("-MJ", "-gen-cdb-fragment-path", "-serialize-diagnostics"):
            if next(arguments, None) is None:
                raise ValueError(f"missing compiler output argument: {arg}")
        elif arg.startswith(
            ("-MJ", "-gen-cdb-fragment-path=", "-serialize-diagnostics=")
        ):
            continue
        else:
            driver.append(arg)
    frontends = {}
    for line in run([*driver, "-###"], cwd=directory, diagnostics=True).splitlines():
        if '"-cc1"' not in line:
            continue
        frontend = response_arguments(line)
        if "-fcuda-is-device" not in frontend:
            continue

        if (
            option(frontend, "-triple") != "amdgcn-amd-amdhsa"
            or "-emit-obj" not in frontend
        ):
            raise ValueError("unsupported device compiler frontend")
        features = []
        for index, arg in enumerate(frontend[:-1]):
            if arg == "-target-feature":
                feature = frontend[index + 1]
                if feature.startswith(("+", "-")) and feature[1:] in (
                    "sramecc",
                    "xnack",
                ):
                    features.append(feature[1:] + feature[0])
        arch = (
            option(frontend, "-target-cpu"),
            tuple(sorted(features)),
        )
        if arch in frontends:
            raise ValueError(f"duplicate device compiler frontend: {arch}")
        frontends[arch] = frontend
    if set(frontends) != expected:
        raise ValueError("compiler device targets do not match object offload targets")

    dependencies = set()
    with tempfile.TemporaryDirectory(prefix="rocm-device-deps-") as temporary:
        for index, frontend in enumerate(frontends.values()):
            command = []
            arguments = iter(frontend)
            for arg in arguments:
                if arg in (
                    "-o",
                    "-dependency-file",
                    "-MT",
                    "-MQ",
                    "-serialize-diagnostic-file",
                    "-coverage-notes-file",
                    "-coverage-data-file",
                    "-split-dwarf-output",
                    "-opt-record-file",
                    "-dependency-dot",
                    "-diagnostic-log-file",
                    "-header-include-file",
                ):
                    if next(arguments, None) is None:
                        raise ValueError(f"missing compiler output argument: {arg}")
                elif arg in ("-emit-obj", "-MP") or arg.startswith(
                    ("-ftime-trace", "-stats-file=")
                ):
                    continue
                elif arg.startswith(("-fmodule", "-include-pch")):
                    raise ValueError("device PCH/modules need expanded dependencies")
                else:
                    command.append(arg)
            depfile = Path(temporary) / f"device-{index}.d"
            run(
                [
                    *command,
                    "-Eonly",
                    "-dependency-file",
                    str(depfile),
                    "-MT",
                    "kernrec_device_deps",
                    "-sys-header-deps",
                ],
                cwd=directory,
            )
            paths = {absolute(p, directory) for p in parse_depfile(depfile.read_text())}
            if source not in paths:
                raise ValueError(f"device dependencies omit compiled source: {source}")
            dependencies.update(paths)
    return dependencies


def build_map(
    build_root: Path,
    source_root: Path,
    tools: dict[str, str],
    *,
    commit: str = "",
    hipify_map: Path | None = None,
) -> dict:
    start = time.monotonic()
    source_root, build_root = source_root.resolve(), build_root.resolve()
    result: dict = {
        "version": 2,
        "backend": "rocm",
        "commit": commit,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "incomplete": False,
        "objects": [],
        "errors": [],
        "build_dirs": [],
        "tools": {},
    }
    try:
        for name, executable in tools.items():
            version = run([executable, "--version"]).splitlines()
            if not version:
                raise ValueError(f"no version information from {name}")
            result["tools"][name] = version[0]
        databases = sorted(build_root.rglob("compile_commands.json"))
        if not databases:
            raise ValueError(
                "no compile_commands.json; enable CMAKE_EXPORT_COMPILE_COMMANDS"
            )
        build_dirs = sorted({build_root, *(p.parent for p in databases)})
        result["build_dirs"] = [relative(p, source_root) for p in build_dirs]
        manifests = (
            [hipify_map]
            if hipify_map
            else sorted(build_root.rglob("hipify-source-map.json"))
        )
        provenance: dict[Path, Path] = {}
        for manifest in manifests:
            for generated, original in load_provenance(manifest, source_root).items():
                if generated in provenance and provenance[generated] != original:
                    raise ValueError(f"conflicting HIPify provenance: {generated}")
                provenance[generated] = original
        result["hipify_manifests"] = [relative(p, source_root) for p in manifests]
        generated = generated_dependencies(build_root, source_root)
        seen: dict[Path, tuple] = {}
        for database in databases:
            build_dir = database.parent
            if not (build_dir / ".ninja_deps").is_file():
                raise ValueError(f"missing Ninja dependency log: {build_dir}")
            dependencies = {
                absolute(obj, build_dir): paths
                for obj, paths in parse_deps(
                    run([tools["ninja"], "-C", str(build_dir), "-t", "deps"])
                ).items()
            }
            commands = json.loads(database.read_text())
            if not isinstance(commands, list) or not commands:
                raise ValueError(f"empty or invalid compilation database: {database}")
            for command in commands:
                obj = compilation_output(command)
                directory = Path(command["directory"]).resolve()
                source = absolute(command["file"], directory)
                response_files: set[Path] = set()
                args = compilation_arguments(command, response_files)
                identity = (source, directory, tuple(args))
                if obj in seen:
                    if seen[obj] != identity:
                        raise ValueError(f"conflicting compilation records: {obj}")
                    continue
                seen[obj] = identity
                if not obj.is_file():
                    raise ValueError(f"missing compiled object: {obj}")
                paths = dependencies.get(obj)
                if not paths:
                    raise ValueError(f"missing object dependencies: {obj}")
                deps = {absolute(p, build_dir) for p in paths}
                if source not in deps:
                    raise ValueError(f"dependency log omits compiled source: {obj}")
                symbols, targets = device_symbols(obj, tools)
                deps.update(device_dependencies(args, directory, source, targets))
                original_deps = {
                    original
                    for path in deps
                    for original in expand_dependencies(path, generated)
                }
                for path in (
                    deps
                    | original_deps
                    | response_files
                    | {provenance[p] for p in original_deps if p in provenance}
                ):
                    if not path.is_file():
                        raise ValueError(f"missing source/header: {path}")
                    if path.stat().st_mtime_ns > obj.stat().st_mtime_ns:
                        raise ValueError(f"source/header is newer than object: {path}")
                mapped_deps = sorted(
                    {
                        source_path(p, source_root, build_dirs, provenance)
                        for p in original_deps
                    }
                )
                mapped_source = source_path(source, source_root, build_dirs, provenance)
                target = TARGET.search(obj.as_posix())
                result["objects"].append(
                    {
                        "source": mapped_source,
                        "compiled_source": relative(source, source_root),
                        "target": target.group(1) if target else "",
                        "object": relative(obj, build_dir),
                        "device": bool(targets),
                        "symbols": symbols,
                        "deps": mapped_deps,
                        "offload_targets": targets,
                    }
                )
        for obj in build_root.rglob("*.o"):
            if any(
                "CompilerId" in part or part == "CMakeScratch" for part in obj.parts
            ):
                continue
            if obj.resolve() not in seen:
                raise ValueError(f"object is absent from compilation databases: {obj}")
        result["objects"].sort(key=lambda entry: (entry["source"], entry["target"]))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        result["objects"] = []
        result["incomplete"] = True
        result["reason"] = str(error)
        result["errors"].append({"error": str(error)})
    result["stats"] = {
        "objects": len(result["objects"]),
        "device_objects": sum(bool(o["device"]) for o in result["objects"]),
        "symbols": sum(len(o["symbols"]) for o in result["objects"]),
        "seconds": round(time.monotonic() - start, 2),
    }
    result["offload_targets"] = sorted(
        {target for obj in result["objects"] for target in obj["offload_targets"]}
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--hipify-map", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", default=os.environ.get("BUILDKITE_COMMIT", ""))
    parser.add_argument("--rocm-path", type=Path, default=Path("/opt/rocm"))
    args = parser.parse_args()
    tools = {}
    for name in ("llvm-readelf", "llvm-objcopy", "clang-offload-bundler", "ninja"):
        candidate = args.rocm_path / "llvm" / "bin" / name
        tools[name] = (
            str(candidate) if candidate.is_file() else shutil.which(name) or name
        )
    result = build_map(
        args.build_root,
        args.source_root,
        tools,
        commit=args.commit,
        hipify_map=args.hipify_map,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "stats": result["stats"],
                "reason": result.get("reason", ""),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
