#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile vLLM render protobuf definitions into Python code.

Generates *_pb2.py, *_pb2_grpc.py, and *_pb2.pyi for vllm_render.proto.

vllm_render.proto imports "vllm_engine.proto", which now lives in the
external smg-grpc-proto package (the in-tree copy was removed when
vllm/grpc/ was extracted to smg-grpc-servicer in #36169). We resolve
that include path at runtime and then patch the generated files' bare
imports so they point at ``smg_grpc_proto.vllm_engine_pb2`` and
``vllm.entrypoints.grpc`` respectively.

Proto compilation also runs automatically during package build (see
``setup.py``); this script exists for developers regenerating after
editing the .proto.

Usage:
    python -m vllm.entrypoints.grpc.compile_protos

Requirements:
    pip install grpcio-tools smg-grpc-proto
"""

from __future__ import annotations

import sys
from pathlib import Path

import regex as re

SPDX_HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "# mypy: ignore-errors\n"
)

PROTO_NAMES: tuple[str, ...] = ("vllm_render",)


def _smg_grpc_proto_dir() -> Path:
    """Resolve the proto/ directory shipped inside the smg-grpc-proto package."""
    import smg_grpc_proto  # type: ignore[import-untyped]

    return Path(smg_grpc_proto.__file__).parent / "proto"


def _well_known_protos_dir() -> Path:
    """Resolve grpc_tools' bundled well-known proto directory (google/*)."""
    import grpc_tools

    return Path(grpc_tools.__file__).parent / "_proto"


def _patch_generated(path: Path) -> None:
    """Rewrite bare imports in-place and prepend SPDX + mypy headers."""
    text = path.read_text()

    text = re.sub(
        r"^import vllm_engine_pb2 as ([A-Za-z_][A-Za-z0-9_]*)$",
        r"from smg_grpc_proto import vllm_engine_pb2 as \1",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^import vllm_engine_pb2$",
        r"from smg_grpc_proto import vllm_engine_pb2",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"^import vllm_render_pb2 as ([A-Za-z_][A-Za-z0-9_]*)$",
        r"from vllm.entrypoints.grpc import vllm_render_pb2 as \1",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^import vllm_render_pb2$",
        r"from vllm.entrypoints.grpc import vllm_render_pb2",
        text,
        flags=re.MULTILINE,
    )

    if not text.startswith("# SPDX-License-Identifier"):
        text = SPDX_HEADER + text

    path.write_text(text)


def compile_protos() -> int:
    try:
        from grpc_tools import protoc
    except ImportError:
        print("Error: grpcio-tools not installed", file=sys.stderr)
        print("Install with: pip install grpcio-tools", file=sys.stderr)
        return 1

    try:
        smg_proto_dir = _smg_grpc_proto_dir()
    except ImportError:
        print(
            "Error: smg-grpc-proto not installed — it provides vllm_engine.proto.",
            file=sys.stderr,
        )
        print("Install with: pip install smg-grpc-proto", file=sys.stderr)
        return 1

    script_dir = Path(__file__).parent
    well_known_dir = _well_known_protos_dir()

    for proto_name in PROTO_NAMES:
        proto_file = script_dir / f"{proto_name}.proto"
        if not proto_file.exists():
            print(f"Error: proto file not found at {proto_file}", file=sys.stderr)
            return 1

        print(f"Compiling {proto_file}")
        # NOTE: --proto_path order matters. `script_dir` must come first so
        # protoc picks up *our* vllm_render.proto rather than the stale copy
        # still shipped inside the smg-grpc-proto package.
        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"--proto_path={script_dir}",
                f"--proto_path={smg_proto_dir}",
                f"--proto_path={well_known_dir}",
                f"--python_out={script_dir}",
                f"--grpc_python_out={script_dir}",
                f"--pyi_out={script_dir}",
                str(proto_file),
            ]
        )
        if result != 0:
            print(f"Error: protoc returned {result}", file=sys.stderr)
            return result

        for suffix in ("_pb2.py", "_pb2_grpc.py", "_pb2.pyi"):
            generated = script_dir / f"{proto_name}{suffix}"
            if generated.exists():
                _patch_generated(generated)
                print(f"  Patched {generated}")

    print("Protobuf compilation successful.")
    return 0


if __name__ == "__main__":
    sys.exit(compile_protos())
