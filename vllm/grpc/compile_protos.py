#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compile vLLM protobuf definitions into Python code.

This script uses grpcio-tools to generate *_pb2.py, *_pb2_grpc.py, and
*_pb2.pyi (type stubs) files from proto definitions.

NOTE: Proto compilation happens automatically during package build (via setup.py).
This script is provided for developers who want to regenerate protos manually,
e.g., after modifying .proto files.

Usage:
    python vllm/grpc/compile_protos.py

Requirements:
    pip install grpcio-tools
"""

import sys
from pathlib import Path

# All proto files to compile
PROTO_FILES = [
    "vllm_engine.proto",
    "render.proto",
]


def compile_protos():
    """Compile protobuf definitions."""
    script_dir = Path(__file__).parent
    vllm_package_root = script_dir.parent.parent  # vllm/vllm/grpc -> vllm/

    try:
        import grpc_tools
        from grpc_tools import protoc
    except ImportError:
        print("Error: grpcio-tools not installed")
        print("Install with: pip install grpcio-tools")
        return 1

    # Include path for well-known types (google/protobuf/struct.proto etc.)
    grpc_tools_proto_path = Path(grpc_tools.__file__).parent / "_proto"

    spdx_header = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        "# mypy: ignore-errors\n"
    )

    for proto_name in PROTO_FILES:
        proto_file = script_dir / proto_name
        if not proto_file.exists():
            print(f"Warning: Proto file not found at {proto_file}, skipping")
            continue

        print(f"Compiling protobuf: {proto_file}")

        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"--proto_path={vllm_package_root}",
                f"--proto_path={grpc_tools_proto_path}",
                f"--python_out={vllm_package_root}",
                f"--grpc_python_out={vllm_package_root}",
                f"--pyi_out={vllm_package_root}",
                str(proto_file),
            ]
        )

        if result != 0:
            print(f"Error: protoc returned {result} for {proto_name}")
            return result

        # Add SPDX headers to generated files
        stem = proto_name.replace(".proto", "")
        for suffix in ["_pb2.py", "_pb2_grpc.py", "_pb2.pyi"]:
            generated_file = script_dir / f"{stem}{suffix}"
            if generated_file.exists():
                content = generated_file.read_text()
                if not content.startswith("# SPDX-License-Identifier"):
                    generated_file.write_text(spdx_header + content)
                print(f"  Generated: {generated_file}")

    print("✓ Protobuf compilation successful!")
    return 0


if __name__ == "__main__":
    sys.exit(compile_protos())
