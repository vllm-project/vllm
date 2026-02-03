#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compile vLLM protobuf definitions into Python code.

This script uses grpcio-tools to generate *_pb2.py, *_pb2_grpc.py, and
*_pb2.pyi (type stubs) files from the proto definitions.

NOTE: Proto compilation happens automatically during package build (via setup.py).
This script is provided for developers who want to regenerate protos manually,
e.g., after modifying vllm_engine.proto or render.proto.

Usage:
    python vllm/grpc/compile_protos.py

Requirements:
    pip install grpcio-tools
"""

import sys
from pathlib import Path


def compile_proto(proto_name: str, script_dir: Path, vllm_package_root: Path) -> int:
    """Compile a single proto file."""
    from grpc_tools import protoc

    proto_file = script_dir / f"{proto_name}.proto"

    if not proto_file.exists():
        print(f"Error: Proto file not found at {proto_file}")
        return 1

    print(f"Compiling protobuf: {proto_file}")

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"--proto_path={vllm_package_root}",
            f"--python_out={vllm_package_root}",
            f"--grpc_python_out={vllm_package_root}",
            f"--pyi_out={vllm_package_root}",
            str(proto_file),
        ]
    )

    if result != 0:
        print(f"Error: protoc returned {result}")
        return result

    # Add SPDX headers to generated files
    spdx_header = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    )

    for suffix in ["_pb2.py", "_pb2_grpc.py", "_pb2.pyi"]:
        generated_file = script_dir / f"{proto_name}{suffix}"
        if generated_file.exists():
            content = generated_file.read_text()
            if not content.startswith("# SPDX-License-Identifier"):
                header = spdx_header + "# mypy: ignore-errors\n"
                generated_file.write_text(header + content)

    print(f"  Generated: {script_dir / f'{proto_name}_pb2.py'}")
    print(f"  Generated: {script_dir / f'{proto_name}_pb2_grpc.py'}")
    print(f"  Generated: {script_dir / f'{proto_name}_pb2.pyi'} (type stubs)")

    return 0


def compile_protos():
    """Compile all protobuf definitions."""
    # Get the vllm package root directory
    script_dir = Path(__file__).parent
    vllm_package_root = script_dir.parent.parent  # vllm/vllm/grpc -> vllm/

    print(f"Output directory: {script_dir}")

    proto_files = ["vllm_engine", "render"]

    try:
        from grpc_tools import protoc  # noqa: F401

        for proto_name in proto_files:
            result = compile_proto(proto_name, script_dir, vllm_package_root)
            if result != 0:
                return result

        print("✓ All protobuf compilations successful!")
        return 0

    except ImportError:
        print("Error: grpcio-tools not installed")
        print("Install with: pip install grpcio-tools")
        return 1
    except Exception as e:
        print(f"Error during compilation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(compile_protos())
