#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compile vLLM protobuf definitions into Python code.

This script uses grpcio-tools to generate *_pb2.py, *_pb2_grpc.py, and
*_pb2.pyi (type stubs) files from the vllm_engine.proto definition.

NOTE: Proto compilation happens automatically during package build (via setup.py).
This script is provided for developers who want to regenerate protos manually,
e.g., after modifying vllm_engine.proto.

Usage:
    python vllm/grpc/compile_protos.py

Requirements:
    pip install grpcio-tools
"""

import sys
from pathlib import Path


def compile_protos():
    """Compile protobuf definitions."""
    # Get the vllm package root directory
    script_dir = Path(__file__).parent
    vllm_package_root = script_dir.parent.parent  # vllm/vllm/grpc -> vllm/

    proto_file = script_dir / "vllm_engine.proto"

    if not proto_file.exists():
        print(f"Error: Proto file not found at {proto_file}")
        return 1

    print(f"Compiling protobuf: {proto_file}")
    print(f"Output directory: {script_dir}")

    # Compile the proto file
    # We use vllm/vllm as the proto_path so that the package is vllm.grpc.engine
    try:
        from grpc_tools import protoc

        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"--proto_path={vllm_package_root}",
                f"--python_out={vllm_package_root}",
                f"--grpc_python_out={vllm_package_root}",
                f"--pyi_out={vllm_package_root}",  # Generate type stubs
                str(script_dir / "vllm_engine.proto"),
            ]
        )

        if result == 0:
            # Add SPDX headers to generated files
            spdx_header = (
                "# SPDX-License-Identifier: Apache-2.0\n"
                "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
            )

            for generated_file in [
                script_dir / "vllm_engine_pb2.py",
                script_dir / "vllm_engine_pb2_grpc.py",
                script_dir / "vllm_engine_pb2.pyi",
            ]:
                if generated_file.exists():
                    content = generated_file.read_text()
                    if not content.startswith("# SPDX-License-Identifier"):
                        # Add mypy ignore-errors comment for all generated files
                        header = spdx_header + "# mypy: ignore-errors\n"
                        generated_file.write_text(header + content)

            print("✓ Protobuf compilation successful!")
            print(f"  Generated: {script_dir / 'vllm_engine_pb2.py'}")
            print(f"  Generated: {script_dir / 'vllm_engine_pb2_grpc.py'}")
            print(f"  Generated: {script_dir / 'vllm_engine_pb2.pyi'} (type stubs)")
            return 0
        else:
            print(f"Error: protoc returned {result}")
            return result

    except ImportError:
        print("Error: grpcio-tools not installed")
        print("Install with: pip install grpcio-tools")
        return 1
    except Exception as e:
        print(f"Error during compilation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(compile_protos())
