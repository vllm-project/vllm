#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from dataclasses import dataclass, field

import regex as re


@dataclass
class ForbiddenImport:
    pattern: str
    tip: str
    allowed_pattern: re.Pattern = re.compile(r"^$")  # matches nothing by default
    allowed_files: set[str] = field(default_factory=set)


CHECK_IMPORTS = {
    "pickle/cloudpickle": ForbiddenImport(
        pattern=(
            r"^\s*(import\s+(pickle|cloudpickle)(\s|$|\sas)"
            r"|from\s+(pickle|cloudpickle)\s+import\b)"
        ),
        tip=(
            "Avoid using pickle or cloudpickle or add this file to "
            "tools/pre_commit/check_forbidden_imports.py."
        ),
        allowed_files={
            # pickle
            "vllm/multimodal/hasher.py",
            "vllm/transformers_utils/config.py",
            "vllm/model_executor/models/registry.py",
            "vllm/compilation/caching.py",
            "vllm/compilation/piecewise_backend.py",
            "vllm/distributed/utils.py",
            "vllm/distributed/parallel_state.py",
            "vllm/distributed/device_communicators/all_reduce_utils.py",
            "vllm/distributed/device_communicators/shm_broadcast.py",
            "vllm/distributed/device_communicators/shm_object_storage.py",
            "vllm/distributed/weight_transfer/ipc_engine.py",
            "vllm/utils/hashing.py",
            "tests/multimodal/media/test_base.py",
            "tests/tokenizers_/test_hf.py",
            "tests/utils_/test_hashing.py",
            "tests/compile/test_aot_compile.py",
            "benchmarks/kernels/graph_machete_bench.py",
            "benchmarks/kernels/benchmark_lora.py",
            "benchmarks/kernels/benchmark_machete.py",
            "benchmarks/fused_kernels/layernorm_rms_benchmarks.py",
            "benchmarks/cutlass_benchmarks/w8a8_benchmarks.py",
            "benchmarks/cutlass_benchmarks/sparse_benchmarks.py",
            # cloudpickle
            "vllm/v1/executor/multiproc_executor.py",
            "vllm/v1/executor/ray_executor.py",
            "vllm/entrypoints/llm.py",
            "tests/utils.py",
            # pickle and cloudpickle
            "vllm/v1/serial_utils.py",
        },
    ),
    "re": ForbiddenImport(
        pattern=r"^\s*(?:import\s+re(?:$|\s|,)|from\s+re\s+import)",
        tip="Replace 'import re' with 'import regex as re' or 'import regex'.",
        allowed_pattern=re.compile(r"^\s*import\s+regex(\s*|\s+as\s+re\s*)$"),
        allowed_files={"setup.py"},
    ),
    "triton": ForbiddenImport(
        pattern=r"^(from|import)\s+triton(\s|\.|$)",
        tip="Use 'from vllm.triton_utils import triton' instead.",
        allowed_pattern=re.compile(
            "from vllm.triton_utils import (triton|tl|tl, triton)"
        ),
        allowed_files={"vllm/triton_utils/importing.py"},
    ),
}


def check_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return_code = 0
    # Check all patterns in the whole file
    for import_name, forbidden_import in CHECK_IMPORTS.items():
        # Skip files that are allowed for this import
        if path in forbidden_import.allowed_files:
            continue
        # Search for forbidden imports
        for match in re.finditer(forbidden_import.pattern, content, re.MULTILINE):
            # Check if it's allowed
            if forbidden_import.allowed_pattern.match(match.group()):
                continue
            # Calculate line number from match position
            line_num = content[: match.start() + 1].count("\n") + 1
            print(
                f"{path}:{line_num}: "
                "\033[91merror:\033[0m "  # red color
                f"Found forbidden import: {import_name}. {forbidden_import.tip}"
            )
            return_code = 1
    return return_code


def main():
    returncode = 0
    for path in sys.argv[1:]:
        returncode |= check_file(path)
    return returncode


def test_regex():
    test_cases = [
        # Should match
        ("import pickle", True),
        ("import cloudpickle", True),
        ("import pickle as pkl", True),
        ("import cloudpickle as cpkl", True),
        ("from pickle import *", True),
        ("from cloudpickle import dumps", True),
        ("from pickle import dumps, loads", True),
        ("from cloudpickle import (dumps, loads)", True),
        ("    import pickle", True),
        ("\timport cloudpickle", True),
        ("from   pickle   import   loads", True),
        # Should not match
        ("import somethingelse", False),
        ("from somethingelse import pickle", False),
        ("# import pickle", False),
        ("print('import pickle')", False),
        ("import pickleas as asdf", False),
    ]
    for i, (line, should_match) in enumerate(test_cases):
        result = bool(CHECK_IMPORTS["pickle/cloudpickle"].pattern.match(line))
        assert result == should_match, (
            f"Test case {i} failed: '{line}' (expected {should_match}, got {result})"
        )
    print("All regex tests passed.")


if __name__ == "__main__":
    if "--test-regex" in sys.argv:
        test_regex()
    else:
        sys.exit(main())
