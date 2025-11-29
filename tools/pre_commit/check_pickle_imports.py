#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

import regex as re

# List of files (relative to repo root) that are allowed to import pickle or
# cloudpickle
#
# STOP AND READ BEFORE YOU ADD ANYTHING ELSE TO THIS LIST:
#  The pickle and cloudpickle modules are known to be unsafe when deserializing
#  data from potentially untrusted parties. They have resulted in multiple CVEs
#  for vLLM and numerous vulnerabilities in the Python ecosystem more broadly.
#  Before adding new uses of pickle/cloudpickle, please consider safer
#  alternatives like msgpack or pydantic that are already in use in vLLM. Only
#  add to this list if absolutely necessary and after careful security review.
ALLOWED_FILES = {
    # pickle
    "vllm/multimodal/hasher.py",
    "vllm/transformers_utils/config.py",
    "vllm/model_executor/models/registry.py",
    "vllm/compilation/caching.py",
    "vllm/distributed/utils.py",
    "vllm/distributed/parallel_state.py",
    "vllm/distributed/device_communicators/all_reduce_utils.py",
    "vllm/distributed/device_communicators/shm_broadcast.py",
    "vllm/distributed/device_communicators/shm_object_storage.py",
    "vllm/utils/hashing.py",
    "tests/tokenizers_/test_cached_tokenizer.py",
    "tests/utils_/test_hashing.py",
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
}

PICKLE_RE = re.compile(
    r"^\s*(import\s+(pickle|cloudpickle)(\s|$|\sas)"
    r"|from\s+(pickle|cloudpickle)\s+import\b)"
)


def scan_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if PICKLE_RE.match(line):
                print(
                    f"{path}:{i}: "
                    "\033[91merror:\033[0m "  # red color
                    "Found pickle/cloudpickle import"
                )
                return 1
    return 0


def main():
    returncode = 0
    for filename in sys.argv[1:]:
        if filename in ALLOWED_FILES:
            continue
        returncode |= scan_file(filename)
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
        result = bool(PICKLE_RE.match(line))
        assert result == should_match, (
            f"Test case {i} failed: '{line}' (expected {should_match}, got {result})"
        )
    print("All regex tests passed.")


if __name__ == "__main__":
    if "--test-regex" in sys.argv:
        test_regex()
    else:
        sys.exit(main())
