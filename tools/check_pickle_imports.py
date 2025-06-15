#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import os
import sys

import regex as re

try:
    import pathspec
except ImportError:
    print(
        "ERROR: The 'pathspec' library is required. "
        "Install it with 'pip install pathspec'.",
        file=sys.stderr)
    sys.exit(2)

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
ALLOWED_FILES = set([
    # pickle
    'vllm/utils.py',
    'vllm/v1/serial_utils.py',
    'vllm/v1/executor/multiproc_executor.py',
    'vllm/multimodal/hasher.py',
    'vllm/transformers_utils/config.py',
    'vllm/model_executor/models/registry.py',
    'tests/test_utils.py',
    'tests/tokenization/test_cached_tokenizer.py',
    'tests/model_executor/test_guided_processors.py',
    'vllm/distributed/utils.py',
    'vllm/distributed/parallel_state.py',
    'vllm/engine/multiprocessing/client.py',
    'vllm/distributed/device_communicators/custom_all_reduce_utils.py',
    'vllm/distributed/device_communicators/shm_broadcast.py',
    'vllm/engine/multiprocessing/engine.py',
    'benchmarks/kernels/graph_machete_bench.py',
    'benchmarks/kernels/benchmark_lora.py',
    'benchmarks/kernels/benchmark_machete.py',
    'benchmarks/fused_kernels/layernorm_rms_benchmarks.py',
    'benchmarks/cutlass_benchmarks/w8a8_benchmarks.py',
    'benchmarks/cutlass_benchmarks/sparse_benchmarks.py',
    # cloudpickle
    'vllm/worker/worker_base.py',
    'vllm/executor/mp_distributed_executor.py',
    'vllm/executor/ray_distributed_executor.py',
    'vllm/entrypoints/llm.py',
    'tests/utils.py',
    # pickle and cloudpickle
    'vllm/utils.py',
    'vllm/v1/serial_utils.py',
    'vllm/v1/executor/multiproc_executor.py',
    'vllm/transformers_utils/config.py',
    'vllm/model_executor/models/registry.py',
    'vllm/engine/multiprocessing/client.py',
    'vllm/engine/multiprocessing/engine.py',
])

PICKLE_RE = re.compile(r"^\s*(import\s+(pickle|cloudpickle)(\s|$|\sas)"
                       r"|from\s+(pickle|cloudpickle)\s+import\b)")


def is_python_file(path):
    return path.endswith('.py')


def scan_file(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if PICKLE_RE.match(line):
                return True
    return False


def load_gitignore(repo_root):
    gitignore_path = os.path.join(repo_root, '.gitignore')
    patterns = []
    if os.path.exists(gitignore_path):
        with open(gitignore_path, encoding='utf-8') as f:
            patterns = f.read().splitlines()
    # Always ignore .git directory
    patterns.append('.git/')
    return pathspec.PathSpec.from_lines('gitwildmatch', patterns)


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = load_gitignore(repo_root)
    bad_files = []
    for dirpath, _, filenames in os.walk(repo_root):
        for filename in filenames:
            if not is_python_file(filename):
                continue
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, repo_root)
            # Skip ignored files
            if spec.match_file(rel_path):
                continue
            if scan_file(abs_path) and rel_path not in ALLOWED_FILES:
                bad_files.append(rel_path)
    if bad_files:
        print("\nERROR: The following files import 'pickle' or 'cloudpickle' "
              "but are not in the allowed list:")
        for f in bad_files:
            print(f"  {f}")
        print("\nIf this is intentional, update the allowed list in "
              "tools/check_pickle_imports.py.")
        sys.exit(1)
    sys.exit(0)


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
            f"Test case {i} failed: '{line}' "
            f"(expected {should_match}, got {result})")
    print("All regex tests passed.")


if __name__ == '__main__':
    if '--test-regex' in sys.argv:
        test_regex()
    else:
        main()
