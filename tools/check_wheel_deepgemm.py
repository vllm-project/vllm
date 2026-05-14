# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Assert the installed vLLM has a `_C.cpython-X.Y-*.so` for every CPython
covered by `requires-python`. Fails closed if a Python's `.so` is missing
from the wheel — i.e. the regression that surfaced in #41476/#41512.

Run from a CI test job after vLLM is installed, e.g. the H100 deepgemm
kernel tests in .buildkite/test_areas/kernels.yaml.
"""

import importlib.util
import os
import sys
from pathlib import Path

import regex as re
import tomllib

SO_RE = re.compile(r"^_C\.cpython-(\d)(\d+)-")


def required_pythons() -> list[str]:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    spec = tomllib.loads(pyproject.read_text())["project"]["requires-python"]
    m = re.match(r">=3\.(\d+),<3\.(\d+)", spec)
    if not m:
        sys.exit(f"unexpected requires-python format: {spec!r}")
    return [f"3.{v}" for v in range(int(m[1]), int(m[2]))]


spec = importlib.util.find_spec("vllm.third_party.deep_gemm")
if spec is None or spec.origin is None:
    sys.exit("vllm.third_party.deep_gemm not importable; is vllm installed?")
pkg_dir = Path(spec.origin).parent

found = {f"{m[1]}.{m[2]}" for f in os.listdir(pkg_dir) if (m := SO_RE.match(f))}
required = required_pythons()
missing = [v for v in required if v not in found]
print(f"deepgemm _C: found {sorted(found)}, required {required}, missing {missing}")
sys.exit(1 if missing else 0)
