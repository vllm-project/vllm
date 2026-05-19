# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small native Windows CUDA smoke test.

Run from a VS Developer Command Prompt with CUDA and torch DLLs on PATH.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO_ROOT)
os.environ["PYTHONPATH"] = (
    REPO_ROOT + os.pathsep + os.environ["PYTHONPATH"]
    if os.environ.get("PYTHONPATH")
    else REPO_ROOT
)

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="facebook/opt-125m",
        enforce_eager=True,
        max_model_len=64,
        gpu_memory_utilization=0.20,
        attention_backend="TRITON_ATTN",
    )
    outputs = llm.generate(
        ["Windows CUDA source build test"],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
