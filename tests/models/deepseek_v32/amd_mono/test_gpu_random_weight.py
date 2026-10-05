# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU test: one MonoKernel decoder layer at TP8 on random weights against the FP32
golden layer (FlyDSL's torch reference), batch 1 and 8.

Requires 8x gfx950 (MI350X / MI355X), FlyDSL, and FlyDSL's repository on PYTHONPATH for
the golden model (``kernels.monokernel.glm.reference``). Runs the random-weight harness
(tools/mono_check/random_weight_check.py) in a subprocess (it spawns one process per
rank).
"""

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

from vllm.platforms import current_platform

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))


def _gfx950_count() -> int:
    if not current_platform.is_rocm() or not torch.cuda.is_available():
        return 0
    from vllm.platforms.rocm import on_gfx950

    return torch.accelerator.device_count() if on_gfx950() else 0


pytestmark = [
    pytest.mark.skipif(_gfx950_count() < 8, reason="requires 8x gfx950"),
    pytest.mark.skipif(
        importlib.util.find_spec("flydsl") is None, reason="requires FlyDSL"
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("kernels") is None,
        reason="requires FlyDSL's repository on PYTHONPATH (golden model)",
    ),
]


@pytest.mark.parametrize("batch", [1, 8])
def test_random_weight_layer_vs_golden(batch):
    cmd = [
        sys.executable,
        os.path.join(ROOT, "tools", "mono_check", "random_weight_check.py"),
        "--npes", "8",
        "--batch", str(batch),
        "--samples", str(batch),
        "--ctx", "3000",
        "--iters", "2",
    ]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    assert res.stdout.strip().splitlines()[-1] == "PASS", res.stdout[-4000:]
