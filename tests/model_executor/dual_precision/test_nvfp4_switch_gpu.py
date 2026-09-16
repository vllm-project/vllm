# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode with the NVFP4 shadow actually bound, not merely attached.

``test_nvfp4_shadow_gpu.py`` checks that the shadow attaches and that each shadow
linear tracks its BF16 twin, but the generation there runs with the binding still
on BF16, because the model runner rebinds before every forward from the
scheduler's decision. Manually binding would be overwritten. So this test drives
the production path instead: ``VLLM_DUAL_PRECISION_POLICY=uniform_w4`` makes the
scheduler ask for INT4 on every step, and the assertion is that all 152 bindings
really point at the shadow while the engine still emits sensible text.

One GPU, two sequential child processes, about 4 minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu_smoke

HERE = Path(__file__).resolve().parent
CHILD = HERE / "nvfp4_switch_child.py"
HF_HUB = os.environ.get("DUAL_PRECISION_HF_HUB", "/data/huggingface/hub")
QWEN35_9B_BF16 = (
    f"{HF_HUB}/models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
QWEN35_9B_NVFP4 = (
    f"{HF_HUB}/models--AxionML--Qwen3.5-9B-NVFP4/snapshots/"
    "97aef92393f126bf649f310cd40861be8dad3279"
)

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
for _path in (QWEN35_9B_BF16, QWEN35_9B_NVFP4):
    if not os.path.isdir(_path):
        pytest.skip(f"checkpoint not available: {_path}", allow_module_level=True)


def run_child(mode: str, tmp_path: Path) -> dict:
    report = tmp_path / f"{mode}.json"
    environment = {
        **os.environ,
        "BF16_MODEL": QWEN35_9B_BF16,
        "NVFP4_MODEL": QWEN35_9B_NVFP4,
    }
    result = subprocess.run(
        [sys.executable, str(CHILD), mode, str(report)],
        capture_output=True,
        text=True,
        timeout=1800,
        env=environment,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(report.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reports(tmp_path_factory) -> dict[str, dict]:
    tmp_path = tmp_path_factory.mktemp("nvfp4_switch")
    return {mode: run_child(mode, tmp_path) for mode in ("w4", "bf16")}


def test_uniform_w4_binds_every_shadow_layer(reports):
    w4 = reports["w4"]

    assert w4["active_precision"] == "int4"
    assert w4["shadow_bindings_total"] > 0
    # Not "some": uniform_w4 with BF16_LAYERS=none must bind all of them.
    assert w4["bindings_on_shadow"] == w4["shadow_bindings_total"], w4


def test_bf16_leaves_the_shadow_attached_but_unbound(reports):
    bf16 = reports["bf16"]

    assert bf16["active_precision"] == "bf16"
    assert bf16["bindings_on_shadow"] == 0, bf16
    # The shadow is loaded either way; only the binding differs.
    assert bf16["shadow_bindings_total"] == reports["w4"]["shadow_bindings_total"]


def test_nvfp4_bound_engine_still_answers_sensibly(reports):
    capital, arithmetic = reports["w4"]["texts"]

    assert "Paris" in capital, capital
    # 17 + 25 = 42: a shadow that decoded garbage would not land on this.
    assert "42" in arithmetic, arithmetic
