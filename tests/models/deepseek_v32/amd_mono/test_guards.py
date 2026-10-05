# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of mono/guards.check_before_install on fake model / config objects: RoPE
table length, KV cache < 4 GiB per layer, FULL-graph capture sizes == kernel widths;
and the output rank's fail-stop latch."""

import time
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.models.deepseek_v32.amd.mono import guards
from vllm.models.deepseek_v32.amd.mono.guards import check_before_install
from vllm.models.deepseek_v32.amd.mono.live import LiveConfig


class FakeKV:
    def __init__(self, nbytes):
        self.n = nbytes // 2

    def numel(self):
        return self.n

    def element_size(self):
        return 2


def model(kv_bytes, rope_rows=8192):
    rope = NS(cos_sin_cache=torch.zeros(rope_rows, 64))
    lay = NS(self_attn=NS(kv_cache=FakeKV(kv_bytes), rotary_emb=rope))
    return NS(model=NS(layers=[lay] * 78))


def vc(mml=4096, mode=CUDAGraphMode.FULL_DECODE_ONLY, sizes=(1, 2, 4, 5, 6, 8)):
    return NS(
        model_config=NS(max_model_len=mml),
        compilation_config=NS(cudagraph_mode=mode, cudagraph_capture_sizes=list(sizes)),
    )


KV_OK = 151 << 20


def test_accepts_default_config():
    check_before_install(model(KV_OK), LiveConfig(ckpt="x"), vc())


def test_rope_table_raised_to_max_model_len():
    cfg = LiveConfig(ckpt="x")
    check_before_install(model(KV_OK), cfg, vc(mml=8000))
    assert cfg.max_model_len == 8000


@pytest.mark.parametrize(
    "mdl, cfg_vc",
    [
        pytest.param(model(KV_OK, rope_rows=4096), vc(mml=8000), id="rope_rows"),
        pytest.param(model(5 << 30), vc(), id="kv_ge_4GiB"),
        pytest.param(model(KV_OK), vc(sizes=(1, 2, 4, 8)), id="capture_ne_widths"),
        pytest.param(
            model(KV_OK), vc(sizes=(1, 2, 4, 5, 6, 8, 16)), id="capture_gt_widths"
        ),
    ],
)
def test_refusals(mdl, cfg_vc):
    with pytest.raises(RuntimeError):
        check_before_install(mdl, LiveConfig(ckpt="x"), cfg_vc)


def test_eager_has_no_width_constraint():
    check_before_install(
        model(KV_OK), LiveConfig(ckpt="x"), vc(mode=CUDAGraphMode.NONE, sizes=())
    )


def test_output_rank_failstop_latch(failstop_latch):
    """The output rank latches its first fail-stop, so the worker can refuse the
    sample_tokens / execute_model calls that follow (an exception raised in the forward
    alone is lost under async scheduling)."""
    assert guards.failstop_error() is None
    with pytest.raises(RuntimeError, match="first -> fail-stop"):
        guards.fail_stop("first", rank=0)
    with pytest.raises(RuntimeError, match="second"):
        guards.fail_stop("second", rank=0)
    assert guards.failstop_error() == "first -> fail-stop"
    assert len(failstop_latch) == 1  # exit watchdog armed once


def test_exit_watchdog_exits_worker():
    """The watchdog ends a process whose main thread is stuck (here: a sleep standing in
    for a device sync waiting on exited peers) and dumps its stack."""
    import subprocess
    import sys

    code = (
        "import time\n"
        "from vllm.models.deepseek_v32.amd.mono.guards import _arm_exit_watchdog\n"
        "_arm_exit_watchdog(0.5)\n"
        "time.sleep(60)\n"
    )
    t0 = time.monotonic()
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=50
    )
    assert p.returncode != 0 and time.monotonic() - t0 < 50
    assert "time.sleep" in p.stderr or "<string>" in p.stderr, p.stderr[-2000:]
