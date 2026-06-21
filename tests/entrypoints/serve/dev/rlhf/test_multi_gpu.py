# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tier 3 — Multi-GPU RL lifecycle tests (TP=2).

These tests exercise the sleep/wake/pause lifecycle across Tensor Parallel
workers.  In TP=2, CuMemAllocator must unmap/remap on every rank; any missed
rank will cause an illegal-memory-access on the first forward pass after wake.

All tests are automatically skipped when fewer than 2 GPUs are available.

Test classes
------------
TestTP2SleepWakeMemory     GPU free-bytes changes on every rank (measured via TP=2 server)
TestTP2OutputCorrectness   golden output roundtrip across sleep/wake with TP=2
TestTP2StagedWake          wake(weights) → wake(kv_cache) staged pattern at TP=2

Reference implementations:
  - sglang test/registered/rl/test_multi_instance_release_memory_occupation.py
    _run_sglang_subprocess (6-step release/resume, dp=2, tp=2)
  - trl tests/test_vllm_client_server.py
    TestVLLMClientServerTP — generate + weight update at TP=2 (3 GPU)

RFC: https://github.com/vllm-project/vllm/issues/45585
PR:  https://github.com/vllm-project/vllm/pull/45586
"""

import contextlib
import os
import subprocess
import sys
import time
from contextlib import contextmanager

import pytest
import requests

MODEL_NAME = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")

# ---------------------------------------------------------------------------
# GPU availability guard — skip entire module when < 2 GPUs
# ---------------------------------------------------------------------------


def _gpu_count() -> int:
    try:
        out = subprocess.check_output(
            [sys.executable, "-c",
             "import torch; print(torch.cuda.device_count())"],
            timeout=15,
        )
        return int(out.strip())
    except Exception:
        return 0


_NGPUS = _gpu_count()
pytestmark = pytest.mark.skipif(
    _NGPUS < 2,
    reason=f"Tier 3 multi-GPU tests require >= 2 GPUs (found {_NGPUS})",
)

# ---------------------------------------------------------------------------
# Server harness
# ---------------------------------------------------------------------------

_TP2_ARGS = [
    "--dtype", "bfloat16",
    "--max-model-len", "2048",
    "--max-num-seqs", "32",
    "--gpu-memory-utilization", "0.70",
    "--enable-sleep-mode",
    "--enforce-eager",
    "--tensor-parallel-size", "2",
]


@contextmanager
def _server_tp2(extra_args=None, port: int = 8790, timeout: float = 240.0):
    """Launch a TP=2 vLLM server with dev router; yield base URL."""
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", str(port),
        "--served-model-name", "m",
        *(_TP2_ARGS + (extra_args or [])),
    ]
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    url = f"http://localhost:{port}"
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                err = (
                    proc.stderr.read(4000).decode(errors="replace")
                    if proc.stderr else ""
                )
                raise RuntimeError(f"TP=2 vllm server exited during startup:\n{err}")
            with contextlib.suppress(Exception):
                if requests.get(f"{url}/health", timeout=3).status_code == 200:
                    break
            time.sleep(1)
        else:
            proc.terminate()
            raise RuntimeError("TP=2 vllm server did not start in time")
        yield url
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        if proc.poll() is None:
            proc.kill()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _gen(url, prompt="The capital of France is", max_tokens=8, timeout=30):
    try:
        r = requests.post(
            f"{url}/v1/completions",
            json={"model": "m", "prompt": prompt,
                  "max_tokens": max_tokens, "temperature": 0},
            timeout=timeout,
        )
        return r.json()
    except Exception:
        return None


def _ok(resp) -> bool:
    return (
        resp is not None
        and "choices" in resp
        and bool(resp["choices"])
        and "error" not in resp
    )


def _sleep(url, level=1, mode="abort"):
    return requests.post(
        f"{url}/sleep", params={"level": level, "mode": mode}, timeout=20
    ).status_code


def _wake(url, tags=None):
    params = {"tags": tags} if tags else {}
    return requests.post(f"{url}/wake_up", params=params, timeout=30).status_code


def _is_sleeping(url) -> bool:
    return requests.get(f"{url}/is_sleeping", timeout=5).json()["is_sleeping"]


def _health(url) -> int:
    try:
        return requests.get(f"{url}/health", timeout=5).status_code
    except Exception:
        return 0


def _gpu_free_bytes(device: int = 0) -> int:
    out = subprocess.check_output(
        [sys.executable, "-c",
         f"import torch; f,_=torch.cuda.mem_get_info({device}); print(f)"],
        timeout=10,
    )
    return int(out.strip())


# ---------------------------------------------------------------------------
# TestTP2SleepWakeMemory
# ---------------------------------------------------------------------------


class TestTP2SleepWakeMemory:
    """TP=2 sleep releases GPU memory; wake restores it.

    With TP=2 the model is sharded across 2 GPUs.  CuMemAllocator must
    unmap on both ranks during sleep and remap on both during wake.
    We verify GPU-0 free bytes to confirm at least one rank's cumem
    operations are being executed.

    Reference: sglang test_multi_instance_release_memory_occupation.py
               _run_sglang_subprocess (6-step release/resume, tp=2)
    """

    def test_tp2_sleep_frees_gpu_memory(self):
        """TP=2 sleep(1) releases GPU memory on both ranks; wake restores it.

        Checks GPU-0 and GPU-1 independently: each should hold roughly half
        the model weight shard plus KV allocations.
        """
        with _server_tp2(port=8790) as url:
            _gen(url)  # warm up
            free_awake_0 = _gpu_free_bytes(device=0)
            free_awake_1 = _gpu_free_bytes(device=1)

            assert _sleep(url, level=1) == 200
            assert _is_sleeping(url) is True
            free_sleep_0 = _gpu_free_bytes(device=0)
            free_sleep_1 = _gpu_free_bytes(device=1)

            freed_0 = (free_sleep_0 - free_awake_0) / 2**30
            freed_1 = (free_sleep_1 - free_awake_1) / 2**30
            # Each rank holds ~half the 0.6B BF16 model (~0.6 GiB) + KV pool
            assert freed_0 > 0.3, (
                f"TP=2 sleep freed only {freed_0:.2f} GiB on GPU-0 — "
                "CuMemAllocator may not be operating on rank 0"
            )
            assert freed_1 > 0.3, (
                f"TP=2 sleep freed only {freed_1:.2f} GiB on GPU-1 — "
                "CuMemAllocator may not be operating on rank 1"
            )

            assert _wake(url) == 200
            assert _is_sleeping(url) is False
            free_wake_0 = _gpu_free_bytes(device=0)
            free_wake_1 = _gpu_free_bytes(device=1)

            # After wake, memory is re-allocated: free bytes decrease vs sleeping.
            re_alloc_0 = (free_sleep_0 - free_wake_0) / 2**30
            re_alloc_1 = (free_sleep_1 - free_wake_1) / 2**30
            assert re_alloc_0 > 0.2, (
                f"TP=2 wake re-allocated only {re_alloc_0:.2f} GiB on GPU-0"
            )
            assert re_alloc_1 > 0.2, (
                f"TP=2 wake re-allocated only {re_alloc_1:.2f} GiB on GPU-1"
            )
            assert _health(url) == 200
            assert _ok(_gen(url))


# ---------------------------------------------------------------------------
# TestTP2OutputCorrectness
# ---------------------------------------------------------------------------


class TestTP2OutputCorrectness:
    """TP=2 sleep/wake must not corrupt weights; golden output must be stable.

    In TP=2 each rank holds a shard of the weight tensors.  If any rank's
    cumem remap is incorrect, the allreduce on the forward pass will produce
    wrong logits.  We verify with a golden output roundtrip and 3 cycles.

    Reference: sglang test_update_weights_from_disk.py
               TestServerUpdateWeightsFromDisk with tp=2
    """

    def test_tp2_full_wake_restores_output(self):
        """sleep(1) → wake_up() → output matches golden at TP=2."""
        with _server_tp2(port=8791, extra_args=["--seed", "42"]) as url:
            golden = _gen(url)
            assert golden and _ok(golden)
            golden_text = golden["choices"][0]["text"]
            assert golden_text.strip()

            assert _sleep(url, level=1) == 200
            assert _wake(url) == 200
            assert _health(url) == 200

            resp = _gen(url)
            assert resp and _ok(resp), "generate failed after TP=2 sleep/wake"
            assert resp["choices"][0]["text"] == golden_text, (
                "output changed after TP=2 sleep/wake — "
                "weight restore on one or more TP ranks may be broken"
            )

    def test_tp2_three_cycles_stable(self):
        """3× sleep/wake cycles at TP=2 — output stable, engine alive."""
        with _server_tp2(port=8792, extra_args=["--seed", "42"]) as url:
            golden = _gen(url)
            assert golden and _ok(golden), "initial generate failed before TP=2 cycles"
            golden_text = golden["choices"][0]["text"]
            assert golden_text.strip(), "golden output must be non-empty"

            for cycle in range(3):
                assert _sleep(url, level=1) == 200
                assert _wake(url) == 200
                assert _health(url) == 200

                resp = _gen(url)
                assert resp and _ok(resp), f"generate failed on TP=2 cycle {cycle}"
                assert resp["choices"][0]["text"] == golden_text, (
                    f"output drifted on TP=2 cycle {cycle}"
                )


# ---------------------------------------------------------------------------
# TestTP2StagedWake
# ---------------------------------------------------------------------------


class TestTP2StagedWake:
    """TP=2 staged wake: wake(weights) then wake(kv_cache) independently.

    This is the colocate RL pattern at TP=2.  Between the two wake calls,
    is_sleeping must remain True (kv_cache still unmapped on all ranks).
    After both wakes, the engine must generate correctly.

    Reference: sglang test_multi_instance_release_memory_occupation.py
               test_multi_stage_release_and_resume with tp=2
    """

    def test_tp2_staged_wake_keeps_sleeping_between_tags(self):
        """wake(["weights"]) at TP=2 must leave is_sleeping=True.

        Guards against the #44483 regression at TP=2: if any rank resumes
        the scheduler before kv_cache is remapped, the first forward pass
        will hit an unmapped KV VA and crash with CUDA IMA.
        """
        with _server_tp2(port=8793) as url:
            assert _sleep(url, level=1) == 200
            assert _is_sleeping(url) is True

            assert _wake(url, tags=["weights"]) == 200
            # kv_cache still unmapped — must still be sleeping
            assert _is_sleeping(url) is True, (
                "is_sleeping went False after weights-only wake at TP=2 — "
                "scheduler may have resumed before kv_cache is remapped "
                "(PR #44483 regression in TP=2 path)"
            )

            assert _wake(url, tags=["kv_cache"]) == 200
            assert _is_sleeping(url) is False
            assert _health(url) == 200
            assert _ok(_gen(url)), "generate failed after TP=2 staged wake"
