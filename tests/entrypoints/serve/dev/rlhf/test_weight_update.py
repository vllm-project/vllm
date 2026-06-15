# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for the vLLM RL weight-transfer protocol.

Endpoint surface under test
---------------------------
rlhf/api_router : POST /start_weight_update
                  POST /update_weights
                  POST /finish_weight_update
                  POST /init_weight_transfer_engine
                  GET  /get_world_size

Also covers the **compound RL training-step sequence** — the end-to-end
workflow that combines sleep/wake (sleep/api_router) with weight transfer:

    rollout  →  sleep  →  start_weight_update  →  finish_weight_update
             →  wake(weights)  →  wake(kv_cache)  →  rollout

This sequence is the exact lifecycle that PR #44483 protects: wake(weights)
must not resume the scheduler until kv_cache is also resident.

All tests require:
  --enable-sleep-mode   KV cache allocated via CuMemAllocator
  VLLM_SERVER_DEV_MODE=1

RFC: https://github.com/vllm-project/vllm/issues/45585
"""

import contextlib
import os
import subprocess
import sys
import time
from contextlib import contextmanager

import pytest
import requests

MODEL_NAME = os.environ.get("VLLM_TEST_MODEL", "meta-llama/Llama-3.2-1B-Instruct")

_BASE_ARGS = [
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "2048",
    "--max-num-seqs",
    "32",
    "--gpu-memory-utilization",
    "0.75",
    "--enable-sleep-mode",
    "--enforce-eager",
]


# ---------------------------------------------------------------------------
# Server harness
# ---------------------------------------------------------------------------


@contextmanager
def _server(extra_args=None, port: int = 8771, timeout: float = 180.0):
    env = {**os.environ, "VLLM_SERVER_DEV_MODE": "1"}
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--port",
        str(port),
        "--served-model-name",
        "m",
        *(_BASE_ARGS + (extra_args or [])),
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
                    if proc.stderr
                    else ""
                )
                raise RuntimeError(f"vllm server exited during startup:\n{err}")
            with contextlib.suppress(Exception):
                if requests.get(f"{url}/health", timeout=3).status_code == 200:
                    break
            time.sleep(1)
        else:
            proc.terminate()
            raise RuntimeError("vllm server did not start in time")
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
            json={
                "model": "m",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
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


def _sleep(url, level=1):
    return requests.post(
        f"{url}/sleep", params={"level": level, "mode": "abort"}, timeout=15
    ).status_code


def _wake(url, tags=None):
    params = {"tags": tags} if tags else {}
    return requests.post(f"{url}/wake_up", params=params, timeout=20).status_code


def _health(url) -> int:
    try:
        return requests.get(f"{url}/health", timeout=5).status_code
    except Exception:
        return 0


def _start_weight_update(url):
    return requests.post(f"{url}/start_weight_update", json={}, timeout=10)


def _finish_weight_update(url):
    return requests.post(f"{url}/finish_weight_update", timeout=10)


# ---------------------------------------------------------------------------
# TestWeightTransferProtocol
# ---------------------------------------------------------------------------


class TestWeightTransferProtocol:
    """HTTP weight-transfer protocol framing tests.

    These tests verify the protocol endpoints are correctly wired and
    the engine survives the full start → (update) → finish sequence.
    They do NOT push actual tensor data; see TestWeightUpdateWithTensors.
    """

    def test_start_finish_no_tensors_engine_survives(self):
        """/start_weight_update → /finish_weight_update with no tensors.

        Protocol framing test: the endpoints must accept an empty update
        and leave the engine healthy.
        """
        with _server() as url:
            r = _start_weight_update(url)
            assert r.status_code == 200, f"start_weight_update failed: {r.text}"

            r = _finish_weight_update(url)
            assert r.status_code == 200, f"finish_weight_update failed: {r.text}"

            assert _health(url) == 200
            assert _ok(_gen(url))

    def test_get_world_size_positive(self):
        with _server() as url:
            r = requests.get(f"{url}/get_world_size", timeout=5)
            assert r.status_code == 200
            ws = r.json()["world_size"]
            assert isinstance(ws, int) and ws >= 1

    def test_finish_without_start_handled(self):
        """/finish_weight_update without a preceding /start must not crash."""
        with _server() as url:
            r = _finish_weight_update(url)
            assert r.status_code in (200, 400, 409), (
                f"unexpected status {r.status_code} for finish-without-start"
            )
            assert _health(url) == 200

    def test_prefix_cache_flushed_after_finish(self):
        """/finish_weight_update must flush prefix cache.

        If flush is skipped, a subsequent sleep cycle on the same prompt
        could reuse a stale KV entry pointing to a released physical page.
        """
        with _server() as url:
            prompt = "The capital of France is"
            _gen(url, prompt=prompt)  # populate cache

            _start_weight_update(url)
            _finish_weight_update(url)

            # same prompt after update — must succeed (not IMA on stale cache)
            resp = _gen(url, prompt=prompt)
            assert _ok(resp), (
                "generate failed after finish_weight_update with cached prompt — "
                "prefix cache may not have been flushed"
            )


# ---------------------------------------------------------------------------
# TestWeightUpdateWithTensors
# ---------------------------------------------------------------------------


class TestWeightUpdateWithTensors:
    """Push real weight tensors and verify output changes.

    Requires VLLM_TEST_MODEL_ALT to point to a *different* checkpoint than
    VLLM_TEST_MODEL.  If not set the test is skipped.
    """

    @pytest.fixture(autouse=True)
    def _require_alt_model(self):
        alt = os.environ.get("VLLM_TEST_MODEL_ALT")
        if not alt or alt == MODEL_NAME:
            pytest.skip(
                "Set VLLM_TEST_MODEL_ALT to a different checkpoint to run "
                "weight-update output-change tests"
            )
        self.alt_model_path = alt

    def test_weight_update_changes_output(self):
        """After pushing different weights, output must change.

        This is the core RL use-case: post-training weights pushed into
        the rollout engine must produce different (updated) completions.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError:
            pytest.skip("transformers not available")

        with _server() as url:
            golden_resp = _gen(url)
            assert golden_resp
            golden_text = golden_resp["choices"][0]["text"]

            alt = AutoModelForCausalLM.from_pretrained(
                self.alt_model_path, torch_dtype=torch.bfloat16
            )
            _start_weight_update(url)
            for name, tensor in alt.named_parameters():
                import pybase64 as base64

                update_info = {
                    "name": name,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "data": base64.b64encode(
                        tensor.cpu().contiguous().numpy().tobytes()
                    ).decode(),
                }
                r = requests.post(
                    f"{url}/update_weights",
                    json={"update_info": update_info},
                    timeout=30,
                )
                assert r.status_code == 200, f"update_weights failed for {name}"
            del alt
            _finish_weight_update(url)

            resp_after = _gen(url)
            assert _ok(resp_after), "generate failed after weight update"
            assert resp_after["choices"][0]["text"] != golden_text, (
                "output did not change after pushing different weights — "
                "update_weights may be a no-op"
            )
            assert _health(url) == 200


# ---------------------------------------------------------------------------
# TestCompoundRLStep
# ---------------------------------------------------------------------------


class TestCompoundRLStep:
    """Full RL training-step sequence end-to-end.

    Tests the exact lifecycle that PR #44483 protects:
      rollout → sleep → weight_update → wake(weights) → wake(kv_cache) → rollout

    The window between wake(weights) and wake(kv_cache) is the "danger zone"
    where pre-#44483 vLLM would dispatch a generate step with kv_cache
    unmapped → TMA descriptor 700 / illegal memory access.
    """

    def test_single_rl_step_engine_survives(self):
        """One complete RL step via HTTP endpoints."""
        with _server() as url:
            # 1. rollout
            assert _ok(_gen(url))

            # 2. hand GPU to trainer
            assert _sleep(url, level=1) == 200

            # 3. weight update (framing only; no real tensors)
            assert _start_weight_update(url).status_code == 200
            assert _finish_weight_update(url).status_code == 200

            # 4. partial wake — weights only (kv_cache still unmapped)
            assert _wake(url, tags=["weights"]) == 200

            # 5. complete wake — kv_cache remapped; guard lifts
            assert _wake(url, tags=["kv_cache"]) == 200

            assert _health(url) == 200

            # 6. next rollout
            assert _ok(_gen(url)), "generate failed after full RL step sequence"

    def test_two_rl_steps_stable(self):
        """Two consecutive RL steps — engine alive and healthy throughout."""
        with _server() as url:
            for step in range(2):
                assert _ok(_gen(url)), f"rollout failed at step {step}"

                assert _sleep(url, level=1) == 200
                assert _start_weight_update(url).status_code == 200
                assert _finish_weight_update(url).status_code == 200
                assert _wake(url, tags=["weights"]) == 200
                assert _wake(url, tags=["kv_cache"]) == 200

                assert _health(url) == 200, f"engine died after step {step}"
