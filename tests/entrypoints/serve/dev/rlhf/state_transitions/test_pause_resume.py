# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for the vLLM RL pause/resume lifecycle.

Endpoint surface under test
---------------------------
rlhf/api_router   : POST /pause  POST /resume   GET /is_paused

Test classes
------------------------------------------------
TestPauseResume             /pause /resume /is_paused independent of sleep

RFC: https://github.com/vllm-project/vllm/issues/45585
Fixes regression introduced by: https://github.com/vllm-project/vllm/pull/44483
"""

import threading
import time

from tests.entrypoints.serve.dev.rlhf.conftest import (
    gen,
    ok,
    pause,
    resume,
    server,
)


# ---------------------------------------------------------------------------
# TestPauseResume
# ---------------------------------------------------------------------------


class TestPauseResume:
    """POST /pause  POST /resume  GET /is_paused are independent of sleep.

    /pause blocks scheduling without releasing GPU memory (level=0 equivalent
    from the GPU side, but a distinct code path and distinct state flag).
    """

    def test_pause_mode_wait_drains_inflight_request(self):
        """mode='wait' lets an in-flight request complete, then blocks new ones."""
        with server() as url:
            result: dict = {}

            def _bg():
                result["r"] = gen(url, max_tokens=32, timeout=60)

            t = threading.Thread(target=_bg)
            t.start()
            time.sleep(0.5)

            assert pause(url, mode="wait") == 200
            t.join(timeout=30)
            assert result.get("r") is not None, (
                "in-flight request not completed after pause(mode=wait)"
            )

            resp = gen(url, timeout=5)
            assert not ok(resp)
            assert resume(url) == 200

    def test_pause_mode_keep_resumes_frozen_request(self):
        """mode='keep' freezes the request; it must complete after /resume."""
        with server() as url:
            result: dict = {}

            def _bg():
                result["r"] = gen(url, max_tokens=16, timeout=60)

            t = threading.Thread(target=_bg)
            t.start()
            time.sleep(0.3)

            assert pause(url, mode="keep") == 200
            time.sleep(1)

            # request must NOT have completed yet
            assert not ok(result.get("r")), (
                "request completed before resume in mode=keep"
            )

            assert resume(url) == 200
            t.join(timeout=30)
            assert ok(result.get("r")), (
                "request not completed after resume in mode=keep"
            )
