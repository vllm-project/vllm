# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep discovery heartbeats independent of the model worker's GIL."""

import json
import logging
import select
import subprocess
import sys
from typing import Any

import msgpack
import zmq

logger = logging.getLogger(__name__)


class MoRIIOHeartbeat:
    def __init__(
        self,
        endpoint: str,
        payload: dict[str, Any],
        interval: float,
        max_retries: int = 100,
    ):
        self._process = subprocess.Popen(
            [sys.executable, __file__, endpoint, str(interval), str(max_retries)],
            stdin=subprocess.PIPE,
            text=True,
        )
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
        except BaseException:
            self.shutdown()
            raise

    def shutdown(self) -> None:
        assert self._process.stdin is not None
        try:
            if not self._process.stdin.closed:
                self._process.stdin.close()
        except BrokenPipeError:
            pass
        finally:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=1)


def main() -> None:
    endpoint, interval_arg, max_retries_arg = sys.argv[1:]
    initial = sys.stdin.readline()
    if not initial:
        return
    payload = msgpack.dumps(json.loads(initial))
    interval = float(interval_arg)
    max_retries = int(max_retries_arg)
    failures = 0
    with zmq.Context() as context, context.socket(zmq.DEALER) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDTIMEO, 1000)
        sock.connect(endpoint)
        while not select.select([sys.stdin], [], [], 0)[0]:
            try:
                sock.send(payload)
                failures = 0
            except zmq.ZMQError:
                failures += 1
                logger.exception("MoRIIO discovery heartbeat send failed")
                if failures >= max_retries:
                    raise
            # EOF also stops the child when the worker exits without cleanup.
            if select.select([sys.stdin], [], [], interval)[0]:
                break


if __name__ == "__main__":
    main()
