# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Private, one-shot file control for an external startup capturer."""

import asyncio
import json
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from vllm.logger import init_logger
from vllm.snapshot.manifest import _write_json_atomic, validate_artifact_root
from vllm.snapshot.types import Oracle

logger = init_logger(__name__)


class Activation(BaseModel):
    """Fresh deployment context, supplied after native CUDA restore/unlock."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1]
    capture_id: str
    activation_id: str = Field(min_length=1, max_length=128, pattern=r"^[\w.-]+$")
    kind: Literal["donor", "copy"]
    native_complete: bool
    host: str | None
    port: int = Field(ge=1, le=65535)


class FileSnapshotControl:
    """The capturer provides a fresh private directory for every activation.

    It must capture before writing activation.json. A challenge generated
    afterward prevents a release saved by an earlier copy from being replayed.
    The capturer owns artifact compatibility, namespace/stdio restoration and
    native completion. This protocol does not invoke or retry native recovery.
    """

    def __init__(self, directory: Path, timeout_s: float):
        validate_artifact_root(directory, creating=False)
        if any(directory.iterdir()):
            raise ValueError("startup snapshot control directory must be empty")
        self.directory = directory
        self.timeout_s = timeout_s
        self.capture_id = uuid4().hex
        self.activation: Activation | None = None
        self._wait_started = False

    def publish_ready(self, oracle: Oracle) -> None:
        """Publish only after all worker preparation and rehearsal complete."""
        _write_json_atomic(
            self.directory / "capture-ready.json",
            {
                "version": 1,
                "capture_id": self.capture_id,
                "pid": os.getpid(),
                "oracle": asdict(oracle),
            },
        )

    def _read(self, name: str) -> dict:
        validate_artifact_root(self.directory, creating=False)
        path = self.directory / name
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"snapshot control must be a regular file: {name}")
        with path.open("rb") as source:
            data = source.read(65537)
        if len(data) > 65536:
            raise ValueError(f"snapshot control is too large: {name}")
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError(f"snapshot control must be a JSON object: {name}")
        return payload

    async def _wait(self, name: str) -> dict:
        # Frozen time is owned by the capturer's deadline, not a deadline
        # captured in the event loop. Bound this instance's polling budget.
        remaining = self.timeout_s
        while True:
            try:
                return self._read(name)
            except FileNotFoundError:
                if remaining <= 0:
                    raise TimeoutError(
                        f"snapshot control timed out waiting for {name}"
                    ) from None
                delay = min(0.05, remaining)
                await asyncio.sleep(delay)
                remaining -= delay

    async def wait_for_activation(self) -> Activation:
        """Require fresh native completion and a response to a new challenge."""
        if self.activation is not None:
            return self.activation
        if self._wait_started:
            raise RuntimeError("snapshot activation wait already started")
        self._wait_started = True
        payload = await self._wait("activation.json")
        activation = Activation.model_validate(payload)
        if activation.capture_id != self.capture_id or not activation.native_complete:
            raise ValueError("snapshot activation does not match a completed capture")
        for name in ("challenge.json", "release.json", "status.json", "error.json"):
            if os.path.lexists(self.directory / name):
                raise ValueError(f"stale snapshot activation control: {name}")
        challenge = {
            "version": 1,
            "capture_id": self.capture_id,
            "activation_id": activation.activation_id,
            "nonce": uuid4().hex,
        }
        _write_json_atomic(self.directory / "challenge.json", challenge)
        release = await self._wait("release.json")
        if release != challenge or self._read("activation.json") != payload:
            raise ValueError(
                "snapshot release does not match this activation challenge"
            )
        self.activation = activation
        return activation

    def write_status(self, phase: str) -> None:
        """Record recovery progress; validated is not HTTP serving readiness."""
        assert self.activation is not None
        _write_json_atomic(
            self.directory / "status.json",
            {
                "capture_id": self.capture_id,
                "activation_id": self.activation.activation_id,
                "phase": phase,
            },
            overwrite=True,
        )

    def write_error(self, phase: str, error: BaseException) -> None:
        """Retain the primary failure even if restored standard streams fail."""
        try:
            _write_json_atomic(
                self.directory / "error.json",
                {
                    "capture_id": self.capture_id,
                    "activation_id": (
                        self.activation.activation_id if self.activation else None
                    ),
                    "phase": phase,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                overwrite=True,
            )
        except Exception:
            logger.exception("Could not record startup snapshot failure")
