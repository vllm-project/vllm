# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections.abc import Callable
from typing import Generic, TypeVar

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ResourceT = TypeVar("ResourceT")

DEFAULT_CLOSE_TIMEOUT_S = 5.0


class AsyncHostBuffer(Generic[ResourceT]):
    """Build a host-backed resource without blocking the main thread.

    ``allocate`` runs on a dedicated thread and must release the GIL during its
    long native operations. The accelerator device is thread-local, so the one selected
    on the main thread is re-applied before ``allocate`` runs.

    The resource stays reachable only from that thread until ``poll`` hands it
    over, so it always has a single owner and teardown can never race a
    half-built resource.

    The thread is a daemon: building can take minutes, and neither shutdown nor
    interpreter exit should wait that long.

    Args:
        allocate: Build and return the resource.
        cleanup: Release a resource that was never adopted.
        thread_name: Name of the initializer thread.
    """

    def __init__(
        self,
        allocate: Callable[[], ResourceT],
        cleanup: Callable[[ResourceT], None],
        thread_name: str,
    ) -> None:
        self._cleanup = cleanup
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._resource: ResourceT | None = None
        self._failed = False
        self._adopted = False
        self._abandoned = False
        self._device = (
            torch.accelerator.current_device_index()
            if torch.accelerator.is_available()
            else None
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(allocate,),
            name=thread_name,
            daemon=True,
        )
        self._thread.start()

    @property
    def failed(self) -> bool:
        return self._failed

    def _run(self, allocate: Callable[[], ResourceT]) -> None:
        resource: ResourceT | None = None
        try:
            try:
                if self._device is not None:
                    torch.accelerator.set_device_index(self._device)
                resource = allocate()
            except BaseException:
                # Catch everything: leaving _done clear would strand every
                # future poll() at "still initializing".
                logger.exception("Asynchronous host-buffer initialization failed")
        finally:
            with self._lock:
                if self._abandoned and resource is not None:
                    # close() stopped waiting for us; nobody will adopt this.
                    self._cleanup_safely(resource)
                else:
                    self._resource = resource
                self._done.set()

    def poll(self) -> ResourceT | None:
        """Take ownership of the resource once built, else None.

        Sets `failed` if the build finished without producing one.
        """
        if self._adopted or self._failed or not self._done.is_set():
            return None
        with self._lock:
            resource = self._resource
            self._resource = None
        if resource is None:
            self._failed = True
            return None
        self._adopted = True
        return resource

    def _cleanup_safely(self, resource: ResourceT) -> None:
        try:
            self._cleanup(resource)
        except Exception:
            logger.exception("Failed to clean up asynchronous host buffer")

    def close(self, timeout: float = DEFAULT_CLOSE_TIMEOUT_S) -> None:
        """Release an unadopted resource, waiting at most ``timeout`` seconds.

        On timeout the initializer is abandoned; it releases the resource itself
        once it finishes, so the caller never blocks on a long-running build.
        """
        if self._adopted:
            return
        if not self._done.wait(timeout):
            with self._lock:
                if not self._done.is_set():
                    self._abandoned = True
                    logger.warning(
                        "Abandoning in-flight host-buffer initialization after "
                        "%.1fs; it will release its own resources.",
                        timeout,
                    )
                    return
        with self._lock:
            resource = self._resource
            self._resource = None
        if resource is not None:
            self._cleanup_safely(resource)
