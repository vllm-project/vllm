# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Generic, Optional, TypeVar
import threading

# First Party
from lmcache import torch_dev
from lmcache.utils import lmcache_deprecate
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

T = TypeVar("T")


class MessagingFuture(Generic[T]):
    def __init__(self):
        self.is_done_ = threading.Event()
        self.result_ = None

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        return self.is_done_.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            bool: True if the future is done, False if the timeout was reached.
        """
        return self.is_done_.wait(timeout)

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            raise LMCacheTimeoutError("Future result not available within timeout")
        return self.result_

    def set_result(self, result: T) -> None:
        """
        Set the result of the future and mark it as done. This function is NOT
        SUPPOSED TO BE CALLED by users directly. It should be only called by
        the messaging system when the result is available.

        Args:
            result (T): The result to set.
        """
        self.result_ = result
        self.is_done_.set()

    def to_device_future(
        self,
        device: Any | None = None,
    ) -> "DeviceMessagingFuture":
        """Wrap this future in a device-aware future.

        Args:
            device: The device whose event backend orders completion. Defaults
                to the active device.

        Returns:
            A DeviceMessagingFuture pending on both this future and the event.
        """
        # TODO: need extra type checking for the future type
        return DeviceMessagingFuture.FromMessagingFuture(self, device)  # type: ignore

    @lmcache_deprecate("Use to_device_future() instead")
    def to_cuda_future(
        self,
        device: Any | None = None,
    ) -> "DeviceMessagingFuture[T]":
        """Return a device-aware future using the deprecated CUDA name.

        Args:
            device: Device on which the completion event will be imported.

        Returns:
            A device-aware future wrapping this messaging future.
        """
        return self.to_device_future(device)


class DeviceMessagingFuture(MessagingFuture[T]):
    """
    The future class that wraps both a result and a device IPC event.
    The `query`, `wait`, and `result` methods pend on both the original
    future and the device event, ordered through the platform event backend.
    The original future should return tuple[bytes, T], where the first
    element is the serialized device event handle.
    """

    def __init__(
        self,
        raw_future: MessagingFuture[tuple[bytes, T]],
        device: Any | None = None,
    ) -> None:
        super().__init__()
        self.raw_future_ = raw_future
        self.event_: Any | None = None
        self.result_: T | None = None
        self.device_ = device if device is not None else torch_dev.current_device()
        self._event_backend = get_event_ipc_backend(self.device_)
        self._event_backend.check_event_support(self.device_)

    def _on_raw_future_complete(self) -> None:
        """
        Update the device event and result when the raw future is complete.
        """
        event_bytes, result = self.raw_future_.result()
        self.result_ = result

        self.event_ = self._event_backend.import_event(event_bytes, self.device_)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done, ordered through the device event.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the device event. (NOTE: this could be improved
                with careful threading management)

        Returns:
            bool: True if the future is done, False if the timeout was reached.

        Notes:
            This function does not support waiting for a specific time.
        """
        if self.event_:
            self._event_backend.synchronize_event(self.event_, self.device_)
            return True

        flag = self.raw_future_.wait(timeout)
        if not flag:
            return False

        self._on_raw_future_complete()

        assert self.event_ is not None
        self._event_backend.synchronize_event(self.event_, self.device_)

        return True

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the device event. (NOTE: this could be improved
                with careful threading management)

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            raise LMCacheTimeoutError(
                "DeviceMessagingFuture result not available within timeout"
            )

        assert self.result_ is not None
        return self.result_

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        if self.event_:
            return self._event_backend.query_event(self.event_)

        if self.raw_future_.query():
            self._on_raw_future_complete()
            assert self.event_ is not None
            return self._event_backend.query_event(self.event_)

        return False

    def set_result(self, result: T) -> None:
        raise NotImplementedError(
            "DeviceMessagingFuture does not support set_result directly"
        )

    @staticmethod
    def FromMessagingFuture(
        raw_future: MessagingFuture[tuple[bytes, T]],
        device: Any | None = None,
    ) -> "DeviceMessagingFuture[T]":
        return DeviceMessagingFuture(raw_future, device)


# Backward-compatible alias for existing imports.
CUDAMessagingFuture = DeviceMessagingFuture
