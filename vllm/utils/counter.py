# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading


class Counter:
    def __init__(self, start: int = 0) -> None:
        super().__init__()

        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


class AtomicCounter:
    """An atomic, thread-safe counter"""

    def __init__(self, initial: int = 0) -> None:
        """Initialize a new atomic counter to given initial value"""
        super().__init__()

        self._value = initial
        self._lock = threading.Lock()

    @property
    def value(self) -> int:
        return self._value

    def inc(self, num: int = 1) -> int:
        """Atomically increment the counter by num and return the new value"""
        with self._lock:
            self._value += num
            return self._value

    def dec(self, num: int = 1) -> int:
        """Atomically decrement the counter by num and return the new value"""
        with self._lock:
            self._value -= num
            return self._value
