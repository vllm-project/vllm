# SPDX-License-Identifier: Apache-2.0
"""
Interface for storage controllers

Storage controllers are separate modules/threads that sees the L1 Manager
and can operate on it.
"""

# Standard
from abc import ABC, abstractmethod


class StorageControllerInterface(ABC):
    @abstractmethod
    def start(self):
        """
        Start the storage controller.
        This function should be implemented by subclasses to start
        any necessary threads or processes.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop the storage controller.
        This function should be implemented by subclasses to stop
        any running threads or processes.
        """
        pass

    @abstractmethod
    def report_status(self) -> dict:
        """
        Return a status dict for this controller.

        Must include at least ``is_healthy: bool``.
        """
        pass
