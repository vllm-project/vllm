# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import signal
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Optional

import torch


class BaseConnector(ABC):
    '''
    For fs connector such as s3:
    <connector_type>://<path>/<filename>

    For kv connector such as redis:
    <connector_type>://<host>:<port>/<model_name>/keys/<key>
    <connector_type://<host>:<port>/<model_name>/files/<filename>
    '''

    def __init__(self, url: str, device: torch.device = "cpu"):
        self.url = url
        self.device = device
        self.closed = False
        self.local_dir = tempfile.mkdtemp()
        for sig in (signal.SIGINT, signal.SIGTERM):
            existing_handler = signal.getsignal(sig)
            signal.signal(sig, self._close_by_signal(existing_handler))

    def get_local_dir(self):
        return self.local_dir

    @abstractmethod
    def weight_iterator(
            self,
            rank: int = 0) -> Generator[tuple[str, torch.Tensor], None, None]:
        raise NotImplementedError()

    @abstractmethod
    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:
        raise NotImplementedError()

    def close(self):
        if self.closed:
            return

        self.closed = True
        if os.path.exists(self.local_dir):
            shutil.rmtree(self.local_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def _close_by_signal(self, existing_handler=None):

        def new_handler(signum, frame):
            self.close()
            if existing_handler:
                existing_handler(signum, frame)

        return new_handler


class BaseKVConnector(BaseConnector):

    @abstractmethod
    def get(self, key: str) -> Optional[torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def getstr(self, key: str) -> Optional[str]:
        raise NotImplementedError()

    @abstractmethod
    def set(self, key: str, obj: torch.Tensor) -> None:
        raise NotImplementedError()

    @abstractmethod
    def setstr(self, key: str, obj: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        raise NotImplementedError()


class BaseFileConnector(BaseConnector):
    """
    list full file names from remote fs path and filter by allow pattern.

    Args:
        allow_pattern: A list of patterns of which files to pull.

    Returns:
        list[str]: List of full paths allowed by the pattern
    """

    @abstractmethod
    def glob(self, allow_pattern: str) -> list[str]:
        raise NotImplementedError()
