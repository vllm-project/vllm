# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shutil
import signal
import tempfile
from typing import Optional

from vllm.logger import init_logger
from vllm.utils import PlaceholderModule

logger = init_logger(__name__)

SUPPORTED_SCHEMES = ['s3://', 'gs://']

try:
    from runai_model_streamer import list_safetensors as runai_list_safetensors
    from runai_model_streamer import pull_files as runai_pull_files
except (ImportError, OSError):
    # see https://github.com/run-ai/runai-model-streamer/issues/26
    # OSError will be raised on arm64 platform
    runai_model_streamer = PlaceholderModule(
        "runai_model_streamer")  # type: ignore[assignment]
    runai_pull_files = runai_model_streamer.placeholder_attr("pull_files")
    runai_list_safetensors = runai_model_streamer.placeholder_attr(
        "list_safetensors")


def list_safetensors(path: str = "") -> list[str]:
    """
    List full file names from object path and filter by allow pattern.

    Args:
        path: The object storage path to list from.

    Returns:
        list[str]: List of full object storage paths allowed by the pattern
    """
    return runai_list_safetensors(path)


def is_runai_obj_uri(model_or_path: str) -> bool:
    return model_or_path.lower().startswith(tuple(SUPPORTED_SCHEMES))


class ObjectStorageModel:
    """
    A class representing an ObjectStorage model mirrored into a
    temporary directory.

    Attributes:
        dir: The temporary created directory.

    Methods:
        pull_files(): Pull model from object storage to the temporary directory.
    """

    def __init__(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            existing_handler = signal.getsignal(sig)
            signal.signal(sig, self._close_by_signal(existing_handler))

        self.dir = tempfile.mkdtemp()

    def __del__(self):
        self._close()

    def _close(self) -> None:
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)

    def _close_by_signal(self, existing_handler=None):

        def new_handler(signum, frame):
            self._close()
            if existing_handler:
                existing_handler(signum, frame)

        return new_handler

    def pull_files(self,
                   model_path: str = "",
                   allow_pattern: Optional[list[str]] = None,
                   ignore_pattern: Optional[list[str]] = None) -> None:
        """
        Pull files from object storage into the temporary directory.

        Args:
            model_path: The object storage path of the model.
            allow_pattern: A list of patterns of which files to pull.
            ignore_pattern: A list of patterns of which files not to pull.

        """
        if not model_path.endswith("/"):
            model_path = model_path + "/"
        runai_pull_files(model_path, self.dir, allow_pattern, ignore_pattern)
