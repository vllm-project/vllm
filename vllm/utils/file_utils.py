# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File helpers with no dependency on torch or the model executor."""

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from vllm.logger import init_logger

logger = init_logger(__name__)


@contextmanager
def atomic_writer(
    filepath: str | Path, mode: str = "w", encoding: str | None = None
) -> Generator[IO]:
    """Context manager that provides an atomic file writing routine.

    The context manager writes to a temporary file and, if successful,
    atomically replaces the original file.

    Args:
        filepath (str or Path): The path to the file to write.
        mode (str): The file mode for the temporary file (e.g., 'w', 'wb').
        encoding (str): The encoding for text mode.

    Yields:
        file object: A handle to the temporary file.

    """
    # Create a temporary file in the same directory as the target file
    # to ensure it's on the same filesystem for an atomic replace.
    temp_dir = os.path.dirname(filepath)
    temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)

    try:
        # Open the temporary file for writing
        with os.fdopen(temp_fd, mode=mode, encoding=encoding) as temp_file:
            yield temp_file

        # If the 'with' block completes successfully,
        # perform the atomic replace.
        os.replace(temp_path, filepath)

    except Exception:
        logger.exception(
            "Error during atomic write. Original file '%s' not modified", filepath
        )
        raise
    finally:
        # Clean up the temporary file if it still exists.
        if os.path.exists(temp_path):
            os.remove(temp_path)
