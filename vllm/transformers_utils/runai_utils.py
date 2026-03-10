# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
import shutil
import signal

from vllm import envs
from vllm.assets.base import get_cache_dir
from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule

logger = init_logger(__name__)

SUPPORTED_SCHEMES = ["s3://", "gs://", "az://"]

try:
    from runai_model_streamer import ObjectStorageModel as RunaiObjectStorageModel
    from runai_model_streamer import list_safetensors as runai_list_safetensors
except ImportError:
    runai_model_streamer = PlaceholderModule("runai_model_streamer")  # type: ignore[assignment]
    runai_list_safetensors = runai_model_streamer.placeholder_attr("list_safetensors")
    RunaiObjectStorageModel = runai_model_streamer.placeholder_attr(
        "ObjectStorageModel"
    )


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
    """Represents a model hosted in object storage, mirrored into a local
    cache directory for loading configuration files (e.g. tokenizer, model
    config). Model weights are not pulled here and are streamed separately.

    Wraps the runai_model_streamer ObjectStorageModel as a context manager,
    which makes it safe for concurrent use by multiple processes on the same
    host. When VLLM_ASSETS_CACHE_MODEL_CLEAN is set, the local cache directory
    is also cleaned up on SIGINT/SIGTERM.

    Attributes:
        dir: Path to the local cache directory where files are pulled to.
    """

    def __init__(self, url: str) -> None:
        if envs.VLLM_ASSETS_CACHE_MODEL_CLEAN:
            for sig in (signal.SIGINT, signal.SIGTERM):
                existing_handler = signal.getsignal(sig)
                signal.signal(sig, self._close_by_signal(existing_handler))

        dir_name = os.path.join(
            get_cache_dir(),
            "model_streamer",
            hashlib.sha256(str(url).encode()).hexdigest()[:8],
        )
        self._runai_obj = RunaiObjectStorageModel(model_path=url, dst=dir_name)
        self.dir = self._runai_obj.dir
        logger.debug("Init object storage, model cache path is: %s", dir_name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._runai_obj.__exit__(exc_type, exc_val, exc_tb)

    def _close(self) -> None:
        # ignore_errors=True avoids the TOCTOU race when multiple processes
        # call _close() concurrently on SIGTERM
        shutil.rmtree(self.dir, ignore_errors=True)

    def _close_by_signal(self, existing_handler=None):
        def new_handler(signum, frame):
            self._close()
            if existing_handler:
                existing_handler(signum, frame)

        return new_handler

    def pull_files(
        self,
        model_path: str = "",
        allow_pattern: list[str] | None = None,
        ignore_pattern: list[str] | None = None,
    ) -> None:
        """Pull files from object storage into the local cache directory.

        Args:
            model_path: The object storage path of the model (unused;
                the path is set at construction time).
            allow_pattern: File patterns to include (e.g. ["*.json"]).
            ignore_pattern: File patterns to exclude.
        """
        self._runai_obj.pull_files(allow_pattern, ignore_pattern)
