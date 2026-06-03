"""File management for the OpenAI-compatible Batch API."""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.batch.protocol import (
    FileDeleteResponse,
    FileObject,
)
from vllm.logger import init_logger
from vllm.utils import random_uuid

if TYPE_CHECKING:
    from vllm.entrypoints.openai.serving_batches import OpenAIServingBatches

logger = init_logger(__name__)


class OpenAIServingFiles:
    """Manages file uploads and retrieval for the Batch API.

    Standalone class — does not inherit from OpenAIServing since it has
    no need for an engine client, model config, or tokenizer.
    """

    def __init__(self, storage_dir: str) -> None:
        self.storage_dir = storage_dir
        self.files_dir = os.path.join(storage_dir, "files")
        self.metadata_dir = os.path.join(storage_dir, "metadata")
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        self._files: dict[str, FileObject] = {}
        self._lock = asyncio.Lock()
        self._metadata_path = os.path.join(self.metadata_dir, "files.json")

        self._load_metadata()

    def _load_metadata(self) -> None:
        if os.path.exists(self._metadata_path):
            with open(self._metadata_path) as f:
                data = json.load(f)
            for item in data:
                fo = FileObject.model_validate(item)
                self._files[fo.id] = fo
            logger.info("Loaded %d files from metadata", len(self._files))

    async def _save_metadata(self) -> None:
        data = [fo.model_dump() for fo in self._files.values()]
        with open(self._metadata_path, "w") as f:
            json.dump(data, f)

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        purpose: str,
    ) -> FileObject:
        file_id = f"file-{random_uuid()}"
        file_path = os.path.join(self.files_dir, f"{file_id}.jsonl")

        with open(file_path, "wb") as f:
            f.write(content)

        file_obj = FileObject(
            id=file_id,
            bytes=len(content),
            created_at=int(time.time()),
            filename=filename,
            purpose=purpose,
        )

        async with self._lock:
            self._files[file_id] = file_obj
            await self._save_metadata()

        logger.info("Uploaded file %s (%s, %d bytes)", file_id, filename,
                     len(content))
        return file_obj

    def begin_file(self, filename: str, purpose: str) -> tuple[str, str]:
        """Allocate a file id and on-disk path for streaming writes. The
        file is registered only once ``commit_file`` is called."""
        file_id = f"file-{random_uuid()}"
        file_path = os.path.join(self.files_dir, f"{file_id}.jsonl")
        return file_id, file_path

    async def commit_file(
        self,
        file_id: str,
        filename: str,
        purpose: str,
        num_bytes: int,
    ) -> FileObject:
        """Register a file previously allocated via ``begin_file``."""
        file_obj = FileObject(
            id=file_id,
            bytes=num_bytes,
            created_at=int(time.time()),
            filename=filename,
            purpose=purpose,
        )
        async with self._lock:
            self._files[file_id] = file_obj
            await self._save_metadata()
        logger.info("Committed file %s (%s, %d bytes)", file_id, filename,
                     num_bytes)
        return file_obj

    async def list_files(
        self,
        purpose: str | None = None,
    ) -> list[FileObject]:
        files = list(self._files.values())
        if purpose is not None:
            files = [f for f in files if f.purpose == purpose]
        files.sort(key=lambda f: f.created_at, reverse=True)
        return files

    async def get_file(self, file_id: str) -> FileObject | None:
        return self._files.get(file_id)

    async def get_file_content(self, file_id: str) -> bytes | None:
        if file_id not in self._files:
            return None
        file_path = os.path.join(self.files_dir, f"{file_id}.jsonl")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as f:
            return f.read()

    async def delete_file(
        self,
        file_id: str,
        batch_handler: OpenAIServingBatches | None = None,
    ) -> FileDeleteResponse | dict | None:
        """Delete a file.

        Returns None if the file does not exist (router -> 404), a dict with
        an "error" key if the file is referenced by an active batch
        (router -> 409), or a FileDeleteResponse on success.
        """
        if file_id not in self._files:
            return None

        if (batch_handler is not None
                and batch_handler.is_file_in_active_batch(file_id)):
            return {
                "error": {
                    "message": f"File {file_id} is referenced by an active "
                    "batch and cannot be deleted",
                    "type": "invalid_request_error",
                    "code": 409,
                }
            }

        file_path = os.path.join(self.files_dir, f"{file_id}.jsonl")
        if os.path.exists(file_path):
            os.remove(file_path)

        async with self._lock:
            del self._files[file_id]
            await self._save_metadata()

        logger.info("Deleted file %s", file_id)
        return FileDeleteResponse(id=file_id)
