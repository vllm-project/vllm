# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import time

from vllm.entrypoints.openai.files.protocol import (
    FileDeleteResponse,
    FileObject,
    FilePurpose,
)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import atomic_writer
from vllm.utils import random_uuid

logger = init_logger(__name__)


class OpenAIServingFiles:
    """Stores uploaded and generated JSONL files on local disk."""

    def __init__(self, storage_dir: str) -> None:
        self.files_dir = os.path.join(storage_dir, "files")
        metadata_dir = os.path.join(storage_dir, "metadata")
        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        self._metadata_path = os.path.join(metadata_dir, "files.json")
        self._files: dict[str, FileObject] = {}
        self._load_metadata()

    def file_path(self, file_id: str) -> str:
        return os.path.join(self.files_dir, f"{file_id}.jsonl")

    def allocate_file(self) -> tuple[str, str]:
        """Return a new file id and its path, without registering the file."""
        file_id = f"file-{random_uuid()}"
        return file_id, self.file_path(file_id)

    async def register_file(
        self,
        file_id: str,
        filename: str,
        purpose: FilePurpose,
        num_bytes: int,
    ) -> FileObject:
        """Register a file that was written to the path from `allocate_file`."""
        file_obj = FileObject(
            id=file_id,
            bytes=num_bytes,
            created_at=int(time.time()),
            filename=filename,
            purpose=purpose,
        )
        self._files[file_id] = file_obj
        self._save_metadata()

        logger.info("Stored file %s (%s, %d bytes)", file_id, filename, num_bytes)
        return file_obj

    async def upload_file(
        self,
        content: bytes,
        filename: str,
        purpose: FilePurpose,
    ) -> FileObject:
        file_id, path = self.allocate_file()
        with open(path, "wb") as f:
            f.write(content)
        return await self.register_file(file_id, filename, purpose, len(content))

    async def list_files(self, purpose: FilePurpose | None = None) -> list[FileObject]:
        files = [
            f for f in self._files.values() if purpose is None or f.purpose == purpose
        ]
        return sorted(files, key=lambda f: f.created_at, reverse=True)

    async def get_file(self, file_id: str) -> FileObject | None:
        return self._files.get(file_id)

    async def get_file_content(self, file_id: str) -> bytes | None:
        path = self.file_path(file_id)
        if file_id not in self._files or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    async def delete_file(self, file_id: str) -> FileDeleteResponse | None:
        if file_id not in self._files:
            return None

        path = self.file_path(file_id)
        if os.path.exists(path):
            os.remove(path)

        del self._files[file_id]
        self._save_metadata()

        logger.info("Deleted file %s", file_id)
        return FileDeleteResponse(id=file_id)

    def _load_metadata(self) -> None:
        if not os.path.exists(self._metadata_path):
            return
        with open(self._metadata_path) as f:
            for item in json.load(f):
                file_obj = FileObject.model_validate(item)
                self._files[file_obj.id] = file_obj
        logger.info("Loaded %d files from metadata", len(self._files))

    def _save_metadata(self) -> None:
        with atomic_writer(self._metadata_path) as f:
            json.dump([obj.model_dump() for obj in self._files.values()], f)
