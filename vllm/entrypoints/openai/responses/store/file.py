# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import json
import os
import tempfile
from pathlib import Path

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesResponse,
    serialize_message,
)
from vllm.entrypoints.openai.responses.store.base import ResponsesStore
from vllm.logger import init_logger

logger = init_logger(__name__)


class FileResponsesStore(ResponsesStore):
    """File-based store that persists each response and message set as an
    individual JSON file in a directory. No in-memory cache — every read
    goes to disk, so multiple instances sharing a directory see each
    other's writes immediately.

    Directory layout::

        <base_path>/
        ├── responses/
        │   ├── <response_id>.json
        │   └── ...
        └── messages/
            ├── <response_id>.json
            └── ...
    """

    def __init__(self, base_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._base_path = Path(base_path)
        self._responses_dir = self._base_path / "responses"
        self._messages_dir = self._base_path / "messages"
        self._responses_dir.mkdir(parents=True, exist_ok=True)
        self._messages_dir.mkdir(parents=True, exist_ok=True)
        logger.info("FileResponsesStore initialized at %s", self._base_path)

    def _response_path(self, response_id: str) -> Path:
        return self._responses_dir / f"{response_id}.json"

    def _messages_path(self, response_id: str) -> Path:
        return self._messages_dir / f"{response_id}.json"

    @staticmethod
    def _read_file(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    @staticmethod
    def _write_file_atomic(path: Path, data: str) -> None:
        """Write data to a file atomically using tmp + rename."""
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp", prefix=".store_"
        )
        try:
            os.write(fd, data.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            os.rename(tmp_path, path)
        except BaseException:
            os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    async def get_response(self, response_id: str) -> ResponsesResponse | None:
        path = self._response_path(response_id)
        data = await asyncio.to_thread(self._read_file, path)
        if data is None:
            return None
        return ResponsesResponse.model_validate_json(data)

    async def put_response(self, response_id: str, response: ResponsesResponse) -> None:
        path = self._response_path(response_id)
        data = response.model_dump_json()
        await asyncio.to_thread(self._write_file_atomic, path, data)

    async def get_messages(
        self, response_id: str
    ) -> list[ChatCompletionMessageParam] | None:
        path = self._messages_path(response_id)
        data = await asyncio.to_thread(self._read_file, path)
        if data is None:
            return None
        return self._deserialize_messages(json.loads(data))

    async def put_messages(
        self,
        response_id: str,
        messages: list[ChatCompletionMessageParam],
    ) -> None:
        path = self._messages_path(response_id)
        data = json.dumps([serialize_message(m) for m in messages], ensure_ascii=False)
        await asyncio.to_thread(self._write_file_atomic, path, data)
