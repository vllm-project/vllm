# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from typing import IO

from vllm.entrypoints.openai.batches.jsonl_protocol import BatchRequestOutput
from vllm.entrypoints.openai.files.protocol import FilePurpose
from vllm.entrypoints.openai.files.serving import OpenAIServingFiles


class _StreamedFile:
    """A file that is created on first write and registered on close."""

    def __init__(
        self, serving_files: OpenAIServingFiles, filename: str, purpose: FilePurpose
    ) -> None:
        self._serving_files = serving_files
        self._filename = filename
        self._purpose = purpose
        self._file_id: str | None = None
        self._handle: IO[bytes] | None = None
        self._num_bytes = 0

    async def write(self, data: bytes) -> None:
        if self._handle is None:
            self._file_id, path = self._serving_files.allocate_file()
            self._handle = await asyncio.to_thread(open, path, "wb")
        await asyncio.to_thread(self._handle.write, data)
        self._num_bytes += len(data)

    async def close(self) -> str | None:
        if self._handle is None or self._file_id is None:
            return None
        await asyncio.to_thread(self._handle.close)
        file_obj = await self._serving_files.register_file(
            self._file_id, self._filename, self._purpose, self._num_bytes
        )
        return file_obj.id


class BatchResultWriter:
    """Streams batch results to the output and error files as they complete."""

    def __init__(self, serving_files: OpenAIServingFiles, batch_id: str) -> None:
        self._output = _StreamedFile(
            serving_files, f"{batch_id}_output.jsonl", "batch_output"
        )
        self._errors = _StreamedFile(
            serving_files, f"{batch_id}_errors.jsonl", "batch_error"
        )

    async def write(self, output: BatchRequestOutput) -> None:
        target = self._output if output.error is None else self._errors
        await target.write((output.model_dump_json() + "\n").encode())

    async def close(self) -> tuple[str | None, str | None]:
        """Close both files and return their (output, error) file ids."""
        return await self._output.close(), await self._errors.close()
