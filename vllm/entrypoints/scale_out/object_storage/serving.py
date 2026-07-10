# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import AsyncGenerator
from uuid import UUID, uuid4

from fastapi import HTTPException, Response, UploadFile
from starlette.responses import StreamingResponse

from vllm.renderers.paged_shm.client_async import AsyncPagedShmClient

from .protocol import UUIDResponse


class ServingObjectStorage:
    """
    Asynchronous object storage service backed by PagedShm shared memory.

    Provides upload/download/delete/info endpoints.
    """

    def __init__(self, shm_server_address: str):
        self.client = AsyncPagedShmClient(shm_server_address, pin=False)

    async def upload(self, file: UploadFile, uuid: str | None = None) -> UUIDResponse:
        """
        Upload a file to shared memory.

        If a UUID is provided, it is used as the object key (overwriting any
        existing object with that key). If not provided, a new UUID is generated.
        """
        content = await file.read()

        # Use provided UUID or generate a new one
        if uuid is not None:
            try:
                # Validate UUID format
                UUID(uuid)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid UUID format"
                ) from None
            uid = uuid
        else:
            uid = str(uuid4())

        try:
            size = await self.client.write(uid, content)
        except RuntimeError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to write to shared memory: {e}"
            ) from None

        return UUIDResponse(uuid=uid, size=size)

    async def download(self, uuid: str) -> StreamingResponse:
        """Stream the object identified by the given UUID."""

        try:
            async with self.client.get_iterator_numpy(uuid) as it:

                async def stream_data() -> AsyncGenerator[bytes, None]:
                    for array, offset in it:
                        yield array[:offset].tobytes()

                return StreamingResponse(
                    stream_data(),
                    media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{uuid}"'},
                )
        except RuntimeError as e:
            # Map server-side "not found" errors to 404
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=404, detail="Object not found"
                ) from None
            raise HTTPException(status_code=500, detail=str(e)) from None

    async def delete(self, uuid: str):
        """Delete the object with the specified UUID."""
        try:
            await self.client.delete(uuid)
        except RuntimeError as e:
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=404, detail="Object not found"
                ) from None
            raise HTTPException(status_code=500, detail=str(e)) from None

    async def info(self, uuid: str) -> Response:
        """Return info for the given UUID."""
        try:
            info = await self.client.get_info(uuid)
            return Response(headers={k: str(v) for k, v in info.items()})
        except RuntimeError as e:
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=404, detail="Object not found"
                ) from None
            raise HTTPException(status_code=500, detail=str(e)) from None

    async def close(self):
        """Release resources; call during application shutdown."""
        await self.client.close()
