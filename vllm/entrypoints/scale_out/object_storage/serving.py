# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import AsyncGenerator

from fastapi import HTTPException, UploadFile
from starlette.responses import StreamingResponse

from vllm.renderers.paged_shm.client_async import AsyncPagedShmClient


class ServingObjectStorage:
    """
    Asynchronous object storage service backed by PagedShm shared memory.

    Provides upload/download/delete/info endpoints.
    """

    def __init__(self, shm_server_address: str):
        self.client = AsyncPagedShmClient(shm_server_address, pin=False)

    async def upload(self, file: UploadFile, uuid: str | None = None):
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
                uuid.UUID(uuid)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid UUID format"
                ) from None
            uid = uuid
        else:
            uid = str(uuid.uuid4())

        try:
            await self.client.write(uid, content)
        except RuntimeError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to write to shared memory: {e}"
            ) from None

        return {"uuid": uid}

    async def download(self, uuid: str):
        """Stream the object identified by the given UUID."""

        async def stream_data() -> AsyncGenerator[bytes, None]:
            try:
                async with self.client.get_iterator_numpy(uuid) as it:
                    for array, offset in it:
                        yield array[:offset].tobytes()
            except RuntimeError as e:
                # Map server-side "not found" errors to 404
                if "not found" in str(e).lower():
                    raise HTTPException(
                        status_code=404, detail="Object not found"
                    ) from None
                raise HTTPException(status_code=500, detail=str(e)) from None

        return StreamingResponse(
            stream_data(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{uuid}"'},
        )

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

    async def info(self, uuid: str):
        """Return metadata (currently only size) for the given UUID."""
        try:
            # Temporarily acquire a read lock to obtain the object size
            ctx = self.client.read_context(uuid)
            async with ctx:
                size = ctx.size
            return {"uuid": uuid, "size": size}
        except RuntimeError as e:
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=404, detail="Object not found"
                ) from None
            raise HTTPException(status_code=500, detail=str(e)) from None

    async def close(self):
        """Release resources; call during application shutdown."""
        await self.client.close()
