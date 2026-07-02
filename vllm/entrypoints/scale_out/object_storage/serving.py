# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from starlette.responses import StreamingResponse


class ServingObjectStorage:
    def __init__(self) -> None:
        self.client = None

    async def upload(self, file: UploadFile = File(...)):
        content = await file.read()

        return

    async def download(self, uuid: str):
        try:
            response =
            return StreamingResponse(

                media_type=response.headers.get("Content-Type", "application/octet-stream"),
                headers={"Content-Disposition": f"attachment; filename={uuid}"}
            )
        except :
            if True:
                raise HTTPException(status_code=404, detail="对象不存在")
            raise HTTPException(status_code=500, detail=str(e))

    async def delete(self, uuid: str):
        pass


    async def info(self, uuid: str):
        pass

