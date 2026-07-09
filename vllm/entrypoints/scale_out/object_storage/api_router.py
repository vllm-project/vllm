# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, HTTPException, Request, UploadFile

from .serving import ServingObjectStorage

router = APIRouter()


def get_object_storage(request: Request) -> ServingObjectStorage:
    handler = getattr(request.app.state, "serving_object_storage", None)
    if handler is None:
        raise HTTPException(
            status_code=503, detail="Object storage service is not initialized"
        )
    return handler


@router.put("/object_storage")
async def upload_auto(
    raw_request: Request,
    file: UploadFile,
):
    storage = get_object_storage(raw_request)
    return await storage.upload(file=file)


@router.put("/object_storage/{uuid}")
async def upload_with_uuid(
    uuid: str,
    raw_request: Request,
    file: UploadFile,
):
    storage = get_object_storage(raw_request)
    return await storage.upload(file=file, uuid=uuid)


@router.get("/object_storage/{uuid}")
async def download(
    raw_request: Request,
    uuid: str,
):
    """
    Download the object identified by the given UUID.
    """
    storage = get_object_storage(raw_request)
    return await storage.download(uuid)


@router.delete("/object_storage/{uuid}")
async def delete(
    raw_request: Request,
    uuid: str,
):
    """
    Delete the object identified by the given UUID.
    """
    storage = get_object_storage(raw_request)
    return await storage.delete(uuid)


@router.head("/object_storage/{uuid}")
async def info(
    raw_request: Request,
    uuid: str,
):
    """
    Retrieve metadata (size) of the object identified by the UUID.
    """
    storage = get_object_storage(raw_request)
    return await storage.info(uuid)
