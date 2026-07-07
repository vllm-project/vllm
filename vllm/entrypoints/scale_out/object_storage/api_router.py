# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, Request

from .serving import ServingObjectStorage

router = APIRouter()


def object_storage(request: Request) -> ServingObjectStorage:
    handler = getattr(request.app.state, "serving_object_storage", None)
    if handler is None:
        raise NotImplementedError("")

    return handler


@router.put("/object_storage/")
async def object_storage_upload(raw_request: Request):
    handler = object_storage(raw_request)
    return await handler.upload()


@router.put("/object_storage/{id}")
async def object_storage_upload(raw_request: Request):
    handler = object_storage(raw_request)
    return await handler.upload()


@router.get("/object_storage/{id}/")
async def object_storage_download(raw_request: Request):
    handler = object_storage(raw_request)
    return await handler.download()


@router.delete("/object_storage/{id}/")
async def object_storage_delete(raw_request: Request):
    handler = object_storage(raw_request)
    return await handler.delete()


@router.head("/object_storage/{id}/")
async def object_storage_info(raw_request: Request):
    handler = object_storage(raw_request)
    return await handler.info()
