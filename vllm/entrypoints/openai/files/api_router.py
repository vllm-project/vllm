# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus
from typing import Annotated, Literal

from fastapi import APIRouter, FastAPI, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from vllm.entrypoints.openai.batches.serving import OpenAIServingBatches
from vllm.entrypoints.openai.files.protocol import FileListResponse, FilePurpose
from vllm.entrypoints.openai.files.serving import OpenAIServingFiles
from vllm.entrypoints.serve import create_error_response

router = APIRouter()


def files(request: Request) -> OpenAIServingFiles:
    return request.app.state.openai_serving_files


def batches(request: Request) -> OpenAIServingBatches:
    return request.app.state.openai_serving_batches


def file_not_found(file_id: str) -> JSONResponse:
    error = create_error_response(
        f"File {file_id} not found", "NotFoundError", HTTPStatus.NOT_FOUND
    )
    return JSONResponse(content=error.model_dump(), status_code=error.error.code)


@router.post("/v1/files")
async def upload_file(
    file: UploadFile,
    raw_request: Request,
    purpose: Annotated[Literal["batch"], Form()] = "batch",
):
    content = await file.read()
    file_obj = await files(raw_request).upload_file(
        content, file.filename or "upload.jsonl", purpose
    )
    return JSONResponse(content=file_obj.model_dump())


@router.get("/v1/files")
async def list_files(raw_request: Request, purpose: FilePurpose | None = None):
    file_objs = await files(raw_request).list_files(purpose)
    return JSONResponse(content=FileListResponse(data=file_objs).model_dump())


@router.get("/v1/files/{file_id}")
async def get_file(file_id: str, raw_request: Request):
    file_obj = await files(raw_request).get_file(file_id)
    if file_obj is None:
        return file_not_found(file_id)
    return JSONResponse(content=file_obj.model_dump())


@router.get("/v1/files/{file_id}/content")
async def get_file_content(file_id: str, raw_request: Request):
    content = await files(raw_request).get_file_content(file_id)
    if content is None:
        return file_not_found(file_id)
    return Response(content=content, media_type="application/octet-stream")


@router.delete("/v1/files/{file_id}")
async def delete_file(file_id: str, raw_request: Request):
    if batches(raw_request).is_file_in_active_batch(file_id):
        error = create_error_response(
            f"File {file_id} is referenced by an active batch and cannot be deleted",
            "ConflictError",
            HTTPStatus.CONFLICT,
        )
        return JSONResponse(content=error.model_dump(), status_code=error.error.code)

    result = await files(raw_request).delete_file(file_id)
    if result is None:
        return file_not_found(file_id)
    return JSONResponse(content=result.model_dump())


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
