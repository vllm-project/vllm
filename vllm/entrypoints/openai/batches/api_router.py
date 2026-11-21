# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State

from vllm.entrypoints.openai.batches.protocol import BatchCreateRequest
from vllm.entrypoints.openai.batches.serving import OpenAIServingBatches
from vllm.entrypoints.openai.files.serving import OpenAIServingFiles
from vllm.entrypoints.serve import create_error_response
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.tasks import POOLING_TASKS, SupportedTask


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    serving_batches: OpenAIServingBatches = app.state.openai_serving_batches
    serving_batches.start()
    yield
    await serving_batches.shutdown()


router = APIRouter(lifespan=lifespan)


def batches(request: Request) -> OpenAIServingBatches:
    return request.app.state.openai_serving_batches


def error_json(error: ErrorResponse) -> JSONResponse:
    return JSONResponse(content=error.model_dump(), status_code=error.error.code)


def batch_not_found(batch_id: str) -> JSONResponse:
    return error_json(
        create_error_response(
            f"Batch {batch_id} not found", "NotFoundError", HTTPStatus.NOT_FOUND
        )
    )


@router.post("/v1/batches")
async def create_batch(request: BatchCreateRequest, raw_request: Request):
    result = await batches(raw_request).create_batch(request)
    if isinstance(result, ErrorResponse):
        return error_json(result)
    return JSONResponse(content=result.model_dump())


@router.get("/v1/batches")
async def list_batches(
    raw_request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    after: str | None = None,
):
    result = await batches(raw_request).list_batches(limit=limit, after=after)
    return JSONResponse(content=result.model_dump())


@router.get("/v1/batches/{batch_id}")
async def get_batch(batch_id: str, raw_request: Request):
    result = await batches(raw_request).get_batch(batch_id)
    if result is None:
        return batch_not_found(batch_id)
    return JSONResponse(content=result.model_dump())


@router.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str, raw_request: Request):
    result = await batches(raw_request).cancel_batch(batch_id)
    if result is None:
        return batch_not_found(batch_id)
    if isinstance(result, ErrorResponse):
        return error_json(result)
    return JSONResponse(content=result.model_dump())


def attach_router(app: FastAPI) -> None:
    app.include_router(router)


def init_batch_state(
    state: State, args: Namespace, supported_tasks: tuple[SupportedTask, ...]
) -> None:
    supports_pooling = any(task in POOLING_TASKS for task in supported_tasks)
    serving_files = OpenAIServingFiles(storage_dir=args.batch_storage_dir)
    state.openai_serving_files = serving_files
    state.openai_serving_batches = OpenAIServingBatches(
        storage_dir=args.batch_storage_dir,
        serving_files=serving_files,
        serving_chat=(
            state.openai_serving_chat if "generate" in supported_tasks else None
        ),
        serving_embedding=state.serving_embedding if supports_pooling else None,
        serving_scores=state.serving_scores if supports_pooling else None,
        batch_priority=args.batch_priority,
        retention_hours=args.batch_retention_hours,
    )
