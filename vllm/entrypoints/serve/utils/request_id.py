# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from logging import Logger, LoggerAdapter

from fastapi import Request

from vllm.logger import bind_external_request_id


def get_external_request_id(request: Request | None) -> str | None:
    """Return the canonical ID only after serving has assigned it."""
    if request is None:
        return None
    metadata = getattr(request.state, "request_metadata", None)
    return metadata.request_id if metadata is not None else None


def bind_external_request_id_from_request(
    logger: Logger, request: Request | None
) -> LoggerAdapter:
    """Bind the ID assigned by serving, if the request reached that point."""
    return bind_external_request_id(logger, get_external_request_id(request))
