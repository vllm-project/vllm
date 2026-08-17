# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

import regex as re
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

from ..utils import sanitize_message

logger = init_logger(__name__)


_BRACKETED_INTERNAL_RE = re.compile(r"[\[\]{}()]")

# NOTE: this list is pydantic-core's internal schema-kind vocabulary,
# not a stable public API -- it can grow when pydantic-core adds new
# wrapper/validator kinds. To refresh it after a pydantic upgrade:
#   1. Fuzz the validation-error-prone endpoints (e.g. /tokenize,
#      /v1/completions, /v1/chat/completions) with deliberately
#      malformed values for union-typed and wrapped fields (e.g. `stop`,
#      `prompt`), and inspect the raw `loc` tuples in the response.
#   2. Any *unbracketed* segment that isn't a real field name or list
#      index is a new internal marker -- add it here. Bracketed/
#      parenthesized markers (e.g. "list[...]", "function-wrap[...]")
#      are already caught structurally by _BRACKETED_INTERNAL_RE and
#      don't need a list entry.
#   3. pydantic-core's source (the `error.rs`/schema-kind definitions
#      in the pydantic-core Rust crate) is the canonical reference if
#      you want to check before it shows up in a live fuzz run.
_INTERNAL_LOC_MARKERS = frozenset(
    {
        "function-wrap",
        "function-after",
        "function-before",
        "function-plain",
        "json-or-python",
        "lax-or-strict",
        "chain",
        "default",
        "nullable",
        "tagged-union",
        "union",
        "call",
        "arguments",
        "is-instance",
        "is-subclass",
        "callable",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "list",
        "tuple",
        "dict",
        "set",
        "frozenset",
        "complex",
        "none",
        "nonetype",
    }
)


def _is_internal_loc_segment(segment: str) -> bool:
    """True if `segment` is a Pydantic-internal wrapper/union-branch
    marker rather than a user-meaningful field name or list index."""
    if _BRACKETED_INTERNAL_RE.search(segment):
        return True
    return segment.lower() in _INTERNAL_LOC_MARKERS


def clean_loc_for_param(loc: tuple) -> str:
    """Join a Pydantic error `loc` tuple into a clean dotted `param`
    path, dropping internal wrapper/union-branch markers that don't
    correspond to a real field name an API consumer would recognize.

    E.g. ('body', 'function-wrap[__log_extra_fields__()]', 'prompt')
    -> "body.prompt", not "body.function-wrap[__log_extra_fields__()].prompt".
    """
    parts = [str(p) for p in loc if not _is_internal_loc_segment(str(p))]
    if not parts:
        return ".".join(str(p) for p in loc)
    return ".".join(parts)


async def validation_exception_handler(req: Request, exc: RequestValidationError):
    if req.app.state.args.log_error_stack:
        logger.exception(
            "RequestValidationError caught. Request id: %s",
            req.state.request_metadata.request_id
            if hasattr(req.state, "request_metadata")
            else None,
        )

    param = None
    errors = exc.errors()
    for error in errors:
        if "ctx" in error and "error" in error["ctx"]:
            ctx_error = error["ctx"]["error"]
            if isinstance(ctx_error, VLLMValidationError):
                param = ctx_error.parameter
                break

    if param is None and errors:
        first_error = errors[0]
        loc = first_error.get("loc") if isinstance(first_error, dict) else None
        if loc:
            param = clean_loc_for_param(loc)

    # Build the message from exc.errors() instead of str(exc) - str(exc)
    # leaks the server's file path via FastAPI's endpoint context.
    if errors:
        count = len(errors)
        label = "error" if count == 1 else "errors"
        message = f"{count} validation {label}:\n"
        message += "".join(f"  {err}\n" for err in errors)
        message = message.rstrip()
    else:
        message = "Validation error"

    err = ErrorResponse(
        error=ErrorInfo(
            message=sanitize_message(message),
            type=HTTPStatus.BAD_REQUEST.phrase,
            code=HTTPStatus.BAD_REQUEST,
            param=param,
        )
    )
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)
