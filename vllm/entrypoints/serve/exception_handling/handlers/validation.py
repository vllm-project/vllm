# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

import regex as re
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse
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


# Pydantic repeats the whole offending value under `input` in *every*
# error entry, so one malformed request whose input fails per-element
# echoes that input once per error. Measured on `/v1/responses`: a
# 4,475-byte body produced 12,001 errors and a 23.6 MB response, an
# amplification of ~5,300x that scales linearly with the request size.
# Bound both the number of entries reported and the value echoed in each.
_MAX_REPORTED_ERRORS = 10
_MAX_ERROR_INPUT_CHARS = 200
_MAX_ERROR_CHARS = 1000


def _summarize_error_input(value: object) -> object:
    """A size-bounded stand-in for a validation error's `input` value.

    Containers are described rather than rendered: `repr()` of a large
    list materializes the whole string first, which is the cost this is
    meant to avoid.
    """
    if isinstance(value, str):
        if len(value) > _MAX_ERROR_INPUT_CHARS:
            return value[:_MAX_ERROR_INPUT_CHARS] + "...[truncated]"
        return value
    if isinstance(value, (bytes, bytearray)):
        return f"<{type(value).__name__} of {len(value)} bytes>"
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return f"<{type(value).__name__} of {len(value)} items>"
    try:
        text = str(value)
    except Exception:
        # e.g. an int past `sys.get_int_max_str_digits()`.
        return f"<{type(value).__name__}>"
    if len(text) > _MAX_ERROR_INPUT_CHARS:
        return text[:_MAX_ERROR_INPUT_CHARS] + "...[truncated]"
    return text


def _format_error(err: object) -> str:
    """Render one validation error, bounded in size."""
    if isinstance(err, dict):
        err = dict(err)
        if "input" in err:
            err["input"] = _summarize_error_input(err["input"])
        loc = err.get("loc")
        if isinstance(loc, (tuple, list)):
            # For a union-typed field pydantic spells every branch out in
            # `loc`, which is ~800 characters of type names per entry here.
            # `param` is already cleaned this way.
            err["loc"] = clean_loc_for_param(tuple(loc))
    text = str(err)
    if len(text) > _MAX_ERROR_CHARS:
        text = text[:_MAX_ERROR_CHARS] + "...[truncated]"
    return text


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
        reported = errors[:_MAX_REPORTED_ERRORS]
        message = f"{count} validation {label}:\n"
        message += "".join(f"  {_format_error(err)}\n" for err in reported)
        if count > len(reported):
            message += f"  ...and {count - len(reported)} more {label}\n"
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
