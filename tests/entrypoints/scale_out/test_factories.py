# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace

from fastapi import FastAPI

from vllm.entrypoints.launchers.api_server.routers import register_api_routers
from vllm.entrypoints.scale_out.token_in_token_out.api_router import attach_router
from vllm.tasks import SupportedTask

RENDER_PATHS = {
    "/v1/chat/completions/render",
    "/v1/chat/completions/derender",
    "/v1/completions/render",
    "/v1/completions/derender",
    "/v1/messages/render",
    "/v1/responses/render",
}
GENERATE_PATH = "/inference/v1/generate"
ABORT_PATH = "/abort_requests"
SCALE_OUT_PATHS = RENDER_PATHS | {GENERATE_PATH, ABORT_PATH}


def registered_paths(
    supported_tasks: tuple[SupportedTask, ...],
    *,
    tokens_only: bool = False,
    enable_scale_out: bool = False,
) -> set[str]:
    app = FastAPI()
    args = Namespace(
        tokens_only=tokens_only,
        enable_scale_out=enable_scale_out,
        enable_fault_tolerance=False,
    )
    app.state.args = args
    register_api_routers(args, app, supported_tasks)
    return {route.path for route in app.routes}


def test_scale_out_routes_are_disabled_by_default():
    paths = registered_paths(("generate",))

    assert SCALE_OUT_PATHS.isdisjoint(paths)


def test_disabled_scale_out_routes_are_logged(caplog):
    with caplog.at_level("INFO", logger="vllm.entrypoints.scale_out.factories"):
        registered_paths(("generate",))

    assert any("--enable-scale-out" in record.message for record in caplog.records)


def test_scale_out_routes_can_be_enabled_for_generate_server():
    paths = registered_paths(("generate",), enable_scale_out=True)

    assert paths >= SCALE_OUT_PATHS


def test_render_routes_remain_enabled_for_render_server():
    paths = registered_paths(("render",))

    assert paths >= RENDER_PATHS
    assert GENERATE_PATH not in paths


def test_tokens_only_mode_enables_generate_routes_when_flag_is_unset():
    paths = registered_paths(("generate",), tokens_only=True)

    assert paths >= {GENERATE_PATH, ABORT_PATH}


def test_abort_route_does_not_depend_on_tokens_only():
    """A decode pool that loads a tokenizer serves generate without
    --tokens-only and still needs /abort_requests."""
    paths = registered_paths(("generate",), enable_scale_out=True, tokens_only=False)

    assert ABORT_PATH in paths


def test_abort_route_is_not_registered_twice():
    """The RLHF dev router registers /abort_requests before the scale-out
    routers, so attaching the scale-out one too would add a shadowed duplicate."""
    app = FastAPI()

    @app.post(ABORT_PATH)
    async def existing_abort_requests():
        return None

    attach_router(app)

    assert [route.path for route in app.routes].count(ABORT_PATH) == 1
