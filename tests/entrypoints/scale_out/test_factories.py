# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.cli_args import BaseFrontendArgs
from vllm.entrypoints.scale_out.factories import register_scale_out_api_routers
from vllm.utils.argparse_utils import FlexibleArgumentParser

GENERATE_ENDPOINT = "/inference/v1/generate"
RENDER_ENDPOINT = "/v1/completions/render"
DERENDER_ENDPOINT = "/v1/completions/derender"


@pytest.mark.parametrize(
    ("cli_args", "generate_route_mounted"),
    [
        ([], False),
        (["--enable-scale-out-disaggregation"], True),
    ],
)
def test_disaggregated_generate_endpoint_is_opt_in(
    cli_args: list[str], generate_route_mounted: bool
):
    parser = BaseFrontendArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(cli_args)
    app = FastAPI()
    app.state.args = args

    register_scale_out_api_routers(
        app,
        ("generate",),
        enable_scale_out_disaggregation=args.enable_scale_out_disaggregation,
    )

    route_paths = {route.path for route in app.routes}
    assert (GENERATE_ENDPOINT in route_paths) is generate_route_mounted
    assert RENDER_ENDPOINT in route_paths
    assert DERENDER_ENDPOINT in route_paths
    if not generate_route_mounted:
        assert TestClient(app).post(GENERATE_ENDPOINT).status_code == 404
