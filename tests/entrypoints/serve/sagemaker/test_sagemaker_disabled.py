# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--disable-sagemaker-integration` leaves the SageMaker surface out."""

import sys
from http import HTTPStatus

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm import envs
from vllm.entrypoints.common.cli_args import make_arg_parser
from vllm.entrypoints.launchers.app import build_app
from vllm.utils.argparse_utils import FlexibleArgumentParser

from .conftest import MODEL_NAME_SMOLLM

pytestmark = pytest.mark.skipif(
    envs.VLLM_USE_RUST_FRONTEND, reason="Python frontend only"
)

DISABLED = [["--disable-sagemaker-integration"]]


@pytest.mark.parametrize("basic_server_with_lora", DISABLED, indirect=True)
def test_sagemaker_routes_are_not_registered(
    basic_server_with_lora: RemoteOpenAIServer,
):
    server = basic_server_with_lora
    assert requests.get(server.url_for("health")).status_code == HTTPStatus.OK
    assert requests.get(server.url_for("ping")).status_code == HTTPStatus.NOT_FOUND
    assert (
        requests.post(
            server.url_for("invocations"),
            json={"model": MODEL_NAME_SMOLLM, "prompt": "Hello", "max_tokens": 1},
        ).status_code
        == HTTPStatus.NOT_FOUND
    )
    assert (
        requests.post(server.url_for("adapters"), json={}).status_code
        == HTTPStatus.NOT_FOUND
    )


@pytest.mark.parametrize("basic_server_with_lora", DISABLED, indirect=True)
def test_native_lora_adapter_endpoints_still_work(
    basic_server_with_lora: RemoteOpenAIServer,
    smollm2_lora_files,
):
    """Runtime LoRA updating keeps working without the SageMaker aliases."""
    server = basic_server_with_lora
    adapter_name = "smollm2-native-lora"
    load_response = requests.post(
        server.url_for("v1/load_lora_adapter"),
        json={"lora_name": adapter_name, "lora_path": smollm2_lora_files},
    )
    assert load_response.status_code == HTTPStatus.OK
    unload_response = requests.post(
        server.url_for("v1/unload_lora_adapter"),
        json={"lora_name": adapter_name},
    )
    assert unload_response.status_code == HTTPStatus.OK


@pytest.mark.parametrize(
    "runtime_lora_updating", [False, True], ids=["no-lora", "runtime-lora"]
)
def test_sagemaker_integration_is_not_imported(
    tmp_path, monkeypatch, runtime_lora_updating: bool
):
    """The flag skips the bootstrap, which is what imports `model.py` from
    `SAGEMAKER_MODEL_PATH` and installs the `CUSTOM_*` hooks, and it skips the
    runtime LoRA import of the same package."""
    marker = tmp_path / "imported"
    (tmp_path / "model.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).touch()\n"
        "\n"
        "\n"
        "async def pre_process(request, call_next):\n"
        "    return await call_next(request)\n"
    )
    monkeypatch.setenv("SAGEMAKER_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("CUSTOM_PRE_PROCESS", "model:pre_process")
    monkeypatch.setenv(
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING", "1" if runtime_lora_updating else "0"
    )
    modules_before = set(sys.modules)

    parser = FlexibleArgumentParser()
    serve_parser = parser.add_subparsers().add_parser("serve")
    make_arg_parser(serve_parser)
    args = serve_parser.parse_args(["--disable-sagemaker-integration"])
    app = build_app(args, supported_tasks=())

    assert not marker.exists()
    assert not [
        middleware
        for middleware in app.user_middleware
        if middleware.cls.__module__.startswith("model_hosting_container_standards")
    ]
    assert not [
        name
        for name in set(sys.modules) - modules_before
        if name.startswith("model_hosting_container_standards")
    ]
