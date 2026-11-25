# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`--disable-sagemaker-integration` leaves the SageMaker surface out."""

import json
import os
import subprocess
import sys
from http import HTTPStatus
from pathlib import Path

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm import envs

from .conftest import MODEL_NAME_SMOLLM
from .sagemaker_app_probe import PACKAGE

pytestmark = pytest.mark.skipif(
    envs.VLLM_USE_RUST_FRONTEND, reason="Python frontend only"
)

DISABLED = [["--disable-sagemaker-integration"]]
PROBE = Path(__file__).with_name("sagemaker_app_probe.py")
MARKER = "imported"


def _probe(tmp_path: Path, *cli_args: str, env: dict[str, str] | None = None) -> dict:
    """Build the app in a fresh interpreter and return what the launch left.

    A fresh process is what makes the import check meaningful: `sys.modules`
    in this one may already hold the package because an earlier test built a
    server with the integration enabled.
    """
    (tmp_path / "model.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(tmp_path / MARKER)!r}).touch()\n"
        "\n"
        "\n"
        "async def pre_process(request, call_next):\n"
        "    return await call_next(request)\n"
    )
    probe_env = os.environ | {
        "SAGEMAKER_MODEL_PATH": str(tmp_path),
        "CUSTOM_PRE_PROCESS": "model:pre_process",
    }
    result = subprocess.run(
        [sys.executable, str(PROBE), *cli_args],
        env=probe_env | (env or {}),
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or not lines:
        pytest.fail(f"SageMaker probe failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(lines[-1])


def _sagemaker_middleware(probe: dict) -> list[str]:
    return [
        middleware
        for middleware in probe["middleware"]
        if middleware.startswith(PACKAGE)
    ]


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
def test_sagemaker_integration_is_not_imported(tmp_path, runtime_lora_updating: bool):
    """The flag skips the bootstrap, which is what imports `model.py` from
    `SAGEMAKER_MODEL_PATH` and installs the `CUSTOM_*` hooks, and it skips the
    runtime LoRA import of the same package. The probe imports the launch path
    first, so a module-level import anywhere on it fails this too."""
    probe = _probe(
        tmp_path,
        "--disable-sagemaker-integration",
        env={"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1" if runtime_lora_updating else "0"},
    )

    assert probe["loaded"] == []
    assert _sagemaker_middleware(probe) == []
    assert not (tmp_path / MARKER).exists()


def test_sagemaker_integration_runs_by_default(tmp_path):
    """Positive control for the probe: without the flag the same run sees the
    package, its middleware and the imported `model.py`."""
    probe = _probe(tmp_path, env={"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"})

    assert probe["loaded"]
    assert _sagemaker_middleware(probe)
    assert (tmp_path / MARKER).exists()
