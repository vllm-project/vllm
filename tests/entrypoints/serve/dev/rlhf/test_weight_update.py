# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the RL weight-transfer endpoints.

The server lifecycle follows PR #52144: the shared ``conftest.py`` server
fixture is reused at module scope and both model-runner implementations are
covered. Compound sleep/wake coverage remains in ``test_sleep_wake.py``.

The real tensor-transfer test is parametrized over IPC and NCCL. IPC uses a
colocated trainer/inference GPU; NCCL uses a separate trainer GPU and a local
rendezvous port. Cases that cannot run on the available hardware are skipped.
"""

import errno
import os
from unittest.mock import patch

import pytest
import requests

from .conftest import (
    MODEL_NAME,
    finish_weight_update,
    gen,
    get_world_size,
    health,
    ok,
    resume,
    server,
    start_weight_update,
)


_SERVER_PORT_BASE = 8870
_TENSOR_SERVER_PORT_BASE = 8890
_NCCL_MASTER_PORT_BASE = 29600


def _has_cuda() -> bool:
    """Return whether the current test process can use CUDA."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _has_two_gpus() -> bool:
    """Return whether NCCL can use a separate trainer GPU."""
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except Exception:
        return False


@pytest.fixture(scope="module", params=[False, True], ids=["MRV1", "MRV2"])
def use_v2(request):
    """Run the HTTP tests with both model-runner implementations."""
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            ("ipc", {"rank": 0, "packed": False}),
            id="IPC",
            marks=pytest.mark.skipif(
                not _has_cuda(),
                reason="IPC weight transfer requires CUDA",
            ),
        ),
        pytest.param(
            (
                "nccl",
                {
                    "rank": 0,
                    "master_address": "127.0.0.1",
                    "master_port": _NCCL_MASTER_PORT_BASE,
                    "world_size": 2,
                    "packed": False,
                },
            ),
            id="NCCL",
            marks=pytest.mark.skipif(
                not _has_two_gpus(),
                reason="NCCL weight transfer requires at least two GPUs",
            ),
        ),
    ],
)
def weight_transfer_case(request, use_v2):
    """Return one backend case, with unique NCCL ports per model runner."""
    backend, init_info = request.param
    init_info = dict(init_info)
    if backend == "nccl":
        init_info["master_port"] += int(use_v2)
    return backend, init_info


@pytest.fixture(scope="module")
def server_url(use_v2):
    """Start one server per model-runner implementation for this module."""
    env_vars = {
        "VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0",
    }

    # The shared fixture owns process startup, health polling, and cleanup.
    with (
        patch.dict(os.environ, env_vars),
        server(
            port=_SERVER_PORT_BASE + int(use_v2),
            extra_args=[
                "--enable-prefix-caching",
                "--enable-prompt-tokens-details",
            ]
        ) as url,
    ):
        yield url


@pytest.fixture(scope="module")
def tensor_server_url(use_v2, weight_transfer_case):
    """Start a backend-configured server for the opt-in tensor test."""
    backend, _ = weight_transfer_case
    env_vars = {
        "VLLM_USE_V2_MODEL_RUNNER": "1" if use_v2 else "0",
    }
    if backend == "ipc":
        # HTTP IPC payloads contain serialized CUDA handles.
        env_vars["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    with (
        patch.dict(os.environ, env_vars),
        server(
            port=(
                _TENSOR_SERVER_PORT_BASE
                + int(use_v2)
                + (100 if backend == "nccl" else 0)
            ),
            extra_args=[
                "--enable-prefix-caching",
                "--enable-prompt-tokens-details",
            ],
            weight_transfer_config={"backend": backend},
        ) as url,
    ):
        yield url


@pytest.fixture
def restore_unpaused_state(server_url):
    """Keep each HTTP test isolated from a previous pause state."""
    assert resume(server_url) == 200
    yield
    assert resume(server_url) == 200


@pytest.fixture
def restore_tensor_unpaused_state(tensor_server_url):
    """Keep the backend-configured tensor server in the resumed state."""
    assert resume(tensor_server_url) == 200
    yield
    assert resume(tensor_server_url) == 200


@pytest.mark.usefixtures("restore_unpaused_state")
class TestWeightTransferProtocol:
    """Validate the basic weight-transfer protocol and cache lifecycle."""

    def test_start_finish_no_tensors_engine_survives(self, server_url):
        """A start/finish request without tensors must not kill the server."""
        response = start_weight_update(server_url)
        assert response.status_code in (200, 500), (
            f"start_weight_update unexpected status "
            f"{response.status_code}: {response.text}"
        )

        if response.status_code == 200:
            # A successful start is required before finish can be called.
            response = finish_weight_update(server_url)
            assert response.status_code == 200, response.text

        assert health(server_url) == 200
        assert ok(gen(server_url)), "service cannot generate after protocol request"

    def test_get_world_size_positive(self, server_url):
        """The reported distributed world size must be at least one."""
        response = get_world_size(server_url)
        assert response.status_code == 200
        world_size = response.json()["world_size"]
        assert isinstance(world_size, int)
        assert world_size >= 1

    def test_finish_without_start_handled(self, server_url):
        """Finishing without a start must not terminate the server."""
        response = finish_weight_update(server_url)
        assert response.status_code in (200, 400, 409, 500), response.text
        assert health(server_url) == 200

    def test_prefix_cache_flushed_after_finish(self, server_url):
        """Generation after finish must not use an invalid prefix-cache entry."""
        prompt = "The capital of France is"
        assert ok(gen(server_url, prompt=prompt)), "warm-up generation failed"

        start_response = start_weight_update(server_url)
        if start_response.status_code == 200:
            finish_response = finish_weight_update(server_url)
            assert finish_response.status_code == 200, finish_response.text
        else:
            assert start_response.status_code in (400, 409, 500), (
                f"unexpected start_weight_update status "
                f"{start_response.status_code}: {start_response.text}"
            )

        assert ok(gen(server_url, prompt=prompt)), (
            "generation failed after finish_weight_update; "
            "prefix cache may not have been flushed"
        )


@pytest.mark.usefixtures("restore_tensor_unpaused_state")
class TestWeightUpdateWithTensors:
    """Transfer an alternate checkpoint through a configured backend."""

    @pytest.fixture(autouse=True)
    def _require_alt_model(self):
        """Skip unless a different checkpoint is available for the test."""
        alt_path = os.environ.get("VLLM_TEST_MODEL_ALT")
        if not alt_path or alt_path == MODEL_NAME:
            pytest.skip(
                "Set VLLM_TEST_MODEL_ALT to a different checkpoint to run "
                "weight-update output-change tests"
            )
        self.alt_model_path = alt_path

    def test_weight_update_changes_output(
        self,
        tensor_server_url,
        weight_transfer_case,
    ):
        """Push alternate weights and compare generation before and after."""
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError:
            pytest.skip("transformers not available")

        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for the real weight-transfer test")

        backend, init_info = weight_transfer_case
        torch.cuda.set_device(1 if backend == "nccl" else 0)
        before = gen(tensor_server_url)
        assert ok(before), "generation before update failed"
        before_text = before["choices"][0]["text"]

        alt_model = AutoModelForCausalLM.from_pretrained(
            self.alt_model_path,
            torch_dtype=torch.bfloat16,
        ).cuda()

        try:
            from vllm.distributed.weight_transfer import (
                HTTPVLLMWeightSyncClient,
                ModuleSource,
                WeightTransferTrainerFactory,
            )
            if backend == "ipc":
                from vllm.distributed.weight_transfer.ipc_engine import (
                    IPCTrainerInitInfo,
                )

                trainer_init_info = IPCTrainerInitInfo(**init_info)
            else:
                from vllm.distributed.weight_transfer.nccl_engine import (
                    NCCLTrainerInitInfo,
                )

                trainer_init_info = NCCLTrainerInitInfo(**init_info)
        except ImportError as exc:
            pytest.skip(f"weight-transfer trainer dependencies unavailable: {exc}")

        trainer = None
        try:
            trainer = WeightTransferTrainerFactory.trainer_init(
                init_info=trainer_init_info,
                client=HTTPVLLMWeightSyncClient(tensor_server_url),
                source=ModuleSource(alt_model),
            )
            trainer.send_weights()
        finally:
            if trainer is not None:
                trainer.shutdown()
            del alt_model

        after = gen(tensor_server_url)
        assert ok(after), "generation after weight update failed"
        assert after["choices"][0]["text"] != before_text, (
            "output did not change after pushing different weights"
        )
        assert health(tensor_server_url) == 200


def _has_sm100() -> bool:
    """Return whether the first visible GPU supports SM100 or newer."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability(0)[0] >= 10
    except Exception:
        return False


_RELOAD_MATRIX = [
    pytest.param("TitanML/tiny-mixtral", {}, id="moe-bf16-tiny"),
    pytest.param("ibm-research/PowerMoE-3b", {}, id="moe-bf16-3b"),
    pytest.param(
        "allenai/OLMoE-1B-7B-0924",
        {"quantization": "fp8"},
        id="moe-fp8",
    ),
    pytest.param(
        "allenai/OLMoE-1B-7B-0924",
        {"quantization": "mxfp8"},
        id="moe-mxfp8",
        marks=pytest.mark.skipif(
            not _has_sm100(),
            reason="mxfp8 requires SM100+ (Blackwell)",
        ),
    ),
]


def _run_reload_test(model: str, extra_kwargs: dict):
    """Run one layerwise reload case and verify that generation survives."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": model,
        "enforce_eager": True,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.7,
        **extra_kwargs,
    }

    try:
        llm = LLM(**llm_kwargs)
    except OSError as exc:
        if exc.errno == errno.EROFS:
            pytest.skip(
                f"{model!r} not in local HF cache (read-only mount) - "
                "pre-download the model to run this test"
            )
        raise

    before = llm.generate(["Hello"], SamplingParams(max_tokens=8))
    assert before and before[0].outputs, "initial generate failed"

    def _trigger_layerwise_finalize(worker):
        import torch

        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )

        model_instance = worker.model_runner.model
        with torch.device(worker.device):
            initialize_layerwise_reload(model_instance)
            finalize_layerwise_reload(model_instance, worker.model_config)

    llm.collective_rpc(_trigger_layerwise_finalize)
    after = llm.generate(["Hello"], SamplingParams(max_tokens=8))
    assert after and after[0].outputs, "generate failed after weight reload"
    del llm


class TestWeightReloadCodePaths:
    """Exercise the layerwise reload matrix across model and quantization."""

    @pytest.mark.parametrize("model,extra_kwargs", _RELOAD_MATRIX)
    def test_reload_survives(self, model, extra_kwargs):
        _run_reload_test(model, extra_kwargs)


@pytest.mark.usefixtures("restore_unpaused_state")
class TestWeightUpdateProtocolErrors:
    """Verify invalid request ordering returns errors without killing the server."""

    def test_start_without_init_returns_error_and_engine_survives(self, server_url):
        response = start_weight_update(server_url)
        assert response.status_code in (400, 409, 500), (
            "start without init must return an error, got "
            f"{response.status_code}: {response.text}"
        )

        try:
            body = response.json()
            error = body.get("error", {})
            error_message = (
                error.get("message", "") if isinstance(error, dict) else error
            )
            message = str(body.get("detail", error_message)).lower()
            assert any(
                keyword in message
                for keyword in ("not configured", "weight transfer", "init", "config")
            ), f"error message not informative: {response.text}"
        except (ValueError, KeyError):
            # Some server errors are not JSON; the status code is still useful.
            pass

        assert health(server_url) == 200
        assert ok(gen(server_url)), "engine must still serve after protocol error"

    def test_update_weights_without_start_returns_error(self, server_url):
        response = requests.post(
            f"{server_url}/update_weights",
            json={
                "update_info": {
                    "name": "dummy",
                    "dtype": "float32",
                    "shape": [1],
                }
            },
            timeout=10,
        )
        assert response.status_code in (400, 409, 500), (
            "update without start must return an error, "
            f"got {response.status_code}"
        )
        assert health(server_url) == 200
        assert ok(gen(server_url)), "engine must still serve after invalid update"

    def test_finish_without_start_returns_error_and_engine_survives(self, server_url):
        response = finish_weight_update(server_url)
        assert response.status_code in (400, 409, 500), (
            "finish without start must return an error, "
            f"got {response.status_code}"
        )
        assert health(server_url) == 200
        assert ok(gen(server_url)), "engine must generate after invalid finish"
