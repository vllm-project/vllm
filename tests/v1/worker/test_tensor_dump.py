from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.tensor_dump import TensorDumpConfig, TensorDumper, dump


@pytest.fixture
def batch():
    buffers = InputBuffers(2, 4, torch.device("cpu"))
    result = InputBatch.make_dummy(1, 2, buffers)
    result.num_tokens_after_padding = 4
    result.input_ids = torch.tensor([10, 11, 99, 99])
    result.positions = torch.tensor([0, 1, 99, 99])
    return result


def records(dumper):
    return [
        torch.load(path, weights_only=True)
        for path in sorted(dumper.directory.glob("*.pt"))
    ]


def test_hooks_snapshot_inputs_before_inplace_mutation(tmp_path, batch):
    model = torch.nn.Sequential(torch.nn.ReLU(inplace=True))
    dumper = TensorDumper(str(tmp_path), TensorDumpConfig())
    dumper.register(model)
    inputs = torch.tensor([-2.0, 3.0])
    expected = model(inputs.clone())
    with dumper.capture(batch):
        actual = model(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    saved = {entry["name"]: entry["value"] for entry in records(dumper)}
    torch.testing.assert_close(saved["0.input.args.0"], torch.tensor([-2.0, 3.0]))
    torch.testing.assert_close(saved["0.output"], expected)
    assert saved["batch"]["input_ids"].tolist() == [10, 11]
    assert saved["batch"]["query_start_loc"].tolist() == [0, 2]
    assert saved["batch"]["num_tokens_after_padding"] == 4


def test_explicit_dump_preserves_nested_paths_and_detaches(tmp_path, batch):
    dumper = TensorDumper(str(tmp_path), TensorDumpConfig())
    value = torch.tensor([3.0], requires_grad=True)
    dump("outside", value)
    with dumper.capture(batch):
        dump(
            "inside",
            {"items": (None, value), "pp": IntermediateTensors({"hidden": value})},
        )
    saved = {entry["name"]: entry["value"] for entry in records(dumper)}
    assert set(saved) == {"batch", "inside.items.1", "inside.pp.hidden"}
    assert not saved["inside.items.1"].requires_grad
    assert saved["inside.items.1"].device.type == "cpu"
    value.detach().zero_()
    assert saved["inside.items.1"].item() == 3


@pytest.mark.parametrize("config", [{"enabled": False}, {"ranks": [999]}])
def test_disabled_capture_writes_nothing(tmp_path, batch, config):
    dumper = TensorDumper(str(tmp_path), TensorDumpConfig(**config))
    with dumper.capture(batch):
        dump("value", torch.ones(1))
    assert records(dumper) == []


def test_filters_step_limit_reset_and_reconfigure(tmp_path, batch):
    dumper = TensorDumper(str(tmp_path), TensorDumpConfig(patterns=["selected*"]))
    for _ in range(2):
        with dumper.capture(batch):
            dump("selected", torch.ones(1))
            dump("ignored", torch.ones(1))
    before = records(dumper)
    assert [entry["name"] for entry in before] == ["batch", "selected"]
    dumper.control("reset")
    dumper.control("configure", {"patterns": ["other"]})
    with dumper.capture(batch):
        dump("selected", torch.ones(1))
        dump("other", torch.zeros(1))
    assert [entry["name"] for entry in records(dumper)] == [
        "batch",
        "selected",
        "batch",
        "other",
    ]
    assert dumper.control("get_state")["step"] == 2
    assert dumper.control("get_state")["captured_steps"] == 1
    assert records(dumper)[-1]["step"] == 1


def test_failed_forward_clears_capture_context(tmp_path, batch):
    dumper = TensorDumper(str(tmp_path), TensorDumpConfig())
    with pytest.raises(RuntimeError, match="forward failed"), dumper.capture(batch):
        raise RuntimeError("forward failed")
    dump("outside", torch.ones(1))
    assert [entry["name"] for entry in records(dumper)] == ["batch"]
    assert dumper.step == 1


def test_reregister_and_close_do_not_leave_hooks(tmp_path, batch):
    model = torch.nn.Identity()
    dumper = TensorDumper(str(tmp_path), TensorDumpConfig())
    dumper.register(model)
    dumper.register(model)
    with dumper.capture(batch):
        model(torch.ones(1))
    assert len(records(dumper)) == 3
    dumper.close()
    assert not model._forward_hooks
    assert not model._forward_pre_hooks


def test_workers_do_not_overwrite_each_other(tmp_path, batch):
    first = TensorDumper(str(tmp_path), TensorDumpConfig())
    second = TensorDumper(str(tmp_path), TensorDumpConfig())
    assert first.directory != second.directory
    for dumper in (first, second):
        with dumper.capture(batch):
            dump("value", torch.ones(1))
        assert len(records(dumper)) == 2


def test_saving_a_view_does_not_serialize_its_backing_storage(tmp_path, batch):
    dumper = TensorDumper(str(tmp_path), TensorDumpConfig())
    value = torch.ones(100_000)
    with dumper.capture(batch):
        dump("slice", value[10:13])
    saved = records(dumper)[-1]["value"]
    assert saved.untyped_storage().nbytes() == 3 * value.element_size()


def test_filter_can_select_a_nested_tensor(tmp_path, batch):
    dumper = TensorDumper(str(tmp_path), TensorDumpConfig(patterns=["capture.items.1"]))
    with dumper.capture(batch):
        dump("capture", {"items": [torch.zeros(1), torch.ones(1)]})
    assert [entry["name"] for entry in records(dumper)] == ["batch", "capture.items.1"]


@pytest.mark.parametrize(
    "use_v2,eager,ubatching",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_unsupported_execution_rejected(
    tmp_path, monkeypatch, use_v2, eager, ubatching
):
    from vllm.config import VllmConfig

    monkeypatch.setenv("VLLM_TENSOR_DUMP_DIR", str(tmp_path))
    config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        model_config=SimpleNamespace(enforce_eager=eager),
        parallel_config=SimpleNamespace(use_ubatching=ubatching),
    )
    with pytest.raises(ValueError, match="Tensor dumping"):
        VllmConfig._validate_tensor_dump(config)
    monkeypatch.delenv("VLLM_TENSOR_DUMP_DIR")
    VllmConfig._validate_tensor_dump(config)


def test_runtime_control_validation_and_auth(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware
    from vllm.entrypoints.serve.tensor_dump.api_router import attach_router

    monkeypatch.setenv("VLLM_TENSOR_DUMP_DIR", str(tmp_path))
    app = FastAPI()
    app.add_middleware(AuthenticationMiddleware, tokens=["test-key"])
    app.state.engine_client = SimpleNamespace(
        collective_rpc=AsyncMock(return_value=[{"step": 0}])
    )
    attach_router(app)
    client = TestClient(app)
    assert client.post("/dumper/get_state").status_code == 401
    headers = {"Authorization": "Bearer test-key"}
    assert client.post("/dumper/configure", headers=headers).status_code == 422
    assert (
        client.post(
            "/dumper/configure", headers=headers, json={"max_steps": 0}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/dumper/configure", headers=headers, json={"directory": "/other"}
        ).status_code
        == 422
    )
    result = client.post("/dumper/configure", headers=headers, json={"enabled": False})
    assert result.status_code == 200
    assert result.json() == [{"step": 0}]
    args = app.state.engine_client.collective_rpc.call_args
    assert args.args == ("dumper_control",)
    assert args.kwargs["args"][1]["enabled"] is False
