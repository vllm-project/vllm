from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from pydantic import TypeAdapter

from vllm.config import TensorDumpConfig
from vllm.distributed import parallel_state
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.tensor_dump import TensorDumper, dump


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


@pytest.mark.parametrize("style", ["json", "nested"])
def test_cli_config_round_trip(tmp_path, style, monkeypatch):
    import json

    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    monkeypatch.setattr(current_platform, "device_type", "cpu")
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    assert EngineArgs.from_cli_args(parser.parse_args([])).tensor_dump_config == (
        TensorDumpConfig()
    )
    expected = TensorDumpConfig(
        output_dir=str(tmp_path), patterns=["*attn*"], ranks=[0], max_steps=2
    )
    options = (
        ["--tensor-dump-config", json.dumps(asdict(expected))]
        if style == "json"
        else [
            "--tensor-dump-config.output_dir",
            str(tmp_path),
            "--tensor-dump-config.patterns",
            '["*attn*"]',
            "--tensor-dump-config.ranks",
            "[0]",
            "--tensor-dump-config.max_steps",
            "2",
        ]
    )
    parsed = EngineArgs.from_cli_args(parser.parse_args(options)).tensor_dump_config
    assert parsed == expected
    assert (
        EngineArgs(tensor_dump_config=asdict(expected)).tensor_dump_config == expected
    )
    adapter = TypeAdapter(TensorDumpConfig)
    assert adapter.validate_json(adapter.dump_json(parsed)) == expected
    with pytest.raises(ValueError):
        TensorDumpConfig(output_dir="")


def test_hooks_snapshot_inputs_before_inplace_mutation(tmp_path, batch):
    model = torch.nn.Sequential(torch.nn.ReLU(inplace=True))
    dumper = TensorDumper(TensorDumpConfig(output_dir=str(tmp_path)))
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
    for name in ("input_ids", "positions", "query_start_loc", "seq_lens"):
        snapshot = saved["batch"][name]
        assert snapshot.untyped_storage().nbytes() == (
            snapshot.numel() * snapshot.element_size()
        )


def test_explicit_dump_preserves_nested_paths_and_detaches(tmp_path, batch):
    dumper = TensorDumper(TensorDumpConfig(output_dir=str(tmp_path)))
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
    dumper = TensorDumper(TensorDumpConfig(output_dir=str(tmp_path), **config))
    with dumper.capture(batch):
        dump("value", torch.ones(1))
    assert records(dumper) == []


def test_filters_step_limit_reset_and_reconfigure(tmp_path, batch):
    dumper = TensorDumper(
        TensorDumpConfig(output_dir=str(tmp_path), patterns=["selected*"])
    )
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
    assert dumper.config.output_dir == str(tmp_path)
    with pytest.raises(ValueError):
        dumper.control("configure", {"output_dir": "/other"})


def test_failed_forward_clears_capture_context(tmp_path, batch):
    dumper = TensorDumper(TensorDumpConfig(output_dir=str(tmp_path)))
    with pytest.raises(RuntimeError, match="forward failed"), dumper.capture(batch):
        raise RuntimeError("forward failed")
    dump("outside", torch.ones(1))
    assert [entry["name"] for entry in records(dumper)] == ["batch"]
    assert dumper.step == 1


def test_reregister_and_close_do_not_leave_hooks(tmp_path, batch):
    model = torch.nn.Identity()
    dumper = TensorDumper(TensorDumpConfig(output_dir=str(tmp_path)))
    dumper.register(model)
    dumper.register(model)
    with dumper.capture(batch):
        model(torch.ones(1))
    assert len(records(dumper)) == 3
    dumper.close()
    assert not model._forward_hooks
    assert not model._forward_pre_hooks


def test_workers_do_not_overwrite_each_other(tmp_path, batch):
    first = TensorDumper(TensorDumpConfig(output_dir=str(tmp_path)))
    second = TensorDumper(TensorDumpConfig(output_dir=str(tmp_path)))
    assert first.directory != second.directory
    for dumper in (first, second):
        with dumper.capture(batch):
            dump("value", torch.ones(1))
        assert len(records(dumper)) == 2


def test_saving_a_view_does_not_serialize_its_backing_storage(tmp_path, batch):
    dumper = TensorDumper(TensorDumpConfig(output_dir=str(tmp_path)))
    value = torch.ones(100_000)
    with dumper.capture(batch):
        dump("slice", value[10:13])
    saved = records(dumper)[-1]["value"]
    assert saved.untyped_storage().nbytes() == 3 * value.element_size()


def test_filter_can_select_a_nested_tensor(tmp_path, batch):
    dumper = TensorDumper(
        TensorDumpConfig(output_dir=str(tmp_path), patterns=["capture.items.1"])
    )
    with dumper.capture(batch):
        dump("capture", {"items": [torch.zeros(1), torch.ones(1)]})
    assert [entry["name"] for entry in records(dumper)] == ["batch", "capture.items.1"]


@pytest.mark.parametrize(
    "use_v2,eager,ubatching",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_unsupported_execution_rejected(tmp_path, use_v2, eager, ubatching):
    from vllm.config import VllmConfig

    config = SimpleNamespace(
        tensor_dump_config=TensorDumpConfig(output_dir=str(tmp_path)),
        use_v2_model_runner=use_v2,
        model_config=SimpleNamespace(enforce_eager=eager),
        parallel_config=SimpleNamespace(use_ubatching=ubatching),
    )
    with pytest.raises(ValueError, match="Tensor dumping"):
        VllmConfig._validate_tensor_dump(config)
    config.tensor_dump_config = TensorDumpConfig()
    VllmConfig._validate_tensor_dump(config)


@pytest.mark.parametrize("expert_parallel", [False, True])
def test_batch_records_parallel_groups(tmp_path, batch, monkeypatch, expert_parallel):
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    expected = {}
    for name, ranks in {
        "tp": [2, 3],
        "pp": [1, 3],
        "dp": [3],
        "dcp": [3],
        "pcp": [3],
        "ep": [2, 3],
    }.items():
        group = SimpleNamespace(rank_in_group=ranks.index(3), ranks=ranks)
        monkeypatch.setattr(
            parallel_state, f"get_{name}_group", lambda group=group: group
        )
        expected[name] = {"rank": group.rank_in_group, "ranks": ranks}
    if not expert_parallel:

        def missing_ep():
            raise AssertionError("expert parallel group is not initialized")

        monkeypatch.setattr(parallel_state, "get_ep_group", missing_ep)
        del expected["ep"]
    dumper = TensorDumper(TensorDumpConfig(output_dir=str(tmp_path)))
    with dumper.capture(batch):
        pass
    assert records(dumper)[0]["value"]["parallel_groups"] == expected


def test_runtime_control_uses_existing_rpc(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm.entrypoints.serve.dev.rpc.api_router import attach_router
    from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware

    app = FastAPI()
    app.add_middleware(AuthenticationMiddleware, tokens=["test-key"])
    dumper = TensorDumper(TensorDumpConfig(output_dir=str(tmp_path)))
    app.state.engine_client = SimpleNamespace(
        collective_rpc=AsyncMock(
            side_effect=lambda **kwargs: [dumper.control(*kwargs["args"])]
        )
    )
    attach_router(app)
    headers = {"Authorization": "Bearer test-key"}
    client = TestClient(app)
    assert client.post("/dumper/get_state", headers=headers).status_code == 404
    result = client.post(
        "/collective_rpc",
        json={"method": "dumper_control", "args": ["configure", {"enabled": False}]},
    )
    assert result.status_code == 200
    assert result.json()["results"][0]["config"]["enabled"] is False
    args = app.state.engine_client.collective_rpc.call_args
    assert args.kwargs["method"] == "dumper_control"
    assert args.kwargs["args"][1]["enabled"] is False
