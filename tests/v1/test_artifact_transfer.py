# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import ArtifactTransferConfig, VllmConfig
from vllm.distributed.artifact_transfer import artifact_transfer_state
from vllm.distributed.artifact_transfer.artifact_connector.factory import (
    ArtifactConnectorFactory,
)
from vllm.distributed.artifact_transfer.artifact_connector.v1 import (
    ArtifactConnectorRole,
)
from vllm.distributed.artifact_transfer.artifact_connector.v1.base import (
    ArtifactConnectorOutput,
    ArtifactHandle,
)
from vllm.distributed.artifact_transfer.artifact_connector.v1.transfer_queue_connector import (
    TransferQueueArtifactConnector,
    TransferQueueArtifactConnectorWorkerMetadata,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import LogprobsLists
from vllm.v1.worker.gpu.artifact_connector import NO_OP_ARTIFACT_CONNECTOR


@pytest.fixture(autouse=True)
def fake_transfer_queue_and_ray(monkeypatch):
    tq = types.ModuleType("transfer_queue")
    tq.rows = []
    tq.batch_rows = []
    tq.init_configs = []
    tq.client = SimpleNamespace(closed=False)
    tq.client.close = lambda: setattr(tq.client, "closed", True)
    tq.init = lambda config=None: tq.init_configs.append(config)
    tq.get_client = lambda: tq.client

    tq_interface = types.ModuleType("transfer_queue.interface")
    tq.client_init_configs = []
    tq_interface._maybe_create_tq_client = tq.client_init_configs.append
    monkeypatch.setitem(sys.modules, "transfer_queue.interface", tq_interface)

    def kv_put(**kwargs):
        tq.rows.append(kwargs)

    async def async_kv_batch_put(
        keys, partition_id, fields=None, tags=None, data_parser=None
    ):
        tq.batch_rows.append(
            {
                "keys": keys,
                "partition_id": partition_id,
                "fields": fields,
                "tags": tags,
                "data_parser": data_parser,
            }
        )
        return SimpleNamespace(
            keys=keys,
            partition_id=partition_id,
            fields=list(fields.keys()) if fields is not None else [],
        )

    tq.kv_put = kv_put
    tq.async_kv_batch_put = async_kv_batch_put
    monkeypatch.setitem(sys.modules, "transfer_queue", tq)

    ray = types.ModuleType("ray")
    ray.initialized = False
    ray.init_calls = []
    ray.shutdown_calls = 0
    ray.is_initialized = lambda: ray.initialized

    def init(**kwargs):
        ray.initialized = True
        ray.init_calls.append(kwargs)

    def shutdown():
        ray.initialized = False
        ray.shutdown_calls += 1

    ray.init = init
    ray.shutdown = shutdown
    monkeypatch.setitem(sys.modules, "ray", ray)
    yield tq, ray
    artifact_transfer_state._ARTIFACT_CONNECTOR_AGENT = None


def make_vllm_config(**kwargs):
    config_kwargs = {
        "artifact_connector": "TransferQueueArtifactConnector",
        "artifact_role": "artifact_producer",
        "export_fields": [
            "prompt_token_ids",
            "response_token_ids",
            "response_logprobs",
        ],
        "artifact_connector_extra_config": {
            "run_id": "default-run",
            "policy_version": "default-policy",
            "model_id": "default-model",
            "ray_address": "ray://trainer:10001",
        },
    }
    config_kwargs.update(kwargs)
    return VllmConfig(artifact_transfer_config=ArtifactTransferConfig(**config_kwargs))


def make_request(**overrides):
    values = {
        "request_id": "req-a",
        "prompt_token_ids": [1, 2, 3],
        "output_token_ids": [11, 12],
        "sampling_params": SimpleNamespace(num_logprobs=None),
        "artifact_transfer_params": None,
        "get_finished_reason": lambda: "stop",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_logprobs(token_ids=(11, 12), values=(-0.1, -0.2)):
    return LogprobsLists(
        logprob_token_ids=np.asarray(token_ids, dtype=np.int64)[:, None],
        logprobs=np.asarray(values, dtype=np.float32)[:, None],
        sampled_token_ranks=np.zeros(len(token_ids), dtype=np.int64),
    )


def test_scheduler_producer_publishes_v1alpha1_trajectory(
    fake_transfer_queue_and_ray,
):
    tq, ray = fake_transfer_queue_and_ray
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(), ArtifactConnectorRole.SCHEDULER
    )
    assert isinstance(connector, TransferQueueArtifactConnector)
    request = make_request(
        artifact_transfer_params={
            "run_id": "request-run",
            "policy_version": 7,
            "model_id": "request-model",
            "group_id": "group-a",
            "sample_index": 2,
        }
    )

    connector.on_new_request(request)
    assert request.sampling_params.num_logprobs == 0
    connector.record_request_output(request, [11, 12], make_logprobs())
    handle = connector.request_finished(request)

    assert handle is not None
    assert handle.artifact_id == "request-run:7:req-a:2"
    assert handle.location["partition_id"] == "rollout-request-run-7"
    assert ray.init_calls == [{"address": "ray://trainer:10001"}]
    assert tq.init_configs == [None]
    row = tq.rows[0]
    assert row["key"] == handle.artifact_id
    assert row["partition_id"] == handle.location["partition_id"]
    assert torch.equal(row["fields"]["prompt_token_ids"], torch.tensor([1, 2, 3]))
    assert torch.equal(row["fields"]["response_token_ids"], torch.tensor([11, 12]))
    assert torch.allclose(
        row["fields"]["response_logprobs"], torch.tensor([-0.1, -0.2])
    )
    assert row["tag"]["run_id"] == "request-run"
    assert row["tag"]["policy_version"] == 7
    assert row["tag"]["model_id"] == "request-model"
    assert row["tag"]["group_id"] == "group-a"
    assert row["tag"]["finish_reason"] == "stop"

    connector.shutdown()
    assert tq.client.closed
    assert ray.shutdown_calls == 1


def test_scheduler_producer_uses_connector_metadata_defaults(
    fake_transfer_queue_and_ray,
):
    tq, _ = fake_transfer_queue_and_ray
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(), ArtifactConnectorRole.SCHEDULER
    )
    request = make_request()
    connector.on_new_request(request)
    connector.record_request_output(request, [11, 12], make_logprobs())
    handle = connector.request_finished(request)

    assert handle is not None
    assert handle.artifact_id == "default-run:default-policy:req-a:0"
    assert tq.rows[0]["tag"]["model_id"] == "default-model"


def test_scheduler_producer_exported_config_does_not_initialize_ray(
    fake_transfer_queue_and_ray,
):
    tq, ray = fake_transfer_queue_and_ray
    exported_config = object()
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(
            artifact_connector_extra_config={
                "run_id": "default-run",
                "policy_version": "default-policy",
                "model_id": "default-model",
                "transfer_queue_init_config": exported_config,
            }
        ),
        ArtifactConnectorRole.SCHEDULER,
    )
    request = make_request()
    connector.on_new_request(request)
    connector.record_request_output(request, [11, 12], make_logprobs())

    assert connector.request_finished(request) is not None
    assert tq.client_init_configs == [exported_config]
    assert tq.init_configs == []
    assert ray.init_calls == []

    connector.shutdown()
    assert tq.client.closed
    assert ray.shutdown_calls == 0


def test_scheduler_producer_loads_exported_config_path_without_ray(
    fake_transfer_queue_and_ray,
    tmp_path,
):
    tq, ray = fake_transfer_queue_and_ray
    exported_config = {"controller": "exported"}
    config_path = tmp_path / "client_config.pkl"
    config_path.write_bytes(pickle.dumps(exported_config))
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(
            artifact_connector_extra_config={
                "run_id": "default-run",
                "policy_version": "default-policy",
                "model_id": "default-model",
                "transfer_queue_config_path": str(config_path),
            }
        ),
        ArtifactConnectorRole.SCHEDULER,
    )
    request = make_request()
    connector.on_new_request(request)
    connector.record_request_output(request, [11, 12], make_logprobs())

    assert connector.request_finished(request) is not None
    assert tq.client_init_configs == [exported_config]
    assert tq.init_configs == []
    assert ray.init_calls == []

    connector.shutdown()
    assert tq.client.closed
    assert ray.shutdown_calls == 0


def test_scheduler_producer_writes_debug_metrics(
    fake_transfer_queue_and_ray,
    tmp_path,
):
    metrics_path = tmp_path / "artifact_metrics.jsonl"
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(
            artifact_connector_extra_config={
                "run_id": "default-run",
                "policy_version": "default-policy",
                "model_id": "default-model",
                "ray_address": "ray://trainer:10001",
                "artifact_metrics_path": str(metrics_path),
            }
        ),
        ArtifactConnectorRole.SCHEDULER,
    )
    request = make_request()
    connector.on_new_request(request)
    connector.record_request_output(request, [11, 12], make_logprobs())

    handle = connector.request_finished(request)

    assert handle is not None
    row = json.loads(metrics_path.read_text().strip())
    assert row["status"] == "ok"
    assert row["request_id"] == "req-a"
    assert row["artifact_id"] == handle.artifact_id
    assert row["response_token_count"] == 2
    assert row["response_logprob_count"] == 2
    assert row["schema_build_ms"] >= 0
    assert row["kv_put_ms"] >= 0
    assert row["total_ms"] >= row["kv_put_ms"]


def test_scheduler_producer_async_publish_writes_in_background(
    fake_transfer_queue_and_ray,
    tmp_path,
):
    tq, _ = fake_transfer_queue_and_ray
    metrics_path = tmp_path / "artifact_metrics.jsonl"
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(
            artifact_connector_extra_config={
                "run_id": "default-run",
                "policy_version": "default-policy",
                "model_id": "default-model",
                "ray_address": "ray://trainer:10001",
                "artifact_metrics_path": str(metrics_path),
                "publish_mode": "async",
                "publish_queue_maxsize": 8,
            }
        ),
        ArtifactConnectorRole.SCHEDULER,
    )
    request = make_request()
    connector.on_new_request(request)
    connector.record_request_output(request, [11, 12], make_logprobs())

    handle = connector.request_finished(request)

    assert handle is not None
    connector.shutdown()
    assert len(tq.rows) == 1
    assert tq.rows[0]["key"] == handle.artifact_id
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    queued = [row for row in rows if row["status"] == "queued"]
    published = [row for row in rows if row["status"] == "ok"]
    assert len(queued) == 1
    assert len(published) == 1
    assert queued[0]["async_publish"] is True
    assert published[0]["async_publish"] is True
    assert queued[0]["enqueue_ms"] >= 0
    assert published[0]["queue_wait_ms"] >= 0
    assert published[0]["kv_put_ms"] >= 0


def test_scheduler_producer_async_batch_publish_uses_batch_put(
    fake_transfer_queue_and_ray,
    tmp_path,
):
    tq, _ = fake_transfer_queue_and_ray
    metrics_path = tmp_path / "artifact_metrics.jsonl"
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(
            artifact_connector_extra_config={
                "run_id": "default-run",
                "policy_version": "default-policy",
                "model_id": "default-model",
                "ray_address": "ray://trainer:10001",
                "artifact_metrics_path": str(metrics_path),
                "publish_mode": "async_batch",
                "publish_batch_size": 2,
                "publish_flush_interval_ms": 1000,
            }
        ),
        ArtifactConnectorRole.SCHEDULER,
    )
    first = make_request()
    second = make_request(
        request_id="req-b",
        output_token_ids=[21, 22, 23],
    )
    connector.on_new_request(first)
    connector.record_request_output(first, [11, 12], make_logprobs())
    connector.on_new_request(second)
    connector.record_request_output(
        second,
        [21, 22, 23],
        make_logprobs(token_ids=(21, 22, 23), values=(-0.3, -0.4, -0.5)),
    )

    first_handle = connector.request_finished(first)
    second_handle = connector.request_finished(second)

    assert first_handle is not None
    assert second_handle is not None
    connector.shutdown()
    assert tq.rows == []
    assert len(tq.batch_rows) == 1
    batch = tq.batch_rows[0]
    assert batch["keys"] == [first_handle.artifact_id, second_handle.artifact_id]
    assert batch["partition_id"] == first_handle.location["partition_id"]
    assert batch["tags"][0]["request_id"] == "req-a"
    assert batch["tags"][1]["request_id"] == "req-b"
    assert batch["fields"].batch_size == torch.Size([2])
    assert batch["fields"]["response_token_ids"].is_nested

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    queued = [row for row in rows if row["status"] == "queued"]
    published = [row for row in rows if row["status"] == "ok"]
    assert len(queued) == 2
    assert len(published) == 2
    assert {row["batch_size"] for row in published} == {2}
    assert all(row["async_batch_publish"] is True for row in published)


def test_scheduler_producer_rejects_unknown_publish_mode():
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(artifact_connector_extra_config={"publish_mode": "mystery"}),
        ArtifactConnectorRole.SCHEDULER,
    )
    with pytest.raises(ValueError, match="publish_mode"):
        connector._publish_mode()


def test_sampled_logprob_token_mismatch_obeys_fail_request_policy():
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(), ArtifactConnectorRole.SCHEDULER
    )
    request = make_request()
    connector.on_new_request(request)
    connector.record_request_output(
        request, [11, 12], make_logprobs(token_ids=(11, 99))
    )
    with pytest.raises(RuntimeError, match="Failed to publish trajectory"):
        connector.request_finished(request)


def test_publish_failure_can_fall_back_to_normal_request_output(
    fake_transfer_queue_and_ray,
):
    tq, _ = fake_transfer_queue_and_ray

    def fail_put(**kwargs):
        raise OSError("backend unavailable")

    tq.kv_put = fail_put
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(failure_policy="fallback_to_request_output"),
        ArtifactConnectorRole.SCHEDULER,
    )
    request = make_request()
    connector.on_new_request(request)
    connector.record_request_output(request, [11, 12], make_logprobs())
    assert connector.request_finished(request) is None


def test_shutdown_does_not_stop_ray_owned_by_host(fake_transfer_queue_and_ray):
    tq, ray = fake_transfer_queue_and_ray
    ray.initialized = True
    connector = ArtifactConnectorFactory.create_connector(
        make_vllm_config(), ArtifactConnectorRole.SCHEDULER
    )
    request = make_request()
    connector.on_new_request(request)
    connector.record_request_output(request, [11, 12], make_logprobs())
    assert connector.request_finished(request) is not None
    connector.shutdown()
    assert tq.client.closed
    assert ray.shutdown_calls == 0


def test_no_op_connector_record_output_is_safe():
    NO_OP_ARTIFACT_CONNECTOR.record_model_runner_output(
        SimpleNamespace(req_ids=[], req_id_to_index={}), set()
    )


def test_scheduler_serializes_artifact_handle_once():
    handle = ArtifactHandle(
        backend="transfer_queue",
        artifact_id="partition-0:req-a",
        location={"partition_id": "partition-0", "key": "partition-0:req-a"},
        fields=["token_ids"],
        metadata={"engine_id": "engine-a"},
    )
    connector_output = ArtifactConnectorOutput(
        handles={"req-a": handle},
        worker_meta=TransferQueueArtifactConnectorWorkerMetadata(
            num_published=1, handles={"req-a": handle}
        ),
    )

    class FakeArtifactConnector:
        def __init__(self):
            self.handles = {}

        def update_connector_output(self, output):
            self.handles.update(output.handles)

        def request_finished(self, request):
            return self.handles.pop(request.request_id, None)

    scheduler = object.__new__(Scheduler)
    scheduler.artifact_connector = FakeArtifactConnector()
    Scheduler._update_from_artifact_xfer_finished(scheduler, connector_output)
    params = Scheduler._artifact_connector_finished(
        scheduler, SimpleNamespace(request_id="req-a")
    )
    assert params == {"artifact_handle": handle.to_dict()}
    assert (
        Scheduler._artifact_connector_finished(
            scheduler, SimpleNamespace(request_id="req-a")
        )
        is None
    )
