# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.artifact_transfer.schema import (
    TRAJECTORY_SCHEMA_NAME,
    TRAJECTORY_SCHEMA_VERSION_V1ALPHA1,
    TrajectoryArtifactV1Alpha1,
    build_trajectory_artifact_id,
    build_trajectory_partition_id,
)


def make_artifact(**kwargs):
    defaults = {
        "run_id": "run-a",
        "request_id": "req-a",
        "engine_id": "engine-a",
        "model_id": "model-a",
        "policy_version": 7,
        "sample_index": 2,
        "created_at_ns": 123456789,
        "prompt_token_ids": torch.tensor([1, 2, 3], dtype=torch.int32),
        "response_token_ids": [4, 5],
        "response_logprobs": torch.tensor([-0.1, -0.2], dtype=torch.float64),
    }
    defaults.update(kwargs)
    return TrajectoryArtifactV1Alpha1(**defaults)


def test_stable_key_and_partition_escape_components():
    assert build_trajectory_partition_id("run/a", "policy 7") == (
        "rollout-run%2Fa-policy%207"
    )
    assert build_trajectory_artifact_id("run/a", "policy 7", "req:1", 3) == (
        "run%2Fa:policy%207:req%3A1:3"
    )
    artifact = make_artifact()
    assert artifact.key == "run-a:7:req-a:2"
    assert artifact.partition_id == "rollout-run-a-7"


def test_valid_trajectory_normalizes_tensors_and_builds_handle():
    artifact = make_artifact(
        finish_reason="stop",
        rewards=torch.tensor(1.5, dtype=torch.float64),
        values=[0.4, 0.5],
        loss_mask=[1.0, 0.0],
        routed_experts=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    )
    record = artifact.to_transfer_queue_record()

    assert record.fields["prompt_token_ids"].dtype == torch.int64
    assert record.fields["response_token_ids"].dtype == torch.int64
    assert record.fields["response_logprobs"].dtype == torch.float32
    assert record.fields["rewards"].dtype == torch.float32
    assert all(value.device.type == "cpu" for value in record.fields.values())
    assert all(value.is_contiguous() for value in record.fields.values())
    assert record.tag["schema_name"] == TRAJECTORY_SCHEMA_NAME
    assert record.tag["schema_version"] == TRAJECTORY_SCHEMA_VERSION_V1ALPHA1
    assert record.tag["status"] == "complete"
    assert record.tag["group_id"] == "req-a"
    assert record.tag["finish_reason"] == "stop"

    handle = record.to_handle()
    assert handle.artifact_id == artifact.key
    assert handle.location == {
        "partition_id": artifact.partition_id,
        "key": artifact.key,
    }
    assert handle.metadata["schema_version"] == (TRAJECTORY_SCHEMA_VERSION_V1ALPHA1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prompt_token_ids", torch.tensor([1.0]), "integer dtype"),
        ("response_token_ids", torch.tensor([[1, 2]]), "one-dimensional"),
        ("response_logprobs", torch.tensor([1], dtype=torch.int64), "floating dtype"),
    ],
)
def test_invalid_dtype_or_dimension(field, value, message):
    with pytest.raises(ValueError, match=message):
        make_artifact(**{field: value}).validate()


def test_response_token_and_logprob_lengths_must_match():
    with pytest.raises(ValueError, match="equal length"):
        make_artifact(response_logprobs=[-0.1]).validate()


def test_loss_mask_length_must_match_response():
    with pytest.raises(ValueError, match="loss_mask"):
        make_artifact(loss_mask=[1.0]).validate()


def test_idempotent_reserialization():
    artifact = make_artifact()
    first = artifact.to_transfer_queue_record()
    second = artifact.to_transfer_queue_record()

    assert first.key == second.key
    assert first.partition_id == second.partition_id
    assert first.tag == second.tag
    assert first.fields.keys() == second.fields.keys()
    for name in first.fields:
        assert torch.equal(first.fields[name], second.fields[name])


def test_decode_single_sample_nested_tensor_round_trip():
    source = make_artifact(
        group_id="group-a",
        finish_reason="length",
        prompt_logprobs=[-0.3, -0.2, -0.1],
    )
    record = source.to_transfer_queue_record()
    retrieved_fields = {
        name: torch.nested.nested_tensor([value])
        for name, value in record.fields.items()
    }

    decoded = TrajectoryArtifactV1Alpha1.from_transfer_queue(
        retrieved_fields, record.tag
    )
    decoded_record = decoded.to_transfer_queue_record()

    assert decoded.key == source.key
    assert decoded.partition_id == source.partition_id
    assert decoded.group_id == "group-a"
    assert decoded.finish_reason == "length"
    for name in record.fields:
        assert torch.equal(record.fields[name], decoded_record.fields[name])


def test_decode_rejects_missing_required_field():
    record = make_artifact().to_transfer_queue_record()
    del record.fields["response_logprobs"]

    with pytest.raises(ValueError, match="Missing required trajectory fields"):
        TrajectoryArtifactV1Alpha1.from_transfer_queue(record.fields, record.tag)


@pytest.mark.parametrize(
    ("tag_name", "tag_value"),
    [
        ("schema_name", "other.trajectory"),
        ("schema_version", "v2"),
    ],
)
def test_decode_rejects_unknown_schema(tag_name, tag_value):
    record = make_artifact().to_transfer_queue_record()
    record.tag[tag_name] = tag_value

    with pytest.raises(ValueError, match="Unsupported trajectory schema"):
        TrajectoryArtifactV1Alpha1.from_transfer_queue(record.fields, record.tag)


def test_empty_policy_version_is_rejected():
    with pytest.raises(ValueError, match="policy_version"):
        make_artifact(policy_version="").validate()


def test_decode_rejects_incomplete_status():
    record = make_artifact().to_transfer_queue_record()
    record.tag["status"] = "writing"

    with pytest.raises(ValueError, match="not complete"):
        TrajectoryArtifactV1Alpha1.from_transfer_queue(record.fields, record.tag)
