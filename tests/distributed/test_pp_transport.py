# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.pp_transport import (
    PPTransportAllGatherPolicy,
    PPTransportDataType,
    PPTransportPayload,
    add_pp_transport_tensor,
    copy_pp_transport_tensor,
)
from vllm.sequence import IntermediateTensors


def _run_transport_stage(rank: int, init_method: str):
    dist.init_process_group("gloo", rank=rank, world_size=3, init_method=init_method)
    group = GroupCoordinator(
        [[0, 1, 2]],
        local_rank=rank,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
    )
    policy = PPTransportAllGatherPolicy()
    try:
        for step in range(2):
            if rank == 0:
                payload = PPTransportPayload()
                if step == 0:
                    embedding = torch.arange(15).reshape(3, 5)[:, :4]
                    embedding.modality = "image"
                    payload.set_multimodal_embeddings(
                        [embedding], torch.tensor([False, True, True, True])
                    )
            else:
                tensors, handles, postprocess = group.irecv_tensor_dict(
                    all_gather_tensors=policy
                )
                for handle in handles:
                    handle.wait()
                for callback in postprocess:
                    callback()
                payload = PPTransportPayload(tensors)
                torch.testing.assert_close(
                    tensors[PPTransportDataType.TOPK_INDICES.value],
                    torch.full((4, 2), rank - 1, dtype=torch.int32),
                )
            if rank < 2:
                output = IntermediateTensors({"hidden_states": torch.zeros(4, 4)})
                add_pp_transport_tensor(
                    output,
                    PPTransportDataType.TOPK_INDICES,
                    torch.full((4, 2), rank, dtype=torch.int32),
                )
                handles = group.isend_tensor_dict(
                    payload.relay(output).tensors, all_gather_tensors=policy
                )
                for handle in handles:
                    handle.wait()
            else:
                result = payload.get_multimodal_embeddings()
                if step == 0:
                    assert result is not None
                    embeddings, mask = result
                    torch.testing.assert_close(
                        embeddings, [torch.arange(15).reshape(3, 5)[:, :4]]
                    )
                    assert embeddings[0].modality == "image"
                    torch.testing.assert_close(
                        mask, torch.tensor([False, True, True, True])
                    )
                else:
                    assert result is None
    finally:
        group.destroy()
        dist.destroy_process_group()


def test_three_stage_tensor_dict_transport(tmp_path):
    mp.spawn(_run_transport_stage, args=((tmp_path / "rendezvous").as_uri(),), nprocs=3)


def test_multihop_multimodal_payload():
    image = torch.arange(12).reshape(3, 4)
    image.modality = "image"
    audio = torch.ones(2, 4)
    audio.modality = "audio"
    mask = torch.tensor([False, True, True, True, False, True, True])
    payload = PPTransportPayload()
    payload.set_multimodal_embeddings([image, audio], mask)
    payload.set_target_embeddings(torch.ones(7, 4))

    for stage in range(3):
        output = IntermediateTensors({"hidden_states": torch.full((7, 4), stage)})
        add_pp_transport_tensor(
            output, PPTransportDataType.TOPK_INDICES, torch.full((7, 2), stage)
        )
        # Tensor-dict transport sends values, not Python tensor attributes.
        received = {key: value.clone() for key, value in payload.relay(output).items()}
        payload = PPTransportPayload(received)
        assert PPTransportDataType.TOPK_INDICES.value not in payload.tensors
        assert "hidden_states" not in payload.tensors
        torch.testing.assert_close(received["hidden_states"], output["hidden_states"])

    result = payload.get_multimodal_embeddings()
    assert result is not None
    embeddings, received_mask = result
    torch.testing.assert_close(embeddings, [image, audio])
    assert [embedding.modality for embedding in embeddings] == ["image", "audio"]
    assert received_mask.device.type == "cpu"
    torch.testing.assert_close(received_mask, mask)
    torch.testing.assert_close(payload.get_target_embeddings(), torch.ones(7, 4))
    # A subsequent text-only batch must not reuse the preceding batch's images.
    assert (
        PPTransportPayload(
            {"hidden_states": torch.zeros(2, 4)}
        ).get_multimodal_embeddings()
        is None
    )


def test_payload_replacement_and_invalid_multimodal_layout():
    payload = PPTransportPayload()
    payload.set_multimodal_embeddings(
        [torch.ones(2, 4), torch.zeros(2, 4)], torch.ones(4, dtype=torch.bool)
    )
    payload.set_multimodal_embeddings(
        [torch.ones(1, 4)], torch.ones(1, dtype=torch.bool)
    )
    embeddings, mask = payload.get_multimodal_embeddings()
    assert len(embeddings) == 1
    assert mask.shape == (1,)
    with pytest.raises(ValueError, match="CPU bool"):
        payload.set_multimodal_embeddings([], torch.zeros(1, dtype=torch.int32))

    prefix = PPTransportDataType.MULTIMODAL_EMBEDDINGS.value
    payload.tensors[f"{prefix}.2.image"] = torch.ones(1, 4)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        payload.get_multimodal_embeddings()


@pytest.mark.parametrize("compiled", [False, True])
def test_topk_copy_preserves_buffer_address(compiled):
    buffer = torch.full((8, 4), -1, dtype=torch.int32)
    address = buffer.data_ptr()

    def forward(topk):
        tensors = IntermediateTensors({})
        add_pp_transport_tensor(tensors, PPTransportDataType.TOPK_INDICES, topk)
        copy_pp_transport_tensor(tensors, PPTransportDataType.TOPK_INDICES, buffer)
        return buffer[: topk.shape[0]] + 1

    if compiled:
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    for size in (3, 2, 0):
        previous = buffer.clone()
        topk = torch.arange(size * 4, dtype=torch.int32).reshape(size, 4)
        torch.testing.assert_close(forward(topk), topk + 1)
        assert buffer.data_ptr() == address
        torch.testing.assert_close(buffer[size:], previous[size:])


@pytest.mark.parametrize(
    "topk",
    [
        torch.zeros(9, 4),
        torch.zeros(2, 5),
        torch.zeros(2, 4, 1),
        torch.zeros(2, 4, dtype=torch.int32),
        torch.tensor(1.0),
    ],
)
def test_topk_copy_rejects_invalid_layout(topk):
    tensors = IntermediateTensors({})
    add_pp_transport_tensor(tensors, PPTransportDataType.TOPK_INDICES, topk)
    buffer = torch.full((8, 4), -1.0)
    with pytest.raises(ValueError, match="Invalid pp_transport.topk_indices"):
        copy_pp_transport_tensor(tensors, PPTransportDataType.TOPK_INDICES, buffer)
    torch.testing.assert_close(buffer, torch.full_like(buffer, -1))


@pytest.mark.parametrize("default", [False, True])
def test_transport_stays_on_matching_tp_lane(default):
    policy = PPTransportAllGatherPolicy({"residual": False})
    assert policy.get("hidden_states", default) == default
    assert not policy.get("residual", default)
    for data_type in PPTransportDataType:
        assert not policy.get(data_type.value, default)
    assert not policy.get("pp_transport.multimodal_embeddings.0.image", default)
