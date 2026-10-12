# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.pp_transport import (
    PP_TRANSPORT_REGISTRY,
    PPTransport,
    PPTransportAllGatherPolicy,
    PPTransportDataType,
    PPTransportTensor,
    make_empty_intermediate_tensors_factory,
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
    buffer = torch.full((8, 2), -1, dtype=torch.int32)
    transport = PPTransport(
        {
            PPTransportDataType.HIDDEN_STATES: PPTransportTensor(shape=(4,)),
            PPTransportDataType.TOPK_INDICES: PPTransportTensor(
                buffer=buffer, receive=rank > 0, send=rank < 2
            ),
        },
        payload_types=(PPTransportDataType.MULTIMODAL_EMBEDDINGS,),
    )
    try:
        for step in range(2):
            if rank == 0:
                payload = transport.new_payload()
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
                payload = transport.new_payload(tensors)
                transport.restore_buffers(IntermediateTensors(tensors))
                torch.testing.assert_close(
                    buffer[:4],
                    torch.full((4, 2), rank - 1, dtype=torch.int32),
                )
            if rank < 2:
                buffer.fill_(rank)
                output = transport.make_intermediate_tensors(
                    4, hidden_states=torch.zeros(4, 4)
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
    transport = PPTransport(payload_types=(PPTransportDataType.MULTIMODAL_EMBEDDINGS,))
    payload = transport.new_payload()
    payload.set_multimodal_embeddings([image, audio], mask)

    for stage in range(3):
        output = IntermediateTensors(
            {
                "hidden_states": torch.full((7, 4), stage),
                PPTransportDataType.TOPK_INDICES.value: torch.full((7, 2), stage),
            }
        )
        # Tensor-dict transport sends values, not Python tensor attributes.
        received = {key: value.clone() for key, value in payload.relay(output).items()}
        payload = transport.new_payload(received)
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
    # A subsequent text-only batch must not reuse the preceding batch's images.
    assert (
        transport.new_payload(
            {"hidden_states": torch.zeros(2, 4)}
        ).get_multimodal_embeddings()
        is None
    )


def test_payload_replacement_and_invalid_multimodal_layout():
    payload = PPTransport(
        payload_types=(PPTransportDataType.MULTIMODAL_EMBEDDINGS,)
    ).new_payload()
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
    transport = PPTransport(
        {
            PPTransportDataType.HIDDEN_STATES: PPTransportTensor(shape=(4,)),
            PPTransportDataType.TOPK_INDICES: PPTransportTensor(buffer=buffer),
        }
    )

    def forward(topk):
        tensors = IntermediateTensors({PPTransportDataType.TOPK_INDICES.value: topk})
        transport.restore_buffers(tensors)
        output = transport.make_intermediate_tensors(
            topk.shape[0], hidden_states=topk + 1
        )
        return output[PPTransportDataType.TOPK_INDICES.value] + output["hidden_states"]

    if compiled:
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    for size in (3, 2, 0):
        previous = buffer.clone()
        topk = torch.arange(size * 4, dtype=torch.int32).reshape(size, 4)
        torch.testing.assert_close(forward(topk), topk * 2 + 1)
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
    tensors = IntermediateTensors({PPTransportDataType.TOPK_INDICES.value: topk})
    buffer = torch.full((8, 4), -1.0)
    transport = PPTransport(
        {PPTransportDataType.TOPK_INDICES: PPTransportTensor(buffer=buffer)}
    )
    with pytest.raises(ValueError, match="Invalid pp_transport.topk_indices"):
        transport.restore_buffers(tensors)
    torch.testing.assert_close(buffer, torch.full_like(buffer, -1))


@pytest.mark.parametrize("default", [False, True])
def test_transport_stays_on_matching_tp_lane(default):
    policy = PPTransportAllGatherPolicy({"residual": False})
    assert policy.get("hidden_states", default) == default
    assert not policy.get("residual", default)
    for data_type, spec in PP_TRANSPORT_REGISTRY.items():
        if spec.all_gather is not None:
            assert policy.get(data_type.value, default) == spec.all_gather
    assert not policy.get("pp_transport.multimodal_embeddings.0.image", default)


@pytest.mark.parametrize("receive,send", [(False, False), (False, True), (True, False)])
def test_stage_schema_and_buffer_direction(receive, send):
    """Capture inputs and sends can differ at shared-indexer boundaries."""
    buffer = torch.full((8, 2), -1, dtype=torch.int32)
    transport = PPTransport(
        {
            PPTransportDataType.HIDDEN_STATES: PPTransportTensor(shape=(4,)),
            PPTransportDataType.RESIDUAL: PPTransportTensor(shape=(4,)),
            PPTransportDataType.TOPK_INDICES: PPTransportTensor(
                buffer=buffer, receive=receive, send=send
            ),
        }
    )
    inputs = transport.make_empty_intermediate_tensors(
        3, torch.bfloat16, torch.device("cpu")
    )
    key = PPTransportDataType.TOPK_INDICES.value
    assert (key in inputs.tensors) == receive
    assert inputs["hidden_states"].shape == (3, 4)
    assert inputs["hidden_states"].dtype == torch.bfloat16
    assert inputs["residual"].dtype == torch.bfloat16
    if receive:
        assert inputs[key].shape == (3, 2)
        assert inputs[key].dtype == torch.int32
    transport.restore_buffers(inputs)
    torch.testing.assert_close(
        buffer[:3], torch.full((3, 2), 0 if receive else -1, dtype=torch.int32)
    )
    output = transport.make_intermediate_tensors(
        3, hidden_states=inputs["hidden_states"], residual=inputs["residual"]
    )
    assert (key in output.tensors) == send
    if send:
        assert output[key].data_ptr() == buffer.data_ptr()
        assert output[key].shape == (3, 2)


def test_payload_selection_and_unknown_fields():
    transport = PPTransport()
    payload = transport.new_payload(
        {
            "pp_transport.multimodal_embeddings.0.image": torch.ones(3, 4),
            PPTransportDataType.MULTIMODAL_MASK.value: torch.ones(3, dtype=torch.bool),
        }
    )
    assert not payload.tensors
    with pytest.raises(ValueError, match="not selected"):
        payload.set_multimodal_embeddings([], torch.zeros(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="Unregistered"):
        transport.new_payload({"pp_transport.unknown": torch.zeros(1)})
    with pytest.raises(ValueError, match="Unregistered"):
        PPTransportAllGatherPolicy().get("pp_transport.unknown")
    with pytest.raises(ValueError, match="model input schema"):
        PPTransport(payload_types=(PPTransportDataType.TOPK_INDICES,))
    with pytest.raises(ValueError, match="PP payload"):
        PPTransport(
            {PPTransportDataType.MULTIMODAL_EMBEDDINGS: PPTransportTensor((4,))}
        )


@pytest.mark.parametrize("keys", [["hidden_states"], ["hidden_states", "residual"]])
def test_base_tensor_factory_preserves_layout(keys):
    factory = make_empty_intermediate_tensors_factory(keys, 4)
    tensors = factory(3, torch.float16, torch.device("cpu"))
    assert list(tensors.tensors) == keys
    for tensor in tensors.tensors.values():
        torch.testing.assert_close(tensor, torch.zeros(3, 4, dtype=torch.float16))
