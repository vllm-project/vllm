# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.pp_transport import (
    PP_TRANSPORT_REGISTRY,
    PPTransport,
    PPTransportAllGatherPolicy,
    PPTransportDataType,
    PPTransportTensor,
    copy_intermediate_tensors,
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


def _run_pp2_tp2_transport(rank: int, init_method: str):
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=init_method)
    pp_group = GroupCoordinator(
        [[0, 2], [1, 3]], rank, "gloo", use_device_communicator=False
    )
    tp_group = GroupCoordinator(
        [[0, 1], [2, 3]], rank, "gloo", use_device_communicator=False
    )
    tp_group.device_communicator = DeviceCommunicatorBase(
        cpu_group=tp_group.cpu_group,
        device=torch.device("cpu"),
        device_group=tp_group.device_group,
    )
    lane = tp_group.rank_in_group
    try:
        for sharded_fields in ((), ("hidden_states",), ("residual",)):
            topk_buffer = torch.full((8, 2), -1, dtype=torch.int32)
            transport = PPTransport(
                {
                    PPTransportDataType.HIDDEN_STATES: PPTransportTensor(
                        (4,), sequence_parallel="hidden_states" in sharded_fields
                    ),
                    PPTransportDataType.RESIDUAL: PPTransportTensor(
                        (4,), sequence_parallel="residual" in sharded_fields
                    ),
                    PPTransportDataType.TOPK_INDICES: PPTransportTensor(
                        buffer=topk_buffer
                    ),
                },
                tp_size=2,
            )
            buffers = transport.make_empty_intermediate_tensors(
                8, torch.float32, torch.device("cpu")
            )
            addresses = {key: tensor.data_ptr() for key, tensor in buffers.items()}
            for step, num_tokens in enumerate((5, 2)):
                expected = {}
                for index, key in enumerate(("hidden_states", "residual")):
                    rows = (
                        (num_tokens + 1) // 2 if key in sharded_fields else num_tokens
                    )
                    lane_offset = lane * 100 if key in sharded_fields else 0
                    expected[key] = (
                        torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
                        + step * 1000
                        + index * 200
                        + lane_offset
                    )
                expected["topk_indices"] = (
                    torch.arange(num_tokens * 2, dtype=torch.int32).reshape(-1, 2)
                    + lane * 100
                    + step * 1000
                )
                if pp_group.is_first_rank:
                    topk_buffer[:num_tokens].copy_(expected["topk_indices"])
                    output = transport.make_intermediate_tensors(num_tokens, **expected)
                    handles = pp_group.isend_tensor_dict(
                        output.tensors,
                        all_gather_group=tp_group,
                        all_gather_tensors=transport.all_gather_policy,
                    )
                    for handle in handles:
                        handle.wait()
                else:
                    tensors, handles, postprocess = pp_group.irecv_tensor_dict(
                        all_gather_group=tp_group,
                        all_gather_tensors=transport.all_gather_policy,
                    )
                    for handle in handles:
                        handle.wait()
                    # Replicated fields arrive as slices; lane-local fields do not.
                    assert len(postprocess) == 2 - len(sharded_fields)
                    for key, value in expected.items():
                        wire_value = value
                        if (
                            key in ("hidden_states", "residual")
                            and key not in sharded_fields
                        ):
                            wire_value = value.reshape(2, -1)[lane]
                        torch.testing.assert_close(
                            tensors[key], wire_value, rtol=0, atol=0
                        )
                    for callback in postprocess:
                        callback()
                    inputs = transport.prepare_inputs(
                        buffers, num_tokens, IntermediateTensors(tensors)
                    )
                    for key, value in expected.items():
                        torch.testing.assert_close(inputs[key], value, rtol=0, atol=0)
                        assert inputs[key].data_ptr() == addresses[key]
                    transport.restore_buffers(inputs)
                    assert torch.equal(
                        topk_buffer[:num_tokens], expected["topk_indices"]
                    )
    finally:
        tp_group.destroy()
        pp_group.destroy()
        dist.destroy_process_group()


def test_pp2_tp2_tensor_dict_transport(tmp_path):
    mp.spawn(
        _run_pp2_tp2_transport, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4
    )


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
    with pytest.raises(ValueError, match="Invalid topk_indices"):
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


@pytest.mark.parametrize("tp_size", [1, 2, 3])
def test_mixed_full_and_sharded_allocation(tp_size):
    transport = PPTransport(
        {
            PPTransportDataType.HIDDEN_STATES: PPTransportTensor((4,)),
            PPTransportDataType.RESIDUAL: PPTransportTensor(
                (4,), dtype=torch.float32, sequence_parallel=True
            ),
            PPTransportDataType.TOPK_INDICES: PPTransportTensor(
                (2,), dtype=torch.int32, sequence_parallel=True
            ),
        },
        tp_size=tp_size,
    )
    tensors = transport.make_empty_intermediate_tensors(
        7, torch.bfloat16, torch.device("cpu")
    )
    rows = (7 + tp_size - 1) // tp_size
    expected = {
        "hidden_states": torch.zeros(7, 4, dtype=torch.bfloat16),
        "residual": torch.zeros(rows, 4),
        "topk_indices": torch.zeros(rows, 2, dtype=torch.int32),
    }
    torch.testing.assert_close(tensors.tensors, expected, rtol=0, atol=0)
    assert transport.has_sharded_tensors == (tp_size > 1)
    assert transport.all_gather_policy.get("hidden_states", True)
    assert not transport.all_gather_policy.get("residual", True)
    assert not transport.all_gather_policy.get("topk_indices", True)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sharded_field", ["hidden_states", "residual", "topk_indices"])
def test_prepare_and_pack_per_field_views_preserve_addresses(compiled, sharded_field):
    topk_buffer = torch.full((7, 2), -1, dtype=torch.int32)
    transport = PPTransport(
        {
            PPTransportDataType.HIDDEN_STATES: PPTransportTensor(
                (4,), sequence_parallel=sharded_field == "hidden_states"
            ),
            PPTransportDataType.RESIDUAL: PPTransportTensor(
                (4,), dtype=torch.float64, sequence_parallel=sharded_field == "residual"
            ),
            PPTransportDataType.TOPK_INDICES: PPTransportTensor(
                buffer=topk_buffer, sequence_parallel=sharded_field == "topk_indices"
            ),
        },
        tp_size=2,
    )
    buffers = transport.make_empty_intermediate_tensors(
        7, torch.float32, torch.device("cpu")
    )
    addresses = {key: tensor.data_ptr() for key, tensor in buffers.items()}
    topk_address = topk_buffer.data_ptr()

    def forward(received, num_tokens):
        inputs = transport.prepare_inputs(buffers, num_tokens, received)
        transport.restore_buffers(inputs)
        output = transport.make_intermediate_tensors(num_tokens, **inputs.tensors)
        return inputs, output

    if compiled:
        torch.compiler.reset()
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    for step, num_tokens in enumerate((5, 2, 0, 7)):
        previous = {key: tensor.clone() for key, tensor in buffers.items()}
        previous_topk = topk_buffer.clone()
        received = IntermediateTensors(
            {
                key: torch.arange(tensor.numel(), dtype=tensor.dtype).reshape(
                    tensor.shape
                )
                + step * 100
                for key, tensor in buffers.items()
            }
        )
        profile_inputs = transport.prepare_inputs(buffers, num_tokens)
        inputs, output = forward(received, num_tokens)
        for key, buffer in buffers.items():
            rows = (num_tokens + 1) // 2 if key == sharded_field else num_tokens
            expected = received[key][:rows]
            for tensors in (profile_inputs, inputs, output):
                view = tensors[key]
                torch.testing.assert_close(view, expected, rtol=0, atol=0)
                address = (
                    topk_address
                    if tensors is output and key == "topk_indices"
                    else addresses[key]
                )
                assert view.untyped_storage().data_ptr() == address
                assert view.storage_offset() == 0
            torch.testing.assert_close(
                buffer[rows:], previous[key][rows:], rtol=0, atol=0
            )
        topk_rows = output["topk_indices"].shape[0]
        assert topk_buffer.data_ptr() == topk_address
        assert torch.equal(topk_buffer[topk_rows:], previous_topk[topk_rows:])


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("invalid", ["undersized", "shape", "dtype", "capacity"])
def test_prepare_inputs_rejects_invalid_receives(sequence_parallel, invalid):
    transport = PPTransport(
        {
            PPTransportDataType.TOPK_INDICES: PPTransportTensor(
                (2,), dtype=torch.int32, sequence_parallel=sequence_parallel
            )
        },
        tp_size=2,
    )
    buffers = transport.make_empty_intermediate_tensors(
        8, torch.float32, torch.device("cpu")
    )
    num_tokens = 9 if invalid == "capacity" else 5
    rows = (num_tokens + 1) // 2 if sequence_parallel else num_tokens
    source = torch.ones(
        rows - (invalid == "undersized"),
        3 if invalid == "shape" else 2,
        dtype=torch.int64 if invalid == "dtype" else torch.int32,
    )
    with pytest.raises(ValueError, match="topk_indices"):
        transport.prepare_inputs(
            buffers, num_tokens, IntermediateTensors({"topk_indices": source})
        )
    assert torch.equal(
        buffers["topk_indices"], torch.zeros_like(buffers["topk_indices"])
    )


@pytest.mark.parametrize("tp_size", [0, -1])
def test_transport_rejects_nonpositive_tp_size(tp_size):
    with pytest.raises(ValueError, match="positive TP size"):
        PPTransport(tp_size=tp_size)


def test_compatibility_factory_preserves_arbitrary_buffer_layouts():
    buffers = {
        "residual": torch.full((8, 2), -1, dtype=torch.float64),
        "router_state": torch.full((8, 2, 3), -1, dtype=torch.int32),
    }
    factory = make_empty_intermediate_tensors_factory(
        ["hidden_states", "residual"], 4, buffers=buffers
    )
    tensors = factory(3, torch.bfloat16, torch.device("cpu"))
    assert set(tensors.tensors) == {"hidden_states", "residual", "router_state"}
    torch.testing.assert_close(
        tensors["hidden_states"], torch.zeros(3, 4, dtype=torch.bfloat16)
    )
    for key, buffer in buffers.items():
        torch.testing.assert_close(tensors[key], torch.zeros_like(buffer[:3]))
        assert tensors[key].data_ptr() != buffer.data_ptr()
        assert torch.equal(buffer, torch.full_like(buffer, -1))


def test_legacy_receive_preserves_dtype_conversion():
    """Models with FP32 residual connections can receive into BF16 buffers."""
    factory = make_empty_intermediate_tensors_factory(["hidden_states"], 4)
    buffers = factory(8, torch.bfloat16, torch.device("cpu"))
    received = IntermediateTensors(
        {"hidden_states": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    )
    inputs = PPTransport().prepare_inputs(buffers, 3, received)
    torch.testing.assert_close(
        inputs["hidden_states"], received["hidden_states"].to(torch.bfloat16)
    )
    assert inputs["hidden_states"].data_ptr() == buffers["hidden_states"].data_ptr()


@pytest.mark.parametrize("compiled", [False, True])
def test_graph_output_copy_preserves_individual_views(compiled):
    buffers = IntermediateTensors(
        {
            "hidden_states": torch.full((8, 4), -1.0),
            "residual": torch.full((4, 4), -1.0),
            "topk_indices": torch.full((8, 2), -1, dtype=torch.int32),
        }
    )
    addresses = {key: tensor.data_ptr() for key, tensor in buffers.items()}
    copy_outputs = copy_intermediate_tensors
    if compiled:
        torch.compiler.reset()
        copy_outputs = torch.compile(copy_outputs, backend="eager", fullgraph=True)
    for step, (full_rows, sharded_rows) in enumerate(((5, 3), (2, 1), (0, 0))):
        previous = {key: tensor.clone() for key, tensor in buffers.items()}
        outputs = IntermediateTensors(
            {
                key: torch.full_like(
                    buffer[: sharded_rows if key == "residual" else full_rows], step
                )
                for key, buffer in buffers.items()
            }
        )
        views = copy_outputs(buffers, outputs)
        for key, value in outputs.items():
            torch.testing.assert_close(views[key], value, rtol=0, atol=0)
            assert views[key].untyped_storage().data_ptr() == addresses[key]
            assert views[key].storage_offset() == 0
            rows = value.shape[0]
            assert torch.equal(buffers[key][rows:], previous[key][rows:])
