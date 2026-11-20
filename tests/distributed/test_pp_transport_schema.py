# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from operator import setitem

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.pp_transport.fields import PPTransportField
from vllm.distributed.pp_transport.ops import copy_intermediate_tensors
from vllm.distributed.pp_transport.schema import (
    PPTransportSchema,
    PPTransportTensorSpec,
)
from vllm.sequence import IntermediateTensors


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
    # Exercise Gloo even when this CPU test runs in a CUDA installation.
    tp_group.use_custom_op_call = False
    lane = tp_group.rank_in_group
    try:
        for sharded_fields in ((), ("hidden_states",), ("residual",)):
            transport = PPTransportSchema(
                {
                    PPTransportField.HIDDEN_STATES: PPTransportTensorSpec(
                        (4,), sequence_parallel="hidden_states" in sharded_fields
                    ),
                    PPTransportField.RESIDUAL: PPTransportTensorSpec(
                        (4,), sequence_parallel="residual" in sharded_fields
                    ),
                },
                tp_size=2,
            )
            transport.register_tensor(
                "downstream.route_state",
                PPTransportTensorSpec((4,), sequence_parallel=True),
                receive=not pp_group.is_first_rank,
                send=pp_group.is_first_rank,
            )
            buffers = transport.make_empty_intermediate_tensors(
                8, torch.float32, torch.device("cpu")
            )
            addresses = {key: tensor.data_ptr() for key, tensor in buffers.items()}
            for step, (num_tokens, padded_tokens) in enumerate(
                ((5, 8), (1, 8), (2, 2))
            ):
                expected = {}
                for index, key in enumerate(
                    ("hidden_states", "residual", "downstream.route_state")
                ):
                    sharded = key in sharded_fields or key == "downstream.route_state"
                    rows = (padded_tokens + 1) // 2 if sharded else num_tokens
                    lane_offset = lane * 100 if sharded else 0
                    expected[key] = (
                        torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
                        + step * 1000
                        + index * 200
                        + lane_offset
                    )
                if pp_group.is_first_rank:
                    output = IntermediateTensors(
                        {
                            key: value
                            if key in sharded_fields or key == "downstream.route_state"
                            else torch.nn.functional.pad(
                                value, (0, 0, 0, padded_tokens - num_tokens), value=-1
                            )
                            for key, value in expected.items()
                        }
                    )
                    output = transport.prepare_outputs(output, num_tokens)
                    handles = pp_group.isend_tensor_dict(
                        output.tensors,
                        all_gather_group=tp_group,
                        all_gather_tensors=transport.send_all_gather_tensors,
                    )
                    for handle in handles:
                        handle.wait()
                else:
                    tensors, handles, postprocess = pp_group.irecv_tensor_dict(
                        all_gather_group=tp_group,
                        all_gather_tensors=transport.recv_all_gather_tensors,
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
                        buffers,
                        padded_tokens,
                        IntermediateTensors(tensors),
                        num_received_tokens=num_tokens,
                    )
                    for key, value in expected.items():
                        torch.testing.assert_close(
                            inputs[key][: value.shape[0]], value, rtol=0, atol=0
                        )
                        assert inputs[key].data_ptr() == addresses[key]
    finally:
        tp_group.destroy()
        pp_group.destroy()
        dist.destroy_process_group()


def test_pp2_tp2_tensor_dict_transport(tmp_path):
    mp.spawn(
        _run_pp2_tp2_transport, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sharded_field", ["hidden_states", "residual"])
@pytest.mark.parametrize("shape", [(4,), (4, 8)])
def test_mixed_views_keep_capture_shapes_and_addresses(compiled, sharded_field, shape):
    transport = PPTransportSchema(
        {
            field: PPTransportTensorSpec(
                shape, sequence_parallel=field == sharded_field
            )
            for field in ("hidden_states", "residual")
        },
        tp_size=2,
    )
    buffers = transport.make_empty_intermediate_tensors(
        8, torch.float32, torch.device("cpu")
    )
    addresses = {key: value.data_ptr() for key, value in buffers.items()}

    def forward(hidden_states, residual, num_tokens):
        received = IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )
        return transport.prepare_inputs(buffers, num_tokens, received)

    if compiled:
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    captures = []
    for num_tokens in (5, 2):
        values = {
            field: torch.full(
                (
                    (num_tokens + 1) // 2 if field == sharded_field else num_tokens,
                    *shape,
                ),
                num_tokens,
                dtype=torch.float32,
            )
            for field in ("hidden_states", "residual")
        }
        output = forward(**values, num_tokens=num_tokens)
        captured = copy_intermediate_tensors(buffers, output)
        captures.append(captured)
        for key, expected in values.items():
            torch.testing.assert_close(captured[key], expected, rtol=0, atol=0)
            assert captured[key].data_ptr() == addresses[key]
    assert captures[0][sharded_field].shape == (3, *shape)
    assert captures[1][sharded_field].shape == (1, *shape)


@pytest.mark.parametrize("sequence_parallel", [None, False, True])
def test_receive_preserves_padding_and_legacy_dtype_conversion(sequence_parallel):
    specs = (
        None
        if sequence_parallel is None
        else {
            "residual": PPTransportTensorSpec((4,), sequence_parallel=sequence_parallel)
        }
    )
    transport = PPTransportSchema(specs, tp_size=1)
    buffers = IntermediateTensors({"residual": torch.full((8, 4), -1.0)})
    received = IntermediateTensors(
        {"residual": torch.full((3, 4), 2.0, dtype=torch.float64)}
    )
    views = transport.prepare_inputs(buffers, 4, received, num_received_tokens=3)
    torch.testing.assert_close(views["residual"][:3], torch.full((3, 4), 2.0))
    torch.testing.assert_close(views["residual"][3:], torch.full((1, 4), -1.0))
    assert views["residual"].data_ptr() == buffers["residual"].data_ptr()
    # Copy-engine receives already occupy the persistent allocation.
    alias_views = transport.prepare_inputs(buffers, 4, views, num_received_tokens=3)
    torch.testing.assert_close(alias_views.tensors, views.tensors)


def test_sharded_receive_rejects_different_stage_padding():
    transport = PPTransportSchema(
        {"hidden_states": PPTransportTensorSpec((4,), sequence_parallel=True)},
        tp_size=2,
    )
    buffers = transport.make_empty_intermediate_tensors(
        8, torch.float32, torch.device("cpu")
    )
    with pytest.raises(ValueError, match="matching padding"):
        transport.prepare_inputs(
            buffers, 6, IntermediateTensors({"hidden_states": torch.zeros(4, 4)})
        )


@pytest.mark.parametrize(
    "source",
    [torch.zeros(2, 5), torch.zeros(9, 4), torch.zeros(2, 4, dtype=torch.float64)],
)
def test_capture_copy_rejects_invalid_layout_without_changing_buffer(source):
    buffer = torch.full((8, 4), -1.0)
    with pytest.raises(ValueError, match="Invalid residual"):
        copy_intermediate_tensors(
            IntermediateTensors({"residual": buffer}),
            IntermediateTensors({"residual": source}),
        )
    torch.testing.assert_close(buffer, torch.full_like(buffer, -1))


class DownstreamSchema(PPTransportSchema):
    def __init__(self):
        super().__init__({"hidden_states": PPTransportTensorSpec((4,))}, tp_size=2)
        self.register_tensor(
            "downstream.route_state",
            PPTransportTensorSpec((2,), dtype=torch.int32, sequence_parallel=True),
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_downstream_descriptor_drives_all_buffer_views(compiled):
    transport = DownstreamSchema()
    buffers = transport.make_empty_intermediate_tensors(
        20, torch.float32, torch.device("cpu")
    )
    addresses = {key: value.data_ptr() for key, value in buffers.items()}
    assert buffers["downstream.route_state"].shape == (10, 2)
    assert transport.send_all_gather_tensors == {"downstream.route_state": False}

    def forward(hidden, route, num_tokens):
        return transport.prepare_inputs(
            buffers,
            num_tokens,
            IntermediateTensors(
                {"hidden_states": hidden, "downstream.route_state": route}
            ),
        )

    if compiled:
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    for num_tokens, local_rows in ((9, 5), (2, 1)):
        expected = {
            "hidden_states": torch.full((num_tokens, 4), 3.0),
            "downstream.route_state": torch.full((local_rows, 2), 7, dtype=torch.int32),
        }
        output = forward(
            expected["hidden_states"], expected["downstream.route_state"], num_tokens
        )
        captured = copy_intermediate_tensors(buffers, output)
        torch.testing.assert_close(captured.tensors, expected, rtol=0, atol=0)
        assert {key: value.data_ptr() for key, value in captured.items()} == addresses


def test_public_field_descriptions_preserve_directions_and_wire_names():
    schema = PPTransportSchema(tp_size=2)
    incoming, outgoing = schema.recv_specs, schema.send_specs
    sharded = PPTransportTensorSpec((4,), dtype=torch.float32, sequence_parallel=True)
    full = PPTransportTensorSpec((4,), dtype=torch.float32)
    schema.register_tensor(PPTransportField.HIDDEN_STATES, sharded, send=False)
    schema.register_tensor(PPTransportField.HIDDEN_STATES, full, receive=False)
    route = PPTransportTensorSpec((2,), dtype=torch.int32)
    schema.register_tensor("downstream.route_state", route, receive=False)
    assert incoming == {"hidden_states": sharded}
    assert outgoing == {"hidden_states": full, "downstream.route_state": route}
    assert all(type(name) is str for name in (*incoming, *outgoing))
    buffers = schema.make_empty_intermediate_tensors(8, torch.float16, "cpu")
    assert buffers[PPTransportField.HIDDEN_STATES].shape == (4, 4)
    assert buffers[PPTransportField.HIDDEN_STATES].dtype == torch.float32
    for description in (incoming, outgoing):
        with pytest.raises(TypeError):
            setitem(description, "hidden_states", route)
    with pytest.raises(ValueError, match="already registered"):
        schema.register_tensor("hidden_states", full, receive=False)


def test_registration_is_model_local():
    transport = DownstreamSchema()
    other = PPTransportSchema()
    assert not other.has_sequence_parallel_tensors
    assert other.send_all_gather_tensors == {}
    with pytest.raises(ValueError, match="already registered"):
        transport.register_tensor("downstream.route_state", PPTransportTensorSpec((3,)))
    with pytest.raises(ValueError, match="nonempty strings"):
        transport.register_tensor("", PPTransportTensorSpec((3,)))


@pytest.mark.parametrize("tp_size", [1, 2])
def test_send_padding_uses_outgoing_layout(tp_size):
    transport = PPTransportSchema(tp_size=tp_size)
    transport.register_tensor("hidden_states", PPTransportTensorSpec((4,)), send=False)
    transport.register_tensor(
        "hidden_states",
        PPTransportTensorSpec((4,), sequence_parallel=True),
        receive=False,
    )
    transport.register_tensor(
        "residual", PPTransportTensorSpec((4,), sequence_parallel=True), send=False
    )
    transport.register_tensor("residual", PPTransportTensorSpec((4,)), receive=False)
    output = IntermediateTensors(
        {
            "hidden_states": torch.arange(32 // tp_size).reshape(8 // tp_size, 4),
            "residual": torch.arange(32).reshape(8, 4),
            "pp_positions": torch.arange(8).reshape(8, 1),
        }
    )
    sent = transport.prepare_outputs(output, 1)
    for key, tensor in output.items():
        rows = 4 if key == "hidden_states" and tp_size == 2 else 1
        torch.testing.assert_close(sent[key], tensor[:rows], rtol=0, atol=0)
        assert sent[key].data_ptr() == tensor.data_ptr()


@dataclass(frozen=True)
class BoundTensor(PPTransportTensorSpec):
    buffer: torch.Tensor | None = None

    def allocate(self, num_tokens, dtype, device, tp_size):
        assert self.buffer is not None
        rows = self.num_rows(num_tokens, tp_size)
        if rows > self.buffer.shape[0]:
            raise ValueError("Model buffer capacity exceeded")
        return self.buffer[:rows]


def test_directional_cache_manifests_and_bound_model_state():
    transport = PPTransportSchema(
        {"hidden_states": PPTransportTensorSpec((4,))}, tp_size=2
    )
    state = torch.zeros(8, 3, dtype=torch.int32)
    transport.register_tensor(
        "topk", BoundTensor((3,), dtype=torch.int32, buffer=state), send=False
    )
    transport.register_tensor(
        "cache.layer2", PPTransportTensorSpec((2, 4), dtype=torch.uint8), send=False
    )
    transport.register_tensor(
        "cache.layer8", PPTransportTensorSpec((2, 4), dtype=torch.uint8), receive=False
    )
    # The same name can also change SP layout at this stage's boundary.
    transport.register_tensor(
        "mix", PPTransportTensorSpec((4,), sequence_parallel=True), send=False
    )
    transport.register_tensor("mix", PPTransportTensorSpec((4,)), receive=False)
    buffers = transport.make_empty_intermediate_tensors(
        8, torch.float32, torch.device("cpu")
    )
    assert set(buffers.tensors) == {"hidden_states", "topk", "cache.layer2", "mix"}
    assert buffers["topk"].data_ptr() == state.data_ptr()
    assert transport.recv_all_gather_tensors == {"mix": False}
    assert transport.send_all_gather_tensors == {}
