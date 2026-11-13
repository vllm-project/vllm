# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.pp_transport.schema import (
    PPTransportDataType,
    PPTransportSchema,
    PPTransportTensor,
    copy_intermediate_tensors,
    make_empty_intermediate_tensors_factory,
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
                    PPTransportDataType.HIDDEN_STATES: PPTransportTensor(
                        (4,), sequence_parallel="hidden_states" in sharded_fields
                    ),
                    PPTransportDataType.RESIDUAL: PPTransportTensor(
                        (4,), sequence_parallel="residual" in sharded_fields
                    ),
                },
                tp_size=2,
            )
            transport.register_tensor(
                "downstream.route_state",
                PPTransportTensor((4,), sequence_parallel=True),
            )
            buffers = transport.make_empty_intermediate_tensors(
                8, torch.float32, torch.device("cpu")
            )
            addresses = {key: tensor.data_ptr() for key, tensor in buffers.items()}
            for step, num_tokens in enumerate((5, 2)):
                expected = {}
                for index, key in enumerate(
                    ("hidden_states", "residual", "downstream.route_state")
                ):
                    sharded = key in sharded_fields or key == "downstream.route_state"
                    rows = (num_tokens + 1) // 2 if sharded else num_tokens
                    lane_offset = lane * 100 if sharded else 0
                    expected[key] = (
                        torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
                        + step * 1000
                        + index * 200
                        + lane_offset
                    )
                if pp_group.is_first_rank:
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
            field: PPTransportTensor(
                shape, sequence_parallel=field.value == sharded_field
            )
            for field in PPTransportDataType
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
        inputs = transport.prepare_inputs(buffers, num_tokens, received)
        return transport.make_intermediate_tensors(num_tokens, **inputs.tensors)

    if compiled:
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    captures = []
    for num_tokens in (5, 2):
        values = {
            field.value: torch.full(
                (
                    (num_tokens + 1) // 2
                    if field.value == sharded_field
                    else num_tokens,
                    *shape,
                ),
                num_tokens,
                dtype=torch.float32,
            )
            for field in PPTransportDataType
        }
        output = forward(**values, num_tokens=num_tokens)
        captured = copy_intermediate_tensors(buffers, output)
        captures.append(captured)
        for key, expected in values.items():
            torch.testing.assert_close(captured[key], expected, rtol=0, atol=0)
            assert captured[key].data_ptr() == addresses[key]
    assert captures[0][sharded_field].shape == (3, *shape)
    assert captures[1][sharded_field].shape == (1, *shape)


def test_receive_preserves_padding_and_legacy_dtype_conversion():
    transport = PPTransportSchema()
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
        {
            PPTransportDataType.HIDDEN_STATES: PPTransportTensor(
                (4,), sequence_parallel=True
            )
        },
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


def test_legacy_factory_preserves_arbitrary_keys_and_buffer_overrides():
    factory = make_empty_intermediate_tensors_factory(
        ["hidden_states", "custom_state"],
        4,
        buffers={"custom_state": torch.empty(8, 2, dtype=torch.int32)},
    )
    actual = factory(3, torch.float16, torch.device("cpu"))
    torch.testing.assert_close(
        actual.tensors,
        {
            "hidden_states": torch.zeros(3, 4, dtype=torch.float16),
            "custom_state": torch.zeros(3, 2, dtype=torch.int32),
        },
        rtol=0,
        atol=0,
    )


@dataclass(frozen=True)
class AlignedTensor(PPTransportTensor):
    """Example downstream descriptor; no accelerator package is needed."""

    alignment: int = 4

    def num_rows(self, num_tokens: int, tp_size: int) -> int:
        rows = super().num_rows(num_tokens, tp_size)
        return (rows + self.alignment - 1) // self.alignment * self.alignment


class DownstreamSchema(PPTransportSchema):
    def __init__(self):
        super().__init__({"hidden_states": PPTransportTensor((4,))}, tp_size=2)
        self.register_tensor(
            "downstream.route_state",
            AlignedTensor((2,), dtype=torch.int32, sequence_parallel=True),
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_downstream_descriptor_drives_all_buffer_views(compiled):
    transport = DownstreamSchema()
    buffers = transport.make_empty_intermediate_tensors(
        20, torch.float32, torch.device("cpu")
    )
    addresses = {key: value.data_ptr() for key, value in buffers.items()}
    assert buffers["downstream.route_state"].shape == (12, 2)
    assert transport.all_gather_policy == {"downstream.route_state": False}

    def forward(hidden, route, num_tokens):
        inputs = transport.prepare_inputs(
            buffers,
            num_tokens,
            IntermediateTensors(
                {"hidden_states": hidden, "downstream.route_state": route}
            ),
        )
        return transport.make_intermediate_tensors(num_tokens, **inputs.tensors)

    if compiled:
        forward = torch.compile(forward, backend="eager", fullgraph=True)
    for num_tokens, local_rows in ((9, 8), (2, 4)):
        expected = {
            "hidden_states": torch.full((num_tokens, 4), 3.0),
            "downstream.route_state": torch.full((local_rows, 2), 7, dtype=torch.int32),
        }
        output = forward(*expected.values(), num_tokens)
        captured = copy_intermediate_tensors(buffers, output)
        torch.testing.assert_close(captured.tensors, expected, rtol=0, atol=0)
        assert {key: value.data_ptr() for key, value in captured.items()} == addresses


def test_registration_is_model_local_and_sealed_before_execution():
    transport = DownstreamSchema()
    other = PPTransportSchema()
    assert not other.has_sharded_tensors
    assert other.all_gather_policy == {}
    with pytest.raises(ValueError, match="already registered"):
        transport.register_tensor("downstream.route_state", PPTransportTensor((3,)))
    with pytest.raises(ValueError, match="nonempty strings"):
        transport.register_tensor("", PPTransportTensor((3,)))
    transport.make_empty_intermediate_tensors(8, torch.float32, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="after buffer initialization"):
        transport.register_tensor("late_field", PPTransportTensor((3,)))


def test_full_downstream_field_copies_descriptor_padding():
    transport = PPTransportSchema({"custom": AlignedTensor((2,))})
    buffers = transport.make_empty_intermediate_tensors(
        8, torch.float32, torch.device("cpu")
    )
    source = torch.arange(8, dtype=torch.float32).view(4, 2)
    inputs = transport.prepare_inputs(
        buffers, 4, IntermediateTensors({"custom": source}), num_received_tokens=3
    )
    torch.testing.assert_close(inputs["custom"], source, rtol=0, atol=0)
