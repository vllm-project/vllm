# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the supervisor-side Elastic EP control channel.

The control channel's I/O contract is the msgpack tagged-union message contract
(``ScaleUp``/``ScaleDown``/``ScaleDownComplete`` in, ``Ack``/
``ScaleUpResult``/``Ok``/``Err`` out). Guard against: wrong sequencing
(``ScaleDownComplete`` without a pending scale-down), invalid sizes, and malformed
payloads. The engine manager is mocked so no Ray or models are needed.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import msgspec
import pytest
import zmq

from vllm.v1.engine.elastic_ep_control import (
    Ack,
    ControlChannelServer,
    ControlMessageType,
    Err,
    Ok,
    ScaleDown,
    ScaleDownComplete,
    ScaleUp,
    ScaleUpResult,
)


def _make_config() -> Any:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_master_ip="127.0.0.1",
            data_parallel_master_port=0,
            _data_parallel_master_port_list=[],
            _coord_store_port=0,
            data_parallel_size_local=1,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        model_config=SimpleNamespace(get_num_experts=lambda: 8),
    )


def _make_manager(local: int = 4, remote: int = 0) -> Mock:
    manager = Mock()
    manager.local_engine_actors = [object() for _ in range(local)]
    manager.remote_engine_actors = [object() for _ in range(remote)]
    return manager


def _decode(payload: bytes) -> ControlMessageType:
    return msgspec.msgpack.decode(payload, type=ControlMessageType)


@pytest.fixture
def control_channel():
    manager = _make_manager()
    config = _make_config()
    server = ControlChannelServer(manager, config)
    address = server.bind("127.0.0.1")
    server.start()
    ctx = zmq.Context()
    dealer = ctx.socket(zmq.DEALER)
    dealer.RCVTIMEO = 5000
    dealer.connect(address)
    yield manager, config, dealer
    server.close()
    dealer.close(linger=0)
    ctx.term()


def test_scale_up_acks_before_running_scale_up(control_channel) -> None:
    manager, config, dealer = control_channel

    def _scale_up(config: Any, new_size: int, num_redundant_experts: int) -> None:
        manager.local_engine_actors.extend([object() for _ in range(4)])

    manager.scale_up_elastic_ep.side_effect = _scale_up

    dealer.send(msgspec.msgpack.encode(ScaleUp(8)))
    first = _decode(dealer.recv())
    assert isinstance(first, Ack), first
    assert first.new_data_parallel_master_port > 0
    assert len(first.new_data_parallel_master_port_list) == 4
    assert first.coord_store_port > 0

    second = _decode(dealer.recv())
    assert isinstance(second, ScaleUpResult), second
    assert second.ok
    manager.scale_up_elastic_ep.assert_called_once()
    _, new_size, _ = manager.scale_up_elastic_ep.call_args.args
    assert new_size == 8
    # The shared config used for the next spawn must track the new local size.
    assert config.parallel_config.data_parallel_size_local == 8


def test_scale_down_then_complete_frees_placement_groups(control_channel) -> None:
    manager, config, dealer = control_channel

    def _scale_down(old_size: int, new_size: int) -> None:
        for _ in range(old_size - new_size):
            manager.local_engine_actors.pop()

    manager.scale_down_elastic_ep.side_effect = _scale_down

    dealer.send(msgspec.msgpack.encode(ScaleDown(2)))
    ack = _decode(dealer.recv())
    assert isinstance(ack, Ack), ack
    manager.remove_run_refs_for_scale_down.assert_called_once_with(2)

    dealer.send(msgspec.msgpack.encode(ScaleDownComplete(2)))
    ok = _decode(dealer.recv())
    assert isinstance(ok, Ok), ok
    manager.scale_down_elastic_ep.assert_called_once_with(4, 2)
    assert config.parallel_config.data_parallel_size_local == 2


def test_invalid_sizes_and_sequencing_are_rejected(control_channel) -> None:
    manager, _config, dealer = control_channel

    # Same-size and scale-up-to-smaller are invalid.
    dealer.send(msgspec.msgpack.encode(ScaleUp(4)))
    assert isinstance(_decode(dealer.recv()), Err)
    dealer.send(msgspec.msgpack.encode(ScaleDown(4)))
    assert isinstance(_decode(dealer.recv()), Err)

    # ScaleDownComplete without a pending scale-down is invalid.
    dealer.send(msgspec.msgpack.encode(ScaleDownComplete(2)))
    assert isinstance(_decode(dealer.recv()), Err)
    manager.remove_run_refs_for_scale_down.assert_not_called()
    manager.scale_down_elastic_ep.assert_not_called()


def test_malformed_payload_gets_err(control_channel) -> None:
    _manager, _config, dealer = control_channel
    dealer.send(b"\xc1")
    assert isinstance(_decode(dealer.recv()), Err)
