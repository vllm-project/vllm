# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.pp_transport import PPTransport


def test_transport_rejects_unsupported_schema() -> None:
    schema = IntermediateTensors(
        {
            "hidden_states": torch.empty(8, 32, dtype=torch.bfloat16),
            "residual": torch.empty(8, 32, dtype=torch.bfloat16),
        }
    )
    with pytest.raises(ValueError, match="single hidden_states"):
        PPTransport(schema, 8, 2, torch.device("cpu"))


def test_transport_drain_waits_every_ring_slot() -> None:
    transport = PPTransport.__new__(PPTransport)
    work = [[Mock(), Mock()], None, [Mock()]]
    generic_work = Mock()
    transport._send_work = work
    transport._generic_send_work = [[generic_work]]

    transport.drain()

    for slot in work:
        if slot is not None:
            for handle in slot:
                handle.wait.assert_called_once_with()
    generic_work.wait.assert_called_once_with()
    assert transport._send_work == [None, None, None]
    assert transport._generic_send_work == []


def test_transport_reaps_completed_generic_sends() -> None:
    transport = PPTransport.__new__(PPTransport)
    metadata = Mock()
    tensor = Mock()
    tensor.is_completed.return_value = True
    transport._generic_send_work = [[metadata, tensor]]

    transport._reap_generic_send_work()

    metadata.wait.assert_called_once_with()
    tensor.wait.assert_called_once_with()
    assert transport._generic_send_work == []
