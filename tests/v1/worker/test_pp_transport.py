# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import has_aiter
from vllm.v1.worker.gpu.pp_transport import (
    PPTransport,
    dequant_fp8_per_token,
)


def test_transport_rejects_unsupported_schema() -> None:
    schema = IntermediateTensors(
        {
            "hidden_states": torch.empty(8, 32, dtype=torch.bfloat16),
            "residual": torch.empty(8, 32, dtype=torch.bfloat16),
        }
    )
    with pytest.raises(ValueError, match="single hidden_states"):
        PPTransport("stream", schema, 8, 2, torch.device("cpu"))


def test_dequant_rejects_noncanonical_scale_shape() -> None:
    src = torch.empty(4, 32, dtype=current_platform.fp8_dtype())
    scales = torch.empty(4, dtype=torch.float32)
    dst = torch.empty_like(src, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="scale shape"):
        dequant_fp8_per_token(src, scales, dst)


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


@pytest.mark.skipif(
    not current_platform.is_rocm() or not has_aiter(),
    reason="requires ROCm and AITER",
)
def test_fp8_quant_dequant_matches_bf16() -> None:
    from aiter import dynamic_per_token_scaled_quant

    torch.manual_seed(0)
    src = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    quant = torch.empty_like(src, dtype=current_platform.fp8_dtype())
    scales = torch.empty(8, 1, device="cuda", dtype=torch.float32)
    restored = torch.empty_like(src)

    dynamic_per_token_scaled_quant(quant, src, scales)
    dequant_fp8_per_token(quant, scales, restored)

    torch.testing.assert_close(restored, src, rtol=0.15, atol=0.15)
