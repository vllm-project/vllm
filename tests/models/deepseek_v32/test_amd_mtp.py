# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``compact_topk_indices`` on the ROCm DeepSeek-V3.2 / GLM MTP drafter.

With ``index_share_for_mtp_iteration``, draft step 0 writes top-k rows for
every query token and the proposer compacts each request's last-token row to
the front of the buffer so steps 1+ can reuse it. The ROCm drafter keeps
``topk_indices_buffer`` on ``self_attn`` itself, not under ``self_attn.mla_attn``
as the generic drafter does, so the generic method body would find no buffer
and silently leave it uncompacted.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.deepseek_mtp import DeepSeekMultiTokenPredictor
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="the amd DeepSeek-V3.2 MTP drafter is ROCm-only",
)


def _amd_predictor(buffers: list[torch.Tensor]):
    # Imported here: the amd model module pulls ROCm-only kernels at import.
    from vllm.models.deepseek_v32.amd.mtp import DeepseekV32MultiTokenPredictor

    predictor = object.__new__(DeepseekV32MultiTokenPredictor)
    torch.nn.Module.__init__(predictor)
    predictor.layers = {
        str(i): SimpleNamespace(
            mtp_block=SimpleNamespace(
                self_attn=SimpleNamespace(topk_indices_buffer=buf, skip_topk=False)
            )
        )
        for i, buf in enumerate(buffers)
    }
    return predictor


def _generic_predictor(buffers: list[torch.Tensor]) -> DeepSeekMultiTokenPredictor:
    predictor = object.__new__(DeepSeekMultiTokenPredictor)
    torch.nn.Module.__init__(predictor)
    predictor.layers = {
        str(i): SimpleNamespace(
            mtp_block=SimpleNamespace(
                self_attn=SimpleNamespace(
                    mla_attn=SimpleNamespace(topk_indices_buffer=buf, skip_topk=False)
                )
            )
        )
        for i, buf in enumerate(buffers)
    }
    return predictor


def _buffer(rows: int = 8, topk: int = 4) -> torch.Tensor:
    return torch.arange(rows * topk, dtype=torch.int32).reshape(rows, topk)


def test_compact_topk_indices_gathers_rows_to_front():
    buf = _buffer()
    before = buf.clone()
    slot_ids = torch.tensor([5, 2, 7])

    _amd_predictor([buf]).compact_topk_indices(slot_ids)

    assert torch.equal(buf[:3], before[slot_ids])
    # Rows past the compacted prefix are left alone.
    assert torch.equal(buf[3:], before[3:])


def test_compact_topk_indices_covers_every_mtp_layer():
    bufs = [_buffer(), _buffer() + 100]
    before = [b.clone() for b in bufs]
    slot_ids = torch.tensor([6, 1])

    _amd_predictor(bufs).compact_topk_indices(slot_ids)

    for buf, orig in zip(bufs, before):
        assert torch.equal(buf[:2], orig[slot_ids])


def test_compact_topk_indices_matches_generic_drafter():
    """Same buffer, same slots: the two drafters must compact identically."""
    amd_buf, generic_buf = _buffer(), _buffer()
    slot_ids = torch.tensor([3, 0, 6, 6])

    _amd_predictor([amd_buf]).compact_topk_indices(slot_ids)
    _generic_predictor([generic_buf]).compact_topk_indices(slot_ids)

    assert torch.equal(amd_buf, generic_buf)


def test_amd_drafter_supports_index_sharing():
    """The proposer only shares top-k indices when both methods exist."""
    from vllm.models.deepseek_v32.amd.mtp import DeepseekV32MultiTokenPredictor

    assert hasattr(DeepseekV32MultiTokenPredictor, "set_skip_topk")
    assert hasattr(DeepseekV32MultiTokenPredictor, "compact_topk_indices")
