# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA MTP selection reuse under DCP."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the localization kernel needs CUDA"
)

PAD = -1
WIDTH = 32


def _pack(rows, width=WIDTH):
    packed = torch.full((len(rows), width + 1), PAD, dtype=torch.int32, device="cuda")
    for i, row in enumerate(rows):
        for j, g in enumerate(row[:width]):
            packed[i, j] = g
        packed[i, width] = len(row[:width])
    return packed


def _localized_rows(source, world, rank, interleave, scratch):
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import qsa_localize_dcp_indices

    local = qsa_localize_dcp_indices(source, scratch, world, rank, interleave)
    return [
        [int(v) for v in local[i, : int(local[i, WIDTH])]]
        for i in range(local.shape[0])
    ]


def _globalize(local_ids, world, rank, interleave):
    """Invert the compact-local id, so ranks can be compared on one scale."""
    return [
        (lid // interleave) * world * interleave
        + rank * interleave
        + (lid % interleave)
        for lid in local_ids
    ]


# Prior lengths chosen to straddle compression-group boundaries at 32 and 48.
STEP0 = [
    list(range(0, 31)),  # a prefill row ending just before 32
    list(range(0, 46, 2)),  # a prefill row straddling 48
    [3, 9, 17, 40],  # a decode row
]


def test_mtp_compacts_and_reuses_packed_qsa_rows_on_two_dcp_ranks():
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMultiTokenPredictor

    class MTP:
        _iter_qsa_attentions = Qwen4ExpMultiTokenPredictor._iter_qsa_attentions

        def __init__(self, attention) -> None:
            self.layers = [SimpleNamespace(self_attn=attention)]

    source = _pack(STEP0)
    row_indices = torch.tensor([2, 0], dtype=torch.long, device="cuda")
    expected = [STEP0[2], STEP0[0]]
    seen: list[list[int]] = [[] for _ in expected]
    for rank in range(2):
        attention = SimpleNamespace(
            indexer=SimpleNamespace(skip_topk=False),
            topk_indices_buffer=source.clone(),
        )
        model = MTP(attention)
        scratch = torch.empty_like(source)
        _localized_rows(attention.topk_indices_buffer, 2, rank, 4, scratch)
        Qwen4ExpMultiTokenPredictor.compact_topk_indices(model, row_indices)
        Qwen4ExpMultiTokenPredictor.set_skip_topk(model, True)
        assert attention.indexer.skip_topk
        packed = attention.topk_indices_buffer
        assert packed[:2, -1].tolist() == [4, 31]
        frozen = packed.clone()

        local = _localized_rows(
            packed[: len(expected)], 2, rank, 4, scratch[: len(expected)]
        )
        assert torch.equal(packed, frozen)
        for row, ids in enumerate(local):
            seen[row].extend(_globalize(ids, 2, rank, 4))

    assert [sorted(ids) for ids in seen] == [sorted(ids) for ids in expected]
