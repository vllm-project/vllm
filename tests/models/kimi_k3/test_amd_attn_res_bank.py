# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.models.kimi_k3.amd import linear
from vllm.models.kimi_k3.amd.linear import KimiLinearModel

MAX_TOKENS = 64


def _make_model() -> tuple[KimiLinearModel, torch.Tensor]:
    model = object.__new__(KimiLinearModel)
    nn.Module.__init__(model)
    bank = torch.empty(MAX_TOKENS, 3, 16)
    model.block_residual_bank = bank
    return model, bank


def test_block_residual_rows_reuse_one_bank() -> None:
    model, bank = _make_model()

    # Growing per-step token counts are what used to leave a new allocation
    # behind on every step.
    for num_tokens in (1, 17, 40, 41, 63, MAX_TOKENS):
        rows = model._block_residual_rows(num_tokens)
        assert rows.shape == (num_tokens, *bank.shape[1:])
        assert rows.is_contiguous()
        assert rows.data_ptr() == bank.data_ptr()


def test_block_residual_rows_give_dbo_ubatches_disjoint_rows(monkeypatch) -> None:
    model, bank = _make_model()
    split = 40

    monkeypatch.setattr(linear, "dbo_current_ubatch_token_offset", lambda: 0)
    first = model._block_residual_rows(split)
    monkeypatch.setattr(linear, "dbo_current_ubatch_token_offset", lambda: split)
    second = model._block_residual_rows(MAX_TOKENS - split)

    first.fill_(1)
    second.fill_(2)
    assert second.data_ptr() == bank[split].data_ptr()
    assert torch.all(bank[:split] == 1)
    assert torch.all(bank[split:] == 2)


def test_block_residual_rows_reject_rows_past_the_bank(monkeypatch) -> None:
    model, _ = _make_model()
    monkeypatch.setattr(linear, "dbo_current_ubatch_token_offset", lambda: 40)

    with pytest.raises(AssertionError):
        model._block_residual_rows(MAX_TOKENS - 40 + 1)
