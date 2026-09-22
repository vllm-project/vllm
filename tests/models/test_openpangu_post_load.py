# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenPangu root dispatch of post-load hooks."""

from unittest.mock import Mock

from torch import nn

from vllm.model_executor.models.openpangu_mtp import OpenPanguMTP
from vllm.model_executor.models.openpangu_vl import (
    OpenPanguVLForConditionalGeneration,
)


class _FakeSinkAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rebuilds = 0

    def post_weight_load(self) -> None:
        self.rebuilds += 1


def test_vl_root_delegates_to_the_language_model() -> None:
    model = object.__new__(OpenPanguVLForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = Mock()

    model.process_weights_after_loading()

    model.language_model.process_weights_after_loading.assert_called_once_with()


def test_mtp_root_rebuilds_nested_derived_tensors() -> None:
    model = object.__new__(OpenPanguMTP)
    nn.Module.__init__(model)
    sink = _FakeSinkAttention()
    model.model = nn.Sequential(nn.Sequential(sink))

    model.process_weights_after_loading()

    assert sink.rebuilds == 1
