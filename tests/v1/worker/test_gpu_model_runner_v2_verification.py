# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_v2_verification_method_token_is_supported():
    assert GPUModelRunner._resolve_verification_method_for_v2("token") == "token"


def test_v2_verification_method_block_falls_back_to_token(caplog):
    with caplog.at_level(logging.WARNING):
        resolved = GPUModelRunner._resolve_verification_method_for_v2("block")
    assert resolved == "token"
    assert "falling back to verification_method='token'" in caplog.text


def test_v2_verification_method_gbv_not_implemented():
    with pytest.raises(NotImplementedError):
        GPUModelRunner._resolve_verification_method_for_v2(
            "greedy_multipath_block"
        )
