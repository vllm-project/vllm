# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for kv_transfer_params validation at the API boundary."""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.engine.protocol import (
    _MAX_KV_TRANSFER_PP_SIZE,
    _MAX_KV_TRANSFER_TP_SIZE,
    validate_kv_transfer_params,
)


class TestValidateKvTransferParams:
    """Unit tests for the shared validate_kv_transfer_params function."""

    def test_none_passthrough(self):
        assert validate_kv_transfer_params(None) is None

    def test_valid_params(self):
        params = {"tp_size": 4, "pp_size": 2, "remote_port": 8080}
        assert validate_kv_transfer_params(params) is params

    def test_missing_optional_fields(self):
        params = {"do_remote_prefill": True}
        assert validate_kv_transfer_params(params) is params

    # --- tp_size ---

    def test_tp_size_zero(self):
        with pytest.raises(ValueError, match="tp_size"):
            validate_kv_transfer_params({"tp_size": 0})

    def test_tp_size_negative(self):
        with pytest.raises(ValueError, match="tp_size"):
            validate_kv_transfer_params({"tp_size": -1})

    def test_tp_size_too_large(self):
        with pytest.raises(ValueError, match="tp_size"):
            validate_kv_transfer_params({"tp_size": _MAX_KV_TRANSFER_TP_SIZE + 1})

    def test_tp_size_not_int(self):
        with pytest.raises(ValueError, match="tp_size"):
            validate_kv_transfer_params({"tp_size": "big"})

    def test_tp_size_at_max(self):
        params = {"tp_size": _MAX_KV_TRANSFER_TP_SIZE}
        assert validate_kv_transfer_params(params) is params

    # --- pp_size ---

    def test_pp_size_zero(self):
        with pytest.raises(ValueError, match="pp_size"):
            validate_kv_transfer_params({"pp_size": 0})

    def test_pp_size_too_large(self):
        with pytest.raises(ValueError, match="pp_size"):
            validate_kv_transfer_params({"pp_size": _MAX_KV_TRANSFER_PP_SIZE + 1})

    def test_pp_size_at_max(self):
        params = {"pp_size": _MAX_KV_TRANSFER_PP_SIZE}
        assert validate_kv_transfer_params(params) is params

    # --- remote_port ---

    def test_port_zero(self):
        with pytest.raises(ValueError, match="remote_port"):
            validate_kv_transfer_params({"remote_port": 0})

    def test_port_too_large(self):
        with pytest.raises(ValueError, match="remote_port"):
            validate_kv_transfer_params({"remote_port": 65536})

    def test_port_valid_range(self):
        for port in (1, 80, 443, 8080, 65535):
            params = {"remote_port": port}
            assert validate_kv_transfer_params(params) is params


class TestCompletionRequestKvTransferValidation:
    """Verify CompletionRequest rejects bad kv_transfer_params."""

    def test_rejects_huge_tp_size(self):
        from vllm.entrypoints.openai.completion.protocol import (
            CompletionRequest,
        )

        with pytest.raises(ValidationError):
            CompletionRequest(
                model="test",
                prompt="hello",
                kv_transfer_params={"tp_size": 20_000_000},
            )

    def test_accepts_valid_params(self):
        from vllm.entrypoints.openai.completion.protocol import (
            CompletionRequest,
        )

        req = CompletionRequest(
            model="test",
            prompt="hello",
            kv_transfer_params={"tp_size": 4, "remote_port": 8080},
        )
        assert req.kv_transfer_params["tp_size"] == 4

    def test_accepts_none(self):
        from vllm.entrypoints.openai.completion.protocol import (
            CompletionRequest,
        )

        req = CompletionRequest(model="test", prompt="hello")
        assert req.kv_transfer_params is None


class TestChatCompletionRequestKvTransferValidation:
    """Verify ChatCompletionRequest rejects bad kv_transfer_params."""

    def test_rejects_huge_tp_size(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                kv_transfer_params={"tp_size": 20_000_000},
            )

    def test_rejects_invalid_port(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                kv_transfer_params={"remote_port": 0},
            )


class TestResponsesRequestKvTransferValidation:
    """Verify ResponsesRequest rejects bad kv_transfer_params."""

    def test_rejects_huge_tp_size(self):
        from vllm.entrypoints.openai.responses.protocol import (
            ResponsesRequest,
        )

        with pytest.raises(ValidationError):
            ResponsesRequest(
                model="test",
                input="hello",
                kv_transfer_params={"tp_size": 20_000_000},
            )
