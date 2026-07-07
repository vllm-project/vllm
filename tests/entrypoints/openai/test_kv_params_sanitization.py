# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for CWE-918: client-supplied peer-topology fields
must be stripped from kv_transfer_params before they reach the engine."""

from vllm.entrypoints.openai.kv_transfer_params_sanitizer import (
    sanitize_kv_transfer_params,
)


class TestSanitizeKvTransferParams:
    def test_none_passthrough(self):
        assert sanitize_kv_transfer_params(None) is None

    def test_empty_dict_passthrough(self):
        assert not sanitize_kv_transfer_params({})

    def test_strips_remote_host_from_prefill(self):
        params = {
            "prefill": {
                "kv_request_id": "req-1",
                "remote_host": "attacker.example",
                "remote_port": 4444,
            }
        }
        result = sanitize_kv_transfer_params(params)
        assert result is not None
        assert result["prefill"] == {"kv_request_id": "req-1"}

    def test_strips_remote_host_from_decode(self):
        params = {
            "decode": {
                "kv_request_id": "req-1",
                "remote_host": "attacker.example",
                "remote_port": 4444,
            }
        }
        result = sanitize_kv_transfer_params(params)
        assert result is not None
        assert result["decode"] == {"kv_request_id": "req-1"}

    def test_preserves_kv_request_id(self):
        params = {
            "prefill": {
                "kv_request_id": "req-42",
                "remote_host": "10.0.0.1",
                "remote_port": 7777,
            }
        }
        result = sanitize_kv_transfer_params(params)
        assert result is not None
        assert result["prefill"]["kv_request_id"] == "req-42"

    def test_preserves_non_topology_keys(self):
        params = {
            "prefill": {
                "kv_request_id": "req-1",
                "custom_key": "value",
                "remote_host": "x",
                "remote_port": 1,
            }
        }
        result = sanitize_kv_transfer_params(params)
        assert result is not None
        assert result["prefill"]["custom_key"] == "value"
        assert "remote_host" not in result["prefill"]
        assert "remote_port" not in result["prefill"]

    def test_does_not_mutate_original(self):
        original_prefill = {
            "kv_request_id": "req-1",
            "remote_host": "10.0.0.1",
            "remote_port": 7777,
        }
        params = {"prefill": original_prefill}
        sanitize_kv_transfer_params(params)
        assert "remote_host" in original_prefill

    def test_no_prefill_or_decode_passthrough(self):
        params = {"do_remote_prefill": True, "other_key": "val"}
        result = sanitize_kv_transfer_params(params)
        assert result == params

    def test_prefill_without_topology_keys_unchanged(self):
        params = {"prefill": {"kv_request_id": "req-1"}}
        result = sanitize_kv_transfer_params(params)
        assert result is not None
        assert result["prefill"] == {"kv_request_id": "req-1"}

    def test_strips_from_both_prefill_and_decode(self):
        params = {
            "prefill": {
                "kv_request_id": "req-1",
                "remote_host": "host-a",
                "remote_port": 1111,
            },
            "decode": {
                "kv_request_id": "req-2",
                "remote_host": "host-b",
                "remote_port": 2222,
            },
        }
        result = sanitize_kv_transfer_params(params)
        assert result is not None
        assert "remote_host" not in result["prefill"]
        assert "remote_port" not in result["prefill"]
        assert "remote_host" not in result["decode"]
        assert "remote_port" not in result["decode"]
        assert result["prefill"]["kv_request_id"] == "req-1"
        assert result["decode"]["kv_request_id"] == "req-2"

    def test_non_dict_prefill_left_alone(self):
        params = {"prefill": "not-a-dict"}
        result = sanitize_kv_transfer_params(params)
        assert result == {"prefill": "not-a-dict"}

    def test_top_level_remote_host_stripped(self):
        """Top-level remote_host/remote_port (used by NIXL) must also be
        stripped -- the NIXL connector reads them to initiate outbound
        connections (same CWE-918 vector)."""
        params = {
            "do_remote_prefill": True,
            "remote_host": "10.0.0.1",
            "remote_port": 5555,
        }
        result = sanitize_kv_transfer_params(params)
        assert result is not None
        assert "remote_host" not in result
        assert "remote_port" not in result
        assert result["do_remote_prefill"] is True
