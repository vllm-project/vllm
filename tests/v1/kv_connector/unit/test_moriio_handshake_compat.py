# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoRIIO handshake compatibility checks (Gap 12)."""
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MORIIO_HANDSHAKE_VERSION,
    HandshakeError,
    MoRIIOAgentMetadata,
    MoRIIOMode,
)


def _make_meta(**overrides):
    defaults = dict(
        engine_id="e1",
        agent_metadata=b"",
        kv_caches_base_addr=[0],
        num_blocks=1,
        block_len=128,
        attn_backend_name="flash_attn",
    )
    defaults.update(overrides)
    return MoRIIOAgentMetadata(**defaults)


class TestMoRIIOAgentMetadataCompat:
    """Verify that the new compatibility fields exist and serialize correctly."""

    def test_handshake_version_default_is_zero(self):
        meta = _make_meta()
        assert meta.handshake_version == 0

    def test_handshake_version_set_explicitly(self):
        meta = _make_meta(handshake_version=MORIIO_HANDSHAKE_VERSION)
        assert meta.handshake_version == MORIIO_HANDSHAKE_VERSION

    def test_mode_field_default(self):
        meta = _make_meta()
        assert meta.mode == ""

    def test_mode_field_set(self):
        meta = _make_meta(mode=MoRIIOMode.WRITE.value)
        assert meta.mode == "write"

    def test_msgspec_roundtrip_preserves_new_fields(self):
        orig = _make_meta(
            handshake_version=MORIIO_HANDSHAKE_VERSION,
            mode="write",
            block_len=256,
        )
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
        decoded = decoder.decode(encoder.encode(orig))

        assert decoded.handshake_version == MORIIO_HANDSHAKE_VERSION
        assert decoded.mode == "write"
        assert decoded.block_len == 256

    def test_omit_defaults_keeps_nonzero_version_on_wire(self):
        """handshake_version defaults to 0, so omit_defaults drops 0 but keeps
        MORIIO_HANDSHAKE_VERSION (>0) on the wire."""
        meta = _make_meta(handshake_version=MORIIO_HANDSHAKE_VERSION)
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
        decoded = decoder.decode(encoder.encode(meta))
        assert decoded.handshake_version == MORIIO_HANDSHAKE_VERSION

    def test_legacy_peer_missing_new_fields(self):
        """A legacy peer that omits handshake_version/mode decodes with
        defaults (0 and "") thanks to omit_defaults + default=0."""
        legacy_data = {
            "engine_id": "legacy",
            "agent_metadata": b"",
            "kv_caches_base_addr": [0],
            "num_blocks": 1,
            "block_len": 128,
            "attn_backend_name": "flash_attn",
        }
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
        meta = decoder.decode(encoder.encode(legacy_data))

        assert meta.handshake_version == 0
        assert meta.mode == ""

    def test_handshake_version_constant(self):
        assert isinstance(MORIIO_HANDSHAKE_VERSION, int)
        assert MORIIO_HANDSHAKE_VERSION >= 1


class TestValidateHandshakeCompat:
    """Tests for MoRIIOConnector._validate_handshake_compat."""

    def _make_connector_stub(self, mode="write", block_len=128,
                             backend_name="flash_attn"):
        conn = MagicMock()
        conn.mode = MoRIIOMode(mode)
        conn.block_len = block_len
        conn.backend_name = backend_name
        # Import the real method and bind it
        from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (  # noqa: E501
            MoRIIOConnector,
        )
        conn._validate_handshake_compat = (
            MoRIIOConnector._validate_handshake_compat.__get__(conn)
        )
        return conn

    def test_legacy_peer_version_zero_skips_checks(self):
        conn = self._make_connector_stub()
        remote = _make_meta(handshake_version=0)
        conn._validate_handshake_compat(remote, "legacy-engine")

    def test_matching_version_passes(self):
        conn = self._make_connector_stub()
        remote = _make_meta(
            handshake_version=MORIIO_HANDSHAKE_VERSION,
            mode="write",
        )
        conn._validate_handshake_compat(remote, "peer-engine")

    def test_version_mismatch_raises(self):
        conn = self._make_connector_stub()
        remote = _make_meta(handshake_version=999)
        with pytest.raises(HandshakeError, match="version mismatch"):
            conn._validate_handshake_compat(remote, "peer-engine")

    def test_block_len_mismatch_raises(self):
        conn = self._make_connector_stub(block_len=128)
        remote = _make_meta(
            handshake_version=MORIIO_HANDSHAKE_VERSION,
            block_len=256,
        )
        with pytest.raises(HandshakeError, match="Block length mismatch"):
            conn._validate_handshake_compat(remote, "peer-engine")

    def test_mode_mismatch_warns(self):
        conn = self._make_connector_stub(mode="write")
        remote = _make_meta(
            handshake_version=MORIIO_HANDSHAKE_VERSION,
            mode="read",
        )
        with patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio"
            ".moriio_connector.logger"
        ) as mock_logger:
            conn._validate_handshake_compat(remote, "peer-engine")
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "mode" in warning_msg.lower() or "differs" in warning_msg.lower()

    def test_backend_mismatch_warns(self):
        conn = self._make_connector_stub(backend_name="flash_attn")
        remote = _make_meta(
            handshake_version=MORIIO_HANDSHAKE_VERSION,
            attn_backend_name="triton",
        )
        with patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio"
            ".moriio_connector.logger"
        ) as mock_logger:
            conn._validate_handshake_compat(remote, "peer-engine")
            mock_logger.warning.assert_called()
