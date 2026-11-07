# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoRIIO handshake compatibility checks (Gap 12)."""
import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MORIIO_HANDSHAKE_VERSION,
    HandshakeError,
    MoRIIOAgentMetadata,
    MoRIIOMode,
)


class TestMoRIIOAgentMetadataCompat:
    """Verify that the new compatibility fields exist and serialize correctly."""

    def test_handshake_version_default(self):
        meta = MoRIIOAgentMetadata(
            engine_id="e1",
            agent_metadata=b"",
            kv_caches_base_addr=[0],
            num_blocks=1,
            block_len=128,
            attn_backend_name="flash_attn",
        )
        assert meta.handshake_version == MORIIO_HANDSHAKE_VERSION
        assert MORIIO_HANDSHAKE_VERSION >= 1

    def test_mode_field_default(self):
        meta = MoRIIOAgentMetadata(
            engine_id="e1",
            agent_metadata=b"",
            kv_caches_base_addr=[0],
            num_blocks=1,
            block_len=128,
            attn_backend_name="flash_attn",
        )
        assert meta.mode == ""

    def test_mode_field_set(self):
        meta = MoRIIOAgentMetadata(
            engine_id="e1",
            agent_metadata=b"",
            kv_caches_base_addr=[0],
            num_blocks=1,
            block_len=128,
            attn_backend_name="flash_attn",
            mode=MoRIIOMode.WRITE.value,
        )
        assert meta.mode == "write"

    def test_msgspec_roundtrip_preserves_new_fields(self):
        orig = MoRIIOAgentMetadata(
            engine_id="e1",
            agent_metadata=b"\x01\x02",
            kv_caches_base_addr=[100, 200],
            num_blocks=10,
            block_len=256,
            attn_backend_name="flash_attn",
            handshake_version=MORIIO_HANDSHAKE_VERSION,
            mode="write",
        )
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
        data = encoder.encode(orig)
        decoded = decoder.decode(data)

        assert decoded.handshake_version == MORIIO_HANDSHAKE_VERSION
        assert decoded.mode == "write"
        assert decoded.engine_id == "e1"
        assert decoded.block_len == 256

    def test_legacy_peer_missing_new_fields(self):
        """A legacy peer that omits handshake_version/mode should decode
        with defaults (0 and "" respectively) thanks to omit_defaults."""
        legacy_data = {
            "engine_id": "legacy",
            "agent_metadata": b"",
            "kv_caches_base_addr": [0],
            "num_blocks": 1,
            "block_len": 128,
            "attn_backend_name": "flash_attn",
        }
        encoder = msgspec.msgpack.Encoder()
        raw = encoder.encode(legacy_data)
        decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
        meta = decoder.decode(raw)

        assert meta.handshake_version == MORIIO_HANDSHAKE_VERSION
        assert meta.mode == ""

    def test_handshake_version_constant(self):
        assert isinstance(MORIIO_HANDSHAKE_VERSION, int)
        assert MORIIO_HANDSHAKE_VERSION >= 1
