# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.engine.utils import CoreEngine, _get_engine_for_handshake_identity


def test_get_engine_for_handshake_identity_matches_expected_engine():
    engines = [CoreEngine(index=0), CoreEngine(index=1)]

    engine = _get_engine_for_handshake_identity(engines[1].identity, engines)

    assert engine is engines[1]


def test_get_engine_for_handshake_identity_ignores_non_engine_identity():
    engines = [CoreEngine(index=0)]

    engine = _get_engine_for_handshake_identity(b"GET /metrics HTTP/1.1", engines)

    assert engine is None


def test_get_engine_for_handshake_identity_rejects_unexpected_engine_rank():
    engines = [CoreEngine(index=0)]

    with pytest.raises(
        RuntimeError, match="unexpected data parallel rank: 1"
    ):
        _get_engine_for_handshake_identity((1).to_bytes(2, "little"), engines)
