# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``InputProcessor._get_mm_identifier``.

Regression coverage for https://github.com/vllm-project/vllm/issues/44939:
a same-name in-place LoRA reload (same ``lora_name`` / reused
``lora_int_id``, new ``lora_path``) must produce a distinct multi-modal
encoder-cache identifier, otherwise the request reuses the previous
adapter's stale encoder embeddings.
"""

from types import SimpleNamespace

from vllm.lora.request import LoRARequest
from vllm.v1.engine.input_processor import InputProcessor

MM_HASH = "deadbeefcafef00d"


def _identifier(lora_request, *, enable_tower_connector_lora=True, lora_config=True):
    """Call the unbound method with a minimal stub as ``self``.

    ``_get_mm_identifier`` only reads ``self.lora_config``.
    """
    cfg = (
        SimpleNamespace(enable_tower_connector_lora=enable_tower_connector_lora)
        if lora_config
        else None
    )
    stub = SimpleNamespace(lora_config=cfg)
    return InputProcessor._get_mm_identifier(stub, MM_HASH, lora_request)


def _lora(name, path, int_id=1):
    return LoRARequest(lora_name=name, lora_int_id=int_id, lora_path=path)


def test_no_lora_returns_bare_mm_hash():
    assert _identifier(None) == MM_HASH


def test_feature_disabled_returns_bare_mm_hash():
    lora = _lora("tenant", "/adapters/a")
    assert _identifier(lora, enable_tower_connector_lora=False) == MM_HASH


def test_no_lora_config_returns_bare_mm_hash():
    lora = _lora("tenant", "/adapters/a")
    assert _identifier(lora, lora_config=False) == MM_HASH


def test_identifier_includes_name_and_mm_hash():
    lora = _lora("tenant", "/adapters/a")
    identifier = _identifier(lora)
    assert identifier != MM_HASH
    assert identifier.startswith("tenant:")
    assert identifier.endswith(f":{MM_HASH}")


def test_same_name_different_path_differ():
    # The #44939 regression: in-place reload keeps the name and int_id but
    # swaps the path. The identifier must change so the encoder cache misses
    # and recomputes with the new adapter.
    a = _lora("tenant", "/adapters/a", int_id=1)
    b = _lora("tenant", "/adapters/b", int_id=1)  # same name, same id, new path
    assert _identifier(a) != _identifier(b)


def test_same_name_same_path_match():
    # An unchanged adapter must keep a stable identifier so cache hits work.
    a = _lora("tenant", "/adapters/a")
    b = _lora("tenant", "/adapters/a")
    assert _identifier(a) == _identifier(b)


def test_different_name_same_path_differ():
    a = _lora("tenant", "/adapters/a")
    b = _lora("other", "/adapters/a")
    assert _identifier(a) != _identifier(b)
