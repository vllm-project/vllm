# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for InputProcessor._get_mm_identifier domain separation.

Covers cross-tenant encoder-cache poisoning prevention (#66885), same-name
in-place LoRA adapter reload invalidation (#44939), and colon-delimiter
framing invariants when enable_tower_connector_lora is active.
"""

from types import SimpleNamespace

import pytest

from vllm.lora.request import LoRARequest
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.skip_global_cleanup

MM_HASH = "0f30991f73a00c08d7ab0bbdb53be39cdce4589ec6883c13d61fc5e121eb3450"
NUM_ENCODER_EMBEDS = 64


def _make_processor(
    enable_tower_connector_lora: bool | None = True,
) -> SimpleNamespace:
    """Build a minimal InputProcessor stub carrying only lora_config."""
    if enable_tower_connector_lora is None:
        return SimpleNamespace(lora_config=None)
    lora_cfg = SimpleNamespace(enable_tower_connector_lora=enable_tower_connector_lora)
    return SimpleNamespace(lora_config=lora_cfg)


def _make_lora(
    name: str = "vision-lora-a",
    path: str = "/adapters/vision-lora-a",
) -> LoRARequest:
    """Construct a LoRARequest fixture for vision-tower adapter tests."""
    return LoRARequest(lora_name=name, lora_int_id=1, lora_path=path)


def _make_encoder_request(request_id: str, identifier: str) -> SimpleNamespace:
    """Build a lightweight request stub for EncoderCacheManager lookups."""
    return SimpleNamespace(
        request_id=request_id,
        mm_features=[SimpleNamespace(identifier=identifier)],
        get_num_encoder_embeds=lambda input_id: NUM_ENCODER_EMBEDS,
    )


def test_disabled_or_missing_lora_config_returns_bare_mm_hash() -> None:
    """When tower-connector LoRA is inactive, all requests share bare mm_hash."""
    lora = _make_lora()
    proc_disabled = _make_processor(enable_tower_connector_lora=False)
    proc_no_cfg = _make_processor(enable_tower_connector_lora=None)

    assert InputProcessor._get_mm_identifier(proc_disabled, MM_HASH, None) == MM_HASH
    assert InputProcessor._get_mm_identifier(proc_disabled, MM_HASH, lora) == MM_HASH
    assert InputProcessor._get_mm_identifier(proc_no_cfg, MM_HASH, lora) == MM_HASH


def test_enabled_tags_base_and_lora_domains() -> None:
    """Active tower-connector LoRA tags both base and adapter namespaces."""
    proc = _make_processor(enable_tower_connector_lora=True)
    lora = _make_lora(name="tenant-a", path="/models/lora-a")
    expected_digest = sha256_cbor((lora.lora_name, lora.lora_path)).hex()

    base_id = InputProcessor._get_mm_identifier(proc, MM_HASH, None)
    lora_id = InputProcessor._get_mm_identifier(proc, MM_HASH, lora)

    assert base_id == f"base:{MM_HASH}"
    assert lora_id == f"lora:{expected_digest}:{MM_HASH}"
    assert base_id != lora_id


def test_base_caller_forged_uuid_cannot_poison_lora_encoder_cache() -> None:
    """Regression test for #66885: forged caller UUIDs stay in the base domain.

    Even when a base-model caller knows the public adapter name, path, and
    target image digest and supplies a crafted UUID matching the LoRA
    identifier format, EncoderCacheManager must not alias the two entries.
    """
    proc = _make_processor(enable_tower_connector_lora=True)
    lora = _make_lora()
    lora_id = InputProcessor._get_mm_identifier(proc, MM_HASH, lora)

    forged_candidates = [
        lora_id,
        f"{lora.lora_name}:{MM_HASH}",
        f"uuid:{lora_id}",
        MM_HASH,
    ]
    cache_manager = EncoderCacheManager(cache_size=8 * NUM_ENCODER_EMBEDS)

    for idx, forged_mm_hash in enumerate(forged_candidates):
        base_id = InputProcessor._get_mm_identifier(proc, forged_mm_hash, None)
        assert base_id != lora_id
        req = _make_encoder_request(f"base-caller-{idx}", base_id)
        cache_manager.allocate(req, 0)

    lora_req = _make_encoder_request("vision-lora-tenant", lora_id)
    assert not cache_manager.check_and_update_cache(lora_req, 0)


def test_same_name_inplace_reload_invalidates_identifier() -> None:
    """Regression test for #44939: re-pointing lora_path changes the digest."""
    proc = _make_processor(enable_tower_connector_lora=True)
    lora_v1 = _make_lora(name="tenant", path="/adapters/v1")
    lora_v2 = _make_lora(name="tenant", path="/adapters/v2")
    lora_v1_repeat = _make_lora(name="tenant", path="/adapters/v1")

    id_v1 = InputProcessor._get_mm_identifier(proc, MM_HASH, lora_v1)
    id_v2 = InputProcessor._get_mm_identifier(proc, MM_HASH, lora_v2)
    id_v1_repeat = InputProcessor._get_mm_identifier(proc, MM_HASH, lora_v1_repeat)

    assert id_v1 != id_v2
    assert id_v1 == id_v1_repeat


def test_colon_in_lora_name_or_mm_hash_cannot_collide() -> None:
    """Fixed-width CBOR tuple hashing prevents colon-delimiter ambiguity."""
    proc = _make_processor(enable_tower_connector_lora=True)
    lora_short = _make_lora(name="a", path="/adapters/p")
    lora_colon = _make_lora(name="a:b", path="/adapters/p")

    id_short = InputProcessor._get_mm_identifier(proc, f"b:{MM_HASH}", lora_short)
    id_colon = InputProcessor._get_mm_identifier(proc, MM_HASH, lora_colon)

    assert id_short != id_colon
