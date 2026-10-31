# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV continuation helpers using the shared Generate multimodal feature format."""

from typing import Any

from pydantic import TypeAdapter

from vllm.inputs import MultiModalInput
from vllm.inputs.mm_features.convert import (
    engine_input_from_features,
    extract_mm_features,
)
from vllm.inputs.mm_features.specs import (
    KV_HANDOFF_VERSION,
    MultiModalFeatures,
    MultiModalKvInfo,
)
from vllm.multimodal.inputs import MultiModalKwargsItems

_KV_INFO_ADAPTER = TypeAdapter(MultiModalKvInfo)


def validate_kv_handoff_info(info: Any) -> None:
    """Check the handoff format at engine admission, including Generate requests."""
    _KV_INFO_ADAPTER.validate_python(info, strict=True)


def export_multimodal_kv_handoff(engine_input: MultiModalInput) -> dict[str, Any]:
    """Export a media-free Generate input from the exact rendered prefill input.

    The caller binds this input to the matching KV. Version 1 supports ordinary
    sequence positions only; model capability checks belong to InputProcessor.
    """
    continuation = engine_input.copy()
    continuation["mm_kwargs"] = MultiModalKwargsItems(
        {
            modality: [None] * len(hashes)
            for modality, hashes in engine_input["mm_hashes"].items()
        }
    )
    continuation["mm_requires_kv"] = True
    continuation["mm_kv_handoff"] = MultiModalKvInfo(
        version=KV_HANDOFF_VERSION, position_state="sequence"
    )
    features = extract_mm_features(continuation)
    assert features is not None
    # Use the same bounds validation and reconstruction as Generate admission.
    engine_input_from_features(engine_input["prompt_token_ids"], features)
    return {
        "token_ids": list(engine_input["prompt_token_ids"]),
        "features": features.model_dump(mode="json"),
        "cache_salt": engine_input.get("cache_salt"),
    }


def restore_multimodal_kv_handoff(payload: dict[str, Any]) -> MultiModalInput:
    """Restore a Generate-compatible input without invoking a media processor."""
    features = MultiModalFeatures.model_validate(payload["features"], strict=True)
    if not features.requires_kv:
        raise ValueError("KV handoff requires features.requires_kv")
    validate_kv_handoff_info(features.kv_handoff)
    return engine_input_from_features(
        payload["token_ids"], features, cache_salt=payload.get("cache_salt")
    )
