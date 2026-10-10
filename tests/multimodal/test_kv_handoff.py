# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify media-free handoff identity, input fidelity and cold-cache semantics."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from vllm.entrypoints.common.offline import OfflineInferenceMixin
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.exceptions import VLLMValidationError
from vllm.inputs import mm_input
from vllm.multimodal.cache import (
    LruKeyReplicatedReceiverCache,
    MultiModalCacheMissError,
)
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.kv_handoff import (
    export_multimodal_kv_handoff,
    restore_multimodal_kv_handoff,
)
from vllm.renderers.base import BaseRenderer
from vllm.renderers.params import TokenizeParams
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def rendered_input(image_hash="a"):
    return mm_input(
        prompt_token_ids=[1, 99, 99, 99, 2, 99, 99, 3],
        mm_kwargs={"image": [None, None]},
        mm_hashes={"image": [image_hash, "second-image"]},
        mm_placeholders={
            "image": [
                PlaceholderRange(
                    offset=1, length=3, is_embed=torch.tensor([True, False, True])
                ),
                PlaceholderRange(offset=5, length=2),
            ]
        },
        cache_salt="tenant-a",
    )


def payload(image_hash="a"):
    state = export_multimodal_kv_handoff(rendered_input(image_hash), "model-a")
    return json.loads(json.dumps(state))


def test_roundtrip_preserves_expansion_masks_hashes_and_cache_scope():
    restored = restore_multimodal_kv_handoff(payload(), "model-a")
    original = rendered_input()
    assert restored["prompt_token_ids"] == original["prompt_token_ids"]
    assert restored["mm_hashes"] == original["mm_hashes"]
    assert restored["cache_salt"] == "tenant-a"
    assert restored["mm_requires_kv"] is True
    assert restored["mm_kwargs"] == {"image": [None, None]}
    for got, expected in zip(
        restored["mm_placeholders"]["image"], original["mm_placeholders"]["image"]
    ):
        assert (got.offset, got.length) == (expected.offset, expected.length)
        if expected.is_embed is not None:
            assert torch.equal(got.is_embed, expected.is_embed)


def test_repeated_image_identity_is_stable_and_other_images_stay_distinct():
    a = restore_multimodal_kv_handoff(payload("a"), "model-a")
    b = restore_multimodal_kv_handoff(payload("b"), "model-a")
    repeat = restore_multimodal_kv_handoff(payload("a"), "model-a")
    assert a["mm_hashes"] == repeat["mm_hashes"] != b["mm_hashes"]
    assert a["cache_salt"] == b["cache_salt"] == repeat["cache_salt"]


def test_native_block_hashes_match_prefill_and_reuse_without_request_salts():
    init_none_hash(sha256)

    def block_hashes(prompt):
        features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_hash=mm_hash,
                mm_position=position,
                requires_kv=prompt.get("mm_requires_kv", False),
            )
            for mm_hash, position in zip(
                prompt["mm_hashes"]["image"], prompt["mm_placeholders"]["image"]
            )
        ]
        return Request(
            request_id="any-request",
            prompt_token_ids=prompt["prompt_token_ids"],
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            mm_features=features,
            cache_salt=prompt["cache_salt"],
            block_hasher=get_request_block_hasher(2, sha256),
        ).block_hashes

    prefill = block_hashes(rendered_input("a"))
    decode = block_hashes(restore_multimodal_kv_handoff(payload("a"), "model-a"))
    other = block_hashes(restore_multimodal_kv_handoff(payload("b"), "model-a"))
    assert prefill == decode
    assert all(a != b for a, b in zip(decode, other))


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "model",
        "positions",
        "negative",
        "overlap",
        "bounds",
        "mask",
        "hash",
        "type",
        "empty",
        "tokens",
        "requirement",
        "compatibility",
        "mask_type",
    ],
)
def test_malformed_or_incompatible_handoff_is_rejected(change):
    state = payload()
    features = state["features"]
    positions = features["mm_placeholders"]["image"]
    if change == "version":
        features["kv_handoff"]["version"] = 2
    elif change == "model":
        features["kv_handoff"]["model_fingerprint"] = "other-model"
    elif change == "positions":
        features["kv_handoff"]["position_state"] = "mrope"
    elif change == "negative":
        positions[0]["offset"] = -1
    elif change == "overlap":
        positions[1]["offset"] = 2
    elif change == "bounds":
        positions[1]["length"] = 20
    elif change == "mask":
        positions[0]["is_embed"] = [True]
    elif change == "hash":
        features["mm_hashes"]["image"][0] = ""
    elif change == "type":
        positions[0]["offset"] = True
    elif change == "empty":
        features["mm_hashes"]["image"] = []
    elif change == "requirement":
        features.pop("requires_kv")
    elif change == "compatibility":
        features.pop("kv_handoff")
    elif change == "mask_type":
        positions[0]["is_embed"] = [1, 0, 1]
    else:
        state["token_ids"][0] = -1
    with pytest.raises(ValueError):
        restore_multimodal_kv_handoff(state, "model-a")


def test_cold_receiver_bypasses_only_explicit_kv_features():
    cache = LruKeyReplicatedReceiverCache(
        SimpleNamespace(
            get_multimodal_config=lambda: SimpleNamespace(mm_processor_cache_gb=0.01)
        )
    )
    feature = MultiModalFeatureSpec(
        None,
        "image",
        "image-a",
        PlaceholderRange(1, 3),
        mm_hash="image-a",
        requires_kv=True,
    )
    assert cache.get_and_update_features([feature])[0].data is None
    feature.requires_kv = False
    with pytest.raises(MultiModalCacheMissError):
        cache.get_and_update_features([feature])


def test_kv_requirement_survives_engine_ipc_serialization():
    feature = MultiModalFeatureSpec(
        None,
        "image",
        "a",
        PlaceholderRange(1, 3, torch.tensor([True, False, True])),
        mm_hash="a",
        requires_kv=True,
    )
    restored = MsgpackDecoder(MultiModalFeatureSpec).decode(
        MsgpackEncoder().encode(feature)
    )
    assert restored.requires_kv is True
    assert restored.data is None
    assert restored.identifier == feature.identifier
    assert torch.equal(restored.mm_position.is_embed, feature.mm_position.is_embed)


@pytest.mark.asyncio
async def test_engine_exports_once_then_restores_without_renderer_calls(monkeypatch):
    processor = object.__new__(InputProcessor)
    monkeypatch.setattr(processor, "get_external_kv_handoff_error", lambda: None)
    monkeypatch.setattr(processor, "_kv_handoff_model_fingerprint", lambda: "model-a")
    processor.model_config = SimpleNamespace(is_encoder_decoder=False)
    processor.renderer = SimpleNamespace(
        render_cmpl_async=AsyncMock(return_value=[rendered_input()])
    )
    prompt, state = await processor.prepare_multimodal_kv_handoff(
        {"prompt_token_ids": [1, 99, 2]}
    )
    assert prompt["prompt_token_ids"] == rendered_input()["prompt_token_ids"]
    restored = processor.restore_multimodal_kv_handoff(state)
    assert restored["mm_hashes"] == prompt["mm_hashes"]
    processor.renderer.render_cmpl_async.assert_awaited_once()


def test_input_processor_rejects_media_payload_in_kv_only_input(monkeypatch):
    # The native InputProcessor is also exercised by the engine API tests;
    # this admission check must reject a media payload on a KV-only request.
    processor = object.__new__(InputProcessor)
    monkeypatch.setattr(processor, "get_external_kv_handoff_error", lambda: None)
    monkeypatch.setattr(processor, "_kv_handoff_model_fingerprint", lambda: "model-a")
    prompt = restore_multimodal_kv_handoff(payload(), "model-a")
    processor._validate_external_kv_input(prompt, SamplingParams())
    prompt["mm_kwargs"] = MultiModalKwargsItems[MultiModalKwargsItem | None](
        {"image": [MultiModalKwargsItem({}), None]}
    )
    with pytest.raises(VLLMValidationError, match="no media payload"):
        processor._validate_external_kv_input(prompt, SamplingParams())


@pytest.fixture
def admission_processor(monkeypatch):
    processor = supported_handoff_processor()
    monkeypatch.setattr(processor, "_kv_handoff_model_fingerprint", lambda: "model-a")
    monkeypatch.setattr(processor, "_validate_params", Mock(return_value=False))
    processor.vllm_config.parallel_config = SimpleNamespace(
        data_parallel_size=1, data_parallel_size_local=1, local_engines_only=False
    )
    processor.vllm_config.watermark_config = None
    processor.vllm_config.cache_config = SimpleNamespace(kv_sharing_fast_prefill=False)
    processor.model_config.max_model_len = 64
    processor.model_config.runner_type = "generate"
    processor.renderer = SimpleNamespace(
        tokenizer=None, get_eos_token_id=lambda: 2, validate_token_ids=Mock()
    )
    processor.generation_config_fields = {}
    processor.skip_prompt_length_check = False
    processor.supports_mm_inputs = True
    processor.mm_encoder_cache_size = 0
    return processor


@pytest.mark.asyncio
@pytest.mark.parametrize("encoder_cache_size", [0, 16])
async def test_async_submission_preserves_kv_requirement_and_identity(
    admission_processor,
    encoder_cache_size,
):
    from vllm.v1.engine.async_llm import AsyncLLM

    processor = admission_processor
    processor.mm_encoder_cache_size = encoder_cache_size
    prompt = restore_multimodal_kv_handoff(payload(), "model-a")
    engine = SimpleNamespace(
        errored=False,
        vllm_config=processor.vllm_config,
        model_config=processor.model_config,
        input_processor=processor,
        get_supported_tasks=AsyncMock(return_value=("generate",)),
        _run_output_handler=Mock(),
        _add_request=AsyncMock(),
    )
    await AsyncLLM.add_request(engine, "decode", prompt, SamplingParams(max_tokens=1))
    engine._add_request.assert_awaited_once()
    request = engine._add_request.call_args.args[0]
    assert request.prompt_token_ids == prompt["prompt_token_ids"]
    assert request.cache_salt == "tenant-a"
    assert request.mm_features is not None
    assert [f.identifier for f in request.mm_features] == ["a", "second-image"]
    assert all(f.requires_kv and f.data is None for f in request.mm_features)


@pytest.mark.parametrize(
    ("invalid", "error"),
    [
        ("flag", "mm_requires_kv must be a boolean"),
        ("model", "model configuration does not match"),
        ("media", "no media payload"),
        ("prompt_logprobs", "without prompt logprobs"),
    ],
)
def test_zero_encoder_cache_requires_valid_kv_contract(
    admission_processor, invalid, error
):
    prompt = restore_multimodal_kv_handoff(payload(), "model-a")
    params = SamplingParams(max_tokens=1)
    if invalid == "flag":
        prompt["mm_requires_kv"] = 1  # type: ignore[typeddict-item]
    elif invalid == "model":
        prompt["mm_kv_handoff"]["model_fingerprint"] = "other-model"
    elif invalid == "media":
        prompt["mm_kwargs"] = MultiModalKwargsItems(
            {"image": [MultiModalKwargsItem({}), None]}
        )
    else:
        params.prompt_logprobs = 1
    with pytest.raises(VLLMValidationError, match=error):
        admission_processor.process_inputs("decode", prompt, params, ("generate",))


@pytest.mark.parametrize("has_media", [False, True])
def test_ordinary_multimodal_input_still_requires_encoder_capacity(
    admission_processor, has_media
):
    prompt = rendered_input()
    if has_media:
        prompt["mm_kwargs"] = MultiModalKwargsItems(
            {"image": [MultiModalKwargsItem({}), None]}
        )
    with pytest.raises(VLLMValidationError, match="pre-allocated encoder cache size 0"):
        admission_processor.process_inputs(
            "prefill", prompt, SamplingParams(max_tokens=1), ("generate",)
        )


@pytest.mark.parametrize("invalid", ["length", "token_ids"])
def test_kv_only_admission_still_validates_prompt(admission_processor, invalid):
    prompt = restore_multimodal_kv_handoff(payload(), "model-a")
    if invalid == "length":
        admission_processor.model_config.max_model_len = 4
        error = "maximum model length"
    else:
        error = "invalid token ID"
        admission_processor.renderer.validate_token_ids.side_effect = (
            VLLMValidationError(error)
        )
    with pytest.raises(VLLMValidationError, match=error):
        admission_processor.process_inputs(
            "decode", prompt, SamplingParams(max_tokens=1), ("generate",)
        )


@pytest.fixture
def prompt_renderer():
    class TestRenderer(BaseRenderer):
        def __init__(self):
            self.tokenizer = None
            self.mm_processor = None
            self.model_config = SimpleNamespace(
                max_model_len=64, encoder_config=None, is_encoder_decoder=False
            )

        def render_messages(self, *args, **kwargs):
            raise NotImplementedError

    return TestRenderer()


@pytest.mark.parametrize("restored", [False, True])
def test_offline_preprocessing_rejects_processed_multimodal_inputs(
    prompt_renderer, restored
):
    # Exercise the real LLM.generate preprocessing path, including its renderer.
    llm = OfflineInferenceMixin()
    llm.renderer = prompt_renderer
    llm.model_config = prompt_renderer.model_config
    prompt = (
        restore_multimodal_kv_handoff(payload(), "model-a")
        if restored
        else rendered_input()
    )
    with pytest.raises(VLLMValidationError, match="AsyncLLM.generate"):
        llm._preprocess_cmpl([prompt])  # type: ignore[list-item]
    (ordinary,) = llm._preprocess_cmpl([{"prompt_token_ids": [1, 2, 3]}])
    assert ordinary["type"] == "token"
    assert ordinary["prompt_token_ids"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_async_renderer_rejects_restored_input(prompt_renderer):
    prompt = restore_multimodal_kv_handoff(payload(), "model-a")
    with pytest.raises(VLLMValidationError, match="AsyncLLM.generate"):
        await prompt_renderer.render_cmpl_async(
            [prompt], TokenizeParams(max_total_tokens=64)
        )


def supported_handoff_processor():
    processor = object.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlavaForConditionalGeneration"]),
        is_encoder_decoder=False,
        uses_mrope=False,
    )
    processor.vllm_config = SimpleNamespace(
        lora_config=None,
        speculative_config=None,
        ec_transfer_config=None,
        kv_transfer_config=None,
    )
    return processor


@pytest.mark.parametrize(
    "unsupported", [None, "model", "mrope", "lora", "speculation", "encoder"]
)
def test_engine_capability_checks_model_and_input_contracts(
    unsupported,
):
    processor = supported_handoff_processor()
    model_config = processor.model_config
    config = processor.vllm_config
    if unsupported == "model":
        model_config.hf_config.architectures = ["Qwen3VLForConditionalGeneration"]
    elif unsupported == "mrope":
        model_config.uses_mrope = True
    elif unsupported == "lora":
        config.lora_config = object()
    elif unsupported == "speculation":
        config.speculative_config = object()
    elif unsupported == "encoder":
        config.ec_transfer_config = object()
    assert (processor.get_external_kv_handoff_error() is None) == (unsupported is None)


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", [None, "NixlConnector", "OtherConnector"])
@pytest.mark.parametrize("policy", ["fail", "recompute"])
async def test_handoff_export_restore_and_admission_are_transport_independent(
    monkeypatch, connector, policy
):
    processor = supported_handoff_processor()
    if connector is not None:
        processor.vllm_config.kv_transfer_config = SimpleNamespace(
            kv_connector=connector, kv_load_failure_policy=policy
        )
    monkeypatch.setattr(processor, "_kv_handoff_model_fingerprint", lambda: "model-a")
    processor.renderer = SimpleNamespace(
        render_cmpl_async=AsyncMock(return_value=[rendered_input()])
    )
    _, state = await processor.prepare_multimodal_kv_handoff(
        {"prompt_token_ids": [1, 99, 2]}
    )
    restored = processor.restore_multimodal_kv_handoff(state)
    processor._validate_external_kv_input(restored, SamplingParams())
    assert restored["mm_requires_kv"] is True


def test_python_export_is_a_generate_request_input():
    state = payload()
    request = GenerateRequest.model_validate_json(
        json.dumps({**state, "sampling_params": {"max_tokens": 1}})
    )
    assert request.features is not None
    assert request.features.requires_kv
    assert request.features.kv_handoff is not None
    assert request.features.kv_handoff["model_fingerprint"] == "model-a"
    assert request.token_ids == rendered_input()["prompt_token_ids"]
    assert request.cache_salt == "tenant-a"
    assert request.kv_transfer_params is None


@pytest.mark.parametrize(
    "info",
    [
        None,
        {"version": 2},
        {
            "version": 1,
            "model_fingerprint": "wrong-model",
            "position_state": "sequence",
        },
    ],
)
def test_native_admission_checks_source_compatibility(monkeypatch, info):
    processor = object.__new__(InputProcessor)
    monkeypatch.setattr(processor, "get_external_kv_handoff_error", lambda: None)
    monkeypatch.setattr(processor, "_kv_handoff_model_fingerprint", lambda: "model-a")
    prompt = restore_multimodal_kv_handoff(payload(), "model-a")
    if info is None:
        prompt.pop("mm_kv_handoff")
    else:
        prompt["mm_kv_handoff"] = info
    with pytest.raises(VLLMValidationError):
        processor._validate_external_kv_input(prompt, SamplingParams())
