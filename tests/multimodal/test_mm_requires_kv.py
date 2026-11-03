# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side contract for multimodal inputs that require existing KV.

A decode instance without media or encoder capacity continues a multimodal
request purely from decoder KV. Such an input carries hashes and placeholder
positions but no media and sets ``mm_requires_kv``: admission accepts it with a
zero encoder budget, its block hashes equal the prefill engine's, the receiver
cache skips it, and renderers refuse to render it again. The scheduler side
(fail the request instead of computing when KV coverage is missing) is covered
in tests/v1/core/test_scheduler.py.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.entrypoints.common.offline import OfflineInferenceMixin
from vllm.exceptions import VLLMValidationError
from vllm.inputs import MultiModalInput, mm_input
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
from vllm.pooling_params import PoolingParams
from vllm.renderers.base import BaseRenderer
from vllm.renderers.params import TokenizeParams
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

PROMPT_TOKEN_IDS = [1, 99, 99, 99, 2, 99, 99, 3]


def rendered_input(image_hash: str = "a") -> MultiModalInput:
    """A processed prompt as the renderer hands it to the prefill engine."""
    return mm_input(
        prompt_token_ids=list(PROMPT_TOKEN_IDS),
        mm_kwargs={"image": [None, None]},
        mm_hashes={"image": [image_hash, "second-image"]},
        mm_placeholders={
            "image": [
                PlaceholderRange(offset=1, length=3),
                PlaceholderRange(offset=5, length=2),
            ]
        },
        cache_salt="tenant-a",
    )


def kv_only_input(image_hash: str = "a") -> MultiModalInput:
    """The same prompt as a decode instance receives it: no media, KV required."""
    prompt = rendered_input(image_hash)
    prompt["mm_requires_kv"] = True
    return prompt


def without_text_suffix(prompt: MultiModalInput) -> MultiModalInput:
    # Extend the last image span onto the final prompt token.
    prompt["mm_placeholders"] = {
        "image": [
            PlaceholderRange(offset=1, length=3),
            PlaceholderRange(offset=5, length=3),
        ]
    }
    return prompt


def test_block_hashes_match_prefill_and_differ_across_images():
    init_none_hash(sha256)

    def block_hashes(prompt: MultiModalInput):
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
    decode = block_hashes(kv_only_input("a"))
    other = block_hashes(kv_only_input("b"))
    # The continuation reproduces prefill's block hashes exactly, so the local
    # prefix cache and every hash-keyed KV consumer see the same blocks on both
    # sides, while a different image still changes every hash.
    assert prefill == decode
    assert all(a != b for a, b in zip(decode, other))


def test_receiver_cache_skips_only_kv_only_features():
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


@pytest.fixture
def admission_processor(monkeypatch):
    """A bare InputProcessor for a decode instance with no encoder capacity."""
    processor = object.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(
        max_model_len=64,
        runner_type="generate",
        is_encoder_decoder=False,
        uses_mrope=False,
    )
    processor.vllm_config = SimpleNamespace(
        model_config=processor.model_config,
        lora_config=None,
        speculative_config=None,
        ec_transfer_config=None,
        kv_transfer_config=None,
        watermark_config=None,
        parallel_config=SimpleNamespace(
            data_parallel_size=1, data_parallel_size_local=1, local_engines_only=False
        ),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
    )
    processor.lora_config = None
    processor.renderer = SimpleNamespace(
        tokenizer=None, get_eos_token_id=lambda: 2, validate_token_ids=Mock()
    )
    processor.generation_config_fields = {}
    processor.skip_prompt_length_check = False
    processor.supports_mm_inputs = True
    processor.mm_encoder_cache_size = 0
    monkeypatch.setattr(processor, "_validate_params", Mock(return_value=False))
    return processor


@pytest.mark.parametrize("encoder_cache_size", [0, 16])
def test_kv_only_input_is_admitted_without_encoder_capacity(
    admission_processor, encoder_cache_size
):
    admission_processor.mm_encoder_cache_size = encoder_cache_size
    request = admission_processor.process_inputs(
        "decode", kv_only_input(), SamplingParams(max_tokens=1), ("generate",)
    )
    assert request.prompt_token_ids == PROMPT_TOKEN_IDS
    assert request.cache_salt == "tenant-a"
    assert request.mm_features is not None
    assert [f.identifier for f in request.mm_features] == ["a", "second-image"]
    assert all(f.requires_kv and f.data is None for f in request.mm_features)


@pytest.mark.parametrize(
    ("invalid", "error"),
    [
        ("suffix", "after the last multimodal span"),
        ("media", "no media payload"),
        ("hash", "nonempty hash"),
        ("bounds", "exceeds the prompt length"),
        ("prompt_logprobs", "cannot be rescored"),
        ("pooling", "cannot be rescored"),
        ("mrope", "M-RoPE"),
    ],
)
def test_kv_only_admission_rejects_invalid_contracts(
    admission_processor, invalid, error
):
    prompt = kv_only_input()
    params: SamplingParams | PoolingParams = SamplingParams(max_tokens=1)
    if invalid == "suffix":
        without_text_suffix(prompt)
    elif invalid == "media":
        prompt["mm_kwargs"] = MultiModalKwargsItems(
            {"image": [MultiModalKwargsItem({}), None]}
        )
    elif invalid == "hash":
        prompt["mm_hashes"] = {"image": ["", "second-image"]}
    elif invalid == "bounds":
        prompt["mm_placeholders"] = {
            "image": [
                PlaceholderRange(offset=1, length=3),
                PlaceholderRange(offset=5, length=20),
            ]
        }
    elif invalid == "prompt_logprobs":
        params = SamplingParams(max_tokens=1, prompt_logprobs=1)
    elif invalid == "pooling":
        params = PoolingParams()
    else:
        admission_processor.model_config.uses_mrope = True
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
    prompt = kv_only_input()
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


def test_offline_preprocessing_rejects_only_kv_only_inputs(prompt_renderer):
    # Exercise the real LLM.generate preprocessing path, including its renderer.
    llm = OfflineInferenceMixin()
    llm.renderer = prompt_renderer
    llm.model_config = prompt_renderer.model_config
    with pytest.raises(VLLMValidationError, match="AsyncLLM.generate"):
        llm._preprocess_cmpl([kv_only_input()])  # type: ignore[list-item]
    # Pre-rendered inputs without the KV requirement keep their existing path
    # (tests/entrypoints/multimodal/llm/test_mm_cache_external_injection.py).
    prerendered = rendered_input()
    (kept,) = llm._preprocess_cmpl([prerendered])  # type: ignore[list-item]
    assert kept.get("prompt_token_ids") == prerendered["prompt_token_ids"]
    (ordinary,) = llm._preprocess_cmpl([{"prompt_token_ids": [1, 2, 3]}])
    assert ordinary["type"] == "token"
    assert ordinary["prompt_token_ids"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_async_renderer_rejects_kv_only_input(prompt_renderer):
    with pytest.raises(VLLMValidationError, match="AsyncLLM.generate"):
        await prompt_renderer.render_cmpl_async(
            [kv_only_input()], TokenizeParams(max_total_tokens=64)
        )
