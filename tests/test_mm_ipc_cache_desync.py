# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig
from vllm.multimodal.cache import (
    MultiModalProcessorSenderCache,
    MultiModalReceiverCache,
)
from vllm.multimodal.inputs import MultiModalKwargsItem

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_rejected_request_cleanup_removes_sender_only_cache_entry():
    model_config = ModelConfig(
        model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        mm_processor_cache_gb=1,
    )
    sender_cache = MultiModalProcessorSenderCache(model_config)
    receiver_cache = MultiModalReceiverCache(model_config)

    first_item = MultiModalKwargsItem.dummy()
    sender_cache.get_and_update_item((first_item, []), "image_X")
    assert sender_cache.is_cached_item("image_X")

    sender_cache.discard_sender_cache_item("image_X")
    assert not sender_cache.is_cached_item("image_X")

    second_item = MultiModalKwargsItem.dummy()
    forwarded_item, _ = sender_cache.get_and_update_item((second_item, []), "image_X")

    assert forwarded_item is second_item
    assert receiver_cache.get_and_update_item(forwarded_item, "image_X") is second_item
