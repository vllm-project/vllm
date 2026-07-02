# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.inputs.engine import MM_CACHE_TXN_FIELD
from vllm.multimodal.cache import MultiModalProcessorCacheTransaction
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import input_processor as input_processor_mod
from vllm.v1.engine.input_processor import InputProcessor


class FakeMMCacheTransaction:
    def __init__(self):
        self.commit_count = 0
        self.rollback_count = 0

    def commit(self):
        self.commit_count += 1

    def rollback(self):
        self.rollback_count += 1


class FakeMMCache:
    def __init__(self):
        self.removed_hashes: list[str] = []

    def remove_items(self, items):
        self.removed_hashes.extend(mm_hash for mm_hash, _ in items)


def make_multimodal_prompt(txn: FakeMMCacheTransaction):
    return {
        "type": "multimodal",
        "prompt_token_ids": [1],
        "mm_kwargs": {},
        "mm_hashes": {},
        "mm_placeholders": {},
        MM_CACHE_TXN_FIELD: txn,
    }


def make_input_processor():
    processor = object.__new__(InputProcessor)
    processor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        )
    )
    processor.model_config = SimpleNamespace(max_model_len=16)
    processor.generation_config_fields = {}

    class FakeRenderer:
        tokenizer = None

        def clear_mm_cache(self):
            raise AssertionError("process_inputs should not clear the whole MM cache")

        def get_eos_token_id(self):
            return None

    processor.renderer = FakeRenderer()
    processor._validate_params = lambda *args, **kwargs: None
    processor._validate_lora = lambda *args, **kwargs: None
    return processor


def test_mm_processor_cache_txn_rolled_back_after_post_render_error(monkeypatch):
    processor = make_input_processor()

    def raise_after_render(*args, **kwargs):
        raise ValueError("post-render validation failed")

    processor._validate_model_inputs = raise_after_render
    monkeypatch.setattr(
        input_processor_mod.current_platform,
        "validate_request",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="post-render validation failed"):
        processor.process_inputs(
            request_id="req-mm-cache-cleanup",
            prompt=make_multimodal_prompt(txn := FakeMMCacheTransaction()),
            params=SamplingParams(),
            supported_tasks=("generate",),
        )

    assert txn.rollback_count == 1
    assert txn.commit_count == 0


def test_mm_processor_cache_txn_rolled_back_when_preprocess_raises():
    processor = make_input_processor()
    fake_cache = FakeMMCache()

    def raise_after_mm_processing(*args, **kwargs):
        with MultiModalProcessorCacheTransaction(fake_cache) as txn:
            txn.record_inserted("new_image", object())
        raise ValueError("preprocess failed after MM processing")

    processor.input_preprocessor = SimpleNamespace(preprocess=raise_after_mm_processing)

    with pytest.raises(ValueError, match="preprocess failed after MM processing"):
        processor.process_inputs(
            request_id="req-mm-cache-preprocess-cleanup",
            prompt="raw prompt",
            params=SamplingParams(),
            supported_tasks=("generate",),
        )

    assert fake_cache.removed_hashes == ["new_image"]


def test_mm_processor_cache_txn_committed_after_request_validation(monkeypatch):
    processor = make_input_processor()
    processor._validate_model_inputs = lambda *args, **kwargs: None
    monkeypatch.setattr(
        input_processor_mod.current_platform,
        "validate_request",
        lambda *args, **kwargs: None,
    )

    txn = FakeMMCacheTransaction()
    processor.process_inputs(
        request_id="req-mm-cache-commit",
        prompt=make_multimodal_prompt(txn),
        params=SamplingParams(max_tokens=1),
        supported_tasks=("generate",),
    )

    assert txn.commit_count == 1
    assert txn.rollback_count == 0
