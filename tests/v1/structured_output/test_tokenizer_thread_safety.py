# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that ``StructuredOutputManager`` hands out a thread-safe tokenizer.

``cached_tokenizer_from_config`` returns a single process-global instance, and
``StructuredOutputManager`` runs it concurrently across its ``self.executor``
threads (grammar compilation + the request-scoped reasoner). HF fast tokenizers
mutate shared state inside ``encode`` (``set_truncation_and_padding``), so
concurrent calls raise ``RuntimeError: Already borrowed``. The manager guards
against this by routing the tokenizer through ``maybe_make_thread_pool``.
"""

import concurrent.futures as cf

import pytest
from transformers import AutoTokenizer

from vllm.config import StructuredOutputsConfig, VllmConfig
from vllm.config.model import ModelConfig
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tokenizers.hf import ThreadSafeHFTokenizerMixin
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

TOKENIZER = "gpt2"
_N_THREADS = 32
_N_ITERS = 200
_TEXT = "hello world this is a concurrency test " * 4


def _make_manager() -> StructuredOutputManager:
    vllm_config = VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER),
        structured_outputs_config=StructuredOutputsConfig(backend="guidance"),
    )
    return StructuredOutputManager(vllm_config)


def _hammer(tokenizer) -> list[str]:
    """Encode concurrently, toggling truncation so each call mutates the
    underlying Rust tokenizer. Returns the exceptions raised, if any."""
    errors: list[str] = []

    def work(_: int) -> None:
        try:
            for j in range(_N_ITERS):
                if j % 2 == 0:
                    tokenizer(_TEXT, truncation=True, max_length=8)
                else:
                    tokenizer(_TEXT)
        except Exception as e:  # noqa: BLE001 - record, don't fail the worker
            errors.append(repr(e))

    with cf.ThreadPoolExecutor(max_workers=_N_THREADS) as ex:
        list(ex.map(work, range(_N_THREADS)))
    return errors


def test_manager_tokenizer_is_thread_safe_wrapped():
    manager = _make_manager()
    assert isinstance(manager.tokenizer, ThreadSafeHFTokenizerMixin)


def test_manager_does_not_mutate_shared_cache():
    # The shared, process-global cache entry must stay un-wrapped: the manager
    # `copy.copy`s before wrapping so the in-place class swap can't leak.
    manager = _make_manager()
    shared = cached_tokenizer_from_config(model_config=manager.vllm_config.model_config)
    assert not isinstance(shared, ThreadSafeHFTokenizerMixin)
    assert manager.tokenizer is not shared


def test_manager_tokenizer_survives_concurrent_encode():
    manager = _make_manager()
    errors = _hammer(manager.tokenizer)
    assert errors == [], f"pooled tokenizer raced under concurrency: {errors[:3]}"


def test_raw_fast_tokenizer_is_thread_unsafe():
    """Control: the same workload reliably trips an un-pooled fast tokenizer,
    which is the failure mode the manager's pooling prevents. Skipped (not
    failed) if the race can't be reproduced, to avoid coupling CI to the
    timing of transformers' internals."""
    raw = AutoTokenizer.from_pretrained(TOKENIZER)
    assert raw.is_fast
    for _ in range(3):
        errors = _hammer(raw)
        if errors:
            assert all("Already borrowed" in e for e in errors)
            return
    pytest.skip("could not reproduce the 'Already borrowed' race here")
