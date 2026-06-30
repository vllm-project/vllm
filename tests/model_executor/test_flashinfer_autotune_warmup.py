# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import vllm.distributed.parallel_state as parallel_state
import vllm.utils.flashinfer as fi_utils
from vllm.model_executor.warmup import kernel_warmup


def test_flashinfer_autotune_includes_lm_head(monkeypatch):
    active = False
    events = []

    @contextmanager
    def fake_autotune():
        nonlocal active
        active = True
        events.append("enter")
        try:
            yield
        finally:
            active = False
            events.append("exit")

    all_hidden_states = object()
    sample_hidden_states = MagicMock()
    sample_hidden_states.numel.return_value = 1

    def dummy_run(**kwargs):
        assert active
        events.append("dummy")
        return all_hidden_states, sample_hidden_states

    def compute_logits(hidden_states):
        assert active
        assert hidden_states is sample_hidden_states
        events.append("logits")
        return object()

    def barrier():
        assert not active
        events.append("barrier")

    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        is_pooling_model=False,
        _dummy_run=MagicMock(side_effect=dummy_run),
        model=SimpleNamespace(compute_logits=MagicMock(side_effect=compute_logits)),
    )
    world = SimpleNamespace(barrier=MagicMock(side_effect=barrier))

    monkeypatch.setattr(kernel_warmup, "_FLASHINFER_USE_PERSISTENT_CACHE", False)
    monkeypatch.setattr(fi_utils, "autotune", fake_autotune)
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: world)

    kernel_warmup.flashinfer_autotune(runner)

    runner._dummy_run.assert_called_once_with(
        num_tokens=256,
        skip_eplb=True,
        is_profile=True,
    )
    runner.model.compute_logits.assert_called_once_with(sample_hidden_states)
    world.barrier.assert_called_once_with()
    assert events == ["enter", "dummy", "logits", "exit", "barrier"]


def test_flashinfer_autotune_covers_max_cudagraph_lm_head(monkeypatch):
    sample_hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    seen = []

    @contextmanager
    def fake_autotune():
        yield

    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4, 8])
        ),
        is_pooling_model=False,
        _dummy_run=lambda **_kwargs: (object(), sample_hidden_states),
        model=SimpleNamespace(compute_logits=lambda hidden: seen.append(hidden)),
    )

    monkeypatch.setattr(kernel_warmup, "_FLASHINFER_USE_PERSISTENT_CACHE", False)
    monkeypatch.setattr(fi_utils, "autotune", fake_autotune)
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    monkeypatch.setattr(
        parallel_state,
        "get_world_group",
        lambda: SimpleNamespace(barrier=lambda: None),
    )

    kernel_warmup.flashinfer_autotune(runner)

    assert len(seen) == 1
    assert seen[0].shape == (8, 4)
    torch.testing.assert_close(
        seen[0], sample_hidden_states.repeat(4, 1), rtol=0, atol=0
    )
