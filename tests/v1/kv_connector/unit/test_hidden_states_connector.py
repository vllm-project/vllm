# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for ExampleHiddenStatesConnector."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector as connector,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

ExampleHiddenStatesConnector = connector.ExampleHiddenStatesConnector
pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_copy_from_kv_cache_in_chunks_bounds_staging_and_preserves_output(
    monkeypatch, dtype
):
    kv_cache = torch.arange(4 * 3 * 2 * 2, dtype=dtype).reshape(4, 3, 2, 2)
    slot_mapping = torch.tensor([0, 5, 2, 7, 9, 1, 11, 4, 8, 3, 6, 10])
    num_tokens = 10
    output = torch.empty((num_tokens, 2, 2), dtype=kv_cache.dtype)
    bytes_per_token = 2 * 2 * kv_cache.element_size()
    staging_tokens = 3

    chunk_slots = []
    original_extract = connector.extract_from_kv_cache

    def recorded_extract(kv_cache, slot_mapping, num_tokens):
        chunk_slots.append(slot_mapping.tolist())
        return original_extract(kv_cache, slot_mapping, num_tokens)

    monkeypatch.setattr(connector, "extract_from_kv_cache", recorded_extract)
    connector._copy_from_kv_cache_in_chunks(
        kv_cache,
        slot_mapping,
        output,
        max_device_staging_bytes=staging_tokens * bytes_per_token,
    )

    expected = kv_cache.flatten(0, 1)[slot_mapping[:num_tokens]]
    torch.testing.assert_close(output, expected)
    assert chunk_slots == [[0, 5, 2], [7, 9, 1], [11, 4, 8], [3]]


@pytest.mark.parametrize("budget", [0, 7])
def test_copy_from_kv_cache_in_chunks_rejects_invalid_budget(budget):
    kv_cache = torch.empty((1, 2, 1, 2), dtype=torch.float32)
    slot_mapping = torch.tensor([0, 1])
    output = torch.empty((2, 1, 2), dtype=kv_cache.dtype)

    with pytest.raises(ValueError, match="max_device_staging_bytes"):
        connector._copy_from_kv_cache_in_chunks(
            kv_cache, slot_mapping, output, max_device_staging_bytes=budget
        )


def test_submit_async_write_uses_bounded_gather_chunks(monkeypatch, tmp_path):
    kv_cache = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2)
    token_ids = torch.arange(10)
    bytes_per_token = 2 * 2 * kv_cache.element_size()
    timeline = []

    class FakeEvent:
        def record(self, stream=None):
            timeline.append(("event", stream))

    class FakeStream:
        def wait_event(self, event):
            timeline.append(("wait", event))

    fake_stream = FakeStream()
    fake_future = MagicMock()
    fake_executor = MagicMock()
    fake_executor.submit.return_value = fake_future
    instance = object.__new__(connector.ExampleHiddenStatesConnector)
    instance._is_tp_rank_zero = True
    instance._kv_cache = kv_cache
    instance._block_size = kv_cache.shape[1]
    instance._max_device_staging_bytes = 3 * bytes_per_token
    instance._get_copy_stream = lambda: fake_stream
    instance._executor = fake_executor
    instance._req_futures = {}
    instance._req_copy_events = {}
    instance._lock_fds = {}
    instance.use_lock = False

    original_empty = torch.empty
    original_empty_like = torch.empty_like
    original_extract = connector.extract_from_kv_cache

    def unpinned_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    def unpinned_empty_like(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty_like(*args, **kwargs)

    def recorded_extract(kv_cache, slot_mapping, num_tokens):
        timeline.append(("gather", num_tokens))
        return original_extract(kv_cache, slot_mapping, num_tokens)

    monkeypatch.setattr(connector.torch, "empty", unpinned_empty)
    monkeypatch.setattr(connector.torch, "empty_like", unpinned_empty_like)
    monkeypatch.setattr(
        connector, "extract_from_kv_cache", MagicMock(side_effect=recorded_extract)
    )
    monkeypatch.setattr(connector.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        connector.torch.cuda, "stream", lambda stream: nullcontext(stream)
    )

    pending = connector.PendingSave(
        req_id="request-0",
        filename=str(tmp_path / "hidden-states.safetensors"),
        token_ids=token_ids,
        block_ids=[0, 1, 2, 3],
    )
    instance._submit_async_write(pending)

    assert [item[1] for item in timeline if item[0] == "gather"] == [3, 3, 3, 1]
    assert timeline[-1] == ("event", fake_stream)
    fake_executor.submit.assert_called_once()

    tensors = fake_executor.submit.call_args.args[1]
    expected_slots = torch.arange(12)[: token_ids.numel()]
    expected = kv_cache.flatten(0, 1)[expected_slots]
    torch.testing.assert_close(tensors["hidden_states"], expected)


def _full(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )


def _hidden(block_size: int) -> HiddenStateCacheSpec:
    return HiddenStateCacheSpec(
        block_size=block_size, num_kv_heads=6, head_size=2048, dtype=torch.bfloat16
    )


def _config(*specs):
    """Minimal stand-in exposing only ``kv_cache_groups`` (all the helpers read)."""
    return SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[f"layer.{i}"], kv_cache_spec=spec)
            for i, spec in enumerate(specs)
        ]
    )


# ---- _find_cache_kv_group_id ------------------------------------------------


def test_find_group_id_none_config_returns_zero():
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(None) == 0


def test_find_group_id_single_non_hidden_group_returns_zero():
    # Uniform (dense) model: one group, no HiddenStateCacheSpec -> group 0.
    cfg = _config(_full(16))
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == 0


def test_find_group_id_locates_hidden_group_when_not_first():
    # Hybrid layout: the hidden-states group is not group 0.
    cfg = _config(_full(528), _hidden(22), _full(528))
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == 1


def test_find_group_id_locates_hidden_group_last():
    cfg = _config(_full(528), _full(528), _hidden(22))
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == 2


def test_find_group_id_raises_when_no_hidden_group_and_multiple_groups():
    cfg = _config(_full(16), _full(16))
    with pytest.raises(ValueError, match="Could not uniquely identify"):
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg)


def test_find_group_id_raises_when_multiple_hidden_groups():
    cfg = _config(_hidden(22), _hidden(22))
    with pytest.raises(ValueError, match="Could not uniquely identify"):
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg)


# ---- _get_cache_block_size --------------------------------------------------


def test_get_block_size_reads_hidden_group_spec_not_global():
    # Hidden group keeps block size 22; the global is bumped to 528 for hybrids.
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=528))
    cfg = _config(_full(528), _hidden(22))
    block_size = ExampleHiddenStatesConnector._get_cache_block_size(
        vllm_config, cfg, cache_kv_group_id=1
    )
    assert block_size == 22


def test_get_block_size_falls_back_to_cache_config_when_no_kv_cache_config():
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    block_size = ExampleHiddenStatesConnector._get_cache_block_size(
        vllm_config, None, cache_kv_group_id=0
    )
    assert block_size == 16


# ---- MLA-verifier absorption ------------------------------------------------


def test_find_group_id_errors_clearly_when_absorbed_by_mla_swa_verifier():
    # HiddenStateCacheSpec subclasses MLAAttentionSpec, so an MLA + sliding-
    # window MLA verifier absorbs it into the MLA group instead of isolating it.
    dt = torch.bfloat16
    spec = {
        "layers.0.mla": MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=576, dtype=dt
        ),
        "layers.1.swa": SlidingWindowMLASpec(
            block_size=64, num_kv_heads=1, head_size=576, dtype=dt, sliding_window=512
        ),
        "cache_only_layers.61": _hidden(64),
    }
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
    )
    groups = get_kv_cache_groups(vllm_config, spec)
    assert not any(isinstance(g.kv_cache_spec, HiddenStateCacheSpec) for g in groups)
    cfg = SimpleNamespace(kv_cache_groups=groups)
    with pytest.raises(ValueError, match="MLA verifiers are unsupported"):
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg)
