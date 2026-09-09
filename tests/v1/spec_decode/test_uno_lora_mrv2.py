# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the native Punica mapping plans used by Uno."""

from types import SimpleNamespace

import pytest
import torch

from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
from vllm.v1.worker.gpu.spec_decode.uno_lora import (
    UnoLoRAPlanCache,
    UnoLoRAState,
    base_lora_mapping,
    base_prompt_mapping,
    draft_lora_mapping,
    wrapper_fingerprint,
)


class FakePunicaWrapper:
    """A CPU wrapper with the same buffers as PunicaWrapperGPU."""

    def __init__(self, max_tokens=32, max_loras=2, vocab_size=16):
        self.device = torch.device("cpu")
        self.max_loras = max_loras
        self.vocab_size = vocab_size
        self._token_lora_indices = torch.empty(max_tokens, dtype=torch.long)
        self._sampler_indices = torch.empty(max_tokens, dtype=torch.long)
        self._sampler_indices_padded = torch.empty(max_tokens, dtype=torch.long)
        self._embeddings_indices = torch.empty(max_tokens, dtype=torch.long)
        self.indices_len: list[int | None] = [None] * 4
        self.is_prefill = False
        self.token_mapping_meta = LoRAKernelMeta.make(
            max_loras, max_tokens, device=self.device
        )
        self.prompt_mapping_meta = LoRAKernelMeta.make(
            max_loras, max_tokens, device=self.device
        )
        self.native_calls = 0

    def update_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size):
        self.native_calls += 1
        self.is_prefill = mapping.is_prefill
        id_to_index = {
            lora_id: index
            for index, lora_id in enumerate(lora_index_to_id)
            if lora_id is not None
        }
        base = torch.tensor(
            [id_to_index[x] if x > 0 else -1 for x in mapping.index_mapping],
            dtype=torch.long,
        )
        sampler = torch.tensor(
            [id_to_index[x] if x > 0 else -1 for x in mapping.prompt_mapping],
            dtype=torch.long,
        )
        sampler_padded = torch.where(sampler == -1, max_loras - 1, sampler)
        sampler_padded = torch.arange(sampler.numel()) + (
            sampler_padded * sampler.numel()
        )
        embeddings = torch.tensor(
            [
                id_to_index[x] * vocab_size if x > 0 else 0
                for x in mapping.index_mapping
            ],
            dtype=torch.long,
        )
        lengths = [
            base.numel(),
            sampler.numel(),
            sampler_padded.numel(),
            embeddings.numel(),
        ]
        for target, value in zip(
            (
                self._token_lora_indices,
                self._sampler_indices,
                self._sampler_indices_padded,
                self._embeddings_indices,
            ),
            (base, sampler, sampler_padded, embeddings),
            strict=True,
        ):
            target[: value.shape[0]].copy_(value)
        self.indices_len[:] = lengths
        self.token_mapping_meta.prepare_tensors(self._token_lora_indices[: lengths[0]])
        self.prompt_mapping_meta.prepare_tensors(self._sampler_indices[: lengths[1]])


def _mapping(token, prompt):
    return SimpleNamespace(
        index_mapping=tuple(token),
        prompt_mapping=tuple(prompt),
        is_prefill=True,
    )


def _state(wrapper):
    fields: dict[object, object] = {}
    for name, length in zip(
        (
            "_token_lora_indices",
            "_sampler_indices",
            "_sampler_indices_padded",
            "_embeddings_indices",
        ),
        wrapper.indices_len,
        strict=True,
    ):
        fields[name] = getattr(wrapper, name)[:length].clone()
    fields["indices_len"] = tuple(wrapper.indices_len)
    fields["is_prefill"] = wrapper.is_prefill
    for owner, length in zip(
        ("token_mapping_meta", "prompt_mapping_meta"),
        wrapper.indices_len[:2],
        strict=True,
    ):
        meta = getattr(wrapper, owner)
        for name in (
            "active_lora_ids",
            "num_tokens_per_lora",
            "lora_token_start_loc",
            "no_lora_flag_cpu",
            "num_active_loras_cpu",
        ):
            fields[(owner, name)] = getattr(meta, name).clone()
        if not bool(meta.no_lora_flag_cpu[0]):
            for name in ("token_lora_mapping", "token_indices_sorted_by_lora_ids"):
                fields[(owner, name)] = getattr(meta, name)[:length].clone()
    return fields


def _native_install(cache, wrapper, kind, shape, slots, mapping):
    return cache.install(
        wrapper,
        kind,
        shape,
        slots,
        lambda: wrapper.update_metadata(mapping, list(slots), 3, 16),
    )


def _assert_state_equal(got, expected):
    assert got.keys() == expected.keys()
    for key in got:
        if isinstance(got[key], torch.Tensor):
            assert torch.equal(got[key], expected[key]), key
        else:
            assert got[key] == expected[key], key


@pytest.mark.parametrize("k", [1, 4, 8])
def test_draft_mapping_has_base_seed_and_padded_base_rows(k):
    mapping = draft_lora_mapping(3 * k + 2, 3, k, 1_000_003)
    assert mapping[:k] == (0,) + (1_000_003,) * (k - 1)
    assert mapping[k : 2 * k] == mapping[:k]
    assert mapping[3 * k :] == (0, 0)


def test_draft_mapping_rejects_truncation_and_zero_dimensions():
    with pytest.raises(ValueError):
        draft_lora_mapping(7, 2, 4, 1_000_003)
    with pytest.raises(ValueError):
        draft_lora_mapping(4, 0, 4, 1_000_003)


def test_cached_plan_restores_native_metadata_in_place():
    wrapper = FakePunicaWrapper()
    cache = UnoLoRAPlanCache()
    slots = (1_000_003, None)
    mapping = _mapping(draft_lora_mapping(8, 2, 4, 1_000_003), (0, 0))
    shape = (8, 2, 4, 1_000_003)

    assert _native_install(cache, wrapper, "draft", shape, slots, mapping) is False
    expected = _state(wrapper)
    addresses = {
        name: getattr(wrapper, name).data_ptr()
        for name in (
            "_token_lora_indices",
            "_sampler_indices",
            "_sampler_indices_padded",
            "_embeddings_indices",
        )
    }

    for name in addresses:
        getattr(wrapper, name).fill_(-99)
    for owner in ("token_mapping_meta", "prompt_mapping_meta"):
        meta = getattr(wrapper, owner)
        for name in (
            "token_lora_mapping",
            "token_indices_sorted_by_lora_ids",
            "active_lora_ids",
            "num_tokens_per_lora",
            "lora_token_start_loc",
            "num_active_loras_cpu",
        ):
            getattr(meta, name).fill_(-99)
        meta.no_lora_flag_cpu.fill_(True)
    wrapper.indices_len[:] = [0, 0, 0, 0]
    wrapper.is_prefill = False

    assert _native_install(cache, wrapper, "draft", shape, slots, mapping) is True
    assert wrapper.native_calls == 1
    _assert_state_equal(_state(wrapper), expected)
    assert {name: getattr(wrapper, name).data_ptr() for name in addresses} == addresses
    assert cache.stats()["hits"] == 1


def test_all_base_plan_preserves_native_no_lora_semantics():
    wrapper = FakePunicaWrapper()
    cache = UnoLoRAPlanCache()
    slots = (1_000_003, None)
    mapping = _mapping(base_lora_mapping(6), base_prompt_mapping(2))
    shape = (6, 2)

    assert _native_install(cache, wrapper, "base", shape, slots, mapping) is False
    expected = _state(wrapper)
    assert bool(wrapper.token_mapping_meta.no_lora_flag_cpu[0])

    wrapper.token_mapping_meta.no_lora_flag_cpu.fill_(False)
    wrapper.prompt_mapping_meta.no_lora_flag_cpu.fill_(False)
    assert _native_install(cache, wrapper, "base", shape, slots, mapping) is True
    _assert_state_equal(_state(wrapper), expected)
    assert bool(wrapper.token_mapping_meta.no_lora_flag_cpu[0])
    assert bool(wrapper.prompt_mapping_meta.no_lora_flag_cpu[0])


@pytest.mark.parametrize(
    ("max_tokens", "expected_buckets"),
    [
        (16, (8, 8, 8, 16, 16)),
        (10, (8, 8, 8, 10, 10)),
    ],
)
def test_base_plan_reuses_physical_bucket_across_logits_counts(
    max_tokens, expected_buckets
):
    """Base plans round up safely while actual logits vary."""
    wrapper = FakePunicaWrapper(max_tokens=max_tokens)
    adapter_id = 1_000_003

    class Manager:
        lora_index_to_id = [adapter_id, None]
        lora_slots = 2
        vocab_size = 16

        def _get_punica_wrapper(self, _):
            return wrapper

    class Worker:
        def __init__(self, manager):
            self._adapter_manager = manager

    state = UnoLoRAState(
        Worker(Manager()),
        SimpleNamespace(lora_int_id=1_000_003),
        mapping_cls=lambda token, prompt, is_prefill: SimpleNamespace(
            index_mapping=token,
            prompt_mapping=prompt,
            is_prefill=is_prefill,
        ),
    )

    physical_counts = (5, 7, 8, 9, 10)
    actual_logits_counts = (2, 5, 8, 9, 10)
    assert state.install_base(physical_counts[0], actual_logits_counts[0]) is False
    assert wrapper.native_calls == 1
    assert wrapper.indices_len == [expected_buckets[0]] * 4
    assert torch.equal(
        wrapper._token_lora_indices[: expected_buckets[0]],
        torch.full((expected_buckets[0],), -1, dtype=torch.long),
    )
    assert torch.equal(
        wrapper._embeddings_indices[: expected_buckets[0]],
        torch.zeros(expected_buckets[0], dtype=torch.long),
    )

    # Change the persistent buffers to a real draft mapping before the first
    # cache hit.  A restore must remove the adapter rows from every actual
    # token/embedding row rather than relying on the prior all-base contents.
    wrapper.update_metadata(
        _mapping(
            draft_lora_mapping(expected_buckets[0], 1, 8, adapter_id),
            base_prompt_mapping(expected_buckets[0]),
        ),
        [adapter_id, None],
        3,
        16,
    )
    assert wrapper.native_calls == 2
    assert not bool(wrapper.token_mapping_meta.no_lora_flag_cpu[0])
    assert torch.any(wrapper._token_lora_indices[: expected_buckets[0]] == 0)

    expected_hits = (True, True, False, True)
    for physical_count, actual_logits_count, expected_bucket, expected_hit in zip(
        physical_counts[1:],
        actual_logits_counts[1:],
        expected_buckets[1:],
        expected_hits,
        strict=True,
    ):
        restored = state.install_base(physical_count, actual_logits_count)
        assert restored is expected_hit
        assert wrapper.indices_len == [expected_bucket] * 4
        assert torch.equal(
            wrapper._token_lora_indices[:expected_bucket],
            torch.full((expected_bucket,), -1, dtype=torch.long),
        )
        assert torch.equal(
            wrapper._embeddings_indices[:expected_bucket],
            torch.zeros(expected_bucket, dtype=torch.long),
        )
        assert bool(wrapper.token_mapping_meta.no_lora_flag_cpu[0])
        assert bool(wrapper.prompt_mapping_meta.no_lora_flag_cpu[0])

    assert wrapper.native_calls == 3
    assert state.plan_cache.stats() == {
        "hits": 3,
        "misses": 2,
        "bypasses": 0,
        "invalidations": 0,
        "entries": 2,
    }
    with pytest.raises(ValueError, match="physical model rows"):
        state.install_base(5, 6)
    with pytest.raises(ValueError, match="native LoRA metadata capacity"):
        state.install_base(max_tokens + 1, 1)


def test_non_vector_native_buffer_bypasses_plan_cache():
    """Unknown native buffer rank must fail closed instead of slicing it."""
    wrapper = FakePunicaWrapper()
    wrapper._token_lora_indices = torch.empty((32, 1), dtype=torch.long)
    assert wrapper_fingerprint(wrapper) is None

    cache = UnoLoRAPlanCache()
    builds = 0

    def build():
        nonlocal builds
        builds += 1

    assert cache.install(wrapper, "base", (8,), (None, None), build) is False
    assert builds == 1
    assert cache.stats()["entries"] == 0
    assert cache.stats()["bypasses"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_gpu_state_reuses_rounded_base_plan():
    """The real Punica GPU wrapper accepts an upward rounded base bucket."""
    pytest.importorskip("triton")

    from vllm.config.lora import LoRAConfig
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

    device = torch.device("cuda")
    adapter_id = 1_000_003
    wrapper = PunicaWrapperGPU(
        max_num_batched_tokens=16,
        max_batches=4,
        device=device,
        lora_config=LoRAConfig(
            max_loras=2,
            max_lora_rank=8,
            lora_dtype=torch.bfloat16,
        ),
    )

    class Manager:
        lora_index_to_id = [adapter_id, None]
        lora_slots = 2
        vocab_size = 16

        def _get_punica_wrapper(self, _):
            return wrapper

    class Worker:
        def __init__(self, manager):
            self._adapter_manager = manager

    state = UnoLoRAState(
        Worker(Manager()),
        SimpleNamespace(lora_int_id=adapter_id),
        mapping_cls=LoRAMapping,
    )
    assert state.install_base(5, 2) is False
    assert state.install_base(7, 3) is True
    torch.accelerator.synchronize()

    assert wrapper.indices_len == [8, 8, 8, 8]
    assert bool(wrapper.token_mapping_meta.no_lora_flag_cpu[0])
    assert bool(wrapper.prompt_mapping_meta.no_lora_flag_cpu[0])
    assert state.plan_cache.stats() == {
        "hits": 1,
        "misses": 1,
        "bypasses": 0,
        "invalidations": 0,
        "entries": 1,
    }


def test_slot_reassignment_does_not_reuse_a_plan():
    wrapper = FakePunicaWrapper()
    cache = UnoLoRAPlanCache()
    mapping = _mapping(draft_lora_mapping(8, 2, 4, 1_000_003), (0, 0))
    shape = (8, 2, 4, 1_000_003)
    assert not _native_install(
        cache, wrapper, "draft", shape, (1_000_003, None), mapping
    )
    assert not _native_install(
        cache, wrapper, "draft", shape, (None, 1_000_003), mapping
    )
    assert wrapper.native_calls == 2
    assert cache.stats()["entries"] == 2


def test_state_uses_manager_for_one_time_adapter_lifecycle():
    wrapper = FakePunicaWrapper()

    class Manager:
        lora_index_to_id = [None, None]
        lora_slots = 2
        vocab_size = 16
        _last_mapping = object()
        _last_slot_layout = object()

        def __init__(self):
            self.adapters = set()
            self.activations = 0

        def _get_punica_wrapper(self, _):
            return wrapper

        def list_adapters(self):
            return self.adapters

        def add_adapter(self, request):
            self.adapters.add(request.lora_int_id)

        def activate_adapter(self, adapter_id):
            self.activations += 1
            self.lora_index_to_id[0] = adapter_id

    class Worker:
        def __init__(self, manager):
            self._adapter_manager = manager

        def list_adapters(self):
            return self._adapter_manager.list_adapters()

        def add_adapter(self, request):
            return self._adapter_manager.add_adapter(request)

    manager = Manager()
    request = SimpleNamespace(lora_int_id=1_000_003)
    mapping_cls = lambda token, prompt, is_prefill: SimpleNamespace(
        index_mapping=token,
        prompt_mapping=prompt,
        is_prefill=is_prefill,
    )
    state = UnoLoRAState(Worker(manager), request, mapping_cls=mapping_cls)
    # Full draft replay computes logits for padded requests too; their
    # sample indices are -1 but the base head metadata keeps the full shape.
    assert state.install_draft(16, 2, 4) is False
    assert manager.adapters == {1_000_003}
    assert manager.activations == 1
    assert wrapper.indices_len[0] == 16
    assert wrapper.indices_len[1] == 16
    assert state.install_draft(16, 2, 4) is True
    assert manager.activations == 2
    assert wrapper.native_calls == 1
    assert manager._last_mapping is None
    assert manager._last_slot_layout is None


def _poison_gpu_wrapper(wrapper):
    """Destroy prepared values while keeping every native allocation intact."""
    for name in (
        "_token_lora_indices",
        "_sampler_indices",
        "_sampler_indices_padded",
        "_embeddings_indices",
    ):
        getattr(wrapper, name).fill_(-99)
    for owner in ("token_mapping_meta", "prompt_mapping_meta"):
        meta = getattr(wrapper, owner)
        for name in (
            "token_lora_mapping",
            "token_indices_sorted_by_lora_ids",
            "active_lora_ids",
            "num_tokens_per_lora",
            "lora_token_start_loc",
            "num_active_loras_cpu",
        ):
            getattr(meta, name).fill_(-99)
        meta.no_lora_flag_cpu.fill_(True)
    wrapper.indices_len[:] = [0, 0, 0, 0]
    wrapper.is_prefill = False


@pytest.mark.parametrize("specialize_active_lora", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_gpu_cached_plans_match_native_punica_across_shapes_and_slots(
    specialize_active_lora,
):
    """Compare in-place cached restores with fresh native GPU metadata builds."""
    pytest.importorskip("triton")

    from vllm.config.lora import LoRAConfig
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

    device = torch.device("cuda")
    adapter_id = 1_000_003
    config = LoRAConfig(
        max_loras=2,
        max_lora_rank=8,
        lora_dtype=torch.bfloat16,
        specialize_active_lora=specialize_active_lora,
    )

    def make_wrapper():
        return PunicaWrapperGPU(
            max_num_batched_tokens=32,
            max_batches=4,
            device=device,
            lora_config=config,
        )

    cached = make_wrapper()
    native = make_wrapper()
    cache = UnoLoRAPlanCache()
    slots_for_adapter = (
        (adapter_id, None),
        (None, adapter_id),
    )
    addresses = {
        name: getattr(cached, name).data_ptr()
        for name in (
            "_token_lora_indices",
            "_sampler_indices",
            "_sampler_indices_padded",
            "_embeddings_indices",
        )
    }
    meta_addresses = {
        (owner, name): getattr(getattr(cached, owner), name).data_ptr()
        for owner in ("token_mapping_meta", "prompt_mapping_meta")
        for name in (
            "token_lora_mapping",
            "token_indices_sorted_by_lora_ids",
            "active_lora_ids",
            "num_tokens_per_lora",
            "lora_token_start_loc",
            "no_lora_flag_cpu",
            "num_active_loras_cpu",
        )
    }

    def install_inputs(kind, shape):
        if kind == "draft":
            num_tokens, batch_size, k, _ = shape
            mapping = LoRAMapping(
                draft_lora_mapping(num_tokens, batch_size, k, adapter_id),
                base_prompt_mapping(num_tokens),
                is_prefill=True,
            )
        else:
            (num_tokens,) = shape
            mapping = LoRAMapping(
                base_lora_mapping(num_tokens),
                base_prompt_mapping(num_tokens),
                is_prefill=True,
            )
        return mapping

    cases = [
        ("draft", (8, 2, 4, adapter_id), slots_for_adapter[0], False),
        ("draft", (8, 2, 4, adapter_id), slots_for_adapter[0], True),
        ("draft", (12, 3, 4, adapter_id), slots_for_adapter[0], False),
        ("base", (12,), slots_for_adapter[0], False),
        ("base", (12,), slots_for_adapter[0], True),
        ("base", (8,), slots_for_adapter[1], False),
        ("draft", (8, 2, 4, adapter_id), slots_for_adapter[1], False),
        ("draft", (8, 2, 4, adapter_id), slots_for_adapter[1], True),
    ]
    for kind, shape, slots, expected_hit in cases:
        mapping = install_inputs(kind, shape)
        _poison_gpu_wrapper(cached)
        restored = cache.install(
            cached,
            kind,
            shape,
            slots,
            lambda mapping=mapping, slots=slots: cached.update_metadata(
                mapping, list(slots), 3, 16
            ),
        )
        native.update_metadata(mapping, list(slots), 3, 16)
        torch.accelerator.synchronize()
        assert restored is expected_hit
        _assert_state_equal(_state(cached), _state(native))

    assert {name: getattr(cached, name).data_ptr() for name in addresses} == addresses
    assert {
        (owner, name): getattr(getattr(cached, owner), name).data_ptr()
        for owner, name in meta_addresses
    } == meta_addresses
    assert cache.stats()["hits"] == 3
    assert cache.stats()["misses"] == 5
