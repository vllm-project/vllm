# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``LegacyToUnifiedBackendAdapter`` (RFC #42449, PR 1/N).

The adapter re-hosts the runner's build decision tree
(``cudagraph-capture`` / cross-group ``update_block_table`` cache-hit /
``build``) behind the unified ``prep_forward()`` and stores the built metadata
on the instance (``self.attn_metadata``). PR 1's invariant is *zero behavior
change*, so these tests pin the routing decisions and the cross-group cache
(the highest-risk piece) exactly.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from tests.v1.attention.utils import (
    BatchSpec,
    MockMambaBuilder,
    create_common_attn_metadata,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import override_forward_context
from vllm.v1.attention.backend import (
    AttentionBackend,
    LayerConfig,
    LegacyToUnifiedBackendAdapter,
    UnifiedToLegacyBackendAdapter,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.utils import AttentionGroup

# --------------------------------------------------------------------------- #
# Fake backend/builder: lets us assert the adapter's decision tree on CPU with
# no GPU/kernel dependency. The builder records which path was taken.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _FakeSpec:
    """Hashable stand-in for a resolved KVCacheSpec (used as the cache key)."""

    tag: str = "spec"

    def copy_with_new_block_size(self, block_size: int) -> "_FakeSpec":
        return self


class _FakeBuilder:
    def __init__(self, spec, layer_names, vllm_config, device, *, supports=True):
        self.spec = spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.supports_update_block_table = supports
        self.build_calls = 0
        self.update_calls = 0
        self.capture_calls = 0

    def build(self, common_prefix_len, common_attn_metadata, **extra):
        self.build_calls += 1
        return {
            "kind": "build",
            "prefix": common_prefix_len,
            "block": common_attn_metadata.block_table_tensor,
            "extra": extra,
        }

    def update_block_table(self, metadata, blk_table, slot_mapping):
        self.update_calls += 1
        return {**metadata, "kind": "update", "block": blk_table, "slot": slot_mapping}

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self.capture_calls += 1
        return {"kind": "capture", "block": common_attn_metadata.block_table_tensor}


class _FakeImpl:
    """Stand-in AttentionImpl recording its construction args and forward calls."""

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.forward_calls: list = []

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        *,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        self.forward_calls.append((layer, attn_metadata))

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        self.kv_calls = getattr(self, "kv_calls", [])
        self.kv_calls.append((layer, slot_mapping))


def _make_fake_backend(supports_update_block_table=True):
    # A not-yet-ported (legacy) backend: get_builder_cls returns a separate
    # builder class, so create_metadata_builders wraps it in the
    # LegacyToUnifiedBackendAdapter.
    _subt = supports_update_block_table

    class _FakeBackend:
        @staticmethod
        def get_builder_cls():
            def factory(spec, layer_names, vllm_config, device):
                return _FakeBuilder(
                    spec, layer_names, vllm_config, device, supports=_subt
                )

            return factory

        @staticmethod
        def get_impl_cls():
            return _FakeImpl

    return _FakeBackend


def _make_adapter(supports_update_block_table=True, num_ubatches=1, spec=None):
    spec = spec if spec is not None else _FakeSpec()
    return LegacyToUnifiedBackendAdapter(
        _make_fake_backend(supports_update_block_table),
        spec,
        ["layer0"],
        vllm_config=None,
        device=torch.device("cpu"),
        num_ubatches=num_ubatches,
    )


def _common_metadata(block_offset: int = 0):
    cm = create_common_attn_metadata(
        BatchSpec(seq_lens=[8, 9], query_lens=[1, 1]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    if block_offset:
        cm = cm.replace(block_table_tensor=cm.block_table_tensor + block_offset)
    return cm


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_prep_forward_build_stores_metadata_on_instance():
    adapter = _make_adapter()
    cm = _common_metadata()

    adapter.prep_forward(cm, common_prefix_len=7, extra_metadata_args={"foo": 1})

    md = adapter.attn_metadata
    assert md["kind"] == "build"
    assert md["prefix"] == 7
    assert md["extra"] == {"foo": 1}
    assert adapter.get_builder(0).build_calls == 1
    assert adapter.get_builder(0).capture_calls == 0


def test_prep_forward_cudagraph_capture_is_first_branch():
    """for_cudagraph_capture must route to build_for_cudagraph_capture and
    never consult the cache (some backends' capture build differs)."""
    adapter = _make_adapter()
    cache: dict = {}

    adapter.prep_forward(
        _common_metadata(),
        for_cudagraph_capture=True,
        extra_metadata_args={"foo": 1},
        metadata_cache=cache,
    )

    assert adapter.attn_metadata["kind"] == "capture"
    assert adapter.get_builder(0).capture_calls == 1
    assert adapter.get_builder(0).build_calls == 0
    assert cache == {}


def test_prep_forward_selects_ubatch_builder():
    adapter = _make_adapter(num_ubatches=2)

    adapter.prep_forward(_common_metadata(), ubatch_id=1)

    assert adapter.get_builder(1).build_calls == 1
    assert adapter.get_builder(0).build_calls == 0


# --------------------------------------------------------------------------- #
# Cross-group update_block_table cache (highest-risk behavior)
# --------------------------------------------------------------------------- #


def test_cross_group_cache_reuses_via_update_block_table():
    """Two sibling groups with equal spec + same builder type share a
    runner-owned cache. The first builds and caches; the second must reuse via
    update_block_table (not rebuild) and carry the second group's block table.
    """
    spec = _FakeSpec()
    cache: dict = {}
    adapter_a = _make_adapter(spec=spec)
    adapter_b = _make_adapter(spec=spec)

    cm_a = _common_metadata()
    cm_b = _common_metadata(block_offset=100)

    adapter_a.prep_forward(cm_a, metadata_cache=cache)
    adapter_b.prep_forward(cm_b, metadata_cache=cache)

    assert adapter_a.attn_metadata["kind"] == "build"
    # Second sibling reused the cached metadata via update_block_table.
    assert adapter_b.attn_metadata["kind"] == "update"
    assert adapter_b.get_builder(0).build_calls == 0
    assert adapter_b.get_builder(0).update_calls == 1
    # ...and carries its own (second group's) block table.
    torch.testing.assert_close(
        adapter_b.attn_metadata["block"], cm_b.block_table_tensor
    )


def test_no_cache_when_metadata_cache_is_none():
    """V2 passes metadata_cache=None: every call rebuilds, nothing is cached."""
    adapter = _make_adapter()

    adapter.prep_forward(_common_metadata(), metadata_cache=None)
    adapter.prep_forward(_common_metadata(block_offset=100), metadata_cache=None)

    assert adapter.get_builder(0).build_calls == 2
    assert adapter.get_builder(0).update_calls == 0


def test_unsupported_builder_never_caches_or_updates():
    """When the builder does not support update_block_table, the adapter must
    rebuild every time and never populate the cache."""
    spec = _FakeSpec()
    cache: dict = {}
    adapter_a = _make_adapter(supports_update_block_table=False, spec=spec)
    adapter_b = _make_adapter(supports_update_block_table=False, spec=spec)

    adapter_a.prep_forward(_common_metadata(), metadata_cache=cache)
    adapter_b.prep_forward(_common_metadata(block_offset=100), metadata_cache=cache)

    assert cache == {}
    assert adapter_a.attn_metadata["kind"] == "build"
    assert adapter_b.attn_metadata["kind"] == "build"
    assert adapter_b.get_builder(0).build_calls == 1
    assert adapter_b.get_builder(0).update_calls == 0


# --------------------------------------------------------------------------- #
# Authoritative block_tables / slot_mappings dicts (PR 2)
# --------------------------------------------------------------------------- #


def test_block_tables_dict_is_authoritative():
    """When block_tables/slot_mappings dicts are passed, the adapter sources
    the per-group table from them (keyed by kv_cache_group_id), overriding
    whatever common_attn_metadata carried."""
    spec = _FakeSpec()
    adapter = LegacyToUnifiedBackendAdapter(
        _make_fake_backend(),
        spec,
        ["layer0"],
        vllm_config=None,
        device=torch.device("cpu"),
        kv_cache_group_ids=[2],
    )
    cm = _common_metadata()  # carries some unrelated block table
    authoritative = _common_metadata(block_offset=500).block_table_tensor
    authoritative_slot = torch.arange(7, dtype=torch.int64)

    adapter.prep_forward(
        cm,
        block_tables={2: authoritative},
        slot_mappings={2: authoritative_slot},
    )

    md = adapter.attn_metadata
    torch.testing.assert_close(md["block"], authoritative)


def test_block_tables_none_uses_common_metadata():
    """With no dicts (e.g. the ubatch path), the builder uses the block table
    already on common_attn_metadata."""
    adapter = _make_adapter()
    cm = _common_metadata(block_offset=321)

    adapter.prep_forward(cm, block_tables=None)

    torch.testing.assert_close(adapter.attn_metadata["block"], cm.block_table_tensor)


# --------------------------------------------------------------------------- #
# Instance forward() routing (PR 3 / RFC #42449)
# --------------------------------------------------------------------------- #


def _layer_config(layer_name="layer0", **kw):
    base = dict(layer_name=layer_name, num_heads=8, head_size=64, scale=0.125)
    base.update(kw)
    return LayerConfig(**base)


class _UnifiedBackend(AttentionBackend):
    """Fake unified backend: build records args, forward records the call."""

    @staticmethod
    def get_name():
        return "FAKE_UNIFIED"

    @classmethod
    def get_builder_cls(cls):
        return cls

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    def _build_step(
        self, ubatch_id, common_attn_metadata, common_prefix_len=0, fast_build=False
    ):
        self.built = (
            ubatch_id,
            common_attn_metadata.block_table_tensor,
            common_prefix_len,
        )

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        self.ran = (layer.layer_config.layer_name, self.built[1])
        return output


def test_unified_backend_build_and_forward():
    """A unified backend is the per-group instance. create_metadata_builders
    exposes its builder through a UnifiedToLegacyBackendAdapter so the legacy V1
    runner can build via builder.build(); the op calls the backend's own forward
    directly (no impl shim — the backend reads its per-step state)."""
    group = AttentionGroup(
        _UnifiedBackend, ["layer0"], _FakeSpec(), kv_cache_group_ids=[0]
    )
    group.create_metadata_builders(vllm_config=None, device=torch.device("cpu"))

    assert isinstance(group.backend_instance, _UnifiedBackend)
    assert isinstance(group.legacy_adapter, UnifiedToLegacyBackendAdapter)
    builder = group.get_metadata_builder(0)
    assert builder is group.legacy_adapter.get_builder()

    # Build: builder.build drives _build_step (lane 0) and returns the backend.
    cm = _common_metadata()
    md = builder.build(common_prefix_len=3, common_attn_metadata=cm)
    assert md is group.backend_instance
    assert group.backend_instance.built[0] == 0
    assert group.backend_instance.built[2] == 3

    # Forward: the backend runs its own forward, reading per-step state.
    layer = SimpleNamespace(layer_config=_layer_config())
    group.backend_instance.forward(
        layer,
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(2),
        torch.zeros(1),
        output=torch.zeros(2),
    )
    assert group.backend_instance.ran[0] == "layer0"


def test_legacy_adapter_forward_and_kv_update_delegate_to_impl():
    """A legacy backend runs through the unified interface via
    LegacyToUnifiedBackendAdapter: forward() re-resolves the layer's metadata
    from the forward context and calls the module's legacy impl; the cache write
    delegates likewise."""
    adapter = _make_adapter()
    layer = SimpleNamespace(layer_name="layer0", impl=_FakeImpl())
    ctx = SimpleNamespace(attn_metadata={"layer0": "MD"})

    with override_forward_context(ctx):
        adapter.forward(
            layer,
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(2),
            torch.zeros(1),
            output=torch.zeros(2),
        )
    assert len(layer.impl.forward_calls) == 1
    # The adapter resolved this layer's built metadata and passed it through.
    assert layer.impl.forward_calls[0][1] == "MD"

    adapter.do_kv_cache_update(
        layer, torch.zeros(2), torch.zeros(2), torch.zeros(1), torch.zeros(2)
    )
    assert len(layer.impl.kv_calls) == 1


# --------------------------------------------------------------------------- #
# Builder aliasing via AttentionGroup
# --------------------------------------------------------------------------- #


def test_attention_group_aliases_builders_to_adapter():
    group = AttentionGroup(
        _make_fake_backend(),
        ["layer0"],
        _FakeSpec(),
        kv_cache_group_ids=[0],
    )

    group.create_metadata_builders(
        vllm_config=None,
        device=torch.device("cpu"),
        num_metadata_builders=2,
    )

    assert group.backend_instance is not None
    # The builder list every existing consumer reads must BE the adapter's.
    assert group.metadata_builders is group.backend_instance._builders
    assert len(group.metadata_builders) == 2
    assert group.get_metadata_builder(1) is group.backend_instance.get_builder(1)


# --------------------------------------------------------------------------- #
# Real builder: prep_forward(build) == builder.build with identical args
# --------------------------------------------------------------------------- #


def _make_mamba_vllm_config(block_size: int = 16, max_model_len: int = 256):
    """Minimal config exposing only the fields a mamba builder reads (mirrors
    tests/v1/attention/test_mamba_update_block_table.py)."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size, mamba_cache_mode="all"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=None,
        num_speculative_tokens=0,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )


def test_real_builder_prep_forward_matches_direct_build():
    block_size = 16
    device = torch.device("cpu")
    vllm_config = _make_mamba_vllm_config(block_size=block_size)
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="all",
    )

    class _MambaBackend:
        @staticmethod
        def get_builder_cls():
            return MockMambaBuilder

    cm = create_common_attn_metadata(
        BatchSpec(seq_lens=[8, 9, 1], query_lens=[1, 1, 1]),
        block_size=block_size,
        device=device,
        arange_block_indices=True,
    ).replace(is_prefilling=torch.tensor([False, False, True], dtype=torch.bool))

    # Direct build from a bare builder.
    direct_builder = MockMambaBuilder(spec, ["layer0"], vllm_config, device)
    expected = direct_builder.build(common_prefix_len=0, common_attn_metadata=cm)

    # Build through the adapter.
    adapter = LegacyToUnifiedBackendAdapter(
        _MambaBackend, spec, ["layer0"], vllm_config, device
    )
    adapter.prep_forward(cm)
    got = adapter.attn_metadata

    assert got.num_prefills == expected.num_prefills
    assert got.num_decodes == expected.num_decodes
    assert got.num_decode_tokens == expected.num_decode_tokens
    torch.testing.assert_close(got.seq_lens, expected.seq_lens)
    torch.testing.assert_close(got.has_initial_states_p, expected.has_initial_states_p)
