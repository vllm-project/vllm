# SPDX-License-Identifier: Apache-2.0
"""Unit tests for V3 load/store optimizations: L1 (batched rope), L2
(obj_keys cache), S1 (async fingerprint).

These tests exercise the wiring/state changes without touching CUDA or
the storage controller. The CUDA kernel inside ``_apply_cb_rope_batched``
is mocked; the matcher inside the async fingerprint worker is mocked.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import threading
import time

# Third Party
import pytest
import torch

# ---------------------------------------------------------------------------
# S1: async fingerprint registration
# ---------------------------------------------------------------------------


def _make_engine_with_mocked_matcher():
    """Construct a real BlendV3Module with the matcher mocked so we can
    observe `on_new_token_hashes` calls without setting up storage."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    eng_mock = MagicMock(spec=v3_mod.BlendV3Module)
    eng_mock._fingerprint_stop = threading.Event()
    eng_mock._token_range_matcher = MagicMock()
    eng_mock._pending_fp_lock = threading.Lock()
    eng_mock._pending_fp_hashes = set()
    # Bind the real drainer method to our mock.
    eng_mock._drain_fingerprint_queue = (
        v3_mod.BlendV3Module._drain_fingerprint_queue.__get__(eng_mock)
    )
    return eng_mock


def test_fingerprint_queue_drains_in_order():
    """Jobs enqueued by store() flow through the worker in submission order."""
    # Standard
    from queue import Queue

    eng = _make_engine_with_mocked_matcher()
    eng._fingerprint_queue = Queue()

    worker = threading.Thread(target=eng._drain_fingerprint_queue, daemon=True)
    worker.start()
    try:
        jobs = [
            ([1, 2, 3], [b"h1"], 0, 0),
            ([4, 5, 6], [b"h2"], 1, 3),
            ([7, 8, 9], [b"h3"], 0, 6),
        ]
        for j in jobs:
            eng._fingerprint_queue.put(j)
        # Wait for the queue to drain (worker calls task_done implicitly
        # only via get(); we just poll until matcher has all calls).
        deadline = time.monotonic() + 2.0
        while (
            eng._token_range_matcher.on_new_token_hashes.call_count < len(jobs)
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
    finally:
        eng._fingerprint_stop.set()
        worker.join(timeout=1.0)

    # All three were registered.
    assert eng._token_range_matcher.on_new_token_hashes.call_count == 3
    # In submission order.
    calls = eng._token_range_matcher.on_new_token_hashes.call_args_list
    assert calls[0].args[0] == [1, 2, 3]
    assert calls[1].args[0] == [4, 5, 6]
    assert calls[2].args[0] == [7, 8, 9]
    # kwargs are preserved (start_chunk_idx, position_offset).
    assert calls[1].kwargs == {"start_chunk_idx": 1, "position_offset": 3}


def test_fingerprint_worker_survives_kernel_exception():
    """A failing matcher call doesn't kill the worker."""
    # Standard
    from queue import Queue

    eng = _make_engine_with_mocked_matcher()
    eng._fingerprint_queue = Queue()
    # First call raises, subsequent succeed.
    eng._token_range_matcher.on_new_token_hashes.side_effect = [
        RuntimeError("boom"),
        None,
    ]

    worker = threading.Thread(target=eng._drain_fingerprint_queue, daemon=True)
    worker.start()
    try:
        eng._fingerprint_queue.put(([1], [b"h1"], 0, 0))
        eng._fingerprint_queue.put(([2], [b"h2"], 0, 1))
        deadline = time.monotonic() + 2.0
        while (
            eng._token_range_matcher.on_new_token_hashes.call_count < 2
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
    finally:
        eng._fingerprint_stop.set()
        worker.join(timeout=1.0)

    assert eng._token_range_matcher.on_new_token_hashes.call_count == 2
    assert not worker.is_alive()


def test_fingerprint_worker_stops_on_signal():
    """``_fingerprint_stop`` event halts the drainer cleanly."""
    # Standard
    from queue import Queue

    eng = _make_engine_with_mocked_matcher()
    eng._fingerprint_queue = Queue()
    worker = threading.Thread(target=eng._drain_fingerprint_queue, daemon=True)
    worker.start()
    eng._fingerprint_stop.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()


# ---------------------------------------------------------------------------
# L2: obj_keys cache lifecycle
# ---------------------------------------------------------------------------


def _fake_obj_key(chunk_hash: bytes, worker_id: int) -> SimpleNamespace:
    return SimpleNamespace(chunk_hash=chunk_hash, worker_id=worker_id)


def test_obj_keys_cache_round_trip_tp1():
    """At world_size=1, retrieve can rebuild from the cache exactly."""
    eng = MagicMock()
    eng._lookup_obj_keys_cache = {}
    eng._lookup_obj_keys_lock = threading.Lock()

    # Simulate what cb_lookup_subsequences stores.
    chunk_hashes = [b"h1", b"h2", b"h3"]
    obj_keys_per_chunk = {h: [_fake_obj_key(h, 0)] for h in chunk_hashes}
    with eng._lookup_obj_keys_lock:
        eng._lookup_obj_keys_cache["req-1"] = obj_keys_per_chunk

    # Simulate retrieve consuming the cache.
    matches_sorted = [
        SimpleNamespace(hash=h, cur_st=i) for i, h in enumerate(chunk_hashes)
    ]
    with eng._lookup_obj_keys_lock:
        cached = eng._lookup_obj_keys_cache.pop("req-1", None)

    assert cached is not None
    assert all(r.hash in cached for r in matches_sorted)
    rebuilt = [k for r in matches_sorted for k in cached[r.hash]]
    assert len(rebuilt) == 3
    assert [k.chunk_hash for k in rebuilt] == chunk_hashes
    # Cache is now empty for this request.
    with eng._lookup_obj_keys_lock:
        assert "req-1" not in eng._lookup_obj_keys_cache


def test_obj_keys_cache_round_trip_tp_expanded():
    """world_size>1: cached entry per hash is a list of length world_size,
    rebuilt list is flat chunk-major."""
    eng = MagicMock()
    eng._lookup_obj_keys_cache = {}
    eng._lookup_obj_keys_lock = threading.Lock()

    ws = 4
    chunk_hashes = [b"h1", b"h2"]
    per_hash = {h: [_fake_obj_key(h, w) for w in range(ws)] for h in chunk_hashes}
    with eng._lookup_obj_keys_lock:
        eng._lookup_obj_keys_cache["req-tp"] = per_hash

    matches_sorted = [
        SimpleNamespace(hash=h, cur_st=i) for i, h in enumerate(chunk_hashes)
    ]
    with eng._lookup_obj_keys_lock:
        cached = eng._lookup_obj_keys_cache.pop("req-tp", None)
    rebuilt = [k for r in matches_sorted for k in cached[r.hash]]
    # Length = 2 chunks × 4 workers.
    assert len(rebuilt) == 8
    # Chunk-major: first 4 entries are h1's workers 0..3, then h2's.
    assert [k.chunk_hash for k in rebuilt[:4]] == [b"h1"] * 4
    assert [k.worker_id for k in rebuilt[:4]] == [0, 1, 2, 3]
    assert [k.chunk_hash for k in rebuilt[4:]] == [b"h2"] * 4


def test_obj_keys_cache_miss_falls_back():
    """If the cache doesn't contain every match's hash, retrieve must
    fall back to recompute (handled in the engine; this test just pins
    the detection logic)."""
    cached = {b"h1": ["k1"]}
    matches = [SimpleNamespace(hash=b"h1"), SimpleNamespace(hash=b"h_missing")]
    all_present = all(r.hash in cached for r in matches)
    assert all_present is False


# ---------------------------------------------------------------------------
# L1: batched rope structure
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal stand-in for the torch tensors used inside _apply_cb_rope_batched.
    Tracks shape so the kernel mock can assert on it.
    """

    def __init__(self, shape):
        self.shape = shape
        self.device = "cpu"
        # rot_for_group takes the buffer dtype (declared-map quant skip).
        self.dtype = torch.bfloat16

    def __getitem__(self, idx):
        # tmp[0] selects K from the (2, num_layers, slots, hidden_dim) tensor.
        return _FakeTensor(self.shape[1:] if isinstance(idx, int) else self.shape)

    def reshape(self, *new_shape):
        return _FakeTensor(tuple(new_shape))

    def view(self, *new_shape):
        return _FakeTensor(tuple(new_shape))


def _build_fake_gpu_context(batch_size: int, num_groups: int):
    """Returns a MagicMock matching the minimal GPUCacheContext surface
    used by _apply_cb_rope_batched."""
    gpu_context = MagicMock()
    gpu_context.kv_layer_groups_manager.num_kernel_groups = num_groups
    # All groups: uncompressed (tokens_per_block == slots_per_block), kv_size=2.
    groups = [
        SimpleNamespace(tokens_per_block=4, slots_per_block=4, engine_group_idx=idx)
        for idx in range(num_groups)
    ]
    gpu_context.kv_layer_groups_manager.kernel_groups = groups

    # Each per-(slot, group) buffer has shape
    # (2 kv, num_layers, slots_per_block, hidden_dim).
    num_layers, slots_per_block, hidden_dim = 2, 4, 64
    head_size = 32

    def _get_temp_kernel_group_buffer(batch_idx, kernel_group_idx):
        return _FakeTensor((2, num_layers, slots_per_block, hidden_dim))

    gpu_context.get_temp_kernel_group_buffer.side_effect = _get_temp_kernel_group_buffer
    return gpu_context, head_size


def test_batched_rope_calls_kernel_per_group_per_slot():
    """For N non-prefix slots and G groups, kernel is called N*G times
    (matching today's CUDA-level work) but the Python ``per-group setup``
    runs only G times (vs N*G under the legacy path)."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    gpu_context, head_size = _build_fake_gpu_context(batch_size=4, num_groups=2)
    rope_state = v3_mod._CBRopeState(
        head_size=head_size,
        is_neox_style=True,
        cos_sin_caches=[MagicMock()],
        group_to_cache=[],
    )

    eng = MagicMock(spec=v3_mod.BlendV3Module)
    eng._apply_cb_rope_batched = v3_mod.BlendV3Module._apply_cb_rope_batched.__get__(
        eng
    )

    slots_to_rope = [(0, 100, 200), (2, 300, 400)]  # 2 non-prefix slots

    with (
        patch.object(v3_mod, "lmc_ops") as ops,
        patch.object(v3_mod, "torch") as torch_mod,
    ):
        torch_mod.long = "long"

        # Build a fake positions tensor that supports + and .repeat()
        class _Pos:
            def __add__(self, other):
                return _Pos()

            def __radd__(self, other):
                return _Pos()

            def repeat(self, n):
                return _Pos()

        torch_mod.arange.return_value = _Pos()

        eng._apply_cb_rope_batched(gpu_context, rope_state, 4, slots_to_rope)

    # all_slots is built once per group (G=2), each fetching the full batch
    # of slot buffers => batch_len(4) × G(2) = 8 buffer fetches, independent
    # of how many slots are actually re-RoPE'd.
    assert gpu_context.get_temp_kernel_group_buffer.call_count == 8
    # Kernel called N=2 slots × G=2 groups = 4 times.
    assert ops.rotary_embedding_k_fused.call_count == 4


def test_batched_rope_noop_on_empty_slots():
    """No non-prefix slots → no setup, no kernel calls."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    gpu_context, head_size = _build_fake_gpu_context(batch_size=2, num_groups=2)
    rope_state = v3_mod._CBRopeState(
        head_size=head_size,
        is_neox_style=False,
        cos_sin_caches=[MagicMock()],
        group_to_cache=[],
    )
    eng = MagicMock(spec=v3_mod.BlendV3Module)
    eng._apply_cb_rope_batched = v3_mod.BlendV3Module._apply_cb_rope_batched.__get__(
        eng
    )

    with patch.object(v3_mod, "lmc_ops") as ops:
        eng._apply_cb_rope_batched(gpu_context, rope_state, 2, [])

    assert gpu_context.get_temp_kernel_group_buffer.call_count == 0
    assert ops.rotary_embedding_k_fused.call_count == 0


def test_batched_rope_raises_on_compressed_layout():
    """A compressed group (tokens_per_block != slots_per_block) → RuntimeError."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    gpu_context = MagicMock()
    gpu_context.kv_layer_groups_manager.num_kernel_groups = 1
    gpu_context.kv_layer_groups_manager.kernel_groups = [
        SimpleNamespace(tokens_per_block=8, slots_per_block=4, engine_group_idx=0)
    ]
    gpu_context.get_temp_kernel_group_buffer.return_value = SimpleNamespace(
        shape=(2, 2, 4, 64), dtype=torch.bfloat16
    )
    # Real rope state: the batched path resolves the group's rot window
    # (rot_for_group) before the geometry check that this test targets.
    rope_state = v3_mod._CBRopeState(
        head_size=32,
        is_neox_style=True,
        cos_sin_caches=[MagicMock()],
        group_to_cache=[],
    )

    eng = MagicMock(spec=v3_mod.BlendV3Module)
    eng._apply_cb_rope_batched = v3_mod.BlendV3Module._apply_cb_rope_batched.__get__(
        eng
    )

    with pytest.raises(RuntimeError, match="is compressed"):
        eng._apply_cb_rope_batched(gpu_context, rope_state, 2, [(0, 1, 2)])


# ---------------------------------------------------------------------------
# Coordinator (global) leg: conversion to retrievable CBMatchResult + deadline
# ---------------------------------------------------------------------------


def _coord_engine(chunk_size: int = 4):
    """A BlendV3Module mock with the coordinator-leg methods bound."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    eng = MagicMock(spec=v3_mod.BlendV3Module)
    eng._ctx = SimpleNamespace(chunk_size=chunk_size)
    # _event_bus is an instance attr (set in __init__), so spec= omits it;
    # _poll_coordinator_match publishes CB_COORDINATOR_MATCH_END through it.
    eng._event_bus = MagicMock()
    eng._build_global_segments = v3_mod.BlendV3Module._build_global_segments.__get__(
        eng
    )
    eng._poll_coordinator_match = v3_mod.BlendV3Module._poll_coordinator_match.__get__(
        eng
    )
    return eng


def test_build_global_segments_are_retrievable_cbmatchresults():
    """Coordinator object_key hex round-trips to the hash the retrieve path
    resolves via ipc_key_to_object_keys; positions span one chunk."""
    # First Party
    from lmcache.v1.mp_coordinator.blend_client import RemoteMatch
    from lmcache.v1.multiprocess.custom_types import CBMatchResult

    eng = _coord_engine(chunk_size=4)
    raw = bytes.fromhex("00") * 0 + b"\xab\xcd\xef\x01"
    matches = [RemoteMatch(object_key=raw.hex(), old_st=8, cur_st=20)]

    segs = eng._build_global_segments(matches)

    assert len(segs) == 1
    seg = segs[0]
    assert isinstance(seg, CBMatchResult)
    assert seg.hash == raw  # hex -> exact bytes the retrieve path expands
    assert (seg.old_st, seg.old_ed, seg.cur_st, seg.cur_ed) == (8, 12, 20, 24)


def test_poll_coordinator_match_deferred_then_resolved():
    """PENDING within deadline defers (None); a list resolves to segments."""
    # First Party
    from lmcache.v1.mp_coordinator.blend_client import PENDING, RemoteMatch

    eng = _coord_engine(chunk_size=4)
    coordinator = MagicMock()
    eng._coordinator = coordinator
    job = SimpleNamespace(coord_submitted=True, coord_deadline=time.monotonic() + 60)

    coordinator.poll_match.return_value = PENDING
    assert eng._poll_coordinator_match(job, "rid") is None  # defer
    coordinator.take_match.assert_not_called()

    coordinator.poll_match.return_value = [RemoteMatch("aa", old_st=0, cur_st=4)]
    out = eng._poll_coordinator_match(job, "rid")
    assert [s.cur_st for s in out] == [4]
    coordinator.take_match.assert_called_once_with("rid")


def test_poll_coordinator_match_gives_up_past_deadline():
    """PENDING past the deadline degrades to local-only ([]) and drops state."""
    # First Party
    from lmcache.v1.mp_coordinator.blend_client import PENDING

    eng = _coord_engine(chunk_size=4)
    coordinator = MagicMock()
    eng._coordinator = coordinator
    coordinator.poll_match.return_value = PENDING
    job = SimpleNamespace(coord_submitted=True, coord_deadline=time.monotonic() - 1)

    assert eng._poll_coordinator_match(job, "rid") == []
    coordinator.take_match.assert_called_once_with("rid")


def test_non_overlapping_after_prefix():
    """Prefix filter + leftmost-greedy overlap dedup, filter applied first."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import CBMatchResult
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    f = v3_mod.BlendV3Module._non_overlapping_after_prefix

    def m(cur_st: int, cur_ed: int) -> CBMatchResult:
        return CBMatchResult(
            old_st=0, old_ed=cur_ed - cur_st, cur_st=cur_st, cur_ed=cur_ed, hash=b""
        )

    assert f([], 0) == []

    # Overlap dedup + ascending cur_st: 10-20 overlaps the kept 5-15, dropped.
    out = f([m(10, 20), m(5, 15), m(15, 25)], 0)
    assert [(r.cur_st, r.cur_ed) for r in out] == [(5, 15), (15, 25)]

    # Prefix filter drops matches starting before the coverage.
    out = f([m(0, 10), m(10, 20)], 5)
    assert [r.cur_st for r in out] == [10]

    # Filter precedes dedup: a prefix-covered match (5-13) must NOT suppress the
    # usable 10-18 in the greedy pass (dedup-first would drop both -> []).
    out = f([m(5, 13), m(10, 18)], 8)
    assert [r.cur_st for r in out] == [10]


# ---------------------------------------------------------------------------
# Dual-RoPE: per-group cache selection + registration validation
# ---------------------------------------------------------------------------


def test_cache_for_group_uniform_and_mapped():
    """Empty map -> every group uses cache 0; a map indexes per group;
    a group past the map's end raises instead of guessing."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    local, global_ = MagicMock(), MagicMock()

    uniform = v3_mod._CBRopeState(
        head_size=32, is_neox_style=True, cos_sin_caches=[local], group_to_cache=[]
    )
    assert uniform.cache_for_group(0) is local
    assert uniform.cache_for_group(5) is local

    mapped = v3_mod._CBRopeState(
        head_size=32,
        is_neox_style=True,
        cos_sin_caches=[local, global_],
        group_to_cache=[0, 1],
    )
    assert mapped.cache_for_group(0) is local
    assert mapped.cache_for_group(1) is global_
    with pytest.raises(RuntimeError, match="no rope cache mapping"):
        mapped.cache_for_group(2)


def _rope_registration_engine(engine_group_indices: list[int]):
    """A BlendV3Module mock with ``cb_register_rope`` bound and a registered
    instance whose kernel groups span the given engine group indices."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    eng = MagicMock(spec=v3_mod.BlendV3Module)
    eng._cb_rope_state = {}
    eng._transfer_module = MagicMock()
    entry = SimpleNamespace(
        cache_context=SimpleNamespace(
            kv_layer_groups_manager=SimpleNamespace(
                kernel_groups=[
                    SimpleNamespace(engine_group_idx=idx)
                    for idx in engine_group_indices
                ]
            )
        )
    )
    eng._transfer_module.get_and_touch_context_entry.return_value = entry
    eng.cb_register_rope = v3_mod.BlendV3Module.cb_register_rope.__get__(eng)
    return eng


def _unit_rope_cache_ipc():
    """An IPC-wrapper mock whose tensor has unit magnitude (cos=1, sin=0),
    so registration skips mscale normalization."""
    # Third Party
    import torch

    cache = torch.zeros(4, 8)
    cache[:, :4] = 1.0
    ipc = MagicMock()
    ipc.to_tensor.return_value = cache
    return ipc


def test_register_rope_dual_cache_round_trip():
    """Two caches + a full engine-group map register and land in rope state."""
    eng = _rope_registration_engine(engine_group_indices=[0, 1])

    eng.cb_register_rope(
        instance_id=7,
        cos_sin_caches_ipc=[_unit_rope_cache_ipc(), _unit_rope_cache_ipc()],
        head_size=8,
        is_neox_style=True,
        group_to_cache=[0, 1],
    )

    state = eng._cb_rope_state[7]
    assert len(state.cos_sin_caches) == 2
    assert state.group_to_cache == [0, 1]
    assert state.cache_for_group(1) is state.cos_sin_caches[1]


def test_register_rope_rejects_invalid_group_to_cache():
    """Out-of-range / negative cache indices and a map that does not cover
    every engine group of the registered model are rejected."""
    eng = _rope_registration_engine(engine_group_indices=[0, 1])
    caches = [_unit_rope_cache_ipc(), _unit_rope_cache_ipc()]

    with pytest.raises(ValueError, match="outside"):
        eng.cb_register_rope(1, caches, 8, True, group_to_cache=[0, 2])
    with pytest.raises(ValueError, match="outside"):
        eng.cb_register_rope(1, caches, 8, True, group_to_cache=[-1, 0])
    # Model has engine groups {0, 1} but the map only covers group 0.
    with pytest.raises(ValueError, match="engine groups up to index 1"):
        eng.cb_register_rope(1, caches, 8, True, group_to_cache=[0])
    with pytest.raises(ValueError, match=">=1 cos/sin cache"):
        eng.cb_register_rope(1, [], 8, True, group_to_cache=[])


def test_register_rope_requires_registered_instance():
    """CB_REGISTER_ROPE_V3 before REGISTER_KV_CACHE is rejected."""
    eng = _rope_registration_engine(engine_group_indices=[0])
    eng._transfer_module.get_and_touch_context_entry.return_value = None

    with pytest.raises(ValueError, match="no paged KV cache registered"):
        eng.cb_register_rope(1, [_unit_rope_cache_ipc()], 8, True, group_to_cache=[])


# ---------------------------------------------------------------------------
# Retrieve: per-slot paged scatter (no torch.cat of the batch)
# ---------------------------------------------------------------------------


def _build_scatter_engine_and_context(
    num_groups: int,
    num_slots: int,
    spc: int = 4,
    num_layers: int = 2,
    hidden_dim: int = 8,
):
    """Engine with the real ``_scatter_batch_to_paged`` bound, plus a fake
    GPU context whose tmp slot buffers are real (CPU) tensors — distinct
    objects per (slot, group) so kernel calls can be identity-checked."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    eng = MagicMock(spec=v3_mod.BlendV3Module)
    eng._scatter_batch_to_paged = v3_mod.BlendV3Module._scatter_batch_to_paged.__get__(
        eng
    )

    # Third Party
    import torch

    gpu_context = MagicMock()
    gpu_context.device = torch.device("cpu")
    gpu_context.kv_layer_groups_manager.num_kernel_groups = num_groups
    gpu_context.kv_layer_groups_manager.kernel_groups = [
        SimpleNamespace(shape_desc=SimpleNamespace(nb=100)) for _ in range(num_groups)
    ]

    buffers = {
        (slot, group): torch.zeros(2, num_layers, spc, hidden_dim)
        for slot in range(num_slots)
        for group in range(num_groups)
    }
    gpu_context.get_temp_kernel_group_buffer.side_effect = lambda s, g: buffers[(s, g)]
    return eng, gpu_context, buffers


def _match(cur_st: int, cur_ed: int):
    return SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=cur_st)


def test_scatter_launches_per_slot_without_cat():
    """N slots × G groups → N*G kernel launches, each fed the slot's OWN
    buffer object (no torch.cat copy), with a per-slot slot_mapping slice
    that matches the group's block table numerically."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    eng, gpu_context, buffers = _build_scatter_engine_and_context(
        num_groups=2, num_slots=3, spc=4
    )
    batch = [(_match(0, 4), None), (_match(4, 8), None), (_match(8, 12), None)]
    # Group 0 and group 1 use different block tables (same bs=4).
    resolved_groups = [
        (torch.tensor([10, 11, 12], dtype=torch.long), 4),
        (torch.tensor([20, 21, 22], dtype=torch.long), 4),
    ]

    with patch.object(v3_mod, "lmc_ops") as ops:
        eng._scatter_batch_to_paged(gpu_context, resolved_groups, batch, 32)

    calls = ops.multi_layer_kv_transfer.call_args_list
    assert len(calls) == 3 * 2  # per (group, slot)

    for call_idx, call in enumerate(calls):
        group_idx, slot_idx = divmod(call_idx, 3)
        key_value = call.args[0]
        # Identity: the kernel scatters straight from the slot buffer.
        assert key_value is buffers[(slot_idx, group_idx)]
        # Per-slot slot_mapping slice: block_ids[tok // bs] * bs + tok % bs.
        block_base = resolved_groups[group_idx][0][slot_idx].item() * 4
        assert call.args[2].tolist() == list(range(block_base, block_base + 4))
        # page_buffer_size = nb * group_bs.
        assert call.args[4] == 100 * 4
        assert call.kwargs["block_size"] == 4
        assert call.kwargs["head_size"] == 32


def test_scatter_narrows_partial_chunk_and_keeps_alignment():
    """A slot holding fewer tokens than its buffer capacity is narrowed to
    the real token count (the kernel scatters ``size(2)`` tokens); later
    slots still get correctly aligned slot_mapping slices — the old cat
    path shifted every subsequent slot off its mapping."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    eng, gpu_context, buffers = _build_scatter_engine_and_context(
        num_groups=1, num_slots=3, spc=4
    )
    # Middle chunk is partial: 2 tokens in a 4-slot buffer.
    batch = [(_match(0, 4), None), (_match(4, 6), None), (_match(6, 10), None)]
    resolved_groups = [(torch.tensor([10, 11, 12], dtype=torch.long), 4)]

    with patch.object(v3_mod, "lmc_ops") as ops:
        eng._scatter_batch_to_paged(gpu_context, resolved_groups, batch, 32)

    calls = ops.multi_layer_kv_transfer.call_args_list
    assert len(calls) == 3

    # Full slot 0: identity, tokens 0..3 -> block 10.
    assert calls[0].args[0] is buffers[(0, 0)]
    assert calls[0].args[2].tolist() == [40, 41, 42, 43]

    # Partial slot 1: narrowed contiguous copy of 2 tokens, mapping 44..45.
    kv1 = calls[1].args[0]
    assert kv1 is not buffers[(1, 0)]
    assert kv1.shape[2] == 2
    assert kv1.is_contiguous()
    assert calls[1].args[2].tolist() == [44, 45]

    # Slot 2 stays aligned after the partial slot: tokens 6..9.
    assert calls[2].args[0] is buffers[(2, 0)]
    assert calls[2].args[2].tolist() == [11 * 4 + 2, 11 * 4 + 3, 12 * 4, 12 * 4 + 1]


# ---------------------------------------------------------------------------
# Retrieve: native plan builder (execute_cb_retrieve_plan fast path)
# ---------------------------------------------------------------------------


def _native_retrieve_plan_available() -> bool:
    """Return whether the C++ native retrieve-plan interfaces are available."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    return v3_mod._HAS_NATIVE_RETRIEVE_PLAN and hasattr(v3_mod.lmc_ops, "CBGroupSpec")


native_retrieve_plan_required = pytest.mark.skipif(
    not _native_retrieve_plan_available(),
    reason="requires native CacheBlend retrieve-plan C++ support",
)


def _build_plan_engine_and_context(
    num_groups: int = 2,
    max_batch: int = 2,
    spc: int = 4,
    num_layers: int = 2,
    head_size: int = 8,
    n_heads: int = 2,
):
    """Engine with the real ``_build_cb_retrieve_plan_flat`` bound, a fake GPU
    context with real CPU tensors, and a real ``_CBRopeState``. Kernel
    groups are plain (non-fused) K/V, so hidden_dim = n_heads * head_size."""
    # Standard
    import weakref

    # Third Party
    import torch

    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod
    import lmcache.c_ops as lmc_ops

    eng = MagicMock(spec=v3_mod.BlendV3Module)
    for name in (
        "_build_cb_retrieve_plan_flat",
        "_resolve_cb_plan_invariants",
        "_cb_slot_buffers",
    ):
        setattr(eng, name, getattr(v3_mod.BlendV3Module, name).__get__(eng))
    eng._cb_plan_invariants = weakref.WeakKeyDictionary()
    eng._cb_slot_staging = weakref.WeakKeyDictionary()

    hidden_dim = n_heads * head_size
    gpu_context = MagicMock()
    gpu_context.device = torch.device("cpu")
    gpu_context.kv_layer_groups_manager.num_kernel_groups = num_groups
    gpu_context.kv_layer_groups_manager.kernel_groups = [
        SimpleNamespace(
            tokens_per_block=4,
            slots_per_block=4,
            engine_group_idx=0,
            engine_kv_format=lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            shape_desc=SimpleNamespace(nb=100),
        )
        for _ in range(num_groups)
    ]
    kv_buffers = {
        (slot, group): torch.zeros(2, num_layers, spc, hidden_dim)
        for slot in range(max_batch)
        for group in range(num_groups)
    }
    gpu_context.get_temp_kernel_group_buffer.side_effect = lambda s, g: kv_buffers[
        (s, g)
    ]
    ptr_tensors = [torch.zeros(num_layers, dtype=torch.long) for _ in range(num_groups)]
    gpu_context.get_kernel_group_kv_pointers.side_effect = lambda g: ptr_tensors[g]
    gpu_context.get_engine_kv_format.side_effect = (
        lambda g: lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    )
    # One object group; each chunk memory object fills one flat slot.
    obj_bytes = sum(kv_buffers[(0, g)].numel() * 4 for g in range(num_groups))
    obj_buffers = [torch.zeros(obj_bytes, dtype=torch.uint8) for _ in range(max_batch)]
    gpu_context.get_temp_object_group_buffer.side_effect = lambda s, og: obj_buffers[s]

    rope_state = v3_mod._CBRopeState(
        head_size=head_size,
        is_neox_style=True,
        cos_sin_caches=[torch.zeros(64, head_size)],
        group_to_cache=[],
    )
    return eng, gpu_context, rope_state, obj_bytes


def _lazy_memory_obj(obj_bytes: int, address: int):
    """MemoryObj stand-in that passes the lazy-allocator gate and
    build_staging_copies' size/pointer checks."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.memory_allocators.lazy_memory_allocator import (
        LazyMemoryAllocator,
    )

    obj = MagicMock()
    obj.parent.return_value = MagicMock(spec=LazyMemoryAllocator)
    obj.raw_tensor = torch.zeros(obj_bytes, dtype=torch.uint8)
    obj.get_size.return_value = obj_bytes
    obj.data_ptr = obj.raw_tensor.data_ptr()
    obj.meta.address = address
    return obj


@native_retrieve_plan_required
def test_native_plan_specs_stamped_and_cached():
    """3 chunks, max_batch=2: per-group slot-mapping rows staged into the
    persistent device buffer and stamped into the cached invariant specs; a
    second build for the same context reuses the same spec objects (and the
    same staging buffer) and re-stamps them."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()

    def pair(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            _lazy_memory_obj(obj_bytes, address=cur_st * 1000),
        )

    # Chunks 0/1 shifted (old != cur), chunk 2 prefix (old == cur).
    runs = [[pair(0, 4, 100), pair(4, 8, 104), pair(8, 12, 8)]]
    cpu_block_tables = [
        (np.array([10, 11, 12], dtype=np.int64), 4),
        (np.array([20, 21, 22], dtype=np.int64), 4),
    ]

    plan = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
    )
    assert plan is not None
    group_specs, (_staging, _ropes, _scatters, step_offsets), keepalive = plan

    assert len(group_specs) == 2
    # keepalive: the persistent (num_groups, cap) device staging buffer.
    assert len(keepalive) == 1
    dev = keepalive[0]
    assert dev[0, :12].tolist() == [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    assert dev[1, :12].tolist() == [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
    # Each cached spec is stamped with its row of the staging buffer.
    assert group_specs[0].slot_mapping_base == dev[0].data_ptr()
    assert group_specs[0].slot_mapping_capacity == 12
    assert group_specs[1].slot_mapping_base == dev[1].data_ptr()
    # Wave split: max_batch=2 -> double-buffered waves of 1 chunk each -> 3 steps.
    assert step_offsets.shape[0] == 3

    # Second build for the same context reuses the cached invariant specs
    # (same objects) and the same staging buffer, re-stamped per request.
    def pair2(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            _lazy_memory_obj(obj_bytes, address=cur_st * 1000),
        )

    plan2 = eng._build_cb_retrieve_plan_flat(
        gpu_context,
        rope_state,
        cpu_block_tables,
        [[pair2(4, 8, 200)]],
        max_batch=2,
    )
    assert plan2 is not None
    group_specs2, _, keepalive2 = plan2
    assert group_specs2[0] is group_specs[0]  # cached, not rebuilt
    assert keepalive2[0] is dev  # staging buffer reused, not reallocated
    assert group_specs2[0].slot_mapping_base == keepalive2[0][0].data_ptr()
    assert group_specs2[0].slot_mapping_capacity == 4
    # pos 4..8 -> block 11 -> slots 44..47 for group 0.
    assert keepalive2[0][0, :4].tolist() == [44, 45, 46, 47]


@native_retrieve_plan_required
def test_flat_plan_tables_encode_every_work_item():
    """The flat tables encode one staging row per chunk (dest = its wave
    slot's buffer), rope rows only for shifted chunks x groups, scatter rows
    for all chunks x groups with cumulative token offsets, and monotone
    per-step CSR offsets."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()

    def pair(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            _lazy_memory_obj(obj_bytes, address=cur_st * 1000),
        )

    # Chunks 0/1 shifted, chunk 2 prefix (old == cur).
    runs = [[pair(0, 4, 100), pair(4, 8, 104), pair(8, 12, 8)]]
    cpu_block_tables = [
        (np.array([10, 11, 12], dtype=np.int64), 4),
        (np.array([20, 21, 22], dtype=np.int64), 4),
    ]

    plan = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
    )
    assert plan is not None
    _specs, (staging, ropes, scatters, step_offsets), _keep = plan

    # 3 chunks -> 3 staging rows; wave=1 alternates slots 0,1,0.
    assert staging.shape == (3, 4)
    slot_bufs = [gpu_context.get_temp_object_group_buffer(s, 0) for s in (0, 1)]
    assert staging[:, 0].tolist() == [
        slot_bufs[0].data_ptr(),
        slot_bufs[1].data_ptr(),
        slot_bufs[0].data_ptr(),
    ]
    # Rope rows: 2 shifted chunks x 2 groups.
    assert ropes.shape == (4, 4)
    assert sorted(set(ropes[:, 2].tolist())) == [100, 104]  # old_st values
    # Scatter rows: 3 chunks x 2 groups, token offsets 0,4,8 repeated per group.
    assert scatters.shape == (6, 4)
    assert scatters[:, 2].tolist() == [0, 0, 4, 4, 8, 8]
    assert scatters[:, 3].tolist() == [4] * 6
    # Step CSR: 3 steps of 1 chunk; scatter ends = chunks x groups.
    assert step_offsets.shape == (3, 3)
    assert step_offsets[:, 0].tolist() == [1, 2, 3]
    assert step_offsets[:, 2].tolist() == [2, 4, 6]
    assert bool(np.all(np.diff(step_offsets[:, 1]) >= 0))


@native_retrieve_plan_required
def test_flat_tables_alternate_disjoint_slot_halves():
    """Same double-buffer contract, asserted on the flat-table encoding."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context(
        max_batch=4
    )
    runs = [
        [
            (
                SimpleNamespace(cur_st=i * 4, cur_ed=i * 4 + 4, old_st=i * 4 + 100),
                _lazy_memory_obj(obj_bytes, address=i * 4),
            )
            for i in range(6)
        ]
    ]
    cpu_block_tables = [
        (np.arange(12, dtype=np.int64), 4),
        (np.arange(12, dtype=np.int64) + 100, 4),
    ]
    plan = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=4
    )
    assert plan is not None
    _specs, (_staging, _ropes, scatters, step_offsets), _keep = plan

    prev_slots: set[int] | None = None
    c0 = 0
    for c1 in step_offsets[:, 2].tolist():
        slots = set(np.asarray(scatters[c0:c1, 1]).tolist())
        assert slots <= {0, 1} or slots <= {2, 3}, "step must stay in one half"
        if prev_slots is not None:
            assert not (slots & prev_slots)
        prev_slots = slots
        c0 = c1


@native_retrieve_plan_required
def test_native_plan_falls_back_for_non_lazy_objects():
    """A non-lazy-allocator memory object disables the native plan."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()
    obj = _lazy_memory_obj(obj_bytes, address=0)
    obj.parent.return_value = object()  # not a LazyMemoryAllocator
    runs = [[(SimpleNamespace(cur_st=0, cur_ed=4, old_st=100), obj)]]
    cpu_block_tables = [
        (np.array([10], dtype=np.int64), 4),
        (np.array([20], dtype=np.int64), 4),
    ]
    assert (
        eng._build_cb_retrieve_plan_flat(
            gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
        )
        is None
    )


@native_retrieve_plan_required
def test_native_plan_falls_back_for_compressed_group():
    """A compressed group (tokens != slots per block) disables the plan."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()
    gpu_context.kv_layer_groups_manager.kernel_groups[1].slots_per_block = 2
    runs = [
        [
            (
                SimpleNamespace(cur_st=0, cur_ed=4, old_st=100),
                _lazy_memory_obj(obj_bytes, address=0),
            )
        ]
    ]
    cpu_block_tables = [
        (np.array([10], dtype=np.int64), 4),
        (np.array([20], dtype=np.int64), 4),
    ]
    assert (
        eng._build_cb_retrieve_plan_flat(
            gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
        )
        is None
    )
