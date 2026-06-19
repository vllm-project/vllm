# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capture import (
    FileRoutedExpertsStore,
    FullAttnBlockMap,
    RoutedExpertsCapturer,
    RoutedExpertsManager,
    compute_full_attn_block_map,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.v1.kv_cache_interface import FullAttentionSpec

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capture.capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    # Before binding, no capture hook.
    assert dummy_module.router.capture_fn is None

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    # After binding, hook should exist and be callable.
    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[9, 10]]))
    assert len(capturer.calls) == 1


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """n == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """n == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


_BLOCK_SIZE = 4
_NUM_LAYERS = 3
_TOP_K = 2


def _make_manager(
    *,
    num_blocks: int = 8,
    num_offload_blocks: int | None = None,
    block_size_factor: int = 1,
) -> RoutedExpertsManager:
    spec = FullAttentionSpec(
        block_size=_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
    )
    hf_config = SimpleNamespace(
        num_experts=16,
        num_experts_per_tok=_TOP_K,
        num_hidden_layers=_NUM_LAYERS,
    )
    # Unique instance_id per manager so the shared /dev/shm slot mmap never
    # collides across tests (the region O_EXCL-creates its file).
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_config),
        instance_id=f"retest_{uuid.uuid4().hex}",
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    mgr = RoutedExpertsManager(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        num_offload_blocks=num_offload_blocks,
        block_size_factor=block_size_factor,
    )
    return mgr


def _routing_for_blocks(block_ids: list[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        16,
        size=(len(block_ids) * _BLOCK_SIZE, _NUM_LAYERS, _TOP_K),
        dtype=np.uint8,
    )


def _slots_for_blocks(block_ids: list[int]) -> np.ndarray:
    return (
        np.array(block_ids).reshape(-1, 1) * _BLOCK_SIZE + np.arange(_BLOCK_SIZE)
    ).flatten()


def _map_1to1(gpu_ids: list[int], cpu_ids: list[int]) -> "FullAttnBlockMap":
    """Build a factor=1 single-group 1:1 block map (legacy semantics)."""
    return compute_full_attn_block_map(
        gpu_block_ids=np.array(gpu_ids, dtype=np.int64),
        cpu_block_ids=np.array(cpu_ids, dtype=np.int64),
        group_sizes=[len(gpu_ids)],
        block_indices=[0],
        attn_gid=0,
        block_size_factor=1,
    )


def test_manager_offload_store_load_roundtrip():
    """Routing data survives GPU block reuse via the offload buffer."""
    mgr = _make_manager(num_offload_blocks=4)

    # A request computes 2 full blocks; routing lands in the slot buffer.
    orig_blocks = [2, 5]
    orig_data = _routing_for_blocks(orig_blocks, seed=0)
    mgr.store_batch(orig_data, _slots_for_blocks(orig_blocks))

    # Connector stores the blocks to CPU blocks [1, 3]; offload buffer follows.
    cpu_blocks = [1, 3]
    mgr.store_routed_experts_to_cpu_blocks(_map_1to1(orig_blocks, cpu_blocks))

    # GPU blocks are freed and reused by another request: slots clobbered.
    mgr.store_batch(
        _routing_for_blocks(orig_blocks, seed=1), _slots_for_blocks(orig_blocks)
    )

    # CPU hit loads the blocks back into fresh GPU blocks [7, 0].
    new_blocks = [7, 0]
    mgr.load_routed_experts_from_cpu_blocks(_map_1to1(new_blocks, cpu_blocks))

    got = mgr.get(new_blocks, num_tokens=len(orig_blocks) * _BLOCK_SIZE)
    assert np.array_equal(got, orig_data)


def test_manager_offload_factor_gt1_roundtrip():
    """factor>1: several GPU blocks pack into one offloaded block."""
    factor = 3
    mgr = _make_manager(num_offload_blocks=4, block_size_factor=factor)

    # Request: 3 GPU blocks = exactly one offloaded block (factor=3).
    gpu_blocks = [2, 5, 6]
    orig_data = _routing_for_blocks(gpu_blocks, seed=7)
    mgr.store_batch(orig_data, _slots_for_blocks(gpu_blocks))

    # One store job covering the whole offloaded block (cpu block 1).
    block_map = compute_full_attn_block_map(
        gpu_block_ids=np.array(gpu_blocks, dtype=np.int64),
        cpu_block_ids=np.array([1], dtype=np.int64),
        group_sizes=[factor],
        block_indices=[0],
        attn_gid=0,
        block_size_factor=factor,
    )
    # Each GPU block lands in a distinct sub-block of CPU block 1.
    assert np.array_equal(block_map.cpu_block_ids, [1, 1, 1])
    assert np.array_equal(block_map.sub_offsets, [0, 1, 2])
    mgr.store_routed_experts_to_cpu_blocks(block_map)

    # Clobber the GPU slots.
    mgr.store_batch(
        _routing_for_blocks(gpu_blocks, seed=8), _slots_for_blocks(gpu_blocks)
    )

    # Load back into fresh GPU blocks [7, 0, 3].
    new_blocks = [7, 0, 3]
    load_map = compute_full_attn_block_map(
        gpu_block_ids=np.array(new_blocks, dtype=np.int64),
        cpu_block_ids=np.array([1], dtype=np.int64),
        group_sizes=[factor],
        block_indices=[0],
        attn_gid=0,
        block_size_factor=factor,
    )
    mgr.load_routed_experts_from_cpu_blocks(load_map)

    got = mgr.get(new_blocks, num_tokens=len(gpu_blocks) * _BLOCK_SIZE)
    assert np.array_equal(got, orig_data)


def test_manager_offload_buffer_not_initialized_raises():
    mgr = _make_manager(num_offload_blocks=None)
    with pytest.raises(RuntimeError, match="offload buffer is not initialized"):
        mgr.store_routed_experts_to_cpu_blocks(_map_1to1([0], [0]))
    with pytest.raises(RuntimeError, match="offload buffer is not initialized"):
        mgr.load_routed_experts_from_cpu_blocks(_map_1to1([0], [0]))


def test_manager_offload_empty_jobs_are_noop():
    mgr = _make_manager(num_offload_blocks=4)
    before = mgr.routed_experts_by_slot.copy()
    empty = compute_full_attn_block_map(
        gpu_block_ids=np.array([], dtype=np.int64),
        cpu_block_ids=np.array([], dtype=np.int64),
        group_sizes=[0],
        block_indices=[0],
        attn_gid=0,
        block_size_factor=1,
    )
    mgr.store_routed_experts_to_cpu_blocks(empty)
    mgr.load_routed_experts_from_cpu_blocks(empty)
    assert np.array_equal(mgr.routed_experts_by_slot, before)


def test_compute_full_attn_block_map_factor1_identity():
    """factor=1: 1:1 mapping, sub-offsets all zero."""
    m = compute_full_attn_block_map(
        gpu_block_ids=np.array([4, 9, 1], dtype=np.int64),
        cpu_block_ids=np.array([2, 5, 8], dtype=np.int64),
        group_sizes=[3],
        block_indices=[0],
        attn_gid=0,
        block_size_factor=1,
    )
    assert np.array_equal(m.gpu_block_ids, [4, 9, 1])
    assert np.array_equal(m.cpu_block_ids, [2, 5, 8])
    assert np.array_equal(m.sub_offsets, [0, 0, 0])


def test_compute_full_attn_block_map_first_block_skip():
    """block_indices unaligned to factor folds into the sub-block index."""
    factor = 4
    # Group starts at GPU logical block 6 -> skip = 6 % 4 = 2: the first
    # offloaded block in this job is partial (sub-blocks 2,3 used).
    # 5 GPU blocks => offloaded blocks: cdiv(5 + 2, 4) = 2.
    m = compute_full_attn_block_map(
        gpu_block_ids=np.array([10, 11, 12, 13, 14], dtype=np.int64),
        cpu_block_ids=np.array([7, 9], dtype=np.int64),
        group_sizes=[5],
        block_indices=[6],
        attn_gid=0,
        block_size_factor=factor,
    )
    # sub = (2 + [0..4]) % 4 = [2,3,0,1,2]; cpu_loc = (2+[0..4])//4 = [0,0,1,1,1]
    assert np.array_equal(m.sub_offsets, [2, 3, 0, 1, 2])
    assert np.array_equal(m.cpu_block_ids, [7, 7, 9, 9, 9])
    assert np.array_equal(m.gpu_block_ids, [10, 11, 12, 13, 14])


def test_compute_full_attn_block_map_multi_group_anchor():
    """Anchor group at gid=1 (e.g. MLA after a linear group): only it is sliced."""
    factor = 2
    # group 0 (non-anchor): 3 GPU blocks, skip = block_indices[0]%2 = 0 ->
    #   consumes cdiv(3,2)=2 CPU blocks. group 1 (anchor): 4 GPU blocks.
    m = compute_full_attn_block_map(
        gpu_block_ids=np.array([0, 1, 2, 20, 21, 22, 23], dtype=np.int64),
        cpu_block_ids=np.array([100, 101, 200, 201], dtype=np.int64),
        group_sizes=[3, 4],
        block_indices=[0, 0],
        attn_gid=1,
        block_size_factor=factor,
    )
    assert np.array_equal(m.gpu_block_ids, [20, 21, 22, 23])
    # cpu_off = cdiv(3+0,2)=2 -> anchor CPU blocks start at index 2: [200,201]
    assert np.array_equal(m.sub_offsets, [0, 1, 0, 1])
    assert np.array_equal(m.cpu_block_ids, [200, 200, 201, 201])


def test_compute_full_attn_block_map_contract_violation_raises():
    """The group-major flat-order contract is validated inside the helper."""
    # len(cpu) != sum(cdiv per group): 3 GPU blocks at factor=2 need 2 CPU
    # blocks, but only 1 is supplied.
    with pytest.raises(RuntimeError, match="flat-order contract"):
        compute_full_attn_block_map(
            gpu_block_ids=np.array([0, 1, 2], dtype=np.int64),
            cpu_block_ids=np.array([100], dtype=np.int64),
            group_sizes=[3],
            block_indices=[0],
            attn_gid=0,
            block_size_factor=2,
        )
    # expected_num_groups mismatch (job spans 1 group, 2 declared).
    with pytest.raises(RuntimeError, match="flat-order contract"):
        compute_full_attn_block_map(
            gpu_block_ids=np.array([0, 1, 2], dtype=np.int64),
            cpu_block_ids=np.array([100, 101, 102], dtype=np.int64),
            group_sizes=[3],
            block_indices=[0],
            attn_gid=0,
            block_size_factor=1,
            expected_num_groups=2,
        )


def test_file_routed_experts_store_roundtrip(tmp_path):
    """FileRoutedExpertsStore persists and restores offloaded-block rows by key."""
    row_shape = (2, _BLOCK_SIZE, _NUM_LAYERS, _TOP_K)
    dtype = np.dtype(np.uint8)

    class _StubMapper:
        def get_file_name(self, key: bytes) -> str:
            return str(tmp_path / f"{key.hex()}.bin")

    store = FileRoutedExpertsStore(
        root_dir=str(tmp_path),
        file_mapper=_StubMapper(),
        row_shape=row_shape,
        dtype=dtype,
    )
    keys = [b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"]
    rng = np.random.default_rng(3)
    rows = rng.integers(0, 16, size=(2, *row_shape), dtype=np.uint8)

    store.persist(keys, rows)
    got = store.restore(keys)
    assert got is not None
    assert np.array_equal(got, rows)

    # Missing key -> None (never partial).
    assert store.restore([keys[0], b"\xde\xad\xbe\xef"]) is None


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1
