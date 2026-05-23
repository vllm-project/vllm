# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.mamba_utils import (
    MambaCopyBuffers,
    MambaSpecDecodeGPUContext,
    collect_mamba_copy_meta,
    do_mamba_copy_block,
    preprocess_mamba,
)

MambaStateCopyFunc = Callable[..., Any]

# Conv + temporal copy specs, in the order the tests' MambaSpec shapes expect.
_COPY_FUNCS: tuple[MambaStateCopyFunc, ...] = (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)


def postprocess_mamba(
    scheduler_output: "SchedulerOutput",
    kv_cache_config: "KVCacheConfig",
    input_batch: Any,
    requests: dict[str, Any],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: "MambaCopyBuffers",
):
    """CPU reference for the align-mode postprocess.

    Used as a golden against the GPU fused kernel (``postprocess_mamba_align_gpu``).
    Mirrors what the production code did before the fused kernel replaced it;
    kept here because production no longer has a CPU implementation.
    """
    assert input_batch.mamba_state_idx_cpu is not None
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    mamba_state_idx_cpu = input_batch.mamba_state_idx_cpu
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx_cpu[i]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


def _make_scheduler_output(
    finished_req_ids: set[str],
    preempted_req_ids: set[str] | None,
    resumed_req_ids: set[str],
) -> SchedulerOutput:
    cached = CachedRequestData.make_empty()
    cached.resumed_req_ids = resumed_req_ids
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids,
        free_encoder_mm_hashes=[],
        preempted_req_ids=preempted_req_ids,
    )


def test_resumed_req_ids_cleared_from_mamba_state_idx():
    """When a request is force-preempted (e.g. reset_prefix_cache),
    it appears in resumed_req_ids but NOT in preempted_req_ids.
    preprocess_mamba must still clear its mamba_state_idx entry,
    otherwise stale indices can point beyond the new block allocation.
    """
    spec = MagicMock(block_size=64, num_speculative_blocks=0)
    cache_config = MagicMock(enable_prefix_caching=True)
    input_batch = MagicMock(req_ids=[])
    copy_bufs = MagicMock(mamba_group_ids=[0], mamba_spec=spec)

    mamba_state_idx: dict[str, int] = {
        "finished": 1,
        "preempted": 2,
        "resumed": 3,  # only in resumed_req_ids, NOT in preempted
        "keep": 99,
    }
    sched = _make_scheduler_output(
        finished_req_ids={"finished"},
        preempted_req_ids={"preempted"},
        resumed_req_ids={"resumed"},
    )

    with patch(
        "vllm.v1.worker.mamba_utils.get_mamba_groups",
        return_value=([0], spec),
    ):
        preprocess_mamba(
            sched,
            MagicMock(),  # kv_cache_config
            cache_config,
            mamba_state_idx,
            input_batch,
            {},  # requests
            {},  # forward_context
            (),  # mamba_state_copy_funcs
            copy_bufs,
        )

    assert mamba_state_idx == {"keep": 99}


# -----------------------------------------------------------------------------
# Golden tests for postprocess_mamba_fused_kernel
# -----------------------------------------------------------------------------


@dataclass
class _TestConfig:
    """Common test configuration for fused kernel tests."""

    block_size: int = 16
    num_blocks: int = 32
    num_layers: int = 2
    num_reqs: int = 4
    max_num_reqs: int = 8
    # Conv state shape: [num_blocks, conv_width, inner_dim]
    conv_width: int = 4
    conv_inner_dim: int = 64
    # Temporal state shape: [num_blocks, state_dim]
    temporal_state_dim: int = 128
    dtype: torch.dtype = torch.float16


class _MockCpuGpuBuffer:
    """Mock CpuGpuBuffer for testing without pinned memory."""

    def __init__(self, size: int, dtype: torch.dtype, device: torch.device):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.gpu = torch.zeros(size, dtype=dtype, device=device)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)


def _make_postprocess_scheduler_output(
    req_ids: list[str],
    num_scheduled_tokens: dict[str, int],
    scheduled_spec_decode_tokens: dict[str, list] | None = None,
) -> SchedulerOutput:
    """Create a minimal SchedulerOutput for postprocess testing."""
    cached = CachedRequestData.make_empty()
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens or {},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=set(),
    )


def _make_mock_attention(
    conv_state: torch.Tensor, temporal_state: torch.Tensor
) -> MagicMock:
    """Create a mock attention object with kv_cache."""
    attention = MagicMock()
    attention.kv_cache = [conv_state, temporal_state]
    return attention


def _make_dual_states(
    cfg: "_TestConfig",
    layer_names: list[str],
    device: torch.device,
    *,
    num_blocks: int | None = None,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    dict[str, MagicMock],
    dict[str, MagicMock],
]:
    """Allocate conv+temporal state tensors for the Python path, clone them for
    the GPU path, and build matching ``forward_context`` dicts for both.

    Returns ``(conv_py, temporal_py, conv_gpu, temporal_gpu, fwd_py, fwd_gpu)``
    where the four state lists are parallel to ``layer_names``.
    """
    n_blocks = num_blocks if num_blocks is not None else cfg.num_blocks
    conv_py = [
        torch.randn(
            n_blocks,
            cfg.conv_width,
            cfg.conv_inner_dim,
            dtype=cfg.dtype,
            device=device,
        )
        for _ in layer_names
    ]
    temporal_py = [
        torch.randn(n_blocks, cfg.temporal_state_dim, dtype=cfg.dtype, device=device)
        for _ in layer_names
    ]
    conv_gpu = [s.clone() for s in conv_py]
    temporal_gpu = [s.clone() for s in temporal_py]
    fwd_py = {
        name: _make_mock_attention(c, t)
        for name, c, t in zip(layer_names, conv_py, temporal_py)
    }
    fwd_gpu = {
        name: _make_mock_attention(c, t)
        for name, c, t in zip(layer_names, conv_gpu, temporal_gpu)
    }
    return conv_py, temporal_py, conv_gpu, temporal_gpu, fwd_py, fwd_gpu


def _make_dual_layer_state(
    cfg: "_TestConfig",
    device: torch.device,
    *,
    num_blocks: int | None = None,
    layer_name: str = "layer_0",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, MagicMock],
    dict[str, MagicMock],
]:
    """Single-layer convenience form of ``_make_dual_states``."""
    conv_py, temporal_py, conv_gpu, temporal_gpu, fwd_py, fwd_gpu = _make_dual_states(
        cfg, [layer_name], device, num_blocks=num_blocks
    )
    return conv_py[0], temporal_py[0], conv_gpu[0], temporal_gpu[0], fwd_py, fwd_gpu


def _make_kv_cache_config(cfg: _TestConfig, layer_names: list[str]) -> KVCacheConfig:
    """Create a KVCacheConfig with mamba groups."""
    mamba_spec = MambaSpec(
        block_size=cfg.block_size,
        shapes=(
            (cfg.conv_width, cfg.conv_inner_dim),
            (cfg.temporal_state_dim,),
        ),
        dtypes=(cfg.dtype, cfg.dtype),
        mamba_cache_mode="all",
    )
    group = KVCacheGroupSpec(
        layer_names=layer_names,
        kv_cache_spec=mamba_spec,
    )
    return KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[group],
    )


def _make_input_batch(
    req_ids: list[str],
    num_accepted_tokens: list[int],
    mamba_state_idx: list[int],
) -> MagicMock:
    """Create a mock GPUInputBatch."""
    batch = MagicMock()
    batch.req_ids = req_ids
    batch.req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
    # Use numpy arrays so modifications persist
    batch.num_accepted_tokens_cpu = np.array(num_accepted_tokens, dtype=np.int32)
    batch.mamba_state_idx_cpu = np.array(mamba_state_idx, dtype=np.int32)
    return batch


def _make_requests(
    req_ids: list[str],
    num_computed_tokens: list[int],
    block_ids_per_req: list[list[int]],
) -> dict[str, MagicMock]:
    """Create mock CachedRequestState objects."""
    requests = {}
    for i, req_id in enumerate(req_ids):
        req = MagicMock()
        req.num_computed_tokens = num_computed_tokens[i]
        req.block_ids = {0: block_ids_per_req[i]}  # group_id=0
        requests[req_id] = req
    return requests


def _make_copy_bufs(
    cfg: _TestConfig, kv_cache_config: KVCacheConfig, device: torch.device
) -> MambaCopyBuffers:
    """Create MambaCopyBuffers for the Python path."""

    def make_buffer(n, dtype):
        return _MockCpuGpuBuffer(n, dtype, device)

    return MambaCopyBuffers.create(
        max_num_reqs=cfg.max_num_reqs,
        kv_cache_config=kv_cache_config,
        copy_funcs=(get_conv_copy_spec, get_temporal_copy_spec),
        make_buffer=make_buffer,
    )


def _make_gpu_ctx(
    cfg: _TestConfig, kv_cache_config: KVCacheConfig, device: torch.device
) -> MambaSpecDecodeGPUContext:
    """Create MambaSpecDecodeGPUContext for the GPU path."""

    def make_buffer(n, dtype):
        return _MockCpuGpuBuffer(n, dtype, device)

    return MambaSpecDecodeGPUContext.create(
        max_num_reqs=cfg.max_num_reqs,
        kv_cache_config=kv_cache_config,
        num_state_types=2,
        device=device,
        make_buffer=make_buffer,
    )


def _run_gpu_postprocess(
    gpu_ctx: MambaSpecDecodeGPUContext,
    *,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    copy_funcs: tuple,
    block_table: torch.Tensor,
    req_ids: list[str],
    num_accepted_tokens: list[int],
    mamba_state_idx: list[int],
    num_scheduled_tokens: dict[str, int],
    num_computed_tokens: list[int],
    num_draft_tokens: dict[str, int],
    device: torch.device,
) -> None:
    """Initialize the GPU context against `block_table`, run the fused
    postprocess kernel for `req_ids`, and synchronize."""

    def t(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    gpu_ctx.initialize_from_forward_context(
        kv_cache_config, forward_context, copy_funcs, [block_table]
    )
    gpu_ctx.run_fused_postprocess(
        num_reqs=len(req_ids),
        num_accepted_tokens_gpu=t(num_accepted_tokens),
        mamba_state_idx_gpu=t(mamba_state_idx),
        num_scheduled_tokens_gpu=t([num_scheduled_tokens[r] for r in req_ids]),
        num_computed_tokens_gpu=t(num_computed_tokens),
        num_draft_tokens_gpu=t([num_draft_tokens.get(r, 0) for r in req_ids]),
    )
    torch.accelerator.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPostprocessMambaFusedKernel:
    """Tests for postprocess_mamba_fused_kernel comparing GPU vs CPU paths."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @pytest.fixture
    def test_config(self):
        return _TestConfig()

    def test_matches_python_postprocess_mamba(self, device, test_config):
        """
        Golden test: GPU kernel produces identical results to Python impl.

        This test:
        1. Sets up identical initial state for both paths
        2. Runs Python postprocess_mamba (modifies states via batch_memcpy)
        3. Runs GPU fused kernel (modifies states directly)
        4. Compares resulting state tensors and num_accepted_tokens
        """
        cfg = test_config
        torch.manual_seed(42)

        # Test scenario: 4 requests with different copy conditions
        # Copy needed when: aligned_new_computed >= num_tokens_running_state
        # where: num_tokens_running_state = num_computed + num_scheduled - num_draft
        #        new_num_computed = num_tokens_running_state + num_accepted - 1
        #        aligned_new_computed = (new_num_computed // block_size) * block_size
        req_ids = ["req_0", "req_1", "req_2", "req_3"]

        # Configure requests so some need copies, some don't
        # block_size = 16
        # req_0: running=60+5-2=63, new=63+3-1=65, aligned=64 >= 63 -> COPY
        # req_1: running=30+3-0=33, new=33+2-1=34, aligned=32 < 33 -> NO COPY
        # req_2: running=45+8-3=50, new=50+4-1=53, aligned=48 < 50 -> NO COPY
        # req_3: running=10+6-0=16, new=16+2-1=17, aligned=16 >= 16 -> COPY
        num_computed_tokens = [60, 30, 45, 10]
        num_scheduled_tokens = {"req_0": 5, "req_1": 3, "req_2": 8, "req_3": 6}
        num_draft_tokens = {"req_0": 2, "req_1": 0, "req_2": 3, "req_3": 0}
        num_accepted_tokens = [3, 2, 4, 2]
        mamba_state_idx = [3, 1, 2, 0]  # source block indices

        # Block IDs for each request (simulate block table)
        block_ids_per_req = [
            list(range(8)),  # req_0: blocks 0-7
            list(range(8, 16)),  # req_1: blocks 8-15
            list(range(16, 24)),  # req_2: blocks 16-23
            list(range(24, 32)),  # req_3: blocks 24-31
        ]

        layer_names = [f"layer_{i}" for i in range(cfg.num_layers)]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        (
            conv_states_py,
            temporal_states_py,
            conv_states_gpu,
            temporal_states_gpu,
            forward_context_py,
            forward_context_gpu,
        ) = _make_dual_states(cfg, layer_names, device)

        # --- Run Python path ---
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # --- Run GPU path ---
        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)

        num_reqs = len(req_ids)
        max_blocks = max(len(b) for b in block_ids_per_req)
        block_table_gpu = torch.zeros(
            num_reqs, max_blocks, dtype=torch.int32, device=device
        )
        for i, block_ids in enumerate(block_ids_per_req):
            block_table_gpu[i, : len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )
        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # --- Compare results ---
        # 1. Compare state tensors
        for i in range(cfg.num_layers):
            torch.testing.assert_close(
                conv_states_gpu[i],
                conv_states_py[i],
                msg=f"Conv state mismatch at layer {i}",
            )
            torch.testing.assert_close(
                temporal_states_gpu[i],
                temporal_states_py[i],
                msg=f"Temporal state mismatch at layer {i}",
            )

        # 2. Compare num_accepted_tokens updates
        expected_accepted = torch.tensor(
            input_batch_py.num_accepted_tokens_cpu[:num_reqs],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(
            gpu_ctx.num_accepted_tokens_out[:num_reqs],
            expected_accepted,
            msg="num_accepted_tokens mismatch",
        )

    def test_no_copy_when_not_needed(self, device, test_config):
        """Kernel should not modify state when no copy is needed."""
        cfg = test_config
        torch.manual_seed(123)

        # Single request where no copy is needed:
        # running = 30 + 3 = 33, new = 33 + 1 - 1 = 33, aligned = 32 < 33
        req_ids = ["req_0"]
        num_computed_tokens = [30]
        num_scheduled_tokens = {"req_0": 3}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [1]
        mamba_state_idx = [1]
        block_ids_per_req = [list(range(8))]

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        # Create state tensor
        conv_state = torch.randn(
            cfg.num_blocks,
            cfg.conv_width,
            cfg.conv_inner_dim,
            dtype=cfg.dtype,
            device=device,
        )
        temporal_state = torch.randn(
            cfg.num_blocks, cfg.temporal_state_dim, dtype=cfg.dtype, device=device
        )

        # Clone to verify no modification
        conv_state_orig = conv_state.clone()
        temporal_state_orig = temporal_state.clone()

        forward_context = {"layer_0": _make_mock_attention(conv_state, temporal_state)}

        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)

        num_reqs = len(req_ids)
        block_table_gpu = torch.zeros(num_reqs, 8, dtype=torch.int32, device=device)
        block_table_gpu[0, :8] = torch.tensor(block_ids_per_req[0], dtype=torch.int32)

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # State should be unchanged
        torch.testing.assert_close(conv_state, conv_state_orig)
        torch.testing.assert_close(temporal_state, temporal_state_orig)

    @pytest.mark.parametrize("num_reqs", [1, 2, 8, 16])
    def test_various_batch_sizes(self, device, test_config, num_reqs):
        """Verify kernel works correctly with different batch sizes."""
        cfg = _TestConfig(max_num_reqs=max(16, num_reqs))
        torch.manual_seed(456)

        req_ids = [f"req_{i}" for i in range(num_reqs)]
        # All requests will trigger a copy
        num_computed_tokens = [60] * num_reqs
        num_scheduled_tokens = {r: 5 for r in req_ids}
        num_draft_tokens = {r: 0 for r in req_ids}
        num_accepted_tokens = [3] * num_reqs
        mamba_state_idx = [3] * num_reqs
        # Each request gets unique blocks
        block_ids_per_req = [list(range(i * 8, (i + 1) * 8)) for i in range(num_reqs)]

        # Ensure we have enough blocks
        total_blocks = num_reqs * 8
        cfg = _TestConfig(num_blocks=total_blocks, max_num_reqs=max(16, num_reqs))

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        (
            conv_state_py,
            temporal_state_py,
            conv_state_gpu,
            temporal_state_gpu,
            forward_context_py,
            forward_context_gpu,
        ) = _make_dual_layer_state(cfg, device)

        # Run Python path
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # Run GPU path
        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)

        max_blocks_per_req = 8
        block_table_gpu = torch.zeros(
            num_reqs, max_blocks_per_req, dtype=torch.int32, device=device
        )
        for i, block_ids in enumerate(block_ids_per_req):
            block_table_gpu[i, : len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # Compare results
        torch.testing.assert_close(
            conv_state_gpu, conv_state_py, msg="Conv state mismatch"
        )
        torch.testing.assert_close(
            temporal_state_gpu, temporal_state_py, msg="Temporal state mismatch"
        )

    def test_block_table_with_realistic_stride(self, device, test_config):
        """
        Test kernel with realistic block table strides.

        In real usage, the block table is pre-allocated with shape
        [max_num_reqs, max_num_blocks_per_req] and then sliced to
        [:num_reqs]. This means stride(0) = max_num_blocks_per_req,
        which is typically much larger than the actual blocks used.

        This test verifies the kernel handles non-tight strides correctly,
        catching bugs where stride is incorrectly treated as bytes vs elements.
        """
        cfg = test_config
        torch.manual_seed(789)

        # Use multiple requests to exercise stride-based indexing
        num_reqs = 4
        req_ids = [f"req_{i}" for i in range(num_reqs)]

        # All requests trigger copies (same setup as test_various_batch_sizes)
        num_computed_tokens = [60] * num_reqs
        num_scheduled_tokens = {r: 5 for r in req_ids}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [3] * num_reqs
        mamba_state_idx = [3] * num_reqs

        # Each request uses only 8 blocks, but we allocate much more
        blocks_used_per_req = 8
        block_ids_per_req = [
            list(range(i * blocks_used_per_req, (i + 1) * blocks_used_per_req))
            for i in range(num_reqs)
        ]

        total_blocks = num_reqs * blocks_used_per_req
        cfg = _TestConfig(num_blocks=total_blocks, max_num_reqs=max(16, num_reqs))

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        (
            conv_state_py,
            temporal_state_py,
            conv_state_gpu,
            temporal_state_gpu,
            forward_context_py,
            forward_context_gpu,
        ) = _make_dual_layer_state(cfg, device)

        # Run Python path
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # Run GPU path with REALISTIC block table stride
        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)

        # KEY DIFFERENCE: Create a large block table like real code does
        # Real system has max_num_blocks_per_req >> blocks actually used
        max_num_reqs_full = 16
        max_blocks_per_req_full = 512  # Much larger than blocks_used_per_req=8

        # Allocate full-size table (simulates pre-allocated CpuGpuBuffer)
        block_table_full = torch.zeros(
            max_num_reqs_full, max_blocks_per_req_full, dtype=torch.int32, device=device
        )

        # Fill in actual block IDs (only first few columns used)
        for i, block_ids in enumerate(block_ids_per_req):
            block_table_full[i, : len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )

        # Slice like real code: block_table.gpu[:num_reqs]
        # This preserves stride(0) = 512, not 8!
        block_table_gpu = block_table_full[:num_reqs]

        # Verify stride is large (the key property we're testing)
        assert block_table_gpu.stride(0) == max_blocks_per_req_full, (
            f"Expected stride {max_blocks_per_req_full}, "
            f"got {block_table_gpu.stride(0)}"
        )

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # Compare results - this will fail if stride handling is incorrect
        torch.testing.assert_close(
            conv_state_gpu,
            conv_state_py,
            msg="Conv state mismatch - possible stride bug in kernel",
        )
        torch.testing.assert_close(
            temporal_state_gpu,
            temporal_state_py,
            msg="Temporal state mismatch - possible stride bug in kernel",
        )

    def test_src_addr_equals_dst_addr_skips_copy_and_sets_accepted_to_1(
        self, device, test_config
    ):
        """
        Test the ``src_addr == dst_addr`` early-return path in
        postprocess_mamba_fused_kernel matches Python behavior.

        When src_addr == dst_addr (source and destination memory addresses are
        identical), both implementations should:
        1. Skip the copy (state unchanged)
        2. Set num_accepted_tokens to 1

        This condition occurs when:
        - src_block_idx == dest_block_idx (same logical block)
        - accept_token_bias == 0 (no offset within the block)

        Python reference (collect_mamba_copy_meta):
            if src_block_idx == dest_block_idx and accept_token_bias == 0:
                return  # No copy added

        Python reference (postprocess_mamba):
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1

        Test setup (block_size=16):
        - num_tokens_running_state = 30 + 2 - 0 = 32
        - new_num_computed = 32 + 1 - 1 = 32
        - aligned_new_computed = 32
        - accept_token_bias = 32 - 32 = 0
        - dest_block_idx = 32 // 16 - 1 = 1
        - src_block_idx = 1 (set explicitly)
        """
        cfg = test_config
        torch.manual_seed(1001)

        req_ids = ["req_0"]
        num_computed_tokens = [30]
        num_scheduled_tokens = {"req_0": 2}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [5]  # Initial value, should become 1
        mamba_state_idx = [1]  # src_block_idx = 1 = dest_block_idx
        block_ids_per_req = [list(range(8))]

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        (
            conv_state_py,
            temporal_state_py,
            conv_state_gpu,
            temporal_state_gpu,
            forward_context_py,
            forward_context_gpu,
        ) = _make_dual_layer_state(cfg, device)

        # Also clone to verify no modification
        conv_state_orig = conv_state_py.clone()
        temporal_state_orig = temporal_state_py.clone()

        # --- Run Python path ---
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # --- Run GPU path ---
        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)

        num_reqs = len(req_ids)
        block_table_gpu = torch.zeros(num_reqs, 8, dtype=torch.int32, device=device)
        block_table_gpu[0, :8] = torch.tensor(block_ids_per_req[0], dtype=torch.int32)

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # --- Verify Python behavior (ground truth) ---
        # State should be unchanged (no copy when src_addr == dst_addr)
        torch.testing.assert_close(
            conv_state_py,
            conv_state_orig,
            msg="Python: Conv state should be unchanged when src==dst",
        )
        torch.testing.assert_close(
            temporal_state_py,
            temporal_state_orig,
            msg="Python: Temporal state should be unchanged when src==dst",
        )
        # num_accepted_tokens should be 1
        assert input_batch_py.num_accepted_tokens_cpu[0] == 1, (
            f"Python: num_accepted_tokens should be 1, "
            f"got {input_batch_py.num_accepted_tokens_cpu[0]}"
        )

        # --- Verify GPU matches Python ---
        torch.testing.assert_close(
            conv_state_gpu,
            conv_state_py,
            msg="GPU conv state should match Python",
        )
        torch.testing.assert_close(
            temporal_state_gpu,
            temporal_state_py,
            msg="GPU temporal state should match Python",
        )
        expected_accepted = torch.tensor(
            input_batch_py.num_accepted_tokens_cpu[:num_reqs],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(
            gpu_ctx.num_accepted_tokens_out[:num_reqs],
            expected_accepted,
            msg="GPU num_accepted_tokens should match Python",
        )

    def test_same_block_idx_with_offset_copies_then_sets_accepted_to_1(
        self, device, test_config
    ):
        """
        Test the ``src_block_idx == dest_block_idx`` post-copy update in
        postprocess_mamba_fused_kernel matches Python behavior.

        When src_block_idx == dest_block_idx but accept_token_bias > 0, both
        implementations should:
        1. Perform the copy (src_addr != dst_addr due to offset)
        2. Set num_accepted_tokens to 1 AFTER the copy

        Python reference (postprocess_mamba):
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1

        For conv states: copies state[block, offset:] to
            state[block, :] (shifted window)
        For temporal states: copies state[block_ids[src_idx + offset]] to
            state[block_ids[dest_idx]]

        Test setup (block_size=16):
        - num_tokens_running_state = 30 + 1 - 0 = 31
        - new_num_computed = 31 + 2 - 1 = 32
        - aligned_new_computed = 32
        - accept_token_bias = 32 - 31 = 1 (> 0, so copy happens)
        - dest_block_idx = 32 // 16 - 1 = 1
        - src_block_idx = 1 (set explicitly, == dest_block_idx)
        """
        cfg = test_config
        torch.manual_seed(1002)

        req_ids = ["req_0"]
        num_computed_tokens = [30]
        num_scheduled_tokens = {"req_0": 1}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [2]  # Results in accept_token_bias = 1
        mamba_state_idx = [1]  # src_block_idx = 1 = dest_block_idx
        block_ids_per_req = [list(range(8))]

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        (
            conv_state_py,
            temporal_state_py,
            conv_state_gpu,
            temporal_state_gpu,
            forward_context_py,
            forward_context_gpu,
        ) = _make_dual_layer_state(cfg, device)

        # Clone to verify modification
        conv_state_orig = conv_state_py.clone()
        temporal_state_orig = temporal_state_py.clone()

        # --- Run Python path ---
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # --- Run GPU path ---
        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)

        num_reqs = len(req_ids)
        block_table_gpu = torch.zeros(num_reqs, 8, dtype=torch.int32, device=device)
        block_table_gpu[0, :8] = torch.tensor(block_ids_per_req[0], dtype=torch.int32)

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # --- Verify Python behavior (ground truth) ---
        dest_block_id = block_ids_per_req[0][1]  # dest_block_idx = 1

        # Conv state should be modified (shifted copy within block)
        conv_changed = not torch.allclose(
            conv_state_py[dest_block_id], conv_state_orig[dest_block_id]
        )
        assert conv_changed, (
            "Python: Conv state should be modified when accept_token_bias > 0"
        )

        # Temporal state should be modified (copy from different block)
        src_block_id_temporal = block_ids_per_req[0][2]  # actual_src_block_idx = 2
        dest_block_id_temporal = block_ids_per_req[0][1]  # dest_block_idx = 1
        torch.testing.assert_close(
            temporal_state_py[dest_block_id_temporal],
            temporal_state_orig[src_block_id_temporal],
            msg="Python: Temporal state copy should have happened",
        )

        # num_accepted_tokens should be 1
        assert input_batch_py.num_accepted_tokens_cpu[0] == 1, (
            f"Python: num_accepted_tokens should be 1, "
            f"got {input_batch_py.num_accepted_tokens_cpu[0]}"
        )

        # --- Verify GPU matches Python ---
        torch.testing.assert_close(
            conv_state_gpu,
            conv_state_py,
            msg="GPU conv state should match Python",
        )
        torch.testing.assert_close(
            temporal_state_gpu,
            temporal_state_py,
            msg="GPU temporal state should match Python",
        )
        expected_accepted = torch.tensor(
            input_batch_py.num_accepted_tokens_cpu[:num_reqs],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(
            gpu_ctx.num_accepted_tokens_out[:num_reqs],
            expected_accepted,
            msg="GPU num_accepted_tokens should match Python",
        )

    def test_different_block_idx_copies_without_setting_accepted_to_1(
        self, device, test_config
    ):
        """
        Test that neither special-case path triggers when
        src_block_idx != dest_block_idx, and GPU matches Python behavior.

        When copying between different blocks:
        1. src_addr != dst_addr (different blocks = different addresses)
        2. src_block_idx != dest_block_idx

        Therefore:
        - The ``src_addr == dst_addr`` early-return does NOT trigger
        - The ``src_block_idx == dest_block_idx`` post-copy update does NOT trigger
        - Copy happens normally
        - num_accepted_tokens remains UNCHANGED

        Test setup (block_size=16):
        - num_tokens_running_state = 60 + 3 - 0 = 63
        - new_num_computed = 63 + 3 - 1 = 65
        - aligned_new_computed = 64
        - accept_token_bias = 64 - 63 = 1
        - dest_block_idx = 64 // 16 - 1 = 3
        - src_block_idx = 2 (set explicitly, != dest_block_idx)
        """
        cfg = test_config
        torch.manual_seed(1003)

        req_ids = ["req_0"]
        num_computed_tokens = [60]
        num_scheduled_tokens = {"req_0": 3}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [3]  # Should remain 3, NOT set to 1
        mamba_state_idx = [2]  # src_block_idx = 2, dest_block_idx will be 3
        block_ids_per_req = [list(range(8))]

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        (
            conv_state_py,
            temporal_state_py,
            conv_state_gpu,
            temporal_state_gpu,
            forward_context_py,
            forward_context_gpu,
        ) = _make_dual_layer_state(cfg, device)

        # Clone to verify modification
        conv_state_orig = conv_state_py.clone()

        # --- Run Python path ---
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # --- Run GPU path ---
        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)

        num_reqs = len(req_ids)
        block_table_gpu = torch.zeros(num_reqs, 8, dtype=torch.int32, device=device)
        block_table_gpu[0, :8] = torch.tensor(block_ids_per_req[0], dtype=torch.int32)

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # --- Verify Python behavior (ground truth) ---
        dest_block_id = block_ids_per_req[0][3]  # dest_block_idx = 3

        # Copy DID happen (dest block should be modified)
        conv_changed = not torch.allclose(
            conv_state_py[dest_block_id], conv_state_orig[dest_block_id]
        )
        assert conv_changed, "Python: Conv state copy should have happened"

        # num_accepted_tokens should NOT be changed to 1
        assert input_batch_py.num_accepted_tokens_cpu[0] == num_accepted_tokens[0], (
            f"Python: num_accepted_tokens should remain {num_accepted_tokens[0]}, "
            f"got {input_batch_py.num_accepted_tokens_cpu[0]}"
        )

        # --- Verify GPU matches Python ---
        torch.testing.assert_close(
            conv_state_gpu,
            conv_state_py,
            msg="GPU conv state should match Python",
        )
        torch.testing.assert_close(
            temporal_state_gpu,
            temporal_state_py,
            msg="GPU temporal state should match Python",
        )
        expected_accepted = torch.tensor(
            input_batch_py.num_accepted_tokens_cpu[:num_reqs],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(
            gpu_ctx.num_accepted_tokens_out[:num_reqs],
            expected_accepted,
            msg="GPU num_accepted_tokens should match Python",
        )

    def test_prefix_caching_shared_block_does_not_set_accepted_to_1(
        self, device, test_config
    ):
        """
        Regression test: with prefix caching, different logical block indices
        can map to the same physical block. The kernel must NOT set
        num_accepted_tokens to 1 in that case.

        When src_block_idx != dest_block_idx but block_table maps both to the
        same physical block ID, src_addr == dst_addr. The copy is correctly
        skipped (self-copy is a no-op), but num_accepted_tokens must be
        preserved — only logical-index equality justifies setting it to 1.

        Test setup (block_size=16):
        - num_tokens_running_state = 30 + 2 - 0 = 32
        - new_num_computed = 32 + 3 - 1 = 34
        - aligned_new_computed = 32
        - accept_token_bias = 32 - 32 = 0
        - dest_block_idx = 32 // 16 - 1 = 1
        - src_block_idx = 0 (set explicitly, != dest_block_idx)
        - block_ids = [5, 5, ...] — prefix caching: both logical indices
          map to the same physical block 5
        """
        cfg = test_config
        torch.manual_seed(2001)

        req_ids = ["req_0"]
        num_computed_tokens = [30]
        num_scheduled_tokens = {"req_0": 2}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [3]  # Must stay 3, NOT become 1
        mamba_state_idx = [0]  # src_block_idx = 0, dest_block_idx will be 1

        # Prefix caching: logical blocks 0 and 1 share physical block 5
        block_ids_per_req = [[5, 5, 2, 3, 4, 6, 7, 8]]

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        (
            conv_state_py,
            temporal_state_py,
            conv_state_gpu,
            temporal_state_gpu,
            forward_context_py,
            forward_context_gpu,
        ) = _make_dual_layer_state(cfg, device)

        conv_state_orig = conv_state_py.clone()
        temporal_state_orig = temporal_state_py.clone()

        # --- Run Python path ---
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # --- Run GPU path ---
        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)
        num_reqs = len(req_ids)
        block_table_gpu = torch.zeros(num_reqs, 8, dtype=torch.int32, device=device)
        block_table_gpu[0, :8] = torch.tensor(block_ids_per_req[0], dtype=torch.int32)

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # --- Verify Python behavior (ground truth) ---
        # Copy is self-to-self (same physical block), state unchanged
        torch.testing.assert_close(
            conv_state_py,
            conv_state_orig,
            msg="Python: Conv state should be unchanged (self-copy)",
        )
        torch.testing.assert_close(
            temporal_state_py,
            temporal_state_orig,
            msg="Python: Temporal state should be unchanged (self-copy)",
        )
        # num_accepted_tokens must NOT be set to 1 (src_block_idx != dest_block_idx)
        assert input_batch_py.num_accepted_tokens_cpu[0] == num_accepted_tokens[0], (
            f"Python: num_accepted_tokens should remain {num_accepted_tokens[0]}, "
            f"got {input_batch_py.num_accepted_tokens_cpu[0]}"
        )

        # --- Verify GPU matches Python ---
        torch.testing.assert_close(
            conv_state_gpu,
            conv_state_py,
            msg="GPU conv state should match Python",
        )
        torch.testing.assert_close(
            temporal_state_gpu,
            temporal_state_py,
            msg="GPU temporal state should match Python",
        )
        expected_accepted = torch.tensor(
            input_batch_py.num_accepted_tokens_cpu[:num_reqs],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(
            gpu_ctx.num_accepted_tokens_out[:num_reqs],
            expected_accepted,
            msg="GPU num_accepted_tokens should match Python (must NOT be 1)",
        )

    def test_prefix_caching_nonsequential_block_ids_boundary(self, device, test_config):
        """
        Regression test: non-sequential physical block IDs under prefix caching
        with the needs_copy boundary at exact equality.

        Under PC, the block allocator assigns physical block IDs in arbitrary
        order (e.g., [17, 3, 42, 9] instead of [0, 1, 2, 3]). The needs_copy
        condition is purely token-count based and must evaluate identically
        regardless of the physical block IDs assigned. This test verifies that
        the kernel's address arithmetic (block_table lookup, stride computation)
        produces correct copies when physical IDs are non-sequential.

        Two requests exercise different boundary behaviors:
        - req_0: aligned_new_computed == num_tokens_running_state (exact boundary)
          This is the tightest edge: one fewer accepted token and no copy needed.
        - req_1: aligned_new_computed == num_tokens_running_state (exact boundary)
          Different block layout, src!=dest, real copy happens.

        Both use non-sequential block IDs typical of PC reuse patterns.

        Test setup (block_size=16):
        req_0:
        - num_tokens_running_state = 48 + 0 - 0 = 48
        - new_num_computed = 48 + 1 - 1 = 48
        - aligned_new_computed = 48
        - needs_copy = (48 >= 48) = True (exact boundary!)
        - accept_token_bias = 48 - 48 = 0
        - dest_block_idx = 48 // 16 - 1 = 2
        - src_block_idx = 2 (same as dest -> num_accepted = 1)

        req_1:
        - num_tokens_running_state = 31 + 1 - 0 = 32
        - new_num_computed = 32 + 3 - 1 = 34
        - aligned_new_computed = 32
        - needs_copy = (32 >= 32) = True (exact boundary!)
        - accept_token_bias = 32 - 32 = 0
        - dest_block_idx = 32 // 16 - 1 = 1
        - src_block_idx = 0 (diff from dest -> num_accepted unchanged)
        """
        cfg = test_config
        torch.manual_seed(4001)

        req_ids = ["req_0", "req_1"]
        num_computed_tokens = [48, 31]
        num_scheduled_tokens = {"req_0": 0, "req_1": 1}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [1, 3]
        mamba_state_idx = [2, 0]

        # Non-sequential block IDs typical of prefix caching allocation
        block_ids_per_req = [
            [17, 3, 42, 9, 25, 11, 30, 2],  # req_0: scattered physical blocks
            [41, 7, 22, 15, 38, 19, 4, 28],  # req_1: different scattered blocks
        ]

        layer_names = [f"layer_{i}" for i in range(cfg.num_layers)]
        # Need enough physical blocks for the scattered IDs
        num_blocks = 50
        local_cfg = _TestConfig(num_blocks=num_blocks, max_num_reqs=cfg.max_num_reqs)
        kv_cache_config = _make_kv_cache_config(local_cfg, layer_names)

        (
            conv_states_py,
            temporal_states_py,
            conv_states_gpu,
            temporal_states_gpu,
            forward_context_py,
            forward_context_gpu,
        ) = _make_dual_states(local_cfg, layer_names, device)

        # --- Run Python path ---
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(local_cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # --- Run GPU path ---
        gpu_ctx = _make_gpu_ctx(local_cfg, kv_cache_config, device)
        num_reqs = len(req_ids)
        max_blocks = max(len(b) for b in block_ids_per_req)
        block_table_gpu = torch.zeros(
            num_reqs, max_blocks, dtype=torch.int32, device=device
        )
        for i, block_ids in enumerate(block_ids_per_req):
            block_table_gpu[i, : len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # --- Compare results ---
        for i in range(cfg.num_layers):
            torch.testing.assert_close(
                conv_states_gpu[i],
                conv_states_py[i],
                msg=f"Conv state mismatch at layer {i} with non-sequential block IDs",
            )
            torch.testing.assert_close(
                temporal_states_gpu[i],
                temporal_states_py[i],
                msg=(
                    f"Temporal state mismatch at layer {i} "
                    f"with non-sequential block IDs"
                ),
            )

        expected_accepted = torch.tensor(
            input_batch_py.num_accepted_tokens_cpu[:num_reqs],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(
            gpu_ctx.num_accepted_tokens_out[:num_reqs],
            expected_accepted,
            msg="num_accepted_tokens mismatch with non-sequential block IDs",
        )

        # Verify req_0 had num_accepted set to 1 (src==dest) and req_1 unchanged
        assert input_batch_py.num_accepted_tokens_cpu[0] == 1
        assert input_batch_py.num_accepted_tokens_cpu[1] == num_accepted_tokens[1]

    def test_prefix_caching_mixed_shared_and_distinct_blocks(self, device, test_config):
        """
        Regression test: mixed batch under prefix caching where some requests
        have shared physical blocks (aliased) and others have distinct blocks,
        with the needs_copy boundary at various positions.

        This tests the interaction between:
        1. PC block aliasing (src and dest map to same physical block)
        2. The needs_copy boundary (exact equality vs well-past vs no-copy)
        3. Non-sequential physical block IDs

        Batch of 4 requests:
        - req_0: needs_copy=True, src!=dest, shared physical block (PC aliased)
                 -> copy skipped (src_addr==dst_addr), num_accepted PRESERVED
        - req_1: needs_copy=True, src!=dest, distinct blocks, non-sequential IDs
                 -> real copy happens, num_accepted PRESERVED
        - req_2: needs_copy=False (below boundary)
                 -> no action at all
        - req_3: needs_copy=True, src==dest (exact boundary, zero bias)
                 -> copy skipped (self-copy), num_accepted SET TO 1

        Test setup (block_size=16):
        req_0: running=30+2-0=32, new=32+3-1=34, aligned=32, 32>=32 -> COPY
               bias=0, dest=32//16-1=1, src=0 (!=dest)
               block_ids=[5,5,...] -> same physical -> skip, keep accepted=3

        req_1: running=60+5-2=63, new=63+3-1=65, aligned=64, 64>=63 -> COPY
               bias=1, dest=64//16-1=3, src=2 (!=dest)
               block_ids=[41,7,22,15,...] -> distinct -> real copy, keep accepted=3

        req_2: running=30+3-0=33, new=33+1-1=33, aligned=32, 32<33 -> NO COPY

        req_3: running=48+0-0=48, new=48+1-1=48, aligned=48, 48>=48 -> COPY
               bias=0, dest=48//16-1=2, src=2 (==dest)
               block_ids=[10,20,30,...] -> distinct IDs, same logical idx
               -> self-copy (src_addr==dst_addr), set accepted=1
        """
        cfg = test_config
        torch.manual_seed(5001)

        req_ids = ["req_0", "req_1", "req_2", "req_3"]
        num_computed_tokens = [30, 60, 30, 48]
        num_scheduled_tokens = {"req_0": 2, "req_1": 5, "req_2": 3, "req_3": 0}
        num_draft_tokens = {"req_1": 2}
        num_accepted_tokens = [3, 3, 1, 1]
        mamba_state_idx = [0, 2, 1, 2]

        # Block IDs with various PC patterns:
        # req_0: shared blocks (PC alias: logical 0 and 1 -> physical 5)
        # req_1: distinct non-sequential blocks
        # req_2: doesn't matter (no copy)
        # req_3: distinct sequential blocks (no aliasing)
        block_ids_per_req = [
            [5, 5, 12, 18, 23, 31, 44, 2],  # req_0: blocks 0,1 share phys 5
            [41, 7, 22, 15, 38, 19, 4, 28],  # req_1: all distinct
            [10, 20, 30, 40, 1, 6, 8, 14],  # req_2: irrelevant
            [10, 20, 30, 40, 1, 6, 8, 14],  # req_3: distinct, dest=src=idx 2
        ]

        layer_names = [f"layer_{i}" for i in range(cfg.num_layers)]
        num_blocks = 50
        local_cfg = _TestConfig(num_blocks=num_blocks, max_num_reqs=cfg.max_num_reqs)
        kv_cache_config = _make_kv_cache_config(local_cfg, layer_names)

        (
            conv_states_py,
            temporal_states_py,
            conv_states_gpu,
            temporal_states_gpu,
            forward_context_py,
            forward_context_gpu,
        ) = _make_dual_states(local_cfg, layer_names, device)

        # --- Run Python path ---
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(local_cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # --- Run GPU path ---
        gpu_ctx = _make_gpu_ctx(local_cfg, kv_cache_config, device)
        num_reqs = len(req_ids)
        max_blocks = max(len(b) for b in block_ids_per_req)
        block_table_gpu = torch.zeros(
            num_reqs, max_blocks, dtype=torch.int32, device=device
        )
        for i, block_ids in enumerate(block_ids_per_req):
            block_table_gpu[i, : len(block_ids)] = torch.tensor(
                block_ids, dtype=torch.int32
            )

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # --- Compare all state tensors ---
        for i in range(cfg.num_layers):
            torch.testing.assert_close(
                conv_states_gpu[i],
                conv_states_py[i],
                msg=(
                    f"Conv state mismatch at layer {i} — "
                    f"mixed PC batch with shared/distinct blocks"
                ),
            )
            torch.testing.assert_close(
                temporal_states_gpu[i],
                temporal_states_py[i],
                msg=(
                    f"Temporal state mismatch at layer {i} — "
                    f"mixed PC batch with shared/distinct blocks"
                ),
            )

        # --- Compare num_accepted_tokens ---
        expected_accepted = torch.tensor(
            input_batch_py.num_accepted_tokens_cpu[:num_reqs],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(
            gpu_ctx.num_accepted_tokens_out[:num_reqs],
            expected_accepted,
            msg="num_accepted_tokens mismatch in mixed PC batch",
        )

        # Verify per-request expectations:
        # req_0: src!=dest, shared block -> preserved (3)
        assert input_batch_py.num_accepted_tokens_cpu[0] == 3
        # req_1: src!=dest, distinct blocks -> preserved (3)
        assert input_batch_py.num_accepted_tokens_cpu[1] == 3
        # req_2: no copy -> preserved (1)
        assert input_batch_py.num_accepted_tokens_cpu[2] == 1
        # req_3: src==dest -> set to 1
        assert input_batch_py.num_accepted_tokens_cpu[3] == 1

    def test_pc_aliased_blocks_skip_must_use_logical_idx_not_addr(
        self, device, test_config
    ):
        """
        Regression test for 6466ce0d vs 959ca0fd: the kernel's early-return
        guard must compare logical block indices, not physical addresses.

        Under prefix caching, different logical blocks (src_block_idx=0,
        dest_block_idx=1) can map to the same physical block. When
        accept_token_bias=0, this makes src_addr == dst_addr for BOTH conv
        and temporal states. A buggy guard `if src_addr == dst_addr` would
        incorrectly set num_accepted_tokens=1; the correct guard is
        `if src_block_idx == dest_block_idx and accept_token_bias == 0`.

        The Python reference only sets num_accepted_tokens=1 when
        src_block_idx == dest_block_idx (line 79 of postprocess_mamba).
        With src_block_idx=0, dest_block_idx=1, num_accepted_tokens must
        be preserved even though the physical addresses match.

        Test setup (block_size=16):
        - num_tokens_running_state = 30 + 2 - 0 = 32
        - new_num_computed = 32 + 3 - 1 = 34
        - aligned_new_computed = 32
        - needs_copy = (32 >= 32) = True
        - accept_token_bias = 32 - 32 = 0
        - dest_block_idx = 32 // 16 - 1 = 1
        - src_block_idx = 0 (explicitly, != dest_block_idx)
        - block_ids = [7, 7, ...] -> physical aliasing via prefix caching

        Expected: num_accepted_tokens stays 3 (not set to 1).
        Bug (959ca0fd): kernel saw src_addr == dst_addr, set it to 1.
        """
        cfg = test_config
        torch.manual_seed(6001)

        req_ids = ["req_0"]
        num_computed_tokens = [30]
        num_scheduled_tokens = {"req_0": 2}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [3]
        mamba_state_idx = [0]  # src_block_idx = 0

        # Prefix caching: logical blocks 0 and 1 both map to physical block 7.
        block_ids_per_req = [[7, 7, 10, 11, 12, 13, 14, 15]]

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        (
            conv_state_py,
            temporal_state_py,
            conv_state_gpu,
            temporal_state_gpu,
            forward_context_py,
            forward_context_gpu,
        ) = _make_dual_layer_state(cfg, device)

        # --- Run Python path ---
        scheduler_output = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        input_batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            scheduler_output,
            kv_cache_config,
            input_batch_py,
            requests,
            forward_context_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # Python reference: src_block_idx(0) != dest_block_idx(1) -> no change
        assert input_batch_py.num_accepted_tokens_cpu[0] == 3, (
            f"Python: num_accepted_tokens should remain 3, "
            f"got {input_batch_py.num_accepted_tokens_cpu[0]}"
        )

        # --- Run GPU path ---
        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)
        num_reqs = len(req_ids)
        block_table_gpu = torch.zeros(num_reqs, 8, dtype=torch.int32, device=device)
        block_table_gpu[0, :8] = torch.tensor(block_ids_per_req[0], dtype=torch.int32)

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table_gpu,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # The critical assertion: kernel must NOT set num_accepted_tokens to 1
        # when src_block_idx != dest_block_idx, even though src_addr == dst_addr
        # due to prefix caching aliasing.
        #
        # Old kernel (959ca0fd): `if src_addr == dst_addr` -> FAILS here (sets 1)
        # Fixed kernel (6466ce0d): `if src_block_idx == dest_block_idx and
        #   accept_token_bias == 0` -> PASSES (preserves 3)
        kernel_accepted = gpu_ctx.num_accepted_tokens_out[0].item()
        assert kernel_accepted == 3, (
            f"Kernel set num_accepted_tokens to {kernel_accepted} but expected 3. "
            f"The early-return guard likely compared physical addresses "
            f"(src_addr == dst_addr) instead of logical block indices "
            f"(src_block_idx == dest_block_idx). Under prefix caching, "
            f"different logical blocks can share the same physical block."
        )

        # Also verify state tensors match Python
        torch.testing.assert_close(
            conv_state_gpu,
            conv_state_py,
            msg="GPU conv state should match Python",
        )
        torch.testing.assert_close(
            temporal_state_gpu,
            temporal_state_py,
            msg="GPU temporal state should match Python",
        )

    def test_as_strided_temporal_copy_size(self, device, test_config):
        """
        Regression test for 240723d46: temporal copy_size must be
        inner_size * elem_size, not state_block_stride.

        In production (gpu_model_runner.py), conv and temporal states share
        a raw buffer via torch.as_strided where stride(0) equals
        page_size_bytes / elem_size — larger than either state's natural
        element count.  Using stride(0) as copy_size for temporal states
        overwrites into the next block's conv region.

        Layout per page (384 float16 elements = 768 bytes):
            [conv: 256 elems | temporal: 128 elems]

        The test triggers a temporal copy from block 4 to block 3.  With the
        bug the kernel copies 768 bytes (page stride) instead of 256 bytes
        (128 * 2), overwriting conv_state[4] with conv_state[5]'s data.

        Test setup (block_size=16):
        - running = 60 + 5 - 2 = 63
        - new = 63 + 3 - 1 = 65
        - aligned = 64 >= 63 -> COPY needed
        - accept_token_bias = 64 - 63 = 1
        - dest_block_idx = 64 // 16 - 1 = 3
        - temporal: actual_src_block_idx = 3 + 1 = 4  (block_ids[4] = 4)
        """
        cfg = test_config
        torch.manual_seed(7001)

        req_ids = ["req_0"]
        num_computed_tokens = [60]
        num_scheduled_tokens = {"req_0": 5}
        num_draft_tokens = {"req_0": 2}
        num_accepted_tokens = [3]
        mamba_state_idx = [3]
        block_ids_per_req = [list(range(8))]

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        # --- Production-like packed layout (mirrors gpu_model_runner.py) ---
        conv_shape = (cfg.conv_width, cfg.conv_inner_dim)
        temporal_shape = (cfg.temporal_state_dim,)
        dtype = cfg.dtype
        elem_size = torch.tensor([], dtype=dtype).element_size()

        conv_natural_elems = cfg.conv_width * cfg.conv_inner_dim
        temporal_natural_elems = cfg.temporal_state_dim
        page_size_bytes = (conv_natural_elems + temporal_natural_elems) * elem_size
        num_element_per_page = page_size_bytes // elem_size

        assert num_element_per_page > temporal_natural_elems, (
            "Test requires padded stride; page must be larger than one state"
        )

        raw_py = torch.randn(
            cfg.num_blocks * num_element_per_page, dtype=dtype, device=device
        )
        raw_gpu = raw_py.clone()

        def make_views(raw):
            conv_tgt = (cfg.num_blocks, *conv_shape)
            conv_nat_stride = torch.empty(conv_tgt).stride()
            conv = torch.as_strided(
                raw,
                size=conv_tgt,
                stride=(num_element_per_page, *conv_nat_stride[1:]),
                storage_offset=0,
            )

            temp_tgt = (cfg.num_blocks, *temporal_shape)
            temp_nat_stride = torch.empty(temp_tgt).stride()
            temp = torch.as_strided(
                raw,
                size=temp_tgt,
                stride=(num_element_per_page, *temp_nat_stride[1:]),
                storage_offset=conv_natural_elems,
            )
            return conv, temp

        conv_py, temp_py = make_views(raw_py)
        conv_gpu, temp_gpu = make_views(raw_gpu)

        fwd_py = {"layer_0": _make_mock_attention(conv_py, temp_py)}
        fwd_gpu = {"layer_0": _make_mock_attention(conv_gpu, temp_gpu)}

        # --- Python reference ---
        sched = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            sched,
            kv_cache_config,
            batch_py,
            requests,
            fwd_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # --- GPU fused kernel ---
        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)
        num_reqs = 1
        block_table = torch.zeros(num_reqs, 8, dtype=torch.int32, device=device)
        block_table[0, :8] = torch.tensor(block_ids_per_req[0], dtype=torch.int32)

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=fwd_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # --- Assertions ---
        # With the bug (pre-240723d46), the kernel copies page_size_bytes
        # (768) for temporal state instead of 256 bytes, overwriting
        # conv_state[4] with conv_state[5]'s data.
        torch.testing.assert_close(
            conv_gpu,
            conv_py,
            msg=(
                "Conv state corrupted: temporal copy_size was likely "
                "state_block_stride instead of inner_size * elem_size"
            ),
        )
        torch.testing.assert_close(
            temp_gpu,
            temp_py,
            msg="Temporal state mismatch",
        )

        expected_accepted = torch.tensor(
            batch_py.num_accepted_tokens_cpu[:num_reqs],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(
            gpu_ctx.num_accepted_tokens_out[:num_reqs],
            expected_accepted,
            msg="num_accepted_tokens mismatch",
        )

    def test_temporal_copy_with_bias_ge_2(self, device, test_config):
        """
        Coverage test for the temporal-state block-table stride arithmetic
        when ``accept_token_bias >= 2``.

        The kernel computes, for temporal (non-conv) states::

            actual_src_block_idx = src_block_idx + accept_token_bias
            actual_src_block_id = block_table[req, actual_src_block_idx]

        All prior regression tests exercise only ``bias == 1``, i.e. they
        only ever read one slot ahead of ``src_block_idx`` in the block
        table. An off-by-one (or missing scale) in the address computation
        on line 143 of ``mamba_utils.py`` would be invisible to every
        existing test but would silently read the wrong physical block on
        any speculative-decode cycle that accepts multiple tokens across a
        block boundary, feeding a stale hidden state forward one step.

        Setup (block_size=16):
        - running   = 28 + 2 - 0 = 30
        - new       = 30 + 3 - 1 = 32
        - aligned   = 32 >= 30 -> COPY needed
        - bias      = 32 - 30 = 2             (key: >= 2)
        - dest_idx  = 32 // 16 - 1 = 1
        - src_idx   = 1 (same as dest -> exercises post-copy accepted=1 write)
        - temporal actual_src_block_idx = 1 + 2 = 3 (reads block_table[0, 3])

        With identity block_ids = [0,1,2,3,...], an off-by-one that used
        bias=1 would copy from block_ids[2]=2 instead of block_ids[3]=3,
        producing a clear state-value mismatch against the Python
        reference.
        """
        cfg = test_config
        torch.manual_seed(7002)

        req_ids = ["req_0"]
        num_computed_tokens = [28]
        num_scheduled_tokens = {"req_0": 2}
        num_draft_tokens: dict[str, int] = {}
        num_accepted_tokens = [3]  # -> accept_token_bias = 2
        mamba_state_idx = [1]  # src_block_idx = 1 = dest_block_idx
        block_ids_per_req = [list(range(8))]

        layer_names = ["layer_0"]
        kv_cache_config = _make_kv_cache_config(cfg, layer_names)

        (
            conv_state_py,
            temporal_state_py,
            conv_state_gpu,
            temporal_state_gpu,
            fwd_py,
            fwd_gpu,
        ) = _make_dual_layer_state(cfg, device)
        temporal_state_orig = temporal_state_py.clone()

        # --- Python reference ---
        sched = _make_postprocess_scheduler_output(
            req_ids,
            num_scheduled_tokens,
            {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
        )
        batch_py = _make_input_batch(
            req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy()
        )
        requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
        copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

        postprocess_mamba(
            sched,
            kv_cache_config,
            batch_py,
            requests,
            fwd_py,
            _COPY_FUNCS,
            copy_bufs,
        )
        torch.accelerator.synchronize()

        # --- GPU fused kernel ---
        gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)
        num_reqs = 1
        block_table = torch.zeros(num_reqs, 8, dtype=torch.int32, device=device)
        block_table[0, :8] = torch.tensor(block_ids_per_req[0], dtype=torch.int32)

        _run_gpu_postprocess(
            gpu_ctx,
            kv_cache_config=kv_cache_config,
            forward_context=fwd_gpu,
            copy_funcs=_COPY_FUNCS,
            block_table=block_table,
            req_ids=req_ids,
            num_accepted_tokens=num_accepted_tokens,
            mamba_state_idx=mamba_state_idx,
            num_scheduled_tokens=num_scheduled_tokens,
            num_computed_tokens=num_computed_tokens,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )

        # --- Ground truth: Python must have sourced temporal from block 3 ---
        actual_src_block_id = block_ids_per_req[0][3]  # == 3
        dest_block_id = block_ids_per_req[0][1]  # == 1
        torch.testing.assert_close(
            temporal_state_py[dest_block_id],
            temporal_state_orig[actual_src_block_id],
            msg=(
                "Python reference did not copy from block_ids[src+bias]=3; "
                "test preconditions are wrong"
            ),
        )

        # --- GPU kernel must match Python byte-for-byte ---
        torch.testing.assert_close(
            conv_state_gpu,
            conv_state_py,
            msg="Conv state mismatch at accept_token_bias=2",
        )
        torch.testing.assert_close(
            temporal_state_gpu,
            temporal_state_py,
            msg=(
                "Temporal state mismatch at accept_token_bias=2: the kernel "
                "likely read the wrong slot of the block table "
                "(actual_src_block_idx stride arithmetic)"
            ),
        )

        expected_accepted = torch.tensor(
            batch_py.num_accepted_tokens_cpu[:num_reqs],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(
            gpu_ctx.num_accepted_tokens_out[:num_reqs],
            expected_accepted,
            msg="num_accepted_tokens mismatch at accept_token_bias=2",
        )
