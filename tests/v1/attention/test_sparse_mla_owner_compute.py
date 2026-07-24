# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm.distributed as distributed
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import (
    flashinfer_mla_sparse,
    flashmla_sparse,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseImpl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.owner_compute import (
    filter_peer_slots_to_owner_local_reference,
    merge_owner_compute_partials_reference,
    should_use_owner_compute,
    validate_owner_compute_scope,
    validate_owner_prefill_materialization_support,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    filter_peer_slots_to_owner_local,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs_batch


class _FakeGroup:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.rank_in_group = 0


class _FakeOwnerComputeGroup:
    world_size = 4
    rank_in_group = 1

    def __init__(self, slots_per_peer: int) -> None:
        self.slots_per_peer = slots_per_peer
        self.gather_inputs: list[torch.Tensor] = []

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0
        self.gather_inputs.append(tensor.clone())
        if tensor.is_floating_point():
            source_rows = [tensor + source * 10 for source in range(self.world_size)]
        else:
            source_rows = [
                torch.where(
                    tensor >= 0,
                    tensor + source * self.slots_per_peer,
                    tensor,
                )
                for source in range(self.world_size)
            ]
        return torch.cat(source_rows, dim=0)


@pytest.mark.parametrize("num_actual", [2, 0])
def test_forward_owner_compute_routes_padded_rows_and_restores_origin(
    monkeypatch: pytest.MonkeyPatch,
    num_actual: int,
) -> None:
    source_stride = 4
    topk = 3
    num_heads = 2
    output_dim = 3
    block_size = 4
    blocks_per_peer = 8
    slots_per_peer = block_size * blocks_per_peer

    group = _FakeOwnerComputeGroup(slots_per_peer)
    monkeypatch.setattr(distributed, "get_dcp_group", lambda: group)
    monkeypatch.setattr(distributed, "get_pcp_group", lambda: group)

    converted_slots = torch.tensor(
        [[1, slots_per_peer + 2, -1], [2 * slots_per_peer + 3, 4, 5]],
        dtype=torch.int32,
    )[:num_actual]
    cache_get_calls: list[tuple[int, torch.Tensor, dict[str, int]]] = []

    def fake_cache_get_or_build(
        rows: int,
        block_table: torch.Tensor,
        *,
        build,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_get_calls.append((rows, block_table, kwargs))
        assert kwargs == {
            "source_stride": source_stride,
            "owner_rank": group.rank_in_group,
            "dcp_world_size": 4,
            "blocks_per_peer": blocks_per_peer,
            "cp_kv_cache_interleave_size": block_size,
            "block_size": block_size,
        }
        padded_slots = torch.full((source_stride, topk), -1, dtype=torch.int32)
        padded_slots[:rows] = converted_slots
        return build(padded_slots)

    routed_rows = group.world_size * source_stride
    local_slots = torch.arange(routed_rows * topk, dtype=torch.int32).view(
        routed_rows, topk
    )
    local_selected_counts = torch.full((routed_rows,), topk, dtype=torch.int32)
    local_selected_counts[1::4] = 0
    filter_inputs: list[torch.Tensor] = []

    def fake_filter(peer_slots: torch.Tensor, **kwargs):
        filter_inputs.append(peer_slots.clone())
        assert kwargs == {
            "owner_rank": group.rank_in_group,
            "dcp_world_size": group.world_size,
            "blocks_per_peer": blocks_per_peer,
            "block_size": block_size,
        }
        return local_slots, local_selected_counts

    monkeypatch.setattr(
        flashinfer_mla_sparse,
        "filter_peer_slots_to_owner_local",
        fake_filter,
    )

    kernel_calls: list[dict[str, object]] = []

    def fake_flashinfer_kernel(**kwargs):
        kernel_calls.append(kwargs)
        output = torch.ones((routed_rows, 1, num_heads, output_dim))
        output[local_selected_counts == 0] = float("nan")
        return (
            output,
            torch.zeros((routed_rows, num_heads, 1)),
        )

    fake_flashinfer = ModuleType("flashinfer")
    fake_flashinfer_decode = ModuleType("flashinfer.decode")
    fake_flashinfer_decode.trtllm_batch_decode_with_kv_cache_mla = (
        fake_flashinfer_kernel
    )
    fake_flashinfer.decode = fake_flashinfer_decode
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", fake_flashinfer_decode)

    restored = torch.arange(
        source_stride * num_heads * output_dim, dtype=torch.float32
    ).view(source_stride, num_heads, output_dim)
    reduce_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_reduce(
        output: torch.Tensor,
        lse: torch.Tensor,
        cp_group: _FakeOwnerComputeGroup,
        *,
        is_lse_base_on_e: bool,
    ) -> torch.Tensor:
        reduce_calls.append((output.clone(), lse.clone()))
        assert cp_group is group
        assert not is_lse_base_on_e
        empty_rows = local_selected_counts == 0
        assert output[empty_rows].isnan().all()
        assert lse[empty_rows].isneginf().all()
        assert torch.all(output[~empty_rows] == 1)
        assert torch.all(lse[~empty_rows] == 0)
        return restored

    import vllm.v1.attention.ops.common as common_ops

    monkeypatch.setattr(common_ops, "cp_lse_ag_out_rs_batch", fake_reduce)

    impl = object.__new__(FlashInferMLASparseImpl)
    impl.topk_indices_buffer = torch.arange(
        source_stride * topk, dtype=torch.int32
    ).view(source_stride, topk)
    impl._owner_compute_workspace_buffer = torch.empty(1, dtype=torch.int8)
    impl.bmm1_scale = 0.5
    impl.bmm2_scale = 1.25
    impl.qk_nope_head_dim = 4
    impl.kv_lora_rank = 4
    impl.qk_rope_head_dim = 4

    q = torch.arange(source_stride * 8, dtype=torch.float32).view(source_stride, 8)
    kv_cache = torch.empty((6, block_size, 8))
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_decodes=0,
        num_actual_tokens=num_actual,
        req_id_per_token=torch.arange(num_actual, dtype=torch.int32),
        block_table=block_table,
        topk_tokens=topk,
        block_size=block_size,
        cp_kv_cache_interleave_size=block_size,
    )
    layer = SimpleNamespace(
        pcp_owner_compute_source_stride=source_stride,
        pcp_peer_block_stride=blocks_per_peer,
        owner_peer_slot_cache=SimpleNamespace(
            get_or_build_owner_local=fake_cache_get_or_build
        ),
    )

    output, lse = impl._forward_owner_compute(  # type: ignore[arg-type]
        q,
        kv_cache,
        metadata,
        layer,
    )

    torch.testing.assert_close(output, restored[:num_actual])
    assert lse is None
    assert len(group.gather_inputs) == 2
    torch.testing.assert_close(group.gather_inputs[0], q)
    expected_padded_slots = torch.full((source_stride, topk), -1, dtype=torch.int32)
    expected_padded_slots[:num_actual] = converted_slots
    torch.testing.assert_close(group.gather_inputs[1], expected_padded_slots)
    assert len(filter_inputs) == 1
    expected_routed_slots = torch.cat(
        [
            torch.where(
                expected_padded_slots >= 0,
                expected_padded_slots + source * slots_per_peer,
                expected_padded_slots,
            )
            for source in range(group.world_size)
        ],
        dim=0,
    )
    torch.testing.assert_close(filter_inputs[0], expected_routed_slots)
    assert len(kernel_calls) == 1
    assert len(reduce_calls) == 1

    kernel_call = kernel_calls[0]
    assert kernel_call["query"].shape == (routed_rows, 1, q.shape[1])
    assert kernel_call["kv_cache"].shape == (kv_cache.shape[0], 1, *kv_cache.shape[1:])
    torch.testing.assert_close(kernel_call["block_tables"], local_slots.unsqueeze(1))
    torch.testing.assert_close(kernel_call["seq_lens"], local_selected_counts)
    assert kernel_call["return_lse"] is True

    assert len(cache_get_calls) == 1
    cached_rows, converted_block_table, _ = cache_get_calls[0]
    assert cached_rows == num_actual
    assert converted_block_table is block_table


@pytest.mark.parametrize("num_actual", [2, 0])
@pytest.mark.parametrize("lse_layout", ["batch_seq_heads", "batch_heads_seq"])
def test_flashmla_forward_owner_compute_uses_row_batches_and_natural_lse(
    monkeypatch: pytest.MonkeyPatch,
    num_actual: int,
    lse_layout: str,
) -> None:
    source_stride = 4
    topk = 3
    num_heads = 2
    head_dim = 5
    kv_lora_rank = 3
    v_head_dim = 2
    block_size = 4
    blocks_per_peer = 8
    slots_per_peer = block_size * blocks_per_peer

    monkeypatch.setattr(
        flashmla_sparse.current_platform,
        "is_device_capability_family",
        lambda capability: capability == 100,
    )
    group = _FakeOwnerComputeGroup(slots_per_peer)
    monkeypatch.setattr(distributed, "get_dcp_group", lambda: group)
    monkeypatch.setattr(distributed, "get_pcp_group", lambda: group)

    converted_slots = torch.tensor(
        [[1, slots_per_peer + 2, -1], [2 * slots_per_peer + 3, 4, 5]],
        dtype=torch.int32,
    )[:num_actual]

    def fake_cache_get_or_build(
        rows: int,
        block_table: torch.Tensor,
        *,
        build,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert rows == num_actual
        assert block_table is metadata.block_table
        assert kwargs == {
            "source_stride": source_stride,
            "owner_rank": group.rank_in_group,
            "dcp_world_size": group.world_size,
            "blocks_per_peer": blocks_per_peer,
            "cp_kv_cache_interleave_size": block_size,
            "block_size": block_size,
        }
        padded_slots = torch.full((source_stride, topk), -1, dtype=torch.int32)
        padded_slots[:rows] = converted_slots
        return build(padded_slots)

    routed_rows = group.world_size * source_stride
    local_slots = torch.arange(routed_rows * topk, dtype=torch.int32).view(
        routed_rows, topk
    )
    local_selected_counts = torch.tensor(
        [3, 0, 2, 1] * group.world_size,
        dtype=torch.int32,
    )
    filter_inputs: list[torch.Tensor] = []

    def fake_filter(peer_slots: torch.Tensor, **kwargs):
        filter_inputs.append(peer_slots.clone())
        assert kwargs == {
            "owner_rank": group.rank_in_group,
            "dcp_world_size": group.world_size,
            "blocks_per_peer": blocks_per_peer,
            "block_size": block_size,
        }
        return local_slots, local_selected_counts

    monkeypatch.setattr(
        flashmla_sparse,
        "filter_peer_slots_to_owner_local",
        fake_filter,
    )
    fresh_scheduler = flashmla_sparse.FlashMLASchedMeta()
    monkeypatch.setattr(
        flashmla_sparse,
        "get_mla_metadata",
        lambda: (fresh_scheduler, object()),
    )

    kernel_calls: list[dict[str, object]] = []
    padded_heads = num_heads + 3
    raw_lse = torch.arange(
        routed_rows * padded_heads,
        dtype=torch.float32,
    ).view(routed_rows, padded_heads)

    kernel_output = torch.arange(
        routed_rows * num_heads * kv_lora_rank,
        dtype=torch.bfloat16,
    ).view(routed_rows, 1, num_heads, kv_lora_rank)

    def fake_flashmla_kernel(**kwargs):
        kernel_calls.append(kwargs)
        if lse_layout == "batch_seq_heads":
            lse = raw_lse.unsqueeze(1)
        else:
            lse = raw_lse.unsqueeze(-1)
        return kernel_output, lse

    w_uv = (
        torch.arange(
            num_heads * kv_lora_rank * v_head_dim,
            dtype=torch.bfloat16,
        ).view(num_heads, kv_lora_rank, v_head_dim)
        / 8
    )
    expected_projected = torch.bmm(
        kernel_output.squeeze(1).transpose(0, 1),
        w_uv,
    ).transpose(0, 1)
    restored = torch.arange(
        source_stride * num_heads * v_head_dim, dtype=torch.bfloat16
    ).view(source_stride, num_heads, v_head_dim)
    reduce_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_reduce(
        output: torch.Tensor,
        lse: torch.Tensor,
        cp_group: _FakeOwnerComputeGroup,
        *,
        is_lse_base_on_e: bool,
    ) -> torch.Tensor:
        reduce_calls.append((output.clone(), lse.clone()))
        assert cp_group is group
        assert is_lse_base_on_e
        assert output.shape == (routed_rows, num_heads, v_head_dim)
        torch.testing.assert_close(output, expected_projected)
        assert lse.shape == (routed_rows, num_heads)
        empty_rows = local_selected_counts == 0
        assert lse[empty_rows].isneginf().all()
        torch.testing.assert_close(lse[~empty_rows], raw_lse[~empty_rows, :num_heads])
        return restored

    import vllm.v1.attention.ops.common as common_ops

    monkeypatch.setattr(common_ops, "cp_lse_ag_out_rs_batch", fake_reduce)

    placeholder_scheduler = object()
    dummy_block_table = torch.tensor([[0]], dtype=torch.int32)
    cache_lens = torch.tensor([topk], dtype=torch.int32)
    fp8_metadata = FlashMLASparseMetadata.FP8KernelMetadata(
        scheduler_metadata=placeholder_scheduler,
        dummy_block_table=dummy_block_table,
        cache_lens=cache_lens,
    )
    metadata = SimpleNamespace(
        num_decodes=0,
        num_actual_tokens=num_actual,
        req_id_per_token=torch.arange(num_actual, dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        topk_tokens=topk,
        block_size=block_size,
        cp_kv_cache_interleave_size=block_size,
        fp8_extra_metadata=fp8_metadata,
    )
    layer = SimpleNamespace(
        pcp_owner_compute_source_stride=source_stride,
        pcp_peer_block_stride=blocks_per_peer,
        W_UV=w_uv,
        v_head_dim=v_head_dim,
        owner_peer_slot_cache=SimpleNamespace(
            get_or_build_owner_local=fake_cache_get_or_build,
            get_or_build_owner_local_metadata=lambda _key, build: build(),
        ),
    )

    impl = object.__new__(FlashMLASparseImpl)
    impl.topk_indices_buffer = torch.arange(
        source_stride * topk, dtype=torch.int32
    ).view(source_stride, topk)
    impl.num_heads = num_heads
    impl.kv_lora_rank = kv_lora_rank
    impl.fp8_decode_padded_heads = 64
    impl._fp8_flash_mla_kernel = fake_flashmla_kernel

    q = torch.arange(
        source_stride * num_heads * head_dim,
        dtype=torch.bfloat16,
    ).view(source_stride, num_heads, head_dim)
    kv_cache = torch.empty((6, block_size, 656), dtype=torch.uint8)

    output, lse = impl._forward_owner_compute(  # type: ignore[arg-type]
        q,
        kv_cache,
        metadata,
        layer,
    )

    torch.testing.assert_close(output, restored[:num_actual])
    assert lse is None
    assert len(group.gather_inputs) == 2
    torch.testing.assert_close(group.gather_inputs[0], q)
    expected_padded_slots = torch.full((source_stride, topk), -1, dtype=torch.int32)
    expected_padded_slots[:num_actual] = converted_slots
    torch.testing.assert_close(group.gather_inputs[1], expected_padded_slots)
    assert len(filter_inputs) == 1

    assert len(kernel_calls) == 1
    kernel_call = kernel_calls[0]
    assert kernel_call["q"].shape == (
        routed_rows,
        1,
        num_heads,
        head_dim,
    )
    torch.testing.assert_close(
        kernel_call["q"].squeeze(1),
        torch.cat(
            [q + source * 10 for source in range(group.world_size)],
            dim=0,
        ),
    )
    assert kernel_call["kv_c_and_k_pe_cache"] is kv_cache
    torch.testing.assert_close(
        kernel_call["topk_indices"],
        local_slots.unsqueeze(1),
    )
    torch.testing.assert_close(
        kernel_call["topk_length"],
        local_selected_counts,
    )
    owner_kernel_metadata = kernel_call["kernel_metadata"]
    assert isinstance(
        owner_kernel_metadata,
        FlashMLASparseMetadata.FP8KernelMetadata,
    )
    assert owner_kernel_metadata.scheduler_metadata is fresh_scheduler
    assert owner_kernel_metadata.dummy_block_table is dummy_block_table
    assert owner_kernel_metadata.cache_lens is cache_lens
    assert len(reduce_calls) == 1


@pytest.mark.parametrize(
    ("w_uv", "v_head_dim", "match"),
    [
        (None, 2, "requires W_UV"),
        (torch.empty((2, 4, 2), dtype=torch.bfloat16), 2, "requires W_UV"),
        (torch.empty((2, 3, 2), dtype=torch.bfloat16), 4, "requires W_UV"),
        (torch.empty((2, 3, 2), dtype=torch.float32), 2, "query device and dtype"),
    ],
)
def test_flashmla_owner_compute_validates_value_projection(
    w_uv: torch.Tensor | None,
    v_head_dim: int,
    match: str,
) -> None:
    impl = object.__new__(FlashMLASparseImpl)
    impl.num_heads = 2
    impl.kv_lora_rank = 3
    q = torch.empty((4, 2, 5), dtype=torch.bfloat16)
    layer = SimpleNamespace(W_UV=w_uv, v_head_dim=v_head_dim)

    with pytest.raises(RuntimeError, match=match):
        impl._validate_owner_value_projection(q, layer)  # type: ignore[arg-type]


def test_flashmla_owner_compute_accepts_value_projection() -> None:
    impl = object.__new__(FlashMLASparseImpl)
    impl.num_heads = 2
    impl.kv_lora_rank = 3
    q = torch.empty((4, 2, 5), dtype=torch.bfloat16)
    w_uv = torch.empty((2, 3, 7), dtype=torch.bfloat16)
    layer = SimpleNamespace(W_UV=w_uv, v_head_dim=7)

    actual_weight, actual_dim = impl._validate_owner_value_projection(  # type: ignore[arg-type]
        q,
        layer,
    )

    assert actual_weight is w_uv
    assert actual_dim == 7


def test_owner_value_projection_commutes_with_lse_merge() -> None:
    torch.manual_seed(7)
    owners, rows, heads, latent_dim, value_dim = 4, 3, 2, 5, 3
    partials = torch.randn(owners, rows, heads, latent_dim, dtype=torch.float64)
    weights = torch.randn(heads, latent_dim, value_dim, dtype=torch.float64)
    lse = torch.randn(owners, rows, heads, dtype=torch.float64)
    owner_weights = torch.softmax(lse, dim=0)

    merge_then_project = torch.einsum("orh,orhl->rhl", owner_weights, partials)
    merge_then_project = torch.einsum("rhl,hlv->rhv", merge_then_project, weights)
    project_then_merge = torch.einsum("orhl,hlv->orhv", partials, weights)
    project_then_merge = torch.einsum(
        "orh,orhv->rhv", owner_weights, project_then_merge
    )

    torch.testing.assert_close(
        project_then_merge,
        merge_then_project,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("lse", "expected"),
    [
        (
            torch.arange(30, dtype=torch.float32).view(2, 1, 15),
            torch.arange(30, dtype=torch.float32).view(2, 15)[:, :3],
        ),
        (
            torch.arange(30, dtype=torch.float32).view(2, 15, 1),
            torch.arange(30, dtype=torch.float32).view(2, 15)[:, :3],
        ),
    ],
)
def test_flashmla_owner_lse_layout_normalization(
    lse: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual = FlashMLASparseImpl._normalize_owner_lse(
        lse,
        batch_size=2,
        seq_len=1,
        num_heads=3,
    )
    torch.testing.assert_close(actual, expected)


def test_flashmla_owner_compute_rejects_hopper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        flashmla_sparse.current_platform,
        "is_device_capability_family",
        lambda _capability: False,
    )
    impl = object.__new__(FlashMLASparseImpl)
    with pytest.raises(RuntimeError, match="requires SM100"):
        impl._forward_owner_compute(
            torch.empty((1, 1, 1)),
            torch.empty((1, 1, 1)),
            SimpleNamespace(),
            SimpleNamespace(),
        )


def test_owner_compute_selection_is_uniform_for_zero_token_prefill_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PCP_OWNER_PREFILL_MODE", "auto")
    for max_prefill_seq_len in (1024, 1535, 1536):
        assert not should_use_owner_compute(
            owner_history_enabled=True,
            num_decodes=0,
            max_prefill_seq_len=max_prefill_seq_len,
        )
    for _local_rank_max in (2048, 1792, 1536, 1280):
        # Every rank receives the same scheduler-global maximum even when its
        # localized DualChunkSwap rows have different sequence ends.
        assert should_use_owner_compute(
            owner_history_enabled=True,
            num_decodes=0,
            max_prefill_seq_len=1537,
        )
    # Mixed and decode-only batches retain direct peer reads in the MVP.
    assert not should_use_owner_compute(
        owner_history_enabled=True,
        num_decodes=1,
        max_prefill_seq_len=3584,
    )

    with pytest.raises(ValueError, match="cannot be negative"):
        should_use_owner_compute(
            owner_history_enabled=True,
            num_decodes=0,
            max_prefill_seq_len=-1,
        )
    assert not should_use_owner_compute(
        owner_history_enabled=False,
        num_decodes=0,
        max_prefill_seq_len=3584,
    )


@pytest.mark.parametrize("mode", ["direct", "materialize"])
def test_explicit_owner_prefill_mode_disables_owner_compute(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setenv("VLLM_PCP_OWNER_PREFILL_MODE", mode)
    assert not should_use_owner_compute(
        owner_history_enabled=True,
        num_decodes=0,
        max_prefill_seq_len=3584,
    )


def test_invalid_owner_prefill_mode_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PCP_OWNER_PREFILL_MODE", "unknown")
    with pytest.raises(ValueError, match="must be auto, direct, or materialize"):
        should_use_owner_compute(
            owner_history_enabled=True,
            num_decodes=0,
            max_prefill_seq_len=3584,
        )


def test_explicit_materialization_requires_backend_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PCP_OWNER_PREFILL_MODE", "materialize")
    validate_owner_prefill_materialization_support(
        owner_history_enabled=True,
        supports_materialization=True,
    )
    validate_owner_prefill_materialization_support(
        owner_history_enabled=False,
        supports_materialization=False,
    )
    with pytest.raises(RuntimeError, match="requires an attention backend"):
        validate_owner_prefill_materialization_support(
            owner_history_enabled=True,
            supports_materialization=False,
        )


def test_owner_compute_scope_is_exact_pcp4_dcp4() -> None:
    validate_owner_compute_scope(
        pcp_world_size=4,
        dcp_world_size=4,
        pcp_rank=2,
        dcp_rank=2,
        cp_kv_cache_interleave_size=64,
        block_size=64,
    )

    with pytest.raises(RuntimeError, match="PCP4=DCP4"):
        validate_owner_compute_scope(
            pcp_world_size=2,
            dcp_world_size=2,
            pcp_rank=0,
            dcp_rank=0,
            cp_kv_cache_interleave_size=64,
            block_size=64,
        )
    with pytest.raises(RuntimeError, match="identical PCP/DCP rank ordering"):
        validate_owner_compute_scope(
            pcp_world_size=4,
            dcp_world_size=4,
            pcp_rank=1,
            dcp_rank=2,
            cp_kv_cache_interleave_size=64,
            block_size=64,
        )
    with pytest.raises(RuntimeError, match="must equal block_size"):
        validate_owner_compute_scope(
            pcp_world_size=4,
            dcp_world_size=4,
            pcp_rank=1,
            dcp_rank=1,
            cp_kv_cache_interleave_size=32,
            block_size=64,
        )


def test_batch_lse_reduce_scatter_single_rank_and_shape_guard() -> None:
    output = torch.randn((3, 2, 4))
    lse = torch.randn((3, 2))
    actual_output, actual_lse = cp_lse_ag_out_rs_batch(
        output,
        lse,
        _FakeGroup(1),  # type: ignore[arg-type]
        return_lse=True,
    )
    assert actual_output is output
    assert actual_lse is lse

    with pytest.raises(ValueError, match="leading dimension divisible"):
        cp_lse_ag_out_rs_batch(
            output,
            lse,
            _FakeGroup(4),  # type: ignore[arg-type]
        )


def test_filter_peer_slots_to_owner_local_is_stable() -> None:
    # 8 slots/peer. Owner 1 occupies rank-major peer interval [8, 16).
    peer_slots = torch.tensor(
        [
            [0, 8, 17, 9, -1, 15, 16, 7],
            [12, 11, 10, 9, 8, 15, 14, 13],
        ],
        dtype=torch.int32,
    )
    local, counts = filter_peer_slots_to_owner_local_reference(
        peer_slots,
        owner_rank=1,
        dcp_world_size=4,
        blocks_per_peer=2,
        block_size=4,
    )

    torch.testing.assert_close(
        local[0],
        torch.tensor([0, 1, 7, -1, -1, -1, -1, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        local[1],
        torch.tensor([4, 3, 2, 1, 0, 7, 6, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(counts, torch.tensor([3, 8], dtype=torch.int32))


def test_filter_peer_slots_handles_empty_owner_rows() -> None:
    peer_slots = torch.tensor([[0, 1, -1], [24, 31, 32]], dtype=torch.int32)
    local, counts = filter_peer_slots_to_owner_local_reference(
        peer_slots,
        owner_rank=2,
        dcp_world_size=4,
        blocks_per_peer=2,
        block_size=4,
    )
    torch.testing.assert_close(local, torch.full_like(local, -1))
    torch.testing.assert_close(counts, torch.zeros_like(counts))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
def test_filter_peer_slots_cuda_matches_reference() -> None:
    peer_slots = torch.tensor(
        [
            [0, 8, 17, 9, -1, 15, 16, 7],
            [12, 11, 10, 9, 8, 15, 14, 13],
            [0, 1, -1, 24, 31, 32, 7, 16],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    expected_slots, expected_counts = filter_peer_slots_to_owner_local_reference(
        peer_slots,
        owner_rank=1,
        dcp_world_size=4,
        blocks_per_peer=2,
        block_size=4,
    )
    actual_slots, actual_counts = filter_peer_slots_to_owner_local(
        peer_slots,
        owner_rank=1,
        dcp_world_size=4,
        blocks_per_peer=2,
        block_size=4,
    )

    torch.testing.assert_close(actual_slots, expected_slots)
    torch.testing.assert_close(actual_counts, expected_counts)


def test_owner_compute_merge_matches_base2_partitioned_attention() -> None:
    outputs = torch.tensor(
        [
            [[[1.0, 2.0]], [[5.0, 6.0]]],
            [[[3.0, 4.0]], [[7.0, 8.0]]],
            [[[0.0, 0.0]], [[9.0, 9.0]]],
            [[[0.0, 0.0]], [[9.0, 9.0]]],
        ]
    )
    lses = torch.tensor(
        [
            [[1.0], [2.0]],
            [[3.0], [2.0]],
            [[float("-inf")], [float("-inf")]],
            [[float("-inf")], [float("-inf")]],
        ]
    )

    merged, merged_lse = merge_owner_compute_partials_reference(
        outputs,
        lses,
        return_lse=True,
    )

    # Row 0 weights are 2^1 and 2^3; row 1 has equal LSEs.
    torch.testing.assert_close(merged[0], torch.tensor([[2.6, 3.6]]))
    torch.testing.assert_close(merged[1], torch.tensor([[6.0, 7.0]]))
    torch.testing.assert_close(
        merged_lse,
        torch.tensor([[torch.log2(torch.tensor(10.0))], [3.0]]),
    )


def test_owner_compute_merge_zeroes_rows_with_no_selected_keys() -> None:
    outputs = torch.randn((4, 2, 3, 5))
    lses = torch.full((4, 2, 3), float("-inf"))
    merged, merged_lse = merge_owner_compute_partials_reference(
        outputs,
        lses,
        return_lse=True,
    )
    torch.testing.assert_close(merged, torch.zeros_like(merged))
    assert merged_lse.isneginf().all()
