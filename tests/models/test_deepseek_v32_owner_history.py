# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v32.nvidia.attention import (
    DeepseekV32Attention,
    _owner_history_uses_peer_slots,
)
from vllm.v1.attention.backends.mla.owner_history import (
    select_direct_owner_slot_mapping,
    validate_direct_pcp_fused_cache_contract,
)


class _OwnerOutputImpl:
    def __init__(self, projected: bool) -> None:
        self.owner_compute_returns_projected_values = projected


@pytest.mark.parametrize(
    "supports_materialization,num_prefills,num_decodes,use_mixed,expect_peer_slots",
    [
        (True, 1, 0, False, False),
        (True, 1, 0, True, True),
        (True, 1, 1, True, True),
        (True, 0, 1, False, True),
        (False, 1, 0, False, True),
    ],
)
def test_owner_history_peer_slot_refresh_matches_selected_path(
    supports_materialization: bool,
    num_prefills: int,
    num_decodes: int,
    use_mixed: bool,
    expect_peer_slots: bool,
) -> None:
    impl = type(
        "Impl",
        (),
        {"supports_owner_history_prefill_materialization": (supports_materialization)},
    )()
    metadata = type(
        "Metadata",
        (),
        {
            "num_prefills": num_prefills,
            "num_decodes": num_decodes,
            "fp8_use_mixed_batch": use_mixed,
        },
    )()
    assert _owner_history_uses_peer_slots(impl, metadata) is expect_peer_slots


def _attention_for_output_test(*, projected: bool) -> DeepseekV32Attention:
    layer = DeepseekV32Attention.__new__(DeepseekV32Attention)
    torch.nn.Module.__init__(layer)
    layer.num_local_heads = 2
    layer.kv_lora_rank = 3
    layer.v_head_dim = 2
    layer.impl = _OwnerOutputImpl(projected)  # type: ignore[assignment]
    layer.W_UV = torch.arange(12, dtype=torch.float32).view(2, 3, 2) / 8
    return layer


def test_projected_owner_output_skips_origin_value_projection() -> None:
    layer = _attention_for_output_test(projected=True)
    projected = torch.arange(12, dtype=torch.float32).view(3, 2, 2)
    output = torch.full((3, 4), float("nan"))

    layer._write_mqa_output(
        projected,
        output,
        num_actual=3,
        owner_compute=True,
    )

    torch.testing.assert_close(output.view(3, 2, 2), projected)


@pytest.mark.parametrize("owner_compute", [False, True])
def test_ordinary_output_keeps_origin_value_projection(owner_compute: bool) -> None:
    layer = _attention_for_output_test(projected=False)
    latent = torch.arange(18, dtype=torch.float32).view(3, 2, 3)
    expected = torch.bmm(latent.transpose(0, 1), layer.W_UV).transpose(0, 1)
    output = torch.empty((3, 4))

    layer._write_mqa_output(
        latent,
        output,
        num_actual=3,
        owner_compute=owner_compute,
    )

    torch.testing.assert_close(output.view(3, 2, 2), expected)


def test_projected_owner_output_rejects_latent_shape() -> None:
    layer = _attention_for_output_test(projected=True)
    latent = torch.empty((3, 2, 3))
    output = torch.empty((3, 4))

    with pytest.raises(RuntimeError, match="invalid shape"):
        layer._write_mqa_output(
            latent,
            output,
            num_actual=3,
            owner_compute=True,
        )


def _select(
    mapping: torch.Tensor | None,
    *,
    expected: bool = True,
    rank: int = 2,
    size: int = 4,
    tokens: int = 3,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if device is None:
        device = torch.device("cpu")
    return select_direct_owner_slot_mapping(
        mapping,
        owner_history_expected=expected,
        pcp_rank=rank,
        pcp_size=size,
        num_tokens=tokens,
        device=device,
    )


def test_owner_slot_selection_preserves_non_owner_slot_representation() -> None:
    assert _select(torch.arange(3), expected=False) is None


def test_owner_slot_selection_requires_mapping() -> None:
    with pytest.raises(RuntimeError, match="requires an owner-slot mapping"):
        _select(None)


@pytest.mark.parametrize(
    "mapping",
    [
        torch.empty(24, dtype=torch.int64),
        torch.empty((12, 3), dtype=torch.int64),
        torch.empty((11, 2), dtype=torch.int64),
    ],
)
def test_owner_slot_selection_rejects_malformed_shape(
    mapping: torch.Tensor,
) -> None:
    with pytest.raises(RuntimeError, match="mapping shape"):
        _select(mapping)


def test_owner_slot_selection_rejects_float_mapping() -> None:
    with pytest.raises(RuntimeError, match="integer"):
        _select(torch.empty((12, 2), dtype=torch.float32))


def test_owner_slot_selection_rejects_wrong_device() -> None:
    with pytest.raises(RuntimeError, match="mapping on"):
        _select(
            torch.empty((12, 2), dtype=torch.int64, device="meta"),
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("rank", [-1, 4])
def test_owner_slot_selection_rejects_invalid_rank(rank: int) -> None:
    with pytest.raises(RuntimeError, match="invalid PCP rank"):
        _select(torch.empty((12, 2), dtype=torch.int64), rank=rank)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_owner_slot_selection_returns_rank_view_without_copy(
    dtype: torch.dtype,
) -> None:
    mapping = torch.arange(24, dtype=dtype).view(12, 2)
    selected = _select(mapping)
    assert selected is not None
    torch.testing.assert_close(selected, mapping.view(4, 3, 2)[2])
    assert selected.untyped_storage().data_ptr() == mapping.untyped_storage().data_ptr()


def _peer_cache(block_size: int = 8) -> torch.Tensor:
    return torch.empty((4, 3, block_size, 16), dtype=torch.uint8)


def test_fused_cache_contract_accepts_shared_slot_and_block_size() -> None:
    slot = torch.empty((12, 2), dtype=torch.int64)
    validate_direct_pcp_fused_cache_contract(
        mla_slot=slot,
        indexer_slot=slot,
        mla_peer_cache=_peer_cache(),
        indexer_peer_cache=_peer_cache(),
    )


def test_fused_cache_contract_rejects_value_equal_slot_clone() -> None:
    slot = torch.zeros((12, 2), dtype=torch.int64)
    with pytest.raises(RuntimeError, match="exact same slot tensor"):
        validate_direct_pcp_fused_cache_contract(
            mla_slot=slot,
            indexer_slot=slot.clone(),
            mla_peer_cache=_peer_cache(),
            indexer_peer_cache=_peer_cache(),
        )


def test_fused_cache_contract_requires_indexer_peer_cache() -> None:
    slot = torch.empty((12, 2), dtype=torch.int64)
    with pytest.raises(RuntimeError, match="requires an indexer peer cache"):
        validate_direct_pcp_fused_cache_contract(
            mla_slot=slot,
            indexer_slot=slot,
            mla_peer_cache=_peer_cache(),
            indexer_peer_cache=None,
        )


def test_fused_cache_contract_rejects_different_block_sizes() -> None:
    slot = torch.empty((12, 2), dtype=torch.int64)
    with pytest.raises(RuntimeError, match="identical MLA and indexer"):
        validate_direct_pcp_fused_cache_contract(
            mla_slot=slot,
            indexer_slot=slot,
            mla_peer_cache=_peer_cache(8),
            indexer_peer_cache=_peer_cache(4),
        )
