# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.pp_payload import (
    AUX_HIDDEN_STATES_NAMESPACE,
    PP_SIDECAR_PREFIX,
    TOPK_INDICES_NAMESPACE,
    PPForwardPayload,
    merge_pp_aux_hidden_states,
)
from vllm.sequence import IntermediateTensors

pytestmark = pytest.mark.skip_global_cleanup


def _intermediate(value: float) -> IntermediateTensors:
    return IntermediateTensors(
        {
            "hidden_states": torch.full((2, 3), value, dtype=torch.float32),
            "residual": torch.full((2, 3), -value, dtype=torch.float32),
        }
    )


def test_pp_forward_payload_roundtrip_and_multihop():
    aux_2 = torch.full((2, 3), 2.0)
    aux_10 = torch.full((2, 3), 10.0)
    topk_0 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    topk_1 = torch.tensor([[4, 5], [6, 7]], dtype=torch.int32)

    rank_0 = PPForwardPayload(_intermediate(0))
    rank_0.add_aux_hidden_states([2], [aux_2])
    rank_0.set_topk_indices(topk_0)

    rank_1_in = PPForwardPayload.from_intermediate_tensors(
        rank_0.to_intermediate_tensors()
    )
    local_buffer = torch.empty_like(topk_0)
    rank_1_in.copy_topk_indices_to(local_buffer, 2)
    torch.testing.assert_close(local_buffer, topk_0)

    rank_1_local = PPForwardPayload(_intermediate(1))
    rank_1_local.add_aux_hidden_states([10], [aux_10])
    rank_1_out = rank_1_in.carry(rank_1_local.to_intermediate_tensors())
    rank_1_out.set_topk_indices(topk_1)

    rank_2_in = PPForwardPayload.from_intermediate_tensors(
        rank_1_out.to_intermediate_tensors()
    )
    torch.testing.assert_close(
        rank_2_in.intermediate_tensors["hidden_states"],
        torch.full((2, 3), 1.0),
    )
    aux_by_layer = rank_2_in.pop_aux_hidden_states()
    assert aux_by_layer[2] is aux_2
    assert aux_by_layer[10] is aux_10
    assert rank_2_in.pop_topk_indices() is topk_1


def test_merge_pp_aux_hidden_states_uses_configured_global_order():
    aux_2 = torch.tensor([2])
    aux_10 = torch.tensor([10])
    aux_20 = torch.tensor([20])
    payload = PPForwardPayload(_intermediate(0))
    payload.add_aux_hidden_states([10, 2], [aux_10, aux_2])

    merged = merge_pp_aux_hidden_states(payload, (2, 10, 20), [aux_20])

    assert merged[0] is aux_2
    assert merged[1] is aux_10
    assert merged[2] is aux_20


@pytest.mark.parametrize(
    ("layer_ids", "local_states", "message"),
    [
        ((2, 2), [], "Duplicate aux hidden-state layer ids"),
        ((2, 10), [], "last PP stage"),
    ],
)
def test_merge_pp_aux_hidden_states_rejects_invalid_counts(
    layer_ids: tuple[int, ...],
    local_states: list[torch.Tensor],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        merge_pp_aux_hidden_states(None, layer_ids, local_states)


def test_pp_forward_payload_validates_topk_shape():
    payload = PPForwardPayload(_intermediate(0))
    payload.set_topk_indices(torch.zeros((2, 4), dtype=torch.int32))

    with pytest.raises(ValueError, match="top-k indices shape"):
        payload.copy_topk_indices_to(
            torch.empty((2, 3), dtype=torch.int32), num_tokens=2
        )


def test_pp_forward_payload_sidecars_disable_implicit_tp_all_gather():
    policy = PPForwardPayload.make_all_gather_policy({"residual": True})
    aux_key = f"{PP_SIDECAR_PREFIX}{AUX_HIDDEN_STATES_NAMESPACE}/2"
    topk_key = f"{PP_SIDECAR_PREFIX}{TOPK_INDICES_NAMESPACE}/buffer"

    assert policy.get("residual", False)
    assert policy.get("hidden_states", True)
    assert not policy.get(aux_key, True)
    assert not policy.get(topk_key, True)


def test_pp_forward_payload_rejects_malformed_sidecar_key():
    malformed_key = f"{PP_SIDECAR_PREFIX}missing_name"

    with pytest.raises(ValueError, match="Malformed PP sidecar key"):
        PPForwardPayload.from_intermediate_tensors(
            IntermediateTensors({malformed_key: torch.tensor([1])})
        )
