# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm.distributed.kv_transfer.kv_connector.v1.kda_recoverssm_transport import (
    KDA_BASE_RECURRENT_REGION,
    KDA_TARGET_CONV_REGION,
    KDATargetStateLayerTransport,
)
from vllm.v1.kv_cache_interface import MambaSpec


def _page_cache(spec: MambaSpec, num_blocks: int) -> torch.Tensor:
    return torch.empty(
        (num_blocks, 1, 1, spec.state_content_size_bytes), dtype=torch.int8
    )


def _prefill_spec() -> MambaSpec:
    return MambaSpec(
        block_size=256,
        shapes=((3, 6), (2, 2, 2)),
        dtypes=(torch.float16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


def _recover_spec(num_speculative_tokens: int) -> MambaSpec:
    query_len = num_speculative_tokens + 1
    return MambaSpec(
        block_size=256,
        shapes=(
            (6, 3 + num_speculative_tokens),
            (2, 2, 2),
            (2, query_len, 2),
            (2, query_len, 4),
        ),
        dtypes=(torch.float16, torch.float32, torch.float32, torch.float16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_speculative_tokens", [4, 7])
@pytest.mark.parametrize("dcp_size", [1, 4])
def test_no_spec_prefill_to_recoverssm_target_state_transport(
    num_speculative_tokens: int,
    dcp_size: int,
):
    num_blocks = 4
    prefill_spec = _prefill_spec()
    prefill = KDATargetStateLayerTransport(
        "linear_attn",
        1,
        _page_cache(prefill_spec, num_blocks),
        prefill_spec,
        conv_state_dim_first=False,
    )
    prefill.states[0].copy_(
        torch.arange(prefill.states[0].numel(), dtype=torch.float16).view_as(
            prefill.states[0]
        )
    )
    prefill.recurrent_state.copy_(
        torch.arange(prefill.recurrent_state.numel(), dtype=torch.float32).view_as(
            prefill.recurrent_state
        )
    )
    prefill.stage_blocks([1, 2])

    assert [region.kind for region in prefill.regions] == [
        KDA_TARGET_CONV_REGION,
        KDA_BASE_RECURRENT_REGION,
    ]

    for _ in range(dcp_size):
        decode_spec = _recover_spec(num_speculative_tokens)
        decode = KDATargetStateLayerTransport(
            "linear_attn",
            1,
            _page_cache(decode_spec, num_blocks),
            decode_spec,
            conv_state_dim_first=True,
        )
        for state in decode.states:
            state.fill_(99)

        assert [region.content_len_bytes for region in decode.regions] == [
            region.content_len_bytes for region in prefill.regions
        ]
        decode.target_conv[1:3].copy_(prefill.target_conv[1:3])
        decode.recurrent_state[1:3].copy_(prefill.recurrent_state[1:3])
        decode.materialize_blocks([1, 2])

        expected_conv = prefill.states[0][1:3].transpose(-1, -2)
        torch.testing.assert_close(decode.local_conv[1:3, ..., :3], expected_conv)
        torch.testing.assert_close(
            decode.recurrent_state[1:3], prefill.recurrent_state[1:3]
        )
        assert torch.count_nonzero(decode.local_conv[1:3, ..., 3:]) == 0
        for record in decode.states[2:]:
            assert torch.count_nonzero(record[1:3]) == 0

        assert torch.all(decode.local_conv[0] == 99)


@pytest.mark.cpu_test
def test_homogeneous_recoverssm_transport_resets_local_records():
    spec = _recover_spec(7)
    producer = KDATargetStateLayerTransport(
        "linear_attn", 0, _page_cache(spec, 2), spec, conv_state_dim_first=True
    )
    consumer = KDATargetStateLayerTransport(
        "linear_attn", 0, _page_cache(spec, 2), spec, conv_state_dim_first=True
    )
    producer.local_conv.fill_(3)
    producer.recurrent_state.fill_(5)
    producer.stage_blocks([1])
    for state in consumer.states:
        state.fill_(99)

    consumer.target_conv[1].copy_(producer.target_conv[1])
    consumer.recurrent_state[1].copy_(producer.recurrent_state[1])
    consumer.materialize_blocks([1])

    assert torch.all(consumer.local_conv[1, ..., :3] == 3)
    assert torch.count_nonzero(consumer.local_conv[1, ..., 3:]) == 0
    assert torch.all(consumer.recurrent_state[1] == 5)
    for record in consumer.states[2:]:
        assert torch.count_nonzero(record[1]) == 0


@pytest.mark.cpu_test
def test_ordinary_speculative_kda_page_is_rejected():
    spec = MambaSpec(
        block_size=256,
        shapes=((6, 7), (2, 2, 2)),
        dtypes=(torch.float16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        num_speculative_blocks=4,
    )
    with pytest.raises(ValueError, match="ordinary speculative"):
        KDATargetStateLayerTransport(
            "linear_attn",
            1,
            _page_cache(spec, 2),
            spec,
            conv_state_dim_first=True,
        )
