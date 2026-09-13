# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.ubatch_utils import UBatchSlice


@pytest.mark.parametrize(
    "builder_type", [GDNAttentionMetadataBuilder, Mamba2AttentionMetadataBuilder]
)
@pytest.mark.parametrize("use_spec_decode", [False, True])
@pytest.mark.parametrize("ubatching", [False, True])
def test_accepted_metadata_rejects_ubatching(builder_type, use_spec_decode, ubatching):
    """Both speculative and recovery-only paths reject unsliced accepted counts."""
    builder = Mock(spec=builder_type)
    builder.supports_update_block_table = False
    group = SimpleNamespace(
        get_metadata_builder=Mock(return_value=builder), layer_names=["layer.0"]
    )
    spec = MambaSpec(block_size=16, shapes=((16, 64),), dtypes=(torch.float16,))
    block_table = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    query_start = torch.tensor([0, 1, 2], dtype=torch.int32)
    counts = torch.tensor([3, 1], dtype=torch.int32)
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
        ),
        optimistic_seq_lens_cpu=torch.tensor([65, 65], dtype=torch.int32),
        input_batch=SimpleNamespace(
            block_table=[
                SimpleNamespace(get_device_tensor=Mock(return_value=block_table))
            ],
            num_computed_tokens_cpu_tensor=torch.tensor([64, 64]),
            num_prompt_tokens_cpu_tensor=torch.tensor([8, 8]),
        ),
        routed_experts_initialized=False,
        use_async_spec_decode=False,
        is_mm_prefix_lm=False,
        model_config=SimpleNamespace(rswa_window=None),
        cache_config=SimpleNamespace(
            use_replayssm=False, kv_sharing_fast_prefill=False
        ),
        query_start_loc=SimpleNamespace(gpu=query_start, cpu=query_start),
        seq_lens=torch.tensor([65, 65], dtype=torch.int32),
        positions=torch.tensor([64, 64]),
        dcp_world_size=1,
        speculative_config=object(),
        drafter=None,
        attn_groups=[[group]],
        num_accepted_tokens=SimpleNamespace(gpu=counts),
        num_decode_draft_tokens=SimpleNamespace(cpu=torch.tensor([0, 0])),
        mamba_prev_last_scheduled_idx=None,
        _get_encoder_seq_lens=Mock(return_value=(None, None)),
    )
    slices = (
        [UBatchSlice(slice(0, 1), slice(0, 1)), UBatchSlice(slice(1, 2), slice(1, 2))]
        if ubatching
        else None
    )
    context = (
        pytest.raises(AssertionError, match="UBatching not supported")
        if ubatching
        else nullcontext()
    )
    with context:
        GPUModelRunner._build_attention_metadata(
            runner,
            num_tokens=2,
            num_reqs=2,
            max_query_len=1,
            ubatch_slices=slices,
            use_spec_decode=use_spec_decode,
            slot_mappings={0: torch.tensor([1, 2])},
        )
    if ubatching:
        builder.build.assert_not_called()
    else:
        builder.build.assert_called_once()
        torch.testing.assert_close(
            builder.build.call_args.kwargs["num_accepted_tokens"], counts
        )
