# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A Mamba1 prefill exports its internal checkpoint without changing its result.

The mixer runs on metadata from the real builder. The checkpoint block must
hold what a prefill of only the tokens before the checkpoint ends with, and the
outputs and final states must match a run that takes no checkpoint.
"""

from dataclasses import replace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.mamba_mixer import MambaMixer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

if not current_platform.is_cuda_alike():
    pytest.skip("selective_scan_fn uses a CUDA-only custom op", allow_module_level=True)

DEVICE = torch.device(current_platform.device_type)
PREFIX = "mixer"
BLOCK_SIZE = 64
CACHE_HIT_ALIGNMENT = 16
# Row 0 checkpoints mid-block at 208. Row 1 resumes from a cached state at 192
# and checkpoints at 304. Row 2 is too short to checkpoint.
SEQ_LENS = [220, 320, 40]
QUERY_LENS = [220, 128, 40]
CHECKPOINTS = [208, 304, 0]
# Block tables are arange(15).view(3, 5): row r's column c is block 5r + c.
RUNNING_BLOCKS = [3, 9, 10]
CHECKPOINT_BLOCKS = [2, 8]
ROW1_INITIAL_BLOCK = 9


@pytest.fixture
def mixer(dist_init):
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B", dtype=torch.float32, block_size=BLOCK_SIZE
    )
    cache_config = vllm_config.cache_config
    cache_config.mamba_cache_mode = "align"
    cache_config.mamba_block_size = BLOCK_SIZE
    with set_current_vllm_config(vllm_config):
        layer = MambaMixer(
            hidden_size=64,
            ssm_state_size=16,
            conv_kernel_size=4,
            intermediate_size=128,
            time_step_rank=8,
            use_conv_bias=True,
            use_bias=False,
            use_rms_norm=False,
            model_config=vllm_config.model_config,
            cache_config=cache_config,
            prefix=PREFIX,
        ).to(DEVICE)
    torch.manual_seed(0)
    for param in layer.parameters():
        torch.nn.init.normal_(param, std=0.1)
    layer.A.data = -torch.rand_like(layer.A) - 0.5
    builder = Mamba1AttentionMetadataBuilder(
        layer.get_kv_cache_spec(vllm_config), [PREFIX], vllm_config, DEVICE
    )
    cache_config.hash_block_size = CACHE_HIT_ALIGNMENT
    cache_config.cache_hit_alignment_tokens = CACHE_HIT_ALIGNMENT
    return layer, builder, vllm_config


def _build(builder, seq_lens, query_lens):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        BLOCK_SIZE,
        DEVICE,
        arange_block_indices=True,
    )
    common = common.replace(is_prefilling=torch.ones(len(seq_lens), dtype=torch.bool))
    return builder.build(common_prefix_len=0, common_attn_metadata=common)


def _run(mixer, metadata, hidden_states, initial_state):
    layer, _, vllm_config = mixer
    layer.kv_cache = tuple(
        torch.zeros(15, *shape, dtype=dtype, device=DEVICE)
        for shape, dtype in zip(layer.get_state_shape(), layer.get_state_dtype())
    )
    for state, initial in zip(layer.kv_cache, initial_state):
        state[ROW1_INITIAL_BLOCK] = initial
    output = torch.zeros_like(hidden_states)
    with set_forward_context({PREFIX: metadata}, vllm_config):
        layer.forward_impl(hidden_states, output)
    return output, layer.kv_cache


def test_checkpoint_matches_a_prefix_run_and_leaves_the_prefill_unchanged(mixer):
    layer, builder, _ = mixer
    hidden_states = torch.randn(sum(QUERY_LENS), 64, device=DEVICE)
    initial_state = tuple(
        torch.randn(shape, device=DEVICE) for shape in layer.get_state_shape()
    )

    metadata = _build(builder, SEQ_LENS, QUERY_LENS)
    assert metadata.checkpoint_meta is not None
    assert metadata.checkpoint_meta.offsets == [
        max(ckpt - seq + query, 0)
        for ckpt, seq, query in zip(CHECKPOINTS, SEQ_LENS, QUERY_LENS)
    ]
    output, states = _run(mixer, metadata, hidden_states, initial_state)

    plain = replace(metadata, checkpoint_meta=None, checkpoint_segments=None)
    plain_output, plain_states = _run(mixer, plain, hidden_states, initial_state)
    close = dict(atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(output, plain_output, **close)
    for state, plain_state in zip(states, plain_states):
        torch.testing.assert_close(
            state[RUNNING_BLOCKS], plain_state[RUNNING_BLOCKS], **close
        )

    # Prefill rows 0 and 1 only up to their checkpoints.
    prefix_seq_lens = CHECKPOINTS[:2]
    prefix_query_lens = [208, 112]
    prefix = replace(
        _build(builder, prefix_seq_lens, prefix_query_lens),
        checkpoint_meta=None,
        checkpoint_segments=None,
    )
    prefix_hidden_states = torch.cat(
        [hidden_states[:208], hidden_states[220 : 220 + 112]]
    )
    _, prefix_states = _run(mixer, prefix, prefix_hidden_states, initial_state)
    conv_state, ssm_state = states
    prefix_conv_state, prefix_ssm_state = prefix_states
    torch.testing.assert_close(
        conv_state[CHECKPOINT_BLOCKS], prefix_conv_state[RUNNING_BLOCKS[:2]]
    )
    torch.testing.assert_close(
        ssm_state[CHECKPOINT_BLOCKS], prefix_ssm_state[RUNNING_BLOCKS[:2]], **close
    )

    written = set(RUNNING_BLOCKS + CHECKPOINT_BLOCKS)
    for state in states:
        untouched = [b for b in range(15) if b not in written]
        assert not state[untouched].any(), "wrote outside the request's blocks"


def test_update_block_table_regathers_every_pass_slot(mixer):
    """Groups sharing a spec reuse one group's metadata via update_block_table;
    both scan passes and the checkpoint must then use the new group's blocks."""
    _, builder, _ = mixer
    metadata = _build(builder, SEQ_LENS, QUERY_LENS)
    other_table = torch.arange(100, 115, dtype=torch.int32, device=DEVICE).view(3, 5)

    updated = builder.update_block_table(
        metadata, other_table, torch.zeros(1, dtype=torch.int64, device=DEVICE)
    )

    segments = updated.checkpoint_segments
    assert segments is not None and updated.checkpoint_meta is not None
    null = NULL_BLOCK_ID
    # Segments: row 0 head, row 0 tail, row 1 head, row 1 tail, row 2.
    assert segments.head_state_indices.tolist() == [103, null, 109, null, 110]
    assert segments.tail_state_indices.tolist() == [null, 103, null, 109, null]
    assert updated.checkpoint_meta.state_indices.tolist() == [102, 108, null]
