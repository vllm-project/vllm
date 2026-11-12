# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3-Flash KDA prefill checkpoint on the Triton backend.

The export splits each checkpointed prefill row at its checkpoint, reruns the
tail from the exported state and stores both halves through the shared KDA
exporter. These tests pin the split plan, its numerical equivalence with the
unsplit chunk kernel, and where the metadata places the checkpoints.
"""

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.model_executor.layers.mamba.kda_checkpoint import (
    SPLIT_KDA_CHECKPOINT_ALIGNMENT,
    build_kda_prefill_checkpoint_split,
    chunk_kda_with_checkpoints,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

if current_platform.is_rocm():
    from vllm.models.glm5next.amd.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )
else:
    from vllm.models.glm5next.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )

gpu_test = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Needs a CUDA or ROCm device"
)

# A mamba block size the chunk walk lands on; any multiple of the kernel's
# 64-token chunk behaves the same and keeps the fixtures small.
BLOCK_SIZE = 64


def test_split_plan_cuts_only_checkpointed_rows() -> None:
    # One decode row, a checkpointed 200-token prefill and a plain 128-token
    # one; the decode row stays its own segment.
    qsl = [0, 1, 201, 329]
    offsets = [0, 192, 0]
    split = build_kda_prefill_checkpoint_split(
        qsl, offsets, num_decodes=1, device=torch.device("cpu")
    )

    assert split is not None
    torch.testing.assert_close(split.seg_bounds, torch.tensor([0, 1, 193, 201, 329]))
    torch.testing.assert_close(split.first_seg, torch.tensor([0, 1, 3]))
    torch.testing.assert_close(split.ckpt_rows, torch.tensor([1]))
    torch.testing.assert_close(split.tail_tokens, torch.arange(193, 201))
    torch.testing.assert_close(split.tail_bounds, torch.tensor([0, 8]))
    # rows[1] indexes row 1's checkpoint in the compact list; the zeros for
    # the other rows alias checkpoint 0 and are masked off in the exporter.
    torch.testing.assert_close(split.rows, torch.tensor([0, 0, 0]))


def test_split_plan_segments_tile_all_tokens() -> None:
    # The segment run must write every output row the unsplit call would.
    qsl = [0, 5, 133, 260, 316]
    offsets = [0, 64, 64, 0]
    split = build_kda_prefill_checkpoint_split(
        qsl, offsets, num_decodes=1, device=torch.device("cpu")
    )

    assert split is not None
    spans = list(zip(split.seg_bounds[:-1].tolist(), split.seg_bounds[1:].tolist()))
    covered = [t for start, end in spans for t in range(start, end)]
    assert covered == list(range(qsl[-1]))


def test_split_plan_declines_without_checkpoints() -> None:
    split = build_kda_prefill_checkpoint_split(
        [0, 100, 250], [0, 0], num_decodes=0, device=torch.device("cpu")
    )
    assert split is None


def test_split_plan_rejects_decode_checkpoints() -> None:
    with pytest.raises(AssertionError):
        build_kda_prefill_checkpoint_split(
            [0, 1, 100], [8, 0], num_decodes=1, device=torch.device("cpu")
        )


def _builder(
    num_prefill_checkpoint_blocks: int = 1,
    mamba_cache_mode: str = "align",
    block_size: int = BLOCK_SIZE,
) -> GDNAttentionMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name="zai-org/GLM-5.3-Flash", block_size=block_size
    )
    vllm_config.cache_config.mamba_cache_mode = mamba_cache_mode
    vllm_config.cache_config.hash_block_size = block_size
    vllm_config.cache_config.cache_hit_alignment_tokens = block_size
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=block_size,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode=mamba_cache_mode,
            num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
            prefill_checkpoint_alignment=(
                SPLIT_KDA_CHECKPOINT_ALIGNMENT
                if num_prefill_checkpoint_blocks
                else None
            ),
        ),
        layer_names=["model.layers.0.kda"],
        vllm_config=vllm_config,
        device=torch.device("cuda"),
    )


def _i32(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


@gpu_test
def test_checkpoint_metadata_targets_last_aligned_boundary() -> None:
    # Slot p holds the state after (p + 1) * block_size tokens, so a 200-token
    # prompt checkpoints at 192 into slot 2 and a 128-token one at 64 into
    # slot 0. The block table is an arange, so request 1 owns blocks 4..7.
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[200, 128], query_lens=[200, 128]),
        BLOCK_SIZE,
        torch.device("cuda"),
        arange_block_indices=True,
    )
    md = _builder().build(0, common)

    assert md.checkpoint is not None
    torch.testing.assert_close(md.checkpoint.checkpoint_offsets, _i32([192, 64]))
    torch.testing.assert_close(md.checkpoint.state_indices, _i32([2, 4]))
    assert md.checkpoint_split is not None
    torch.testing.assert_close(
        md.checkpoint_split.ckpt_rows, torch.tensor([0, 1], device="cuda")
    )
    expected_tails = torch.cat(
        (
            torch.arange(192, 200, device="cuda"),
            torch.arange(264, 328, device="cuda"),
        )
    )
    torch.testing.assert_close(md.checkpoint_split.tail_tokens, expected_tails)


@gpu_test
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_chunk_kda_with_checkpoints_matches_unsplit(
    state_dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, head_dim = 4, 128
    lens = [200, 128]
    offsets = [192, 64]
    total = sum(lens)
    shape = (1, total, num_heads, head_dim)
    q = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    raw_g = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    beta = torch.rand(1, total, num_heads, device=device, dtype=torch.float32)
    initial_state = torch.randn(
        len(lens), num_heads, head_dim, head_dim, device=device, dtype=state_dtype
    )
    kwargs = dict(
        A_log=torch.full((1, 1, num_heads, 1), -1.0, device=device),
        g_bias=torch.zeros(num_heads * head_dim, device=device),
        use_qk_l2norm_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    cu = _i32([0, lens[0], total])

    def head_only(row: int, start: int, offset: int):
        end = start + offset
        return chunk_kda_with_fused_gate(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            raw_g=raw_g[:, start:end],
            beta=beta[:, start:end],
            initial_state=initial_state[row : row + 1],
            output_final_state=True,
            cu_seqlens=_i32([0, offset]),
            **kwargs,
        )

    def full_pass():
        return chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu,
            **kwargs,
        )

    def split_pass():
        split = build_kda_prefill_checkpoint_split(
            cu.tolist(), offsets, num_decodes=0, device=device
        )
        assert split is not None
        return chunk_kda_with_checkpoints(
            chunk_kda_with_fused_gate,
            split,
            q,
            k,
            v,
            raw_g,
            beta,
            initial_state,
            **kwargs,
        )

    # The chunk path's stored final state reads not-yet-written memory on
    # cold buffers and converges over identical reruns, so every call shape
    # runs once before the comparisons below use it.
    full_pass()
    split_pass()
    for row, (start, offset) in enumerate(zip([0, lens[0]], offsets)):
        head_only(row, start, offset)

    o_ref, final_ref = full_pass()
    o, final_state, checkpoint_state = split_pass()

    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(final_state, final_ref, rtol=1e-2, atol=1e-2)

    # Each checkpoint is the state the head-only prefill finishes with.
    for row, (start, offset) in enumerate(zip([0, lens[0]], offsets)):
        _, head_state = head_only(row, start, offset)
        torch.testing.assert_close(
            checkpoint_state[row], head_state[0], rtol=1e-2, atol=1e-2
        )
