# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        reason="FlashInfer GDN prefill is not supported on ROCm.",
        allow_module_level=True,
    )

import flashinfer.gdn_prefill  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm.model_executor.layers.mamba.checkpoint import (  # noqa: E402
    MambaPrefillCheckpointExporter,
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fi_chunk_gated_delta_rule,
)  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    plan_flashinfer_checkpoints,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID  # noqa: E402


def test_flashinfer_gdn_prefill_passes_cu_seqlens_through(monkeypatch):
    """FlashInfer accepts int32 cu_seqlens; the wrapper must not cast them."""
    captured_cu_seqlens = None

    def fake_chunk_gated_delta_rule(**kwargs):
        nonlocal captured_cu_seqlens
        captured_cu_seqlens = kwargs["cu_seqlens"]
        return kwargs["q"]

    monkeypatch.setattr(
        flashinfer.gdn_prefill,
        "chunk_gated_delta_rule",
        fake_chunk_gated_delta_rule,
    )
    q = torch.zeros(1, 2, 1, 2)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    output, final_state = fi_chunk_gated_delta_rule(
        q=q,
        k=q,
        v=q,
        g=torch.zeros(1, 2, 1),
        beta=torch.zeros(1, 2, 1),
        initial_state=torch.zeros(1, 1, 2, 2),
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    assert captured_cu_seqlens is not None
    assert captured_cu_seqlens is cu_seqlens
    assert output.shape == q.shape
    assert final_state is None


@pytest.mark.skipif(
    not (
        current_platform.is_device_capability(90)
        or current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    ),
    reason="FlashInfer GDN prefill checkpoints need SM90, SM100 or SM120.",
)
@pytest.mark.parametrize("use_cp", [True, False])
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
def test_flashinfer_checkpoints_export_the_state_at_each_offset(use_cp, ssm_dtype):
    """Each exported checkpoint is the state of the prefill stopped there.

    The leading decode row is outside the prefill kernel. The two checkpointed
    prefills put FlashInfer's stride (their offsets' gcd) below both offsets,
    so the exporter must pick each state out of the shared emitted buffer;
    the third prefill takes no checkpoint. Both FlashInfer routes are covered
    because ``use_cp="auto"`` may take either. The conv cache has MTP's spec
    slots after the window, which the conv kernels leave alone.
    """
    torch.manual_seed(0)
    device = "cuda"
    num_qk_heads, num_v_heads, head_dim, conv_dim, state_len = 2, 4, 128, 16, 3
    num_spec = 2
    offsets = [0, 128, 0, 192]
    query_lens = [200, 50, 300]
    state_indices = [NULL_BLOCK_ID, 3, NULL_BLOCK_ID, 5]
    num_blocks = 8

    num_tokens = sum(query_lens)
    q = F.normalize(
        torch.randn(1, num_tokens, num_qk_heads, head_dim, device=device), dim=-1
    ).bfloat16()
    k = F.normalize(
        torch.randn(1, num_tokens, num_qk_heads, head_dim, device=device), dim=-1
    ).bfloat16()
    v = torch.randn(1, num_tokens, num_v_heads, head_dim, device=device).bfloat16()
    g = F.logsigmoid(torch.randn(1, num_tokens, num_v_heads, device=device)).exp()
    beta = torch.sigmoid(torch.randn(1, num_tokens, num_v_heads, device=device))
    initial_state = 0.1 * torch.randn(
        len(query_lens), num_v_heads, head_dim, head_dim, device=device
    )
    prefill_cu = [0]
    for query_len in query_lens:
        prefill_cu.append(prefill_cu[-1] + query_len)

    every_n_tokens, cu_starts, rows = plan_flashinfer_checkpoints(offsets, query_lens)
    state_checkpoints = torch.empty(
        cu_starts[-1], num_v_heads, head_dim, head_dim, device=device
    )
    fi_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=torch.tensor(prefill_cu, device=device),
        use_qk_l2norm_in_kernel=False,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=torch.tensor(cu_starts, device=device),
        checkpoint_every_n_tokens=every_n_tokens,
        use_cp=use_cp,
    )

    # The exporter sees the decode row's token in front of the prefills.
    raw_qkv = torch.randn(1 + num_tokens, conv_dim, device=device).bfloat16()
    cu_seqlens = torch.tensor([0] + [1 + n for n in prefill_cu], device=device)
    conv_state = torch.zeros(
        num_blocks,
        state_len + num_spec,
        conv_dim,
        device=device,
        dtype=torch.bfloat16,
    ).transpose(-1, -2)
    ssm_state = torch.zeros(
        num_blocks, num_v_heads, head_dim, head_dim, device=device, dtype=ssm_dtype
    )
    MambaPrefillCheckpointExporter(state_len=state_len).export(
        MambaPrefillCheckpointMetadata(
            checkpoint_offsets=torch.tensor(offsets, device=device, dtype=torch.int32),
            state_indices=torch.tensor(state_indices, device=device),
        ),
        raw_qkv=raw_qkv,
        conv_state=conv_state,
        recurrent_checkpoint=state_checkpoints,
        recurrent_state=ssm_state,
        cu_seqlens=cu_seqlens,
        recurrent_rows=torch.tensor(rows, device=device),
    )

    written = []
    for row, (offset, block) in enumerate(zip(offsets, state_indices)):
        if not offset:
            continue
        written.append(block)
        seq = row - 1
        start = prefill_cu[seq]
        _, expected_state = fi_chunk_gated_delta_rule(
            q=q[:, start : start + offset],
            k=k[:, start : start + offset],
            v=v[:, start : start + offset],
            g=g[:, start : start + offset],
            beta=beta[:, start : start + offset],
            initial_state=initial_state[seq : seq + 1],
            output_final_state=True,
            cu_seqlens=torch.tensor([0, offset], device=device),
            use_qk_l2norm_in_kernel=False,
        )
        torch.testing.assert_close(
            ssm_state[block].float(), expected_state[0], atol=2e-2, rtol=2e-2
        )
        end = cu_seqlens[row].item() + offset
        torch.testing.assert_close(
            conv_state[block, :, :state_len],
            raw_qkv[end - state_len : end].T,
            atol=0,
            rtol=0,
        )
        assert not conv_state[block, :, state_len:].any()

    untouched = [i for i in range(num_blocks) if i not in written]
    assert not ssm_state[untouched].any()
    assert not conv_state[untouched].any()
