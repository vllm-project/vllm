# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo-V2 vision window attention has to apply the per-head sink logits."""

import pytest
import torch

from tests.utils import ensure_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

EMBED_DIM = 256
NUM_HEADS = 4
HEAD_DIM = 64
WINDOW = 8
# One sequence shorter than the window, one longer.
SEQ_LENS = [5, 37]


@pytest.fixture(scope="module")
def vision_attn_env():
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
        backend="nccl",
    )
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with ensure_current_vllm_config():
        initialize_model_parallel(tensor_model_parallel_size=1)
        yield
    torch.set_default_dtype(default_dtype)


def _reference(q, k, v, cu_seqlens, sinks, scale):
    """Dense windowed softmax with the sink added to each sequence's key 0."""
    groups = q.shape[1] // k.shape[1]
    out = torch.empty_like(q, dtype=torch.float32)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        qs = q[start:end].float()
        ks = k[start:end].float().repeat_interleave(groups, dim=1)
        vs = v[start:end].float().repeat_interleave(groups, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        scores[..., 0] += sinks.float().view(-1, 1)
        pos = torch.arange(end - start, device=q.device)
        outside = (pos.view(-1, 1) - pos.view(1, -1)).abs() > WINDOW
        scores.masked_fill_(outside, -torch.inf)
        out[start:end] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vs)
    return out


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires flash-attn")
@pytest.mark.parametrize("num_kv_heads", [NUM_HEADS, NUM_HEADS // 2])
def test_window_attention_applies_sinks(vision_attn_env, num_kv_heads):
    from vllm.model_executor.models.mimo_v2_omni import MiMoVisionAttention

    torch.manual_seed(0)
    attn = MiMoVisionAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        qk_channels=HEAD_DIM,
        kv_channels=HEAD_DIM,
        use_sink=True,
        visual_token_window_size=WINDOW,
    ).cuda()
    attn.sinks.data.normal_()

    total = sum(SEQ_LENS)
    opts = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn(total, NUM_HEADS, HEAD_DIM, **opts)
    k = torch.randn(total, num_kv_heads, HEAD_DIM, **opts)
    v = torch.randn(total, num_kv_heads, HEAD_DIM, **opts)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(SEQ_LENS).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    out = attn._forward_window_attn(q, k, v, cu_seqlens, max(SEQ_LENS))
    ref = _reference(q, k, v, cu_seqlens, attn.sinks, attn.scale)

    # bf16 attention lands at ~2e-3 here; dropping the sinks lands at ~1e-1.
    error = ((out.float() - ref).norm() / ref.norm()).item()
    assert error < 1e-2, f"sink-corrected output is off by {error:.2e}"
