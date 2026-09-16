# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_MOONMATH_MLA decode under decode context parallelism."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_moonmath_amd

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and has_moonmath_amd()),
    reason="ROCM_MOONMATH_MLA requires ROCm and moonmath_amd",
)

_LAT, _ROPE = 512, 64


@pytest.fixture(autouse=True)
def _cuda_default_device():
    with torch.device("cuda"):
        yield


def test_dcp_shards_merge_to_reference():
    """Each rank attends its round-robin KV shard; merging the outputs by the
    returned LSE must reproduce unsharded causal attention over the whole KV."""
    from vllm.v1.attention.backends.mla.moonmath_mla import MoonmathMLAImpl

    world, num_reqs, seq_len, q_len, heads = 2, 2, 257, 4, 12
    num_tokens = num_reqs * q_len
    scale = 1.0 / math.sqrt(_LAT + _ROPE)
    torch.manual_seed(0)
    q = torch.randn(num_tokens, heads, _LAT + _ROPE, dtype=torch.bfloat16)
    kv = (torch.randn(num_reqs, seq_len, _LAT + _ROPE) * 4).to(torch.float8_e4m3fnuz)

    impl = MoonmathMLAImpl.__new__(MoonmathMLAImpl)
    impl.scale = scale
    impl._mm_kv_scale = 1.0
    impl._max_model_len = seq_len
    impl.dcp_world_size = world

    outs, lses = [], []
    for rank in range(world):
        shard = kv[:, rank::world]
        local_len = shard.shape[1]
        decode = SimpleNamespace(
            seq_lens=torch.full((num_reqs,), local_len, dtype=torch.int32),
            paged_kv_indices=torch.arange(num_reqs * local_len, dtype=torch.int32),
            paged_kv_indptr=torch.arange(
                0, (num_reqs + 1) * local_len, local_len, dtype=torch.int32
            ),
            dcp_tot_seq_lens=torch.full((num_reqs,), seq_len, dtype=torch.int32),
        )
        impl.dcp_rank = rank
        out, lse = impl.forward_mqa(
            q,
            shard.reshape(-1, _LAT + _ROPE).contiguous(),
            SimpleNamespace(decode=decode, causal=True),
            layer=None,
        )
        outs.append(out.float())
        lses.append(lse)
    weights = torch.softmax(torch.stack(lses), dim=0).unsqueeze(-1)
    merged = (weights * torch.stack(outs)).sum(dim=0)

    latent, rope = kv[..., :_LAT].float(), kv[..., _LAT:].float()
    q_lat, q_pe = q.float().split((_LAT, _ROPE), dim=-1)
    ref = torch.empty_like(merged)
    for req in range(num_reqs):
        for pos in range(q_len):
            row = req * q_len + pos
            visible = seq_len - (q_len - 1 - pos)
            scores = (
                q_lat[row] @ latent[req, :visible].T + q_pe[row] @ rope[req, :visible].T
            ) * scale
            ref[row] = torch.softmax(scores, dim=-1) @ latent[req, :visible]

    rel_err = (merged - ref).abs().max() / ref.abs().max()
    assert rel_err < 1e-2, f"relative error {rel_err:.3e}"
