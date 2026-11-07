# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inside graph capture, the ROCm CSA layer forks its input chains onto the aux
streams only up to VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD tokens, like
DeepseekV4Attention._run_parallel_input_projections. The eager profile run forks
at any size, so the aux streams' hipBLASLt handles exist before capture."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm DeepSeek V4.1 attention layer"
)


class Forked(Exception):
    pass


@pytest.mark.parametrize(
    "threshold, num_tokens, capturing, forks",
    [
        (1024, 400, True, True),
        (0, 400, True, False),
        (400, 400, True, True),
        (399, 400, True, False),
        # The eager profile run (non-dict metadata) forks above the threshold.
        (0, 400, False, True),
    ],
)
def test_csa_streams_follow_token_threshold(
    monkeypatch, threshold, num_tokens, capturing, forks
):
    import vllm.models.deepseek_v41.amd.rocm as rocm
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterMLAAttention
    from vllm.models.deepseek_v41.attention import DeepseekV4Attention

    monkeypatch.setattr(rocm.envs, "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD", threshold)
    monkeypatch.setattr(
        rocm.current_platform,
        "enable_multi_stream_overlap",
        lambda streams, metadata: streams is not None,
    )
    monkeypatch.setattr(
        rocm, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    seen = []

    def sequential(self, positions, hidden_states, *args):
        seen.append(self.aux_stream_list)
        return "sequential"

    def fork(self, *args):
        raise Forked

    monkeypatch.setattr(DeepseekV4Attention, "forward", sequential)
    monkeypatch.setattr(
        DeepseekV41ROCMAiterMLAAttention, "_alloc_attn_out", lambda self, n, h: None
    )
    monkeypatch.setattr(DeepseekV41ROCMAiterMLAAttention, "_forward_csa2_full", fork)

    layer = DeepseekV41ROCMAiterMLAAttention.__new__(DeepseekV41ROCMAiterMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.indexer = object()
    layer.compressor = object()
    layer._prepare_and_attn_fn = None
    layer.aux_stream_list = ["s0", "s1", "s2"]

    hidden = torch.empty(num_tokens, 8)
    if forks:
        with pytest.raises(Forked):
            layer.forward(None, hidden, None)
        assert seen == []
    else:
        assert layer.forward(None, hidden, None) == "sequential"
        # The sequential path runs without aux streams and restores them.
        assert seen == [None]
    assert layer.aux_stream_list == ["s0", "s1", "s2"]
