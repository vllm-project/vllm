# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test: DiffusionGemmaModelState._causal_buf must stay visible to
an already-captured CUDA graph.

A captured graph binds to tensor addresses at capture time, so the causal
flags can only reach replay through in-place updates to a persistent buffer.
That requires the buffer to be int32: FlashAttentionMetadataBuilder.build()
rejects other dtypes rather than casting out-of-place, because such a cast
would allocate a fresh tensor each call and freeze the graph at the
capture-time snapshot.
"""

import types

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.model_executor.models.diffusion_gemma import DiffusionGemmaModelState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA"
)

DEVICE = torch.device("cuda:0")


def _make_model_state() -> DiffusionGemmaModelState:
    vllm_config = create_vllm_config(model_name="facebook/opt-125m", max_num_seqs=4)
    vllm_config.model_config.try_get_generation_config = types.MethodType(
        lambda self: {"stability_threshold": 1, "max_denoising_steps": 48},
        vllm_config.model_config,
    )
    # model/encoder_cache are only dereferenced when encoder_cache is not None.
    return DiffusionGemmaModelState(
        vllm_config=vllm_config, model=None, encoder_cache=None, device=DEVICE
    )


def test_causal_buf_survives_cudagraph_replay():
    causal_buf = _make_model_state()._causal_buf
    assert causal_buf.dtype == torch.int32, (
        "FlashAttentionMetadataBuilder.build() requires int32 causal tensors; "
        "any other dtype raises there instead of silently breaking replay"
    )

    out = torch.zeros(causal_buf.shape[0], dtype=torch.int32, device=DEVICE)
    causal_buf[:] = 0

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out.copy_(causal_buf)
    torch.cuda.current_stream().wait_stream(stream)

    causal_buf[:] = 1  # e.g. a request entering its causal (encoder) phase
    graph.replay()
    torch.accelerator.synchronize()

    assert out.tolist() == causal_buf.tolist(), (
        f"replay produced {out.tolist()}, but _causal_buf now holds "
        f"{causal_buf.tolist()} -- graph is reading a stale capture-time tensor"
    )
