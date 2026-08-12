# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for DiffusionGemmaModelState's per-request causal buffer
under CUDA graph capture.

``DiffusionGemmaModelState._causal_buf`` holds each request's per-sequence
causal/bidirectional attention flag and feeds FlashAttention's per-sequence
causal path. That backend's metadata builder casts non-int32 causal tensors
via ``causal.to(torch.int32)`` (see ``build()`` in
``vllm/v1/attention/backends/flash_attn.py``) -- an out-of-place op that
allocates a *new* tensor on every call. Under ``CUDAGraphMode.FULL``, the
address returned at *capture* time is what the replayed kernel reads for the
graph's entire lifetime, so if ``_causal_buf`` isn't already int32, later
updates to it never reach an already-captured graph: replay keeps producing
whatever causal/bidirectional setting existed at capture time.
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


def _make_model_state(max_num_seqs: int = 4) -> DiffusionGemmaModelState:
    """Construct a real DiffusionGemmaModelState without loading any weights.

    ``__init__`` never touches ``model`` on the path that builds
    ``_causal_buf`` (it's only read later, by ``custom_sampler()``), so
    passing ``encoder_cache=None`` -- which skips the ``EncoderRunner``
    construction that would need a real vision-capable model -- lets
    ``model`` stay ``None``.
    """
    vllm_config = create_vllm_config(
        model_name="facebook/opt-125m", max_num_seqs=max_num_seqs
    )
    # DiffusionGemma-specific generation_config.json keys __init__ reads;
    # opt-125m has neither, so supply them the same way the real checkpoint
    # would.
    vllm_config.model_config.try_get_generation_config = types.MethodType(
        lambda self: {"stability_threshold": 1, "max_denoising_steps": 48},
        vllm_config.model_config,
    )
    return DiffusionGemmaModelState(
        vllm_config=vllm_config,
        model=None,
        encoder_cache=None,
        device=DEVICE,
    )


def test_causal_buf_survives_cudagraph_replay():
    """A CUDA graph that reads _causal_buf (through FlashAttention's dtype
    cast) must reflect later in-place updates to it on replay."""
    model_state = _make_model_state(max_num_seqs=4)
    causal_buf = model_state._causal_buf
    n = causal_buf.shape[0]

    def cast_like_flash_attn_build(causal: torch.Tensor) -> torch.Tensor:
        # Mirrors flash_attn.py's build():
        #   if isinstance(causal, torch.Tensor) and causal.dtype != torch.int32:
        #       causal = causal.to(torch.int32)
        if isinstance(causal, torch.Tensor) and causal.dtype != torch.int32:
            causal = causal.to(torch.int32)
        return causal

    out = torch.zeros(n, dtype=torch.int32, device=DEVICE)

    # Capture-time state: mirrors the all-False default of a dummy warmup
    # request, which never went through the real is_encoder_phase state
    # machine.
    causal_buf[:] = 0
    causal_for_capture = cast_like_flash_attn_build(causal_buf[:n])

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out.copy_(causal_for_capture)
    torch.cuda.current_stream().wait_stream(s)

    # A real step: e.g. a request entering its causal (prefill/encoder) phase.
    causal_buf[:] = 1
    graph.replay()
    torch.accelerator.synchronize()

    assert out.tolist() == causal_buf.tolist(), (
        f"replayed graph produced {out.tolist()}, but _causal_buf currently "
        f"holds {causal_buf.tolist()} -- the captured graph is reading a "
        "stale, capture-time causal tensor instead of the live buffer"
    )
