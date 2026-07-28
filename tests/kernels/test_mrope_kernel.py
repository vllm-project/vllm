# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Regression test: M-RoPE Triton kernel must respect ``is_neox_style``.

Bug (https://github.com/vllm-project/vllm/issues/49290):
``_triton_mrope_forward`` hardcoded Neox (half-split) rotation pairing and
ignored the ``is_neox_style`` flag. For GPT-J (interleaved even/odd) style
models such as GLM-OCR, the CUDA fast-path silently produced wrong positional
encodings in eager mode (graph mode was unaffected because ``compile_native``
forces ``forward_native``).

Fix: the kernel gains an ``is_neox_style: tl.constexpr`` switch selecting
between Neox pairing ``(i, i + rd/2)`` and GPT-J pairing ``(2i, 2i+1)``;
``forward_cuda`` passes the instance flag through.

This test compares ``forward_cuda`` (Triton kernel) against the independent
pure-PyTorch ``forward_native`` reference for both rotation styles, across
multiple token counts, interleaved modes, and position patterns.

Usage
-----
.. code-block:: bash

    # Run all mrope kernel regression tests
    pytest tests/kernels/test_mrope_kernel.py -v

    # Run only GPT-J style tests
    pytest tests/kernels/test_mrope_kernel.py -v -k "gptj"

    # Run only Neox style tests
    pytest tests/kernels/test_mrope_kernel.py -v -k "neox"

    # Run only the backward-compatibility check
    pytest tests/kernels/test_mrope_kernel.py -v -k "default"

    # Stop at the first failure
    pytest tests/kernels/test_mrope_kernel.py -v -x

    # Run with extra verbose output (prints test IDs)
    pytest tests/kernels/test_mrope_kernel.py -vv
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import mrope
from vllm.platforms import current_platform

# ---------------------------------------------------------------------------
# Test configuration — matches GLM-OCR (GPT-J style multimodal model)
# ---------------------------------------------------------------------------
HEAD_SIZE = 128          # dimension of each attention head
ROTARY_DIM = 128         # number of rotary dimensions (full head_size)
MAX_POS = 16384          # maximum position index in the cos/sin cache
BASE = 10000.0           # RoPE base frequency (theta)
MROPE_SECTION = [16, 24, 24]   # t/h/w split — sums to rotary_dim // 2 = 64
NUM_Q_HEADS = 16         # query attention heads
NUM_KV_HEADS = 8         # key/value attention heads

# Cosine similarity threshold for Triton vs PyTorch agreement.
COS_SIM_THRESHOLD = 0.9999


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rope(is_neox_style: bool, mrope_interleaved: bool,
               dtype: torch.dtype, device: str):
    """Create an MRotaryEmbedding instance inside a vLLM config context.

    ``MRotaryEmbedding`` inherits from ``CustomOp``, which requires
    ``set_current_vllm_config`` to be active during construction.
    """
    with set_current_vllm_config(VllmConfig()):
        return mrope.MRotaryEmbedding(
            head_size=HEAD_SIZE,
            rotary_dim=ROTARY_DIM,
            max_position_embeddings=MAX_POS,
            base=BASE,
            is_neox_style=is_neox_style,
            dtype=dtype,
            mrope_section=MROPE_SECTION,
            mrope_interleaved=mrope_interleaved,
        ).to(device)


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened tensors (in float32)."""
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0).item()


# ---------------------------------------------------------------------------
# forward_cuda must match forward_native for every configuration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not current_platform.is_cuda(),
                    reason="Triton mrope kernel requires CUDA")
@pytest.mark.parametrize("is_neox_style", [True, False],
                         ids=["neox", "gptj"])
@pytest.mark.parametrize("mrope_interleaved", [False, True],
                         ids=["chunked", "interleaved"])
@pytest.mark.parametrize("num_tokens", [1, 4, 129],
                         ids=["decode1", "short", "long"])
@pytest.mark.parametrize("uniform_positions", [True, False],
                         ids=["uniform_pos", "distinct_pos"])
def test_mrope_forward_cuda_matches_native(is_neox_style: bool,
                                           mrope_interleaved: bool,
                                           num_tokens: int,
                                           uniform_positions: bool):
    """``forward_cuda`` (Triton) must match ``forward_native`` (PyTorch).

    Covers both rotation pairing styles (Neox / GPT-J), both t/h/w frequency
    mask layouts (chunked / interleaved), three token counts (decode-1,
    short prefill, >1-page prefill), and two position patterns (uniform
    t/h/w positions vs fully distinct per-dimension positions).
    """
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # Build the RoPE operator under test
    rope = _make_rope(is_neox_style, mrope_interleaved=mrope_interleaved,
                      dtype=dtype, device=device)

    # ---- position tensor ------------------------------------------------
    # M-RoPE uses 3D positions: [3, num_tokens] for (temporal, height, width).
    # "uniform_positions" simulates the decode stage where all three position
    # rows carry the same token indices.  "distinct_pos" simulates prefill
    # with genuinely different t/h/w position values.
    if uniform_positions:
        row = torch.randint(0, MAX_POS, (num_tokens,), device=device)
        positions = row.unsqueeze(0).repeat(3, 1)          # [3, num_tokens]
    else:
        positions = torch.randint(0, MAX_POS, (3, num_tokens), device=device)

    # ---- input tensors --------------------------------------------------
    # vLLM stores q/k flattened: [num_tokens, num_heads * head_size]
    query = torch.randn(num_tokens, NUM_Q_HEADS * HEAD_SIZE,
                        dtype=dtype, device=device)
    key = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE,
                      dtype=dtype, device=device)

    # ---- run both paths ------------------------------------------------
    # forward_cuda → _triton_mrope_forward (Triton kernel)
    # forward_native → ApplyRotaryEmb.forward_native (pure PyTorch reference)
    with torch.no_grad():
        q_cuda, k_cuda = rope.forward_cuda(positions, query.clone(),
                                           key.clone())
        q_native, k_native = rope.forward_native(positions, query.clone(),
                                                 key.clone())

    # ---- compare --------------------------------------------------------
    cs_q = _cos_sim(q_cuda, q_native)
    cs_k = _cos_sim(k_cuda, k_native)
    max_diff_q = (q_cuda.float() - q_native.float()).abs().max().item()
    max_diff_k = (k_cuda.float() - k_native.float()).abs().max().item()

    style = "neox" if is_neox_style else "gptj"
    assert cs_q > COS_SIM_THRESHOLD, (
        f"[{style}] query mismatch: cos_sim={cs_q:.8f} "
        f"max|diff|={max_diff_q:.4e}")
    assert cs_k > COS_SIM_THRESHOLD, (
        f"[{style}] key mismatch: cos_sim={cs_k:.8f} "
        f"max|diff|={max_diff_k:.4e}")


# ---------------------------------------------------------------------------
# backward compatibility of the triton_mrope() public API
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not current_platform.is_cuda(),
                    reason="Triton mrope kernel requires CUDA")
def test_triton_mrope_default_is_neox_style():
    """``triton_mrope()`` defaults to Neox for backward compatibility.

    The unified kernel added an ``is_neox_style`` parameter that defaults
    to ``True``.  Callers that do not pass the new argument (all existing
    callers before this fix) must see **identical** behaviour to explicitly
    passing ``is_neox_style=True``.  This is enforced with a strict
    bit-exact comparison (``rtol=0, atol=0``).
    """
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    num_tokens = 4

    # Synthetic M-RoPE cos/sin — shape [3, num_tokens, head_size // 2]
    cos = torch.randn(3, num_tokens, HEAD_SIZE // 2,
                      dtype=torch.float32, device=device)
    sin = torch.randn_like(cos)

    # Flattened q/k tensors as consumed by triton_mrope()
    q = torch.randn(num_tokens, NUM_Q_HEADS * HEAD_SIZE,
                    dtype=dtype, device=device)
    k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE,
                    dtype=dtype, device=device)

    # ---- default (no is_neox_style argument) ----------------------------
    q_default, k_default = mrope.triton_mrope(
        q.clone(), k.clone(), cos, sin,
        MROPE_SECTION, HEAD_SIZE, ROTARY_DIM, False)

    # ---- explicit is_neox_style=True ------------------------------------
    q_neox, k_neox = mrope.triton_mrope(
        q.clone(), k.clone(), cos, sin,
        MROPE_SECTION, HEAD_SIZE, ROTARY_DIM, False,
        is_neox_style=True)

    # Bit-exact comparison: the default must produce the same result as
    # explicitly requesting Neox style.
    torch.testing.assert_close(q_default, q_neox, rtol=0, atol=0)
    torch.testing.assert_close(k_default, k_neox, rtol=0, atol=0)
